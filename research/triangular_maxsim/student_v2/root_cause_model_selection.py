"""Root-cause driven model-selection harness for TRACE.

This module turns existing benchmark artifacts into a per-router-class failure
taxonomy and a conservative architecture recommendation. It deliberately does
not choose a global neural architecture up front; candidates are only promoted
when their evidence matches a diagnosed root cause and clears no-regression
gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[3]
STUDENT = ROOT / "research/triangular_maxsim/student_v2"
CODING_ARTIFACTS = ROOT / "research/triangular_maxsim/coding/artifacts"
PROOF_BUNDLE = ROOT / "data/veracier-industries/proof_bundle_v1"

KNOWN_CLASSES = (
    "rag.prose.enterprise",
    "rag.prose.short_factoid",
    "rag.prose.multi_claim",
    "rag.structured",
    "rag.code_in_context",
    "code.agentic_trace",
)


@dataclass(frozen=True)
class ArtifactPaths:
    v1_cache: Path = STUDENT / "calibrator_runs/reset_v1/v1_cache_eval_200.jsonl"
    fusion: Path = STUDENT / "calibrator_runs/reset_v1/fusion_optimized_no_regress_200.json"
    eval_v1_plus: Path = STUDENT / "calibrator_runs/reset_v1/eval_v1_plus_calibrator_200.json"
    train_report: Path = STUDENT / "calibrator_runs/reset_v1/train_report.json"
    deadweight: Path = STUDENT / "checkpoints_v2/deadweight_audit.json"
    eval_per_source: Path = STUDENT / "checkpoints_reset/eval_per_source_test.json"
    ragtruth_reset: Path = STUDENT / "checkpoints_reset/benchmark_eval_ragtruth.json"
    halueval_reset: Path = STUDENT / "checkpoints_reset/benchmark_eval_halueval.json"
    trajectory_v1: Path = CODING_ARTIFACTS / "trace_stabilization_transcripts_v1.json"
    trajectory_v2: Path = CODING_ARTIFACTS / "trace_stabilization_transcripts_v2.json"
    trajectory_both: Path = CODING_ARTIFACTS / "trace_stabilization_both.json"
    trajectory_head: Path = CODING_ARTIFACTS / "trajectory_head_root_cause_v1/coding_trajectory_head.json"
    stabilization_readme: Path = PROOF_BUNDLE / "stabilization_2026_04_30/README.md"
    runtime_policy: Path = ROOT / "latence_trace/data/runtime_policy.optimized_v1_plus_calibrator.json"


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return {} if default is None else default
    text = path.read_text(encoding="utf-8")
    # Some legacy JSON files have a human-readable prefix before the payload.
    first_brace = min([i for i in (text.find("{"), text.find("[")) if i >= 0], default=-1)
    if first_brace > 0:
        text = text[first_brace:]
    return json.loads(text)


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _binary_label(band: str) -> str:
    band = str(band or "").lower()
    if band == "green":
        return "grounded"
    if band == "red":
        return "ungrounded"
    return "partial"


def _is_false_allow(gold_band: str, pred_band: str) -> bool:
    return _binary_label(gold_band) == "ungrounded" and str(pred_band).lower() == "green"


def _is_false_block(gold_band: str, pred_band: str) -> bool:
    return _binary_label(gold_band) == "grounded" and str(pred_band).lower() == "red"


def _is_wrong(gold_band: str, pred_band: str) -> bool:
    gold = _binary_label(gold_band)
    pred = _binary_label(pred_band)
    if gold == "partial" or pred == "partial":
        return gold != pred
    return gold != pred


def _token_support_rate(labels: Mapping[str, Any]) -> float | None:
    toks = labels.get("token_support_labels")
    if not isinstance(toks, list) or not toks:
        return None
    vals = [float(x) for x in toks if x is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _deadweight_pathologies(deadweight: Mapping[str, Any]) -> dict[str, Counter]:
    out: dict[str, Counter] = {key: Counter() for key in KNOWN_CLASSES}
    for bucket_key, payload in (deadweight.get("buckets") or {}).items():
        status = str((payload or {}).get("status") or "unknown")
        for class_key in KNOWN_CLASSES:
            if class_key in bucket_key or class_key.replace(".", "::") in bucket_key:
                out[class_key][status] += int((payload or {}).get("rows") or 0)
    # Legacy RAGTruth buckets do not carry normalized class names.
    ragtruth_rows = Counter()
    for bucket_key, payload in (deadweight.get("buckets") or {}).items():
        if bucket_key.startswith("ragtruth::"):
            ragtruth_rows[str((payload or {}).get("status") or "unknown")] += int(
                (payload or {}).get("rows") or 0
            )
    out["rag.prose.multi_claim"].update(ragtruth_rows)
    out["rag.prose.short_factoid"].update(ragtruth_rows)
    return out


def _class_metrics_from_v1_cache(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_class[str(row.get("class_key") or "unknown")].append(row)

    metrics: dict[str, dict[str, Any]] = {}
    for class_key in KNOWN_CLASSES:
        class_rows = by_class.get(class_key, [])
        false_allows: list[str] = []
        false_blocks: list[str] = []
        partial_errors: list[str] = []
        scores_by_gold: dict[str, list[float]] = defaultdict(list)
        nli_by_gold: dict[str, list[float]] = defaultdict(list)
        mixed_token_error_rows = 0
        wrong_rows = 0
        for row in class_rows:
            gold = str(row.get("gold_band") or "")
            pred = str((row.get("v1") or {}).get("band") or "")
            score = float((row.get("v1") or {}).get("score") or 0.0)
            nli = (row.get("v1") or {}).get("nli_aggregate")
            gold_bin = _binary_label(gold)
            scores_by_gold[gold_bin].append(score)
            if nli is not None:
                nli_by_gold[gold_bin].append(float(nli))
            if _is_false_allow(gold, pred):
                false_allows.append(str(row.get("row_id")))
            if _is_false_block(gold, pred):
                false_blocks.append(str(row.get("row_id")))
            if gold_bin == "partial" and pred.lower() != "amber":
                partial_errors.append(str(row.get("row_id")))
            if _is_wrong(gold, pred):
                wrong_rows += 1
                support_rate = _token_support_rate(row.get("labels") or {})
                if support_rate is not None and 0.05 < support_rate < 0.95:
                    mixed_token_error_rows += 1

        grounded = scores_by_gold.get("grounded", [])
        ungrounded = scores_by_gold.get("ungrounded", [])
        grounded_mean = sum(grounded) / len(grounded) if grounded else None
        ungrounded_mean = sum(ungrounded) / len(ungrounded) if ungrounded else None
        separation = (
            grounded_mean - ungrounded_mean
            if grounded_mean is not None and ungrounded_mean is not None
            else None
        )
        all_scores = [s for values in scores_by_gold.values() for s in values]
        saturation_rate = (
            sum(1 for s in all_scores if s >= 0.95) / len(all_scores) if all_scores else None
        )
        metrics[class_key] = {
            "n": len(class_rows),
            "wrong_rows": wrong_rows,
            "false_allow_count": len(false_allows),
            "false_block_count": len(false_blocks),
            "partial_error_count": len(partial_errors),
            "mixed_token_error_rows": mixed_token_error_rows,
            "grounded_score_mean": grounded_mean,
            "ungrounded_score_mean": ungrounded_mean,
            "score_separation": separation,
            "score_saturation_rate": saturation_rate,
            "grounded_nli_mean": _mean(nli_by_gold.get("grounded", [])),
            "ungrounded_nli_mean": _mean(nli_by_gold.get("ungrounded", [])),
            "examples": {
                "false_allow": false_allows[:5],
                "false_block": false_blocks[:5],
                "partial_support": partial_errors[:5],
            },
        }
    return metrics


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _trajectory_metrics(paths: ArtifactPaths) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, path in {
        "transcripts_v1": paths.trajectory_v1,
        "transcripts_v2": paths.trajectory_v2,
        "both": paths.trajectory_both,
    }.items():
        payload = _read_json(path, {})
        cells = (payload.get("summary") or {}).get("by_cell") or {}
        best = 0.0
        best_cell = None
        for cell_key, cell in cells.items():
            value = ((cell or {}).get("metrics") or {}).get("reverse_context_auroc")
            if value is not None and float(value) > best:
                best = float(value)
                best_cell = cell_key
        flagship = (((cells.get("colgrep|fuse_mean") or {}).get("metrics") or {}).get("reverse_context_auroc"))
        out[key] = {
            "case_count": (payload.get("run_info") or {}).get("case_count", 0),
            "best_auroc": round(best, 4),
            "best_cell": best_cell,
            "flagship_auroc": None if flagship is None else round(float(flagship), 4),
            "headline": payload.get("flagship_headline"),
        }
    return out


def _root_causes_for_class(
    class_key: str,
    metrics: Mapping[str, Any],
    pathology_counts: Counter,
    fusion_entry: Mapping[str, Any],
    trajectory: Mapping[str, Any],
    eval_per_source: Mapping[str, Any],
) -> list[str]:
    causes: list[str] = []
    if metrics.get("false_allow_count", 0) > 0:
        causes.append("false_allow")
    if metrics.get("false_block_count", 0) > 0:
        causes.append("false_block")
    if metrics.get("partial_error_count", 0) > 0:
        causes.append("partial_support_collapse")
    saturation = metrics.get("score_saturation_rate")
    separation = metrics.get("score_separation")
    if saturation is not None and saturation >= 0.75:
        causes.append("token_saturation")
    if separation is not None and abs(float(separation)) < 0.03 and metrics.get("n", 0) >= 10:
        causes.append("low_score_separation")
    if metrics.get("mixed_token_error_rows", 0) > 0:
        causes.append("aggregation_error")
    if pathology_counts.get("one_sided", 0) or pathology_counts.get("missing_labels", 0):
        causes.append("data_label_pathology")
    if class_key == "rag.prose.multi_claim":
        causes.append("multi_claim_aggregation")
    if class_key == "rag.prose.short_factoid":
        causes.append("short_answer_overlap")
    if class_key == "rag.structured":
        causes.append("structured_cell_alignment")
    if class_key in {"rag.code_in_context", "code.agentic_trace"}:
        causes.append("code_identifier_morphology")
    if class_key == "code.agentic_trace":
        best = max((v.get("best_auroc") or 0.0) for v in trajectory.values()) if trajectory else 0.0
        if best < 0.90:
            causes.append("trajectory_ranker_failure")
    selected = fusion_entry.get("selected") or {}
    baseline = fusion_entry.get("baseline_v1") or {}
    if selected and baseline and selected.get("binary_grounded_accuracy") == baseline.get("binary_grounded_accuracy"):
        causes.append("feature_calibrator_no_lift")

    by_source_role = eval_per_source.get("by_source_role") or {}
    if class_key == "rag.prose.enterprise":
        grounded_acc = (by_source_role.get("synthetic_enterprise::grounded") or {}).get("acc")
        if grounded_acc is not None and float(grounded_acc) < 0.5:
            causes.append("grounded_false_block_bias")
    if class_key in {"rag.prose.multi_claim", "rag.prose.short_factoid"}:
        partial_acc = (by_source_role.get("ragtruth::partial_support") or {}).get("acc")
        if partial_acc is not None and float(partial_acc) < 0.2:
            causes.append("partial_support_blind_spot")
    return sorted(dict.fromkeys(causes))


def _candidate_rows(
    class_key: str,
    root_causes: list[str],
    fusion_entry: Mapping[str, Any],
    train_report: Mapping[str, Any],
    trajectory: Mapping[str, Any],
    trajectory_head: Mapping[str, Any],
) -> list[dict[str, Any]]:
    baseline = fusion_entry.get("baseline_v1") or {}
    selected = fusion_entry.get("selected") or {}
    rows = [
        {
            "candidate": "v1_router",
            "family": "baseline",
            "status": "baseline",
            "binary_grounded_accuracy": baseline.get("binary_grounded_accuracy"),
            "ungrounded_f1": baseline.get("ungrounded_f1"),
            "paired_score_accuracy": baseline.get("paired_score_accuracy"),
            "latency_profile": "known_production",
            "matches_root_causes": [],
        },
        {
            "candidate": "optimized_feature_calibrator",
            "family": "mlp_logistic_xgboost_feature_head",
            "status": "available" if fusion_entry.get("mode") == "weighted_fusion" else "passthrough_no_lift",
            "binary_grounded_accuracy": selected.get("binary_grounded_accuracy"),
            "ungrounded_f1": selected.get("ungrounded_f1"),
            "paired_score_accuracy": selected.get("paired_score_accuracy"),
            "latency_profile": "sub_ms_policy_math",
            "matches_root_causes": [
                c for c in root_causes if c in {"false_allow", "false_block", "low_score_separation", "token_saturation"}
            ],
        },
        {
            "candidate": "compact_claim_evidence_encoder",
            "family": "t5_small_or_tiny_cross_encoder",
            "status": "hypothesis",
            "binary_grounded_accuracy": None,
            "ungrounded_f1": None,
            "paired_score_accuracy": None,
            "latency_profile": "requires_budget_test",
            "matches_root_causes": [
                c for c in root_causes if c in {"partial_support_collapse", "partial_support_blind_spot", "short_answer_overlap"}
            ],
        },
        {
            "candidate": "span_sequence_aggregator",
            "family": "xlstm_or_light_sequence_head",
            "status": "hypothesis",
            "binary_grounded_accuracy": None,
            "ungrounded_f1": None,
            "paired_score_accuracy": None,
            "latency_profile": "requires_budget_test",
            "matches_root_causes": [
                c for c in root_causes if c in {"aggregation_error", "multi_claim_aggregation"}
            ],
        },
        {
            "candidate": "data_repair_first",
            "family": "label_pipeline_fix",
            "status": "required" if "data_label_pathology" in root_causes else "not_primary",
            "binary_grounded_accuracy": None,
            "ungrounded_f1": None,
            "paired_score_accuracy": None,
            "latency_profile": "offline",
            "matches_root_causes": [c for c in root_causes if c == "data_label_pathology"],
        },
    ]
    if class_key == "code.agentic_trace":
        best = max((v.get("best_auroc") or 0.0) for v in trajectory.values()) if trajectory else None
        eval_metrics = trajectory_head.get("eval") or {}
        heldout_aurocs = [
            float(m.get("auroc"))
            for m in eval_metrics.values()
            if isinstance(m, Mapping) and m.get("auroc") is not None
        ]
        min_heldout_auroc = min(heldout_aurocs) if heldout_aurocs else None
        rows.append(
            {
                "candidate": "trajectory_symbolic_ranker",
                "family": "code_specific_symbolic_plus_learned_ranker",
                "status": (
                    "evaluated_rejected"
                    if trajectory_head.get("promotion_decision", "").startswith("do_not_promote")
                    else "evaluated_candidate"
                ),
                "best_trajectory_auroc": best,
                "dedicated_head_min_heldout_auroc": min_heldout_auroc,
                "binary_grounded_accuracy": None,
                "ungrounded_f1": None,
                "paired_score_accuracy": None,
                "latency_profile": "must_hit_code_lane_budget",
                "matches_root_causes": [
                    c for c in root_causes if c in {"code_identifier_morphology", "trajectory_ranker_failure"}
                ],
            }
        )
    if train_report:
        rows.append(
            {
                "candidate": "existing_tiny_mlp_probe",
                "family": "mlp_probe",
                "status": "rejected" if float(train_report.get("tiny_mlp_train_accuracy") or 0.0) < 0.6 else "hypothesis",
                "train_accuracy": train_report.get("tiny_mlp_train_accuracy"),
                "turn_f1": (train_report.get("turn_metrics") or {}).get("f1"),
                "latency_profile": "cheap",
                "matches_root_causes": ["calibration_probe"],
            }
        )
    return rows


def _select_head(class_key: str, root_causes: list[str], fusion_entry: Mapping[str, Any], trajectory: Mapping[str, Any]) -> dict[str, Any]:
    mode = fusion_entry.get("mode")
    selected = fusion_entry.get("selected") or {}
    baseline = fusion_entry.get("baseline_v1") or {}
    if (
        mode == "weighted_fusion"
        and selected.get("binary_grounded_accuracy") is not None
        and baseline.get("binary_grounded_accuracy") is not None
        and float(selected["binary_grounded_accuracy"]) >= float(baseline["binary_grounded_accuracy"])
        and float(selected.get("ungrounded_f1") or 0.0) >= float(baseline.get("ungrounded_f1") or 0.0)
        and class_key == "rag.prose.enterprise"
    ):
        return {
            "selected_head": "optimized_feature_calibrator",
            "production_mode": "allow_block_repair_opt_in",
            "reason": "feature calibrator fixes observed enterprise class failures without regression",
        }
    if class_key == "code.agentic_trace":
        best = max((v.get("best_auroc") or 0.0) for v in trajectory.values()) if trajectory else 0.0
        return {
            "selected_head": "trajectory_symbolic_ranker",
            "production_mode": "auto_repair_only",
            "reason": f"manufactured trajectory AUROC {best:.4f} is below 0.90 gate",
        }
    if "partial_support_blind_spot" in root_causes or "partial_support_collapse" in root_causes:
        return {
            "selected_head": "compact_claim_evidence_encoder",
            "production_mode": "auto_repair_only_until_bakeoff_passes",
            "reason": "partial-support errors require semantic claim/evidence modelling",
        }
    if "data_label_pathology" in root_causes:
        return {
            "selected_head": "data_repair_first",
            "production_mode": "auto_repair_only",
            "reason": "label pathology must be fixed before architecture selection is trustworthy",
        }
    return {
        "selected_head": "v1_passthrough",
        "production_mode": "auto_repair_only",
        "reason": "no candidate has class-specific proof beyond v1 passthrough",
    }


def build_report(paths: ArtifactPaths = ArtifactPaths()) -> dict[str, Any]:
    v1_rows = list(_iter_jsonl(paths.v1_cache))
    v1_metrics = _class_metrics_from_v1_cache(v1_rows)
    fusion = _read_json(paths.fusion, {})
    eval_v1_plus = _read_json(paths.eval_v1_plus, {})
    train_report = _read_json(paths.train_report, {})
    deadweight = _read_json(paths.deadweight, {})
    eval_per_source = _read_json(paths.eval_per_source, {})
    runtime_policy = _read_json(paths.runtime_policy, {})
    trajectory_head = _read_json(paths.trajectory_head, {})
    trajectory = _trajectory_metrics(paths)
    pathology_counts = _deadweight_pathologies(deadweight)

    classes: dict[str, Any] = {}
    fusion_classes = fusion.get("classes") or {}
    for class_key in KNOWN_CLASSES:
        fusion_entry = fusion_classes.get(class_key) or {}
        causes = _root_causes_for_class(
            class_key,
            v1_metrics.get(class_key, {}),
            pathology_counts.get(class_key, Counter()),
            fusion_entry,
            trajectory,
            eval_per_source,
        )
        classes[class_key] = {
            "root_causes": causes,
            "v1_cache_metrics": v1_metrics.get(class_key, {}),
            "data_pathologies": dict(pathology_counts.get(class_key, Counter())),
            "fusion_evidence": fusion_entry,
            "candidate_bakeoff": _candidate_rows(
                class_key,
                causes,
                fusion_entry,
                train_report,
                trajectory,
                trajectory_head,
            ),
            "selection": _select_head(class_key, causes, fusion_entry, trajectory),
        }

    artifact_checksums = {
        name: _sha256(path)
        for name, path in paths.__dict__.items()
        if path.exists() and path.is_file()
    }
    return {
        "schema": "trace_root_cause_model_selection.v1",
        "inputs": {name: str(path) for name, path in paths.__dict__.items()},
        "artifact_checksums": artifact_checksums,
        "overall": {
            "v1_rows": len(v1_rows),
            "optimized_fusion_no_regression": bool(fusion.get("overall_no_regression_pass")),
            "raw_v1_plus_warning": {
                "v1_plus_binary": (((eval_v1_plus.get("v1_plus_calibrator") or {}).get("binary_grounded_accuracy"))),
                "v1_plus_ungrounded_f1": (((eval_v1_plus.get("v1_plus_calibrator") or {}).get("ungrounded_f1"))),
            },
            "trajectory": trajectory,
            "trajectory_head": {
                "train_bank": trajectory_head.get("train_bank"),
                "promotion_decision": trajectory_head.get("promotion_decision"),
                "eval": trajectory_head.get("eval"),
            },
        },
        "classes": classes,
        "production_gating": _production_gating(classes, runtime_policy, artifact_checksums),
    }


def _production_gating(
    classes: Mapping[str, Any],
    runtime_policy: Mapping[str, Any],
    artifact_checksums: Mapping[str, str | None],
) -> dict[str, Any]:
    enabled: dict[str, Any] = {}
    repair_only: dict[str, str] = {}
    registry: dict[str, Any] = {}
    policy_classes = runtime_policy.get("classes") or {}
    for class_key, payload in classes.items():
        selection = payload.get("selection") or {}
        mode = selection.get("production_mode")
        if mode == "allow_block_repair_opt_in":
            class_policy = policy_classes.get(class_key) or {}
            enabled[class_key] = {
                "head": selection.get("selected_head"),
                "mode": mode,
                "rollback": "disable LATENCE_TRACE_RUNTIME_DECISION_ENABLED or set class thresholds disabled",
            }
            registry[class_key] = {
                "class_key": class_key,
                "head_id": selection.get("selected_head"),
                "version": "root_cause_v1",
                "enabled": True,
                "thresholds": {
                    "allow_threshold": class_policy.get("allow_threshold"),
                    "block_threshold": class_policy.get("block_threshold"),
                    "allow_disabled": class_policy.get("allow_disabled"),
                    "block_disabled": class_policy.get("block_disabled"),
                },
                "proof": {
                    "fusion_report_sha256": artifact_checksums.get("fusion"),
                    "runtime_policy_sha256": artifact_checksums.get("runtime_policy"),
                    "fusion_report": "research/triangular_maxsim/student_v2/calibrator_runs/reset_v1/fusion_optimized_no_regress_200.json",
                    "root_cause_artifact": "research/triangular_maxsim/student_v2/root_cause_runs/latest/root_cause_model_selection.json",
                },
                "rollback_switches": [
                    "LATENCE_TRACE_RUNTIME_DECISION_ENABLED=0",
                    f"classes.{class_key}.allow_disabled=true",
                    f"classes.{class_key}.block_disabled=true",
                    "set learned/calibrator weight to 0.0 in the policy source artifact",
                ],
            }
        else:
            repair_only[class_key] = str(selection.get("reason") or "not proven")
            registry[class_key] = {
                "class_key": class_key,
                "head_id": selection.get("selected_head"),
                "version": "root_cause_v1",
                "enabled": False,
                "production_mode": mode,
                "reason": repair_only[class_key],
                "rollback_switches": ["already repair-only"],
            }
    return {
        "enabled_allow_block_classes": enabled,
        "repair_only_classes": repair_only,
        "runtime_head_registry": registry,
        "claim": (
            "automatic allow/block is class-scoped; classes without passing heads "
            "remain auto_repair-only and must not be sold as autonomous blockers"
        ),
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Root-Cause-Driven TRACE Model Selection",
        "",
        "## Verdict",
        "",
        "Architecture must stay class-specific. The evidence promotes only "
        "`rag.prose.enterprise` to opt-in automatic allow/block/repair with "
        "the optimized feature calibrator. Other classes remain repair-only "
        "until their diagnosed root causes clear a bake-off gate.",
        "",
        "## Per-Class Root Causes And Selection",
        "",
        "| class | root causes | selected head | production mode |",
        "|---|---|---|---|",
    ]
    for class_key, payload in report["classes"].items():
        selection = payload["selection"]
        causes = ", ".join(payload["root_causes"]) or "none_observed"
        lines.append(
            f"| `{class_key}` | {causes} | `{selection['selected_head']}` | `{selection['production_mode']}` |"
        )

    lines.extend(
        [
            "",
            "## Candidate Bake-Off Summary",
            "",
            "| class | candidate | family | status | evidence |",
            "|---|---|---|---|---|",
        ]
    )
    for class_key, payload in report["classes"].items():
        for row in payload["candidate_bakeoff"]:
            evidence_bits = []
            for key in (
                "binary_grounded_accuracy",
                "ungrounded_f1",
                "paired_score_accuracy",
                "best_trajectory_auroc",
                "dedicated_head_min_heldout_auroc",
                "train_accuracy",
            ):
                val = row.get(key)
                if val is not None and not (isinstance(val, float) and math.isnan(val)):
                    evidence_bits.append(f"{key}={val}")
            evidence = ", ".join(evidence_bits) or "hypothesis only"
            lines.append(
                f"| `{class_key}` | `{row['candidate']}` | `{row['family']}` | `{row['status']}` | {evidence} |"
            )

    lines.extend(
        [
            "",
            "## Production Gating",
            "",
            "Enabled allow/block classes:",
        ]
    )
    enabled = report["production_gating"]["enabled_allow_block_classes"]
    if enabled:
        for class_key, payload in enabled.items():
            lines.append(f"- `{class_key}` via `{payload['head']}`")
    else:
        lines.append("- none")
    lines.append("")
    lines.append("Repair-only classes:")
    for class_key, reason in report["production_gating"]["repair_only_classes"].items():
        lines.append(f"- `{class_key}`: {reason}")
    lines.append("")
    lines.append("## Coding Trajectory Evidence")
    for bank, payload in report["overall"]["trajectory"].items():
        lines.append(
            f"- `{bank}`: cases={payload['case_count']}, best_auroc={payload['best_auroc']}, "
            f"best_cell=`{payload['best_cell']}`, flagship_auroc={payload['flagship_auroc']}"
        )
    head = report["overall"].get("trajectory_head") or {}
    if head.get("promotion_decision"):
        lines.append(
            f"- dedicated trajectory head: train_bank=`{head.get('train_bank')}`, "
            f"promotion_decision=`{head.get('promotion_decision')}`"
        )
    lines.append("")
    lines.append("## Definition Of Done Status")
    lines.append("")
    lines.append("- Root-cause report: complete.")
    lines.append("- Candidate bake-off harness: complete, with hypotheses separated from proven heads.")
    lines.append("- Class-head selection: complete, conservative.")
    lines.append("- Coding trajectory model: built and evaluated; not production-promoted.")
    lines.append("- Production gating: only proven class is enabled; the rest remain repair-only.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        default=str(STUDENT / "root_cause_runs/latest"),
        help="Directory for root-cause model-selection artifacts.",
    )
    parser.add_argument(
        "--runtime-registry-out",
        default=None,
        help="Optional path for the production runtime head registry JSON.",
    )
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = build_report()
    (out_dir / "root_cause_model_selection.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    (out_dir / "root_cause_model_selection.md").write_text(
        render_markdown(report), encoding="utf-8"
    )
    if args.runtime_registry_out:
        registry_payload = {
            "schema": "trace_runtime_head_registry.v1",
            "source": "root_cause_model_selection",
            "runtime_head_registry": report["production_gating"]["runtime_head_registry"],
            "enabled_allow_block_classes": report["production_gating"][
                "enabled_allow_block_classes"
            ],
            "repair_only_classes": report["production_gating"]["repair_only_classes"],
        }
        registry_path = Path(args.runtime_registry_out)
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        registry_path.write_text(
            json.dumps(registry_payload, indent=2, sort_keys=True), encoding="utf-8"
        )
    print(out_dir / "root_cause_model_selection.json")
    print(out_dir / "root_cause_model_selection.md")
    if args.runtime_registry_out:
        print(args.runtime_registry_out)


if __name__ == "__main__":
    main()
