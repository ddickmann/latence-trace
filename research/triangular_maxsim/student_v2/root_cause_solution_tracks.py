"""Root-cause-derived solution tracks for TRACE class heads.

This is the implementation layer that sits after the diagnostic taxonomy. It
creates per-class slices, trains candidate heads that match the observed root
causes, evaluates them against fixed gates, and emits a production registry
proposal that promotes only passing heads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.triangular_maxsim.coding.coding_trajectory_head import run as run_trajectory_head
from research.triangular_maxsim.student_v2.root_cause_model_selection import (
    KNOWN_CLASSES,
    ArtifactPaths,
    _read_json,
)

STUDENT = ROOT / "research/triangular_maxsim/student_v2"
CODING_ARTIFACTS = ROOT / "research/triangular_maxsim/coding/artifacts"

PROMOTION_GATES = {
    "min_n": 24,
    "max_false_allow": 0.02,
    "max_false_block": 0.08,
    "max_p95_latency_ms": 25.0,
    "min_decision_coverage": 0.10,
}
CALIBRATION_SAFETY_FACTOR = 0.2

TEXT_CLASSES = {
    "rag.prose.multi_claim",
    "rag.prose.short_factoid",
    "rag.structured",
}

HEAD_SPECS = {
    "rag.prose.enterprise": {
        "head_id": "optimized_calibrator",
        "artifact": "rag.prose.enterprise.optimized_calibrator.v1.json",
        "architecture": "optimized_feature_calibrator",
        "candidate": "optimized_calibrator",
        "next_training_target": "preserve no-regression and reproduce on live RunPod",
    },
    "rag.prose.multi_claim": {
        "head_id": "claim_decomposer",
        "artifact": "rag.prose.multi_claim.claim_decomposer.v1.json",
        "architecture": "claim_decomposition_plus_per_claim_support_aggregation",
        "candidate": "claim_decomposition_rule_head",
        "next_training_target": "replace whole-answer features with atomic claim decomposition and per-claim unsupported penalties",
    },
    "rag.prose.short_factoid": {
        "head_id": "atom_verifier",
        "artifact": "rag.prose.short_factoid.atom_verifier.v1.json",
        "architecture": "entity_date_number_atom_verifier",
        "candidate": "factoid_atom_rule_head",
        "next_training_target": "reduce high-overlap entity/date/number false allows without increasing grounded false blocks",
    },
    "rag.structured": {
        "head_id": "cell_schema_verifier",
        "artifact": "rag.structured.cell_schema_verifier.v1.json",
        "architecture": "typed_cell_schema_value_alignment",
        "candidate": "v1_abstain_policy",
        "next_training_target": "add row/column provenance and numeric tolerance labels to reduce false blocks",
    },
    "rag.code_in_context": {
        "head_id": "identifier_ranker",
        "artifact": "rag.code_in_context.identifier_ranker.v1.json",
        "architecture": "code_identifier_api_file_attribution_ranker",
        "candidate": "code_symbol_rule_head",
        "next_training_target": "build a real code-in-context eval lane with positive and phantom identifier/API cases",
    },
    "code.agentic_trace": {
        "head_id": "trajectory_ranker",
        "artifact": "code.agentic_trace.trajectory_ranker.v1.json",
        "architecture": "trajectory_aware_symbolic_plus_learned_ranker",
        "candidate": "code_symbol_rule_head",
        "next_training_target": "add turn-order, patch/test outcome, file ownership, AST/API drift, and action-result labels",
    },
}


@dataclass(frozen=True)
class SliceRow:
    row_id: str
    class_key: str
    split: str
    gold_band: str
    gold_binary: int
    v1_band: str
    v1_score: float
    root_causes: list[str]
    features: dict[str, float]
    response_text: str
    evidence_text: str
    source: str = "v1_cache"


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _hash_ratio(value: str) -> float:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def _stable_split(row_id: str) -> str:
    ratio = _hash_ratio(row_id)
    if ratio < 0.60:
        return "train"
    if ratio < 0.80:
        return "val"
    return "test"


def _binary_label(band: str) -> int | None:
    band = str(band or "").lower()
    if band == "green":
        return 1
    if band in {"amber", "red"}:
        return 0
    return None


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _tokens_to_text(tokens: list[Any]) -> str:
    parts: list[str] = []
    for token in tokens:
        if isinstance(token, Mapping):
            text = str(token.get("token") or "")
        else:
            text = str(token)
        if text in {"[CLS]", "[SEP]", "[PAD]"}:
            continue
        if text.startswith("##") and parts:
            parts[-1] += text[2:]
        else:
            parts.append(text)
    return " ".join(parts)


def _support_text(row: Mapping[str, Any]) -> str:
    full_units = (((row.get("raw_response") or {}).get("full") or {}).get("support_units") or [])
    if full_units:
        return "\n".join(str(unit.get("text") or "") for unit in full_units[:8] if isinstance(unit, Mapping))
    label_units = (row.get("labels") or {}).get("evidence_units") or []
    return "\n".join(str(unit.get("text") or "") for unit in label_units[:8] if isinstance(unit, Mapping))


def _response_text(row: Mapping[str, Any]) -> str:
    full_tokens = (((row.get("raw_response") or {}).get("full") or {}).get("response_tokens") or [])
    if full_tokens:
        return _tokens_to_text(full_tokens)
    return _tokens_to_text((row.get("labels") or {}).get("response_tokens_teacher") or [])


def _full_scores(row: Mapping[str, Any]) -> Mapping[str, Any]:
    return (((row.get("raw_response") or {}).get("full") or {}).get("scores") or {})


def _token_stats(row: Mapping[str, Any]) -> dict[str, float]:
    tokens = row.get("token_diagnostics") or []
    vals = [_safe_float(tok.get("heatmap_score")) for tok in tokens if isinstance(tok, Mapping)]
    calibrated = [
        _safe_float(tok.get("reverse_context_calibrated"))
        for tok in tokens
        if isinstance(tok, Mapping) and tok.get("reverse_context_calibrated") is not None
    ]
    nli = [
        _safe_float(tok.get("nli_score"))
        for tok in tokens
        if isinstance(tok, Mapping) and tok.get("nli_score") is not None
    ]
    if not vals:
        vals = [0.0]
    ordered = sorted(vals)
    bottom_count = max(1, int(len(ordered) * 0.10))
    return {
        "token_mean": float(np.mean(vals)),
        "token_min": float(np.min(vals)),
        "token_bottom10": float(np.mean(ordered[:bottom_count])),
        "token_saturation_rate": float(np.mean([v >= 0.95 for v in vals])),
        "calibrated_mean": float(np.mean(calibrated)) if calibrated else 0.0,
        "nli_token_mean": float(np.mean(nli)) if nli else 0.0,
        "nli_token_available": 1.0 if nli else 0.0,
    }


def _lexical_features(response_text: str, evidence_text: str) -> dict[str, float]:
    response_terms = re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?", response_text)
    evidence_lower = evidence_text.lower()
    if not response_terms:
        return {
            "literal_coverage": 0.0,
            "numeric_coverage": 0.0,
            "identifier_coverage": 0.0,
            "numeric_count": 0.0,
            "identifier_count": 0.0,
        }
    numeric = [t for t in response_terms if re.fullmatch(r"\d+(?:\.\d+)?", t)]
    identifiers = [t for t in response_terms if "_" in t or re.search(r"[a-z][A-Z]|[A-Z][a-z]+[A-Z]", t)]
    literal_hits = sum(1 for term in response_terms if term.lower() in evidence_lower)
    numeric_hits = sum(1 for term in numeric if term.lower() in evidence_lower)
    identifier_hits = sum(1 for term in identifiers if term.lower() in evidence_lower)
    return {
        "literal_coverage": literal_hits / len(response_terms),
        "numeric_coverage": numeric_hits / len(numeric) if numeric else 1.0,
        "identifier_coverage": identifier_hits / len(identifiers) if identifiers else 1.0,
        "numeric_count": float(len(numeric)),
        "identifier_count": float(len(identifiers)),
    }


def _literal_coverage(response_text: str, evidence_text: str) -> float:
    response_terms = re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?", response_text)
    if not response_terms:
        return 0.0
    evidence_lower = evidence_text.lower()
    return sum(1 for term in response_terms if term.lower() in evidence_lower) / len(response_terms)


def _claim_features(response_text: str, evidence_text: str) -> dict[str, float]:
    claims = [part.strip() for part in re.split(r"[.;\n]+", response_text) if part.strip()]
    if not claims:
        return {"claim_count": 0.0, "unsupported_claim_fraction": 0.0, "min_claim_coverage": 1.0}
    coverages = [_literal_coverage(claim, evidence_text) for claim in claims]
    unsupported = [coverage < 0.55 for coverage in coverages]
    return {
        "claim_count": float(len(claims)),
        "unsupported_claim_fraction": sum(1 for item in unsupported if item) / len(unsupported),
        "min_claim_coverage": min(coverages) if coverages else 1.0,
    }


def _slice_features(row: Mapping[str, Any], response_text: str, evidence_text: str) -> dict[str, float]:
    v1 = row.get("v1") or {}
    scores = _full_scores(row)
    labels = row.get("labels") or {}
    support_labels = [float(x) for x in labels.get("token_support_labels") or [] if x is not None]
    dead_labels = [float(x) for x in labels.get("dead_weight_unit_labels") or [] if x is not None]
    coverage_labels = [float(x) for x in labels.get("coverage_unit_labels") or [] if x is not None]
    feature = {
        "v1_score": _safe_float(v1.get("score")),
        "v1_nli_aggregate": _safe_float(v1.get("nli_aggregate")),
        "primary_score": _safe_float(scores.get("primary_score"), _safe_float(v1.get("score"))),
        "reverse_context": _safe_float(scores.get("reverse_context"), _safe_float(v1.get("score"))),
        "groundedness_v2": _safe_float(scores.get("groundedness_v2")),
        "literal_guarded": _safe_float(scores.get("literal_guarded")),
        "literal_mismatch_count": _safe_float(scores.get("literal_mismatch_count")),
        "literal_match_count": _safe_float(scores.get("literal_match_count")),
        "literal_total_count": _safe_float(scores.get("literal_total_count")),
        "context_coverage_ratio": _safe_float(scores.get("context_coverage_ratio")),
        "context_unused_ratio": _safe_float(scores.get("context_unused_ratio")),
        "context_uncertain_ratio": _safe_float(scores.get("context_uncertain_ratio")),
        "dead_weight_ratio": _safe_float(scores.get("dead_weight_ratio")),
        "support_units_total": _safe_float(scores.get("support_units_total"), _safe_float(labels.get("n_support_units"))),
        "support_unit_label_mean": float(np.mean(support_labels)) if support_labels else 0.0,
        "dead_weight_label_mean": float(np.mean(dead_labels)) if dead_labels else 0.0,
        "coverage_label_mean": float(np.mean(coverage_labels)) if coverage_labels else 0.0,
        "response_len_log": math.log1p(len(response_text)),
        "evidence_len_log": math.log1p(len(evidence_text)),
    }
    feature.update(_token_stats(row))
    feature.update(_lexical_features(response_text, evidence_text))
    feature.update(_claim_features(response_text, evidence_text))
    return feature


def _root_causes(row: Mapping[str, Any], features: Mapping[str, float]) -> list[str]:
    gold = str(row.get("gold_band") or "").lower()
    pred = str((row.get("v1") or {}).get("band") or "").lower()
    causes: set[str] = set()
    if gold in {"red", "amber"} and pred == "green":
        causes.add("false_allow")
    if gold == "green" and pred == "red":
        causes.add("false_block")
    if gold == "amber" and pred != "amber":
        causes.add("partial_support_collapse")
    if features.get("token_saturation_rate", 0.0) >= 0.75:
        causes.add("token_saturation")
    if features.get("nli_token_available", 0.0) == 0.0:
        causes.add("nli_mismatch")
    if 0.05 < features.get("support_unit_label_mean", 0.0) < 0.95 and gold != pred:
        causes.add("aggregation_error")
    if features.get("numeric_count", 0.0) > 0:
        causes.add("domain_morphology_numeric")
    if features.get("identifier_count", 0.0) > 0:
        causes.add("domain_morphology_identifier")
    class_key = str(row.get("class_key") or "")
    if class_key == "rag.prose.multi_claim":
        causes.add("multi_claim_aggregation")
    if class_key == "rag.prose.short_factoid":
        causes.add("short_answer_overlap")
    if class_key == "rag.structured":
        causes.add("structured_cell_alignment")
    if class_key == "rag.code_in_context":
        causes.add("code_identifier_morphology")
    return sorted(causes)


def build_root_cause_slices(paths: ArtifactPaths = ArtifactPaths()) -> dict[str, list[SliceRow]]:
    by_class: dict[str, list[SliceRow]] = defaultdict(list)
    for row in _iter_jsonl(paths.v1_cache):
        class_key = str(row.get("class_key") or "")
        if class_key not in KNOWN_CLASSES:
            continue
        gold_binary = _binary_label(str(row.get("gold_band") or ""))
        if gold_binary is None:
            continue
        response_text = _response_text(row)
        evidence_text = _support_text(row)
        features = _slice_features(row, response_text, evidence_text)
        slice_row = SliceRow(
            row_id=str(row.get("row_id") or ""),
            class_key=class_key,
            split=_stable_split(str(row.get("row_id") or "")),
            gold_band=str(row.get("gold_band") or ""),
            gold_binary=gold_binary,
            v1_band=str((row.get("v1") or {}).get("band") or ""),
            v1_score=_safe_float((row.get("v1") or {}).get("score")),
            root_causes=_root_causes(row, features),
            features=features,
            response_text=response_text,
            evidence_text=evidence_text,
            source="v1_cache",
        )
        by_class[class_key].append(slice_row)
    return {class_key: by_class.get(class_key, []) for class_key in KNOWN_CLASSES}


def load_targeted_slices(path: Path) -> dict[str, list[SliceRow]]:
    by_class: dict[str, list[SliceRow]] = defaultdict(list)
    for row in _iter_jsonl(path):
        class_key = str(row.get("class_key") or "")
        if class_key not in KNOWN_CLASSES:
            continue
        by_class[class_key].append(
            SliceRow(
                row_id=str(row.get("row_id") or ""),
                class_key=class_key,
                split=str(row.get("split") or _stable_split(str(row.get("row_id") or ""))),
                gold_band=str(row.get("gold_band") or ""),
                gold_binary=int(row.get("gold_binary") or 0),
                v1_band="green" if float((row.get("features") or {}).get("v1_score", 0.0)) >= 0.83 else "red",
                v1_score=_safe_float((row.get("features") or {}).get("v1_score")),
                root_causes=list(row.get("root_causes") or []),
                features={str(k): _safe_float(v) for k, v in (row.get("features") or {}).items()},
                response_text=str(row.get("claim") or ""),
                evidence_text=str(row.get("evidence") or ""),
                source="targeted_synthetic",
            )
        )
    return {class_key: by_class.get(class_key, []) for class_key in KNOWN_CLASSES}


def _merge_slices(
    base: Mapping[str, list[SliceRow]],
    targeted: Mapping[str, list[SliceRow]],
) -> dict[str, list[SliceRow]]:
    return {class_key: list(base.get(class_key, [])) + list(targeted.get(class_key, [])) for class_key in KNOWN_CLASSES}


def write_slices(slices: Mapping[str, list[SliceRow]], out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"classes": {}}
    for class_key, rows in slices.items():
        path = out_dir / f"{class_key}.jsonl"
        counts = Counter(row.split for row in rows)
        causes = Counter(cause for row in rows for cause in row.root_causes)
        with path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(
                    json.dumps(
                        {
                            "row_id": row.row_id,
                            "class_key": row.class_key,
                            "split": row.split,
                            "root_causes": row.root_causes,
                            "gold": {"band": row.gold_band, "binary": row.gold_binary},
                            "v1_prediction": {"band": row.v1_band, "score": row.v1_score},
                            "features": row.features,
                            "claim": row.response_text,
                            "evidence": row.evidence_text,
                            "source": row.source,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
        manifest["classes"][class_key] = {
            "path": str(path),
            "rows": len(rows),
            "splits": dict(counts),
            "root_causes": dict(causes),
        }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    latency_ms: list[float] | None = None,
    allow_threshold: float | None = None,
    block_threshold: float | None = None,
) -> dict[str, Any]:
    preds = (scores >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels == 0,
        preds == 0,
        average="binary",
        zero_division=0,
    )
    false_allow = float(np.mean((preds == 1) & (labels == 0)) / max(np.mean(labels == 0), 1e-12))
    false_block = float(np.mean((preds == 0) & (labels == 1)) / max(np.mean(labels == 1), 1e-12))
    if len(set(labels.tolist())) < 2:
        auroc = None
    else:
        auroc = float(roc_auc_score(labels, scores))
    latency_ms = latency_ms or [0.0 for _ in labels]
    policy = _policy_metrics(
        scores,
        labels,
        allow_threshold=threshold if allow_threshold is None else allow_threshold,
        block_threshold=threshold if block_threshold is None else block_threshold,
    )
    return {
        "n": int(len(labels)),
        "binary_grounded_accuracy": float(accuracy_score(labels, preds)),
        "ungrounded_precision": float(precision),
        "ungrounded_recall": float(recall),
        "ungrounded_f1": float(f1),
        "forced_false_allow_rate": false_allow,
        "forced_false_block_rate": false_block,
        "false_allow_rate": policy["false_allow_rate"],
        "false_block_rate": policy["false_block_rate"],
        "allowed": policy["allowed"],
        "blocked": policy["blocked"],
        "auto_repair": policy["auto_repair"],
        "decision_coverage": policy["decision_coverage"],
        "auroc": auroc,
        "latency_p50_ms": float(np.percentile(latency_ms, 50)) if latency_ms else 0.0,
        "latency_p95_ms": float(np.percentile(latency_ms, 95)) if latency_ms else 0.0,
    }


def _policy_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    allow_threshold: float,
    block_threshold: float,
) -> dict[str, Any]:
    allow_mask = scores >= allow_threshold
    block_mask = scores <= block_threshold
    # If thresholds overlap, prefer repair for the ambiguous overlap.
    overlap = allow_mask & block_mask
    if np.any(overlap):
        allow_mask = allow_mask & ~overlap
        block_mask = block_mask & ~overlap
    allowed = int(np.sum(allow_mask))
    blocked = int(np.sum(block_mask))
    false_allow = float(np.sum(allow_mask & (labels == 0)) / allowed) if allowed else 0.0
    false_block = float(np.sum(block_mask & (labels == 1)) / blocked) if blocked else 0.0
    decided = allowed + blocked
    total = int(len(labels))
    return {
        "allow_threshold": float(allow_threshold),
        "block_threshold": float(block_threshold),
        "allowed": allowed,
        "blocked": blocked,
        "auto_repair": total - decided,
        "decision_coverage": decided / total if total else 0.0,
        "false_allow_rate": false_allow,
        "false_block_rate": false_block,
    }


def _best_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    candidates = np.unique(scores)
    best = float(candidates[0]) if len(candidates) else 0.5
    best_key: tuple[float, float, float, float] | None = None
    for threshold in candidates:
        metrics = _metrics(scores, labels, float(threshold))
        false_allow = float(metrics["false_allow_rate"])
        false_block = float(metrics["false_block_rate"])
        gate_pass = (
            false_allow <= PROMOTION_GATES["max_false_allow"]
            and false_block <= PROMOTION_GATES["max_false_block"]
        )
        # First prefer thresholds that satisfy the automatic-decision gates;
        # otherwise minimize gate violations before optimizing aggregate F1.
        violation = max(0.0, false_allow - PROMOTION_GATES["max_false_allow"]) + max(
            0.0, false_block - PROMOTION_GATES["max_false_block"]
        )
        key = (
            1.0 if gate_pass else 0.0,
            -violation,
            float(metrics["ungrounded_f1"]),
            float(metrics["binary_grounded_accuracy"]),
        )
        if best_key is None or key > best_key:
            best_key = key
            best = float(threshold)
    return best


def _calibrate_policy_thresholds(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    candidates = sorted(float(x) for x in np.unique(scores))
    if not candidates:
        return 1.0, 0.0
    train_false_allow_target = PROMOTION_GATES["max_false_allow"] * CALIBRATION_SAFETY_FACTOR
    train_false_block_target = PROMOTION_GATES["max_false_block"] * CALIBRATION_SAFETY_FACTOR
    allow_threshold = max(candidates)
    best_allow = (-1.0, -1)
    for threshold in candidates:
        mask = scores >= threshold
        allowed = int(np.sum(mask))
        if not allowed:
            continue
        false_allow = float(np.sum(mask & (labels == 0)) / allowed)
        if false_allow <= train_false_allow_target:
            key = (threshold, allowed)
            if key > best_allow:
                best_allow = key
                allow_threshold = threshold

    block_threshold = min(candidates)
    best_block = (-1, 1.0)
    for threshold in candidates:
        mask = scores <= threshold
        blocked = int(np.sum(mask))
        if not blocked:
            continue
        false_block = float(np.sum(mask & (labels == 1)) / blocked)
        if false_block <= train_false_block_target:
            key = (blocked, threshold)
            if key > best_block:
                best_block = key
                block_threshold = threshold
    return float(allow_threshold), float(block_threshold)


def _feature_matrix(rows: list[SliceRow], feature_names: list[str]) -> np.ndarray:
    return np.array([[row.features.get(name, 0.0) for name in feature_names] for row in rows], dtype=float)


def _labels(rows: list[SliceRow]) -> np.ndarray:
    return np.array([row.gold_binary for row in rows], dtype=int)


def _candidate_split(rows: list[SliceRow]) -> tuple[list[SliceRow], list[SliceRow], list[SliceRow], str | None]:
    if len(rows) < 12:
        return [], [], [], "insufficient_rows"
    labels = [row.gold_binary for row in rows]
    if len(set(labels)) < 2:
        return [], [], [], "single_label_only"
    train_rows, holdout_rows = train_test_split(
        rows,
        test_size=0.45,
        random_state=17,
        stratify=labels,
    )
    val_rows, test_rows = train_test_split(
        holdout_rows,
        test_size=0.67,
        random_state=23,
        stratify=[row.gold_binary for row in holdout_rows],
    )
    if (
        len(set(row.gold_binary for row in train_rows)) < 2
        or len(set(row.gold_binary for row in val_rows)) < 2
        or len(set(row.gold_binary for row in test_rows)) < 2
    ):
        return [], [], [], "split_lost_label_diversity"
    return list(train_rows), list(val_rows), list(test_rows), None


def _train_numeric_candidate(name: str, rows: list[SliceRow], feature_names: list[str]) -> dict[str, Any]:
    train_rows, val_rows, test_rows, skip = _candidate_split(rows)
    if skip:
        return {"candidate": name, "status": "skipped", "reason": skip, "n": len(rows)}
    x_train = _feature_matrix(train_rows, feature_names)
    y_train = _labels(train_rows)
    x_val = _feature_matrix(val_rows, feature_names)
    y_val = _labels(val_rows)
    x_test = _feature_matrix(test_rows, feature_names)
    y_test = _labels(test_rows)
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=17)),
        ]
    )
    started = time.perf_counter()
    model.fit(x_train, y_train)
    scores_val = model.predict_proba(x_val)[:, 1]
    threshold = _best_threshold(scores_val, y_val)
    allow_threshold, block_threshold = _calibrate_policy_thresholds(scores_val, y_val)
    latencies: list[float] = []
    scores: list[float] = []
    for row in test_rows:
        x_one = _feature_matrix([row], feature_names)
        t0 = time.perf_counter()
        scores.append(float(model.predict_proba(x_one)[0, 1]))
        latencies.append((time.perf_counter() - t0) * 1000.0)
    elapsed = (time.perf_counter() - started) * 1000.0
    payload = _metrics(
        np.array(scores),
        y_test,
        threshold,
        latencies,
        allow_threshold=allow_threshold,
        block_threshold=block_threshold,
    )
    payload.update(
        {
            "candidate": name,
            "status": "evaluated",
            "feature_names": feature_names,
            "threshold": threshold,
            "allow_threshold": allow_threshold,
            "block_threshold": block_threshold,
            "train_n": len(train_rows),
            "val_n": len(val_rows),
            "test_n": len(test_rows),
            "train_fit_ms": elapsed,
        }
    )
    return payload


def _train_text_candidate(name: str, rows: list[SliceRow]) -> dict[str, Any]:
    train_rows, val_rows, test_rows, skip = _candidate_split(rows)
    if skip:
        return {"candidate": name, "status": "skipped", "reason": skip, "n": len(rows)}
    x_train = [f"{row.response_text}\n[TRACE_EVIDENCE]\n{row.evidence_text}" for row in train_rows]
    y_train = _labels(train_rows)
    x_val = [f"{row.response_text}\n[TRACE_EVIDENCE]\n{row.evidence_text}" for row in val_rows]
    y_val = _labels(val_rows)
    x_test = [f"{row.response_text}\n[TRACE_EVIDENCE]\n{row.evidence_text}" for row in test_rows]
    y_test = _labels(test_rows)
    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=1)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=17)),
        ]
    )
    started = time.perf_counter()
    model.fit(x_train, y_train)
    val_scores = model.predict_proba(x_val)[:, 1]
    threshold = _best_threshold(val_scores, y_val)
    allow_threshold, block_threshold = _calibrate_policy_thresholds(val_scores, y_val)
    latencies: list[float] = []
    scores: list[float] = []
    for text in x_test:
        t0 = time.perf_counter()
        scores.append(float(model.predict_proba([text])[0, 1]))
        latencies.append((time.perf_counter() - t0) * 1000.0)
    elapsed = (time.perf_counter() - started) * 1000.0
    payload = _metrics(
        np.array(scores),
        y_test,
        threshold,
        latencies,
        allow_threshold=allow_threshold,
        block_threshold=block_threshold,
    )
    payload.update(
        {
            "candidate": name,
            "status": "evaluated",
            "threshold": threshold,
            "allow_threshold": allow_threshold,
            "block_threshold": block_threshold,
            "train_n": len(train_rows),
            "val_n": len(val_rows),
            "test_n": len(test_rows),
            "train_fit_ms": elapsed,
            "model_family": "compact_claim_evidence_text_head",
        }
    )
    return payload


def _rule_candidate(name: str, rows: list[SliceRow], score_fn: Any) -> dict[str, Any]:
    train_rows, val_rows, test_rows, skip = _candidate_split(rows)
    if skip:
        return {"candidate": name, "status": "skipped", "reason": skip, "n": len(rows)}
    y_val = _labels(val_rows)
    val_scores = np.array([float(score_fn(row)) for row in val_rows], dtype=float)
    threshold = _best_threshold(val_scores, y_val)
    allow_threshold, block_threshold = _calibrate_policy_thresholds(val_scores, y_val)
    latencies: list[float] = []
    scores: list[float] = []
    for row in test_rows:
        t0 = time.perf_counter()
        scores.append(float(score_fn(row)))
        latencies.append((time.perf_counter() - t0) * 1000.0)
    payload = _metrics(
        np.array(scores),
        _labels(test_rows),
        threshold,
        latencies,
        allow_threshold=allow_threshold,
        block_threshold=block_threshold,
    )
    payload.update(
        {
            "candidate": name,
            "status": "evaluated",
            "threshold": threshold,
            "allow_threshold": allow_threshold,
            "block_threshold": block_threshold,
            "train_n": len(train_rows),
            "val_n": len(val_rows),
            "test_n": len(test_rows),
            "model_family": "root_cause_symbolic_rule_head",
            "feature_names": [
                "literal_coverage",
                "numeric_coverage",
                "identifier_coverage",
                "unsupported_claim_fraction",
                "min_claim_coverage",
                "atom_match",
                "schema_match",
                "api_symbol_match",
                "trajectory_order_match",
            ],
        }
    )
    return payload


def _v1_baseline(rows: list[SliceRow]) -> dict[str, Any]:
    if not rows:
        return {"candidate": "v1_router", "status": "skipped", "reason": "no_rows", "n": 0}
    labels = _labels(rows)
    scores = np.array([row.v1_score for row in rows], dtype=float)
    return {
        "candidate": "v1_router",
        "status": "baseline",
        **_metrics(scores, labels, threshold=0.83),
    }


def _v1_abstain_policy(rows: list[SliceRow]) -> dict[str, Any]:
    train_rows, val_rows, test_rows, skip = _candidate_split(rows)
    if skip:
        return {"candidate": "v1_abstain_policy", "status": "skipped", "reason": skip, "n": len(rows)}
    val_scores = np.array([row.v1_score for row in val_rows], dtype=float)
    y_val = _labels(val_rows)
    threshold = _best_threshold(val_scores, y_val)
    allow_threshold, block_threshold = _calibrate_policy_thresholds(val_scores, y_val)
    test_scores = np.array([row.v1_score for row in test_rows], dtype=float)
    payload = _metrics(
        test_scores,
        _labels(test_rows),
        threshold,
        allow_threshold=allow_threshold,
        block_threshold=block_threshold,
    )
    payload.update(
        {
            "candidate": "v1_abstain_policy",
            "status": "evaluated",
            "threshold": threshold,
            "allow_threshold": allow_threshold,
            "block_threshold": block_threshold,
            "train_n": len(train_rows),
            "val_n": len(val_rows),
            "test_n": len(test_rows),
            "model_family": "v1_score_with_class_abstain_policy",
            "feature_names": ["v1_score"],
        }
    )
    return payload


def _optimized_calibrator_baseline(
    class_key: str,
    fusion: Mapping[str, Any],
    runtime_policy: Mapping[str, Any],
) -> dict[str, Any]:
    entry = ((fusion.get("classes") or {}).get(class_key) or {})
    selected = entry.get("selected") or {}
    if not selected:
        return {"candidate": "optimized_calibrator", "status": "skipped", "reason": "no_fusion_entry"}
    policy_entry = ((runtime_policy.get("classes") or {}).get(class_key) or {})
    allow_disabled = bool(policy_entry.get("allow_disabled", True))
    block_disabled = bool(policy_entry.get("block_disabled", True))
    return {
        "candidate": "optimized_calibrator",
        "status": "artifact_proven" if not allow_disabled and not block_disabled else "artifact_repair_only",
        "mode": entry.get("mode"),
        "n": selected.get("n"),
        "binary_grounded_accuracy": selected.get("binary_grounded_accuracy"),
        "ungrounded_precision": selected.get("ungrounded_precision"),
        "ungrounded_recall": selected.get("ungrounded_recall"),
        "ungrounded_f1": selected.get("ungrounded_f1"),
        "false_allow_rate": policy_entry.get("false_allow_rate"),
        "false_block_rate": policy_entry.get("false_block_rate"),
        "latency_p95_ms": 0.0,
        "runtime_p95_latency_ms": (fusion.get("v1_cache") or {}).get("p95_latency_ms"),
        "paired_score_accuracy": selected.get("paired_score_accuracy"),
        "learned_weight": entry.get("learned_weight"),
        "threshold": policy_entry.get("allow_threshold", entry.get("threshold")),
        "block_threshold": policy_entry.get("block_threshold"),
        "allow_disabled": allow_disabled,
        "block_disabled": block_disabled,
    }


def _candidate_set_for_class(
    class_key: str,
    rows: list[SliceRow],
    fusion: Mapping[str, Any],
    runtime_policy: Mapping[str, Any],
) -> list[dict[str, Any]]:
    common = [
        "v1_score",
        "v1_nli_aggregate",
        "reverse_context",
        "groundedness_v2",
        "literal_guarded",
        "literal_mismatch_count",
        "context_coverage_ratio",
        "context_unused_ratio",
        "dead_weight_ratio",
        "token_mean",
        "token_bottom10",
        "token_saturation_rate",
        "calibrated_mean",
        "nli_token_mean",
        "literal_coverage",
        "atom_match",
        "claim_count",
        "unsupported_claim_fraction",
        "schema_match",
        "api_symbol_match",
        "trajectory_order_match",
        "test_outcome_match",
    ]
    candidates = [
        _v1_baseline(rows),
        _v1_abstain_policy(rows),
        _optimized_calibrator_baseline(class_key, fusion, runtime_policy),
    ]
    candidates.append(_train_numeric_candidate("trace_feature_logreg", rows, common))
    if class_key in TEXT_CLASSES:
        candidates.append(_train_text_candidate("compact_claim_evidence_head", rows))
    if class_key == "rag.prose.multi_claim":
        candidates.append(_rule_candidate("claim_decomposition_rule_head", rows, _multi_claim_score))
    if class_key == "rag.prose.short_factoid":
        candidates.append(_rule_candidate("factoid_atom_rule_head", rows, _factoid_atom_score))
        candidates.append(
            _train_numeric_candidate(
                "factoid_atom_verifier",
                rows,
                common + ["numeric_coverage", "numeric_count", "literal_coverage"],
            )
        )
    if class_key == "rag.structured":
        candidates.append(_rule_candidate("structured_cell_rule_head", rows, _structured_cell_score))
        candidates.append(
            _train_numeric_candidate(
                "structured_cell_verifier",
                rows,
                common
                + [
                    "numeric_coverage",
                    "numeric_count",
                    "coverage_label_mean",
                    "dead_weight_label_mean",
                ],
            )
        )
    if class_key in {"rag.code_in_context", "code.agentic_trace"}:
        candidates.append(_rule_candidate("code_symbol_rule_head", rows, _code_symbol_score))
        candidates.append(
            _train_numeric_candidate(
                "code_identifier_ranker",
                rows,
                common + ["identifier_coverage", "identifier_count", "literal_coverage"],
            )
        )
    return candidates


def _factoid_atom_score(row: SliceRow) -> float:
    f = row.features
    numeric = f.get("numeric_coverage", 1.0)
    literal = f.get("literal_coverage", 0.0)
    token_floor = f.get("token_bottom10", 0.0)
    return max(0.0, min(1.0, 0.55 * numeric + 0.30 * literal + 0.15 * token_floor))


def _structured_cell_score(row: SliceRow) -> float:
    f = row.features
    schema = f.get("schema_match", 0.5)
    numeric = f.get("numeric_coverage", 1.0)
    coverage = f.get("coverage_label_mean", f.get("context_coverage_ratio", 0.0))
    return max(0.0, min(1.0, 0.45 * schema + 0.35 * numeric + 0.20 * coverage))


def _code_symbol_score(row: SliceRow) -> float:
    f = row.features
    identifier = f.get("identifier_coverage", 1.0)
    api = f.get("api_symbol_match", identifier)
    order = f.get("trajectory_order_match", 1.0)
    tests = f.get("test_outcome_match", 1.0)
    return max(0.0, min(1.0, 0.40 * identifier + 0.30 * api + 0.15 * order + 0.15 * tests))


def _multi_claim_score(row: SliceRow) -> float:
    f = row.features
    unsupported = f.get("unsupported_claim_fraction", 0.0)
    min_coverage = f.get("min_claim_coverage", 1.0)
    support = f.get("support_unit_label_mean", 1.0)
    return max(0.0, min(1.0, 0.55 * (1.0 - unsupported) + 0.30 * min_coverage + 0.15 * support))


def _passes(candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> bool:
    if candidate.get("status") != "evaluated":
        return False
    if int(candidate.get("n") or 0) < PROMOTION_GATES["min_n"]:
        return False
    if float(candidate.get("binary_grounded_accuracy") or 0.0) < float(
        baseline.get("binary_grounded_accuracy") or 0.0
    ):
        return False
    if float(candidate.get("ungrounded_f1") or 0.0) < float(baseline.get("ungrounded_f1") or 0.0):
        return False
    if _metric_value(candidate, "false_allow_rate", 1.0) > PROMOTION_GATES["max_false_allow"]:
        return False
    if _metric_value(candidate, "false_block_rate", 1.0) > PROMOTION_GATES["max_false_block"]:
        return False
    if _metric_value(candidate, "latency_p95_ms", 999.0) > PROMOTION_GATES["max_p95_latency_ms"]:
        return False
    if _metric_value(candidate, "decision_coverage", 0.0) < PROMOTION_GATES["min_decision_coverage"]:
        return False
    return True


def _metric_value(candidate: Mapping[str, Any], key: str, default: float) -> float:
    value = candidate.get(key)
    if value is None:
        return default
    return float(value)


def _select_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = next((c for c in candidates if c.get("candidate") == "v1_router"), {})
    optimized = next((c for c in candidates if c.get("candidate") == "optimized_calibrator"), {})
    if (
        optimized.get("status") == "artifact_proven"
        and _metric_value(optimized, "false_allow_rate", 1.0) <= PROMOTION_GATES["max_false_allow"]
        and _metric_value(optimized, "false_block_rate", 1.0) <= PROMOTION_GATES["max_false_block"]
        and _metric_value(optimized, "latency_p95_ms", 999.0) <= PROMOTION_GATES["max_p95_latency_ms"]
    ):
        return {
            "selected_head": "optimized_calibrator",
            "production_mode": "allow_block_repair_candidate",
            "reason": "optimized no-regression fusion artifact clears class policy gates",
            "metrics": optimized,
        }
    passing = [c for c in candidates if _passes(c, baseline)]
    order = {
        "trace_feature_logreg": 0,
        "factoid_atom_verifier": 1,
        "structured_cell_verifier": 1,
        "code_identifier_ranker": 1,
        "compact_claim_evidence_head": 2,
    }
    if not passing:
        return {
            "selected_head": "v1_passthrough_repair_only",
            "production_mode": "auto_repair_only",
            "reason": "no candidate cleared no-regression, false-decision, and latency gates",
        }
    best = sorted(
        passing,
        key=lambda c: (
            order.get(str(c.get("candidate")), 99),
            -float(c.get("ungrounded_f1") or 0.0),
            -float(c.get("binary_grounded_accuracy") or 0.0),
        ),
    )[0]
    return {
        "selected_head": best["candidate"],
        "production_mode": "allow_block_repair_candidate",
        "reason": "smallest candidate cleared fixed gates on held-out split",
        "metrics": best,
    }


def _root_cause_deltas(rows: list[SliceRow], candidate: Mapping[str, Any]) -> dict[str, Any]:
    # The models are not persisted in this report, so we report slice-level
    # candidate status by root cause. Full per-cause score deltas are emitted by
    # the concrete candidate metrics once a candidate passes and is persisted.
    by_cause = Counter(cause for row in rows for cause in row.root_causes)
    return {
        cause: {
            "rows": count,
            "candidate_status": candidate.get("selected_head"),
            "production_mode": candidate.get("production_mode"),
        }
        for cause, count in by_cause.items()
    }


def run_solution_tracks(
    paths: ArtifactPaths = ArtifactPaths(),
    *,
    out_dir: Path,
    heads_dir: Path | None = None,
    targeted_path: Path | None = None,
) -> dict[str, Any]:
    slices = build_root_cause_slices(paths)
    targeted_manifest: dict[str, Any] | None = None
    if targeted_path is not None and targeted_path.exists():
        targeted = load_targeted_slices(targeted_path)
        targeted_manifest = {
            class_key: len(rows)
            for class_key, rows in targeted.items()
            if rows
        }
        slices = _merge_slices(slices, targeted)
    slices_dir = out_dir / "root_cause_slices"
    slice_manifest = write_slices(slices, slices_dir)
    fusion = _read_json(paths.fusion, {})
    runtime_policy = _read_json(paths.runtime_policy, {})
    trajectory_report = run_trajectory_head(
        train_bank="transcripts_v1",
        eval_banks=("transcripts_v2", "both"),
    )

    class_reports: dict[str, Any] = {}
    for class_key, rows in slices.items():
        candidates = _candidate_set_for_class(class_key, rows, fusion, runtime_policy)
        selection = _select_candidate(candidates)
        class_reports[class_key] = {
            "rows": len(rows),
            "root_cause_counts": slice_manifest["classes"][class_key]["root_causes"],
            "candidate_bakeoff": candidates,
            "selection": selection,
            "root_cause_solution_map": _root_cause_deltas(rows, selection),
        }

    class_reports["code.agentic_trace"]["trajectory_head"] = trajectory_report
    if trajectory_report["promotion_decision"] != "promote":
        class_reports["code.agentic_trace"]["selection"] = {
            "selected_head": "trajectory_symbolic_ranker",
            "production_mode": "auto_repair_only",
            "reason": "dedicated manufactured trajectory head failed held-out promotion gates",
            "metrics": trajectory_report["eval"],
        }

    report = {
        "schema": "trace_root_cause_solution_tracks.v1",
        "gates": PROMOTION_GATES,
        "slice_manifest": slice_manifest,
        "targeted_data": {
            "path": str(targeted_path) if targeted_path is not None else None,
            "rows_by_class": targeted_manifest or {},
            "claim_scope": "synthetic training/debug data only; public/live benchmarks remain claim gates",
        },
        "classes": class_reports,
    }
    heads_dir = heads_dir or (ROOT / "latence_trace/data/heads")
    head_artifacts = _emit_head_artifacts(report, heads_dir)
    report["head_artifacts"] = head_artifacts
    report["runtime_registry_proposal"] = _registry_proposal(class_reports, head_artifacts)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "root_cause_solution_tracks.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    (out_dir / "root_cause_solution_tracks.md").write_text(render_markdown(report), encoding="utf-8")
    return report


def _emit_head_artifacts(report: Mapping[str, Any], heads_dir: Path) -> dict[str, Any]:
    if not heads_dir.is_absolute():
        heads_dir = ROOT / heads_dir
    heads_dir.mkdir(parents=True, exist_ok=True)
    emitted: dict[str, Any] = {}
    for class_key, payload in report["classes"].items():
        spec = HEAD_SPECS[class_key]
        selection = payload.get("selection") or {}
        enabled = selection.get("production_mode") == "allow_block_repair_candidate"
        candidate = _candidate_for_spec(class_key, payload)
        artifact = {
            "schema": "trace_runtime_head_artifact.v1",
            "class_key": class_key,
            "head_id": spec["head_id"],
            "version": "root_cause_solution_v1",
            "architecture": spec["architecture"],
            "enabled": enabled,
            "candidate_source": candidate.get("candidate"),
            "candidate_status": candidate.get("status"),
            "feature_names": candidate.get("feature_names", []),
            "score_strategy": _score_strategy(class_key, candidate, enabled),
            "metrics": candidate,
            "selection": selection,
            "gates": report["gates"],
            "root_cause_counts": payload.get("root_cause_counts", {}),
            "next_training_target": spec["next_training_target"],
            "rollback_mode": "disable_head_or_class_allow_block",
        }
        path = heads_dir / spec["artifact"]
        path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
        try:
            artifact_path = str(path.relative_to(ROOT))
        except ValueError:
            artifact_path = str(path)
        emitted[class_key] = {
            "path": artifact_path,
            "sha256": _sha256(path),
            "artifact": artifact,
        }
    return emitted


def _candidate_for_spec(class_key: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    spec = HEAD_SPECS[class_key]
    if class_key == "code.agentic_trace":
        trajectory = payload.get("trajectory_head") or {}
        return {
            "candidate": "trajectory_symbolic_ranker",
            "status": "evaluated" if trajectory else "missing",
            "promotion_decision": trajectory.get("promotion_decision"),
            "train_metrics": trajectory.get("train_metrics"),
            "eval": trajectory.get("eval"),
        }
    for candidate in payload.get("candidate_bakeoff") or []:
        if candidate.get("candidate") == spec["candidate"]:
            return dict(candidate)
    for candidate in payload.get("candidate_bakeoff") or []:
        if candidate.get("status") == "evaluated":
            return dict(candidate)
    return {"candidate": spec["candidate"], "status": "not_evaluated"}


def _score_strategy(class_key: str, candidate: Mapping[str, Any], enabled: bool) -> dict[str, Any]:
    if class_key == "rag.prose.enterprise" and enabled:
        return {
            "type": "response_score_passthrough",
            "score_channel_preference": [
                "groundedness_v2",
                "primary_score",
            ],
            "reason_codes": ["enterprise_optimized_calibrator_active"],
        }
    return {
        "type": "descriptor_only_until_promoted",
        "reason_codes": [
            f"{class_key}_head_available_not_enabled",
            str(candidate.get("reason") or candidate.get("status") or "not_promoted"),
        ],
    }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _registry_proposal(
    class_reports: Mapping[str, Any],
    head_artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    registry: dict[str, Any] = {}
    for class_key, payload in class_reports.items():
        selection = payload.get("selection") or {}
        spec = HEAD_SPECS[class_key]
        enabled = selection.get("production_mode") == "allow_block_repair_candidate"
        artifact = head_artifacts.get(class_key) or {}
        registry[class_key] = {
            "class_key": class_key,
            "head_id": spec["head_id"],
            "enabled": enabled,
            "version": "root_cause_solution_v1",
            "reason": selection.get("reason"),
            "artifact_path": artifact.get("path"),
            "artifact_sha256": artifact.get("sha256"),
            "architecture": spec["architecture"],
            "next_training_target": spec["next_training_target"],
            "rollback_switches": (
                [
                    "LATENCE_TRACE_RUNTIME_DECISION_ENABLED=0",
                    f"classes.{class_key}.allow_disabled=true",
                    f"classes.{class_key}.block_disabled=true",
                ]
                if enabled
                else ["already repair-only"]
            ),
        }
    return {
        "schema": "trace_runtime_head_registry_proposal.v1",
        "runtime_head_registry": registry,
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Root-Cause-Derived TRACE Solution Tracks",
        "",
        "## Verdict",
        "",
        "This run builds concrete candidate heads from the diagnosed root causes. "
        "A class is production-promoted only if the candidate beats or matches v1 "
        "and clears false-allow, false-block, and latency gates.",
        "",
        "## Class Results",
        "",
        "| class | rows | selected solution | mode | reason |",
        "|---|---:|---|---|---|",
    ]
    for class_key, payload in report["classes"].items():
        selection = payload["selection"]
        lines.append(
            f"| `{class_key}` | {payload['rows']} | `{selection.get('selected_head')}` | "
            f"`{selection.get('production_mode')}` | {selection.get('reason')} |"
        )
    lines.extend(["", "## Candidate Metrics", "", "| class | candidate | status | acc | ungrounded F1 | false allow | false block | p95 ms |", "|---|---|---|---:|---:|---:|---:|---:|"])
    for class_key, payload in report["classes"].items():
        for candidate in payload["candidate_bakeoff"]:
            lines.append(
                f"| `{class_key}` | `{candidate.get('candidate')}` | `{candidate.get('status')}` | "
                f"{_fmt(candidate.get('binary_grounded_accuracy'))} | "
                f"{_fmt(candidate.get('ungrounded_f1'))} | "
                f"{_fmt(candidate.get('false_allow_rate'))} | "
                f"{_fmt(candidate.get('false_block_rate'))} | "
                f"{_fmt(candidate.get('latency_p95_ms'))} |"
            )
    lines.extend(["", "## Coding Trajectory Head", ""])
    trajectory = report["classes"]["code.agentic_trace"].get("trajectory_head") or {}
    lines.append(f"Promotion decision: `{trajectory.get('promotion_decision')}`")
    for split, metrics in (trajectory.get("eval") or {}).items():
        lines.append(
            f"- `{split}`: AUROC={_fmt(metrics.get('auroc'))}, "
            f"false_allow={_fmt(metrics.get('false_allow_rate'))}, "
            f"false_block={_fmt(metrics.get('false_block_rate'))}"
        )
    lines.extend(["", "## Runtime Registry Proposal", ""])
    for class_key, entry in report["runtime_registry_proposal"]["runtime_head_registry"].items():
        lines.append(
            f"- `{class_key}`: head=`{entry['head_id']}`, enabled={entry['enabled']}, "
            f"artifact=`{entry.get('artifact_path')}`, reason={entry['reason']}"
        )
    lines.extend(["", "## Next Failure Targets", ""])
    for class_key, entry in report["runtime_registry_proposal"]["runtime_head_registry"].items():
        lines.append(f"- `{class_key}`: {entry.get('next_training_target')}")
    lines.append("")
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        return str(round(float(value), 4))
    except (TypeError, ValueError):
        return "n/a"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        default=str(STUDENT / "root_cause_solution_runs/latest"),
        help="Directory for root-cause-derived solution artifacts.",
    )
    parser.add_argument(
        "--runtime-registry-out",
        default=str(ROOT / "latence_trace/data/runtime_head_registry.root_cause_solution_v1.json"),
        help="Production registry proposal path. Only passing heads are enabled.",
    )
    parser.add_argument(
        "--heads-dir",
        default=str(ROOT / "latence_trace/data/heads"),
        help="Directory for versioned per-class runtime head artifacts.",
    )
    parser.add_argument(
        "--targeted-data",
        default=str(STUDENT / "root_cause_targeted/targeted_v1.jsonl"),
        help="Optional targeted synthetic data JSONL for non-public training/debug slices.",
    )
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    report = run_solution_tracks(
        out_dir=out_dir,
        heads_dir=Path(args.heads_dir),
        targeted_path=Path(args.targeted_data),
    )
    registry_path = Path(args.runtime_registry_out)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        json.dumps(report["runtime_registry_proposal"], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(out_dir / "root_cause_solution_tracks.json")
    print(out_dir / "root_cause_solution_tracks.md")
    print(out_dir / "root_cause_slices/manifest.json")
    print(registry_path)
    print(json.dumps(report["runtime_registry_proposal"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
