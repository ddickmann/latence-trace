from __future__ import annotations

import json
from pathlib import Path

from research.triangular_maxsim.student_v2.root_cause_model_selection import (
    ArtifactPaths,
    build_report,
)


def _write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def test_root_cause_report_promotes_only_enterprise_head(tmp_path: Path) -> None:
    v1_cache = _write_jsonl(
        tmp_path / "v1_cache.jsonl",
        [
            {
                "row_id": "enterprise-red-accepted",
                "class_key": "rag.prose.enterprise",
                "gold_band": "red",
                "v1": {"band": "green", "score": 0.97, "nli_aggregate": 0.2},
                "labels": {"token_support_labels": [1, 0, 1, 0]},
            },
            {
                "row_id": "enterprise-green-blocked",
                "class_key": "rag.prose.enterprise",
                "gold_band": "green",
                "v1": {"band": "red", "score": 0.45, "nli_aggregate": 0.9},
                "labels": {"token_support_labels": [1, 1, 1]},
            },
        ],
    )
    classes = {
        "rag.prose.enterprise": {
            "mode": "weighted_fusion",
            "baseline_v1": {"binary_grounded_accuracy": 0.8, "ungrounded_f1": 0.8},
            "selected": {"binary_grounded_accuracy": 0.95, "ungrounded_f1": 0.96},
        },
        "rag.prose.multi_claim": {
            "mode": "v1_passthrough",
            "baseline_v1": {"binary_grounded_accuracy": 0.4, "ungrounded_f1": 0.5},
            "selected": {"binary_grounded_accuracy": 0.4, "ungrounded_f1": 0.5},
        },
        "rag.prose.short_factoid": {
            "mode": "v1_passthrough",
            "baseline_v1": {"binary_grounded_accuracy": 0.6, "ungrounded_f1": 0.6},
            "selected": {"binary_grounded_accuracy": 0.6, "ungrounded_f1": 0.6},
        },
        "rag.structured": {
            "mode": "v1_passthrough",
            "baseline_v1": {"binary_grounded_accuracy": 0.6, "ungrounded_f1": 0.7},
            "selected": {"binary_grounded_accuracy": 0.6, "ungrounded_f1": 0.7},
        },
        "rag.code_in_context": {
            "mode": "v1_passthrough",
            "baseline_v1": {"binary_grounded_accuracy": 1.0, "ungrounded_f1": 0.0},
            "selected": {"binary_grounded_accuracy": 1.0, "ungrounded_f1": 0.0},
        },
        "code.agentic_trace": {
            "mode": "v1_passthrough",
            "baseline_v1": {"binary_grounded_accuracy": 1.0, "ungrounded_f1": 1.0},
            "selected": {"binary_grounded_accuracy": 1.0, "ungrounded_f1": 1.0},
        },
    }
    fusion = _write_json(
        tmp_path / "fusion.json",
        {"overall_no_regression_pass": True, "classes": classes},
    )
    train_report = _write_json(
        tmp_path / "train_report.json",
        {"tiny_mlp_train_accuracy": 0.49, "turn_metrics": {"f1": 0.5}},
    )
    policy = _write_json(
        tmp_path / "runtime_policy.json",
        {
            "classes": {
                "rag.prose.enterprise": {
                    "allow_threshold": 0.83,
                    "block_threshold": 0.5,
                    "allow_disabled": False,
                    "block_disabled": False,
                }
            }
        },
    )
    trajectory = {
        "run_info": {"case_count": 2},
        "summary": {
            "by_cell": {
                "sentence_packed|gte_only": {"metrics": {"reverse_context_auroc": 0.79}},
                "colgrep|fuse_mean": {"metrics": {"reverse_context_auroc": 0.65}},
            }
        },
    }
    paths = ArtifactPaths(
        v1_cache=v1_cache,
        fusion=fusion,
        eval_v1_plus=_write_json(tmp_path / "eval_v1_plus.json", {}),
        train_report=train_report,
        deadweight=_write_json(tmp_path / "deadweight.json", {"buckets": {}}),
        eval_per_source=_write_json(tmp_path / "eval_per_source.json", {}),
        ragtruth_reset=_write_json(tmp_path / "ragtruth.json", {}),
        halueval_reset=_write_json(tmp_path / "halueval.json", {}),
        trajectory_v1=_write_json(tmp_path / "trajectory_v1.json", trajectory),
        trajectory_v2=_write_json(tmp_path / "trajectory_v2.json", trajectory),
        trajectory_both=_write_json(tmp_path / "trajectory_both.json", trajectory),
        trajectory_head=_write_json(
            tmp_path / "trajectory_head.json",
            {
                "promotion_decision": "do_not_promote_keep_code_agentic_trace_repair_only",
                "eval": {"transcripts_v2": {"auroc": 0.69}},
            },
        ),
        stabilization_readme=tmp_path / "missing.md",
        runtime_policy=policy,
    )

    report = build_report(paths)

    assert report["classes"]["rag.prose.enterprise"]["selection"]["selected_head"] == (
        "optimized_feature_calibrator"
    )
    assert "rag.prose.enterprise" in report["production_gating"]["enabled_allow_block_classes"]
    assert report["production_gating"]["runtime_head_registry"]["rag.prose.enterprise"][
        "thresholds"
    ]["allow_threshold"] == 0.83
    assert report["classes"]["code.agentic_trace"]["selection"]["production_mode"] == (
        "auto_repair_only"
    )
