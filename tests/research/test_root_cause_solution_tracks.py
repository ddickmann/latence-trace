from __future__ import annotations

import json
from pathlib import Path

from research.triangular_maxsim.student_v2.root_cause_model_selection import ArtifactPaths
from research.triangular_maxsim.student_v2.root_cause_solution_tracks import (
    build_root_cause_slices,
    run_solution_tracks,
)


def _json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


def _row(row_id: str, class_key: str, gold: str, pred: str, score: float) -> dict:
    return {
        "row_id": row_id,
        "class_key": class_key,
        "gold_band": gold,
        "v1": {"band": pred, "score": score, "nli_aggregate": score - 0.5},
        "labels": {
            "response_tokens_teacher": ["The", "value", "is", "42"],
            "token_support_labels": [1, 1, 0, 0],
            "evidence_units": [{"text": "The value is 41.", "char_start": 0, "char_end": 16}],
            "dead_weight_unit_labels": [0],
            "coverage_unit_labels": [1],
            "n_support_units": 1,
        },
        "token_diagnostics": [
            {"token": "The", "heatmap_score": score, "reverse_context_calibrated": score},
            {"token": "42", "heatmap_score": score, "reverse_context_calibrated": score},
        ],
        "raw_response": {
            "full": {
                "scores": {
                    "primary_score": score,
                    "reverse_context": score,
                    "groundedness_v2": score,
                    "literal_guarded": score,
                    "literal_total_count": 4,
                    "literal_match_count": 3,
                    "literal_mismatch_count": 1,
                },
                "response_tokens": [{"token": "The"}, {"token": "value"}, {"token": "42"}],
                "support_units": [{"text": "The value is 41."}],
            }
        },
    }


def _trajectory(path: Path) -> Path:
    rows = []
    for i in range(4):
        grounded = i < 2
        rows.append(
            {
                "id": f"t{i}",
                "chunker": "sentence_packed",
                "scorer_config": "gte_only",
                "label": "grounded" if grounded else "ungrounded",
                "tier": "correct" if grounded else "wrong",
                "context_token_count": 100,
                "max_top_evidence_score": 0.9 if grounded else 0.1,
                "phantom_flagged": not grounded,
                "scores": {
                    "reverse_context": 0.9 if grounded else 0.1,
                    "consensus_hardened": 0.9 if grounded else 0.1,
                    "groundedness_v2": 0.9 if grounded else 0.1,
                    "triangular": 0.9 if grounded else 0.1,
                    "literal_total_count": 4,
                    "literal_match_count": 4 if grounded else 1,
                    "literal_mismatch_count": 0 if grounded else 3,
                    "context_attribution_ratio": 0.9 if grounded else 0.1,
                    "context_unused_ratio": 0.1 if grounded else 0.9,
                    "context_uncertain_ratio": 0.1 if grounded else 0.9,
                    "dead_weight_ratio": 0.1 if grounded else 0.9,
                    "support_units_total": 4,
                    "support_units_usage_used": 4 if grounded else 1,
                },
            }
        )
    return _json(path, {"rows": rows})


def test_solution_tracks_build_slices_and_registry(tmp_path: Path) -> None:
    rows = []
    for i in range(10):
        rows.append(_row(f"e-green-{i}", "rag.prose.enterprise", "green", "green", 0.95))
        rows.append(_row(f"e-red-{i}", "rag.prose.enterprise", "red", "red", 0.20))
    v1_cache = _jsonl(tmp_path / "v1.jsonl", rows)
    fusion = _json(
        tmp_path / "fusion.json",
        {
            "classes": {
                "rag.prose.enterprise": {
                    "mode": "weighted_fusion",
                    "selected": {
                        "n": 20,
                        "binary_grounded_accuracy": 1.0,
                        "ungrounded_precision": 1.0,
                        "ungrounded_recall": 1.0,
                        "ungrounded_f1": 1.0,
                    },
                }
            }
        },
    )
    policy = _json(
        tmp_path / "policy.json",
        {
            "classes": {
                "rag.prose.enterprise": {
                    "allow_disabled": False,
                    "block_disabled": False,
                    "false_allow_rate": 0.0,
                    "false_block_rate": 0.0,
                    "allow_threshold": 0.8,
                    "block_threshold": 0.3,
                }
            }
        },
    )
    trajectory = _trajectory(tmp_path / "trajectory.json")
    paths = ArtifactPaths(
        v1_cache=v1_cache,
        fusion=fusion,
        eval_v1_plus=_json(tmp_path / "eval.json", {}),
        train_report=_json(tmp_path / "train.json", {}),
        deadweight=_json(tmp_path / "deadweight.json", {}),
        eval_per_source=_json(tmp_path / "source.json", {}),
        ragtruth_reset=_json(tmp_path / "ragtruth.json", {}),
        halueval_reset=_json(tmp_path / "halueval.json", {}),
        trajectory_v1=trajectory,
        trajectory_v2=trajectory,
        trajectory_both=trajectory,
        trajectory_head=_json(tmp_path / "trajectory_head.json", {}),
        stabilization_readme=tmp_path / "missing.md",
        runtime_policy=policy,
    )

    slices = build_root_cause_slices(paths)
    report = run_solution_tracks(paths, out_dir=tmp_path / "out", heads_dir=tmp_path / "heads")

    assert len(slices["rag.prose.enterprise"]) == 20
    assert report["runtime_registry_proposal"]["runtime_head_registry"]["rag.prose.enterprise"][
        "enabled"
    ]
