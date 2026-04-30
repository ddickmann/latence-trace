from __future__ import annotations

import json
from pathlib import Path

from research.triangular_maxsim.coding.coding_trajectory_head import run


def _row(row_id: str, label: str, reverse_context: float) -> dict:
    grounded = label == "grounded"
    return {
        "id": row_id,
        "chunker": "sentence_packed",
        "scorer_config": "gte_only",
        "label": label,
        "tier": "correct" if grounded else "wrong",
        "base_scenario_id": row_id,
        "context_token_count": 1000,
        "max_top_evidence_score": reverse_context,
        "phantom_flagged": not grounded,
        "scores": {
            "reverse_context": reverse_context,
            "consensus_hardened": reverse_context,
            "groundedness_v2": reverse_context,
            "triangular": reverse_context,
            "literal_total_count": 10,
            "literal_match_count": 9 if grounded else 1,
            "literal_mismatch_count": 1 if grounded else 9,
            "context_attribution_ratio": reverse_context,
            "context_unused_ratio": 0.0 if grounded else 1.0,
            "context_uncertain_ratio": 0.1 if grounded else 0.9,
            "dead_weight_ratio": 0.1 if grounded else 0.9,
            "support_units_total": 10,
            "support_units_usage_used": 8 if grounded else 1,
        },
    }


def _bank(path: Path, rows: list[dict]) -> Path:
    path.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    return path


def test_coding_trajectory_head_promotes_on_clean_synthetic_bank(tmp_path: Path) -> None:
    banks = {
        "train": _bank(
            tmp_path / "train.json",
            [
                _row("g1", "grounded", 0.95),
                _row("g2", "grounded", 0.90),
                _row("u1", "ungrounded", 0.10),
                _row("u2", "ungrounded", 0.20),
            ],
        ),
        "eval": _bank(
            tmp_path / "eval.json",
            [
                _row("g3", "grounded", 0.92),
                _row("u3", "ungrounded", 0.12),
            ],
        ),
    }

    report = run(train_bank="train", eval_banks=["eval"], banks=banks)

    assert report["eval"]["eval"]["auroc"] == 1.0
    assert report["promotion_decision"] == "promote"
