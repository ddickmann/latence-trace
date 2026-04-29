"""Unit tests for the teacher-response -> student-label extractor.

Input is the dev_app ``/runsync`` JSON body; we build tiny fixtures
inline to keep the tests fast.
"""

from __future__ import annotations

import pytest

from research.triangular_maxsim.student_v2.teacher_label.extract import (
    extract_labels,
)


def _make_resp(
    *,
    score: float = 0.9,
    band: str | None = None,
    tokens: list[tuple[str, float]] | None = None,
    units: list[dict] | None = None,
    nli_aggregate: float | None = None,
) -> dict:
    response_tokens = [
        {
            "token": t,
            "reverse_context_calibrated": s,
            "heatmap_score": s,
            "nli_score": None,
        }
        for t, s in (tokens or [])
    ]
    support_units = [
        {
            "support_id": u.get("support_id", f"raw-{i}"),
            "text": u.get("text", ""),
            "used": u.get("used", False),
            "usage_state": u.get("usage_state", "unused"),
            "coverage_score": u.get("coverage_score", 0.0),
        }
        for i, u in enumerate(units or [])
    ]
    compact_units = [
        {
            "support_id": u["support_id"],
            "usage_state": u["usage_state"],
            "coverage_score": u["coverage_score"],
            "used": u["used"],
        }
        for u in support_units
    ]
    return {
        "score": score,
        "band": band,
        "nli_aggregate": nli_aggregate,
        "support_units": compact_units,
        "full": {
            "scores": {"groundedness_v2": score, "primary_score": score},
            "response_tokens": response_tokens,
            "support_units": support_units,
        },
    }


def test_turn_score_prefers_top_level_score() -> None:
    resp = _make_resp(score=0.74)
    out = extract_labels(resp)
    assert out["turn_score"] == pytest.approx(0.74)


def test_turn_score_clamped_to_unit_interval() -> None:
    assert extract_labels(_make_resp(score=1.8))["turn_score"] == pytest.approx(1.0)
    assert extract_labels(_make_resp(score=-0.3))["turn_score"] == pytest.approx(0.0)


def test_gold_band_overrides_teacher_band() -> None:
    resp = _make_resp(score=0.95, band="green")
    out = extract_labels(resp, gold_band="red")
    assert out["turn_band"] == 2
    assert out["teacher_band"] == "green"


def test_band_falls_back_to_thresholds_when_missing() -> None:
    assert extract_labels(_make_resp(score=0.85, band=None))["turn_band"] == 0
    assert extract_labels(_make_resp(score=0.1, band=None))["turn_band"] == 2
    assert extract_labels(_make_resp(score=0.5, band=None))["turn_band"] == 1


def test_token_support_labels_use_calibrated_threshold() -> None:
    # Without NLI scores the extractor falls back to calibrated MaxSim
    # with threshold 0.70.
    resp = _make_resp(tokens=[("Hello", 0.9), ("world", 0.1), ("!", 0.55)])
    out = extract_labels(resp)
    assert out["response_tokens_teacher"] == ["Hello", "world", "!"]
    assert out["token_support_labels"] == [1, 0, 0]


def test_token_support_labels_prefer_nli_score() -> None:
    resp = {
        "score": 0.5,
        "support_units": [],
        "full": {
            "response_tokens": [
                {"token": "Paris", "reverse_context_calibrated": 0.95, "nli_score": 0.99},
                {"token": "Berlin", "reverse_context_calibrated": 0.95, "nli_score": -0.99},
                {"token": "[SEP]", "reverse_context_calibrated": 0.10, "nli_score": None},
            ],
            "support_units": [],
        },
    }
    out = extract_labels(resp)
    # Paris: NLI entailed -> 1; Berlin: NLI contradicted -> 0;
    # [SEP]: NLI skipped, calibrated 0.10 < 0.70 -> 0
    assert out["token_support_labels"] == [1, 0, 0]


def test_unit_labels_derive_from_usage_state_and_coverage() -> None:
    resp = _make_resp(
        units=[
            {"text": "unit A", "usage_state": "used", "used": True, "coverage_score": 0.9},
            {"text": "unit B", "usage_state": "unused", "used": False, "coverage_score": 0.05},
            {"text": "unit C", "usage_state": "uncertain", "used": True, "coverage_score": 0.5},
        ]
    )
    out = extract_labels(resp)
    assert out["evidence_units"] == ["unit A", "unit B", "unit C"]
    assert out["dead_weight_unit_labels"] == [0, 1, 0]
    assert out["coverage_unit_labels"] == [1, 0, 1]


def test_empty_response_yields_empty_axes() -> None:
    resp = _make_resp()
    out = extract_labels(resp)
    assert out["token_support_labels"] == []
    assert out["dead_weight_unit_labels"] == []
    assert out["coverage_unit_labels"] == []
    assert out["n_support_units"] == 0
    assert out["n_response_tokens"] == 0


def test_nli_aggregate_passes_through() -> None:
    resp = _make_resp(score=0.0004, nli_aggregate=0.0004)
    out = extract_labels(resp)
    assert out["nli_aggregate"] == pytest.approx(0.0004)
