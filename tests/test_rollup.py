"""Tests for the stateless rollup transform."""

from __future__ import annotations

import pytest

from latence_trace.api.models import (
    CodeLanePerFileUsage,
    FileAttributionDiagnostics,
    RollupRequest,
    RollupTurnInput,
    SessionSignals,
)
from latence_trace.api.rollup import aggregate_turns
from latence_trace.api.service import GroundednessService


def _fa(per_file_specs, histogram=None) -> FileAttributionDiagnostics:
    per_file = [
        CodeLanePerFileUsage(
            path=p,
            n_units=1,
            used=1,
            uncertain=0,
            unused=0,
            coverage=1.0,
            mean_score=1.0,
            max_evidence=1.0,
            owner_tokens=owner,
            owner_share=share,
            query_owner_tokens=0,
            query_owner_share=0.0,
            dead_weight=bool(share < 0.01),
            reason_codes=reasons,
            dominating_peer=None,
        )
        for (p, owner, share, reasons) in per_file_specs
    ]
    return FileAttributionDiagnostics(
        per_file=per_file,
        per_unit=[],
        dead_weight_files=[p for (p, _o, s, _r) in per_file_specs if s < 0.01],
        dead_weight_ratio=(
            sum(1 for (_p, _o, s, _r) in per_file_specs if s < 0.01)
            / max(len(per_file_specs), 1)
        ),
        n_files=len(per_file_specs),
        n_response_tokens=10,
        n_query_tokens=0,
        reason_code_histogram=histogram or {},
    )


def test_rollup_empty_returns_zero_aggregates() -> None:
    response = aggregate_turns([])
    assert response.turns == 0
    assert response.noise_pct == 0.0
    assert response.model_drift_pct == 0.0
    assert response.retrieval_waste_pct == 0.0
    assert response.reason_code_histogram == {}


def test_rollup_noise_pct_is_mean_dead_weight_ratio() -> None:
    turns = [
        RollupTurnInput(scores={"dead_weight_ratio": 0.1, "groundedness_v2": 0.8}),
        RollupTurnInput(scores={"dead_weight_ratio": 0.3, "groundedness_v2": 0.7}),
        RollupTurnInput(scores={"dead_weight_ratio": 0.5, "groundedness_v2": 0.6}),
    ]
    response = aggregate_turns(turns)
    assert response.turns == 3
    assert response.noise_pct == pytest.approx(0.3)


def test_rollup_retrieval_waste_only_counts_grounded_turns() -> None:
    # One grounded-but-noisy turn and one ungrounded turn: waste should
    # reflect only the grounded one (0.6 dead-weight at score 0.8).
    turns = [
        RollupTurnInput(scores={"dead_weight_ratio": 0.6, "groundedness_v2": 0.8}),
        RollupTurnInput(scores={"dead_weight_ratio": 0.2, "groundedness_v2": 0.3}),
    ]
    response = aggregate_turns(turns)
    # Weighted mean with the grounded turn's score 0.8 as weight.
    assert response.retrieval_waste_pct == pytest.approx(0.6)
    # Noise is the plain mean across both turns.
    assert response.noise_pct == pytest.approx(0.4)


def test_rollup_model_drift_combines_zscore_and_ema_gap() -> None:
    turns = [
        RollupTurnInput(
            scores={"groundedness_v2": 0.9},
            session_signals=SessionSignals(
                drift_z_score=0.0, ema_groundedness=0.9, total_turns=1
            ),
        ),
        RollupTurnInput(
            scores={"groundedness_v2": 0.5},
            session_signals=SessionSignals(
                drift_z_score=1.5, ema_groundedness=0.4, total_turns=2
            ),
        ),
    ]
    response = aggregate_turns(turns)
    # Turn 1 contributes max(0/3, 1 - 0.9) = 0.1
    # Turn 2 contributes max(1.5/3, 1 - 0.4) = max(0.5, 0.6) = 0.6
    # Mean = 0.35
    assert response.model_drift_pct == pytest.approx(0.35)


def test_rollup_reason_code_histogram_sums_across_turns() -> None:
    turns = [
        RollupTurnInput(
            file_attribution=_fa(
                [("a.py", 5, 0.5, []), ("b.py", 0, 0.0, ["never_won_argmax"])],
                histogram={"never_won_argmax": 1},
            )
        ),
        RollupTurnInput(
            file_attribution=_fa(
                [
                    ("a.py", 4, 0.4, []),
                    ("b.py", 0, 0.0, ["never_won_argmax", "dominated_by_single_file"]),
                ],
                histogram={
                    "never_won_argmax": 1,
                    "dominated_by_single_file": 1,
                },
            )
        ),
    ]
    response = aggregate_turns(turns)
    assert response.reason_code_histogram == {
        "never_won_argmax": 2,
        "dominated_by_single_file": 1,
    }


def test_rollup_drift_trend_reports_min_max_mean_last() -> None:
    turns = [
        RollupTurnInput(session_signals=SessionSignals(drift_z_score=0.5)),
        RollupTurnInput(session_signals=SessionSignals(drift_z_score=2.0)),
        RollupTurnInput(session_signals=SessionSignals(drift_z_score=1.25)),
    ]
    response = aggregate_turns(turns)
    assert response.drift_trend.min == pytest.approx(0.5)
    assert response.drift_trend.max == pytest.approx(2.0)
    assert response.drift_trend.mean == pytest.approx((0.5 + 2.0 + 1.25) / 3)
    assert response.drift_trend.last == pytest.approx(1.25)


def test_rollup_top_dead_files_uses_latest_session_signals_when_present() -> None:
    turns = [
        RollupTurnInput(
            session_signals=SessionSignals(
                drift_z_score=0.1,
                dead_file_candidates=["old.py"],
                dead_weight_streak=3,
            )
        ),
        RollupTurnInput(
            session_signals=SessionSignals(
                drift_z_score=0.2,
                dead_file_candidates=["current.py"],
                dead_weight_streak=5,
            )
        ),
    ]
    response = aggregate_turns(turns)
    assert [f.path for f in response.top_dead_files] == ["current.py"]
    assert response.top_dead_files[0].dead_turns == 5


def test_rollup_top_dead_files_falls_back_to_file_attribution() -> None:
    turns = [
        RollupTurnInput(
            file_attribution=_fa(
                [
                    ("a.py", 8, 0.8, []),
                    ("b.py", 0, 0.0, ["never_won_argmax"]),
                ]
            ),
        ),
        RollupTurnInput(
            file_attribution=_fa(
                [
                    ("a.py", 7, 0.7, []),
                    ("b.py", 0, 0.0, ["never_won_argmax"]),
                ]
            ),
        ),
    ]
    response = aggregate_turns(turns)
    paths = [f.path for f in response.top_dead_files]
    assert "b.py" in paths
    # The most dead-file appearances should rank first.
    assert paths[0] == "b.py"


def test_rollup_all_percentages_clipped_to_unit_interval() -> None:
    turns = [
        RollupTurnInput(
            scores={"dead_weight_ratio": 2.0, "groundedness_v2": 0.9},
            session_signals=SessionSignals(
                drift_z_score=10.0, ema_groundedness=-1.0, total_turns=1
            ),
        ),
    ]
    response = aggregate_turns(turns)
    assert 0.0 <= response.noise_pct <= 1.0
    assert 0.0 <= response.model_drift_pct <= 1.0
    assert 0.0 <= response.retrieval_waste_pct <= 1.0


def test_service_rollup_emits_heatmap_when_requested() -> None:
    service = GroundednessService(encoder_factory=lambda _model_name=None: None)
    response = service.rollup(
        RollupRequest(
            turns=[
                RollupTurnInput(
                    scores={"dead_weight_ratio": 0.1, "groundedness_v2": 0.8}
                ),
                RollupTurnInput(
                    scores={"dead_weight_ratio": 0.4, "groundedness_v2": 0.5}
                ),
            ],
            heatmap_format="html",
        )
    )
    assert response.heatmap is not None
    assert response.heatmap_html is not None
    assert "lt-heatmap" in response.heatmap_html
    assert response.turns == 2
