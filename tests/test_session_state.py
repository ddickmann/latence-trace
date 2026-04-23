"""Unit tests for the caller-portable session-state transform.

The transform is pure, so the tests focus on:

- Determinism and round-trip safety of the opaque blob.
- Correctness of the derived signals (drift z-score, EMA, dead-file
  eviction, recommendation policy).
- Backwards-compatibility: missing / schema-mismatched blobs should
  never crash and should be treated as a fresh session.
"""

from __future__ import annotations

from latence_trace.api.models import (
    FileSessionStatsPayload,
    RollingStatsPayload,
    SessionStatePayload,
)
from latence_trace.api.service import (
    _session_signals_to_payload,
    _session_state_from_payload,
    _session_state_to_payload,
)
from latence_trace.core.code_lane.session import (
    Recommendation,
    SESSION_STATE_SCHEMA_VERSION,
    SessionState,
    TurnMetrics,
    update_session_state,
)


def _green_turn(files: dict[str, tuple[float, float, bool]] | None = None) -> TurnMetrics:
    return TurnMetrics(
        groundedness=0.85,
        per_token_p10=0.55,
        composite=0.80,
        dead_weight_ratio=0.2,
        cascade_fired=False,
        phantom_verdict=False,
        risk_band="green",
        file_updates=files or {},
    )


def _red_turn(files: dict[str, tuple[float, float, bool]] | None = None) -> TurnMetrics:
    return TurnMetrics(
        groundedness=0.20,
        per_token_p10=0.05,
        composite=0.15,
        dead_weight_ratio=0.9,
        cascade_fired=True,
        phantom_verdict=True,
        risk_band="red",
        file_updates=files or {},
    )


def test_fresh_session_initialises_cleanly():
    state, signals = update_session_state(None, _green_turn(), session_id="sess-1")
    assert state.total_turns == 1
    assert state.session_id == "sess-1"
    assert signals.recommendation == Recommendation.CONTINUE
    assert signals.drift_z_score == 0.0
    assert signals.ema_groundedness == 0.85


def test_transform_is_pure_and_deterministic():
    turn = _green_turn({"src/a.py": (0.6, 0.5, False)})
    s1, sig1 = update_session_state(None, turn, session_id="sess")
    s2, sig2 = update_session_state(None, turn, session_id="sess")
    # Same inputs → bit-identical outputs on each side of the pipeline.
    assert s1 == s2
    assert sig1 == sig2


def test_dead_file_eviction_kicks_in_after_streak():
    state = None
    # Five dead turns for src/b.py, one live turn for src/a.py.
    for _ in range(5):
        state, signals = update_session_state(
            state,
            _green_turn({"src/a.py": (0.5, 0.4, False), "src/b.py": (0.0, 0.0, True)}),
            session_id="sess",
        )
    assert state is not None
    assert state.file_stats["src/b.py"].dead_turns == 5
    assert "src/b.py" in signals.dead_file_candidates
    assert "src/a.py" not in signals.dead_file_candidates


def test_recommendation_escalates_on_sustained_red_turns():
    state = None
    for _ in range(3):
        state, signals = update_session_state(state, _red_turn(), session_id="sess")
    assert signals.red_streak == 3
    assert signals.recommendation == Recommendation.FRESH_CHAT


def test_recommendation_suggests_reanchor_on_single_drift_spike():
    state = None
    # Seed three stable turns so drift std is non-zero.
    for _ in range(3):
        state, _ = update_session_state(state, _green_turn(), session_id="sess")
    # Drift spike: per_token_p10 collapses.
    spike = TurnMetrics(
        groundedness=0.60,
        per_token_p10=0.05,
        composite=0.40,
        dead_weight_ratio=0.3,
        cascade_fired=False,
        phantom_verdict=False,
        risk_band="amber",
    )
    state, signals = update_session_state(state, spike)
    assert signals.drift_z_score >= 1.5  # >= ~2 sigma from baseline
    assert signals.recommendation in {Recommendation.RE_ANCHOR, Recommendation.CONTINUE}


def test_payload_round_trip_preserves_state():
    state = None
    for _ in range(3):
        state, _ = update_session_state(
            state,
            _green_turn({"src/a.py": (0.6, 0.5, False)}),
            session_id="sess",
        )
    assert state is not None
    payload = _session_state_to_payload(state)
    restored = _session_state_from_payload(payload)
    assert restored is not None
    assert restored.total_turns == state.total_turns
    assert restored.per_token_rolling.mean == state.per_token_rolling.mean
    assert restored.file_stats["src/a.py"].ema_owner_share == (
        state.file_stats["src/a.py"].ema_owner_share
    )


def test_schema_mismatch_is_treated_as_fresh_session():
    payload = SessionStatePayload(
        schema_version=SESSION_STATE_SCHEMA_VERSION + 99,  # unknown future version
        total_turns=42,
    )
    restored = _session_state_from_payload(payload)
    # Unknown schema → None → next turn initialises a fresh session.
    assert restored is None


def test_missing_payload_is_treated_as_fresh_session():
    assert _session_state_from_payload(None) is None


def test_signals_payload_mirrors_dataclass_fields():
    state, signals = update_session_state(None, _green_turn(), session_id="sess")
    wire = _session_signals_to_payload(signals)
    dumped = wire.model_dump()
    assert dumped["total_turns"] == 1
    assert dumped["recommendation"] == signals.recommendation


def test_file_stats_cap_bounds_blob_size():
    state: SessionState | None = None
    # Push 200 distinct files; the cap should trim to MAX_TRACKED_FILES.
    files = {f"file-{i}.py": (0.1 + i * 0.001, 0.0, False) for i in range(200)}
    state, _ = update_session_state(state, _green_turn(files=files), session_id="sess")
    assert state is not None
    from latence_trace.core.code_lane.session import MAX_TRACKED_FILES

    assert len(state.file_stats) <= MAX_TRACKED_FILES


def test_empty_payload_matches_defaults():
    # An empty payload must be a valid blob the service can round-trip.
    payload = SessionStatePayload()
    assert payload.per_token_rolling == RollingStatsPayload()
    assert payload.file_stats == {}
    # FileSessionStatsPayload defaults should all be zero.
    fs = FileSessionStatsPayload()
    assert fs.ema_owner_share == 0.0
    assert fs.dead_turns == 0
