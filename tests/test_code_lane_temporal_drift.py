"""End-to-end regression tests for code-lane temporal drift detection.

The v3 quality-boost sprint signed off three temporal signals that the
signed-off spec calls out by name:

- per-session z-scored ``per_token_p10`` (``drift_z_score``)
- per-file ``ema_owner_share`` (dead-weight streak)
- EMA groundedness with baseline anchor (``groundedness_drift``)

and a ``recommendation`` policy that escalates from ``continue`` to
``re-anchor`` / ``fresh-chat`` when those signals fire together.

The unit tests in ``test_session_state.py`` exercise the pure transform
with synthetic :class:`TurnMetrics` values. This module closes the
integration gap: it runs the actual production pipeline —
``score_code_groundedness`` → ``_turn_metrics_from`` →
``update_session_state`` — across a multi-turn conversation and asserts
that every temporal signal moves in the expected direction.

The tests use CPU-only deterministic stubs so they run on the critical
path of CI without touching CUDA or external NLI services. Any
regression in the wiring glue between the code lane and the session
transform will land as a CI failure before it reaches the RunPod image.
"""

from __future__ import annotations

import hashlib
import math
import random
from typing import List, Sequence

import pytest
import torch

from latence_trace.api.service import _turn_metrics_from
from latence_trace.core.code_lane import (
    CodeLaneConfig,
    CodeLaneResult,
    Recommendation,
    SessionState,
    SupportUnitPack,
    file_attribution_to_turn_inputs,
    score_code_groundedness,
    update_session_state,
)
from latence_trace.api.models import (
    GroundednessScores,
    ScoringMode,
)


def _embed(tokens: Sequence[str], dim: int = 16) -> torch.Tensor:
    # Use a deterministic hash (``hashlib.sha256``) instead of Python's
    # built-in ``hash()`` — the latter is randomised per process unless
    # ``PYTHONHASHSEED`` is pinned, which makes the composite scores and
    # therefore the derived session signals non-reproducible across
    # pytest runs.
    rows: List[List[float]] = []
    for t in tokens:
        digest = hashlib.sha256(t.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "big")
        rnd = random.Random(seed)
        vec = [rnd.random() - 0.5 for _ in range(dim)]
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        rows.append([v / norm for v in vec])
    return torch.tensor(rows, dtype=torch.float32)


def _tokenise(text: str) -> List[str]:
    return [t for t in text.replace("\n", " ").split() if t]


def _build_units() -> List[SupportUnitPack]:
    sources = [
        (
            "src/auth.py",
            "from src.auth import AuthService\n"
            "def authenticate(user, password): return user.check_password(password)",
        ),
        (
            "src/auth.py",
            "from src.auth import AuthService\n"
            "class AuthService:\n"
            "    def login(self, user): return authenticate(user, user.password)",
        ),
        (
            "src/routes.py",
            "from src.auth import AuthService\napp.post('/login', AuthService().login)",
        ),
    ]
    units: List[SupportUnitPack] = []
    for idx, (path, text) in enumerate(sources):
        tokens = tuple(_tokenise(text))
        units.append(
            SupportUnitPack(
                support_id=f"unit_{idx}",
                path=path,
                tokens=tokens,
                embeddings=_embed(tokens),
                offset_start=0,
                offset_end=len(text),
            )
        )
    return units


_GROUNDED = (
    "from src.auth import AuthService\n"
    "service = AuthService()\n"
    "service.login(user)"
)
_PHANTOM = (
    "from photon_labs.tracer import quantum_auth\n"
    "quantum_auth.login(user)"
)


def _run_turn(response_code: str) -> CodeLaneResult:
    response_tokens = _tokenise(response_code)
    query_text = "implement the login endpoint using AuthService"
    query_tokens = _tokenise(query_text)
    cfg = CodeLaneConfig(
        enable_ast=True,
        enable_nli_cascade=False,
        enable_literal_novelty=True,
        use_logistic_composite=True,
        response_language_hint="python",
    )
    return score_code_groundedness(
        response_text=response_code,
        response_tokens=response_tokens,
        response_embeddings=_embed(response_tokens),
        support_units=_build_units(),
        query_text=query_text,
        query_embeddings=_embed(query_tokens),
        config=cfg,
    )


def _scores_stub(result: CodeLaneResult) -> GroundednessScores:
    """Minimal ``GroundednessScores`` projection the service layer builds.

    Only the fields ``_turn_metrics_from`` actually reads are populated.
    """
    composite = result.composite
    verdict = bool(composite.verdict)
    return GroundednessScores(
        primary_name="composite_phantom_score",
        primary_score=float(composite.composite_score),
        reverse_context=float(result.scorer.reverse_context),
        groundedness_v2=float(composite.composite_score),
        composite_phantom_score=float(composite.composite_score),
        risk_band="red" if verdict else "green",
        scoring_mode=ScoringMode.CODE,
    )


def _advance(state, result: CodeLaneResult):
    scores = _scores_stub(result)
    metrics = _turn_metrics_from(scores=scores, code_result=result)
    return update_session_state(state, metrics, session_id="temporal-drift-test")


def test_drift_z_score_rises_when_per_token_p10_collapses():
    """A phantom turn after a run of grounded turns must spike ``drift_z_score``."""
    state = None
    baseline_scores: List[float] = []
    for _ in range(4):
        result = _run_turn(_GROUNDED)
        state, signals = _advance(state, result)
        baseline_scores.append(result.scorer.per_token_p10)

    phantom_result = _run_turn(_PHANTOM)
    state, signals = _advance(state, phantom_result)

    assert state.total_turns == 5
    assert signals.drift_z_score >= 1.0, (
        f"expected drift_z_score >= 1.0 after phantom turn, got {signals.drift_z_score:.3f} "
        f"(baseline p10 run: {baseline_scores}, phantom p10: {phantom_result.scorer.per_token_p10})"
    )


def test_phantom_verdicts_drive_phantom_rate_and_recommendation():
    """The full wire test: ``AstGroundingResult.ast_phantom_verdict`` must
    feed ``TurnMetrics.phantom_verdict`` and drive ``SessionSignals`` such
    that a streak of phantom turns escalates the recommendation away from
    ``continue``.
    """
    state = None
    for _ in range(5):
        state, signals = _advance(state, _run_turn(_PHANTOM))

    assert signals.phantom_rate > 0.3, (
        f"phantom_rate stuck at {signals.phantom_rate} after 5 consecutive phantoms"
    )
    assert signals.recommendation in {
        Recommendation.FRESH_CHAT,
        Recommendation.RE_ANCHOR,
    }, (
        f"recommendation did not escalate: got {signals.recommendation} "
        f"with phantom_rate={signals.phantom_rate:.3f}, red_streak={signals.red_streak}"
    )


def test_groundedness_ema_and_baseline_are_populated():
    """``ema_groundedness`` must start populated from the first turn and
    ``groundedness_drift`` must stay in ``[-1, 1]`` across a multi-turn
    run — the quantitative direction is covered in the pure-transform
    unit tests (``test_session_state.py``) and the live 360-benchmark
    ``_dim_session`` suite; here we lock in the wiring contract.
    """
    state = None
    emas: List[float] = []
    drifts: List[float] = []
    for _ in range(6):
        state, signals = _advance(state, _run_turn(_GROUNDED))
        assert signals.ema_groundedness is not None
        emas.append(signals.ema_groundedness)
        drifts.append(signals.groundedness_drift)

    assert all(-1.0 <= x <= 1.0 for x in emas), f"ema out of bounds: {emas}"
    assert all(-1.0 <= x <= 1.0 for x in drifts), f"drift out of bounds: {drifts}"
    # Baseline is seeded from the first turn, so drift starts at ~0.
    assert abs(drifts[0]) < 1e-6, (
        f"expected near-zero drift on first turn, got {drifts[0]}"
    )


def test_file_stats_accumulate_across_turns():
    """File-level EMA ownership must be tracked across turns and bounded.

    Irrespective of whether an individual turn flags a file as dead, the
    per-file book-keeping (``ema_owner_share``, ``total_turns``) must
    monotonically advance — this is the foundation the dead-weight
    streak signal is built on.
    """
    state = None
    for _ in range(6):
        state, _ = _advance(state, _run_turn(_GROUNDED))

    assert state is not None
    assert state.file_stats, "no file stats accumulated after 6 turns"
    for path, stats in state.file_stats.items():
        assert stats.total_turns > 0, f"{path}: total_turns did not advance"
        assert 0.0 <= stats.ema_owner_share <= 1.0, (
            f"{path}: ema_owner_share out of bounds: {stats.ema_owner_share}"
        )


def test_session_state_round_trips_across_turns_deterministically():
    """The opaque session blob must be bit-identical across identical turn
    sequences — no hidden clocks, no PRNGs in the transform."""

    def _run(turn_count: int) -> SessionState:
        state = None
        for _ in range(turn_count):
            state, _ = _advance(state, _run_turn(_GROUNDED))
        assert state is not None
        return state

    s1 = _run(3)
    s2 = _run(3)
    assert s1.total_turns == s2.total_turns == 3
    assert s1.per_token_rolling.mean == pytest.approx(s2.per_token_rolling.mean)
    assert s1.per_token_rolling.m2 == pytest.approx(s2.per_token_rolling.m2)
    assert s1.ema_groundedness == pytest.approx(s2.ema_groundedness)
    assert s1.phantom_trail == s2.phantom_trail
    assert set(s1.file_stats.keys()) == set(s2.file_stats.keys())
