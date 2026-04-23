"""Caller-portable session state for code-lane temporal signals.

The HTTP API is **stateless by design** — every ``/groundedness``
request is scored in isolation, no conversation storage on the server.
Temporal judgments (drift, EMA groundedness, dead-weight streaks,
per-session anomalies) are *session-level* questions though, and they
need memory *somewhere*. This module makes that memory a first-class,
**caller-carried** concern.

The contract
------------
1. The client hits turn 1 **without** a ``session_state`` field.
2. The server scores the turn and returns a fresh
   ``next_session_state`` (+ ``session_signals``) alongside the usual
   per-turn diagnostics.
3. The client includes ``session_state=<that blob>`` on turn 2.
4. The server runs :func:`update_session_state` — a pure deterministic
   transform — and emits an updated blob.
5. Rinse, repeat.

The server never persists the blob. It is an opaque, bounded payload
the caller owns end-to-end. This keeps the privacy posture simple
(no server-side conversation storage), aligns with MCP / plugin
runtimes that already carry session memory, and matches the
sensor-layer / control-layer split that enterprise buyers expect.

Everything in here is pure Python with **O(bounded_window)** space
complexity per call — safe to run on the critical path.

See ``docs/session_semantics.md`` for the wire protocol, reset
events, and recommendation policy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "DEFAULT_EMA_HALF_LIFE_TURNS",
    "DEFAULT_VERDICT_WINDOW",
    "DEFAULT_RISK_BAND_WINDOW",
    "DEFAULT_GROUNDEDNESS_EMA_HALF_LIFE",
    "MAX_TRACKED_FILES",
    "SESSION_STATE_SCHEMA_VERSION",
    "Recommendation",
    "RollingStats",
    "FileSessionStats",
    "TurnMetrics",
    "SessionState",
    "SessionSignals",
    "update_session_state",
    "file_attribution_to_turn_inputs",
]

#: One release bump per breaking change to the state shape. Callers can
#: ignore blobs with a newer version than they know about.
SESSION_STATE_SCHEMA_VERSION: int = 1

#: Roughly: after this many turns the EMA decays by 1/e.
DEFAULT_EMA_HALF_LIFE_TURNS: int = 5

#: Half-life used for the groundedness-score EMA. A shorter window than
#: the per-file EMA because we want faster re-anchor recommendations.
DEFAULT_GROUNDEDNESS_EMA_HALF_LIFE: int = 3

#: Retention window for the phantom verdict trail and risk-band trail.
DEFAULT_VERDICT_WINDOW: int = 20

#: Retention window for the risk-band trail; kept separate so callers
#: can dial it up without blowing up the verdict window.
DEFAULT_RISK_BAND_WINDOW: int = 20

#: Upper bound on tracked files. Beyond this we evict the lowest-EMA
#: file, which is the right thing to do for dead-weight detection: the
#: blob stays O(1) in serialised size even on 10k-turn agentic runs.
MAX_TRACKED_FILES: int = 64

#: Threshold below which a file is considered "effectively dead" for
#: the eviction recommendation. Matches the value used in the TS
#: client helper so the two are interchangeable.
DEFAULT_DEAD_FILE_EMA_THRESHOLD: float = 0.02

#: Minimum number of consecutive dead turns before a file is surfaced
#: in ``dead_file_candidates``.
DEFAULT_DEAD_FILE_MIN_TURNS: int = 5


# ---------------------------------------------------------------------------
# Enums and compact typed records
# ---------------------------------------------------------------------------


class Recommendation:
    """Session-level recommendation (plain strings for wire compatibility)."""

    CONTINUE = "continue"
    RE_ANCHOR = "re-anchor"
    FRESH_CHAT = "fresh-chat"


@dataclass
class RollingStats:
    """Welford-style running mean + variance accumulator.

    Fields are plain ``float`` / ``int`` so the struct is trivially
    JSON-serialisable and bit-for-bit reproducible.
    """

    n: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, value: float) -> "RollingStats":
        n = self.n + 1
        delta = value - self.mean
        mean = self.mean + delta / n
        m2 = self.m2 + delta * (value - mean)
        return RollingStats(n=n, mean=mean, m2=m2)

    @property
    def std(self) -> float:
        return math.sqrt(self.m2 / (self.n - 1)) if self.n > 1 else 0.0


@dataclass
class FileSessionStats:
    """Per-file rolling stats inside :class:`SessionState`.

    ``dead_turns`` resets to 0 whenever the file is not flagged dead.
    """

    ema_owner_share: float = 0.0
    ema_query_owner_share: float = 0.0
    dead_turns: int = 0
    total_turns: int = 0


@dataclass
class TurnMetrics:
    """Flat per-turn summary the session transform consumes.

    This is a deliberate **projection** of the full scoring response —
    only the scalars the session layer needs. Keeping this tight means
    new code-lane diagnostics never force a session schema bump.
    """

    groundedness: Optional[float] = None
    per_token_p10: Optional[float] = None
    composite: Optional[float] = None
    dead_weight_ratio: Optional[float] = None
    cascade_fired: bool = False
    phantom_verdict: Optional[bool] = None
    phantom_probability: Optional[float] = None
    risk_band: Optional[str] = None
    # ``file_updates[file_id] = (owner_share, query_owner_share,
    # dead_weight_bool)`` for the files attributed this turn.
    file_updates: Dict[str, Tuple[float, float, bool]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# The state itself
# ---------------------------------------------------------------------------


@dataclass
class SessionState:
    """Caller-portable session memory.

    The state is intentionally **additive** and bounded in size:

    - Rolling Welford stats for ``per_token_p10``, ``composite``, and
      ``groundedness`` (3 × 3 floats each).
    - ``ema_groundedness`` (single float).
    - ``groundedness_baseline`` — Welford stats over the first N
      "stable" turns, used as the anchor for ``groundedness_drift``.
    - ``file_stats`` capped at :data:`MAX_TRACKED_FILES`.
    - ``phantom_trail`` / ``risk_band_trail`` capped by their window.

    The blob round-trips verbatim; :func:`update_session_state` is the
    only code path that writes to it.
    """

    schema_version: int = SESSION_STATE_SCHEMA_VERSION
    session_id: Optional[str] = None
    total_turns: int = 0
    cascade_fires: int = 0

    per_token_rolling: RollingStats = field(default_factory=RollingStats)
    composite_rolling: RollingStats = field(default_factory=RollingStats)
    groundedness_rolling: RollingStats = field(default_factory=RollingStats)
    groundedness_baseline: RollingStats = field(default_factory=RollingStats)

    ema_groundedness: Optional[float] = None

    file_stats: Dict[str, FileSessionStats] = field(default_factory=dict)
    phantom_trail: List[Optional[bool]] = field(default_factory=list)
    risk_band_trail: List[str] = field(default_factory=list)

    # Tunables (kept inside the blob so both sides of the wire agree).
    ema_half_life_turns: int = DEFAULT_EMA_HALF_LIFE_TURNS
    groundedness_ema_half_life: int = DEFAULT_GROUNDEDNESS_EMA_HALF_LIFE
    verdict_window: int = DEFAULT_VERDICT_WINDOW
    risk_band_window: int = DEFAULT_RISK_BAND_WINDOW


@dataclass
class SessionSignals:
    """Derived session-level signals emitted next to ``next_session_state``.

    These are cheap to compute and callers can inspect them without
    ever cracking the state blob open.
    """

    total_turns: int = 0
    drift_z_score: float = 0.0
    ema_groundedness: Optional[float] = None
    groundedness_drift: float = 0.0
    dead_file_candidates: List[str] = field(default_factory=list)
    dead_weight_streak: int = 0
    cascade_density: float = 0.0
    phantom_rate: float = 0.0
    red_streak: int = 0
    recommendation: str = Recommendation.CONTINUE


# ---------------------------------------------------------------------------
# Pure transform
# ---------------------------------------------------------------------------


def update_session_state(
    prior: Optional[SessionState],
    turn: TurnMetrics,
    *,
    session_id: Optional[str] = None,
    dead_file_ema_threshold: float = DEFAULT_DEAD_FILE_EMA_THRESHOLD,
    dead_file_min_turns: int = DEFAULT_DEAD_FILE_MIN_TURNS,
) -> Tuple[SessionState, SessionSignals]:
    """Update ``prior`` with ``turn`` and emit derived session signals.

    The function is a **pure deterministic transform**:

    - Same inputs → bit-identical outputs.
    - No hidden state, no I/O, no global reads.
    - O(len(turn.file_updates) + constant-bounded trail lengths).

    Callers that want to start a new session pass ``prior=None``.
    """
    state = _clone_or_default(prior, session_id=session_id)
    state.total_turns += 1

    if turn.cascade_fired:
        state.cascade_fires += 1

    if turn.per_token_p10 is not None:
        state.per_token_rolling = state.per_token_rolling.update(
            float(turn.per_token_p10)
        )

    if turn.composite is not None:
        state.composite_rolling = state.composite_rolling.update(float(turn.composite))

    if turn.groundedness is not None:
        state.groundedness_rolling = state.groundedness_rolling.update(
            float(turn.groundedness)
        )
        # EMA, half-life expressed in turns.
        alpha = _half_life_to_alpha(state.groundedness_ema_half_life)
        if state.ema_groundedness is None:
            state.ema_groundedness = float(turn.groundedness)
        else:
            state.ema_groundedness = (
                alpha * float(turn.groundedness) + (1.0 - alpha) * state.ema_groundedness
            )
        # Baseline captures the first N turns, giving ``groundedness_drift``
        # something stable to anchor against. Ten turns is enough to get a
        # reasonable mean+std without dragging drift detection on forever.
        if state.groundedness_baseline.n < 10:
            state.groundedness_baseline = state.groundedness_baseline.update(
                float(turn.groundedness)
            )

    _trim_and_append(state.phantom_trail, turn.phantom_verdict, state.verdict_window)
    if turn.risk_band is not None:
        _trim_and_append(state.risk_band_trail, turn.risk_band, state.risk_band_window)

    if turn.file_updates:
        _update_file_stats(state, turn)

    signals = _derive_signals(
        state=state,
        turn=turn,
        dead_file_ema_threshold=dead_file_ema_threshold,
        dead_file_min_turns=dead_file_min_turns,
    )
    return state, signals


def file_attribution_to_turn_inputs(
    per_file: Sequence[Mapping[str, object]],
) -> Dict[str, Tuple[float, float, bool]]:
    """Build ``TurnMetrics.file_updates`` from a ``PerFileUsage`` dict seq.

    This is a convenience for the service layer — it lets the wiring in
    ``api/service.py`` stay one line long.
    """
    updates: Dict[str, Tuple[float, float, bool]] = {}
    for entry in per_file:
        path = str(entry.get("path") or "")
        if not path:
            continue
        owner_share = float(entry.get("owner_share") or 0.0)
        query_owner_share = float(entry.get("query_owner_share") or 0.0)
        dead_weight = bool(entry.get("dead_weight") or False)
        updates[path] = (owner_share, query_owner_share, dead_weight)
    return updates


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _clone_or_default(
    prior: Optional[SessionState],
    *,
    session_id: Optional[str],
) -> SessionState:
    """Deep-copy the dataclasses we mutate; leave scalars shared.

    The transform is meant to be functional at the caller boundary
    even though we rebuild mutable containers here.
    """
    if prior is None:
        return SessionState(session_id=session_id)
    return SessionState(
        schema_version=prior.schema_version,
        session_id=session_id or prior.session_id,
        total_turns=prior.total_turns,
        cascade_fires=prior.cascade_fires,
        per_token_rolling=RollingStats(
            n=prior.per_token_rolling.n,
            mean=prior.per_token_rolling.mean,
            m2=prior.per_token_rolling.m2,
        ),
        composite_rolling=RollingStats(
            n=prior.composite_rolling.n,
            mean=prior.composite_rolling.mean,
            m2=prior.composite_rolling.m2,
        ),
        groundedness_rolling=RollingStats(
            n=prior.groundedness_rolling.n,
            mean=prior.groundedness_rolling.mean,
            m2=prior.groundedness_rolling.m2,
        ),
        groundedness_baseline=RollingStats(
            n=prior.groundedness_baseline.n,
            mean=prior.groundedness_baseline.mean,
            m2=prior.groundedness_baseline.m2,
        ),
        ema_groundedness=prior.ema_groundedness,
        file_stats={
            path: FileSessionStats(
                ema_owner_share=stats.ema_owner_share,
                ema_query_owner_share=stats.ema_query_owner_share,
                dead_turns=stats.dead_turns,
                total_turns=stats.total_turns,
            )
            for path, stats in prior.file_stats.items()
        },
        phantom_trail=list(prior.phantom_trail),
        risk_band_trail=list(prior.risk_band_trail),
        ema_half_life_turns=prior.ema_half_life_turns,
        groundedness_ema_half_life=prior.groundedness_ema_half_life,
        verdict_window=prior.verdict_window,
        risk_band_window=prior.risk_band_window,
    )


def _update_file_stats(state: SessionState, turn: TurnMetrics) -> None:
    alpha = _half_life_to_alpha(state.ema_half_life_turns)
    for path, (owner, query_owner, dead) in turn.file_updates.items():
        stats = state.file_stats.get(path)
        if stats is None:
            stats = FileSessionStats(
                ema_owner_share=owner,
                ema_query_owner_share=query_owner,
            )
        else:
            stats = FileSessionStats(
                ema_owner_share=alpha * owner + (1.0 - alpha) * stats.ema_owner_share,
                ema_query_owner_share=(
                    alpha * query_owner + (1.0 - alpha) * stats.ema_query_owner_share
                ),
                dead_turns=stats.dead_turns,
                total_turns=stats.total_turns,
            )
        stats.total_turns += 1
        stats.dead_turns = stats.dead_turns + 1 if dead else 0
        state.file_stats[path] = stats

    if len(state.file_stats) > MAX_TRACKED_FILES:
        # Evict the lowest-EMA files first; those are the ones we care
        # about least for both drift and live ownership.
        ranked = sorted(
            state.file_stats.items(),
            key=lambda kv: (kv[1].ema_owner_share, -kv[1].dead_turns),
        )
        for path, _ in ranked[: len(ranked) - MAX_TRACKED_FILES]:
            state.file_stats.pop(path, None)


def _derive_signals(
    *,
    state: SessionState,
    turn: TurnMetrics,
    dead_file_ema_threshold: float,
    dead_file_min_turns: int,
) -> SessionSignals:
    # Drift: absolute z-score of the current ``per_token_p10`` against
    # the session's rolling mean+std. Drift is signed in the literature
    # but we return the absolute value because a one-sided "response is
    # unusually ill-supported" is what operators care about; the sign
    # is recoverable from the last event.
    drift_z = 0.0
    if turn.per_token_p10 is not None and state.per_token_rolling.n > 2:
        std = state.per_token_rolling.std
        if std > 1e-6:
            drift_z = abs(
                (float(turn.per_token_p10) - state.per_token_rolling.mean) / std
            )

    baseline = state.groundedness_baseline
    baseline_mean = baseline.mean if baseline.n > 0 else None
    groundedness_drift = 0.0
    if baseline_mean is not None and state.ema_groundedness is not None:
        groundedness_drift = state.ema_groundedness - baseline_mean

    cascade_density = (
        state.cascade_fires / state.total_turns if state.total_turns else 0.0
    )
    phantom_rate = (
        sum(1 for v in state.phantom_trail if v) / len(state.phantom_trail)
        if state.phantom_trail
        else 0.0
    )

    dead_candidates = [
        path
        for path, stats in state.file_stats.items()
        if stats.dead_turns >= dead_file_min_turns
        and stats.ema_owner_share < dead_file_ema_threshold
    ]
    dead_weight_streak = max(
        (stats.dead_turns for stats in state.file_stats.values()),
        default=0,
    )
    red_streak = _trailing_run(state.risk_band_trail, "red")

    recommendation = _recommend(
        drift_z=drift_z,
        groundedness_drift=groundedness_drift,
        red_streak=red_streak,
        phantom_rate=phantom_rate,
        total_turns=state.total_turns,
    )

    return SessionSignals(
        total_turns=state.total_turns,
        drift_z_score=drift_z,
        ema_groundedness=state.ema_groundedness,
        groundedness_drift=groundedness_drift,
        dead_file_candidates=dead_candidates,
        dead_weight_streak=dead_weight_streak,
        cascade_density=cascade_density,
        phantom_rate=phantom_rate,
        red_streak=red_streak,
        recommendation=recommendation,
    )


def _half_life_to_alpha(half_life_turns: int) -> float:
    # ``0.5 ** (1 / H)`` is the per-step decay factor whose half-life
    # is ``H`` turns; the EMA weight on the new sample is 1 − that.
    h = max(1, int(half_life_turns))
    return 1.0 - math.pow(0.5, 1.0 / h)


def _trim_and_append(trail: List, value, window: int) -> None:  # type: ignore[type-arg]
    trail.append(value)
    while len(trail) > max(1, window):
        trail.pop(0)


def _trailing_run(trail: Sequence[str], target: str) -> int:
    count = 0
    for value in reversed(trail):
        if value == target:
            count += 1
        else:
            break
    return count


def _recommend(
    *,
    drift_z: float,
    groundedness_drift: float,
    red_streak: int,
    phantom_rate: float,
    total_turns: int,
) -> str:
    # Policy encoded in one place so the docs and the transform stay in
    # sync. Thresholds are conservative by design — we would rather
    # "continue" a genuine edge case than nudge users to reset sessions.
    if total_turns < 3:
        return Recommendation.CONTINUE
    if red_streak >= 3 or phantom_rate >= 0.3:
        return Recommendation.FRESH_CHAT
    if drift_z >= 2.0 or groundedness_drift <= -0.15 or red_streak >= 2:
        return Recommendation.RE_ANCHOR
    return Recommendation.CONTINUE
