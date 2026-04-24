"""Stateless session-level rollup.

The rollup endpoint is a pure deterministic transform:

- input: a list of per-turn records the caller already has
- output: conversation-level aggregates (noise, drift, waste,
  reason-code histogram, top dead files, risk-band trail)

Nothing is persisted. Nothing is re-scored. ``O(len(turns))`` in time
and memory, no GPU, no ML calls.

Formula reference
-----------------

``noise_pct``
    Mean ``dead_weight_ratio`` across turns. A high value means a lot
    of retrieved context is unused.

``model_drift_pct``
    Per turn we take::

        max(clip(drift_z_score / 3.0, 0, 1),
            1.0 - (ema_groundedness or 0.0))

    and average across turns. This captures two complementary
    failure modes — the model's ``per_token_p10`` drifting from its
    session baseline (drift z-score) and the rolling EMA of the
    groundedness score dropping below the initial plateau.

``retrieval_waste_pct``
    Weighted mean of ``dead_weight_ratio`` restricted to turns where
    the headline groundedness is >= 0.6 — i.e. the answer looks
    correct but a lot of chunks were dead weight. This is the
    contradictory signal that motivates the retrieval-quality upsell.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from latence_trace.api.models import (
    DriftTrend,
    RollupResponse,
    RollupTopDeadFile,
    RollupTurnInput,
)


def _clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def _score_lookup(scores: Optional[Mapping[str, Any]], *keys: str) -> Optional[float]:
    if not scores:
        return None
    for key in keys:
        value = scores.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _dead_weight_for_turn(turn: RollupTurnInput) -> float:
    value = _score_lookup(turn.scores, "dead_weight_ratio")
    if value is not None:
        return _clip(value)
    if turn.file_attribution is not None:
        try:
            return _clip(float(turn.file_attribution.dead_weight_ratio))
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def _groundedness_for_turn(turn: RollupTurnInput) -> Optional[float]:
    return _score_lookup(
        turn.scores,
        "composite_phantom_score",
        "groundedness_v2",
        "primary_score",
    )


def aggregate_turns(
    turns: List[RollupTurnInput],
    *,
    session_id: Optional[str] = None,
) -> RollupResponse:
    """Pure aggregation — see the module docstring for the formulas."""
    n = len(turns)
    if n == 0:
        return RollupResponse(session_id=session_id)

    dead_weights: List[float] = []
    drift_scores: List[float] = []
    drift_zs: List[float] = []
    risk_trail: List[str] = []
    recommendations: List[str] = []
    reason_counter: Counter[str] = Counter()

    retrieval_waste_weighted_sum = 0.0
    retrieval_waste_weight = 0.0

    for turn in turns:
        dw = _dead_weight_for_turn(turn)
        dead_weights.append(dw)

        groundedness = _groundedness_for_turn(turn)

        signals = turn.session_signals
        drift_z = 0.0
        ema_g: Optional[float] = None
        if signals is not None:
            try:
                drift_z = float(getattr(signals, "drift_z_score", 0.0) or 0.0)
            except (TypeError, ValueError):
                drift_z = 0.0
            ema_candidate = getattr(signals, "ema_groundedness", None)
            if ema_candidate is not None:
                try:
                    ema_g = float(ema_candidate)
                except (TypeError, ValueError):
                    ema_g = None
            rec = getattr(signals, "recommendation", None)
            if rec:
                recommendations.append(str(rec))
        drift_zs.append(drift_z)

        drift_component = max(
            _clip(drift_z / 3.0),
            _clip(1.0 - (ema_g if ema_g is not None else 0.0)),
        )
        drift_scores.append(drift_component)

        if groundedness is not None and groundedness >= 0.6:
            retrieval_waste_weighted_sum += dw * float(groundedness)
            retrieval_waste_weight += float(groundedness)

        if turn.risk_band:
            risk_trail.append(str(turn.risk_band))

        if turn.recommendation and turn.recommendation not in recommendations:
            recommendations.append(str(turn.recommendation))

        if turn.file_attribution is not None:
            for key, val in (turn.file_attribution.reason_code_histogram or {}).items():
                try:
                    reason_counter[str(key)] += int(val)
                except (TypeError, ValueError):
                    continue

    noise_pct = sum(dead_weights) / n
    model_drift_pct = sum(drift_scores) / n
    retrieval_waste_pct = (
        retrieval_waste_weighted_sum / retrieval_waste_weight
        if retrieval_waste_weight > 0
        else 0.0
    )

    trend = DriftTrend(
        min=min(drift_zs) if drift_zs else 0.0,
        max=max(drift_zs) if drift_zs else 0.0,
        mean=(sum(drift_zs) / n) if drift_zs else 0.0,
        last=drift_zs[-1] if drift_zs else 0.0,
    )

    top_dead = _collect_top_dead_files(turns)

    return RollupResponse(
        turns=n,
        noise_pct=_clip(noise_pct),
        model_drift_pct=_clip(model_drift_pct),
        retrieval_waste_pct=_clip(retrieval_waste_pct),
        reason_code_histogram=dict(reason_counter),
        recommendations=_dedupe(recommendations),
        risk_band_trail=risk_trail,
        drift_trend=trend,
        top_dead_files=top_dead,
        session_id=session_id,
    )


def _dedupe(values: Iterable[str]) -> List[str]:
    return list(dict.fromkeys(values))


def _collect_top_dead_files(
    turns: List[RollupTurnInput], *, limit: int = 5
) -> List[RollupTopDeadFile]:
    """Rank dead-file candidates using the most recent session signals.

    Falls back to scanning ``file_attribution.per_file`` over the whole
    window when the caller didn't supply session signals (common for
    pure RAG rollups that don't use the temporal layer).
    """
    # Prefer the latest session signals payload if present — it already
    # carries the dead-file roster (as a list of paths). We cross-
    # reference with the latest file_attribution payload (when
    # available) to pull per-file owner_share so the output is
    # actionable, not just a list of paths.
    latest_attribution: Optional[Mapping[str, Any]] = None
    latest_attribution_paths: Dict[str, Tuple[int, float]] = {}
    for turn in reversed(turns):
        if turn.file_attribution is not None:
            latest_attribution = {
                rec.path: (int(rec.owner_tokens or 0), float(rec.owner_share or 0.0))
                for rec in (turn.file_attribution.per_file or [])
            }
            latest_attribution_paths = latest_attribution
            break

    for turn in reversed(turns):
        signals = turn.session_signals
        if signals is None:
            continue
        dead_candidates = getattr(signals, "dead_file_candidates", None) or []
        if dead_candidates:
            out: List[RollupTopDeadFile] = []
            for path in dead_candidates[:limit]:
                _owner_tokens, ema_owner = latest_attribution_paths.get(
                    path, (0, 0.0)
                )
                out.append(
                    RollupTopDeadFile(
                        path=str(path),
                        dead_turns=int(
                            getattr(signals, "dead_weight_streak", 0) or 0
                        ),
                        ema_owner_share=float(ema_owner or 0.0),
                    )
                )
            return out

    # Fallback: aggregate dead-weight occurrences across turns from
    # file_attribution payloads.
    dead_counts: Counter[str] = Counter()
    owner_share_sum: Dict[str, float] = {}
    owner_share_n: Dict[str, int] = {}
    for turn in turns:
        fa = turn.file_attribution
        if fa is None:
            continue
        for per_file in fa.per_file or []:
            path = str(per_file.path)
            if bool(per_file.dead_weight):
                dead_counts[path] += 1
            owner_share_sum[path] = owner_share_sum.get(path, 0.0) + float(
                per_file.owner_share or 0.0
            )
            owner_share_n[path] = owner_share_n.get(path, 0) + 1

    ranked = sorted(
        dead_counts.items(),
        key=lambda kv: (-kv[1], owner_share_sum.get(kv[0], 0.0) / max(owner_share_n.get(kv[0], 1), 1)),
    )
    out: List[RollupTopDeadFile] = []
    for path, dead_turns in ranked[:limit]:
        denom = max(owner_share_n.get(path, 1), 1)
        out.append(
            RollupTopDeadFile(
                path=path,
                dead_turns=int(dead_turns),
                ema_owner_share=float(owner_share_sum.get(path, 0.0) / denom),
            )
        )
    return out


__all__ = ["aggregate_turns"]
