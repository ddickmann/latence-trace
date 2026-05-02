"""Counterfactual survival labels for TRACE Memory."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from latence_trace.memory.models import MemoryPolicy, MemoryState, MemoryUpdateRequest, SpanRecord
from latence_trace.memory.service import update_memory
from latence_trace.memory.trajectory import CanonicalTrajectory

SurvivalLabel = Literal["anchor", "keep", "demote", "remove", "superseded"]


class CounterfactualLabel(BaseModel):
    span_id: str
    label: SurvivalLabel
    internal_action: str
    confidence: float
    quality_delta: float
    exact_critical_loss: float
    reason: str
    audit: dict[str, Any] = Field(default_factory=dict)


def label_trajectory(
    trajectory: CanonicalTrajectory,
    *,
    policy: MemoryPolicy | None = None,
    learned_weights: dict[str, float] | None = None,
) -> tuple[MemoryState, list[CounterfactualLabel], str]:
    policy = policy or MemoryPolicy(hot_token_budget=160, warm_token_budget=800)
    state = None
    hot_context = ""
    for turn in trajectory.turns:
        result = update_memory(
            MemoryUpdateRequest(
                query_text=turn.query_text,
                turn_text=turn.turn_text,
                response_text=turn.response_text,
                raw_context=_raw_context(turn),
                prior_memory_state=state,
                trace_signals=_trace_signals(turn.trace_scores, learned_weights),
                memory_policy=policy,
                memory_domain=trajectory.domain,
            )
        )
        state = result.next_memory_state
        hot_context = result.hot_context
    if state is None:
        raise ValueError("Cannot label trajectory without turns")
    labels = [
        _label_span(span, hot_context=hot_context, trajectory=trajectory)
        for span in state.spans
    ]
    return state, labels, hot_context


def _trace_signals(
    trace_scores: dict[str, Any],
    learned_weights: dict[str, float] | None,
) -> dict[str, Any]:
    signals = dict(trace_scores or {})
    if learned_weights:
        signals["memory_ranker_weights"] = learned_weights
    return signals


def _label_span(
    span: SpanRecord,
    *,
    hot_context: str,
    trajectory: CanonicalTrajectory,
) -> CounterfactualLabel:
    critical_terms = trajectory.critical_terms
    exact_loss = _exact_loss(span, critical_terms, hot_context)
    survival = span.scores.survival_value
    exact = span.scores.exact_critical
    quality_delta = _quality_delta(span, trajectory, exact_loss)
    if span.compression_level == "tombstone":
        label: SurvivalLabel = "superseded"
        internal_action = "superseded"
        reason = "tombstoned_by_supersession"
    elif exact >= 0.72 or exact_loss > 0.0:
        label = "anchor"
        internal_action = "anchor_exact"
        reason = "exact_critical_or_future_anchor"
    elif quality_delta >= 0.18:
        label = "keep"
        internal_action = "keep_hot" if span.layer == "hot" else "keep_warm"
        reason = "counterfactual_quality_drop"
    elif quality_delta >= 0.06 or survival >= 0.3:
        label = "demote"
        internal_action = "demote_fact" if span.scores.exact_critical >= 0.3 else "demote_summary"
        reason = "summary_or_warm_storage_sufficient"
    else:
        label = "remove"
        internal_action = "remove"
        reason = "no_measurable_degradation"
    confidence = min(1.0, 0.55 + abs(quality_delta) + exact_loss + max(0.0, exact - 0.5))
    return CounterfactualLabel(
        span_id=span.id,
        label=label,
        internal_action=internal_action,
        confidence=round(confidence, 4),
        quality_delta=round(quality_delta, 4),
        exact_critical_loss=round(exact_loss, 4),
        reason=reason,
        audit={
            "span_type": span.span_type,
            "layer": span.layer,
            "salience": round(span.scores.salience, 4),
            "relevance": round(span.scores.relevance, 4),
            "attribution": round(span.scores.attribution, 4),
            "dead_weight": round(span.scores.dead_weight, 4),
            "redundancy": round(span.scores.redundancy, 4),
            "survival_value": round(survival, 4),
            "exact_critical": round(exact, 4),
            "survival_horizon_turns": round(span.scores.survival_horizon_turns, 4),
            "domain_decay_pressure": round(span.scores.domain_decay_pressure, 4),
            "internal_action": internal_action,
            "trajectory_domain": trajectory.domain,
            "dataset": trajectory.dataset,
            "case_id": trajectory.case_id,
        },
    )


def _raw_context(turn: Any) -> str | None:
    parts = [turn.raw_context or ""]
    parts.extend(span.text for span in turn.retrieved_context)
    parts.extend(span.text for span in turn.tool_outputs)
    text = "\n".join(part for part in parts if part.strip())
    return text or None


def _exact_loss(span: SpanRecord, critical_terms: list[str], hot_context: str) -> float:
    if not critical_terms:
        return 0.0
    lost = 0
    for term in critical_terms:
        if term in span.text and term in hot_context:
            lost += 1
    return lost / len(critical_terms)


def _quality_delta(span: SpanRecord, trajectory: CanonicalTrajectory, exact_loss: float) -> float:
    base = 0.0
    if trajectory.outcome.status == "success":
        base += 0.04
    if span.layer == "hot":
        base += 0.08
    if span.scores.attribution >= 0.6:
        base += 0.06
    if span.scores.relevance >= 0.6:
        base += 0.05
    if span.span_type in {"constraint", "decision", "error", "code_symbol", "code_fragment"}:
        base += 0.07
    return min(1.0, base + exact_loss)
