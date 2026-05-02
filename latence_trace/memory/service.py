"""High-level TRACE Memory update operation."""

from __future__ import annotations

from typing import Any

from latence_trace.memory.dedup import merge_spans
from latence_trace.memory.extract import extract_spans
from latence_trace.memory.models import (
    MemoryDiagnostics,
    MemoryState,
    MemoryUpdateRequest,
    MemoryUpdateResponse,
)
from latence_trace.memory.select import assign_layers, hot_context
from latence_trace.memory.survival import update_survival


def update_memory(request: MemoryUpdateRequest) -> MemoryUpdateResponse:
    prior = request.prior_memory_state or MemoryState()
    turn_index = prior.turn_index + 1
    signals = _coerce_signals(request)
    incoming = extract_spans(
        turn_text=request.turn_text,
        response_text=request.response_text,
        raw_context=request.raw_context,
        query_text=request.query_text,
        turn_index=turn_index,
        memory_domain=request.memory_domain,
    )
    merged, dedup_actions = merge_spans(list(prior.spans), incoming)
    scored, survival_actions = update_survival(
        merged,
        turn_index=turn_index,
        policy=request.memory_policy,
        trace_signals=signals,
    )
    selected, selection_diag = assign_layers(scored, policy=request.memory_policy)
    diagnostics = MemoryDiagnostics(
        actions=[*dedup_actions, *survival_actions, *selection_diag.actions],
        hot_tokens=selection_diag.hot_tokens,
        warm_tokens=selection_diag.warm_tokens,
        cold_tokens=selection_diag.cold_tokens,
        removed_tokens=selection_diag.removed_tokens,
        exact_critical_spans=selection_diag.exact_critical_spans,
        top_survival_causes=selection_diag.top_survival_causes,
    )
    state = MemoryState(
        turn_index=turn_index,
        spans=selected,
        cold_provenance=list(prior.cold_provenance),
        metadata=dict(prior.metadata),
    )
    return MemoryUpdateResponse(
        next_memory_state=state,
        hot_context=hot_context(selected),
        actions=diagnostics.actions,
        diagnostics=diagnostics,
    )


def update_memory_from_payload(payload: dict[str, Any]) -> MemoryUpdateResponse:
    return update_memory(MemoryUpdateRequest.model_validate(payload))


def _coerce_signals(request: MemoryUpdateRequest) -> dict[str, Any]:
    signals: dict[str, Any] = {}
    if request.trace_signals:
        signals.update(request.trace_signals)
    if request.memory_ranker_weights:
        signals["memory_ranker_weights"] = request.memory_ranker_weights
    if request.query_text:
        signals["query_text"] = request.query_text
    if request.memory_domain:
        signals["trajectory_domain"] = request.memory_domain
    trace = request.trace_response or {}
    if isinstance(trace, dict):
        for key in (
            "scores",
            "runtime_head_features",
            "trajectory_features",
            "session_signals",
            "runtime_decision",
        ):
            if key in trace and key not in signals:
                signals[key] = trace[key]
    return signals
