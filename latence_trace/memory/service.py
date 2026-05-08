"""High-level TRACE Memory update operation."""

from __future__ import annotations

from time import perf_counter
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
    start_time = perf_counter()
    prior = request.prior_memory_state or MemoryState()
    turn_index = prior.turn_index + 1
    signals = _coerce_signals(request)
    extract_start = perf_counter()
    incoming = extract_spans(
        turn_text=request.turn_text,
        response_text=request.response_text,
        raw_context=request.raw_context,
        query_text=request.query_text,
        turn_index=turn_index,
        memory_domain=request.memory_domain,
        source_pointer=request.source_pointer,
    )
    dedup_start = perf_counter()
    merged, dedup_actions = merge_spans(list(prior.spans), incoming)
    survival_start = perf_counter()
    scored, survival_actions = update_survival(
        merged,
        turn_index=turn_index,
        policy=request.memory_policy,
        trace_signals=signals,
    )
    selection_start = perf_counter()
    selected, selection_diag = assign_layers(scored, policy=request.memory_policy)
    end_time = perf_counter()
    diagnostics = MemoryDiagnostics(
        actions=[*dedup_actions, *survival_actions, *selection_diag.actions],
        hot_tokens=selection_diag.hot_tokens,
        warm_tokens=selection_diag.warm_tokens,
        effective_hot_token_budget=selection_diag.effective_hot_token_budget,
        effective_warm_token_budget=selection_diag.effective_warm_token_budget,
        effective_max_spans=selection_diag.effective_max_spans,
        budget_mode_used=selection_diag.budget_mode_used,
        target_token_reduction=selection_diag.target_token_reduction,
        actual_token_reduction=selection_diag.actual_token_reduction,
        estimated_exact_critical_recall=selection_diag.estimated_exact_critical_recall,
        survival_mass_retained=selection_diag.survival_mass_retained,
        memory_underbudgeted=selection_diag.memory_underbudgeted,
        recommended_hot_token_budget=selection_diag.recommended_hot_token_budget,
        recent_tail_required=selection_diag.recent_tail_required,
        genesis_anchor_spans=selection_diag.genesis_anchor_spans,
        genesis_anchor_recall=selection_diag.genesis_anchor_recall,
        timings_ms={
            "extract": round((dedup_start - extract_start) * 1000.0, 3),
            "dedup": round((survival_start - dedup_start) * 1000.0, 3),
            "survival": round((selection_start - survival_start) * 1000.0, 3),
            "selection": round((end_time - selection_start) * 1000.0, 3),
            "total": round((end_time - start_time) * 1000.0, 3),
        },
        cold_tokens=selection_diag.cold_tokens,
        removed_tokens=selection_diag.removed_tokens,
        exact_critical_spans=selection_diag.exact_critical_spans,
        top_survival_causes=selection_diag.top_survival_causes,
    )
    cold_provenance = list(prior.cold_provenance)
    span_index = {span.id: span for span in selected}
    all_actions = [*dedup_actions, *survival_actions, *selection_diag.actions]
    for action in all_actions:
        if action.action != "superseded":
            continue
        tombstoned = span_index.get(action.span_id)
        if tombstoned is None:
            continue
        cold_provenance.append({
            "span_id": tombstoned.id,
            "span_type": tombstoned.span_type,
            "typed_key": tombstoned.signature.typed_key,
            "token_count": tombstoned.token_count,
            "created_turn": tombstoned.created_turn,
            "tombstoned_turn": turn_index,
            "reason": action.reason,
        })
    state = MemoryState(
        turn_index=turn_index,
        spans=selected,
        cold_provenance=cold_provenance,
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
