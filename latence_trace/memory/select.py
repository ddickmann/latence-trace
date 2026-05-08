"""Hot/warm/cold memory selection."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass

from latence_trace.memory.models import (
    MemoryAction,
    MemoryBudgetMode,
    MemoryDiagnostics,
    MemoryPolicy,
    SpanRecord,
)


@dataclass(frozen=True)
class _BudgetDecision:
    hot_budget: int
    warm_budget: int
    max_spans: int
    mode_used: MemoryBudgetMode
    target_reduction: float | None
    actual_reduction: float
    exact_recall: float
    survival_mass: float
    underbudgeted: bool
    recommended_hot_budget: int
    recent_tail_required: bool
    genesis_anchor_count: int
    genesis_anchor_recall: float


@dataclass(frozen=True)
class _QualityIndex:
    cumulative_tokens: list[int]
    cumulative_exact: list[int]
    cumulative_survival: list[float]
    cumulative_gate: list[int]
    cumulative_genesis: list[int]
    total_exact: int
    total_survival: float
    total_genesis: int


def assign_layers(
    spans: list[SpanRecord],
    *,
    policy: MemoryPolicy,
) -> tuple[list[SpanRecord], MemoryDiagnostics]:
    diagnostics = MemoryDiagnostics()
    candidates = [span for span in spans if span.compression_level != "tombstone"]
    ranked = sorted(
        candidates,
        key=lambda span: _selection_score(span, policy),
        reverse=True,
    )
    budget = _resolve_budget(ranked, policy)
    hot_tokens = 0
    warm_tokens = 0
    kept: list[SpanRecord] = []
    for span in ranked[: budget.max_spans]:
        previous = span.layer
        if hot_tokens + span.token_count <= budget.hot_budget:
            span.layer = "hot"
            hot_tokens += span.token_count
        elif warm_tokens + span.token_count <= budget.warm_budget:
            span.layer = "warm"
            warm_tokens += span.token_count
        else:
            span.layer = "cold"
            diagnostics.cold_tokens += span.token_count
        kept.append(span)
        if previous != span.layer:
            diagnostics.actions.append(
                MemoryAction(
                    action="demoted" if span.layer in {"warm", "cold"} else "kept",
                    span_id=span.id,
                    reason="budgeted_layer_assignment",
                    from_layer=previous,
                    to_layer=span.layer,
                )
            )
    kept_ids = {span.id for span in kept}
    for span in candidates:
        if span.id not in kept_ids:
            span.layer = "cold"
            diagnostics.cold_tokens += span.token_count
            diagnostics.actions.append(
                MemoryAction(
                    action="demoted",
                    span_id=span.id,
                    reason="max_span_budget",
                    to_layer="cold",
                )
            )
            kept.append(span)
    tombstones = [span for span in spans if span.compression_level == "tombstone"]
    diagnostics.hot_tokens = hot_tokens
    diagnostics.warm_tokens = warm_tokens
    diagnostics.effective_hot_token_budget = budget.hot_budget
    diagnostics.effective_warm_token_budget = budget.warm_budget
    diagnostics.effective_max_spans = budget.max_spans
    diagnostics.budget_mode_used = budget.mode_used
    diagnostics.target_token_reduction = budget.target_reduction
    diagnostics.actual_token_reduction = round(
        1.0 - hot_tokens / max(1, sum(max(0, span.token_count) for span in candidates)),
        4,
    )
    diagnostics.estimated_exact_critical_recall = budget.exact_recall
    diagnostics.survival_mass_retained = budget.survival_mass
    diagnostics.memory_underbudgeted = budget.underbudgeted
    diagnostics.recommended_hot_token_budget = budget.recommended_hot_budget
    diagnostics.recent_tail_required = budget.recent_tail_required
    diagnostics.genesis_anchor_spans = budget.genesis_anchor_count
    diagnostics.genesis_anchor_recall = budget.genesis_anchor_recall
    diagnostics.exact_critical_spans = sum(
        1 for span in kept if span.scores.exact_critical >= policy.exact_critical_floor
    )
    diagnostics.top_survival_causes = [
        {
            "span_id": span.id,
            "span_type": span.span_type,
            "layer": span.layer,
            "survival_value": round(span.scores.survival_value, 4),
            "learned_survival": round(span.scores.learned_survival, 4),
            "exact_critical": round(span.scores.exact_critical, 4),
            "survival_horizon_turns": round(span.scores.survival_horizon_turns, 4),
            "domain_decay_pressure": round(span.scores.domain_decay_pressure, 4),
            "reason": _top_reason(span),
        }
        for span in sorted(kept, key=lambda item: item.scores.survival_value, reverse=True)[:8]
    ]
    return kept + tombstones, diagnostics


def _resolve_budget(ranked: list[SpanRecord], policy: MemoryPolicy) -> _BudgetDecision:
    total_tokens = sum(max(0, span.token_count) for span in ranked)
    if not ranked or total_tokens <= 0:
        return _BudgetDecision(
            hot_budget=policy.hot_token_budget,
            warm_budget=policy.warm_token_budget,
            max_spans=policy.max_spans,
            mode_used=policy.memory_budget_mode,
            target_reduction=policy.target_token_reduction,
            actual_reduction=0.0,
            exact_recall=1.0,
            survival_mass=1.0,
            underbudgeted=False,
            recommended_hot_budget=policy.hot_token_budget,
            recent_tail_required=False,
            genesis_anchor_count=0,
            genesis_anchor_recall=1.0,
        )

    min_budget = policy.hot_token_budget
    max_allowed = _max_allowed_hot_budget(total_tokens, policy)
    max_spans = policy.max_spans if max_allowed <= policy.hot_token_budget else max(policy.max_spans, len(ranked))
    quality_index = _build_quality_index(ranked, max_spans, policy)
    mode = policy.memory_budget_mode
    if mode == "fixed":
        hot_budget = min(min_budget, max_allowed)
        exact_recall, survival_mass, underbudgeted, genesis_count, genesis_recall = _budget_quality(
            quality_index,
            hot_budget,
            policy,
        )
        return _decision(
            hot_budget,
            policy,
            max_spans,
            mode,
            exact_recall,
            survival_mass,
            underbudgeted,
            total_tokens,
            recommended_hot_budget=hot_budget,
            genesis_anchor_count=genesis_count,
            genesis_anchor_recall=genesis_recall,
        )
    if mode == "ratio":
        hot_budget = max(min_budget, _ratio_hot_budget(policy))
        hot_budget = min(hot_budget, max_allowed)
        exact_recall, survival_mass, underbudgeted, genesis_count, genesis_recall = _budget_quality(
            quality_index,
            hot_budget,
            policy,
        )
        return _decision(
            hot_budget,
            policy,
            max_spans,
            mode,
            exact_recall,
            survival_mass,
            underbudgeted,
            total_tokens,
            recommended_hot_budget=hot_budget,
            genesis_anchor_count=genesis_count,
            genesis_anchor_recall=genesis_recall,
        )

    target_budget = _target_hot_budget(total_tokens, policy)
    candidate_budgets = _candidate_budgets(ranked, min_budget, target_budget, max_allowed)
    best_budget = max_allowed
    best_quality = _budget_quality(quality_index, best_budget, policy)
    for candidate_budget in candidate_budgets:
        quality = _budget_quality(quality_index, candidate_budget, policy)
        if not quality[2]:
            best_budget = candidate_budget
            best_quality = quality
            break
    recommended = best_budget
    if best_quality[2]:
        recommended = _recommended_hot_budget(quality_index, policy)
    return _decision(
        best_budget,
        policy,
        max_spans,
        "adaptive",
        best_quality[0],
        best_quality[1],
        best_quality[2],
        total_tokens,
        recommended_hot_budget=recommended,
        genesis_anchor_count=best_quality[3],
        genesis_anchor_recall=best_quality[4],
    )


def _decision(
    hot_budget: int,
    policy: MemoryPolicy,
    max_spans: int,
    mode: MemoryBudgetMode,
    exact_recall: float,
    survival_mass: float,
    underbudgeted: bool,
    total_tokens: int,
    *,
    recommended_hot_budget: int,
    genesis_anchor_count: int,
    genesis_anchor_recall: float,
) -> _BudgetDecision:
    warm_budget = max(policy.warm_token_budget, hot_budget * 2)
    actual_reduction = 1.0 - min(hot_budget, total_tokens) / max(1, total_tokens)
    return _BudgetDecision(
        hot_budget=max(1, hot_budget),
        warm_budget=max(1, warm_budget),
        max_spans=max(1, max_spans),
        mode_used=mode,
        target_reduction=policy.target_token_reduction,
        actual_reduction=round(actual_reduction, 4),
        exact_recall=round(exact_recall, 4),
        survival_mass=round(survival_mass, 4),
        underbudgeted=underbudgeted,
        recommended_hot_budget=max(1, recommended_hot_budget),
        recent_tail_required=underbudgeted and policy.recent_tail_token_budget > 0,
        genesis_anchor_count=genesis_anchor_count,
        genesis_anchor_recall=round(genesis_anchor_recall, 4),
    )


def _max_allowed_hot_budget(total_tokens: int, policy: MemoryPolicy) -> int:
    if policy.context_window_tokens:
        available_context = max(1, policy.context_window_tokens - policy.recent_tail_token_budget)
        if policy.memory_context_ratio:
            available_context = min(available_context, _ratio_hot_budget(policy))
        ceiling = max(1, max(policy.hot_token_budget, available_context))
        return min(ceiling, total_tokens) if total_tokens > 0 else ceiling
    if policy.memory_context_ratio:
        ceiling = max(policy.hot_token_budget, _ratio_hot_budget(policy))
        return min(ceiling, total_tokens) if total_tokens > 0 else ceiling
    if policy.target_token_reduction is not None:
        return max(policy.hot_token_budget, total_tokens)
    return policy.hot_token_budget


def _ratio_hot_budget(policy: MemoryPolicy) -> int:
    if not policy.context_window_tokens or not policy.memory_context_ratio:
        return policy.hot_token_budget
    return max(1, int(policy.context_window_tokens * policy.memory_context_ratio))


def _target_hot_budget(total_tokens: int, policy: MemoryPolicy) -> int:
    if policy.target_token_reduction is None:
        return policy.hot_token_budget
    return max(1, int(total_tokens * (1.0 - policy.target_token_reduction)))


def _candidate_budgets(
    ranked: list[SpanRecord],
    min_budget: int,
    target_budget: int,
    max_allowed: int,
) -> list[int]:
    budgets = {max(1, min_budget), max(1, target_budget), max(1, max_allowed)}
    running_tokens = 0
    for span in ranked:
        running_tokens += max(0, span.token_count)
        if min_budget <= running_tokens <= max_allowed and _is_quality_gate_span(span, policy=None):
            budgets.add(running_tokens)
    return sorted(budget for budget in budgets if budget <= max_allowed)


def _build_quality_index(
    ranked: list[SpanRecord],
    max_spans: int,
    policy: MemoryPolicy,
) -> _QualityIndex:
    cumulative_tokens: list[int] = []
    cumulative_exact: list[int] = []
    cumulative_survival: list[float] = []
    cumulative_gate: list[int] = []
    cumulative_genesis: list[int] = []
    running_tokens = 0
    running_exact = 0
    running_survival = 0.0
    running_gate = 0
    running_genesis = 0
    considered = ranked[:max_spans]
    total_exact = sum(1 for span in considered if span.scores.exact_critical >= policy.exact_critical_floor)
    total_survival = sum(max(0.0, span.scores.survival_value) for span in considered) or 1.0
    total_genesis = sum(1 for span in considered if _is_genesis_anchor(span, policy))
    for span in considered:
        running_tokens += max(0, span.token_count)
        if span.scores.exact_critical >= policy.exact_critical_floor:
            running_exact += 1
        running_survival += max(0.0, span.scores.survival_value)
        if _is_quality_gate_span(span, policy=policy):
            running_gate += 1
        if _is_genesis_anchor(span, policy):
            running_genesis += 1
        cumulative_tokens.append(running_tokens)
        cumulative_exact.append(running_exact)
        cumulative_survival.append(running_survival)
        cumulative_gate.append(running_gate)
        cumulative_genesis.append(running_genesis)
    return _QualityIndex(
        cumulative_tokens=cumulative_tokens,
        cumulative_exact=cumulative_exact,
        cumulative_survival=cumulative_survival,
        cumulative_gate=cumulative_gate,
        cumulative_genesis=cumulative_genesis,
        total_exact=total_exact,
        total_survival=total_survival,
        total_genesis=total_genesis,
    )


def _budget_quality(
    index: _QualityIndex,
    budget: int,
    policy: MemoryPolicy,
) -> tuple[float, float, bool, int, float]:
    selected_index = bisect_right(index.cumulative_tokens, budget) - 1
    if selected_index < 0:
        exact_count = 0
        selected_survival = 0.0
        gate_count = 0
        genesis_count = 0
    else:
        exact_count = index.cumulative_exact[selected_index]
        selected_survival = index.cumulative_survival[selected_index]
        gate_count = index.cumulative_gate[selected_index]
        genesis_count = index.cumulative_genesis[selected_index]
    exact_recall = (
        exact_count / index.total_exact
        if index.total_exact
        else 1.0
    )
    survival_mass = min(1.0, selected_survival / index.total_survival)
    stranded_high = bool(index.cumulative_gate and gate_count < index.cumulative_gate[-1])
    genesis_recall = genesis_count / index.total_genesis if index.total_genesis else 1.0
    underbudgeted = (
        exact_recall < policy.min_exact_critical_recall
        or survival_mass < policy.min_survival_mass
        or stranded_high
        or genesis_recall < 1.0
    )
    return exact_recall, survival_mass, underbudgeted, genesis_count, genesis_recall


def _recommended_hot_budget(
    index: _QualityIndex,
    policy: MemoryPolicy,
) -> int:
    recommended = policy.hot_token_budget
    for running_tokens in index.cumulative_tokens:
        exact_recall, survival_mass, underbudgeted, _, _ = _budget_quality(index, running_tokens, policy)
        recommended = running_tokens
        if (
            not underbudgeted
            and exact_recall >= policy.min_exact_critical_recall
            and survival_mass >= policy.min_survival_mass
        ):
            break
    return max(1, recommended)


def _is_quality_gate_span(span: SpanRecord, *, policy: MemoryPolicy | None) -> bool:
    floor = policy.exact_critical_floor if policy is not None else 0.72
    return (
        span.scores.exact_critical >= floor
        or span.scores.survival_value >= floor
        or span.span_type in {"code_symbol", "code_fragment", "error", "tool_result"}
        or span.text.split(maxsplit=1)[0].endswith("_exact_index")
        or (policy is not None and _is_genesis_anchor(span, policy))
    )


def _selection_score(span: SpanRecord, policy: MemoryPolicy) -> float:
    score = span.scores.survival_value / max(1.0, span.token_count) ** policy.rho
    if span.text.split(maxsplit=1)[0].endswith("_exact_index"):
        prefix = span.text.split(maxsplit=1)[0]
        if span.span_type == "code_symbol" and span.source in {"exact_index_raw_context", "exact_index_query"}:
            score *= 128.0
        elif span.span_type == "code_symbol" or span.source in {"exact_index_response", "exact_index_query"}:
            score *= 48.0
        elif span.source == "exact_index_raw_context" and span.span_type == "retrieval_chunk":
            score *= 12.0
        else:
            score *= 24.0
        if prefix in {"rag_exact_index", "search_exact_index"}:
            score *= max(0.15, 1.0 - span.scores.staleness)
        elif prefix == "tool_exact_index":
            score *= 4.0 if span.scores.staleness <= 0.01 else 1.0
        elif prefix == "chat_exact_index":
            score *= max(0.15, 1.0 - span.scores.staleness)
    if _is_genesis_anchor(span, policy):
        score *= 16.0
    return score


def _is_genesis_anchor(span: SpanRecord, policy: MemoryPolicy) -> bool:
    if policy.genesis_anchor_turns <= 0 or span.created_turn > policy.genesis_anchor_turns:
        return False
    if span.span_type not in {"goal", "constraint", "decision", "error", "code_symbol", "code_fragment", "tool_result"}:
        return False
    if span.scores.exact_critical >= policy.genesis_anchor_score_floor:
        return True
    if span.scores.survival_value >= policy.genesis_anchor_score_floor:
        return True
    prefix = span.text.split(maxsplit=1)[0] if span.text else ""
    return prefix.endswith("_exact_index")


def hot_context(spans: list[SpanRecord]) -> str:
    return "\n".join(span.text for span in spans if span.layer == "hot")


def _top_reason(span: SpanRecord) -> str:
    scores = span.scores
    items = {
        "salience": scores.salience,
        "relevance": scores.relevance,
        "attribution": scores.attribution,
        "exact_critical": scores.exact_critical,
    }
    return max(items.items(), key=lambda item: item[1])[0]
