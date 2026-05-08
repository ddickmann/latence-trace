"""Rule-based survival scoring for InfiniMem v1."""

from __future__ import annotations

from typing import Any

from latence_trace.memory.models import MemoryAction, MemoryPolicy, SpanRecord
from latence_trace.memory.ranker import score_span
from latence_trace.memory.signature import extract_exact_critical_terms

_IMPORTANT_TYPES = {"goal", "constraint", "decision", "error", "code_symbol", "code_fragment"}


def update_survival(
    spans: list[SpanRecord],
    *,
    turn_index: int,
    policy: MemoryPolicy,
    trace_signals: dict[str, Any] | None = None,
) -> tuple[list[SpanRecord], list[MemoryAction]]:
    signals = trace_signals or {}
    domain = _memory_domain(signals)
    ranker_weights = _ranker_weights(signals)
    actions: list[MemoryAction] = []
    for span in spans:
        if span.layer == "tombstone":
            continue
        age = max(0, turn_index - span.last_seen_turn)
        horizon = _survival_horizon(span, domain=domain, policy=policy)
        domain_decay = min(1.0, age / horizon)
        exact = _exact_critical(span, domain=domain)
        salience = _salience(span)
        relevance = _relevance(span, signals)
        attribution = _attribution(span, signals)
        redundancy = max(span.scores.redundancy, 0.0)
        dead_weight = _dead_weight(span, signals)
        staleness = domain_decay
        survival = (
            0.30 * salience
            + 0.22 * relevance
            + 0.18 * attribution
            + 0.20 * exact
            - 0.16 * redundancy
            - 0.14 * dead_weight
            - 0.16 * staleness
        )
        learned = score_span(span, ranker_weights, domain=domain) if ranker_weights else 0.0
        if ranker_weights:
            # Learned survival is an advisory annealing signal, not a replacement
            # for the deterministic exact-critical guard below.
            survival = 0.85 * survival + 0.15 * learned
        if exact >= policy.exact_critical_floor:
            survival = max(survival, policy.exact_critical_floor)
        span.scores.salience = _bounded(salience)
        span.scores.relevance = _bounded(relevance)
        span.scores.attribution = _bounded(attribution)
        span.scores.exact_critical = _bounded(exact)
        span.scores.redundancy = _bounded(redundancy)
        span.scores.dead_weight = _bounded(dead_weight)
        span.scores.staleness = _bounded(staleness)
        span.scores.learned_survival = _bounded(learned)
        span.scores.survival_horizon_turns = float(horizon)
        span.scores.domain_decay_pressure = _bounded(domain_decay)
        span.scores.survival_value = _bounded(survival)
        if span.scores.survival_value >= policy.exact_critical_floor and span.span_type in _IMPORTANT_TYPES:
            actions.append(MemoryAction(action="anchored", span_id=span.id, reason=f"{domain}_high_survival_anchor"))
    return spans, actions


def _bounded(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _exact_critical(span: SpanRecord, *, domain: str) -> float:
    score = 0.0
    exact_index_prefix = span.text.split(maxsplit=1)[0] if span.text else ""
    if exact_index_prefix in {"tool_exact_index", "code_exact_index", "grounding_exact_index"}:
        score += 0.55
    if span.span_type in {"constraint", "decision", "code_symbol", "code_fragment", "error"}:
        score += 0.45
    if span.signature.file_paths or span.signature.symbols:
        score += 0.35
    if span.signature.numbers or span.signature.dates:
        score += 0.25
    if "must" in span.text.lower() or "never" in span.text.lower():
        score += 0.15
    exact_terms = extract_exact_critical_terms(span.text, domain=domain)
    if exact_terms:
        score += min(0.55, 0.15 + 0.08 * len(exact_terms))
    if domain == "tool" and span.span_type == "tool_result" and exact_terms:
        score += 0.20
    if domain in {"rag", "search"} and span.span_type in {"retrieval_chunk", "evidence"} and exact_terms:
        score += 0.12
    if domain in {"rag", "search"} and span.span_type in {"retrieval_chunk", "evidence"}:
        score = min(score, 0.68)
    return _bounded(score)


def _salience(span: SpanRecord) -> float:
    if span.span_type in _IMPORTANT_TYPES:
        return 0.85
    if span.span_type in {"retrieval_chunk", "evidence", "tool_result"}:
        return 0.55
    return 0.25


def _relevance(span: SpanRecord, signals: dict[str, Any]) -> float:
    text = span.text.lower()
    query = str(signals.get("query_text") or "").lower()
    if query:
        query_terms = {term for term in query.split() if len(term) > 3}
        if query_terms:
            return min(1.0, len([term for term in query_terms if term in text]) / len(query_terms) + 0.2)
    features = signals.get("runtime_head_features") or signals.get("trajectory_features") or {}
    if span.span_type.startswith("code") and isinstance(features, dict):
        return max(
            float(features.get("file_alignment", 0.0) or 0.0),
            float(features.get("symbol_alignment", 0.0) or 0.0),
        )
    return 0.45


def _attribution(span: SpanRecord, signals: dict[str, Any]) -> float:
    if span.source in {"response", "query"}:
        return 0.65
    scores = signals.get("scores") or {}
    if isinstance(scores, dict):
        base = float(scores.get("context_attribution_ratio", scores.get("context_usage_ratio", 0.4)) or 0.4)
        verdict = scores.get("grounded")
        if verdict is False and span.source.startswith(("raw_context", "exact_index_raw")):
            base *= 0.5
        return base
    return 0.4


def _dead_weight(span: SpanRecord, signals: dict[str, Any]) -> float:
    scores = signals.get("scores") or {}
    if isinstance(scores, dict):
        base = float(scores.get("dead_weight_ratio", scores.get("context_unused_ratio", 0.0)) or 0.0)
        verdict = scores.get("grounded")
        if verdict is False and span.source.startswith(("raw_context", "exact_index_raw")):
            base = max(base, 0.35)
    else:
        base = 0.0
    if span.span_type in _IMPORTANT_TYPES:
        base *= 0.4
    return _bounded(base)


def _memory_domain(signals: dict[str, Any]) -> str:
    raw = str(
        signals.get("trajectory_domain")
        or signals.get("memory_domain")
        or signals.get("domain")
        or ""
    ).strip().lower()
    if raw in {"search", "agentic_search"}:
        return "search"
    if raw in {"rag", "grounding", "agentic_rag"}:
        return "rag"
    if raw in {"code", "coding", "agentic_code", "code.agentic_trace"}:
        return "code"
    if raw in {"tool", "workflow", "tau"}:
        return "tool"
    return "chat"


def _ranker_weights(signals: dict[str, Any]) -> dict[str, float]:
    raw = signals.get("memory_ranker_weights") or signals.get("ranker_weights") or {}
    if not isinstance(raw, dict):
        return {}
    weights: dict[str, float] = {}
    for key, value in raw.items():
        try:
            weights[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return weights


def _survival_horizon(span: SpanRecord, *, domain: str, policy: MemoryPolicy) -> float:
    if domain == "search":
        base = policy.search_horizon_turns
    elif domain == "rag":
        base = policy.rag_horizon_turns
    elif domain == "code":
        base = policy.code_horizon_turns
    elif domain == "tool":
        base = policy.tool_horizon_turns
    else:
        base = policy.chat_horizon_turns

    if domain == "code" and _is_code_anchor(span):
        base = max(base, policy.code_anchor_horizon_turns)
    elif domain in {"rag", "search"} and span.span_type == "retrieval_chunk":
        base = max(1, int(round(base * 0.75)))
    elif span.span_type in {"constraint", "decision"}:
        base = int(round(base * 1.5))
    if span.scores.redundancy >= 0.7:
        base = max(1, int(round(base * 0.5)))
    return float(max(1, base))


def _is_code_anchor(span: SpanRecord) -> bool:
    return (
        span.span_type in {"code_symbol", "code_fragment", "error", "constraint", "decision"}
        or bool(span.signature.file_paths)
        or bool(span.signature.symbols)
    )
