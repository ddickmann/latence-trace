"""Hot/warm/cold memory selection."""

from __future__ import annotations

from latence_trace.memory.models import MemoryAction, MemoryDiagnostics, MemoryPolicy, SpanRecord


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
    hot_tokens = 0
    warm_tokens = 0
    kept: list[SpanRecord] = []
    for span in ranked[: policy.max_spans]:
        previous = span.layer
        if hot_tokens + span.token_count <= policy.hot_token_budget:
            span.layer = "hot"
            hot_tokens += span.token_count
        elif warm_tokens + span.token_count <= policy.warm_token_budget:
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
    return score


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
