"""Measured baseline strategies for TRACE Memory scorecards."""

from __future__ import annotations

from typing import Any

from latence_trace.memory.labels import label_trajectory
from latence_trace.memory.models import MemoryPolicy
from latence_trace.memory.ranker import score_span
from latence_trace.memory.trajectory import CanonicalTrajectory


def evaluate_baselines(
    trajectory: CanonicalTrajectory,
    *,
    learned_weights: dict[str, float] | None = None,
    policy: MemoryPolicy | None = None,
) -> dict[str, dict[str, Any]]:
    policy = policy or MemoryPolicy(hot_token_budget=160, warm_token_budget=800)
    full_text = trajectory.full_history_text()
    critical_terms = trajectory.critical_terms
    full_tokens = _tokens(full_text)
    state, labels, trace_hot = label_trajectory(
        trajectory,
        policy=policy,
        learned_weights=learned_weights,
    )
    spans = list(state.spans)
    baselines = {
        "sliding_window": _metrics(full_text, full_tokens, critical_terms, status="measured"),
        "plain_compression": _metrics(
            _plain_compress(full_text, policy.hot_token_budget),
            full_tokens,
            critical_terms,
            status="measured",
        ),
        "vector_memory": _metrics(
            _vector_select(trajectory, policy.hot_token_budget),
            full_tokens,
            critical_terms,
            status="measured",
        ),
        "summary_memory": _metrics(
            _summary_select(trajectory, policy.hot_token_budget),
            full_tokens,
            critical_terms,
            status="measured",
        ),
        "trace_memory_rule_based": _metrics(
            trace_hot,
            full_tokens,
            critical_terms,
            status="measured",
            diagnostics={"label_counts": _label_counts(labels)},
        ),
        "trace_memory_learned": _metrics(
            _learned_select(spans, policy.hot_token_budget, learned_weights or {}, domain=trajectory.domain),
            full_tokens,
            critical_terms,
            status="measured" if learned_weights else "untrained_fallback",
        ),
    }
    for budget in _matched_budgets(policy.hot_token_budget, full_tokens):
        suffix = f"{budget}"
        baselines[f"sliding_window_{suffix}"] = _metrics(
            _sliding_window(full_text, budget),
            full_tokens,
            critical_terms,
            status="measured",
            diagnostics={"budget": budget},
        )
        baselines[f"plain_compression_{suffix}"] = _metrics(
            _plain_compress(full_text, budget),
            full_tokens,
            critical_terms,
            status="measured",
            diagnostics={"budget": budget},
        )
        baselines[f"trace_memory_rule_based_{suffix}"] = _metrics(
            _budgeted_trace_text(spans, budget, mode="rule"),
            full_tokens,
            critical_terms,
            status="measured",
            diagnostics={"budget": budget, "label_counts": _label_counts(labels)},
        )
        baselines[f"trace_memory_learned_{suffix}"] = _metrics(
            _learned_select(spans, budget, learned_weights or {}, domain=trajectory.domain),
            full_tokens,
            critical_terms,
            status="measured" if learned_weights else "untrained_fallback",
            diagnostics={"budget": budget},
        )
    return baselines


def _metrics(
    text: str,
    full_tokens: int,
    critical_terms: list[str],
    *,
    status: str,
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tokens = _tokens(text)
    preserved = [term for term in critical_terms if term in text]
    exact_preservation = (
        len(preserved) / len(critical_terms)
        if critical_terms
        else 1.0
    )
    return {
        "status": status,
        "tokens": tokens,
        "token_reduction": round(1.0 - tokens / max(1, full_tokens), 4),
        "exact_critical_preservation": round(exact_preservation, 4),
        "exact_critical_term_count": len(critical_terms),
        "preserved_terms": preserved,
        "diagnostics": diagnostics or {},
    }


def _plain_compress(text: str, budget: int) -> str:
    tokens = text.split()
    if len(tokens) <= budget:
        return text
    head = tokens[: max(1, budget // 2)]
    tail = tokens[-max(1, budget - len(head)) :]
    return " ".join([*head, *tail])


def _sliding_window(text: str, budget: int) -> str:
    tokens = text.split()
    return " ".join(tokens[-budget:])


def _vector_select(trajectory: CanonicalTrajectory, budget: int) -> str:
    query_terms = {
        term.lower()
        for turn in trajectory.turns
        for term in (turn.query_text or "").split()
        if len(term) > 3
    }
    candidates = []
    for turn in trajectory.turns:
        parts = [turn.turn_text, turn.response_text or "", turn.raw_context or ""]
        parts.extend(span.text for span in turn.retrieved_context)
        parts.extend(span.text for span in turn.tool_outputs)
        for text in parts:
            score = sum(1 for term in query_terms if term in text.lower())
            candidates.append((score, text))
    selected: list[str] = []
    used = 0
    for _, text in sorted(candidates, reverse=True):
        token_count = _tokens(text)
        if not text.strip() or used + token_count > budget:
            continue
        selected.append(text)
        used += token_count
    return "\n".join(selected)


def _summary_select(trajectory: CanonicalTrajectory, budget: int) -> str:
    bullets = []
    for turn in trajectory.turns:
        source = turn.response_text or turn.turn_text
        if source:
            bullets.append(source.split(".")[0].strip())
    text = ". ".join(item for item in bullets if item)
    return " ".join(text.split()[:budget])


def _learned_select(spans: list[Any], budget: int, weights: dict[str, float], *, domain: str) -> str:
    def score(span: Any) -> float:
        if not weights:
            return span.scores.survival_value
        return score_span(span, weights, domain=domain)

    selected: list[str] = []
    used = 0
    for span in sorted(spans, key=score, reverse=True):
        if used + span.token_count > budget:
            continue
        selected.append(span.text)
        used += span.token_count
    return "\n".join(selected)


def _budgeted_trace_text(spans: list[Any], budget: int, *, mode: str) -> str:
    selected: list[str] = []
    used = 0
    if mode == "rule":
        ordered = sorted(spans, key=lambda span: span.scores.survival_value, reverse=True)
    else:
        ordered = list(spans)
    for span in ordered:
        if used + span.token_count > budget:
            continue
        selected.append(span.text)
        used += span.token_count
    return "\n".join(selected)


def _tokens(text: str) -> int:
    return len(text.split())


def _matched_budgets(hot_budget: int, full_tokens: int) -> list[int]:
    candidates = [hot_budget, 8_000, 32_000, 128_000, 200_000]
    return sorted({min(full_tokens, budget) for budget in candidates if budget > 0})


def _label_counts(labels: list[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for label in labels:
        counts[label.label] = counts.get(label.label, 0) + 1
    return counts
