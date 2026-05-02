from __future__ import annotations

from latence_trace.memory.models import MemoryPolicy, MemoryUpdateRequest
from latence_trace.memory.scorecard import build_scorecard
from latence_trace.memory.service import update_memory
from latence_trace.memory.trajectory import coerce_trajectory


def _age_memory(domain: str, initial_context: str, later_turns: int = 5):
    state = update_memory(
        MemoryUpdateRequest(
            query_text="Investigate active task",
            turn_text="Initial task context",
            response_text="Stored the initial context.",
            raw_context=initial_context,
            memory_domain=domain,
            memory_policy=MemoryPolicy(hot_token_budget=128, warm_token_budget=512),
        )
    ).next_memory_state
    for idx in range(later_turns):
        state = update_memory(
            MemoryUpdateRequest(
                query_text=f"Unrelated turn {idx}",
                turn_text=f"New unrelated work item {idx}",
                response_text=f"Handled unrelated item {idx}.",
                prior_memory_state=state,
                memory_domain=domain,
                memory_policy=MemoryPolicy(hot_token_budget=128, warm_token_budget=512),
            )
        ).next_memory_state
    return state


def test_agentic_search_rag_spans_decay_faster_than_coding_anchors() -> None:
    rag_state = _age_memory(
        "rag",
        "Retrieved support chunk: refund policy was updated to 14 days for this answer only.",
    )
    code_state = _age_memory(
        "code",
        "src/cache.py contains CacheClient.get_many and pytest tests/test_cache.py failed with 120 ms latency.",
    )

    rag_original = next(span for span in rag_state.spans if "refund policy" in span.text)
    code_original = next(span for span in code_state.spans if "src/cache.py" in span.text)

    assert rag_original.scores.survival_horizon_turns < code_original.scores.survival_horizon_turns
    assert rag_original.scores.domain_decay_pressure > code_original.scores.domain_decay_pressure
    assert code_original.scores.survival_value > rag_original.scores.survival_value


def test_scorecard_surfaces_domain_horizon_thesis() -> None:
    rag = coerce_trajectory(
        {
            "id": "rag-short",
            "dataset": "synthetic-rag",
            "domain": "rag",
            "turns": [
                {
                    "query_text": "What is the refund policy?",
                    "turn_text": "Retrieved answer-specific policy.",
                    "response_text": "Refunds take 14 days.",
                    "raw_context": "Refunds take 14 days for this answer.",
                }
            ],
            "critical_terms": ["14 days"],
        }
    )
    code = coerce_trajectory(
        {
            "id": "code-long",
            "dataset": "synthetic-code",
            "domain": "code",
            "turns": [
                {
                    "query_text": "Fix src/cache.py",
                    "turn_text": "Keep CacheClient.get_many stable.",
                    "response_text": "Fixed src/cache.py and preserved CacheClient.get_many.",
                    "raw_context": "src/cache.py contains CacheClient.get_many.",
                }
            ],
            "critical_terms": ["src/cache.py", "CacheClient.get_many"],
        }
    )

    scorecard = build_scorecard([rag, code])

    thesis = scorecard["domain_horizon_thesis"]
    assert thesis["claim"] == "agentic_search_rag_spans_should_have_shorter_survival_than_agentic_coding_spans"
    assert "code" in thesis["anchor_or_keep_ratio_by_domain"]
    assert "rag" in thesis["anchor_or_keep_ratio_by_domain"]
