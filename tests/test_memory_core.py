from __future__ import annotations

from latence_trace.memory.models import MemoryPolicy, MemoryUpdateRequest
from latence_trace.memory.service import update_memory


def test_memory_update_preserves_exact_critical_code_spans() -> None:
    response = update_memory(
        MemoryUpdateRequest(
            query_text="Fix the cache latency regression in src/cache.py",
            turn_text="Always preserve the public API name CacheClient.get_many.",
            response_text=(
                "Implemented the fix in src/cache.py. "
                "Benchmark improved from 120 ms to 84 ms."
            ),
            memory_policy=MemoryPolicy(hot_token_budget=64, warm_token_budget=128),
        )
    )

    hot = response.hot_context
    assert "src/cache.py" in hot
    assert "120 ms" in hot or "84 ms" in hot
    assert response.diagnostics.exact_critical_spans >= 1
    assert any(action.action == "anchored" for action in response.actions)


def test_memory_update_dedups_repeated_constraint() -> None:
    first = update_memory(
        MemoryUpdateRequest(turn_text="Never expose raw patient names in the answer.")
    )
    second = update_memory(
        MemoryUpdateRequest(
            turn_text="Never expose raw patient names in the answer.",
            prior_memory_state=first.next_memory_state,
        )
    )

    assert len(second.next_memory_state.spans) == len(first.next_memory_state.spans)
    assert any(action.action == "deduped" for action in second.actions)


def test_memory_update_accepts_learned_ranker_signal() -> None:
    response = update_memory(
        MemoryUpdateRequest(
            query_text="Keep the relevant support fact.",
            raw_context="Relevant support fact: policy clause 14.2 expires on 2026-05-30.",
            memory_ranker_weights={
                "salience": 1.0,
                "relevance": 1.0,
                "exact_critical": 1.0,
                "dead_weight": -1.0,
            },
            memory_policy=MemoryPolicy(hot_token_budget=64, warm_token_budget=128),
        )
    )

    assert response.next_memory_state.spans
    assert any(span.scores.learned_survival > 0.0 for span in response.next_memory_state.spans)
    assert "learned_survival" in response.diagnostics.top_survival_causes[0]
