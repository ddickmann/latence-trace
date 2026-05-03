from __future__ import annotations

import json

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


def test_memory_policy_context_ratio_scales_hot_budget() -> None:
    payload = json.dumps(
        [
            {"order_id": f"ORD-{idx:04d}", "status": f"valuable_{idx}"}
            for idx in range(8)
        ]
    )
    policy = MemoryPolicy(
        hot_token_budget=8,
        warm_token_budget=16,
        context_window_tokens=1_000,
        memory_context_ratio=0.08,
        max_spans=2,
    )
    response = update_memory(
        MemoryUpdateRequest(
            raw_context=payload,
            memory_domain="tool",
            memory_policy=policy,
        )
    )

    assert response.diagnostics.effective_hot_token_budget == 80
    assert response.diagnostics.effective_warm_token_budget == 160
    assert response.diagnostics.effective_max_spans > policy.max_spans
    assert response.diagnostics.hot_tokens > policy.hot_token_budget
    assert "ORD-0000" in response.hot_context
    assert "ORD-0007" in response.hot_context


def test_memory_policy_context_ratio_does_not_shrink_explicit_budget() -> None:
    response = update_memory(
        MemoryUpdateRequest(
            raw_context='{"order_id":"ORD-1234","status":"paid"}',
            memory_domain="tool",
            memory_policy=MemoryPolicy(
                hot_token_budget=64,
                warm_token_budget=128,
                context_window_tokens=100,
                memory_context_ratio=0.01,
            ),
        )
    )

    assert response.diagnostics.effective_hot_token_budget == 64
    assert "ORD-1234" in response.hot_context


def test_adaptive_memory_expands_beyond_target_when_exact_terms_need_it() -> None:
    payload = json.dumps(
        [
            {
                "order_id": f"ORD-{idx:04d}",
                "payment_id": f"PAY-{idx:04d}",
                "status": f"valuable_{idx}",
            }
            for idx in range(12)
        ]
    )
    policy = MemoryPolicy(
        hot_token_budget=8,
        warm_token_budget=16,
        memory_budget_mode="adaptive",
        context_window_tokens=1_000,
        memory_context_ratio=0.5,
        target_token_reduction=0.95,
        min_exact_critical_recall=0.98,
        min_survival_mass=0.9,
        max_spans=2,
    )
    response = update_memory(
        MemoryUpdateRequest(raw_context=payload, memory_domain="tool", memory_policy=policy)
    )

    target_budget = int(response.diagnostics.hot_tokens * (1.0 - policy.target_token_reduction))
    assert response.diagnostics.budget_mode_used == "adaptive"
    assert response.diagnostics.effective_hot_token_budget > max(policy.hot_token_budget, target_budget)
    assert response.diagnostics.estimated_exact_critical_recall >= policy.min_exact_critical_recall
    assert response.diagnostics.survival_mass_retained >= policy.min_survival_mass
    assert response.diagnostics.memory_underbudgeted is False
    assert "ORD-0000" in response.hot_context
    assert "PAY-0011" in response.hot_context


def test_adaptive_memory_reports_underbudgeted_when_context_bound_is_too_small() -> None:
    payload = json.dumps(
        [
            {
                "order_id": f"ORD-{idx:04d}",
                "payment_id": f"PAY-{idx:04d}",
                "status": f"valuable_{idx}",
            }
            for idx in range(12)
        ]
    )
    response = update_memory(
        MemoryUpdateRequest(
            raw_context=payload,
            memory_domain="tool",
            memory_policy=MemoryPolicy(
                hot_token_budget=8,
                warm_token_budget=16,
                memory_budget_mode="adaptive",
                context_window_tokens=100,
                memory_context_ratio=0.3,
                target_token_reduction=0.95,
                min_exact_critical_recall=0.98,
                min_survival_mass=0.9,
                recent_tail_token_budget=10,
                max_spans=2,
            ),
        )
    )

    assert response.diagnostics.effective_hot_token_budget == 30
    assert response.diagnostics.memory_underbudgeted is True
    assert response.diagnostics.recent_tail_required is True
    assert response.diagnostics.recommended_hot_token_budget > response.diagnostics.effective_hot_token_budget
    assert response.diagnostics.estimated_exact_critical_recall < 0.98


def test_memory_diagnostics_include_stage_timings() -> None:
    response = update_memory(
        MemoryUpdateRequest(
            raw_context='{"order_id":"ORD-5678","status":"paid"}',
            memory_domain="tool",
            memory_policy=MemoryPolicy(memory_budget_mode="adaptive"),
        )
    )

    assert {"extract", "dedup", "survival", "selection", "total"} <= set(response.diagnostics.timings_ms)
    assert response.diagnostics.timings_ms["total"] >= 0.0


def test_adaptive_memory_preserves_first_turn_plan_concepts() -> None:
    policy = MemoryPolicy(
        hot_token_budget=16,
        warm_token_budget=64,
        memory_budget_mode="adaptive",
        context_window_tokens=1_000,
        memory_context_ratio=0.2,
        target_token_reduction=0.95,
        min_exact_critical_recall=0.98,
        min_survival_mass=0.8,
        genesis_anchor_turns=2,
    )
    first_turn = (
        "Plan: build TRACE v2 biaffine student for Real-Time Agentic Runtime Verification. "
        "Core concept: preserve src/trace/core.py, AdaptiveMemoryBudget, ReverseMaxSim, InfiniMemVault."
    )
    response = update_memory(
        MemoryUpdateRequest(
            query_text=first_turn,
            turn_text=first_turn,
            memory_domain="code",
            memory_policy=policy,
        )
    )
    state = response.next_memory_state
    for idx in range(30):
        response = update_memory(
            MemoryUpdateRequest(
                raw_context=f"noise generated_{idx}.py irrelevant status=ok_{idx}",
                response_text="continue",
                memory_domain="code",
                prior_memory_state=state,
                memory_policy=policy,
            )
        )
        state = response.next_memory_state

    assert response.diagnostics.genesis_anchor_spans >= 1
    assert response.diagnostics.genesis_anchor_recall == 1.0
    assert "src/trace/core.py" in response.hot_context
    assert "AdaptiveMemoryBudget" in response.hot_context
    assert "ReverseMaxSim" in response.hot_context
