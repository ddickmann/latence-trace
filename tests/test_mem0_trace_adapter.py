from __future__ import annotations

from latence_trace.memory.models import MemoryPolicy
from scripts.mem0_trace_adapter import TraceMem0AdapterConfig, TraceMemoryClient


def _client(*, hot_token_budget: int = 64, memory_domain: str = "rag") -> TraceMemoryClient:
    return TraceMemoryClient(
        config=TraceMem0AdapterConfig(
            memory_domain=memory_domain,
            memory_policy=MemoryPolicy(
                hot_token_budget=hot_token_budget,
                warm_token_budget=128,
                memory_budget_mode="adaptive",
                context_window_tokens=512,
                memory_context_ratio=0.5,
                target_token_reduction=0.5,
                max_spans=32,
            ),
        )
    )


async def test_trace_mem0_adapter_add_and_search_shape() -> None:
    client = _client(memory_domain="tool")

    response = await client.add(
        [
            {"role": "user", "content": "Check support ticket T-771."},
            {
                "role": "assistant",
                "content": "Tool result: refund_status=approved replacement_order=R-90017.",
            },
        ],
        user_id="u-shape",
    )
    results = await client.search(
        "What is the refund_status for ticket T-771 and replacement order R-90017?",
        user_id="u-shape",
        top_k=5,
        score_debug=True,
    )

    assert response is not None
    assert response["results"]
    assert results
    assert {"memory", "score", "id"}.issubset(results[0])
    assert "score_debug" in results[0]
    assert any("refund_status=approved" in item["memory"] for item in results)


async def test_trace_mem0_adapter_delete_user_resets_search() -> None:
    client = _client()
    await client.add(
        [{"role": "user", "content": "Policy clause 14.2 requires 30 days notice."}],
        user_id="u-delete",
    )

    assert await client.search("What does policy clause 14.2 require?", "u-delete", top_k=5)
    assert await client.delete_user("u-delete")
    assert await client.search("What does policy clause 14.2 require?", "u-delete", top_k=5) == []


async def test_trace_mem0_adapter_includes_source_vault_repair_rows() -> None:
    client = _client(hot_token_budget=1)
    await client.add(
        [
            {
                "role": "user",
                "content": "Legal matter LGL-042 must close on 2026-05-06 after final approval.",
            }
        ],
        user_id="u-repair",
    )

    results = await client.search(
        "What must happen on 2026-05-06?",
        user_id="u-repair",
        top_k=10,
        score_debug=True,
    )

    assert any(str(item["id"]).startswith("repair:") for item in results)
    assert any("2026-05-06" in item["memory"] for item in results)
