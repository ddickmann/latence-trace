"""Score an agentic coding answer against code context."""

from latence_trace_client import LatenceTraceClient


with LatenceTraceClient() as client:
    result = client.grounding.code(
        query="Add retry logic to fetch_user.",
        response_text="```python\nasync def fetch_user(id):\n    return await client.get(f'/users/{id}')\n```",
        raw_context="async def fetch_user(id):\n    async with httpx.AsyncClient() as client:\n        return await client.get(f'/users/{id}')\n",
        extra={"response_language_hint": "python", "emit_chunk_ownership": True},
    )
    print(result.risk_band, result.runtime_decision)
