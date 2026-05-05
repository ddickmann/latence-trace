# latence-trace

Thin Python SDK for **Latence TRACE**. It exposes the production API surface:
privacy redaction, RAG/code grounding, compression, InfiniMem updates, rollup,
and SDK-managed sessions. It does not ship models or runtime dependencies.

## Install

```bash
pip install latence-trace                 # thin SDK
pip install "latence-trace[langchain]"    # optional LangChain adapter
pip install "latence-trace[llama_index]"  # optional LlamaIndex adapter
pip install "latence-trace[openai]"       # optional OpenAI wrapper
pip install "latence-trace[otel]"         # optional OTel propagation
```

## Quick start

```python
from latence_trace_client import LatenceTraceClient

client = LatenceTraceClient(
    base_url="https://lt.acme.internal",
    api_key="lt_live_…",            # the JWT license, set once
    timeout=30.0,
)

result = client.grounding.rag(
    query="When was Newton born?",
    response_text="Newton was born in 1643.",
    raw_context="Sir Isaac Newton was born on 25 December 1642.",
)

print(result.scores.groundedness_v2, result.risk_band, result.scores.coverage_score_u)
```

## Compliance redaction

```python
from latence_trace_client import LatenceTraceClient

with LatenceTraceClient(base_url="https://api.latence.ai", api_key="lt_live_...") as client:
    out = client.privacy.redact(
        text="Contact Jane Doe at jane@example.com before sending the prompt.",
        labels=["person", "email"],
        redact=True,
        include_original_text=False,
    )

print(out.redacted_text)
print(out.entity_count, out.unique_labels)
```

Use `mode="open"` for the full PII catalog or pass `labels` / `categories`
for a tighter allowlist. The SDK returns typed `ComplianceEntity`,
`ComplianceUsage`, and timing metadata; do not persist `entities[*].text` in
analytics.

## Async + retry-on-5xx + OTel

```python
import asyncio
from latence_trace_client import AsyncLatenceTraceClient

async def main():
    async with AsyncLatenceTraceClient(base_url="…", api_key="…") as c:
        out = await c.grounding.rag(query="...", response_text="...", raw_context="...")
        print(out.risk_band)

asyncio.run(main())
```

The client retries on 5xx + 429 with exponential backoff and honors
`Retry-After` on 429 responses. Guard checks default to enabled:

```python
client.grounding.rag(
    query="Can I refund this invoice?",
    response_text="The refund is approved.",
    raw_context="Refund status: pending manager approval.",
    context_trust_enabled=False,  # only disable for trusted latency isolation
)
```

## Sessions

`TraceSession` keeps session state in the SDK while the compute runtime remains
stateless:

```python
from latence_trace_client import InMemorySessionStorage, LatenceTraceClient

storage = InMemorySessionStorage()
with LatenceTraceClient(base_url="https://trace.acme.internal") as client:
    session = client.session(session_id="agent-42", storage=storage)
    session.event("tool", "retrieved invoice and refund policy")
    session.memory_step(turn_text="User needs refund status preserved.")
    result = session.rag(
        query="Can we approve the refund?",
        response_text="Approve the refund now.",
        raw_context="Refund status is pending manager review.",
    )
```

Use `FileSessionStorage` for durable local sessions or implement the
`SessionStorage` protocol for Redis, Postgres, or a customer-owned store.

## Adapters

| Framework | Module | What it does |
|---|---|---|
| LangChain | `latence_trace_client.integrations.langchain` | Convert chain events into TRACE checks |
| LlamaIndex | `latence_trace_client.integrations.llama_index` | Attach TRACE scores to retrieved nodes |
| OpenAI | `latence_trace_client.integrations.openai` | Wrap chat completions and score assistant turns |

See `examples/` for end-to-end snippets.
