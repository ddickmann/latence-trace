# latence-trace-client

Official Python SDK for **Latence TRACE**: calibrated groundedness
verification plus real-time PII compliance redaction for enterprise AI apps.

## Install

```bash
pip install latence-trace-client                 # core client
pip install "latence-trace-client[langchain]"    # + LangChain adapter
pip install "latence-trace-client[llama_index]"  # + LlamaIndex adapter
pip install "latence-trace-client[openai]"       # + OpenAI wrapper
pip install "latence-trace-client[otel]"         # + OTel auto-instrument
```

## Quick start

```python
from latence_trace_client import LatenceTraceClient

client = LatenceTraceClient(
    base_url="https://lt.acme.internal",
    api_key="lt_live_…",            # the JWT license, set once
    timeout=30.0,
)

result = client.score_groundedness(
    query="When was Newton born?",
    response_text="Newton was born in 1643.",
    raw_context=[
        "Sir Isaac Newton (25 December 1642 – 20 March 1726/27) was an English mathematician..."
    ],
)

print(result.scores.groundedness_v2, result.risk_band, result.scores.coverage_score_u)
```

## Compliance redaction

```python
from latence_trace_client import LatenceTraceClient

with LatenceTraceClient(base_url="https://api.latence.ai", api_key="lt_live_...") as client:
    out = client.redact_compliance(
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
        out = await c.score_groundedness(query="…", response_text="…", chunk_ids=["c1","c2"])
        print(out.risk_band)

asyncio.run(main())
```

The client retries on 5xx + 429 with exponential backoff; honors the
`Retry-After` header on 429 responses; propagates W3C trace context
when `[otel]` extras are installed.

## Adapters

| Framework | Module | What it does |
|---|---|---|
| LangChain | `latence_trace_client.integrations.langchain` | `LatenceTraceCallback` scores outputs; `LatenceComplianceRedactor` redacts prompt inputs |
| LlamaIndex | `latence_trace_client.integrations.llama_index` | `LatenceTracePostProcessor` -- attach groundedness to nodes |
| OpenAI | `latence_trace_client.integrations.openai` | `wrap_openai_chat()` -- score the assistant turn against prompt context |

See `examples/` for end-to-end snippets.
