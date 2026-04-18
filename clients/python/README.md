# latence-trace-client

Official Python SDK for the **latence-trace** Groundedness Tracker --
calibrated, auditable hallucination detection for RAG and
evidence-bearing LLM outputs.

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
| LangChain | `latence_trace_client.integrations.langchain` | `LatenceTraceCallback` -- score after every chain output |
| LlamaIndex | `latence_trace_client.integrations.llama_index` | `LatenceTracePostProcessor` -- attach groundedness to nodes |
| OpenAI | `latence_trace_client.integrations.openai` | `wrap_openai_chat()` -- score the assistant turn against prompt context |

See `examples/` for end-to-end snippets.
