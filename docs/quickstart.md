# Quickstart: score your first RAG answer in 5 minutes

## 0. What you need

- Python 3.10+, Node 18+, or `curl`.
- A free-tier API key from `https://latence.ai/signup` (1 000
  scores / month on the `standard` profile; no credit card
  required).

## 1. The contract

TRACE takes three strings and returns a band.

```
POST https://api.latence.ai/v1/score/groundedness
Authorization: Bearer sk_live_xxx
Content-Type: application/json

{
  "question":       "...",            // the user question
  "response_text":  "...",            // the model's candidate answer
  "raw_context":    "...",            // the retrieved evidence
  "profile":        "standard"        // standard | quality | code
}
```

Returns:

```json
{
  "band": "green | amber | red",
  "groundedness": 0.91,
  "profile": "standard",
  "teacher_channels": { "nli_aggregate": 0.93, ... },
  "top_k_attributions": [ { "response_span": "...", "evidence_span": "..." } ],
  "request_id": "rid_..."
}
```

Fail-closed policy: treat `red` as "do not ship the answer"; treat
`amber` as "send to the reviewer queue."

## 2. curl

```bash
export LATENCE_API_KEY=sk_live_xxx
curl -s https://api.latence.ai/v1/score/groundedness \
  -H "Authorization: Bearer $LATENCE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
        "question": "What was our 2023 ARR?",
        "response_text": "ARR reached 12.4M USD in 2023.",
        "raw_context": "FY23 shareholder letter: ARR ended 2023 at 12.4M USD.",
        "profile": "standard"
      }' | jq .band
```

Expected output: `"green"`.

## 3. Python (sync)

```bash
pip install latence-trace-client
```

```python
import os
from latence_trace_client import LatenceTraceClient

trace = LatenceTraceClient(api_key=os.environ["LATENCE_API_KEY"])

res = trace.score_groundedness(
    query="What was our 2023 ARR?",
    response_text="ARR reached 12.4M USD in 2023.",
    raw_context="FY23 shareholder letter: ARR ended 2023 at 12.4M USD.",
    profile="standard",
)
print(res.band, res.groundedness)
```

## 4. TypeScript

```bash
npm install @latence/trace
```

```ts
import { LatenceTrace } from "@latence/trace";

const trace = new LatenceTrace({ apiKey: process.env.LATENCE_API_KEY! });
const res = await trace.scoreGroundedness({
  question: "What was our 2023 ARR?",
  responseText: "ARR reached 12.4M USD in 2023.",
  rawContext: "FY23 shareholder letter: ARR ended 2023 at 12.4M USD.",
});
console.log(res.band);
```

## 5. OpenAI SDK drop-in

```python
import os
from openai import OpenAI
from latence_trace_client import LatenceTraceClient
from latence_trace_client.integrations.openai import wrap_openai_chat

oai = OpenAI()
trace = LatenceTraceClient(api_key=os.environ["LATENCE_API_KEY"])

create_with_score = wrap_openai_chat(oai.chat.completions.create, trace)
resp = create_with_score(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "Answer using only the evidence."},
        {"role": "user", "content": "What was our 2023 ARR?"},
    ],
    raw_context="FY23 shareholder letter: ARR ended 2023 at 12.4M USD.",
)
print(resp.latence_trace.band)
```

## 6. Where to go next

- **Integrate into your stack**: `docs/integrations.md` (LangChain,
  LlamaIndex, LangGraph, CrewAI, AutoGen, Haystack, Pydantic AI).
- **Agents only**: drop-in MCP manifests at
  `clients/mcp-manifests/` so Cursor, Claude Desktop, and any
  MCP-aware agent can score without writing glue code.
- **n8n**: `npm install n8n-nodes-latence-trace` or use the HTTP
  Request blueprint at
  `clients/n8n-nodes-latence-trace/workflows/rag-review-http-request.json`.
- **Understand what TRACE is doing**: `docs/guides/beta-overview.md`
  and the `teacher_channels` section in `docs/api-reference.md`.
- **Prove the claims**:
  `data/veracier-industries/proof_bundle_v1/` + `external_benchmarks.md`.
- **Hosted SLA + pricing**: `commercial/hosted-sla.md` and
  `commercial/hosted-price-card.md`.

## 7. Colab + curl-only variants

- Colab notebook (self-contained, runs against the hosted free
  tier): `docs/quickstart.ipynb`.
- Pure curl script (no language runtime required):
  `docs/quickstart.sh`.
