# End-to-end Groundedness Tracker Tutorial

This tutorial walks through every concept and surface of the
**Groundedness Tracker** ships in `latence-trace`. By the end
you'll be able to:

1. Boot the service locally and confirm it's healthy.
2. Send your first groundedness request and read the response.
3. Choose between the three premise lanes (`raw_context`, `chunk_ids`,
   `support_units`) for your retrieval pipeline.
4. Interpret the headline scores, calibrated risk bands, NLI claim
   verdicts, and the per-token heatmap.
5. Tune retrieval using the **context-coverage observability metric**
   (`context_coverage_ratio`, per-unit `coverage_score` / `used`).
6. Pick a profile (`fast` / `balanced` / `quality`) and switch
   languages (English / German) with one env knob.
7. Wire the service into LangChain, LlamaIndex, raw HTTP, the Python
   SDK, and AI agents (MCP / OpenAI tools).

Every step uses the in-process service so you can follow along without
GPU or HuggingFace downloads. Production paths (vLLM-Factory, GPU,
multilingual ColBERT/NLI) are called out where they differ.

---

## 0. Install

```bash
pip install -e .                     # editable install for local dev
# or, for Python application integration:
pip install latence                   # PyPI SDK
```

---

## 1. Boot the service

The fastest path is to use the in-process service for tests and the
CLI for everything else.

```bash
# Boot the FastAPI server with the balanced profile (default).
latence-trace serve --profile balanced --port 8080
```

In another shell, confirm the service is alive:

```bash
curl -s http://localhost:8080/healthz
# {"status":"ok"}

curl -s http://localhost:8080/readyz | jq
# {
#   "status": "ready",
#   "warmup": {"triton_kernels": "warm", "encoder_loaded": true},
#   "max_inflight": 32
# }
```

Self-describing discovery surfaces for AI agents and developers:

| Endpoint                          | Returns                                                              |
|-----------------------------------|----------------------------------------------------------------------|
| `GET /agent-help`                 | Canonical request shape, profiles, observability axes, endpoint map  |
| `GET /.well-known/ai-plugin.json` | ChatGPT-style plugin descriptor                                      |
| `GET /openapi.json`               | OpenAPI 3.1 schema (operation_id `score_groundedness`)               |
| `GET /docs`                       | Swagger UI                                                           |

---

## 2. Your first request

The minimum payload is `response_text` plus one of the three premise
lanes. Here we use the `raw_context` lane — give it the whole
retrieval string and the engine will sentence-pack it into windows
that fit the encoder's context window.

```bash
curl -s -X POST http://localhost:8080/groundedness \
  -H "content-type: application/json" \
  -d '{
    "query_text": "When was Teardrops by George Harrison released?",
    "raw_context": "Teardrops is a single by George Harrison, released on 20 July 1981 in the United States. The track reached number eight on the Billboard Hot 100 chart.",
    "response_text": "Teardrops, a single by George Harrison, was released on 20 July 1981 in the US and reached number eight on the Billboard Hot 100 chart."
  }' | jq '.scores, .risk_band'
```

Sample output:

```json
{
  "groundedness_v2": 0.91,
  "reverse_context_calibrated": 0.88,
  "context_coverage_ratio": 1.0,
  "context_coverage_threshold": 0.5,
  "support_units_used": 2,
  "support_units_total": 2,
  "context_attribution_ratio": 1.0,
  "context_attribution_used_count": 2
}
"green"
```

What you're seeing:

- `groundedness_v2` ∈ [0, 1] — the calibrated headline score (fused
  reverse-context + NLI when NLI is enabled).
- `risk_band ∈ {green, amber, red, unknown}` — calibrated decision band
  using `thresholds.json` per stratum.
- `context_coverage_ratio = 1.0` — every retrieved chunk was used by
  the response. (More on this in §5.)

---

## 3. Pick a premise lane

`latence-trace` supports three premise lanes; pick whichever your
retrieval pipeline already produces.

### 3a. `raw_context` — give us the text

Easiest path. The engine sentence-packs the string into windows of
`raw_context_chunk_tokens` (default `256`) and concurrently scores
each window.

```json
{
  "raw_context": "<your retrieved text concatenated>",
  "raw_context_chunk_tokens": 256,
  "response_text": "<LLM output>"
}
```

### 3b. `chunk_ids` — caller resolves chunk text

Use this when your retriever stores chunks in a vector DB and you
want the service to dereference IDs through your `chunk_resolver`.

```json
{
  "chunk_ids": ["doc-42:chunk-7", "doc-42:chunk-8"],
  "response_text": "<LLM output>"
}
```

Wire your resolver when constructing the service:

```python
from latence_trace.api.service import GroundednessService

def my_resolver(ids):
    return [{"text": fetch(id), "metadata": {"chunk_id": id}} for id in ids]

service = GroundednessService(chunk_resolver=my_resolver)
```

### 3c. `support_units` — structured premises (recommended for production)

Use this when you want **per-unit attribution metadata** to flow back
into the response (source URL, speaker, timestamp, custom metadata).

```json
{
  "support_units": [
    {
      "text": "Teardrops is a single by George Harrison, released on 20 July 1981.",
      "source_id": "wikipedia:Teardrops_(George_Harrison_song)",
      "metadata": {"section": "Release"}
    },
    {
      "text": "The track reached number eight on the Billboard Hot 100 chart.",
      "source_id": "billboard:archive:1981-08",
      "metadata": {"chart": "Hot 100"}
    }
  ],
  "response_text": "Teardrops by George Harrison reached number 8 on the Billboard Hot 100 in 1981."
}
```

Each `support_units[i]` in the response will carry your `source_id`
and `metadata` along with the new `coverage_score` / `used` flags so
your UI can decorate the per-chunk citation with both attribution
*and* observability signals.

---

## 4. Read the response

Top-level fields you'll care about:

```json
{
  "scores": { ... },
  "risk_band": "green",
  "support_units": [ ... ],
  "response_tokens": [ ... ],
  "claims": [ ... ],
  "evidence": [ ... ]
}
```

### 4a. `response_tokens` — per-token heatmap

Every response token is returned with `char_start` / `char_end`,
`groundedness`, `prompt_echo`, and (when NLI is on) `nli_score`. Use
the character offsets to highlight the original `response_text` —
no need to re-tokenise on the client.

```json
{
  "text": "Teardrops",
  "char_start": 0,
  "char_end": 9,
  "groundedness": 0.94,
  "prompt_echo": 0.12,
  "nli_score": 0.91,
  "best_support_index": 0,
  "best_support_token": 5
}
```

`best_support_index` points at the support unit that best supports
this token (argmax). Use this to draw provenance arrows from token to
chunk in your UI.

### 4b. `support_units[]` — per-unit attribution and coverage

```json
{
  "support_id": "wikipedia:Teardrops_(George_Harrison_song)",
  "score": 0.92,
  "matched_response_tokens": 14,
  "coverage_score": 0.94,
  "used": true,
  "token_scores": [ ... ],
  "metadata": { "section": "Release" }
}
```

| Field                      | Meaning                                                                                                                |
|----------------------------|------------------------------------------------------------------------------------------------------------------------|
| `score`                    | Per-unit aggregate score (token-weight-aware).                                                                         |
| `matched_response_tokens`  | How many response tokens chose this unit as the **argmax** support (competitive signal).                              |
| `coverage_score`           | Max similarity any response token had to this unit's tokens (absolute strength, threshold-free).                      |
| `used`                     | `True` when `coverage_score >= scores.context_coverage_threshold` (default 0.5).                                      |

### 4c. `claims[]` — NLI verdicts

When NLI is enabled, every salient response claim gets `entailed` /
`neutral` / `contradicted` plus the support unit it was tested
against. Use this for human review queues.

### 4d. `risk_band`

Calibrated `green` / `amber` / `red` / `unknown`. Wire it into your
gating logic — for example, return the LLM output to the user only
if `risk_band == "green"`, otherwise route to a fallback.

---

## 5. Retrieval-efficiency observability (context coverage)

This is the section that turns scoring into a **feedback channel for
your retriever**. Two scalars and one per-unit boolean:

```json
"scores": {
  "context_coverage_ratio": 0.40,
  "context_coverage_threshold": 0.5,
  "support_units_used": 2,
  "support_units_total": 5,
  "context_attribution_ratio": 0.40,
  "context_attribution_used_count": 2
}
```

- **`context_coverage_ratio = 0.4`** → 60% of the chunks the
  retriever pulled were dead weight (their strongest match against
  the response was below threshold). Your retriever fetched too much
  irrelevant context for this query.
- **`context_attribution_ratio = 0.4`** → only 2 of 5 chunks won the
  argmax for at least one response token. Independent signal: a chunk
  can have high `coverage_score` (strong match somewhere) yet lose
  every argmax to a sibling chunk.

Coverage and attribution are **independent observability axes**. Pick
the one that maps to your tuning question:

| Question                                                     | Use                                                       |
|--------------------------------------------------------------|-----------------------------------------------------------|
| "Is the retriever pulling dead weight?"                      | `context_coverage_ratio`                                  |
| "Are sibling chunks crowding each other out for the same span?" | Find units with `matched_response_tokens > 0` but `used == False` |
| "Which retrieved chunks contributed nothing?"                | Filter `support_units` where `used == False`              |

### 5a. Tuning the threshold

The per-unit `coverage_score` is **independent of `coverage_threshold`** —
only the `used` flag and the global `coverage_ratio` move.

```json
{
  "raw_context": "...",
  "response_text": "...",
  "coverage_threshold": 0.65
}
```

- Lower threshold (0.3-0.4): tolerant; flags only the very weakest
  matches as dead weight. Useful for noisy multilingual encoders.
- Default (0.5): conservative cutoff for ColBERT-style normalised
  embeddings.
- Higher threshold (0.6-0.7): stricter; surfaces marginal matches as
  dead weight. Useful when your retriever pulls many candidates.

### 5b. Worked example: a noisy retriever

Here's the empirical signal on real HaluEval-QA data with a
deliberately noisy retriever (1 relevant chunk + 4 distractors):

| Scenario                                       | `coverage_ratio` (median) | `support_units_used` (median) |
|------------------------------------------------|---------------------------|-------------------------------|
| Grounded response, 1 retrieved chunk           | 1.000                     | 1 / 1                         |
| Hallucinated response, 1 retrieved chunk       | 1.000                     | 1 / 1                         |
| Grounded response, 1 relevant + 4 distractors  | **0.117**                 | **1.5 / 12**                  |

The signal collapses from 1.0 to ~0.12 when the retriever over-fetches
— exactly the diagnostic we want. Reproduce with:

```bash
python scripts/bench_context_coverage.py --samples 200
```

### 5c. Cost

`compute_unit_coverage` is one `max(dim=0)` reduction over the
`(R, U)` per-unit similarity matrix. CPU microbench:

| `R` × `U`        | median (µs) | p95 (µs) |
|------------------|-------------|----------|
| 64 × 8           | 77          | 91       |
| 1024 × 64        | 83          | 100      |
| 4096 × 256       | 273         | 359      |

Below 0.5 ms even at 4096 response tokens × 256 support units —
statistically free against the per-request encoder/scorer baseline.

---

## 6. Profiles and language

### 6a. Profiles

Three Pareto-optimal presets configure encoder size, NLI lane, fusion
weights, and chunking budgets:

| Profile     | When to use                                              |
|-------------|----------------------------------------------------------|
| `fast`      | Highest QPS, no NLI; "is this output even close?" gating |
| `balanced`  | Default. NLI on, conservative fusion. Good single-GPU box |
| `quality`   | Tightest fusion, NLI + reranker on. For human-in-the-loop |

Set per process:

```bash
LATENCE_TRACE_PROFILE=quality latence-trace serve
```

Or per request: pass `profile` in the JSON body to override.

### 6b. Language (EN / DE)

`latence-trace` is multilingual out of the box. The default
ColBERT/NLI lane is EN-only stub for tests, but in production set:

```bash
# ColBERT (multilingual; bf16)
export VOYAGER_GROUNDEDNESS_MODEL=VAGOsolutions/SauerkrautLM-Multi-Reason-ModernColBERT
export VOYAGER_GROUNDEDNESS_TORCH_DTYPE=bf16

# NLI (multilingual XNLI)
export VOYAGER_GROUNDEDNESS_NLI_MODEL=MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7
```

That's the only difference between an EN-only and an EN+DE deployment.
See `guides/multilingual.md` for the full env table and benchmark
numbers.

---

## 7. Integrations

### 7a. Python SDK (in-process)

```python
from latence_trace.api.models import GroundednessRequest
from latence_trace.api.service import GroundednessService

service = GroundednessService()
result = service.groundedness(GroundednessRequest(
    raw_context="...",
    response_text="...",
    query_text="...",
    coverage_threshold=0.5,
))
print(result.scores.context_coverage_ratio)
print(result.risk_band)
```

### 7b. HTTP / curl

See §2.

### 7c. AI agents (MCP)

```bash
latence-trace mcp-server
```

The stdio MCP loop registers `score_groundedness` as a tool with the
full input schema (including `coverage_threshold`). Cursor, Claude
Desktop, and other MCP-aware agents will discover it automatically.

### 7d. AI agents (OpenAI tools)

```python
import openai

tools = [{
    "type": "function",
    "function": {
        "name": "score_groundedness",
        "description": "...",
        "parameters": {
            "type": "object",
            "properties": {
                "response_text": {"type": "string"},
                "raw_context":   {"type": "string"},
                "query_text":    {"type": "string"},
                "coverage_threshold": {"type": "number", "default": 0.5},
            },
            "required": ["response_text"],
        },
    },
}]
```

When the agent calls the tool, post the arguments to your
`/groundedness` endpoint and return the response.

---

## 8. Production checklist

- [ ] `LATENCE_TRACE_PROFILE` set (default `balanced`).
- [ ] Multilingual encoders pinned (`VOYAGER_GROUNDEDNESS_MODEL`,
      `VOYAGER_GROUNDEDNESS_NLI_MODEL`) if you serve any non-English
      traffic.
- [ ] **English-only deployment?** Switch the NLI peer to
      `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`
      (`VOYAGER_GROUNDEDNESS_NLI_MODEL=...`) for **+10 percentage
      points paired ranking accuracy on HaluEval QA / Summ** vs. the
      multilingual default. Keep the multilingual mDeBERTa for any
      DE / multilingual traffic.
- [ ] `VOYAGER_GROUNDEDNESS_VLLM_ENDPOINT` + `..._VLLM_MODEL` set when
      serving the encoder via vLLM-Factory (BYOP) for max QPS.
- [ ] `LATENCE_TRACE_MAX_INFLIGHT` tuned to GPU budget (default 32;
      see `guides/concurrency.md`).
- [ ] Triton kernels eagerly warmed at startup
      (`latence-trace warm --profile <yours>`).
- [ ] Calibrated thresholds re-fit on your domain
      (`latence-trace calibrate --help`).
- [ ] Coverage threshold tuned per request stratum if your retriever
      varies in over-fetch behaviour (default `0.5`).
- [ ] **Workload boundary acknowledged.** `latence-trace` measures
      whether a response is anchored in the *supplied context*
      (RAG-grounding). It does **not** verify open-domain factuality
      against world knowledge. Pair with a knowledge-base fact-checker
      for workloads like dialogue continuations that introduce new
      real-world facts (HaluEval Dialogue stratum) or open-domain
      biographies (FActScore-style claims). See
      [`docs/algorithm-audit.md`](../algorithm-audit.md) §"Scope and
      Known Mismatches".

---

## 9. What's next

- `guides/profiles.md` — full grid of profile env keys.
- `guides/multilingual.md` — EN+DE benchmark numbers.
- `guides/concurrency.md` — the inflight semaphore and load-test
  budget.
- `algorithm-audit.md` — exact formulas, partition-invariance proofs,
  and per-channel ablation evidence.
- `benchmarks.md` — published numbers on RAGTruth, HaluEval, and
  FActScore.
