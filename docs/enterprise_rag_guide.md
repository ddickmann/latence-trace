# Enterprise RAG integration guide

This is the copy-paste integration recipe for teams running
retrieval-augmented LLM apps in production. The RAG lane is the
default behaviour of `latence-trace` — no flags, no new fields.

## When to use the RAG lane

- Your context is **retrieved documents / passages / chunks** served
  to a generator LLM.
- You want a calibrated `green` / `amber` / `red` risk band per
  response, per-claim NLI evidence, and a retrieval-efficiency signal
  (`context_coverage_ratio`).
- You ship regulated / enterprise workloads and need an auditable
  answer trail.

If your context is opened source files and your output contains code
fences, see [coding_agent_guide.md](coding_agent_guide.md) instead.

## Minimum viable request

```bash
curl -X POST http://127.0.0.1:8090/groundedness \
  -H "Content-Type: application/json" \
  -d '{
    "raw_context": "Paris is the capital of France.",
    "query_text": "What is the capital of France?",
    "response_text": "The capital of France is Paris."
  }'
```

`scoring_mode` defaults to `"rag"`, so no explicit field is needed.
Existing integrations are bitwise-compatible — the RAG parity test
(`tests/api/test_rag_lane_parity.py`) gates every PR on that.

## Recommended request shape (production)

```jsonc
{
  "raw_context": "<retrieved passages concatenated, or see support_units below>",
  "query_text": "<original user question>",
  "response_text": "<LLM answer>",

  // Optional — supply pre-chunked units when your retriever already
  // returns document chunks. Each unit is encoded + scored + attributed
  // independently; coverage and attribution ratios are derived from
  // this set.
  "support_units": [
    { "text": "passage 1 ...", "source_id": "doc-42#chunk-3" },
    { "text": "passage 2 ...", "source_id": "doc-17#chunk-9" }
  ],

  // Optional — override the coverage threshold used to flag a unit
  // as "used". Defaults to 0.5. Tune this against your retriever's
  // empirical coverage distribution.
  "coverage_threshold": 0.5,

  // Optional — request atomic-claim decomposition with per-claim NLI.
  // Strongly recommended for long-form QA and summarization.
  "enable_atomic_claims": true,

  // Optional — request semantic entropy (needs verification_samples).
  "verification_samples": ["sample 1", "sample 2"]
}
```

## What you get back

- `scores.groundedness_v2` — the calibrated headline (0 … 1).
- `risk_band` — one of `green` / `amber` / `red` from the per-stratum
  thresholds artefact.
- `scores.context_coverage_ratio` — the fraction of support units that
  crossed `coverage_threshold`. **Your retrieval-efficiency signal.**
- `scores.context_attribution_ratio` — the fraction of support units
  that won the argmax for at least one response token.
- `support_units[i].coverage_score`, `.used`, `.matched_response_tokens`
  — per-unit rollups for dashboards.
- `literal_diagnostics` — matched / missed dates / numbers / units /
  currencies / URLs / IDs.
- `structured_diagnostics` — typed-cell alignment (JSON / markdown
  tables).
- `nli_diagnostics` — per-atomic-claim entailment / contradiction.
- `per_token_scores` — heatmap at the response-token level.

Full schema: `curl http://<host>/openapi.json | jq`.

## Profiles

Three calibrated profiles ship out of the box:

| Profile | p95 | Use when |
| --- | --- | --- |
| `fast` | ~160 ms | Sub-200 ms SLO; cheap liveness check. |
| `balanced` *(default)* | ~190 ms | Typical RAG QA serving. |
| `quality` | ~195 ms | Long multi-premise contexts; structured tables. |

```bash
latence-trace-server --profile quality
# or
LATENCE_TRACE_PROFILE=quality latence-trace-server
```

See [guides/profiles.md](guides/profiles.md) for the full config
dump per profile.

## Python SDK (sync)

```python
from latence import Latence

client = Latence(base_url="http://latence-trace.internal:8090")
result = client.grounding.rag(
    raw_context=retrieved_passages,
    query=user_question,
    response_text=llm_answer,
    extra={"enable_atomic_claims": True},
)

if result.risk_band == "red":
    log.warning("ungrounded response", extra={
        "groundedness": result.scores.groundedness_v2,
        "coverage": result.scores.context_coverage_ratio,
        "source_ids": [u.source_id for u in result.support_units if u.used],
    })
```

## Python SDK (async)

```python
from latence import AsyncLatence

async with AsyncLatence(base_url="http://latence-trace.internal:8090") as client:
    result = await client.grounding.rag(
        raw_context=retrieved_passages,
        query=user_question,
        response_text=llm_answer,
    )
```

## LangChain / LlamaIndex adapters

See the sibling `latence-trace-python` SDK repository for drop-in callbacks
that attach groundedness scoring to your existing chain / query-engine.

## Retrieval observability loop

Use the per-unit `used` flag to drive retrieval tuning:

```python
if result.scores.context_coverage_ratio < 0.5:
    # Over 50% of what the retriever returned was dead weight.
    # Candidates for eviction / top-k reduction:
    dead = [u.source_id for u in result.support_units if not u.used]
    retrieval_metrics.dead_chunks.labels(retriever="hnsw").inc(len(dead))
```

Combine with `context_attribution_ratio` to spot sibling crowding
(chunks with `matched_response_tokens > 0` but `used == False`): good
signal for chunk-level dedup.

## Threshold calibration

Per-tenant recalibration is a first-class operation. See
[`operations/calibration-runbook.md`](operations/calibration-runbook.md)
for the playbook; it fits new `thresholds.<tenant>.json` and
`fusion_weights.<tenant>.json` artefacts against a tenant-specific
labeled set and hot-swaps them without a restart.

## Also see

- [`docs/algorithm-audit.md`](algorithm-audit.md) — per-stratum
  performance, known mismatches, and reproduction instructions.
- [`docs/benchmarks.md`](benchmarks.md) — reproducing the headline
  numbers.
- [`docs/operations/deployment-checklist.md`](operations/deployment-checklist.md)
  — Helm / Kubernetes rollout checklist.
