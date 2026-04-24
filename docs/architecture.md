# latence-trace — architecture (two-lane)

`latence-trace` exposes a single `/groundedness` endpoint that
dispatches to one of two lanes based on the `scoring_mode` field on
the request body. Both lanes share the heavy infrastructure (GPU
encoder, NLI provider, observability, licensing, rate-limit,
authentication), and each lane owns only the domain-specific signals
it needs.

```text
                        ┌────────────────────────────────────────────┐
                        │              FastAPI front-door            │
                        │  auth │ rate-limit │ request-id │ tracing  │
                        └──────────────┬─────────────────────────────┘
                                       │ POST /groundedness
                                       ▼
                         ┌─────────────────────────────┐
                         │   GroundednessService       │
                         │   scoring_mode dispatcher   │
                         └──────┬───────────────┬──────┘
                scoring_mode=   │               │ scoring_mode=
                    "rag"       │               │     "code"
                                ▼               ▼
                       ┌────────────────┐  ┌─────────────────────────────┐
                       │  RAG scorer    │  │  Code-lane orchestrator     │
                       │  (unchanged)   │  │  GPUScorer  + AST + novelty │
                       │ MaxSim, NLI,   │  │  LogisticComposite          │
                       │ atomic, struct │  │  NLICascade (ambiguity-only)│
                       │ semantic-ent.  │  │  FileAttribution / reasons  │
                       └──────┬─────────┘  └──────────────┬──────────────┘
                              │                           │
                              ▼                           ▼
                       ┌──────────────────────────────────────┐
                       │         Shared infrastructure        │
                       │   ColBERT encoder (bf16, warm)       │
                       │   vLLM-factory NLI (async, pooled)   │
                       │   Triton MaxSim kernel               │
                       │   tree-sitter grammars (Py/TS/JS/…) │
                       │   Prometheus + OTel + JSON logs      │
                       └──────────────────────────────────────┘
                                       │
                                       ▼
                       ┌──────────────────────────────────────┐
                       │           Responses                  │
                       │  RAG:  scores, risk_band, evidence,  │
                       │        per-unit coverage             │
                       │  Code: scores + code_lane_diagnostics│
                       │        (AST drift, cascade, owners)  │
                       └──────────────────────────────────────┘
```

## Per-lane concurrency

`runpod/handler.py` holds **two independent `asyncio.Semaphore`
pools** — one for RAG, one for code — so a burst of code-lane
requests cannot starve the RAG lane (and vice-versa). Budgets are
configurable via `LATENCE_TRACE_RAG_CONCURRENCY` and
`LATENCE_TRACE_CODE_CONCURRENCY`; both default to 32.

## Warmup and readiness

`kernels/warmup.warm_all()` warms the Triton MaxSim kernel.
`kernels/warmup.warm_code_lane()` additionally:

- instantiates the `GPUScorer` singleton and runs a zero-input pass,
- preloads tree-sitter grammars for Python, TypeScript, JavaScript,
  Go, and Rust (via `latence_trace.core.code_lane.SUPPORTED_LANGUAGES`),
- primes the NLI provider with an entailment check.

`/readyz` returns 200 only after **both** warmups complete. This is
how downstream orchestrators know it is safe to direct code-lane
traffic at a pod.

## Observability

Every request emits:

- a `groundedness_turn` structured log line with
  `lane`, `session_id`, `cascade_fired`, `nli_ms`, `ast_ms`,
  `composite_ms`, `total_ms` (no query/response text, no file content),
- four Prometheus counters
  (`latence_trace_lane_requests_total`,
   `latence_trace_cascade_fires_total`,
   `latence_trace_budget_exceeded_total`,
   `latence_trace_phantom_verdict_total`),
- an OpenTelemetry span with the same attributes.

See [`docs/code_lane_turn_event.md`](code_lane_turn_event.md) for the
full JSONL schema.

## Hot-path breakdown

| Step | RAG lane | Code lane |
| --- | --- | --- |
| Encode query + context + response | ColBERT bf16 | ColBERT bf16 (same) |
| Deterministic lexical / structural | Literal guardrails + structured-cell AND-gate | AST symbol extraction + literal novelty |
| Dense scoring | MaxSim + reverse MaxSim + coverage | `GPUScorer` (MaxSim + per-token p10 + owner) |
| NLI | Premise-reranked DeBERTa-MNLI on atomic claims | `NLICascade` — vLLM-factory NLI only when composite ∈ [0.65, 0.90] |
| Semantic entropy | Bidirectional NLI clustering (opt-in) | `compute_semantic_entropy` on caller-supplied samples |
| Final score | `groundedness_v2` (calibrated fusion) | `composite_logistic_v3` (LR with interaction terms) |

## Session state — caller-portable, server-stateless

Temporal signals (drift, EMA groundedness, dead-weight streaks,
recommendations) need memory somewhere, but the API stays stateless
for per-turn measurement. We split the concern:

- **Sensor layer (server)** — pure per-turn scoring, no conversation
  storage, no sticky routing.
- **Control layer (caller)** — a small opaque `session_state` blob
  that round-trips on the request/response. The server runs a pure
  deterministic transform
  (`latence_trace.core.code_lane.session.update_session_state`) and
  emits `next_session_state` + `session_signals` alongside the usual
  per-turn diagnostics.

Callers opt in via `session_id` (first turn) or `session_state`
(subsequent turns). See
[`docs/session_semantics.md`](session_semantics.md) for the full
protocol, derived signals, and recommendation policy. The TypeScript
helpers in [`plugin_client/`](../plugin_client) implement the same
algorithm locally for plugins that want fully-offline computation.

## Shared attribution kernel (lane-neutral)

Both lanes converge on a single per-file attribution kernel:
`latence_trace/core/attribution/file_attribution.py`. The code lane has
always used it; the RAG lane now projects its per-unit argmax evidence
through the same kernel so that every response — regardless of scoring
mode — emits:

- a top-level `file_attribution` payload
  (`per_file` + `per_unit` + `reason_code_histogram`),
- `dead_weight_ratio` and `dead_weight_file_count` on the scores dict,
- the same `ReasonCode` enum values
  (`never_won_argmax`, `all_tokens_below_0_40`,
  `dominated_by_single_file`).

File grouping follows the same precedence rule on both lanes:
`metadata.path` → `metadata.source` → `metadata.source_id` →
`support_id`. This means an IDE plugin that already renders code-lane
attribution needs no extra branching for the RAG lane — the payload
shape is identical.

```text
          per-unit argmax records
          (RAG: MaxSim + tri-state usage,
           Code: GPUScorer + owner / query-owner)
                     │
                     ▼
     resolve_attribution_key(support_unit) ── metadata.path / source / id
                     │
                     ▼
     attribute_files(...)   ⇒  FileAttributionResult
                                    ├── per_file
                                    ├── per_unit
                                    ├── reason_code_histogram  ◀─ new
                                    └── dead_weight_ratio / count
```

## Heatmap (convenience payload)

`latence_trace/api/heatmap.py` is a pure, dependency-free renderer that
projects the per-response scoring output into two surfaces:

- a structured `heatmap` data payload
  (`tokens` + `files` + `summary` + `thresholds`),
- an opt-in self-contained `heatmap_html` fragment — a single
  `<div class="lt-heatmap">` with an inline `<style>` block that scopes
  three band classes (`.lt-band-green` / `.lt-band-amber` /
  `.lt-band-red`). No external stylesheets, no JavaScript, no
  fonts fetched from the network.

Callers opt in via `heatmap_format: "none" | "data" | "html"` on
`GroundednessRequest`. Default is `"data"` — the structured payload
ships on every response with no wire cost unless the caller explicitly
opts out. The HTML fragment is only emitted for `"html"`. See
[`docs/heatmap.md`](heatmap.md) for the exact band thresholds and a
copy-pasteable integration snippet.

## Stateless rollup (session-level aggregation)

`action="rollup"` is a pure, CPU-only transform served by the same
handler. It takes a list of per-turn records (the compact outputs a
plugin already has from scoring) and returns session-level aggregates
(`noise_pct`, `model_drift_pct`, `retrieval_waste_pct`,
`reason_code_histogram`, `risk_band_trail`, `drift_trend`,
`top_dead_files`). No I/O, no model calls, no state — the endpoint is
safe to call on every keystroke if a plugin wants a live session
scoreboard. See [`docs/rollup.md`](rollup.md) for the full request /
response shape and an IDE-plugin integration snippet.

## Also see

- [`docs/code_lane_v3.md`](code_lane_v3.md) — full code-lane design doc.
- [`docs/heatmap.md`](heatmap.md) — band thresholds + HTML template.
- [`docs/rollup.md`](rollup.md) — rollup request / response shape +
  plugin integration snippet.
- [`docs/session_semantics.md`](session_semantics.md) — the
  caller-portable session-state protocol.
- [`docs/code_lane_performance.md`](code_lane_performance.md) —
  per-signal budgets and measured latencies.
- [`docs/code_lane_turn_event.md`](code_lane_turn_event.md) — the
  JSONL event spec for IDE dashboards.
- [`docs/code_lane_review_checklist.md`](code_lane_review_checklist.md)
  — required PR checklist for anything touching the code lane.
- [`docs/enterprise_rag_guide.md`](enterprise_rag_guide.md) —
  integration recipe for enterprise RAG pipelines.
- [`docs/coding_agent_guide.md`](coding_agent_guide.md) —
  integration recipes for IDE coding agents.
