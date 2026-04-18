# API Reference

`POST /groundedness` is the single scoring endpoint. The full schema is
auto-published at `GET /openapi.json` (Swagger at `GET /docs`); this
page is the human-readable companion.

For the end-to-end walkthrough (boot → request → response → tuning),
read [`guides/tutorial.md`](guides/tutorial.md) first.

---

## Endpoint summary

| Method | Path                       | operation_id            |
|--------|----------------------------|-------------------------|
| POST   | `/groundedness`            | `score_groundedness`    |
| GET    | `/agent-help`              | `agent_help`            |
| GET    | `/.well-known/ai-plugin.json` | `ai_plugin_descriptor` |
| GET    | `/openapi.json`            | OpenAPI 3.1 schema      |
| GET    | `/docs`                    | Swagger UI              |
| GET    | `/healthz`                 | Liveness                |
| GET    | `/readyz`                  | Readiness + warmup state|

---

## Request: `GroundednessRequest`

Top-level fields. `response_text` is required; one of the three
premise lanes (`raw_context`, `chunk_ids`, `support_units`) is
required unless you set `attribution_mode = "open_domain"`.

| Field                       | Type             | Default        | Description                                                                                                  |
|-----------------------------|------------------|----------------|--------------------------------------------------------------------------------------------------------------|
| `response_text`             | `string`         | —              | The LLM-generated text to score. Required.                                                                   |
| `query_text`                | `string`         | `null`         | The user query that produced the response. Required when `primary_metric = "triangular"`.                    |
| `raw_context`               | `string`         | `null`         | Free-form premise text. Sentence-packed into windows of `raw_context_chunk_tokens`.                          |
| `chunk_ids`                 | `string[]`       | `[]`           | Stable chunk IDs resolved by your `chunk_resolver`.                                                          |
| `support_units`             | `SupportUnit[]`  | `[]`           | Structured premises with per-unit attribution metadata. See below.                                           |
| `raw_context_chunk_tokens`  | `int`            | `256`          | Token budget per `raw_context` window. The engine packs sentences ≤ this budget per window.                  |
| `response_chunk_tokens`     | `int`            | `256`          | Token budget per response window for long responses.                                                         |
| `primary_metric`            | `enum`           | `reverse_context` | One of `reverse_context`, `triangular`. `triangular` requires `query_text`.                              |
| `attribution_mode`          | `enum`           | `closed_book`  | `closed_book` refuses zero-evidence inputs (returns `risk_band = "unknown"`). `open_domain` is reserved for the post-v1 retrieval-callback lane. |
| `evidence_limit`            | `int`            | `8`            | Cap on returned `evidence[]` entries.                                                                        |
| `coverage_threshold`        | `float [0, 1]`   | `0.5`          | Threshold for the per-unit `used` flag and the global `context_coverage_ratio`. See §"Coverage" below.       |
| `debug_dense_matrices`      | `bool`           | `false`        | When `true`, returns the full per-unit similarity matrix in `debug.dense_matrices`. Heavy; off in production. |
| `profile`                   | `enum`           | env default    | `fast` / `balanced` / `quality`. Overrides `LATENCE_TRACE_PROFILE` for one request.                          |

### `SupportUnit`

Structured premise unit (the recommended lane for production).

| Field        | Type                | Description                                                                       |
|--------------|---------------------|-----------------------------------------------------------------------------------|
| `text`       | `string`            | The premise text. Required.                                                        |
| `source_id`  | `string`            | Stable identifier for the upstream document/chunk. Echoed back in the response.    |
| `speaker`    | `string`            | Optional speaker label (for transcripts).                                          |
| `timestamp`  | `string` (ISO-8601) | Optional timestamp (for transcripts).                                             |
| `metadata`   | `object`            | Free-form metadata; echoed back into the response unchanged.                       |

---

## Response: `GroundednessResponse`

```json
{
  "scores":  { ... },
  "risk_band": "green",
  "support_units":   [ ... ],
  "response_tokens": [ ... ],
  "claims":          [ ... ],
  "evidence":        [ ... ],
  "warnings":        [ ... ],
  "debug":           { ... }
}
```

### `scores: GroundednessScores`

Headline scalars. All scoring fields are in `[0, 1]`. Coverage and
attribution fields are `null` when scoring was refused
(`risk_band = "unknown"`).

| Field                              | Type    | Description                                                                                                                                     |
|------------------------------------|---------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| `groundedness_v2`                  | `float` | Calibrated headline score (fused reverse-context + NLI when NLI is enabled).                                                                    |
| `reverse_context_calibrated`       | `float` | Reverse-MaxSim score, null-bank-calibrated.                                                                                                     |
| `triangular`                       | `float` | (Optional) triangular MaxSim, present when `primary_metric = "triangular"`.                                                                     |
| `prompt_echo`                      | `float` | Mean prompt-echo signal across response tokens.                                                                                                 |
| `nli_entailment_ratio`             | `float` | Fraction of NLI-tested claims that entailed.                                                                                                    |
| **`context_coverage_ratio`**       | `float` | Retrieval-efficiency observability. Fraction of fetched units whose `coverage_score >= context_coverage_threshold`. **`0.4` ⇒ 60% dead weight.** |
| **`context_coverage_threshold`**   | `float` | Threshold echoed from the request. Default `0.5`.                                                                                               |
| **`support_units_used`**           | `int`   | Numerator of `context_coverage_ratio`.                                                                                                          |
| **`support_units_total`**          | `int`   | Denominator of `context_coverage_ratio` (= number of distinct support units after dedup).                                                       |
| **`context_attribution_ratio`**    | `float` | Threshold-free competitive signal. Fraction of units that won the argmax for at least one response token.                                       |
| **`context_attribution_used_count`** | `int` | Numerator of `context_attribution_ratio`.                                                                                                       |

### `risk_band`

`"green" | "amber" | "red" | "unknown"`. Calibrated decision band; use
`"green"` as the gate for fast-path serving, route the others to
human review or fallback prompts.

### `support_units[i]: GroundednessSupportUnit`

| Field                        | Type    | Description                                                                                                                                          |
|------------------------------|---------|------------------------------------------------------------------------------------------------------------------------------------------------------|
| `support_id`                 | `string`| The `source_id` you provided (or auto-assigned for `raw_context` / `chunk_ids` lanes).                                                               |
| `index`                      | `int`   | Global index across all support batches (stable across response chunks).                                                                             |
| `text`                       | `string`| The unit text.                                                                                                                                       |
| `score`                      | `float` | Per-unit aggregate score, token-weight-aware.                                                                                                        |
| `matched_response_tokens`    | `int`   | Number of response tokens that chose this unit as the **argmax** support (competitive signal).                                                       |
| **`coverage_score`**         | `float` | Max similarity any response token had to this unit's tokens. Independent of argmax and of threshold. Range `[0, 1]`.                                 |
| **`used`**                   | `bool`  | `True` when `coverage_score >= scores.context_coverage_threshold`. Filter `used == False` to surface dead-weight chunks.                            |
| `token_scores`               | `float[]`| Per-token max-similarity profile of this unit; used to highlight the most cited spans inside the unit.                                              |
| `metadata`                   | `object`| Echoed from the request `support_units[i].metadata`.                                                                                                 |

### `response_tokens[i]: GroundednessResponseToken`

Per-response-token heatmap data. Use the character offsets to highlight
the original `response_text` directly without re-tokenising.

| Field                  | Type    | Description                                                                                          |
|------------------------|---------|------------------------------------------------------------------------------------------------------|
| `text`                 | `string`| The token text.                                                                                       |
| `char_start` / `char_end` | `int` | Character offsets in `response_text`. Stable across chunked / unchunked paths.                       |
| `groundedness`         | `float` | Per-token reverse-context score.                                                                      |
| `prompt_echo`          | `float` | Per-token prompt-echo score.                                                                          |
| `nli_score`            | `float?` | Per-token NLI score (when NLI is enabled). `null` for tokens outside any tested claim span.          |
| `best_support_index`   | `int`   | Which support unit best supports this token (argmax).                                                |
| `best_support_token`   | `int`   | Token index inside that support unit. Use this to draw provenance arrows.                            |

### `claims[i]: GroundednessClaim`

Present when NLI is enabled.

| Field        | Type    | Description                                                |
|--------------|---------|------------------------------------------------------------|
| `text`       | `string`| The extracted claim.                                        |
| `verdict`    | `enum`  | `entailed` / `neutral` / `contradicted`.                    |
| `score`      | `float` | NLI confidence in the verdict.                              |
| `support_id` | `string`| Which support unit was tested against.                      |

### `warnings[]`

Free-form structured strings the engine emits when something
non-fatal happens (e.g. `"truncated_response_to_max_tokens"`,
`"no_premise_supplied"`). Always check this in production.

---

## Error envelope

All 4xx and 5xx responses share one structured shape:

```json
{
  "detail": {
    "code": "validation_error",
    "message": "human-readable description",
    "hint":    "actionable guidance for the caller",
    "docs_url":"deep link to the relevant doc page"
  }
}
```

Common codes: `validation_error`, `provider_unavailable`,
`request_too_large`, `rate_limited`, `internal_error`. The `code`
strings are stable across releases — agents and SDKs can dispatch on
them.

---

## Coverage in detail

The retrieval-efficiency observability surface in one paragraph:

> Every response carries `scores.context_coverage_ratio` (fraction of
> retrieved units whose strongest token-match cleared
> `coverage_threshold`) and per-unit `support_units[i].coverage_score`
> + `used`. Coverage is the **absolute-strength** signal —
> threshold-aware, independent of competition. Attribution
> (`scores.context_attribution_ratio`,
> `support_units[i].matched_response_tokens`) is the
> **competitive-placement** signal — threshold-free, depends on
> sibling units. Neither dominates the other; pick the one that maps
> to the question you're tuning. See §5 of the
> [tutorial](guides/tutorial.md) for worked examples and
> §"Per-support-unit context coverage" of
> [`algorithm-audit.md`](algorithm-audit.md) for the partition-
> invariance proofs.

---

## Discovery surfaces (live)

Every running server self-describes:

```bash
curl -s http://<host>/agent-help | jq .observability.context_coverage
# {
#   "request_field": "coverage_threshold",
#   "default": 0.5,
#   "range": [0.0, 1.0],
#   "response_globals": [
#     "scores.context_coverage_ratio",
#     "scores.context_coverage_threshold",
#     "scores.support_units_used",
#     "scores.support_units_total",
#     "scores.context_attribution_ratio",
#     "scores.context_attribution_used_count"
#   ],
#   "response_per_unit": [
#     "support_units[*].coverage_score",
#     "support_units[*].used"
#   ],
#   "description": "Retrieval-efficiency observability ..."
# }
```

The MCP adapter (`latence-trace mcp-server`) advertises the same
fields via `tools/list` so AI agents discover them automatically.
