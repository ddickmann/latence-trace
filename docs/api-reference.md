# API Reference

Latence TRACE exposes groundedness verification and compliance redaction
through one runtime. The full schema is auto-published at `GET /openapi.json`
(Swagger at `GET /docs`); this page is the human-readable companion.

For the end-to-end walkthrough (boot → request → response → tuning),
read [`guides/tutorial.md`](guides/tutorial.md) first.

---

## Endpoint summary

| Method | Path                       | operation_id            |
|--------|----------------------------|-------------------------|
| POST   | `/groundedness`            | `score_groundedness`    |
| POST   | `/v1/compliance/redact`    | `v1_compliance_redact`  |
| GET    | `/v1/compliance/schema`    | `v1_compliance_schema`  |
| GET    | `/v1/compliance/healthz`   | `v1_compliance_healthz` |
| GET    | `/agent-help`              | `agent_help`            |
| GET    | `/.well-known/ai-plugin.json` | `ai_plugin_descriptor` |
| GET    | `/openapi.json`            | OpenAPI 3.1 schema      |
| GET    | `/docs`                    | Swagger UI              |
| GET    | `/healthz`                 | Liveness                |
| GET    | `/readyz`                  | Readiness + warmup state|

---

## Compliance Redaction

`POST /v1/compliance/redact` detects GDPR/enterprise PII with
`knowledgator/gliner-pii-large-v1.0`, token-aware chunking, deterministic
sanity checks, optional custom regex overrides, and mask or replacement
redaction. The internal `/compliance/redact` route remains available for
self-hosted deployments, but hosted customers should use the `/v1` alias.

```json
{
  "text": "Jane Doe uses jane@example.com.",
  "mode": "category",
  "labels": ["person", "email"],
  "redact": true,
  "redaction_mode": "mask",
  "include_original_text": false
}
```

The response returns `entities`, `entity_count`, `unique_labels`,
`redacted_text`, `chunks_processed`, `labels_used`, `timings_ms`, and a
privacy-safe `usage` object. Do not log `text`, `entities[*].text`, or
`redacted_text` in customer-facing analytics; the portal insights lane uses
only aggregate usage metadata such as entity count, label count, chunk count,
redaction mode, and latency.

`GET /v1/compliance/schema` returns the supported GDPR categories, full label
catalog, optimized model-label aliases, and modes (`open` or `category`). Use
`open` for broad PII sweeps and `category`/`labels` when the user wants a
tighter allowlist.

Public requests and responses always use canonical GDPR labels such as
`person`, `email`, `phone_number`, and `credit_card_number`. Internally, TRACE
translates those labels to GLiNER-facing aliases discovered by native-model
benchmarking, for example `person -> name`, `social_security_number -> ssn`,
and `drivers_license -> driver's license number`. Predictions are mapped back
to canonical labels before validation, redaction, billing, and insights.

---

## Request: `GroundednessRequest`

Top-level fields. `response_text` is required. At most one premise lane
(`raw_context`, `chunk_ids`, `support_units`) may be supplied per request.
A request with zero premises is still valid: `closed_book` (default) short-
circuits to `risk_band = "unknown"` with `reason = "no_premise_supplied"`;
`open_domain` is reserved for the post-v1 retrieval-callback lane and
currently refuses with `risk_band = "unsupported"` and
`reason = "open_domain_pending_v1_next"` regardless of whether premises are
supplied.

| Field                            | Type             | Default             | Description                                                                                                                                              |
|----------------------------------|------------------|---------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| `response_text`                  | `string`         | —                   | The LLM-generated text to score. Required.                                                                                                                |
| `query_text`                     | `string`         | `null`              | The user query that produced the response. Required when `primary_metric = "triangular"`.                                                                 |
| `raw_context`                    | `string`         | `null`              | Free-form premise text. Sentence-packed into windows of `raw_context_chunk_tokens`.                                                                       |
| `chunk_ids`                      | `(string\|int)[]`| `null`              | Stable chunk IDs resolved by the configured `chunk_resolver`.                                                                                             |
| `support_units`                  | `SupportUnit[]`  | `null`              | Structured premises with per-unit attribution metadata. See below.                                                                                        |
| `raw_context_chunk_tokens`       | `int [1, 8192]`  | `256`               | Token budget per `raw_context` window.                                                                                                                    |
| `response_chunk_tokens`          | `int [1, 8192]`  | `256`               | Token budget per response window for long responses.                                                                                                      |
| `primary_metric`                 | `enum`           | `reverse_context`   | One of `reverse_context`, `triangular`. `triangular` requires `query_text`.                                                                               |
| `attribution_mode`               | `enum`           | `closed_book`       | `closed_book` refuses zero-premise requests with `risk_band = "unknown"` / `reason = "no_premise_supplied"`. `open_domain` is reserved for the post-v1 retrieval-callback lane and currently refuses with `risk_band = "unsupported"` / `reason = "open_domain_pending_v1_next"`, regardless of whether premises are supplied. |
| `segmentation_mode`              | `enum`           | `sentence_packed`   | How `raw_context` is segmented into support units.                                                                                                        |
| `coverage_threshold`             | `float [0, 1]`   | `0.5`               | Threshold for the per-unit legacy `used` flag and the global `context_coverage_ratio`. See §"Coverage" below.                                             |
| `evidence_limit`                 | `int [1, 128]`   | `8`                 | Cap on returned `top_evidence[]` entries.                                                                                                                 |
| `include_triangular_diagnostics` | `bool`           | `true`              | When `query_text` is provided, include optional query-conditioned diagnostics (triangular, echo, grounded coverage).                                       |
| `debug_dense_matrices`           | `bool`           | `false`             | When `true`, returns the full per-unit similarity matrix in `debug`. Heavy; off in production.                                                            |
| `model`                          | `string`         | `null`              | Optional encoder override for response / query / raw-context encoding.                                                                                    |
| `query_prompt_name`              | `string`         | `null`              | Optional asymmetric prompt name for query encoding (e.g. `query`).                                                                                        |
| `document_prompt_name`           | `string`         | `null`              | Optional asymmetric prompt name for response / raw-context encoding (e.g. `document`).                                                                    |
| `verification_samples`           | `string[]`       | `null`              | Alternate responses for the semantic-entropy peer (temperature > 0 siblings).                                                                             |
| `content_type`                   | `string`         | `null`              | Structured-source hint (`application/json`, `text/markdown`, `application/json+schema`). Auto-detected when omitted.                                      |
| `risk_band_stratum`              | `string`         | `null`              | Optional failure-mode hint for the calibrated risk-band classifier (e.g. `entity_swap`, `negation`).                                                      |
| `runtime_head_features`          | `object`         | `null`              | Explicit feature map for promoted runtime heads. Required for feature-gated root-cause heads to emit autonomous decisions.                                 |
| `trajectory_features`            | `object`         | `null`              | Alias for `runtime_head_features` used by agentic coding clients that emit file/symbol/test/patch/order features.                                         |

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
  "collection":       "tutorial-li",
  "mode":             "chunk_ids",
  "model":            "lightonai/GTE-ModernColBERT-v1",
  "attribution_mode": "closed_book",
  "reason":           null,
  "scores":           { "risk_band": "green", "...": "see below" },
  "response_tokens":  [ ... ],
  "support_units":    [ ... ],
  "top_evidence":     [ ... ],
  "eligibility":      { ... },
  "query_tokens":     [ ... ],
  "nli_diagnostics":  { "aggregate_score": 0.92, "claims": [ ... ] },
  "literal_diagnostics":   { ... },
  "semantic_entropy_diagnostics": { ... },
  "structured_diagnostics":       { ... },
  "runtime_decision": { "action": "allow", "head_id": "trajectory_ranker", "...": "..." },
  "warnings":         [ ... ],
  "debug":            { ... },
  "time_ms":          4.2
}
```

### `runtime_decision`

When `LATENCE_TRACE_RUNTIME_DECISION_ENABLED=1`, responses include a stable
runtime decision record:

| Field | Type | Description |
|---|---|---|
| `action` | `"allow" | "auto_repair" | "block"` | Autonomous policy action. Missing required head features force `auto_repair`. |
| `score` | `float` | Decision score after any enabled head override. |
| `score_channel` | `string` | Source of `score`, e.g. `head:trajectory_ranker` or `groundedness_v2`. |
| `head_id` | `string?` | Runtime head selected for the routed class. |
| `head_enabled` | `bool?` | Whether the head was enabled and executable for this request. |
| `head_score` | `float?` | Score emitted by the head. `null` when disabled, missing, corrupt, or feature-gated. |
| `head_features_used` | `string[]` | Feature names or channels consumed by the head. |
| `head_reason_codes` | `string[]` | Machine-readable head status and fallback reasons. |

### `scores: GroundednessScores`

Headline scalars and observability fields. All scoring fields are in
`[0, 1]`. `scores.risk_band` carries the calibrated decision band (see
next subsection). Coverage / attribution / usage fields are `null` when
scoring was refused (`scores.risk_band = "unknown"` or `"unsupported"`).

| Field                                 | Type     | Description                                                                                                                                                                                                                                                                                                                            |
|---------------------------------------|----------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `primary_name`                        | `string` | Name of the headline metric (`reverse_context` or `triangular`).                                                                                                                                                                                                                                                                       |
| `primary_score`                       | `float`  | Value of the headline metric.                                                                                                                                                                                                                                                                                                          |
| `reverse_context`                     | `float`  | Raw reverse-MaxSim score.                                                                                                                                                                                                                                                                                                              |
| `reverse_context_calibrated`          | `float?` | Reverse-MaxSim null-bank-calibrated score, when a null bank is configured.                                                                                                                                                                                                                                                             |
| `groundedness_v2`                     | `float?` | Calibrated headline score fusing reverse-context, literal-guarded, NLI, semantic-entropy, and typed structured evidence.                                                                                                                                                                                                                |
| `consensus_hardened`                  | `float?` | Mean consensus-hardened reverse-context score.                                                                                                                                                                                                                                                                                         |
| `reverse_query_context`               | `float?` | Query-conditioned reverse-context score.                                                                                                                                                                                                                                                                                               |
| `triangular`                          | `float?` | Triangular MaxSim, present when `primary_metric = "triangular"` or `include_triangular_diagnostics = true`.                                                                                                                                                                                                                            |
| `echo_mean`                           | `float?` | Mean response-token echo signal.                                                                                                                                                                                                                                                                                                        |
| `grounded_coverage`                   | `float?` | Query-token grounded-coverage signal.                                                                                                                                                                                                                                                                                                   |
| `literal_guarded`                     | `float?` | Literal-guarded reverse-context score.                                                                                                                                                                                                                                                                                                  |
| `literal_mismatch_count`              | `int?`   | Number of response literals with no support match.                                                                                                                                                                                                                                                                                     |
| `literal_match_count`                 | `int?`   | Number of response literals matched against support.                                                                                                                                                                                                                                                                                   |
| `literal_total_count`                 | `int?`   | Total number of literals extracted from the response.                                                                                                                                                                                                                                                                                  |
| `nli_aggregate`                       | `float?` | Aggregate NLI entailment score across tested claims.                                                                                                                                                                                                                                                                                   |
| `nli_claim_count`                     | `int?`   | Number of claims the NLI peer tested.                                                                                                                                                                                                                                                                                                   |
| `nli_skipped_count`                   | `int?`   | Number of candidate claims the NLI peer skipped.                                                                                                                                                                                                                                                                                        |
| `semantic_entropy_aggregate`          | `float?` | Semantic-entropy peer aggregate.                                                                                                                                                                                                                                                                                                       |
| `semantic_entropy_raw`                | `float?` | Raw semantic-entropy value before fusion.                                                                                                                                                                                                                                                                                               |
| `semantic_entropy_sample_count`       | `int?`   | Number of verification samples used by the semantic-entropy peer.                                                                                                                                                                                                                                                                       |
| `null_bank_size`                      | `int?`   | Size of the calibration null bank used for `reverse_context_calibrated`.                                                                                                                                                                                                                                                                |
| `structured_source`                   | `float?` | Typed structured-evidence AND-gate (`min(entity_align, value_match, unit_match, sign_match)`). `null` when no typed source was detected. When non-null and the AND-gate is enabled, the headline becomes `min(narrative, structured)`.                                                                                                 |
| `structured_source_guarded`          | `float?` | Prose-triple guarded structured-source score.                                                                                                                                                                                                                                                                                           |
| `structured_source_detected`         | `bool?`  | Whether the structured-source adapter detected any typed source.                                                                                                                                                                                                                                                                        |
| `structured_source_typed_aligned`     | `int?`   | Number of typed response claims aligned to a typed source cell.                                                                                                                                                                                                                                                                        |
| `structured_source_typed_count`       | `int?`   | Number of typed claims extracted from `response_text`.                                                                                                                                                                                                                                                                                 |
| `risk_band`                           | `string?`| Calibrated decision band (see below).                                                                                                                                                                                                                                                                                                  |
| **`context_coverage_ratio`**          | `float?` | Retrieval-efficiency observability. Fraction of fetched units whose `coverage_score >= context_coverage_threshold`. **`0.4` ⇒ 60% dead weight.**                                                                                                                                                                                        |
| **`context_coverage_threshold`**      | `float?` | Threshold echoed from the request. Default `0.5`.                                                                                                                                                                                                                                                                                       |
| **`support_units_used`**              | `int?`   | Numerator of `context_coverage_ratio` (legacy coverage view).                                                                                                                                                                                                                                                                          |
| **`support_units_total`**             | `int?`   | Denominator of `context_coverage_ratio` (= number of distinct support units after dedup).                                                                                                                                                                                                                                              |
| **`context_attribution_ratio`**       | `float?` | Threshold-free competitive signal. Fraction of units that won the argmax for at least one response token.                                                                                                                                                                                                                              |
| **`context_attribution_used_count`**  | `int?`   | Numerator of `context_attribution_ratio`.                                                                                                                                                                                                                                                                                               |
| **`support_units_usage_used`**        | `int?`   | Count of support units whose tri-state `usage_state` is `used`. Separate from the legacy coverage-based `support_units_used`.                                                                                                                                                                                                          |
| **`support_units_unused`**            | `int?`   | Count of support units emitted as high-confidence `usage_state = "unused"`. Precision-first: borderline cases abstain instead of flipping here.                                                                                                                                                                                        |
| **`support_units_uncertain`**         | `int?`   | Count of support units emitted as `usage_state = "uncertain"`.                                                                                                                                                                                                                                                                           |
| **`context_usage_ratio`**             | `float?` | Fraction of support units whose tri-state `usage_state` is `used`.                                                                                                                                                                                                                                                                     |
| **`context_unused_ratio`**            | `float?` | Fraction of support units emitted as `usage_state = "unused"`.                                                                                                                                                                                                                                                                           |
| **`context_uncertain_ratio`**         | `float?` | Fraction of support units emitted as `usage_state = "uncertain"`.                                                                                                                                                                                                                                                                        |

### `scores.risk_band`

`"green" | "amber" | "red" | "unknown" | "unsupported"`. Use `"green"`
as the gate for fast-path serving; route the others to human review or
fallback prompts.

- `"green" | "amber" | "red"` are the calibrated decision bands for
  successfully scored requests.
- `"unknown"` is emitted when `attribution_mode = "closed_book"` and
  the request carries zero premises; the companion
  `GroundednessResponse.reason = "no_premise_supplied"` is the
  machine-readable key. `warnings[]` is free-form and must not be
  dispatch-keyed.
- `"unsupported"` is emitted when `attribution_mode = "open_domain"`,
  which is reserved for the post-v1 retrieval-callback lane; the
  companion `reason = "open_domain_pending_v1_next"`.

### `support_units[i]: GroundednessSupportUnit`

| Field                        | Type          | Description                                                                                                                                                               |
|------------------------------|---------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `index`                      | `int`         | Global index across all support batches (stable across response chunks).                                                                                                  |
| `support_id`                 | `string`      | The `source_id` you provided (or auto-assigned for `raw_context` / `chunk_ids` lanes).                                                                                    |
| `chunk_id`                   | `string\|int?`| Stable chunk id when the `chunk_ids` lane is used.                                                                                                                        |
| `source_mode`                | `string`      | Which request lane produced this unit: `chunk_ids`, `raw_context`, or `support_units`.                                                                                     |
| `text`                       | `string`      | The unit text.                                                                                                                                                            |
| `offset_start` / `offset_end`| `int?`        | Character offsets within the source document, when available.                                                                                                             |
| `token_count`                | `int`         | Number of tokens in this unit.                                                                                                                                            |
| `tokens`                     | `string[]`    | Token strings of this unit.                                                                                                                                               |
| `token_scores`               | `float[]`     | Per-token max-similarity profile of this unit; used to highlight the most cited spans inside the unit.                                                                     |
| `score`                      | `float`       | Per-unit aggregate score, token-weight-aware.                                                                                                                              |
| `matched_response_tokens`    | `int`         | Number of response tokens that chose this unit as the **argmax** support (competitive signal).                                                                             |
| **`coverage_score`**         | `float`       | Max similarity any response token had to this unit's tokens. Independent of argmax and of threshold. Range `[0, 1]`.                                                       |
| **`used`**                   | `bool`        | Legacy compatibility view: `True` when `coverage_score >= scores.context_coverage_threshold`. Use `usage_state` for the precision-first tri-state contract.               |
| **`usage_state`**            | `enum`        | Precision-first tri-state label: `used`, `unused`, or `uncertain`. `unused` is emitted only for high-confidence negatives.                                                 |
| `usage_confidence`           | `float?`      | Confidence in the emitted `usage_state` (`[0, 1]`).                                                                                                                        |
| `unused_confidence`          | `float?`      | Confidence that the unit is truly unused (`[0, 1]`). Helpful for sorting or filtering `usage_state = "unused"` units.                                                      |
| `source_id`                  | `string?`     | Echoed from the request `support_units[i].source_id` when supplied.                                                                                                        |
| `speaker`                    | `string?`     | Echoed from the request `support_units[i].speaker` when supplied.                                                                                                          |
| `timestamp`                  | `string?`     | Echoed from the request `support_units[i].timestamp` when supplied.                                                                                                        |
| `metadata`                   | `object?`     | Echoed verbatim from the request `support_units[i].metadata` when supplied.                                                                                                |

### `response_tokens[i]: GroundednessResponseToken`

Per-response-token heatmap data. Use the character offsets to highlight
the original `response_text` directly without re-tokenising.

| Field                            | Type           | Description                                                                                          |
|----------------------------------|----------------|------------------------------------------------------------------------------------------------------|
| `index`                          | `int`          | Zero-based index of the token in the aligned response token stream.                                  |
| `token`                          | `string`       | Token text.                                                                                           |
| `weight`                         | `float`        | Content-mask token weight (`0.0` for non-content tokens, `1.0` otherwise).                           |
| `reverse_context`                | `float`        | Per-token reverse-context MaxSim score.                                                               |
| `reverse_context_calibrated`     | `float?`       | Null-bank-calibrated per-token reverse-context score, when a null bank is configured.                 |
| `reverse_context_z`              | `float?`       | Z-score of the token against the null bank.                                                           |
| `null_mean` / `null_std`         | `float?`       | Null-bank mean / stddev for this token.                                                                |
| `nli_score`                      | `float?`       | Per-token NLI score (when NLI is enabled). `null` for tokens outside any tested claim span.           |
| `consensus_hardened`             | `float?`       | Consensus-hardened reverse-context score for this token.                                              |
| `support_unit_hits_above_threshold` | `int?`      | Count of support units scoring above the consensus threshold for this token.                          |
| `support_unit_soft_breadth`      | `float?`       | Soft count of supporting units above the consensus threshold.                                         |
| `effective_support_units`        | `float?`       | Effective supporting units for consensus hardening.                                                    |
| `reverse_query_context`          | `float?`       | Query-conditioned reverse-context score for this token.                                               |
| `triangular`                     | `float?`       | Per-token triangular score, when triangular diagnostics are active.                                   |
| `echo`                           | `float?`       | Per-token echo score.                                                                                 |
| `support_unit_index`             | `int?`         | Index of the argmax support unit for this token.                                                       |
| `support_token_index`            | `int?`         | Token index inside that support unit.                                                                 |
| `support_token`                  | `string?`      | The argmax support token text (provenance arrows).                                                    |
| `chunk_id`                       | `string\|int?` | Stable chunk id of the argmax support unit, when available.                                            |
| `heatmap_score`                  | `float`        | Scalar used to drive UI heatmap intensity.                                                             |
| `char_start` / `char_end`        | `int?`         | Character offsets in `response_text`. Populated when the encoder's tokenizer exposes offset mappings. |
| `response_chunk_index`           | `int?`         | Index of the response chunk that produced this token when response chunking is active; `null` otherwise. |

### `nli_diagnostics.claims[i]: GroundednessNLIClaim`

Present when NLI is enabled. `nli_diagnostics.aggregate_score` carries
the aggregate entailment across tested claims.

| Field                   | Type                    | Description                                                                                                                                      |
|-------------------------|-------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------|
| `index`                 | `int`                   | Zero-based claim index.                                                                                                                          |
| `text`                  | `string`                | The extracted claim text.                                                                                                                        |
| `char_start`/`char_end` | `int`                   | Character offsets of the claim span within `response_text`.                                                                                      |
| `entailment`            | `float`                 | NLI entailment probability for the claim.                                                                                                        |
| `neutral`               | `float`                 | NLI neutral probability.                                                                                                                          |
| `contradiction`         | `float`                 | NLI contradiction probability.                                                                                                                    |
| `score`                 | `float`                 | Aggregate NLI score used by the fuser.                                                                                                           |
| `skipped`               | `bool`                  | `true` when the NLI peer skipped this claim.                                                                                                      |
| `skip_reason`           | `string?`               | Reason the claim was skipped, when applicable.                                                                                                    |
| `premise_count`         | `int`                   | Number of premise support units tested against this claim.                                                                                        |
| **`support_ids`**       | `string[]`              | Support ids (as in `support_units[i].support_id`) the claim was tested against. Used by the unused-context classifier to attribute NLI evidence. |
| **`support_unit_indices`** | `int[]`              | Global support-unit indices the claim was tested against (mirrors `support_ids`, in sync with `GroundednessSupportUnit.index`).                  |
| `atoms`                 | `GroundednessNLIAtom[]` | Per-atom entailment records produced by atomic-claim decomposition. Each atom mirrors the claim fields above, including `support_ids` and `support_unit_indices`. |

### `top_evidence[i]: GroundednessEvidence`

| Field                   | Type           | Description                                                           |
|-------------------------|----------------|-----------------------------------------------------------------------|
| `response_token_index`  | `int`          | Response-token index that originated this evidence link.              |
| `response_token`        | `string`       | Response token text.                                                  |
| `support_unit_index`    | `int`          | Global support-unit index.                                            |
| `support_token_index`   | `int`          | Token index inside that support unit.                                 |
| `support_token`         | `string`       | Support token text.                                                   |
| `chunk_id`              | `string\|int?` | Stable chunk id, when available.                                      |
| `metric`                | `string`       | Metric used to rank the pair (e.g. `reverse_context`).                |
| `score`                 | `float`        | Metric value.                                                          |

### `warnings[]` and `reason`

`warnings[]` is a free-form list of structured strings the engine emits
for non-fatal conditions (e.g. `"truncated_response_to_max_tokens"`).
It is *not* dispatch-keyed across releases. Use the top-level
`GroundednessResponse.reason` (machine-readable) when you need to
branch on refusal paths such as `"no_premise_supplied"` or
`"open_domain_pending_v1_next"`.

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

## Unused Context Contract

`latence-trace` now emits a second, stricter per-unit usage surface on top of
legacy coverage:

- `support_units[*].usage_state = "used"` means the unit had direct attribution,
  strong non-argmax coverage, or positive NLI evidence.
- `support_units[*].usage_state = "unused"` is **precision-first**. The engine
  only emits it when coverage is low, attribution is absent, NLI found no
  support, and the unit is not just a near-duplicate of a used sibling.
- `support_units[*].usage_state = "uncertain"` is the expected abstention path
  for mixed packed windows, semantically overlapping siblings, or otherwise
  ambiguous cases. Treat `uncertain` as "do not force a negative verdict", not
  as an engine failure.

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

---

## NLI model selection

The NLI peer is the verification channel that catches negation, role
swaps, exact numbers/dates, and entity substitutions that pure
embedding similarity misses. Two production-ready options ship out of
the box:

| Model (env: `VOYAGER_GROUNDEDNESS_NLI_MODEL`) | Languages | HaluEval QA paired acc | HaluEval Summ paired acc | When to use |
|---|---|---|---|---|
| `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` (default) | 100+ incl. EN, DE | 0.68–0.70 | 0.65–0.68 | Multilingual deployments, EN+DE, anything non-English. |
| `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` | EN only | **0.78–0.80** | **0.75–0.78** | English-only deployments. +10pp paired ranking accuracy. |

Same Pydantic schema, same `/groundedness` endpoint, same
`/agent-help` surface, same MCP descriptor. Switch the env variable
and restart the server. Token-level NLI heatmaps and
`claims[i].verdict` / `claims[i].score` are identical in shape across
both models.

> Both models are loaded via vLLM-Factory BYOP when
> `VOYAGER_GROUNDEDNESS_VLLM_ENDPOINT` is configured. The encoder
> (ColBERT) and the NLI peer can share one vLLM endpoint or run on
> separate endpoints — the engine fan-outs handle concurrent batching.

---

## Workload boundary (read before deploying)

`latence-trace` answers **"is this response anchored in the supplied
context?"** (RAG-grounding / faithfulness). It does not answer **"is
this response factually correct against world knowledge?"**
(open-domain factuality). The two questions look identical from the
outside but diverge on these workloads:

| Workload | Suitability | Why |
|---|---|---|
| RAG QA + summarization (anything where retrieved docs are the source) | ✅ Headline use case | What the engine is designed for. RAGTruth qa F1 0.73 / precision 0.98, HaluEval QA paired 0.78–0.80 (English NLI). |
| Tabular / structured-source grounding | ✅ Strong | NLI + structured triples drive 0.93 paired acc on internal hard-structured stratum. |
| Bilingual EN+DE on shared schema | ✅ Verified | German minimal pairs 1.00 paired with the multilingual NLI peer. |
| Distributed-evidence dialogue (support split across speaker turns) | ⚠️ Use with caution | 0.57 paired on internal `hard_dialogue_distributed`; pair with a context-rewriter or longer-premise reranker. |
| Open-domain dialogue continuation (HaluEval Dialogue) | ❌ Out of scope | Hallucinations introduce real-world facts not in the dialogue context; both right and hallucinated answers score ungrounded relative to the supplied premise. Use a knowledge-base fact-checker. |
| Per-claim atomic verification (FActScore-style biographies) | ✅ Supported when source is provided | Each annotation is treated as its own atomic claim and scored against the matching source paragraph (e.g. Wikipedia article enriched via `scripts/enrich_factscore_with_wiki.py`). At F1-optimal threshold the engine reports per-claim precision = 0.61, recall = 0.62, F1 = 0.62 on n = 748 atomic claims (30 biographies, multilingual mDeBERTa NLI). Without a source paragraph the workload degenerates to open-domain factuality and falls into the row above. |

See [`docs/algorithm-audit.md`](algorithm-audit.md) §"Scope and Known
Mismatches" for the full discussion.
