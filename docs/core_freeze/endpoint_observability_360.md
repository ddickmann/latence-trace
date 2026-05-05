# Endpoint Observability 360

This audit checks each public TRACE product path against the website promise:
detailed traces, scores, decisions, redacted evidence, context waste, guard risk,
latency, and operational diagnostics.

## Privacy Redaction

Canonical path: `POST /v1/compliance/redact`

Evidence returned:

- Redacted output: `redacted_text`
- Detection evidence: `entities[]`, canonical labels, scores, offsets, sources
- Privacy-safe usage: `entity_count`, `unique_labels`, `labels_used`,
  `selected_categories`, `usage`
- Timings: `processing_time_ms`, `timings_ms`
- Redaction controls echoed through usage: mode, categories, redaction mode

Verdict: logging-ready. Callers must avoid persisting `entities[*].text` unless
they have a lawful audit reason.

## Grounding: RAG

Canonical path: `POST /groundedness` with `scoring_mode="rag"` or omitted.

Evidence returned:

- Scores: primary, reverse context, calibrated channels, literal/NLI/semantic
  entropy, structured-source, consensus, support-unit usage, context coverage
- Decisions: `risk_band`, `risk_reason`, `runtime_decision`,
  optional `amber_escalation`
- Evidence: response tokens, support units, top evidence, file attribution,
  heatmap data or HTML
- Guard risk: `context_trust_diagnostics`, score channels, per-support-unit
  trust state, labels, spans
- Runtime details: corpus route, runtime head features, warnings, time_ms,
  profile diagnostics, effective profile

Verdict: elite logging-ready. Compact RunPod responses now surface guard fields
at top level and in `score_channels`; canonical responses preserve the full
Pydantic payload.

## Grounding: Code

Canonical path: `POST /groundedness` with `scoring_mode="code"`.

Evidence returned:

- All RAG fields plus code-lane diagnostics
- AST drift, literal novelty, NLI cascade, composite phantom score, file
  attribution, dead-weight file signals, runtime head features
- Caller-carried state: `next_session_state`, `session_signals`

Verdict: logging-ready for IDE and coding-agent traces. The SDK now forwards
`session_id`, memory state, and metadata from `TraceSession.code(...)`.

## Compression

Canonical path: `POST /v1/compression`

Evidence returned:

- Compressed output: `compressed_text`, optional `compressed_messages`
- Savings: original/compressed tokens, ratio, percentage, tokens saved
- Evidence: preserved terms and compression spans
- Runtime diagnostics: provider and `diagnostics`

Verdict: logging-ready. The SDK now parses this as a typed
`CompressionResponse` while preserving raw forward-compatible fields.

## Memory / InfiniMem

Canonical path: `POST /v1/memory/update`

Evidence returned:

- State continuity: `next_memory_state`
- Working context: `hot_context`
- Memory actions: added, kept, demoted, removed, deduped, superseded, anchored
- Diagnostics: hot/warm/cold/removed tokens, effective budgets, actual token
  reduction, quality gates, exact-critical recall, timings

Verdict: logging-ready. The SDK session facade now consumes
`next_memory_state`, sends it on later turns, and can persist snapshots through
in-memory or file storage adapters.

## Rollup

Canonical path: `POST /groundedness/rollup`

Evidence returned:

- Session-level aggregates over supplied turns
- Risk trail, drift/noise/retrieval waste where supplied by turn records
- Reason-code histograms, top dead files, recommendations

Verdict: logging-ready for client-carried sessions. `TraceSession.rollup(...)`
now summarizes the local SDK event buffer without relying on sticky server
state.

## Discovery And Runtime

Canonical discovery:

- `GET /agent-help`
- `GET /healthz`
- `GET /readyz`
- OpenAPI from the generic FastAPI server
- `docs/core_freeze/api_surface_manifest.json`
- `docs/core_freeze/trace_feature_inventory.json`

The discovery gate now cross-checks routes, RunPod actions, Pydantic request
fields, SDK methods, examples, benchmarks, manifest product paths, and env
settings. Environment settings are intentionally emitted for review in
`env_settings_to_review`; blocking drift fails `--check`.

## Residual Risks

- Some operational `LATENCE_TRACE_*` tuning flags are discovered mechanically
  but remain review items rather than release blockers because many are
  deployment-only controls.
- Full managed-vLLM local proof is still required after implementation audit to
  validate these payloads with real model services.
