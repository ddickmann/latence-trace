# latence-trace Documentation

`latence-trace` ships **Latence TRACE** as an enterprise AI compliance
runtime. It scores how well an LLM response is grounded in supporting context,
returns auditable per-claim evidence, redacts PII in real time, and exposes
privacy-safe observability across verification and compliance lanes. It is
**multilingual: English + German** out of the box for groundedness.

- **End-to-end tutorial (recommended starting point):**
  [`guides/tutorial.md`](guides/tutorial.md)
- **API reference (every field, every endpoint):**
  [`api-reference.md`](api-reference.md)
- **Stateful TRACE Sessions:**
  [`trace_sessions.md`](trace_sessions.md)
- **Product overview:** [`guides/beta-overview.md`](guides/beta-overview.md)
- **Pareto-optimal default profiles (`fast` / `balanced` / `quality`):**
  [`guides/profiles.md`](guides/profiles.md)
- **Multilingual configuration (EN + DE):**
  [`guides/multilingual.md`](guides/multilingual.md)
- **Concurrency & batching (inflight semaphore, `LATENCE_TRACE_MAX_INFLIGHT`):**
  [`guides/concurrency.md`](guides/concurrency.md)
- **Benchmarks:** [`benchmarks.md`](benchmarks.md)
- **Veracier enterprise RAG validation:** [`veracier_trace_validation.md`](veracier_trace_validation.md)
- **Algorithm audit (per-channel ablations and verdict):**
  [`algorithm-audit.md`](algorithm-audit.md)
- **Quickstart and integration notes:** see the top-level `README.md`.

## Live discovery surfaces

A running `latence-trace` server self-describes for AI agents and devs:

| Endpoint                          | What it returns                                                                                       |
|-----------------------------------|-------------------------------------------------------------------------------------------------------|
| `GET /agent-help`                 | Canonical request shape, active/default/available profiles, full endpoint map                         |
| `POST /v1/trace/sessions`         | Create a stateful TRACE + InfiniMem session for coding, RAG, or general-purpose agents                |
| `POST /v1/trace/sessions/{id}/score` | Score a turn while the server loads, updates, and returns bounded session memory                    |
| `GET /v1/trace/sessions/{id}/context` | Return the current bounded hot context plus memory counters                                         |
| `GET /v1/trace/sessions/{id}/sources/{source_id}` | Fetch a redacted immutable source-vault record by pointer                              |
| `POST /v1/trace/sessions/{id}/repair` | Build a bounded repair packet from original source history                                          |
| `POST /v1/memory/update`          | Stateless/caller-carried InfiniMem update for custom integrations                                     |
| `POST /v1/compression`            | Standalone LLMLingua2 ingress compression                                                             |
| `POST /v1/compliance/redact`      | PII detection, mask/replacement redaction, custom regex labels, and usage/timing metadata             |
| `GET /v1/compliance/schema`       | Compliance label catalog, GDPR categories, optimized GLiNER aliases, and supported modes              |
| `GET /.well-known/ai-plugin.json` | ChatGPT-style plugin descriptor for AI tool registries                                                |
| `GET /openapi.json`               | OpenAPI 3.1 schema (with `operation_id: score_groundedness`)                                          |
| `GET /docs`                       | Human-readable Swagger UI                                                                             |
| `GET /healthz`                    | Liveness probe                                                                                        |
| `GET /readyz`                     | Readiness probe with warmup status and `max_inflight` echo                                            |

All 4xx and 5xx responses share the same structured envelope
(`{detail: {code, message, hint, docs_url}}`), so agents and SDKs can
parse one shape regardless of the failure mode.

The compliance label catalog includes full-address redaction plus calibrated
address components (`street_address`, `postal_code`, `city`, `country`) for
applications that need finer-grained masking or country-aware synthetic
replacement with `redaction_mode="replace"`.

This product is proprietary. See `LICENSE`. For commercial integrations,
reach out to latence.ai.
