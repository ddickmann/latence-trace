# latence-trace Documentation

`latence-trace` ships the **Groundedness Tracker (Beta)** as a standalone
sidecar in the latence.ai product family. It scores how well an LLM response
is grounded in its supporting context, returns auditable per-claim evidence,
and classifies every output into a calibrated `green` / `amber` / `red`
risk band. It is **multilingual: English + German** out of the box.

- **Product overview:** [`guides/beta-overview.md`](guides/beta-overview.md)
- **Pareto-optimal default profiles (`fast` / `balanced` / `quality`):**
  [`guides/profiles.md`](guides/profiles.md)
- **Multilingual configuration (EN + DE):**
  [`guides/multilingual.md`](guides/multilingual.md)
- **Concurrency & batching (inflight semaphore, `LATENCE_TRACE_MAX_INFLIGHT`):**
  [`guides/concurrency.md`](guides/concurrency.md)
- **Benchmarks:** [`benchmarks.md`](benchmarks.md)
- **Algorithm audit (per-channel ablations and verdict):**
  [`algorithm-audit.md`](algorithm-audit.md)
- **Quickstart and integration notes:** see the top-level `README.md`.

## Live discovery surfaces

A running `latence-trace` server self-describes for AI agents and devs:

| Endpoint                          | What it returns                                                                                       |
|-----------------------------------|-------------------------------------------------------------------------------------------------------|
| `GET /agent-help`                 | Canonical request shape, active/default/available profiles, full endpoint map                         |
| `GET /.well-known/ai-plugin.json` | ChatGPT-style plugin descriptor for AI tool registries                                                |
| `GET /openapi.json`               | OpenAPI 3.1 schema (with `operation_id: score_groundedness`)                                          |
| `GET /docs`                       | Human-readable Swagger UI                                                                             |
| `GET /healthz`                    | Liveness probe                                                                                        |
| `GET /readyz`                     | Readiness probe with warmup status and `max_inflight` echo                                            |

All 4xx and 5xx responses share the same structured envelope
(`{detail: {code, message, hint, docs_url}}`), so agents and SDKs can
parse one shape regardless of the failure mode.

This product is proprietary. See `LICENSE`. For commercial integrations,
reach out to latence.ai.
