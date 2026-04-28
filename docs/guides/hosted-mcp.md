# Hosted MCP endpoint

Latence TRACE exposes a remote Model Context Protocol endpoint at
`https://api.latence.ai/mcp` so any MCP-compatible agent runtime can
discover and call `score_groundedness` with a single HTTP-level
configuration file.

## Transports

Both transports run against the same Cloudflare gateway and the same
upstream RunPod worker.  Pick the one your client speaks natively:

| Transport | URL | Best for |
| --- | --- | --- |
| streamable-HTTP | `POST https://api.latence.ai/mcp` | Cursor, custom orchestrators, short-lived agents |
| SSE | `GET https://api.latence.ai/mcp/sse` + `POST https://api.latence.ai/mcp/messages` | Claude Desktop, long-running review sessions |

Both transports share the same JSON-RPC 2.0 method set:

- `initialize`
- `tools/list`
- `tools/call` (`name="score_groundedness"`)
- `ping`

See `latence_trace/mcp/server.py` for the canonical dispatcher and
`latence_trace/mcp/remote.py` for the HTTP wrapper.

## Authentication

Every request must carry `Authorization: Bearer <api_key>` where
`<api_key>` is a key issued via the portal or the
`/v1/keys/issue` admin endpoint (see
`gateway/cloudflare/README.md`).  The gateway:

1. Resolves the API key from Cloudflare KV (`TENANT_KEYS`).
2. Enforces the per-tenant Durable Object rate limit
   (`RATE_LIMITER`).
3. Checks the monthly quota in D1 (`USAGE_DB`).
4. Forwards the request to the upstream with a
   `X-Latence-Tenant-Id` header.
5. Logs the call to Analytics Engine and D1 for billing.

See `docs/operations/rate-limits.md` for the exact headers and error
shapes on overrun.

## Client manifests

Drop-in manifests for the three most-used clients live under
`clients/mcp-manifests/`:

- `claude-desktop.json` - SSE transport
- `cursor.json` - streamable-HTTP transport
- `openai-agents.py` - reference Python script using the
  Agents SDK `MCPServerSse` transport.

## Local development

Run the HTTP wrapper locally against `runpod/dev_app.py`:

```bash
LATENCE_TRACE_ENABLE_MCP_HTTP=1 \
  uvicorn server.main:app --host 127.0.0.1 --port 8000

curl -N -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' \
  http://127.0.0.1:8000/mcp
```

The gateway-side rate limiting and quota enforcement run only on the
hosted endpoint; self-hosted callers apply their own quotas via
`RateLimitMiddleware` in `server/main.py`.

## Error handling

JSON-RPC errors follow the standard code set:

| Code | Condition |
| --- | --- |
| `-32700` | Malformed JSON |
| `-32600` | Request is not a JSON-RPC 2.0 object |
| `-32601` | Unknown method |
| `-32602` | Invalid params for the tool |
| `-32603` | Internal error; tenant should retry with backoff |

The gateway additionally returns HTTP 401 for unknown keys, 402 for
tier caps, 429 for rate-limit overruns, and 503 when the upstream
worker is unavailable.  Clients should honour `Retry-After` headers
on 429 and 503.

## Observability

Each request emits:

- A Prometheus histogram `latence_trace_mcp_call_duration_seconds`
  labelled by `method` and `tenant_id`.
- An Analytics Engine row in the gateway for per-tenant billing.
- An audit log entry on the worker (see
  `latence_trace/middleware/audit_log.py`).

Grafana dashboard: `docs/operations/grafana/trace-hosted-reliability.json`
(panel "MCP call duration").
