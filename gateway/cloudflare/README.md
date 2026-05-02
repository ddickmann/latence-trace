# Cloudflare Gateway (`api.latence.ai`)

The Cloudflare Worker that sits in front of RunPod TRACE workers.
Implements:

- **B1** — Tenant API key verification, rotation, revocation.
  Keys are stored in KV keyed by `sha256(api_key)`; rotation writes
  a new entry and marks the old one revoked for 30 days.
- **B2** — Per-tenant threshold routing.  When a tenant-specific
  thresholds JSON is present in the `THRESHOLDS` KV namespace, the
  gateway forwards it base64-encoded in the
  `x-latence-tenant-thresholds` header; the Python scorer consumes
  it via `get_risk_band_policy_for_tenant`.
- **B3** — Usage metering.  Every successful score or compliance redaction writes a
  data-point to Analytics Engine and bumps a `usage` row per
  `(tenant_id, month, band)`. Compliance is metered under the `trace`
  service with lane `compliance`. Monthly billing export is driven by
  the `billing_exports` table and the `scripts/export_usage.ts`
  Stripe exporter (see below).
- **Rate limiting** — Durable Object token bucket keyed by
  `tenant_id`, refilled at the tenant's RPS ceiling.
- **C6** — `/mcp/*` paths are proxied to the remote MCP origin with
  tenant context attached.

The gateway is payload-transparent for TRACE v2 runtime-head fields. Requests
may include `runtime_head_features` or `trajectory_features`, and responses may
include `runtime_decision`; the worker forwards those fields unchanged while it
continues to meter by the returned band.

Compliance routes are forwarded to the same TRACE origin:

- `POST /v1/compliance/redact` — authenticated PII detection/redaction.
- `GET /v1/compliance/schema` — authenticated label/category discovery.
- `GET /v1/compliance/healthz` — authenticated readiness check.

Gateway analytics for compliance must remain privacy-safe. Meter aggregate
values only; never write request text, entity text, replacement values, or
redacted output into logs, D1 rows, Analytics Engine blobs, or portal
`usage_details`.

## Deployment

```bash
npm install -g wrangler
cd gateway/cloudflare
wrangler login
wrangler kv namespace create API_KEYS
wrangler kv namespace create THRESHOLDS
wrangler d1 create latence-usage
wrangler d1 create latence-audit
wrangler d1 migrations apply latence-usage
wrangler secret put TRACE_ORIGIN_KEY
wrangler deploy
```

Populate `wrangler.toml` placeholders (`<api-keys-kv-id>` etc.) from
the output of the `kv namespace create` and `d1 create` commands.

## Tenant onboarding

Adding a tenant:

```bash
TENANT=acme
API_KEY="ltk_$(openssl rand -hex 32)"
HASH=$(printf "%s" "$API_KEY" | sha256sum | cut -d' ' -f1)
wrangler kv key put --namespace-id=<api-keys-kv-id> "$HASH" '{
  "tenant_id":"'"$TENANT"'","status":"active","plan":"business",
  "rps_ceiling":100,"monthly_quota":300000
}'
echo "API key for $TENANT: $API_KEY"
wrangler d1 execute latence-usage --command="
INSERT INTO tenants(tenant_id, plan, rps_ceiling, monthly_quota)
VALUES ('$TENANT', 'business', 100, 300000);"
```

Rotation and revocation are customer-self-service:

- `POST /v1/keys/rotate`  (returns a new `api_key`)
- `POST /v1/keys/revoke`  (204 on success)

## Usage export to Stripe

Usage data lives in D1 (`usage` table) and Analytics Engine.  The
monthly export script (`scripts/export_usage.ts`) reads D1,
computes overage, writes a row to `billing_exports`, and posts
usage to Stripe as invoice line items.  Run it on the 1st of each
month via `wrangler cron`.

## Secrets

- `TRACE_ORIGIN_KEY` — upstream RunPod auth token (stored via
  `wrangler secret put`).
- `STRIPE_API_KEY` — Stripe secret key for usage export (stored via
  `wrangler secret put` on the cron worker).

## Observability

- Every request emits a structured log line with `tenant_id`,
  `path`, `band`, and `latency_ms`.  Forward to Datadog via
  `wrangler logs forward`.
- Rate-limit rejections emit `rate_limit_blocked` Analytics Engine
  events keyed by tenant.
- Quota rejections emit `quota_exceeded` events.

## Local development

```bash
wrangler dev
curl -H 'authorization: Bearer ltk_test_key' \
     -H 'content-type: application/json' \
     -d '{"question":"q","response":"r","context":[],"runtime_head_features":{"v1_score":0.99}}' \
     http://localhost:8787/v1/groundedness/score

curl -H 'authorization: Bearer ltk_test_key' \
     -H 'content-type: application/json' \
     -d '{"text":"Contact jane@example.com","labels":["email"],"redact":true,"redaction_mode":"mask"}' \
     http://localhost:8787/v1/compliance/redact
```
