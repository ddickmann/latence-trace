# Minimal SDK / Gateway Contract

This contract is the narrow surface for the locked TRACE sprint. It prevents the
old SaaS scope from returning through SDK, gateway, or portal work.

## Public Pilot Endpoints

Hosted pilots should expose only:

| Capability | Hosted path | Runtime path |
| --- | --- | --- |
| RAG groundedness | `POST /api/v1/trace/rag` | `POST /groundedness` with `scoring_mode="rag"` |
| Code groundedness | `POST /api/v1/trace/code` | `POST /groundedness` with `scoring_mode="code"` |
| Session rollup | `POST /api/v1/trace/rollup` | `POST /groundedness/rollup` or RunPod `action="rollup"` |
| PII redaction | `POST /api/v1/trace/compliance/redact` or `POST /v1/compliance/redact` | `POST /v1/compliance/redact` |
| Compression | `POST /api/v1/trace/compression` only if offered | `POST /v1/compression` |

Do not expose old document-intelligence, dataset, pipeline, ontology, storage,
or billing endpoints as part of TRACE pilot onboarding.

## SDK Methods

The pilot SDK surface should be limited to:

- `trace.rag(...)`
- `trace.code(...)`
- `trace.rollup(...)`
- `trace.redact(...)`
- Optional: `trace.compress(...)`

SDK defaults:

- Require an explicit `base_url` for private/VPC deployments.
- Hosted examples may use `https://api.latence.ai` only after the gateway route
  owner is confirmed.
- Preserve access to raw JSON for forward compatibility.
- Retry GET/health and idempotent stateless calls only.
- Do not hide diagnostics from developers.

## Gateway Responsibilities

The gateway owns:

- API-key authentication.
- Tenant identity.
- Rate limits.
- Privacy-safe usage metering.
- Optional tenant threshold injection.
- Route mapping to the selected TRACE origin.

The runtime owns:

- Groundedness scoring.
- Code-lane scoring.
- PII redaction.
- Compression.
- Runtime decisions.
- Health/readiness.

The portal owns:

- Waitlist and qualification.
- Manual access issuance during pilots.
- Minimal dashboard/insights once data exists.

## Routing Decision Required Before Public Pilot

Only one worker/router should own `api.latence.ai/*`. Before issuing external
keys, choose one:

1. Existing Python gateway with `gateway/src/services/trace.py` active.
2. TRACE-specific Cloudflare Worker under `latence-trace/gateway/cloudflare`.
3. A temporary subdomain such as `trace-api.latence.ai` for pilots.

Document the selected owner in the pilot runbook before sharing API keys.

## Deferred

- Full self-serve key lifecycle in the public UI.
- Stripe/Paddle self-serve billing reconciliation.
- Durable stateful sessions.
- MCP remote exposure unless a pilot explicitly needs it.
- Any integration that bypasses the SDK and calls compute runtime directly.
