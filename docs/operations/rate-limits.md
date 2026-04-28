# Rate limits

This document describes the rate-limit contract for `api.latence.ai`
hosted TRACE and the self-hosted equivalent.

## 1. Contract surface

Rate limits are enforced on the **API gateway** (Cloudflare Worker) in
the hosted deployment, and by `latence_trace/middleware/rate_limit.py`
(per-process token bucket) in the self-hosted deployment.

Each tenant has:

- A **requests-per-second** ceiling (default 20 RPS on Business, 100
  RPS on Enterprise).
- A **monthly quota** (see `commercial/hosted-price-card.md`).
- Optional per-path overrides on Enterprise.

## 2. Response headers

Every response carries:

| Header | Meaning |
| --- | --- |
| `X-RateLimit-Limit` | RPS ceiling for this tenant on this path |
| `X-RateLimit-Remaining` | Tokens remaining in the current 1-second window |
| `X-RateLimit-Reset` | Unix timestamp when the window resets |
| `X-RateLimit-Quota` | Monthly quota |
| `X-RateLimit-Quota-Remaining` | Scores remaining this calendar month |

## 3. HTTP 429 behaviour

When the RPS ceiling is exceeded, the gateway returns `429 Too Many
Requests` with:

```json
{
  "error": {
    "code": "rate_limited",
    "message": "Tenant rate limit exceeded; see Retry-After",
    "type": "rate_limit_error"
  }
}
```

and a `Retry-After: <seconds>` header.  Client SDKs retry with
exponential backoff automatically.

When the **monthly quota** is exceeded, the gateway returns `402
Payment Required` (soft cap on paid tiers) or `429` (hard cap on Free
tier) with `error.code = "quota_exceeded"`.

## 4. Self-hosted enforcement

For self-hosted deployments the per-process token bucket implemented
in `latence_trace/middleware/rate_limit.py` is a **single-process
limit**.  Multi-process deployments should enforce tenant rate limits
at the ingress (e.g. Envoy, Kong, Cloudflare Gateway, NGINX Plus)
because the in-process limiter does not share state across replicas.
This is documented in the Helm chart values
(`helm/latence-trace/values.yaml::rateLimit.mode`).

## 5. Raising a limit

Raise a limit by:

- Enterprise: contact TAM / `support@latence.ai`.
- Business: upgrade tier or request a temporary burst via the portal
  ("Request burst capacity" under `/billing/limits`).
- Self-hosted: change `rateLimit.rps` in the Helm values and restart
  the pods (or rotate with `kubectl rollout restart deploy/latence-trace`).

## 6. Observability

- Prometheus metric `trace_rate_limit_blocked_total{tenant,path}`.
- Grafana dashboard `docs/operations/grafana/api-gateway.json` has a
  `Rate limit saturation` panel.
- Alert `RateLimitSaturationHigh` fires when a single tenant is >80%
  of its RPS ceiling for >5 minutes.
