# Hosted Service Level Agreement — TRACE v1

**Effective date:** 2026-04-28  
**Service:** `api.latence.ai` — TRACE hosted RAG validation API  
**Applies to:** Business and Enterprise tiers on the hosted SaaS plan.

## 1. Uptime commitment

### 1.1 Monthly availability target

Latence commits to a **99.5% monthly availability** for the `Score
Groundedness` API endpoints (`POST /v1/groundedness/score`,
`POST /v1/code/score`) measured on a calendar-month basis, excluding
scheduled maintenance windows.

### 1.2 Measurement

Availability is measured by:

- External synthetic probes from three geographic regions (Frankfurt,
  N. Virginia, Singapore), at 60-second cadence, executing a canonical
  score request against a fixed evidence pack;
- Server-side success rate for 2xx responses from `api.latence.ai`
  after rate-limit filtering.

The lower of the two figures is reported.  Monthly availability reports
are published to `status.latence.ai` within five business days of
month-end.

### 1.3 Scheduled maintenance

- Announced at least 72 hours in advance on `status.latence.ai` and
  by email to the tenant's billing contact.
- Capped at 4 hours per calendar month.
- Scheduled outside 07:00-20:00 UTC Monday-Friday.

## 2. Latency posture

Hosted TRACE provides **two latency figures**, not one, and customers
choose based on their traffic pattern.

| Profile | Isolated p95 target | Sustained (32-concurrency) p95 target |
| --- | ---: | ---: |
| `standard` | <= 1500 ms | <= 4500 ms |
| `quality` | <= 2800 ms | <= 8000 ms |

Isolated p95 is measured on a quiescent worker; sustained p95 is
measured under a constant 32-concurrency load for 10 minutes.  Both
figures are published monthly to `status.latence.ai`.  See
`docs/operations/observability.md` for the exact PromQL.

## 3. Service credits

Monthly availability below 99.5% triggers pro-rata service credits
against the next invoice:

| Measured availability | Credit |
| --- | --- |
| < 99.5% and >= 99.0% | 5% of monthly fee |
| < 99.0% and >= 98.0% | 10% of monthly fee |
| < 98.0% and >= 95.0% | 25% of monthly fee |
| < 95.0% | 50% of monthly fee (plus termination-for-cause right) |

Credits are the customer's sole and exclusive remedy for unavailability.
Claims must be filed in writing to `support@latence.ai` within 30 days
of the affected month-end.

## 4. Support targets

| Severity | Response | Workaround | Fix target |
| --- | ---: | ---: | ---: |
| P1 — production outage, no workaround | 30 minutes | 4 hours | 24 hours |
| P2 — degraded performance, workaround exists | 2 business hours | 1 business day | 5 business days |
| P3 — non-production defect | 1 business day | 5 business days | next minor release |
| P4 — cosmetic / documentation | 5 business days | - | next quarterly release |

Response hours are 08:00-20:00 UTC, Monday-Friday, excluding Latence
company holidays.  Enterprise tier customers may add 24x7 P1 response
for an additional fee (see `order-form-template.md`).

## 5. On-call rotation

- PagerDuty is the pager of record for Latence on-call engineers.
- The `api.latence.ai` canary and error-rate alerts defined in
  `docs/operations/observability.md` fire PagerDuty high-urgency
  incidents.
- Alerts escalate to the secondary on-call at T+15 minutes if
  unacknowledged.

## 6. Exclusions

This SLA does not apply to:

- Customer-caused errors (malformed requests, expired API keys, quota
  exhaustion).
- Force majeure events.
- Unavailability of hyperscaler dependencies (Cloudflare, RunPod)
  outside of reasonable Latence control; Latence will share the
  upstream incident reference.
- Beta / pre-GA endpoints, which are governed by the beta terms at the
  time of access.

## 7. Contact

- Support: `support@latence.ai`
- Incidents: `incidents@latence.ai` (monitored by PagerDuty)
- Status: `https://status.latence.ai`
