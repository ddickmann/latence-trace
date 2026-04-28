# Incident Response Runbook

## 1. Severity matrix

| Severity | Definition | Examples |
| --- | --- | --- |
| SEV-1 | Complete outage or data-integrity incident; no workaround | `api.latence.ai` returns 5xx for >50% of requests for >5 minutes; confirmed tenant cross-contamination; confirmed data loss in audit log |
| SEV-2 | Partial outage or severe degradation; workaround exists | Single region degraded; p95 latency >3x target for >10 min; hosted MCP endpoint down (stdio still works); sampled logs ingestion delayed >1 hour |
| SEV-3 | Minor defect, non-blocking | Isolated tenant issue; one evidence field missing from response; rate-limit headers inconsistent |
| SEV-4 | Cosmetic / documentation | Wrong content-type in an error response; typo in error message |

## 2. Call tree

1. **Primary on-call** — pager via PagerDuty `latence-trace-prod`.
2. **Secondary on-call** — auto-escalate at T+15 minutes.
3. **Incident Commander (IC)** — primary on-call assumes IC role for
   SEV-1 / SEV-2 unless explicitly delegated.
4. **Communications Lead (Comms)** — designated for SEV-1 only; owns
   status page + customer email updates.
5. **Engineering Lead** — paged for SEV-1 by IC; owns root-cause
   investigation.

Out-of-hours rotation calendar lives in Google Calendar
(`latence-oncall@latence.ai`) and is mirrored to PagerDuty.

## 3. SEV-1 playbook

### 3.1 First 15 minutes (triage)

- Acknowledge pager.
- Open incident channel `#inc-YYYYMMDD-<tag>` in Slack.
- Post "investigating" status page update.
- Pull top dashboards:
  - Cloudflare Worker 5xx rate (`gateway_5xx_rate`)
  - RunPod worker health (`worker_ready`, `worker_latency_p95`)
  - TRACE score error rate (`trace_score_error_rate`)
- If >50% 5xx on `api.latence.ai` → confirm SEV-1.

### 3.2 First 30 minutes (containment)

- If regional degradation → fail over to secondary RunPod region
  (`runbooks/failover.md`).
- If rate-limit storm → enable emergency circuit breaker on gateway
  (`cloudflare/workers/ratelimit.ts::CIRCUIT_BREAKER=true`).
- If sampled-logging store is the cause of P95 spike → disable sampled
  logging globally via `latence-trace-admin toggle sampled-logging off`.

### 3.3 First hour (mitigation + comms)

- Comms posts hourly on status page until resolution.
- Customer email to technical-contacts of tenants in the affected
  region at T+45 minutes for SEV-1.
- IC updates the incident channel every 15 minutes.

### 3.4 Resolution

- Confirm metrics back to baseline for 30 consecutive minutes.
- Status page transitions to "resolved".
- Schedule post-mortem within 3 business days.
- Post public post-mortem within 10 business days for SEV-1 that was
  tenant-visible.

## 4. Communications templates

### 4.1 Initial status page

```
Investigating - We are investigating elevated error rates on
api.latence.ai (all regions / <region>) starting at <UTC time>.
Status updates will follow every 15 minutes.
```

### 4.2 Customer email (SEV-1)

```
Subject: [TRACE] Incident <YYYYMMDD-tag> - investigating

Hi <tenant technical contact>,

We are investigating an incident affecting api.latence.ai that began
at <UTC time>. Symptoms: <one-line customer-visible symptom>.
Workaround: <if any, else "none at this time">.

Status page:  https://status.latence.ai/incidents/<tag>
Incident ref: <tag>

We will email you again within 60 minutes with an update.

Incident Commander: <name>
Latence
```

### 4.3 Resolution post

```
Resolved - The incident affecting api.latence.ai is resolved. All
metrics returned to baseline at <UTC time>. A post-mortem will be
published at https://status.latence.ai/postmortems/<tag> within 10
business days.
```

## 5. Post-mortem template

Stored under `docs/operations/postmortems/YYYYMMDD-<tag>.md`:

- Summary (1 paragraph, customer-visible impact)
- Timeline (UTC)
- Detection
- Response
- Root cause (5-whys)
- Remediation (with owners and due dates)
- What went well
- What we are changing (runbook, dashboard, alert, code)

## 6. Status page

`https://status.latence.ai` is the source of truth for customer
communications during an incident.  Do not post incident details
to marketing channels until the incident is resolved.
