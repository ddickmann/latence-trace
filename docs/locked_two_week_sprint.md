# Locked Two-Week TRACE Sprint

Start this only after:

- The waitlist-first marketing page is live.
- `/api/early-access` persists leads and sends founder notifications.
- Old Data Intelligence product surfaces are hidden or feature-flagged.
- `HIBERNATION.md` has been reviewed.

## Sprint Goal

One believable TRACE pilot path from qualified lead to working runtime.

## Non-Negotiables

- No broad portal rebuild.
- No old Data Intelligence scope.
- No self-serve billing detour.
- No new integrations unless they unblock a real pilot.
- No uncalibrated guardrail claims.

## Week 1: Pilot Path Online

| Day | Focus | Done When |
| --- | --- | --- |
| 1 | Select hosted route owner | `api.latence.ai` or pilot subdomain owner is documented. |
| 2 | RunPod/TRACE origin health | Runtime health endpoint and one RAG score work. |
| 3 | Gateway contract | RAG, code, redaction paths route to TRACE with auth. |
| 4 | SDK quickstart | Python quickstart works against the selected endpoint. |
| 5 | Demo dataset | One RAG sample and one code-agent sample are repeatable. |

## Week 2: Pilot-Ready Hardening

| Day | Focus | Done When |
| --- | --- | --- |
| 6 | Observability | JSON logs, request IDs, basic error visibility. |
| 7 | Privacy posture | No raw compliance text in metrics/logs; retention stated. |
| 8 | Deployment docs | Cloud evaluation and VPC/on-prem next steps are documented. |
| 9 | Design-partner pack | One-pager, demo script, security notes, quickstart ready. |
| 10 | Founder demo | End-to-end demo: lead, qualify, issue access, score, show result. |

## Daily Checkpoint

At end of each day, record:

- What works now.
- What is blocked.
- What was deliberately not built.
- The single next action for tomorrow.

## Exit Criteria

- A qualified lead can be onboarded manually.
- The selected endpoint can score RAG and code examples.
- PII redaction is available or explicitly deferred.
- Pilot collateral is ready to send.
- The old product remains hibernated and does not compete for founder focus.
