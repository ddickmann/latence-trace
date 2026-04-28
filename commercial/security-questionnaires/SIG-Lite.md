# SIG-Lite (Shared Assessments) responses - Latence TRACE v1.1

Responses mapped to the 2026 Shared Assessments SIG-Lite questionnaire.
Column `Evidence` links to the artifact under
`commercial/trust-center/` or `docs/operations/`.

This file is intentionally copy-paste friendly; a prospect can paste
this table into their SIG upload tool without reformatting.

## A. Risk management

| # | Question | Answer | Evidence |
|---|---|---|---|
| A.1 | Is there a formal information security programme with executive sponsorship? | Yes | `commercial/trust-center/policies/information-security.md` |
| A.2 | Is there an enterprise risk register updated at least quarterly? | Yes | `commercial/trust-center/policies/risk-register.md` |
| A.3 | Are security KPIs reported to the executive team? | Yes, monthly | Quarterly trust update (under NDA) |

## B. Privacy

| # | Question | Answer | Evidence |
|---|---|---|---|
| B.1 | Do you publish a DPA? | Yes | `commercial/hosted-dpa.md` |
| B.2 | Is customer data geo-restricted on request? | Yes - EU + US regions on request | `commercial/trust-center/data-residency.md` |
| B.3 | Is PII pseudonymised on the hot path? | Hot path does not persist PII; audit log is opt-in and tenant-scoped | `latence_trace/middleware/audit_log.py` |
| B.4 | Is there a data-subject access request (DSAR) process? | Yes, 30-day SLA | `docs/operations/dsar-runbook.md` |

## C. Access control

| # | Question | Answer | Evidence |
|---|---|---|---|
| C.1 | MFA required for all engineer access? | Yes (hardware keys) | `commercial/trust-center/iam.md` |
| C.2 | Privileged access reviewed quarterly? | Yes | `commercial/trust-center/iam.md` section 4 |
| C.3 | API keys scoped per tenant? | Yes, with rotation API | `gateway/cloudflare/src/index.ts` (`handleKeyRotate`, `handleKeyRevoke`) |

## D. Operations

| # | Question | Answer | Evidence |
|---|---|---|---|
| D.1 | Continuous deployment with peer review? | Yes, PR + two reviewers on `main` | Git repo branch-protection settings |
| D.2 | 24x7 on-call? | Yes, via PagerDuty | `docs/operations/incident-response.md` |
| D.3 | RTO / RPO? | RTO 4h / RPO 15min | `docs/operations/incident-response.md` sec 6 |
| D.4 | Public status page? | Yes - status.latence.ai | `docs/operations/status-and-changelog.md` |

## E. Sub-processors

| # | Question | Answer | Evidence |
|---|---|---|---|
| E.1 | Do you publish the list of sub-processors? | Yes | `commercial/trust-center/subprocessors.md` |
| E.2 | Advance notice on sub-processor changes? | 30-day email notice | DPA clause 5.4 |
| E.3 | Are sub-processors SOC 2 / ISO 27001? | Yes (Cloudflare + RunPod) | linked from subprocessors.md |

## F. Security monitoring

| # | Question | Answer | Evidence |
|---|---|---|---|
| F.1 | Centralised logs? | Yes | `docs/operations/observability.md` |
| F.2 | SIEM / alerting on security events? | Yes (Datadog) | `docs/operations/observability.md` sec 8 |
| F.3 | Vulnerability management SLA? | P0 <= 24h, P1 <= 7d | `docs/operations/vulnerability-management.md` |

## G. Known gaps (honest)

* SOC 2 Type I report in audit window; ETA Q3-2026.
* ISO 27001 certification roadmap: readiness review Q4-2026.
* First third-party pen test scheduled Q3-2026; letter of engagement
  available under NDA.
* External RAG benchmarks (HaluEval, RAGTruth) below SOTA; see
  `data/veracier-industries/proof_bundle_v1/failure_modes_v1.md` and
  `commercial/bold-claims-v1.md` for the exact numbers and the v2
  student-training mitigation.

No unmet regulatory obligation is hidden in this document.
