# TRACE Pilot Path

This is the default path from waitlist lead to working TRACE pilot. It keeps the
product narrow enough for a solo-founder launch while preserving enterprise
deployment credibility.

## Qualification Intake

Every lead should be qualified from the `early_access_leads` fields:

- Work email, company, role.
- Use case: RAG, coding agents, agent workflows, or other.
- Deployment need: Cloud evaluation, Dedicated cloud, VPC, or on-prem.
- Data sensitivity.
- Expected trace volume.
- Design-partner interest.
- The specific workflow they want to de-risk.

## Path A: Hosted Cloud Evaluation

Use this for demos, design partners, and low-risk workloads.

| Decision | Default |
| --- | --- |
| Runtime | RunPod all-in-one TRACE worker. |
| Gateway | Existing TRACE route mapping through `gateway/src/services/trace.py` or the TRACE-specific Cloudflare Worker once routing is consolidated. |
| Data policy | No sensitive production data; short retention; privacy-safe metrics only. |
| API surface | RAG groundedness, code groundedness, PII redaction, optional compression. |
| Dashboard | Minimal TRACE dashboard/insights only. |
| Billing | Manual pilot access; no self-serve per-request billing as the main product promise. |

Cloud evaluation promise:

> Cloud is for evaluation and low-risk workloads. Sensitive production data
> should use VPC, dedicated cloud, or on-prem.

## Path B: VPC / On-Prem Qualification

Use this for sensitive production workloads and serious enterprise buyers.

The first response should collect:

- Cloud provider, region, and Kubernetes availability.
- GPU availability or preferred managed GPU provider.
- Retention and raw-payload policy.
- SSO, audit, export/delete, and procurement requirements.
- Expected traces/day and context size.
- Whether air-gapped deployment is required.

VPC/on-prem pilot deliverables:

- Docker or Helm bundle.
- Model/runtime configuration.
- License file or JWT.
- Health check and readiness checklist.
- Privacy-safe logging policy.
- Local dashboard or minimal export path.
- Upgrade/support channel.

## Out Of Scope For First Pilot

- Full self-serve billing.
- Old Data Intelligence pipeline flows.
- Server-side durable TRACE sessions unless the pilot explicitly needs them.
- InfiniMem enforcement for high-stakes decisions before quality gates are complete.
- Prompt-injection/exfiltration guardrails as primary allow/block gates before calibration.

## Success Criteria

- Customer can send one RAG or code-agent trace through the chosen endpoint.
- Response includes score, band, diagnostics, and runtime decision.
- PII redaction works on a representative sample without logging raw sensitive text.
- Founder can show one dashboard or structured output view.
- Next step is clear: continue pilot, move to VPC/dedicated, or disqualify.
