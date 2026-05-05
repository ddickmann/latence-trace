# TRACE Design-Partner Brief

## One-Liner

Latence TRACE is a deployable runtime that verifies whether AI answers and
agent steps are grounded in supplied context, catches coding-agent drift, and
redacts PII before sensitive text crosses a trust boundary.

## Who This Is For

- Teams shipping RAG applications where unsupported claims create business risk.
- Teams running coding agents and wanting drift/phantom-symbol detection.
- Enterprise teams that need private deployment paths for prompts, code, or
  regulated documents.

## What The Pilot Proves

- A TRACE endpoint can score your representative RAG or agent workflow.
- Runtime decisions are actionable: allow, repair/retry, review, or block.
- PII redaction works before prompts/responses leave the intended boundary.
- The right deployment path is clear: Cloud evaluation, Dedicated, VPC, or
  on-prem.

## Qualification Questions

1. Is your primary use case RAG, coding agents, or both?
2. What data would pass through TRACE?
3. Is Cloud evaluation acceptable, or do you need VPC/on-prem from day one?
4. How many traces per day do you expect?
5. What context size do you typically send?
6. What decision should TRACE trigger when support is weak?
7. Do you need audit export, SSO, retention controls, or air-gap support?
8. What would make the pilot successful after two weeks?

## Demo Script

1. Show a grounded RAG answer and its green decision.
2. Show an unsupported claim highlighted as amber/red.
3. Show context utilization so the buyer sees retrieval waste.
4. Show a code-agent turn with a phantom or drift signal.
5. Show PII redaction on a prompt before forwarding.
6. Close with the deployment choice: Cloud evaluation now, VPC/on-prem when
   sensitive data enters the loop.

## Security Posture

- Cloud evaluation is for low-risk data.
- Sensitive production data should use Dedicated, VPC, or on-prem.
- Compliance analytics must never store raw request text, entity text,
  replacement values, or redacted output.
- TRACE is not marketed as SOC 2 or ISO certified unless certification exists.
- TRACE verifies support against supplied context; it is not open-domain truth.

## Follow-Up Assets

- `docs/trace_pilot_path.md`
- `docs/sdk_gateway_minimal_contract.md`
- `docs/locked_two_week_sprint.md`
- `commercial/pilot-kit/onboarding-checklist.md`
- `commercial/pilot-kit/pilot-success-criteria-template.md`
