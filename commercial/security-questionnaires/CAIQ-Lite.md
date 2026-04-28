# CAIQ-Lite (CSA) responses - Latence TRACE v1.1

Responses mapped to the [CSA CAIQ-Lite](https://cloudsecurityalliance.org/research/cai/)
v4 skeleton. Column `Evidence` links to the artifact under
`commercial/trust-center/` or `docs/operations/` that substantiates the
answer. Shortened to the 17 questions that prospects on pilot-size
deals ask first; the full CAIQ is on request under NDA.

| # | Question | Answer | Evidence |
|---|---|---|---|
| AIS-01 | Are information security policies documented and reviewed at least annually? | Yes | `commercial/trust-center/policies/information-security.md` (rev. Q1-2026) |
| AIS-02 | Is there a formal risk-assessment programme? | Yes | `commercial/trust-center/policies/risk-register.md` |
| BCR-01 | Is there a business continuity / disaster recovery plan tested annually? | Yes (RTO 4h, RPO 15min) | `docs/operations/incident-response.md` sections 6-7 |
| CCC-01 | Is change management documented with peer review? | Yes | Git PR workflow + `docs/operations/release.md` |
| CEK-01 | Are cryptographic keys managed with HSM-backed KMS? | Yes | Cloudflare KMS + RunPod secret store; `commercial/trust-center/crypto-architecture.md` |
| DCS-01 | Are data centres SOC 2 / ISO 27001 attested? | Yes - Cloudflare + RunPod | `commercial/trust-center/subprocessors.md` |
| DSP-01 | Is a Data Processing Agreement available? | Yes | `commercial/hosted-dpa.md` |
| DSP-02 | Is GDPR compliance audited? | Readiness review Q4-2026 | `commercial/trust-center/gdpr-readiness.md` |
| HRS-01 | Background checks for engineering staff? | Yes | Internal - redacted HR policy in Trust Center |
| IAM-01 | Is MFA required for all privileged access? | Yes | Cloud IdP + hardware keys; `commercial/trust-center/iam.md` |
| IVS-01 | Is infrastructure vulnerability scanned continuously? | Yes (Trivy + Wiz) | `docs/operations/observability.md` security section |
| IPY-01 | Can data be exported on request? | Yes (audit log export API + portal CSV) | `latence_trace/middleware/audit_log.py`, `latence_trace/middleware/amber_queue.py` |
| LOG-01 | Are audit logs tamper-evident? | Append-only + object-lock S3 mirror | `latence_trace/middleware/audit_log.py` + `docs/operations/audit-log.md` |
| SEF-01 | Is there a documented incident-response plan? | Yes | `docs/operations/incident-response.md` |
| STA-01 | Are sub-processors published? | Yes | `commercial/trust-center/subprocessors.md` |
| TVM-01 | Is a third-party pen test conducted annually? | First pen test scheduled Q3-2026 | `commercial/trust-center/pentest-roadmap.md` |
| UEM-01 | Endpoint-management policy for developer laptops? | Yes (MDM + disk encryption) | `commercial/trust-center/endpoint-policy.md` |

> **Known gap (honest):** TRACE v1.1 is not yet SOC 2 Type I attested.
> Auditor engagement letter with Prescient Assurance is signed; report
> ETA Q3-2026. Under NDA we will share the pentest engagement letter
> and the draft readiness letter.
