# Evidence index for CAIQ-Lite + SIG-Lite

Each question ID on either questionnaire maps to the evidence it
relies on.  Change this file every time the underlying evidence
artifact changes so prospects and auditors can verify freshness in
one hop.

| Q ID | Artifact | Freshness anchor |
|---|---|---|
| AIS-01 | `commercial/trust-center/policies/information-security.md` | git log -1 on that file |
| AIS-02 | `commercial/trust-center/policies/risk-register.md` | git log -1 on that file |
| BCR-01 | `docs/operations/incident-response.md` | Last DR test note in sec 7 |
| CCC-01 | GitHub branch-protection config + `docs/operations/release.md` | Screenshot refreshed each quarter |
| CEK-01 | `commercial/trust-center/crypto-architecture.md` | Last KMS rotation date |
| DCS-01 | `commercial/trust-center/subprocessors.md` | Sub-processor notification ledger |
| DSP-01 | `commercial/hosted-dpa.md` | Legal sign-off date in YAML frontmatter |
| DSP-02 | `commercial/trust-center/gdpr-readiness.md` | Review schedule line in footer |
| HRS-01 | Internal HR policy (Trust Center read-only) | HR sign-off line |
| IAM-01 | `commercial/trust-center/iam.md` | Latest IdP config change |
| IVS-01 | `docs/operations/observability.md` security section | Scanner version tag |
| IPY-01 | `latence_trace/middleware/audit_log.py` + portal export API | `CHANGELOG.md` row |
| LOG-01 | `docs/operations/audit-log.md` | Log retention policy version |
| SEF-01 | `docs/operations/incident-response.md` | Last tabletop exercise note |
| STA-01 | `commercial/trust-center/subprocessors.md` | Version history at top |
| TVM-01 | `commercial/trust-center/pentest-roadmap.md` | Scheduled test date |
| UEM-01 | `commercial/trust-center/endpoint-policy.md` | MDM enrollment report snapshot |
| B.3 | `latence_trace/middleware/audit_log.py` | Feature flag default (off for free tier) |
| C.3 | `gateway/cloudflare/src/index.ts` (`handleKeyRotate`) | Latest commit on that file |
| D.3 | `docs/operations/incident-response.md` sec 6 | Last DR test outcome |
| F.1 | `docs/operations/observability.md` | Log retention clause |

Supporting evidence references to bold-claim rows

| Bold claim row | Questionnaire row backed |
|---|---|
| 1, 2, 3 | n/a (product efficacy, covered by `proof_bundle_v1/*`) |
| 10 (no-persist hot path) | B.3, LOG-01 |
| 11 (per-tenant thresholds) | C.3 |
| 12 (audit log) | IPY-01, LOG-01, F.1 |
| 15 (status + changelog) | D.4 |
