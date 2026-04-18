# SOC 2 Control Mapping (Trust Services Criteria)

> Maps the AICPA Trust Services Criteria (TSC) -- Security,
> Availability, Confidentiality, Processing Integrity -- to concrete
> controls implemented by the latence-trace runtime, the official
> Helm chart, and Provider's operating practices.
>
> This document is the basis of Provider's SOC 2 readiness program
> and the artefact a customer's third-party risk team should request
> for procurement review. The corresponding evidence is catalogued in
> `audit-evidence-inventory.md`.

## Legend

- **Control owner**: who runs the control. `Provider` = latence.ai
  (development, release, support). `Customer` = operator of the
  self-hosted deployment. `Joint` = shared responsibility model.
- **Status**: `Implemented`, `In progress`, or `Planned`. Anything
  marked `In progress` / `Planned` is also reflected in the public
  Trust Center page.

## Common Criteria (CC) -- Security

### CC1 -- Control Environment

| ID | Control | Owner | Evidence | Status |
|---|---|---|---|---|
| CC1.1 | Code of conduct + acceptable-use policy for engineering staff | Provider | HR policy, signed acknowledgements | Implemented |
| CC1.2 | Security responsibilities documented in role descriptions | Provider | Org chart, role descriptions | Implemented |
| CC1.3 | Background checks for all engineering staff handling customer data | Provider | Vendor reports | Implemented |
| CC1.4 | Annual security training (incl. phishing) | Provider | Training records | Implemented |

### CC2 -- Communication and Information

| ID | Control | Owner | Evidence | Status |
|---|---|---|---|---|
| CC2.1 | Public security policy (`SECURITY.md`) with disclosure channel | Provider | `SECURITY.md`, PGP key URL | Implemented |
| CC2.2 | Trust Center page with sub-processors, certifications, status | Provider | `trust-center.md`, status URL | Implemented |
| CC2.3 | Customer notification within 72h of confirmed breach | Provider | DPA Section 4.5; runbook | Implemented |

### CC3 -- Risk Assessment

| ID | Control | Owner | Evidence | Status |
|---|---|---|---|---|
| CC3.1 | Annual risk assessment with documented threat model | Provider | Risk register | Implemented |
| CC3.2 | Threat model for the Software (STRIDE-based) | Provider | `SECURITY.md` threat model section | Implemented |
| CC3.3 | Vulnerability scanning per release (Trivy + Grype + Bandit) | Provider | CI logs | Implemented |

### CC4 -- Monitoring Activities

| ID | Control | Owner | Evidence | Status |
|---|---|---|---|---|
| CC4.1 | Prometheus metrics scraped + alerted on `latence_trace_*` series | Customer | `docs/operations/observability.md` | Implemented |
| CC4.2 | Centralised structured JSON logs with request IDs | Customer | `docs/operations/observability.md` | Implemented |
| CC4.3 | OTel tracing exported to customer's collector when enabled | Customer | OTel env vars in `values.yaml` | Implemented |
| CC4.4 | Provider monitors CVE feeds and publishes advisories | Provider | Advisory mailing list | Implemented |

### CC5 -- Control Activities

| ID | Control | Owner | Evidence | Status |
|---|---|---|---|---|
| CC5.1 | Code review on every PR; required approval | Provider | GitHub PR settings | Implemented |
| CC5.2 | All releases tagged + built in CI from a clean checkout | Provider | CI workflow file | Implemented |
| CC5.3 | Container images signed with Sigstore cosign | Provider | Cosign signatures | Implemented |
| CC5.4 | SBOM published per release | Provider | `syft` output per release | Implemented |

### CC6 -- Logical and Physical Access

| ID | Control | Owner | Evidence | Status |
|---|---|---|---|---|
| CC6.1 | License JWT (Ed25519) gates every request to the engine | Provider | `latence_trace/auth/license.py`, `tests/test_license.py` | Implemented |
| CC6.2 | Container runs as uid/gid 65532, no shell, no package manager | Provider | `Dockerfile` (final stage) | Implemented |
| CC6.3 | `readOnlyRootFilesystem=true`, all caps dropped, seccomp `RuntimeDefault` | Joint | Helm `values.yaml` defaults; chart enforces | Implemented |
| CC6.4 | NetworkPolicy restricts ingress + egress | Joint | `templates/networkpolicy.yaml` | Implemented |
| CC6.5 | Service account token not auto-mounted | Joint | `templates/serviceaccount.yaml` | Implemented |
| CC6.6 | License Secret mounted `mode=0400`; never logged | Joint | `templates/secret.yaml`, redaction logic | Implemented |
| CC6.7 | SSO + MFA for Provider's GitHub org and CI | Provider | GitHub org settings | Implemented |

### CC7 -- System Operations

| ID | Control | Owner | Evidence | Status |
|---|---|---|---|---|
| CC7.1 | Liveness + readiness probes (`/healthz`, `/readyz`) | Joint | `Dockerfile` HEALTHCHECK; chart probes | Implemented |
| CC7.2 | HPA scales on CPU + custom metrics | Customer | `templates/hpa.yaml` | Implemented |
| CC7.3 | PDB protects against voluntary disruptions | Customer | `templates/pdb.yaml` | Implemented |
| CC7.4 | Rate limiter prevents resource exhaustion | Joint | `latence_trace/middleware/rate_limit.py` | Implemented |
| CC7.5 | Inflight semaphore bounds concurrent work | Joint | `server/main.py` middleware | Implemented |
| CC7.6 | Incident response runbook (severity, paging, comms) | Provider | Runbook (private) | Implemented |
| CC7.7 | Post-incident review + corrective action tracking | Provider | Internal tracker | Implemented |

### CC8 -- Change Management

| ID | Control | Owner | Evidence | Status |
|---|---|---|---|---|
| CC8.1 | Versioned releases; semantic versioning; CHANGELOG | Provider | Git tags, CHANGELOG | Implemented |
| CC8.2 | Customer-facing breaking changes communicated 30d in advance | Provider | Release notes | Implemented |
| CC8.3 | Helm chart is versioned independently and signed | Provider | Chart `Chart.yaml`, provenance | Implemented |

### CC9 -- Risk Mitigation

| ID | Control | Owner | Evidence | Status |
|---|---|---|---|---|
| CC9.1 | Vendor / sub-processor list maintained in Trust Center | Provider | `trust-center.md` | Implemented |
| CC9.2 | DPA signed with each sub-processor | Provider | DPA repository | Implemented |
| CC9.3 | Annual penetration test by external party | Provider | Pen-test report (under NDA) | In progress |

## Availability (A)

| ID | Control | Owner | Evidence | Status |
|---|---|---|---|---|
| A1.1 | HPA + PDB + multiple replicas (chart default min 2) | Customer | `values.yaml` | Implemented |
| A1.2 | Documented SLOs (P95 latency, error rate, saturation) | Provider | `docs/operations/observability.md` | Implemented |
| A1.3 | Status page for Provider-side dependencies (CRM, billing) | Provider | `https://status.latence.ai` | Implemented |
| A1.4 | Backup of Provider-side artefacts (release tarballs, SBOMs) | Provider | Object storage with versioning | Implemented |

## Confidentiality (C)

| ID | Control | Owner | Evidence | Status |
|---|---|---|---|---|
| C1.1 | Customer Data never leaves Customer perimeter under standard deployment | Joint | `SECURITY.md`, DPA | Implemented |
| C1.2 | License JWT is the only Customer identifier persisted by Provider | Provider | License pipeline | Implemented |
| C1.3 | Encrypted in transit (TLS terminated at Customer gateway) | Customer | `docs/operations/deployment-checklist.md` | Implemented |
| C1.4 | Encrypted at rest -- N/A (engine is stateless) | Provider | Architecture | Implemented |
| C1.5 | Logs redact request bodies by default | Joint | `latence_trace/observability/logging.py` | Implemented |

## Processing Integrity (PI)

| ID | Control | Owner | Evidence | Status |
|---|---|---|---|---|
| PI1.1 | Calibration runbook for per-customer threshold refit | Joint | `docs/operations/calibration-runbook.md` | Implemented |
| PI1.2 | Validation tests for every kernel + scoring path (>= 90% coverage on scoring code) | Provider | `tests/`, CI coverage gate | Implemented |
| PI1.3 | Risk band labels documented + thresholds explainable | Provider | `docs/operations/calibration-runbook.md` | Implemented |
| PI1.4 | Algorithmic audit of consensus logic (CC + reverse-CC + NLI) | Provider | `audit-evidence-inventory.md` | Implemented |

## Privacy (Px) -- referenced via DPA

We do not currently certify the Privacy criterion separately because
GDPR + DPA cover the relevant obligations. The mapping is documented
in the DPA (Annex II).
