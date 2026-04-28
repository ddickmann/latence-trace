# Trust Center

> Public-facing summary of Provider's security, compliance, and
> reliability posture for the latence-trace Groundedness Tracker. The
> machine-readable version of this page powers
> [`https://latence.ai/trust`](https://latence.ai/trust).

## Snapshot

| Item | Value |
|---|---|
| Product | latence-trace Groundedness Tracker (self-hosted) |
| Deployment model | Customer-hosted container or Helm chart |
| Customer Data exfiltration | None by default |
| TLS | Terminated at Customer gateway (HTTP inside the pod) |
| License | Ed25519 JWT issued per deployment |
| Status page | [`https://status.latence.ai`](https://status.latence.ai) |
| Last security review | 2026-03-18 -- internal architecture review (scope: hosted + self-hosted) |
| Next external pen-test | 2026-06-02 -- Cure53 (black-box + authenticated); summary letter delivered under NDA |
| Last SOC 2 Type I audit | 2026-02-14 -- Prescient Assurance; letter available on request |
| Next SOC 2 Type II window | Window opened 2026-03-01; report expected 2026-11 |
| ISO 27001 status | Gap assessment 2026-01; certification target 2027-Q1 |

## Compliance program

| Framework | Status | Notes |
|---|---|---|
| SOC 2 Type I | **Letter issued 2026-02-14** (Prescient Assurance). | Available on request under NDA. |
| SOC 2 Type II | Type II observation window opened 2026-03-01. | Expected report 2026-11. |
| ISO 27001 | Gap assessment complete 2026-01; certification target 2027-Q1. | -- |
| GDPR | DPA template + Standard Contractual Clauses (Module 2) available. See [`data-processing-addendum.md`](data-processing-addendum.md). | Engine processes no Customer Personal Data outside the Customer's perimeter. |
| UK GDPR / DPA 2018 | Covered by DPA + UK International Data Transfer Addendum. | -- |
| HIPAA | The Software is not a HIPAA "covered entity" or "business associate" by default; available on request for healthcare customers under a BAA. | -- |
| PCI DSS | Out of scope -- no card data ever touches the Software. | -- |

## Security highlights

- **Zero customer data exfiltration.** The runtime is fully
  self-hosted and stateless; nothing leaves the Customer perimeter
  unless the Customer explicitly enables OTel export to their own
  collector.
- **Hard license enforcement.** Every request is gated by an
  Ed25519-signed JWT verified offline against the bundled public key.
- **Hardened container.** Non-root (uid/gid 65532), no shells, no
  package manager in the runtime layer, all capabilities dropped,
  read-only root filesystem, seccomp `RuntimeDefault`.
- **Hardened chart.** ServiceAccount token not auto-mounted, default-
  deny NetworkPolicy, PDB + HPA, ServiceMonitor for Prometheus.
- **Supply chain.** Pinned base image, multi-stage build, dependency
  pinning, SBOM (`spdx-json`) per release, Sigstore-signed images.
- **Documented threat model.** See `SECURITY.md`.

## Sub-processors

The Software runs **inside the Customer's infrastructure** -- there
are no Sub-processors that touch Customer Data in the standard
deployment.

For support correspondence and billing only:

| Sub-processor | Purpose | Region | Safeguard |
|---|---|---|---|
| Linear, Inc. | Support ticket storage | US (EU region on request) | DPA + SCCs |
| Google Workspace (Google Ireland Ltd) | Support email and documents | EU (Ireland) | DPA + SCCs |
| Stripe Payments Europe Ltd | Invoice + payment processing | Ireland | DPA |
| Backblaze Europe B.V. | Encrypted backup of support correspondence | EU (Amsterdam) | DPA + SCCs |

### Hosted SaaS sub-processors (api.latence.ai)

For customers on the hosted plan, additional sub-processors touch
Customer Data.  The full list is maintained in
[`hosted-data-processing-addendum.md`](hosted-data-processing-addendum.md)
§2 and updated at least 30 days before changes.

We notify all customers at least 30 days before any change to this
list.

## Documents available on request

- SOC 2 Type II report (once issued; under NDA)
- External penetration test summary (under NDA)
- Architecture diagrams (under NDA)
- Insurance certificates (cyber liability + E&O)
- ISO 27001 audit reports (once certified)

Email **`compliance@latence.ai`** to request access.

## Vulnerability disclosure

See [`SECURITY.md`](../SECURITY.md). Disclosures go to
**`security@latence.ai`** under a 90-day coordinated disclosure
policy.

## Subscribe to advisories

- Security mailing list: `security-advisories@latence.ai`
  (subscribe at `https://latence.ai/trust/subscribe`)
- RSS / Atom: `https://latence.ai/trust/feed.xml`
- Status page subscriptions: `https://status.latence.ai`
