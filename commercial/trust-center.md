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
| Last security review | <YYYY-MM-DD> -- internal architecture review |
| Next external pen-test | <YYYY-MM-DD> -- scheduled |

## Compliance program

| Framework | Status | Notes |
|---|---|---|
| SOC 2 Type II | Readiness assessment complete (see [`soc2-control-mapping.md`](soc2-control-mapping.md)). External Type I audit scheduled. | Type II report available under NDA once issued. |
| ISO 27001 | Gap assessment complete; certification planned for next fiscal year. | -- |
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
| <CRM provider> | Support ticket storage | <region> | DPA + SCCs |
| <Email provider> | Support email | <region> | DPA + SCCs |
| <Billing provider> | Invoice + payment processing | <region> | DPA |
| <Backup provider> | Encrypted backup of support correspondence | <region> | DPA + SCCs |

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
