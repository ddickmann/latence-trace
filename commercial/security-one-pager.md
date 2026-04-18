# latence-trace Security One-Pager

> Self-hosted groundedness scoring sidecar for RAG and evidence-bearing
> LLM outputs. v1.0 GA, commercial license, deployed inside the
> customer's infrastructure perimeter.

## At a glance

| Topic | Answer |
|---|---|
| Deployment model | Self-hosted in customer infra (container or Helm chart). No multi-tenant SaaS. |
| Data residency | All processing happens in the customer cluster; nothing leaves the perimeter. |
| Customer data persistence | None. Stateless engine; nothing is written to disk outside transient caches. |
| Telemetry to vendor | Disabled by default. Optional OTel export goes to the customer's collector. |
| Authentication | Ed25519-signed JWT license per deployment. |
| Encryption in transit | TLS terminated at the customer's gateway. |
| Encryption at rest | N/A -- the engine has no persistent state. Caches are ephemeral container volumes. |
| Logging | Structured JSON; request bodies redacted by default. |
| Vulnerability disclosure | `security@latence.ai`, 90-day coordinated disclosure. |
| SBOM | Published per release (`spdx-json`). |
| SOC 2 readiness | Control mapping documented (see `soc2-control-mapping.md`). |
| GDPR | DPA template included; engine processes no PII unless the customer's RAG inputs do. |

## Threat model summary

| Threat | Mitigation |
|---|---|
| Unauthorised use of the binary | JWT license enforcement, hard-fail on missing/expired/invalid token. |
| Container escape | uid/gid 65532, no shells, all caps dropped, `readOnlyRootFilesystem`, seccomp `RuntimeDefault`. |
| Lateral movement | NetworkPolicy default-deny, egress restricted to DNS+HTTPS+vLLM. ServiceAccount token not mounted. |
| Secret exfiltration | License Secret mounted `mode=0400`; never logged; redaction list documented. |
| Resource exhaustion / abuse | Inflight semaphore, token-bucket rate limiter (per-license, with `Retry-After`). |
| Supply chain | Pinned base image, multi-stage build, dependency pinning, image signing, SBOM. |
| Model exfiltration | Models loaded from licensed weights; vLLM-Factory sidecar isolated by NetworkPolicy. |

## What we do NOT do

- We do not store, transmit, or process customer prompts or context
  outside the customer cluster.
- We do not exfiltrate telemetry. The default OTel + log destinations
  are unset; the customer points them at their own collector.
- We do not require a connection to `latence.ai` at runtime. License
  validation is offline (Ed25519 signature against a bundled public
  key).

## What we ask the customer to do

- Front the API with a TLS-terminating gateway.
- Manage the license JWT through their secret store of choice
  (Vault, External Secrets, Sealed Secrets).
- Restrict ingress with the chart's NetworkPolicy or their service
  mesh policy.
- Enable Prometheus scraping + alerting on the documented metrics.

## Contact

- Security disclosures: `security@latence.ai` ([SECURITY.md](../SECURITY.md))
- Procurement / TPRM: `compliance@latence.ai`
- Engineering escalation: `support@latence.ai`
