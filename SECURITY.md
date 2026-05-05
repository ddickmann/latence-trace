# Security Policy

`latence-trace` is a commercial product operated as a self-hosted
sidecar inside our customers' infrastructure. We treat security as a
hard requirement, not a feature.

## Reporting a vulnerability

Email **`security@latence.ai`** with:

- A precise description of the vulnerability and its impact.
- Steps to reproduce (PoC, sample request, expected vs actual).
- Affected version(s) and the deployment topology you tested
  (container, Helm chart, source).
- Your name / handle for the security advisory credit (optional).

PGP key: `https://latence.ai/.well-known/security-pgp.asc`

We will:

1. Acknowledge receipt within **2 business days**.
2. Triage and assign a severity (CVSS v4) within **5 business days**.
3. Provide a fix or mitigation within:
   - **7 days** for Critical (CVSS >= 9.0).
   - **30 days** for High (7.0 - 8.9).
   - **90 days** for Medium / Low.
4. Coordinate a disclosure schedule and credit the reporter in the
   advisory unless you ask us not to.

We follow a **90-day coordinated disclosure** policy. We will not
take legal action against good-faith research that respects this
window and the scope below.

## In scope

- The published Docker image (`ghcr.io/latence-ai/latence-trace:*`).
- The published Helm chart (`deploy/helm/latence-trace`).
- The published Python SDK (`latence`, maintained in `latence-trace-python`).
- The HTTP API surface (`/groundedness`, `/healthz`, `/readyz`,
  `/metrics`, `/agent-help`, `/.well-known/ai-plugin.json`).
- The license enforcement, rate limiter, and request middleware.

## Out of scope

- Findings that require a privileged role inside the customer cluster
  (you already had root).
- Denial of service through legitimate load (use the rate limiter).
- Findings against `*.latence.ai` marketing surfaces (please report
  those to the marketing site separately).
- Issues in the upstream models we depend on (HF transformers,
  pylate, Triton). We will route those upstream when applicable.

## Hardening posture (image + chart)

| Control | Posture |
|---|---|
| Container user | uid/gid 65532, no shell, no package manager in runtime layer |
| Root filesystem | `readOnlyRootFilesystem=true` (Helm default) |
| Capabilities | All capabilities dropped; `seccompProfile: RuntimeDefault` |
| Service account | `automountServiceAccountToken=false` |
| Secrets | License Secret mounted with `mode: 0400`; never logged |
| TLS | terminated at the operator's gateway; chart never opens 80 |
| Egress | NetworkPolicy allows DNS, HTTPS, vLLM-Factory port only |
| SBOM | published per release (`syft -o spdx-json`) |
| Image scanning | `trivy` + `grype` gates on every release tag |
| License enforcement | hard-fail on missing/expired/invalid token (HTTP 402) |
| Rate limiting | per-license token bucket, burst capped, 429 + Retry-After |

## Cryptography

- License JWTs: **EdDSA / Ed25519** signature, validated against the
  public key shipped at `latence_trace/auth/license_pubkey.pem` (or
  `LATENCE_TRACE_LICENSE_PUBKEY[_FILE]`).
- TLS for outbound calls: enforced by the operator's gateway and
  cluster CA bundle (we ship `ca-certificates`).
- Inbound TLS: terminated by the operator's gateway; the API itself
  is HTTP-only inside the pod network.
- No plaintext secrets in env vars or logs; license tokens are
  redacted from every observable log line.

## Supply chain

- Pinned base image (`python:3.11-slim` + `tini`).
- Multi-stage build: compilers and `git` only in the build stage.
- All Python dependencies pinned by minimum version in
  `pyproject.toml`; reproducible wheels via `--no-cache-dir` +
  `pip install .`.
- Releases are built in CI; the resulting image is signed and
  attested per Sigstore.

## Data handling

- The runtime is **stateless** -- no scoring inputs are persisted
  outside the request lifecycle.
- Logs include score values and risk band but redact request bodies
  by default; operators can opt into request-body logging via env.
- Telemetry export is disabled unless the operator sets
  `OTEL_EXPORTER_OTLP_ENDPOINT`.

For the full data processing breakdown see
[`commercial/data-processing-addendum.md`](commercial/data-processing-addendum.md).
