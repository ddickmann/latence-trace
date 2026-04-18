# Audit Evidence Inventory

> Mapping from every control in `soc2-control-mapping.md` to a
> concrete artefact (file, configuration, test, runbook, or report).
> Auditors should be able to trace any control to its evidence in one
> hop.
>
> Items marked `[private]` live in Provider's internal compliance
> repository; they are made available under NDA on request.

## Source-tree evidence (this repository)

| Control | Evidence path | What the auditor verifies |
|---|---|---|
| CC2.1 | `SECURITY.md` | Disclosure channel, severities, hardening posture |
| CC2.2 | `commercial/trust-center.md` | Public Trust Center copy |
| CC4.1 | `docs/operations/observability.md` | Prometheus metric names + alert templates |
| CC4.2 | `docs/operations/observability.md` | Log schema + correlation IDs |
| CC5.1 | `.github/CODEOWNERS`, `.github/pull_request_template.md` | Review and approval requirements |
| CC5.2 | `.github/workflows/release.yml` | Clean-checkout build, tagged release |
| CC5.3 | `.github/workflows/release.yml` (cosign step) | Image signing |
| CC5.4 | `.github/workflows/release.yml` (syft step) | SBOM generation |
| CC6.1 | `latence_trace/auth/license.py`, `tests/test_license.py` | License enforcement + tests |
| CC6.2 | `Dockerfile` (final stage) | Non-root, no shell, no package manager |
| CC6.3 | `deploy/helm/latence-trace/values.yaml` (`securityContext`) | Read-only root FS, dropped caps, seccomp |
| CC6.4 | `deploy/helm/latence-trace/templates/networkpolicy.yaml` | Default-deny + explicit egress |
| CC6.5 | `deploy/helm/latence-trace/templates/serviceaccount.yaml` | `automountServiceAccountToken: false` |
| CC6.6 | `deploy/helm/latence-trace/templates/secret.yaml`, `latence_trace/observability/logging.py` | Secret mount mode + log redaction |
| CC7.1 | `Dockerfile` HEALTHCHECK, `templates/deployment.yaml` probes | Liveness + readiness probes |
| CC7.2 | `templates/hpa.yaml` | HPA configuration |
| CC7.3 | `templates/pdb.yaml` | PDB configuration |
| CC7.4 | `latence_trace/middleware/rate_limit.py`, `tests/test_rate_limit.py` | Rate limiter + tests |
| CC7.5 | `server/main.py` (inflight semaphore) | Inflight concurrency bound |
| CC8.1 | `CHANGELOG.md`, git tags | Versioned releases |
| CC8.3 | `deploy/helm/latence-trace/Chart.yaml` | Independent chart versioning |
| CC9.1 | `commercial/trust-center.md` (sub-processors table) | Sub-processor inventory |
| C1.5 | `latence_trace/observability/logging.py` | Body-redaction logic |
| PI1.1 | `docs/operations/calibration-runbook.md` | Calibration workflow |
| PI1.2 | `tests/`, `coverage.xml` | Test coverage gate |
| PI1.3 | `docs/operations/calibration-runbook.md`, README "Risk bands" | Threshold rationale |
| PI1.4 | `commercial/audit-evidence-inventory.md` (this file) | Algorithm audit summary (below) |

## Provider-side evidence (private)

| Control | Evidence | Holder |
|---|---|---|
| CC1.1 | Code of conduct + acknowledgements `[private]` | Provider HR |
| CC1.2 | Role descriptions `[private]` | Provider HR |
| CC1.3 | Background-check vendor reports `[private]` | Provider HR |
| CC1.4 | Security training records `[private]` | Provider Security |
| CC2.3 | Incident notification runbook `[private]` | Provider Security |
| CC3.1 | Annual risk register `[private]` | Provider Security |
| CC4.4 | CVE-feed monitoring + advisory archive `[private]` | Provider Security |
| CC6.7 | GitHub org SSO + MFA settings export `[private]` | Provider IT |
| CC7.6 | Incident response runbook `[private]` | Provider Engineering |
| CC7.7 | Post-incident reviews `[private]` | Provider Engineering |
| CC9.2 | Sub-processor DPAs `[private]` | Provider Legal |
| CC9.3 | External pen-test reports `[private]` | Provider Security |
| A1.3 | Status page operator agreement `[private]` | Provider Operations |
| A1.4 | Backup configuration evidence `[private]` | Provider IT |

## Algorithmic audit (PI1.4)

The scoring stack is layered + auditable:

1. **Per-token similarity** -- ColBERT-style late interaction, with
   the SauerkrautLM-Multi-Reason-ModernColBERT model as the German +
   English default. Verified against the original ColBERT paper's
   cosine semantics.
2. **Reverse cross-context (CC) consensus** -- counts the fraction
   of context chunks that vote a response chunk as grounded. Reduces
   single-chunk false positives. Implementation:
   `latence_trace/scoring/consensus.py`.
3. **NLI fallback** -- DeBERTa-v3-large NLI used to break ties on
   amber-band outputs. Backstop against semantically-similar but
   factually-different rewrites. Implementation:
   `latence_trace/scoring/nli.py`.
4. **Risk bands** -- thresholds documented in
   `docs/operations/calibration-runbook.md`. Per-customer refit is a
   one-time first-week task with a deterministic CLI.

Audit trail per request (when JSON logs enabled):

```json
{
  "level": "info",
  "logger": "latence_trace.scoring",
  "message": "groundedness_scored",
  "request_id": "<uuid>",
  "trace_id": "<otel-trace-id>",
  "score": 0.81,
  "band": "green",
  "components": {"cc": 0.83, "reverse_cc": 0.78, "nli": null},
  "n_chunks_context": 5,
  "n_chunks_response": 2
}
```

## Refresh cadence

| Artefact | Owner | Cadence |
|---|---|---|
| SBOM | Provider release pipeline | Per release tag |
| Image signature | Provider release pipeline | Per release tag |
| Trivy + Grype scans | Provider release pipeline | Per release tag + nightly |
| Risk register | Provider Security | Annual + ad-hoc |
| Pen-test report | External vendor | Annual |
| SOC 2 report | External auditor | Annual (Type II) |
| Trust Center page | Provider Operations | Per change |
| Sub-processor list | Provider Operations | Per change + 30-day notice |
