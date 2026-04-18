# Production Deployment Checklist

Print this. Tick every box before flipping a `latence-trace` namespace
to "production" (i.e. customer-facing).

## Pre-flight

- [ ] Pulled the image by **digest**, not floating tag, in the values
      file (`image.tag: sha256:...`).
- [ ] License JWT issued by `support@latence.ai` and stored in a
      **Secret** (Vault / External Secrets / Sealed Secrets), not
      checked into Git.
- [ ] `LATENCE_TRACE_LICENSE_REQUIRE=true` in the values file.
- [ ] Profile pinned (`profile: balanced` for general RAG; `quality`
      for high-stakes / compliance traffic).
- [ ] CPU and memory `requests` + `limits` sized after a load test on
      a single replica matching production hardware.
- [ ] HPA `minReplicas >= 2` so a pod restart never zeroes capacity.
- [ ] PDB `minAvailable >= 1`.

## Networking

- [ ] Ingress fronted by a **TLS-terminating proxy** (your gateway,
      Istio, NGINX, ALB, ...). Service stays `ClusterIP`.
- [ ] NetworkPolicy enabled, ingress restricted to the namespaces /
      pod selectors that should call the API.
- [ ] Egress restricted to: DNS (53), HTTPS (443), vLLM-Factory port.

## Identity & secrets

- [ ] ServiceAccount has `automountServiceAccountToken: false`
      (chart default).
- [ ] License Secret mounted with `mode: 0400` (chart default).
- [ ] No secrets in env vars; everything sensitive flows through
      Secrets and ConfigMaps.

## Observability

- [ ] `LATENCE_TRACE_LOG_FORMAT=json` (chart default).
- [ ] ServiceMonitor enabled and scraped by Prometheus.
- [ ] Grafana dashboard imported (see `docs/operations/observability.md`).
- [ ] Alerts configured (latency p95, error rate, license expiry,
      rate-limit floor, profile drift).
- [ ] OTel exporter pointed at your collector if you want traces.

## Calibration

- [ ] Per-customer threshold refit run **at least once** before
      go-live (`docs/operations/calibration-runbook.md`).
- [ ] Calibration JSON checked into your config repo.
- [ ] Quarterly recalibration scheduled.

## Disaster recovery

- [ ] License renewal date in the team calendar (alert at -30 days).
- [ ] Runbook on how to rotate the license token (write a
      `kubectl create secret generic ... --dry-run=client -o yaml |
      kubectl apply -f -` example).
- [ ] Failover plan: the engine is stateless. Roll out via the same
      Helm chart in a second cluster; flip your gateway when ready.

## Security review

- [ ] `containerSecurityContext.readOnlyRootFilesystem=true`
      (chart default).
- [ ] No privileged containers; all caps dropped (chart default).
- [ ] CIS Kubernetes benchmark scan passes (`kube-bench`).
- [ ] Image scanned for CVEs in CI (`trivy`, `grype`).
- [ ] SBOM generated and stored (`syft packages docker:lt:<tag> -o spdx-json`).

## Sign-off

| Role | Name | Date |
|---|---|---|
| Platform engineering |   |   |
| Security |   |   |
| Application owner |   |   |
| Compliance |   |   |
