# latence-trace Helm chart

Production Helm chart for the `latence-trace` Groundedness Tracker
sidecar. Targets Kubernetes 1.27+.

## TL;DR

```bash
helm repo add latence-ai https://charts.latence.ai
helm install lt latence-ai/latence-trace \
  --set license.token="$(cat license.jwt)" \
  --set image.tag=1.0.0
```

## What ships

| Resource | Purpose |
|---|---|
| `Deployment` | non-root, read-only-root-fs, GPU-ready pod template |
| `Service` (ClusterIP) | port `8090` |
| `ConfigMap` | profile, rate limit, OTel endpoint, encoder URL |
| `Secret` | optional inline license JWT (use `existingSecret` for Vault/External Secrets) |
| `HPA` | CPU + memory autoscaler (off by default) |
| `PodDisruptionBudget` | `minAvailable=1` so cluster ops can't drain everything |
| `NetworkPolicy` | default-deny ingress, allow-DNS-and-HTTPS egress (off by default) |
| `ServiceMonitor` | kube-prometheus-stack scrape config (off by default) |
| `Ingress` | optional TLS-terminating ingress (off by default) |
| `ServiceAccount` | scoped, no token mount |

## Required values

| Key | Notes |
|---|---|
| `image.tag` | pin to a digest in production |
| `license.token` _or_ `license.existingSecret` | customer JWT |
| `resources.{requests,limits}` | size after a load test |

## Recommended values for prod

```yaml
replicaCount: 3
profile: balanced
license:
  existingSecret: lt-license
  required: true

observability:
  logFormat: json
  otelEndpoint: http://otel-collector.observability:4318

resources:
  requests: { cpu: "2000m", memory: "8Gi" }
  limits:   { cpu: "4000m", memory: "16Gi", "nvidia.com/gpu": 1 }

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 12

pdb: { enabled: true, minAvailable: 2 }

networkPolicy:
  enabled: true
  ingressFrom:
    - namespaceSelector:
        matchLabels:
          name: rag-platform

serviceMonitor:
  enabled: true
  labels:
    release: kube-prometheus-stack
```

## Probes

- Liveness: `GET /healthz` -- 200 as soon as the process is alive.
- Readiness: `GET /readyz` -- 200 only after the Triton kernel JIT
  warmup completes (and, when enforcement is on, the license is valid).

## Security posture

- `runAsNonRoot=true`, uid/gid `65532`, no shells in the runtime layer.
- `readOnlyRootFilesystem=true`; HF cache mounts as `emptyDir` or PVC.
- `automountServiceAccountToken=false`.
- All capabilities dropped, `seccompProfile: RuntimeDefault`.
- License Secret mounted with `mode: 0400`.
