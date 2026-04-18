# Licensing

`latence-trace` is a commercially licensed product. Every production
deployment validates a customer-issued JWT signed with the latence.ai
Ed25519 signing key.

## Receiving your license

1. Contact `support@latence.ai` with your customer name, contact, and
   environment count (prod / staging / DR).
2. Optionally include a deployment fingerprint to pin the license to
   a specific cluster::

       latence-trace license fingerprint --seed "$(cat /etc/machine-id)-prod"

3. We send back a JWT (`<sub>.jwt`). Treat it as a secret -- it grants
   service usage for the duration of the licensed term.

## Storing the license

Recommended: a Kubernetes Secret managed by your secret store::

    kubectl create secret generic lt-license --from-file=license=acme-corp.jwt
    helm install lt latence-ai/latence-trace --set license.existingSecret=lt-license

Other supported locations:

| Location | Use |
|---|---|
| `LATENCE_TRACE_LICENSE` env var (raw JWT) | quick CI / dev |
| `LATENCE_TRACE_LICENSE_FILE` env var (path) | container with bind-mount |
| `/etc/latence-trace/license` | system-wide install |
| `~/.latence-trace/license` | per-user dev install |

## Validating a token locally

```bash
$ latence-trace license inspect
{
  "subject": "acme-corp",
  "tier": "enterprise",
  "issued_at": 1742160000,
  "expires_at": 1773696000,
  "features": ["nli", "reranker", "atomic_claims"],
  "profiles": ["fast", "balanced", "quality"],
  "max_qps": 50,
  "max_workers": 4,
  "fingerprint": null,
  "customer": {"name": "Acme Corp", "contact": "ops@acme.test"},
  "days_until_expiry": 365.0,
  "expiring_soon": false
}
```

`latence-trace license verify` exits 0 / non-zero so it is safe to
chain into a CI gate.

## Rotation

Token rotation is a rolling restart::

    kubectl create secret generic lt-license-new --from-file=license=acme-corp.v2.jwt
    helm upgrade lt latence-ai/latence-trace --reuse-values \
      --set license.existingSecret=lt-license-new
    kubectl rollout status deploy/lt-latence-trace
    kubectl delete secret lt-license

The new pods boot, validate the new JWT, log
`license_loaded` with the new `expires_at`, and start serving. The
old pods stop after the rolling update.

## Expiry alerting

`latence_trace_license_days_until_expiry{subject="acme-corp"}` is the
canonical alert source::

    - alert: LatenceTraceLicenseExpiring
      expr: latence_trace_license_days_until_expiry < 30
      for: 1h
      labels: { severity: warning }

The middleware also emits a structured warning log
(`license_expiring_soon`) on every startup once you cross the 30-day
mark; pipe those into your log alert pipeline as a backup.

## What happens when a license expires

- All requests to `/groundedness` return **HTTP 402** with
  `code: license_expired`.
- `/healthz` and `/readyz` keep returning 200 so the operator can SSH
  into the pod, fix the env, and restart deterministically.
- `/metrics` keeps publishing -- the Grafana board shows traffic
  going to zero with all 402 responses, which is exactly the alert
  signature you want.

## Disabling enforcement (CI / dev)

Set `LATENCE_TRACE_LICENSE_REQUIRE=false` (chart values:
`license.required=false`). The middleware logs a single
`license_enforcement_disabled` warning at startup so production
deploys notice if this gets accidentally shipped.
