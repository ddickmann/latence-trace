# Grafana Dashboards

Importable Grafana dashboard JSONs committed alongside the observability
contract in `docs/operations/observability.md`.

## Files

| File | Dashboard | Notes |
| --- | --- | --- |
| `trace-hosted-reliability.json` | TRACE hosted reliability | Latency, band mix, error rate, rate-limit saturation. Import with `Dashboards > Import > Upload JSON`. |

## Datasource binding

All panels use the `${DS_PROMETHEUS}` variable; select your Prometheus
datasource at import time.

## Panels covered

- Score latency (p50 / p95 / p99) by profile.
- Band mix (green / amber / red share of total scores).
- Error rate.
- Rate limit saturation per tenant.

## Alert rules

The alert rules that sit alongside these panels are documented in
`docs/operations/observability.md#alert-rules` and tracked under
`commercial/hosted-sla.md`.  The incident runbook that references these
alerts is at `docs/operations/incident-response.md`.
