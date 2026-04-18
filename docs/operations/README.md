# Operations Documentation

Documentation for engineers who **deploy and run** `latence-trace` --
not for application developers integrating it.

| Document | When you need it |
|---|---|
| [`deployment-checklist.md`](deployment-checklist.md) | Before flipping a namespace to production |
| [`calibration-runbook.md`](calibration-runbook.md) | First week of every customer onboarding, then quarterly |
| [`observability.md`](observability.md) | Setting up Grafana / Datadog / OTel |
| [`licensing.md`](licensing.md) | Receiving, validating, rotating the JWT |

Operator surface map:

```
HTTP API   /groundedness  /metrics  /healthz  /readyz  /agent-help
CLI        latence-trace  serve | score | calibrate | warm | bench | license | mcp-server
SDK        latence-trace-client  (Python, sync + async)
Helm       deploy/helm/latence-trace/  (chart 1.0.0)
Container  ghcr.io/latence-ai/latence-trace:1.0.0  (uid 65532, RO root)
```

## Cross-references

| Looking for | Go to |
|---|---|
| Vulnerability disclosure + threat model + supply chain | [`../../SECURITY.md`](../../SECURITY.md) |
| Procurement / TPRM artefacts (security one-pager, SOC 2 mapping, MSA, DPA, Trust Center) | [`../../commercial/`](../../commercial/) |
| Pilot agreement, success criteria, ROI calculator, case-study template | [`../../commercial/pilot-kit/`](../../commercial/pilot-kit/) |
| Helm chart values + templates | [`../../deploy/helm/latence-trace/`](../../deploy/helm/latence-trace/) |
| Python SDK | [`../../clients/python/`](../../clients/python/) |
