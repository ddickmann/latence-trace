# Observability Reference

## Metrics

`latence-trace` exposes Prometheus metrics on `GET /metrics`. The
ServiceMonitor in the Helm chart wires this to kube-prometheus-stack
out of the box.

| Metric | Type | Labels | Use |
|---|---|---|---|
| `latence_trace_requests_total` | counter | `route, method, status, license_subject` | RPS + error rate |
| `latence_trace_request_duration_seconds` | histogram | `route, method` | p50/p95/p99 latency |
| `latence_trace_request_inflight` | gauge | `route` | concurrency saturation |
| `latence_trace_encode_seconds` | histogram | `stage` | encoder bottleneck |
| `latence_trace_score_seconds` | histogram | `profile` | scoring path latency |
| `latence_trace_nli_seconds` | histogram | `model` | NLI-only latency |
| `latence_trace_score` | histogram | `kind` | drift detection on the score distribution |
| `latence_trace_active_profile` | gauge | `profile` | catch unexpected profile downgrades |
| `latence_trace_license_days_until_expiry` | gauge | `subject, tier` | renewal alerting |
| `latence_trace_build_info` | info | `version, profile, torch_version` | correlate with deploys |

### Alerts to set on day 1

```promql
# p95 latency above SLO
histogram_quantile(
  0.95,
  rate(latence_trace_request_duration_seconds_bucket{route="/groundedness"}[5m])
) > 0.250

# error rate above 0.5%
sum(rate(latence_trace_requests_total{status=~"5.."}[5m]))
  / sum(rate(latence_trace_requests_total[5m])) > 0.005

# license expiring inside 14 days
latence_trace_license_days_until_expiry < 14

# rate-limited callers
sum(rate(latence_trace_requests_total{status="429"}[5m])) > 1

# unintended profile change
latence_trace_active_profile{profile!~"balanced|quality"} > 0
```

## Tracing

Set `OTEL_EXPORTER_OTLP_ENDPOINT` to enable OpenTelemetry HTTP export
(install the `[otel]` extras to pull in the exporter). The chart's
`observability.otelEndpoint` value sets the env automatically.

Spans you'll see:

- `POST /groundedness` (root, FastAPI auto-instrumentation)
- `encode.{query|context|response}` -- per-stage encoder latency
- `score.maxsim` / `score.literal` / `score.nli` -- attribution math
- `chunk.context` / `chunk.response` -- chunking decisions
- HTTPX outbound spans for vLLM-Factory calls

## Logs

Set `LATENCE_TRACE_LOG_FORMAT=json` (the chart does this by default).
Every log line is a JSON object with:

```json
{"ts": "2026-04-17T18:30:11.123456Z",
 "level": "INFO",
 "logger": "latence_trace.api.routes",
 "message": "groundedness_scored",
 "request_id": "rid_abc",
 "trace_id": "0af7651916cd43dd8448eb211c80319c",
 "span_id": "b7ad6b7169203331",
 "license_subject": "acme-corp",
 "groundedness_v2": 0.91,
 "risk_band": "green"}
```

Pipe these into your log stack (Loki, Datadog, Splunk, ...) -- the
`request_id` and `trace_id` make round-trip debugging trivial.
