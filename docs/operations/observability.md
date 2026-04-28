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

## Latency contract

TRACE reports latency in two lanes.  Both are published in the
Veracier proof bundle (`data/veracier-industries/proof_bundle_v1/`)
and are reproducible via `scripts/bench_latency.py`.

### Isolated single-request lane (concurrency = 1)

This is the number to quote when asked "how fast is one call?".
Measured on a single NVIDIA A5000 GPU worker with the 15-row fixture
set replayed twice, no contention.

| Profile | P50 | P95 | P99 |
| --- | ---: | ---: | ---: |
| `standard` | 124 ms | 166 ms | 171 ms |
| `quality` | 132 ms | 178 ms | 182 ms |
| `code` | (see `docs/benchmarks.md`; code profile has its own fixtures) |

### Sustained concurrency lane

This is the number to quote when asked "how does TRACE behave in a
production pipeline that is already running hot?".  Measured on the
same single-GPU worker so this is a conservative per-worker number;
a multi-worker hosted deployment scales horizontally.

| Profile | Concurrency | P50 | P95 | P99 | RPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| `standard` | 4 | 308 ms | 393 ms | 454 ms | 20.3 |
| `standard` | 8 | 337 ms | 509 ms | 535 ms | 22.7 |
| `standard` | 32 | 1 302 ms | 2 019 ms | 2 031 ms | 19.2 |
| `quality` | 8 | 451 ms | 862 ms | 900 ms | 15.9 |

### Capacity planning

- Steady-state per-worker saturation point is around concurrency = 8
  on the A5000 class instance TRACE targets for hosted SaaS.  Going
  to concurrency = 32 roughly quadruples tail latency *without*
  raising throughput; the right knob is horizontal scale.
- Helm chart default HPA: `targetAverageValue: 6` on the custom
  metric `latence_trace_inflight_requests`.  Tune for your cluster
  using the isolated / sustained numbers above as anchors.
- RunPod standby workers: keep **at least 1 per profile** warm so a
  cold arrival pays only the < 200 ms steady-state p95 instead of
  the 6-8 s first-request warmup (see
  `/workspace/latence-trace/runpod/handler.py`
  `startup warmup request 1/4` log line -- the first call is slow
  while the kernel JITs).
- Autoscale aggressiveness: the hosted SLA
  (`commercial/hosted-sla.md`) budgets a 2 x SLO alert threshold, so
  the autoscaler must bring a new worker up within 60 s of sustained
  queue-time above 25 ms.  The Terraform module in
  `runpod/terraform/` ships this rule.

### Alert thresholds (Prometheus)

```promql
# isolated latency SLO
histogram_quantile(0.95, sum(rate(latence_trace_score_duration_seconds_bucket{profile="standard", concurrency="1"}[5m])) by (le)) > 0.200

# sustained latency SLO (multi-worker fleet)
histogram_quantile(0.95, sum(rate(latence_trace_score_duration_seconds_bucket{profile="standard"}[5m])) by (le)) > 0.900

# queue-time SLO (signal that a worker is saturated and the HPA should kick)
histogram_quantile(0.95, sum(rate(latence_trace_queue_duration_seconds_bucket[5m])) by (le)) > 0.050
```

The reference Grafana dashboard at
`docs/operations/grafana/trace-hosted-reliability.json` panels
"Score latency by profile" and "Queue depth" visualise both
quantities side-by-side.
