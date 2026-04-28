# Worker right-sizing

Concrete recommendations for sizing the TRACE worker fleet on RunPod
and self-hosted Kubernetes.  The numbers come from
`data/veracier-industries/proof_bundle_v1/latency_bench/` and are
reproducible via `scripts/bench_latency.py`.

## Per-worker profile

Single NVIDIA A5000 worker, profile `standard`, steady-state
(no cold start) — behaviour of a single warmed-up pod.

| Concurrency | P95 ms | RPS | Notes |
| ---: | ---: | ---: | --- |
| 1 | 166 | 7.6 | Isolated lane.  Quote this for "how fast is one call?" |
| 4 | 393 | 20.3 | Healthy saturation, plenty of headroom |
| 8 | 509 | 22.7 | **Saturation knee** — HPA target |
| 32 | 2 019 | 19.2 | Over-saturated; tail explodes without throughput gain |

Quality profile at concurrency = 8: p95 862 ms, 15.9 RPS.  Budget ~40%
fewer RPS per worker on quality than standard.

## RunPod standby policy

Hosted SaaS autoscales inside RunPod's serverless engine.  Default
policy:

- **Min workers**: 1 per profile (`standard`, `quality`).  Keeps
  the kernel JIT cache warm so the first customer call pays
  ~200 ms p95, not the 6-8 s first-request cost.
- **Max workers**: 16 (`standard`), 8 (`quality`).  Scales with
  queue depth.
- **Scale-out trigger**: RunPod’s own `queue-age > 30 s` rule,
  mirrored in Prometheus by
  `histogram_quantile(0.95, rate(latence_trace_queue_duration_seconds_bucket[2m])) > 0.05`.
- **Scale-in delay**: 5 min after queue clears, so a traffic dip
  doesn’t cause a thrash of cold starts.

## Self-hosted Helm chart (K8s)

Target metric: `latence_trace_inflight_requests` (custom metric
exported via the OpenTelemetry collector).

```yaml
autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 16
  metrics:
    - type: Pods
      pods:
        metric:
          name: latence_trace_inflight_requests
        target:
          type: AverageValue
          averageValue: "6"
```

Rationale: averaging 6 in-flight requests per pod keeps p95 under
the hosted-SLA ceiling (900 ms) with a 30-40% safety margin.

## Failure modes observed during this run

1. **GPU OOM**: Going to concurrency = 32 on a single A5000 did not
   OOM, but vLLM KV cache hit rate stayed at 0% because batching
   wasn’t effective for short single-request payloads.  Increasing
   `max_num_seqs` in `runpod/vllm_*.yaml` without raising batch size
   will not help — the bottleneck is per-request NLI latency.
2. **Cold-start p95**: `startup warmup request 1/4` in the handler
   log shows ~6.5 s for the first NLI call after a rebuild.
   Keeping at least one standby worker per profile eliminates this
   from the customer-facing p95.
3. **Tail at c=32**: Sustained p99 at c=32 hit 2 031 ms.  The HPA
   rule above prevents this by scaling out before per-worker
   concurrency exceeds 8.

## Next steps if p95 ever regresses

1. Re-run `scripts/bench_latency.py` on the suspect build in both
   lanes.  If isolated p95 is > 200 ms, the regression is in the
   scorer, not the infra.
2. Diff against the committed manifests in
   `data/veracier-industries/proof_bundle_v1/latency_bench/`.
3. If sustained p95 at c=8 exceeds 900 ms but isolated is unchanged,
   the regression is in GPU kernel / vLLM config; check the vLLM
   prefix-cache hit rate and batch-size autotune.
