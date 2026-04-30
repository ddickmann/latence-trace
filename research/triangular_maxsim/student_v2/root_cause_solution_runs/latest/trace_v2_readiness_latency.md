# TRACE V2 Readiness Latency Evidence

Generated locally on the current workspace after enabling executable runtime
heads and cache-stable artifact loading.

## Runtime Decision Overhead

Command:

```bash
LATENCE_TRACE_RUNTIME_DECISION_ENABLED=1 python scripts/verify_trace_v2_readiness.py --skip-pytest
```

Focused micro-benchmark: 500 cached `build_runtime_decision(...)` calls per
class with representative response objects and explicit feature maps where the
head requires them.

| Class | p50 ms | p95 ms | p99 ms |
|---|---:|---:|---:|
| `rag.prose.enterprise` | 0.0334 | 0.0465 | 0.0571 |
| `rag.prose.short_factoid` | 0.0616 | 0.0789 | 0.0879 |
| `rag.prose.multi_claim` | 0.0513 | 0.0732 | 0.0807 |
| `rag.structured` | 0.0547 | 0.0755 | 0.0823 |
| `rag.code_in_context` | 0.0506 | 0.0671 | 0.0757 |
| `code.agentic_trace` | 0.0513 | 0.0764 | 0.0867 |

## Interpretation

Runtime-head decision overhead is sub-millisecond after the first artifact load,
so the v2 head layer itself is not the source of the older 1692 ms p95 policy
artifact value. End-to-end service latency still depends on encoder/NLI/GPU
paths and must be rechecked after RunPod rebuild with:

| Mode | p50 ms | p95 ms | p99 ms |
|---|---:|---:|---:|
| v1 / decision disabled | 0.0011 | 0.0018 | 0.0021 |
| v2 / heads enabled | 0.0304 | 0.0428 | 0.0506 |

The measured runtime-decision delta is roughly +0.041 ms at p95 on the local
workspace. This is below the 25 ms head-layer budget used by the promotion
gates.

```bash
LATENCE_TRACE_RUNTIME_DECISION_ENABLED=1 python scripts/bench_runpod_360_via_sdk.py --dump /tmp/trace-v2-runpod-latency.json
python scripts/verify_trace_v2_readiness.py --live-runpod
```
