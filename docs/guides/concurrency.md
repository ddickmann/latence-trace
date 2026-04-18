# Concurrency & batching

> latence-trace v1 ships with a bounded concurrent scoring surface so a
> single Uvicorn worker can serve real production traffic without
> blocking the asyncio event loop, OOM-ing the GPU, or starving the
> encoder/NLI/reranker batch queues.

## Architecture

The `POST /groundedness` endpoint is `async def`, but the underlying
scoring call is a tightly-coupled CPU + GPU pipeline (encoder, Triton
late-interaction kernel, NLI, reranker). The handler therefore:

1. Acquires a process-wide **inflight semaphore** so the number of
   simultaneously-running scoring calls never exceeds the configured
   cap.
2. Runs the synchronous `service.groundedness(request)` inside the
   FastAPI/Starlette **threadpool** (`run_in_threadpool`). The asyncio
   event loop stays free, so health probes (`/healthz`, `/readyz`) and
   metrics scrapes always respond in single-digit milliseconds even
   while a long-running scoring call is in flight.
3. Releases the semaphore on completion or error, including the K1
   refusal path.

The inflight cap is reported in `/readyz` so operators and
load-test harnesses can verify the deployed configuration:

```
GET /readyz
200 {
  "status": "ready",
  "warmup": { "balanced": { "ok": true, ... } },
  "max_inflight": 8
}
```

## Tuning `LATENCE_TRACE_MAX_INFLIGHT`

| Profile  | VRAM budget | Recommended `LATENCE_TRACE_MAX_INFLIGHT` | Notes                                                                 |
|----------|-------------|------------------------------------------|-----------------------------------------------------------------------|
| fast     | < 12 GB     | 8 (default)                              | Encoder is small, NLI usually disabled.                               |
| balanced | < 18 GB     | 8 (default)                              | Encoder + NLI co-resident; matches the `vllm-factory` `--max-num-seqs`. |
| quality  | < 24 GB     | 4                                        | Reranker enabled, longer claim batches; raise only after a load test.   |

Override at runtime:

```bash
LATENCE_TRACE_MAX_INFLIGHT=12 latence-trace serve
```

> The cap is the **maximum number of concurrently-executing
> `score_groundedness` calls**, not a request rate. Bursts above the
> cap are queued at the asyncio layer; the queue grows with the
> number of clients, not with the cap.

## Per-component batch ceilings

The cross-request audit (PA4) confirmed that each downstream component
already serializes contention behind a thread-safe cache:

| Component           | Cache lock                      | Notes |
|---------------------|---------------------------------|-------|
| Encoder provider    | `_encoder_factory.cache_lock`   | Builds the vLLM-Factory or pylate provider exactly once per `(endpoint, model)` pair, even under cold-start bursts. |
| Null bank pack      | `service._null_bank_lock`       | Encodes the bilingual EN+DE null bank exactly once per provider; the **PA3 pre-stacked pack** is reused across requests, so the dominant `compute_null_distribution` hot path is a single batched matmul. |
| NLI provider        | `service._nli_provider_lock`    | Single shared mDeBERTa instance; ColBERT-style multi-vector. |
| Premise reranker    | `service._nli_reranker_lock`    | Single shared `bge-reranker-v2-m3`. |

## Load-test recipe (PA4 baseline)

The shipped `tests/test_concurrent_batching.py` proves the runtime
contract on every CI run:

- `/healthz` finishes in < 50 ms while a scoring call sleeps for
  120 ms inside the encoder.
- A burst of `3 × cap` requests with `LATENCE_TRACE_MAX_INFLIGHT=3`
  never executes more than 3 scoring calls in parallel.
- `/readyz` echoes the live `max_inflight` value so `kubectl describe`
  / Helm post-deploy hooks can audit the cap.

A heavier `k6` / `locust` load test (P2 in the v1 plan) re-runs this
profile against the real encoder + NLI stack to confirm the
documented p50/p95/p99.
