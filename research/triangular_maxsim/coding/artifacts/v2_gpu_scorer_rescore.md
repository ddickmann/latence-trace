# GPU-slim rescore of v2 bank — `v2_gpu_scorer_rescore`

- cache: `/workspace/latence-trace/research/triangular_maxsim/coding/cache/coding_shared_v5__sentence_packed__dual__cp_gte__bank_transcripts_v2__sw15__ctx256__resp256__pilot20.pt`
- encoder key: `gte`
- device: `cuda`
- n cases: `60`
- total runtime: `1.69s`

## Scorer latency
- n: `60`, mean: `21.69ms`, p50: `19.45ms`, p95: `30.30ms`

## Mean signals by tier

| tier | reverse_context | per_token_p10 |
|---|---:|---:|
| correct | 0.9805 | 0.9548 |
| ambiguous | 0.9802 | 0.9533 |
| wrong | 0.9748 | 0.9322 |
