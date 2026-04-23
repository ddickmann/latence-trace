# GPU-slim rescore of v2 bank — `v2_full_gpu_scorer_rescore`

- cache: `research/triangular_maxsim/coding/cache/coding_shared_v5__sentence_packed__dual__cp_gte__bank_transcripts_v2__sw15__ctx256__resp256.pt`
- encoder key: `gte`
- device: `cuda`
- n cases: `180`
- total runtime: `4.23s`

## Scorer latency
- n: `180`, mean: `21.15ms`, p50: `17.04ms`, p95: `39.28ms`

## Mean signals by tier

| tier | reverse_context | per_token_p10 |
|---|---:|---:|
| correct | 0.9792 | 0.9514 |
| ambiguous | 0.9785 | 0.9485 |
| wrong | 0.9729 | 0.9264 |
