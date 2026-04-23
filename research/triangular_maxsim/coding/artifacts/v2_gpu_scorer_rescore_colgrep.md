# GPU-slim rescore of v2 bank — `v2_gpu_scorer_rescore_colgrep`

- cache: `research/triangular_maxsim/coding/cache/coding_shared_v5__colgrep__dual__cp_gte__bank_transcripts_v2__sw15__ctx256__resp256__pilot20.pt`
- encoder key: `gte`
- device: `cuda`
- n cases: `60`
- total runtime: `2.16s`

## Scorer latency
- n: `60`, mean: `32.09ms`, p50: `17.59ms`, p95: `186.73ms`

## Mean signals by tier

| tier | reverse_context | per_token_p10 |
|---|---:|---:|
| correct | 0.9791 | 0.9489 |
| ambiguous | 0.9788 | 0.9477 |
| wrong | 0.9731 | 0.9275 |
