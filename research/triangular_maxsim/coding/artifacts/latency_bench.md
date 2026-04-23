# Phantom-guard latency bench

- device: `cuda`
- turns per scenario: `15`
- files per turn: `25`
- tokens per file: `40`
- churn: `5.00%`
- embedding dim: `128`
- freeze_gc: `True`

Target: **p95 total <= 150 ms** for the UX-blocking phantom-guard.

## Warm p50 / p95 / p99 per response size

| size | response tokens | scenario | encode p95 | scorer p95 | total p50 | total p95 | total p99 | <=150ms p95 |
|---|---:|---|---:|---:|---:|---:|---:|:---:|
| small | 32 | warm_no_cache | 0.22 | 3.42 | 3.65 | **3.68** | 3.68 | YES |
| small | 32 | warm_with_cache | 0.03 | 3.36 | 3.42 | **3.56** | 3.56 | YES |
| medium | 128 | warm_no_cache | 0.21 | 3.37 | 3.54 | **3.62** | 3.62 | YES |
| medium | 128 | warm_with_cache | 0.03 | 3.49 | 3.58 | **3.63** | 3.63 | YES |
| large | 512 | warm_no_cache | 0.21 | 3.34 | 3.54 | **3.61** | 3.61 | YES |
| large | 512 | warm_with_cache | 0.03 | 3.42 | 3.48 | **3.65** | 3.65 | YES |

## Cold pass (first call per process)

| size | response tokens | total ms | encode ms | scorer ms |
|---|---:|---:|---:|---:|
| small | 32 | 4.61 | 0.23 | 3.42 |
| medium | 128 | 4.23 | 0.20 | 3.39 |
| large | 512 | 4.16 | 0.20 | 3.29 |

