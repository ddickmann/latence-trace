# Noise-injection validation — dead-weight tracer

- device: `cuda`
- dim: `128`
- response tokens: `96`
- chunks/file: `3`, tokens/chunk: `40`
- dead-weight threshold (coverage <): `0.2`
- cases per cell: `6`

## Cells

| cell | n_sig | n_noise | sig cos range | sig chunk ratio | precision | recall | F1 | cov sig (min) | cov noise (max) | scorer p50 ms |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| noise_25pct_strong_clean | 12 | 4 | [0.70,1.00] | 1.00 | **1.000** | **1.000** | **1.000** | 1.000 | 0.000 | 41.45 |
| noise_50pct_strong_clean | 8 | 8 | [0.70,1.00] | 1.00 | **1.000** | **1.000** | **1.000** | 1.000 | 0.000 | 5.86 |
| noise_75pct_strong_clean | 4 | 12 | [0.70,1.00] | 1.00 | **1.000** | **0.972** | **0.986** | 1.000 | 0.000 | 5.88 |
| partial_signal_1_of_3_chunks | 8 | 8 | [0.70,1.00] | 0.33 | **1.000** | **0.833** | **0.906** | 0.333 | 0.000 | 5.91 |
| adversarial_all_uncertain_below_used | 8 | 8 | [0.40,0.54] | 1.00 | **1.000** | **1.000** | **1.000** | 0.000 | 0.000 | 5.88 |
| adversarial_sub_uncertain | 8 | 8 | [0.30,0.44] | 1.00 | **1.000** | **1.000** | **1.000** | 0.000 | 0.000 | 5.89 |

## Confusion totals (summed over cases)

| cell | TP | FP | FN | TN |
|---|---:|---:|---:|---:|
| noise_25pct_strong_clean | 24 | 0 | 0 | 72 |
| noise_50pct_strong_clean | 48 | 0 | 0 | 48 |
| noise_75pct_strong_clean | 70 | 0 | 2 | 24 |
| partial_signal_1_of_3_chunks | 40 | 0 | 8 | 48 |
| adversarial_all_uncertain_below_used | 48 | 0 | 0 | 48 |
| adversarial_sub_uncertain | 48 | 0 | 0 | 48 |

