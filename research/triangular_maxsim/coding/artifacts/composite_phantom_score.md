# Composite phantom-guard score — `composite_phantom_score`

- fit step: `0.05`
- target FP (false-phantom on grounded): `0.05`

## v1 fit

| chunker | w_rc | w_pt | w_lg | AUROC composite | AUROC rc | AUROC p10 | AUROC lg | threshold | actual FP | recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sentence_packed | 0.00 | 0.95 | 0.05 | **0.984** | 0.938 | 0.984 | 0.797 | 0.8820 | 0.000 | 0.875 |
| colgrep | 0.00 | 1.00 | 0.00 | **0.969** | 0.938 | 0.969 | 0.797 | 0.9061 | 0.000 | 0.875 |
