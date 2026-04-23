# Composite phantom-guard score — `composite_phantom_score_v2_full`

- fit step: `0.05`
- target FP (false-phantom on grounded): `0.05`

## v1 fit

| chunker | w_rc | w_pt | w_lg | AUROC composite | AUROC rc | AUROC p10 | AUROC lg | threshold | actual FP | recall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sentence_packed | 0.00 | 0.95 | 0.05 | **0.984** | 0.938 | 0.984 | 0.797 | 0.8820 | 0.000 | 0.875 |
| colgrep | 0.00 | 1.00 | 0.00 | **0.969** | 0.938 | 0.969 | 0.797 | 0.9061 | 0.000 | 0.875 |

## v2 application (weights fit on v1)

- weights source chunker: `sentence_packed`
- weights: `{'w_rc': 0.0, 'w_pt': 0.95, 'w_lg': 0.05}`
- threshold: `0.8820`

- AUROC correct vs wrong:     **0.8027777777777778**
- AUROC correct vs ambiguous: 0.5597222222222222
- counts: {'correct': 60, 'wrong': 60, 'ambiguous': 60}
- flagged at v1 threshold: {'correct': 3, 'wrong': 6, 'ambiguous': 3}

### v2-locked threshold sweep (wrong = positive)

| target FP | threshold | actual FP | recall | flagged wrong | flagged correct |
|---:|---:|---:|---:|---:|---:|
| 0.02 | 0.8594 | 0.000 | 0.033 | 2 | 0 |
| 0.05 | 0.8672 | 0.033 | 0.050 | 3 | 2 |
| 0.10 | 0.8918 | 0.083 | 0.317 | 19 | 5 |
| 0.20 | 0.9147 | 0.183 | 0.733 | 44 | 11 |

