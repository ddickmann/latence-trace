# Learned MaxSim Calibrator Report

## Train Fit

- Token calibrator: `{'loss': 0.6287384629249573, 'accuracy': 0.7357, 'precision': 0.6731, 'recall': 0.734, 'f1': 0.7022, 'n': 200000, 'positive_rate': 0.4245}`
- Turn calibrator: `{'loss': 0.6724032759666443, 'accuracy': 0.6552, 'precision': 0.6233, 'recall': 0.5699, 'f1': 0.5954, 'n': 2152, 'positive_rate': 0.4452}`
- Tiny MLP train accuracy probe: `0.4964`

## Offline Eval

| Channel | Binary Acc | Ungrounded F1 |
|---|---:|---:|
| Raw MaxSim proxy | 0.5147 | 0.3107 |
| Learned calibrator | 0.641 | 0.6717 |
| Raw + calibrator | 0.5632 | 0.4895 |

## Interpretation

- This is an offline frozen-feature experiment, not production wiring.
- `v1_router` and `v1_plus_calibrator` need cached/full v1 predictions before any production claim.
- The calibrator is useful only if fused results improve weak classes without hurting saturated classes.
- Current readout: raw+calibrator improves the proxy baseline overall, so the next gate is full cached v1 predictions and true v1+calibrator fusion.

Verdict: `continue_to_v1_row_cache`
