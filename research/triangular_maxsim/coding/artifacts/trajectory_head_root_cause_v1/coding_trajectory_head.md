# Coding Agent Trajectory Head

Trained on `native_train`, calibrated on `native_val`.

## Selected Evaluation

| split | rows | AUROC | accuracy | false allow | false block | decision coverage |
|---|---:|---:|---:|---:|---:|---:|
| `train` | 756 | 1.0 | 0.9987 | 0.0079 | 0.0 | 1.0 |
| `native_test` | 252 | 1.0 | 1.0 | 0.0079 | 0.0 | 1.0 |

## Candidate Bake-Off

- `symbolic_linear`
  - `native_test`: AUROC=1.0, false_allow=0.0079, false_block=0.0
- `trajectory_logreg`
  - `native_test`: AUROC=1.0, false_allow=0.0079, false_block=0.0
- `extra_trees`
  - `native_test`: AUROC=1.0, false_allow=0.0, false_block=0.0
- `gbdt`
  - `native_test`: AUROC=1.0, false_allow=0.0, false_block=0.0

## Legacy Similarity-Bank Diagnostic

- `transcripts_v2`: AUROC=0.5, false_allow=0.0, false_block=0.5
- `both`: AUROC=0.5225, false_allow=0.0, false_block=0.4727

Promotion decision: `promote`.

Promotion is based on trajectory-native held-out banks. Legacy similarity-only transcript banks are retained as diagnostics because they do not contain the file/test/order facts needed by the runtime head.
