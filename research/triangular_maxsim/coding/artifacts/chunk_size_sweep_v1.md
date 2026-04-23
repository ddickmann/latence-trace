# Chunk-Size Sweep: `chunk_size_sweep_v1`

One row per `(budget, chunker, scorer)` so the ceiling effect is visible.
`separation` is `mean(grounded) − mean(ungrounded)` on `reverse_context`; `saturation@0.95` is the fraction of anchor cases whose score ≥ 0.95.

- max SWE-bench instances: `15`
- phantom threshold: `0.35`

## AUROC across budgets (flagship: colgrep × fuse_mean)

| budget | chunking | chunker | scorer | AUROC | grounded mean | ungrounded mean | separation | saturation@0.95 |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 64 | gte | sentence_packed | `gte_only` | 0.6323 | 0.9834 | 0.9795 | +0.0039 | 1.0000 |
| 64 | gte | sentence_packed | `code_only` | 0.5979 | 0.9190 | 0.9121 | +0.0068 | 0.0513 |
| 64 | gte | sentence_packed | `fuse_mean` | 0.6243 | 0.9512 | 0.9458 | +0.0053 | 0.4615 |
| 64 | gte | sentence_packed | `fuse_max` | 0.6323 | 0.9834 | 0.9795 | +0.0039 | 1.0000 |
| 64 | gte | sentence_packed | `fuse_min` | 0.5979 | 0.9190 | 0.9121 | +0.0068 | 0.0513 |
| 64 | gte | colgrep | `gte_only` | 0.5741 | 0.9820 | 0.9808 | +0.0012 | 1.0000 |
| 64 | gte | colgrep | `code_only` | 0.5265 | 0.9093 | 0.9090 | +0.0003 | 0.0513 |
| 64 | gte | colgrep | `fuse_mean` (flagship) | 0.5265 | 0.9457 | 0.9449 | +0.0008 | 0.5385 |
| 64 | gte | colgrep | `fuse_max` | 0.5741 | 0.9820 | 0.9808 | +0.0012 | 1.0000 |
| 64 | gte | colgrep | `fuse_min` | 0.5265 | 0.9093 | 0.9090 | +0.0003 | 0.0513 |
| 128 | gte | sentence_packed | `gte_only` | 0.6032 | 0.9815 | 0.9791 | +0.0024 | 1.0000 |
| 128 | gte | sentence_packed | `code_only` | 0.6058 | 0.9294 | 0.9231 | +0.0063 | 0.0256 |
| 128 | gte | sentence_packed | `fuse_mean` | 0.6138 | 0.9555 | 0.9511 | +0.0044 | 0.6923 |
| 128 | gte | sentence_packed | `fuse_max` | 0.6032 | 0.9815 | 0.9791 | +0.0024 | 1.0000 |
| 128 | gte | sentence_packed | `fuse_min` | 0.6058 | 0.9294 | 0.9231 | +0.0063 | 0.0256 |
| 128 | gte | colgrep | `gte_only` | 0.6005 | 0.9814 | 0.9798 | +0.0016 | 1.0000 |
| 128 | gte | colgrep | `code_only` | 0.5741 | 0.9270 | 0.9214 | +0.0056 | 0.0513 |
| 128 | gte | colgrep | `fuse_mean` (flagship) | 0.5741 | 0.9542 | 0.9506 | +0.0036 | 0.7179 |
| 128 | gte | colgrep | `fuse_max` | 0.6005 | 0.9814 | 0.9798 | +0.0016 | 1.0000 |
| 128 | gte | colgrep | `fuse_min` | 0.5741 | 0.9270 | 0.9214 | +0.0056 | 0.0513 |
| 256 | gte | sentence_packed | `gte_only` | 0.6058 | 0.9817 | 0.9792 | +0.0025 | 1.0000 |
| 256 | gte | sentence_packed | `code_only` | 0.6111 | 0.9373 | 0.9297 | +0.0076 | 0.2308 |
| 256 | gte | sentence_packed | `fuse_mean` | 0.6296 | 0.9595 | 0.9545 | +0.0050 | 0.7692 |
| 256 | gte | sentence_packed | `fuse_max` | 0.6058 | 0.9817 | 0.9792 | +0.0025 | 1.0000 |
| 256 | gte | sentence_packed | `fuse_min` | 0.6111 | 0.9373 | 0.9297 | +0.0076 | 0.2308 |
| 256 | gte | colgrep | `gte_only` | 0.6243 | 0.9812 | 0.9795 | +0.0017 | 1.0000 |
| 256 | gte | colgrep | `code_only` | 0.6005 | 0.9365 | 0.9297 | +0.0067 | 0.2308 |
| 256 | gte | colgrep | `fuse_mean` (flagship) | 0.6032 | 0.9588 | 0.9546 | +0.0042 | 0.7436 |
| 256 | gte | colgrep | `fuse_max` | 0.6243 | 0.9812 | 0.9795 | +0.0017 | 1.0000 |
| 256 | gte | colgrep | `fuse_min` | 0.6005 | 0.9365 | 0.9297 | +0.0067 | 0.2308 |
| 512 | code | sentence_packed | `gte_only` | 0.5688 | 0.9777 | 0.9762 | +0.0015 | 1.0000 |
| 512 | code | sentence_packed | `code_only` | 0.5926 | 0.9405 | 0.9333 | +0.0072 | 0.3590 |
| 512 | code | sentence_packed | `fuse_mean` | 0.5952 | 0.9591 | 0.9547 | +0.0044 | 0.7436 |
| 512 | code | sentence_packed | `fuse_max` | 0.5688 | 0.9777 | 0.9762 | +0.0015 | 1.0000 |
| 512 | code | sentence_packed | `fuse_min` | 0.5926 | 0.9405 | 0.9333 | +0.0072 | 0.3590 |
| 512 | code | colgrep | `gte_only` | 0.5979 | 0.9789 | 0.9778 | +0.0011 | 1.0000 |
| 512 | code | colgrep | `code_only` | 0.5714 | 0.9397 | 0.9331 | +0.0066 | 0.4103 |
| 512 | code | colgrep | `fuse_mean` (flagship) | 0.5688 | 0.9593 | 0.9554 | +0.0038 | 0.7436 |
| 512 | code | colgrep | `fuse_max` | 0.5979 | 0.9789 | 0.9778 | +0.0011 | 1.0000 |
| 512 | code | colgrep | `fuse_min` | 0.5714 | 0.9397 | 0.9331 | +0.0066 | 0.4103 |
| 1024 | code | sentence_packed | `gte_only` | 0.5569 | 0.9715 | 0.9708 | +0.0007 | 1.0000 |
| 1024 | code | sentence_packed | `code_only` | 0.5926 | 0.9394 | 0.9322 | +0.0072 | 0.2564 |
| 1024 | code | sentence_packed | `fuse_mean` | 0.6058 | 0.9554 | 0.9515 | +0.0039 | 0.6410 |
| 1024 | code | sentence_packed | `fuse_max` | 0.5688 | 0.9721 | 0.9712 | +0.0009 | 1.0000 |
| 1024 | code | sentence_packed | `fuse_min` | 0.5913 | 0.9388 | 0.9318 | +0.0070 | 0.2564 |
| 1024 | code | colgrep | `gte_only` | 0.5410 | 0.9735 | 0.9733 | +0.0002 | 0.8974 |
| 1024 | code | colgrep | `code_only` | 0.5794 | 0.9392 | 0.9320 | +0.0072 | 0.3846 |
| 1024 | code | colgrep | `fuse_mean` (flagship) | 0.5847 | 0.9564 | 0.9527 | +0.0037 | 0.7436 |
| 1024 | code | colgrep | `fuse_max` | 0.5410 | 0.9737 | 0.9734 | +0.0002 | 0.8974 |
| 1024 | code | colgrep | `fuse_min` | 0.5794 | 0.9390 | 0.9319 | +0.0071 | 0.3846 |

## Flagship cell across budgets (colgrep × fuse_mean)

| budget | chunking | AUROC | grounded mean±std | ungrounded mean±std | separation | overlap | saturation@0.95 |
|---:|---|---:|---|---|---:|---:|---:|
| 64 | gte | 0.5265 | 0.9457±0.0226 | 0.9449±0.0211 | +0.0008 | 0.0575 | 0.5385 |
| 128 | gte | 0.5741 | 0.9542±0.0177 | 0.9506±0.0172 | +0.0036 | 0.0452 | 0.7179 |
| 256 | gte | 0.6032 | 0.9588±0.0156 | 0.9546±0.0160 | +0.0042 | 0.0381 | 0.7436 |
| 512 | code | 0.5688 | 0.9593±0.0176 | 0.9554±0.0180 | +0.0038 | 0.0452 | 0.7436 |
| 1024 | code | 0.5847 | 0.9564±0.0190 | 0.9527±0.0187 | +0.0037 | 0.0485 | 0.7436 |

## Per-budget artifacts

Each budget's full report is at:

- `64` tokens (gte chunker): [chunk_size_sweep_v1__b64_gte.md](chunk_size_sweep_v1__b64_gte.md)
- `128` tokens (gte chunker): [chunk_size_sweep_v1__b128_gte.md](chunk_size_sweep_v1__b128_gte.md)
- `256` tokens (gte chunker): [chunk_size_sweep_v1__b256_gte.md](chunk_size_sweep_v1__b256_gte.md)
- `512` tokens (code chunker): [chunk_size_sweep_v1__b512_code.md](chunk_size_sweep_v1__b512_code.md)
- `1024` tokens (code chunker): [chunk_size_sweep_v1__b1024_code.md](chunk_size_sweep_v1__b1024_code.md)
