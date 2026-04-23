# Coding-Agent Groundedness: Scorer Stack x Chunker

> NO — the flagship colgrep x fuse_mean cell did not clear the RAG-style gate: reverse_context AUROC=0.526, phantom-API precision@0.35=0.00, ungrounded minus grounded unused-ratio delta=+0.0000. Stacking lift vs gte_only=-0.0476. Best fusion rule on colgrep is `fuse_max` (AUROC=0.574); consider flipping the flagship.

- run tag: `chunk_size_sweep_v1__b64_gte`
- primary scorer: `lightonai/GTE-ModernColBERT-v1`
- orthogonal scorer: `lightonai/LateOn-Code-edge` (available: `True`)
- shared chunk budget: `64` tokens (GTE tokenizer)
- max SWE-bench instances: `15`
- phantom threshold: `0.35`
- flagship cell: `colgrep` x `fuse_mean` (present: `True`)

## Headline matrix

| chunker | scorer | AUROC | grounded cov | cov delta | unused delta | phantom@thr | grounded held-out rate | p95 ms | support units |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sentence_packed | `gte_only` | 0.6323 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 4390.16 | 20.6 |
| sentence_packed | `code_only` | 0.5979 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 3194.59 | 20.6 |
| sentence_packed | `fuse_mean` | 0.6243 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 4344.17 | 20.6 |
| sentence_packed | `fuse_max` | 0.6323 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 4344.17 | 20.6 |
| sentence_packed | `fuse_min` | 0.5979 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 4344.17 | 20.6 |
| colgrep | `gte_only` | 0.5741 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 4984.60 | 28.4 |
| colgrep | `code_only` | 0.5265 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2903.69 | 28.4 |
| colgrep | `fuse_mean` (flagship) | 0.5265 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 4562.35 | 28.4 |
| colgrep | `fuse_max` | 0.5741 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 4562.35 | 28.4 |
| colgrep | `fuse_min` | 0.5265 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 4562.35 | 28.4 |

`*` marks cells that pass `reverse_context AUROC >= 0.90`.

## Score distribution diagnostics (reverse_context)

| chunker | scorer | grounded mean | grounded p10-p90 | ungrounded mean | ungrounded p10-p90 | separation | saturation@0.95 |
|---|---|---:|---|---:|---|---:|---:|
| sentence_packed | `gte_only` | 0.9834 | 0.9768-0.9888 | 0.9795 | 0.9695-0.9880 | +0.0039 | 1.0000 |
| sentence_packed | `code_only` | 0.9190 | 0.8988-0.9395 | 0.9121 | 0.8894-0.9341 | +0.0068 | 0.0513 |
| sentence_packed | `fuse_mean` | 0.9512 | 0.9393-0.9635 | 0.9458 | 0.9316-0.9615 | +0.0053 | 0.4615 |
| sentence_packed | `fuse_max` | 0.9834 | 0.9768-0.9888 | 0.9795 | 0.9695-0.9880 | +0.0039 | 1.0000 |
| sentence_packed | `fuse_min` | 0.9190 | 0.8988-0.9395 | 0.9121 | 0.8894-0.9341 | +0.0068 | 0.0513 |
| colgrep | `gte_only` | 0.9820 | 0.9702-0.9896 | 0.9808 | 0.9715-0.9895 | +0.0012 | 1.0000 |
| colgrep | `code_only` | 0.9093 | 0.8489-0.9422 | 0.9090 | 0.8445-0.9440 | +0.0003 | 0.0513 |
| colgrep | `fuse_mean` | 0.9457 | 0.9077-0.9652 | 0.9449 | 0.9077-0.9653 | +0.0008 | 0.5385 |
| colgrep | `fuse_max` | 0.9820 | 0.9702-0.9896 | 0.9808 | 0.9715-0.9895 | +0.0012 | 1.0000 |
| colgrep | `fuse_min` | 0.9093 | 0.8489-0.9422 | 0.9090 | 0.8445-0.9440 | +0.0003 | 0.0513 |

`separation` is mean(grounded) − mean(ungrounded) on `reverse_context`. `saturation@0.95` is the fraction of anchor cases (grounded + ungrounded) whose score ≥ 0.95 — a ceiling effect indicator.

## Stacking lift (fuse_* minus gte_only, per chunker)

| chunker | rule | AUROC lift | unused-delta lift | phantom precision lift |
|---|---|---:|---:|---:|
| sentence_packed | fuse_mean | -0.0079 | +0.0000 | +0.0000 |
| sentence_packed | fuse_max | +0.0000 | +0.0000 | +0.0000 |
| sentence_packed | fuse_min | -0.0344 | +0.0000 | +0.0000 |
| colgrep | fuse_mean | -0.0476 | +0.0000 | +0.0000 |
| colgrep | fuse_max | +0.0000 | +0.0000 | +0.0000 |
| colgrep | fuse_min | -0.0476 | +0.0000 | +0.0000 |

## Chunker lift (colgrep minus sentence_packed, per scorer)

| scorer | AUROC lift | unused-delta lift | phantom precision lift |
|---|---:|---:|---:|
| `gte_only` | -0.0582 | +0.0000 | +0.0000 |
| `code_only` | -0.0714 | +0.0000 | +0.0000 |
| `fuse_mean` | -0.0979 | +0.0000 | +0.0000 |
| `fuse_max` | -0.0582 | +0.0000 | +0.0000 |
| `fuse_min` | -0.0714 | +0.0000 | +0.0000 |

## Subcategory snapshot (per cell)

### `sentence_packed` x `gte_only`

| subcategory | count | reverse_context | coverage | unused ratio | max evidence |
|---|---:|---:|---:|---:|---:|
| entity_swap | 10 | 0.9809 | 1.0000 | 0.0000 | 0.9951 |
| grounded | 18 | 0.9834 | 1.0000 | 0.0000 | 0.9957 |
| negation_flip | 3 | 0.9812 | 1.0000 | 0.0000 | 0.9978 |
| parametric | 2 | 0.9660 | 1.0000 | 0.0000 | 0.9964 |
| partial | 2 | 0.9730 | 1.0000 | 0.0000 | 0.9976 |
| phantom_api | 7 | 0.9804 | 1.0000 | 0.0000 | 0.9972 |

### `sentence_packed` x `code_only`

| subcategory | count | reverse_context | coverage | unused ratio | max evidence |
|---|---:|---:|---:|---:|---:|
| entity_swap | 10 | 0.9126 | 1.0000 | 0.0000 | 0.9813 |
| grounded | 18 | 0.9190 | 1.0000 | 0.0000 | 0.9801 |
| negation_flip | 3 | 0.9038 | 1.0000 | 0.0000 | 0.9928 |
| parametric | 2 | 0.8960 | 1.0000 | 0.0000 | 0.9995 |
| partial | 2 | 0.9109 | 1.0000 | 0.0000 | 0.9999 |
| phantom_api | 7 | 0.9156 | 1.0000 | 0.0000 | 0.9878 |

### `sentence_packed` x `fuse_mean`

| subcategory | count | reverse_context | coverage | unused ratio | max evidence |
|---|---:|---:|---:|---:|---:|
| entity_swap | 10 | 0.9468 | 1.0000 | 0.0000 | 0.9882 |
| grounded | 18 | 0.9512 | 1.0000 | 0.0000 | 0.9879 |
| negation_flip | 3 | 0.9425 | 1.0000 | 0.0000 | 0.9953 |
| parametric | 2 | 0.9310 | 1.0000 | 0.0000 | 0.9979 |
| partial | 2 | 0.9420 | 1.0000 | 0.0000 | 0.9988 |
| phantom_api | 7 | 0.9480 | 1.0000 | 0.0000 | 0.9925 |

### `sentence_packed` x `fuse_max`

| subcategory | count | reverse_context | coverage | unused ratio | max evidence |
|---|---:|---:|---:|---:|---:|
| entity_swap | 10 | 0.9809 | 1.0000 | 0.0000 | 0.9958 |
| grounded | 18 | 0.9834 | 1.0000 | 0.0000 | 0.9958 |
| negation_flip | 3 | 0.9812 | 1.0000 | 0.0000 | 0.9987 |
| parametric | 2 | 0.9660 | 1.0000 | 0.0000 | 0.9995 |
| partial | 2 | 0.9730 | 1.0000 | 0.0000 | 0.9999 |
| phantom_api | 7 | 0.9804 | 1.0000 | 0.0000 | 0.9977 |

### `sentence_packed` x `fuse_min`

| subcategory | count | reverse_context | coverage | unused ratio | max evidence |
|---|---:|---:|---:|---:|---:|
| entity_swap | 10 | 0.9126 | 1.0000 | 0.0000 | 0.9805 |
| grounded | 18 | 0.9190 | 1.0000 | 0.0000 | 0.9799 |
| negation_flip | 3 | 0.9038 | 1.0000 | 0.0000 | 0.9919 |
| parametric | 2 | 0.8960 | 1.0000 | 0.0000 | 0.9964 |
| partial | 2 | 0.9109 | 1.0000 | 0.0000 | 0.9976 |
| phantom_api | 7 | 0.9156 | 1.0000 | 0.0000 | 0.9873 |

### `colgrep` x `gte_only`

| subcategory | count | reverse_context | coverage | unused ratio | max evidence |
|---|---:|---:|---:|---:|---:|
| entity_swap | 10 | 0.9817 | 1.0000 | 0.0000 | 0.9955 |
| grounded | 18 | 0.9820 | 1.0000 | 0.0000 | 0.9957 |
| negation_flip | 3 | 0.9814 | 1.0000 | 0.0000 | 0.9978 |
| parametric | 2 | 0.9759 | 1.0000 | 0.0000 | 0.9961 |
| partial | 2 | 0.9794 | 1.0000 | 0.0000 | 0.9980 |
| phantom_api | 7 | 0.9806 | 1.0000 | 0.0000 | 0.9973 |

### `colgrep` x `code_only`

| subcategory | count | reverse_context | coverage | unused ratio | max evidence |
|---|---:|---:|---:|---:|---:|
| entity_swap | 10 | 0.9078 | 1.0000 | 0.0000 | 0.9822 |
| grounded | 18 | 0.9093 | 1.0000 | 0.0000 | 0.9843 |
| negation_flip | 3 | 0.9141 | 1.0000 | 0.0000 | 0.9944 |
| parametric | 2 | 0.9173 | 1.0000 | 0.0000 | 0.9999 |
| partial | 2 | 0.9281 | 1.0000 | 0.0000 | 1.0000 |
| phantom_api | 7 | 0.9086 | 1.0000 | 0.0000 | 0.9875 |

### `colgrep` x `fuse_mean`

| subcategory | count | reverse_context | coverage | unused ratio | max evidence |
|---|---:|---:|---:|---:|---:|
| entity_swap | 10 | 0.9447 | 1.0000 | 0.0000 | 0.9889 |
| grounded | 18 | 0.9457 | 1.0000 | 0.0000 | 0.9900 |
| negation_flip | 3 | 0.9478 | 1.0000 | 0.0000 | 0.9961 |
| parametric | 2 | 0.9466 | 1.0000 | 0.0000 | 0.9980 |
| partial | 2 | 0.9537 | 1.0000 | 0.0000 | 0.9990 |
| phantom_api | 7 | 0.9446 | 1.0000 | 0.0000 | 0.9924 |

### `colgrep` x `fuse_max`

| subcategory | count | reverse_context | coverage | unused ratio | max evidence |
|---|---:|---:|---:|---:|---:|
| entity_swap | 10 | 0.9817 | 1.0000 | 0.0000 | 0.9958 |
| grounded | 18 | 0.9820 | 1.0000 | 0.0000 | 0.9961 |
| negation_flip | 3 | 0.9814 | 1.0000 | 0.0000 | 0.9988 |
| parametric | 2 | 0.9759 | 1.0000 | 0.0000 | 0.9999 |
| partial | 2 | 0.9794 | 1.0000 | 0.0000 | 1.0000 |
| phantom_api | 7 | 0.9806 | 1.0000 | 0.0000 | 0.9979 |

### `colgrep` x `fuse_min`

| subcategory | count | reverse_context | coverage | unused ratio | max evidence |
|---|---:|---:|---:|---:|---:|
| entity_swap | 10 | 0.9078 | 1.0000 | 0.0000 | 0.9819 |
| grounded | 18 | 0.9093 | 1.0000 | 0.0000 | 0.9839 |
| negation_flip | 3 | 0.9141 | 1.0000 | 0.0000 | 0.9933 |
| parametric | 2 | 0.9173 | 1.0000 | 0.0000 | 0.9961 |
| partial | 2 | 0.9281 | 1.0000 | 0.0000 | 0.9980 |
| phantom_api | 7 | 0.9086 | 1.0000 | 0.0000 | 0.9869 |

## Held-out support units (flagship cell)

Flagship cell: `colgrep` x `fuse_mean`. Consensus held-out ids are the intersection of GTE's and LateOn-Code-edge's independently computed `usage_state == 'unused'` sets. `gte` / `code` columns show each encoder's raw unused set.

| case | label | subcategory | consensus held-out | gte unused | code unused | consensus held-out ids |
|---|---|---|---:|---:|---:|---|
| CG1 | grounded | - | 0 | 0 | 0 | - |
| CG2 | grounded | - | 0 | 0 | 0 | - |
| CG3 | grounded | - | 0 | 0 | 0 | - |
| CU1 | ungrounded | phantom_api | 0 | 0 | 0 | - |
| CU2 | ungrounded | phantom_api | 0 | 0 | 0 | - |
| CU3 | ungrounded | entity_swap | 0 | 0 | 0 | - |
| CU4 | ungrounded | entity_swap | 0 | 0 | 0 | - |
| CU5 | ungrounded | parametric | 0 | 0 | 0 | - |
| CU6 | ungrounded | parametric | 0 | 0 | 0 | - |
| CA1 | ambiguous | partial | 0 | 0 | 0 | - |
| CA2 | ambiguous | partial | 0 | 0 | 0 | - |
| CA3 | ambiguous | negation_flip | 0 | 0 | 0 | - |
| SWG1 | grounded | - | 0 | 0 | 0 | - |
| SWU1 | ungrounded | entity_swap | 0 | 0 | 0 | - |
| SWG2 | grounded | - | 0 | 0 | 0 | - |
| SWU2 | ungrounded | phantom_api | 0 | 0 | 0 | - |
| SWG3 | grounded | - | 0 | 0 | 0 | - |
| SWU3 | ungrounded | negation_flip | 0 | 0 | 0 | - |
| SWG4 | grounded | - | 0 | 0 | 0 | - |
| SWU4 | ungrounded | entity_swap | 0 | 0 | 0 | - |
| SWG5 | grounded | - | 0 | 0 | 0 | - |
| SWU5 | ungrounded | entity_swap | 0 | 0 | 0 | - |
| SWG6 | grounded | - | 0 | 0 | 0 | - |
| SWU6 | ungrounded | entity_swap | 0 | 0 | 0 | - |
| SWG7 | grounded | - | 0 | 0 | 0 | - |
| SWU7 | ungrounded | phantom_api | 0 | 0 | 0 | - |
| SWG8 | grounded | - | 0 | 0 | 0 | - |
| SWU8 | ungrounded | entity_swap | 0 | 0 | 0 | - |
| SWG9 | grounded | - | 0 | 0 | 0 | - |
| SWU9 | ungrounded | phantom_api | 0 | 0 | 0 | - |
| SWG10 | grounded | - | 0 | 0 | 0 | - |
| SWU10 | ungrounded | phantom_api | 0 | 0 | 0 | - |
| SWG11 | grounded | - | 0 | 0 | 0 | - |
| SWU11 | ungrounded | phantom_api | 0 | 0 | 0 | - |
| SWG12 | grounded | - | 0 | 0 | 0 | - |
| SWU12 | ungrounded | negation_flip | 0 | 0 | 0 | - |
| SWG13 | grounded | - | 0 | 0 | 0 | - |
| SWU13 | ungrounded | entity_swap | 0 | 0 | 0 | - |
| SWG14 | grounded | - | 0 | 0 | 0 | - |
| SWU14 | ungrounded | entity_swap | 0 | 0 | 0 | - |
| SWG15 | grounded | - | 0 | 0 | 0 | - |
| SWU15 | ungrounded | entity_swap | 0 | 0 | 0 | - |

## Per-case scores

| id | chunker | scorer | label | subcategory | reverse_context | coverage | unused | phantom | max evidence | units |
|---|---|---|---|---|---:|---:|---:|---|---:|---:|
| CG1 | sentence_packed | `gte_only` | grounded | - | 0.9762 | 1.0000 | 0.0000 | False | 1.0000 | 2 |
| CG1 | sentence_packed | `code_only` | grounded | - | 0.9056 | 1.0000 | 0.0000 | False | 0.9962 | 2 |
| CG1 | sentence_packed | `fuse_mean` | grounded | - | 0.9409 | 1.0000 | 0.0000 | False | 0.9981 | 2 |
| CG1 | sentence_packed | `fuse_max` | grounded | - | 0.9762 | 1.0000 | 0.0000 | False | 1.0000 | 2 |
| CG1 | sentence_packed | `fuse_min` | grounded | - | 0.9056 | 1.0000 | 0.0000 | False | 0.9962 | 2 |
| CG2 | sentence_packed | `gte_only` | grounded | - | 0.9853 | 1.0000 | 0.0000 | False | 0.9999 | 3 |
| CG2 | sentence_packed | `code_only` | grounded | - | 0.9378 | 1.0000 | 0.0000 | False | 0.9999 | 3 |
| CG2 | sentence_packed | `fuse_mean` | grounded | - | 0.9615 | 1.0000 | 0.0000 | False | 0.9999 | 3 |
| CG2 | sentence_packed | `fuse_max` | grounded | - | 0.9853 | 1.0000 | 0.0000 | False | 0.9999 | 3 |
| CG2 | sentence_packed | `fuse_min` | grounded | - | 0.9378 | 1.0000 | 0.0000 | False | 0.9999 | 3 |
| CG3 | sentence_packed | `gte_only` | grounded | - | 0.9765 | 1.0000 | 0.0000 | False | 0.9986 | 2 |
| CG3 | sentence_packed | `code_only` | grounded | - | 0.9056 | 1.0000 | 0.0000 | False | 1.0000 | 2 |
| CG3 | sentence_packed | `fuse_mean` | grounded | - | 0.9411 | 1.0000 | 0.0000 | False | 0.9993 | 2 |
| CG3 | sentence_packed | `fuse_max` | grounded | - | 0.9765 | 1.0000 | 0.0000 | False | 1.0000 | 2 |
| CG3 | sentence_packed | `fuse_min` | grounded | - | 0.9056 | 1.0000 | 0.0000 | False | 0.9986 | 2 |
| CU1 | sentence_packed | `gte_only` | ungrounded | phantom_api | 0.9749 | 1.0000 | 0.0000 | False | 0.9977 | 2 |
| CU1 | sentence_packed | `code_only` | ungrounded | phantom_api | 0.9271 | 1.0000 | 0.0000 | False | 1.0000 | 2 |
| CU1 | sentence_packed | `fuse_mean` | ungrounded | phantom_api | 0.9510 | 1.0000 | 0.0000 | False | 0.9988 | 2 |
| CU1 | sentence_packed | `fuse_max` | ungrounded | phantom_api | 0.9749 | 1.0000 | 0.0000 | False | 1.0000 | 2 |
| CU1 | sentence_packed | `fuse_min` | ungrounded | phantom_api | 0.9271 | 1.0000 | 0.0000 | False | 0.9977 | 2 |
| CU2 | sentence_packed | `gte_only` | ungrounded | phantom_api | 0.9698 | 1.0000 | 0.0000 | False | 0.9992 | 2 |
| CU2 | sentence_packed | `code_only` | ungrounded | phantom_api | 0.8676 | 1.0000 | 0.0000 | False | 0.9998 | 2 |
| CU2 | sentence_packed | `fuse_mean` | ungrounded | phantom_api | 0.9187 | 1.0000 | 0.0000 | False | 0.9995 | 2 |
| CU2 | sentence_packed | `fuse_max` | ungrounded | phantom_api | 0.9698 | 1.0000 | 0.0000 | False | 0.9998 | 2 |
| CU2 | sentence_packed | `fuse_min` | ungrounded | phantom_api | 0.8676 | 1.0000 | 0.0000 | False | 0.9992 | 2 |
| CU3 | sentence_packed | `gte_only` | ungrounded | entity_swap | 0.9664 | 1.0000 | 0.0000 | False | 0.9973 | 2 |
| CU3 | sentence_packed | `code_only` | ungrounded | entity_swap | 0.9104 | 1.0000 | 0.0000 | False | 1.0000 | 2 |
| CU3 | sentence_packed | `fuse_mean` | ungrounded | entity_swap | 0.9384 | 1.0000 | 0.0000 | False | 0.9986 | 2 |
| CU3 | sentence_packed | `fuse_max` | ungrounded | entity_swap | 0.9664 | 1.0000 | 0.0000 | False | 1.0000 | 2 |
| CU3 | sentence_packed | `fuse_min` | ungrounded | entity_swap | 0.9104 | 1.0000 | 0.0000 | False | 0.9973 | 2 |
| CU4 | sentence_packed | `gte_only` | ungrounded | entity_swap | 0.9711 | 1.0000 | 0.0000 | False | 0.9974 | 2 |
| CU4 | sentence_packed | `code_only` | ungrounded | entity_swap | 0.8936 | 1.0000 | 0.0000 | False | 1.0000 | 2 |
| CU4 | sentence_packed | `fuse_mean` | ungrounded | entity_swap | 0.9323 | 1.0000 | 0.0000 | False | 0.9987 | 2 |
| CU4 | sentence_packed | `fuse_max` | ungrounded | entity_swap | 0.9711 | 1.0000 | 0.0000 | False | 1.0000 | 2 |
| CU4 | sentence_packed | `fuse_min` | ungrounded | entity_swap | 0.8936 | 1.0000 | 0.0000 | False | 0.9974 | 2 |
| CU5 | sentence_packed | `gte_only` | ungrounded | parametric | 0.9624 | 1.0000 | 0.0000 | False | 0.9962 | 2 |
| CU5 | sentence_packed | `code_only` | ungrounded | parametric | 0.9026 | 1.0000 | 0.0000 | False | 0.9991 | 2 |
| CU5 | sentence_packed | `fuse_mean` | ungrounded | parametric | 0.9325 | 1.0000 | 0.0000 | False | 0.9977 | 2 |
| CU5 | sentence_packed | `fuse_max` | ungrounded | parametric | 0.9624 | 1.0000 | 0.0000 | False | 0.9991 | 2 |
| CU5 | sentence_packed | `fuse_min` | ungrounded | parametric | 0.9026 | 1.0000 | 0.0000 | False | 0.9962 | 2 |
| CU6 | sentence_packed | `gte_only` | ungrounded | parametric | 0.9695 | 1.0000 | 0.0000 | False | 0.9965 | 2 |
| CU6 | sentence_packed | `code_only` | ungrounded | parametric | 0.8894 | 1.0000 | 0.0000 | False | 0.9998 | 2 |
| CU6 | sentence_packed | `fuse_mean` | ungrounded | parametric | 0.9295 | 1.0000 | 0.0000 | False | 0.9982 | 2 |
| CU6 | sentence_packed | `fuse_max` | ungrounded | parametric | 0.9695 | 1.0000 | 0.0000 | False | 0.9998 | 2 |
| CU6 | sentence_packed | `fuse_min` | ungrounded | parametric | 0.8894 | 1.0000 | 0.0000 | False | 0.9965 | 2 |
| CA1 | sentence_packed | `gte_only` | ambiguous | partial | 0.9768 | 1.0000 | 0.0000 | False | 0.9983 | 2 |
| CA1 | sentence_packed | `code_only` | ambiguous | partial | 0.9030 | 1.0000 | 0.0000 | False | 0.9998 | 2 |
| CA1 | sentence_packed | `fuse_mean` | ambiguous | partial | 0.9399 | 1.0000 | 0.0000 | False | 0.9990 | 2 |
| CA1 | sentence_packed | `fuse_max` | ambiguous | partial | 0.9768 | 1.0000 | 0.0000 | False | 0.9998 | 2 |
| CA1 | sentence_packed | `fuse_min` | ambiguous | partial | 0.9030 | 1.0000 | 0.0000 | False | 0.9983 | 2 |
| CA2 | sentence_packed | `gte_only` | ambiguous | partial | 0.9692 | 1.0000 | 0.0000 | False | 0.9970 | 2 |
| CA2 | sentence_packed | `code_only` | ambiguous | partial | 0.9188 | 1.0000 | 0.0000 | False | 1.0000 | 2 |
| CA2 | sentence_packed | `fuse_mean` | ambiguous | partial | 0.9440 | 1.0000 | 0.0000 | False | 0.9985 | 2 |
| CA2 | sentence_packed | `fuse_max` | ambiguous | partial | 0.9692 | 1.0000 | 0.0000 | False | 1.0000 | 2 |
| CA2 | sentence_packed | `fuse_min` | ambiguous | partial | 0.9188 | 1.0000 | 0.0000 | False | 0.9970 | 2 |
| CA3 | sentence_packed | `gte_only` | ambiguous | negation_flip | 0.9765 | 1.0000 | 0.0000 | False | 0.9972 | 2 |
| CA3 | sentence_packed | `code_only` | ambiguous | negation_flip | 0.8838 | 1.0000 | 0.0000 | False | 0.9998 | 2 |
| CA3 | sentence_packed | `fuse_mean` | ambiguous | negation_flip | 0.9302 | 1.0000 | 0.0000 | False | 0.9985 | 2 |
| CA3 | sentence_packed | `fuse_max` | ambiguous | negation_flip | 0.9765 | 1.0000 | 0.0000 | False | 0.9998 | 2 |
| CA3 | sentence_packed | `fuse_min` | ambiguous | negation_flip | 0.8838 | 1.0000 | 0.0000 | False | 0.9972 | 2 |
| SWG1 | sentence_packed | `gte_only` | grounded | - | 0.9782 | 1.0000 | 0.0000 | False | 0.9921 | 10 |
| SWG1 | sentence_packed | `code_only` | grounded | - | 0.8930 | 1.0000 | 0.0000 | False | 0.9490 | 10 |
| SWG1 | sentence_packed | `fuse_mean` | grounded | - | 0.9356 | 1.0000 | 0.0000 | False | 0.9705 | 10 |
| SWG1 | sentence_packed | `fuse_max` | grounded | - | 0.9782 | 1.0000 | 0.0000 | False | 0.9921 | 10 |
| SWG1 | sentence_packed | `fuse_min` | grounded | - | 0.8930 | 1.0000 | 0.0000 | False | 0.9490 | 10 |
| SWU1 | sentence_packed | `gte_only` | ungrounded | entity_swap | 0.9754 | 1.0000 | 0.0000 | False | 0.9921 | 10 |
| SWU1 | sentence_packed | `code_only` | ungrounded | entity_swap | 0.8878 | 1.0000 | 0.0000 | False | 0.9490 | 10 |
| SWU1 | sentence_packed | `fuse_mean` | ungrounded | entity_swap | 0.9316 | 1.0000 | 0.0000 | False | 0.9705 | 10 |
| SWU1 | sentence_packed | `fuse_max` | ungrounded | entity_swap | 0.9754 | 1.0000 | 0.0000 | False | 0.9921 | 10 |
| SWU1 | sentence_packed | `fuse_min` | ungrounded | entity_swap | 0.8878 | 1.0000 | 0.0000 | False | 0.9490 | 10 |
| SWG2 | sentence_packed | `gte_only` | grounded | - | 0.9769 | 1.0000 | 0.0000 | False | 0.9973 | 7 |
| SWG2 | sentence_packed | `code_only` | grounded | - | 0.9096 | 1.0000 | 0.0000 | False | 0.9922 | 7 |
| SWG2 | sentence_packed | `fuse_mean` | grounded | - | 0.9433 | 1.0000 | 0.0000 | False | 0.9948 | 7 |
| SWG2 | sentence_packed | `fuse_max` | grounded | - | 0.9769 | 1.0000 | 0.0000 | False | 0.9973 | 7 |
| SWG2 | sentence_packed | `fuse_min` | grounded | - | 0.9096 | 1.0000 | 0.0000 | False | 0.9922 | 7 |
| SWU2 | sentence_packed | `gte_only` | ungrounded | phantom_api | 0.9752 | 1.0000 | 0.0000 | False | 0.9973 | 7 |
| SWU2 | sentence_packed | `code_only` | ungrounded | phantom_api | 0.9045 | 1.0000 | 0.0000 | False | 0.9922 | 7 |
| SWU2 | sentence_packed | `fuse_mean` | ungrounded | phantom_api | 0.9398 | 1.0000 | 0.0000 | False | 0.9948 | 7 |
| SWU2 | sentence_packed | `fuse_max` | ungrounded | phantom_api | 0.9752 | 1.0000 | 0.0000 | False | 0.9973 | 7 |
| SWU2 | sentence_packed | `fuse_min` | ungrounded | phantom_api | 0.9045 | 1.0000 | 0.0000 | False | 0.9922 | 7 |
| SWG3 | sentence_packed | `gte_only` | grounded | - | 0.9842 | 1.0000 | 0.0000 | False | 0.9979 | 9 |
| SWG3 | sentence_packed | `code_only` | grounded | - | 0.9250 | 1.0000 | 0.0000 | False | 0.9888 | 9 |
| SWG3 | sentence_packed | `fuse_mean` | grounded | - | 0.9546 | 1.0000 | 0.0000 | False | 0.9934 | 9 |
| SWG3 | sentence_packed | `fuse_max` | grounded | - | 0.9842 | 1.0000 | 0.0000 | False | 0.9979 | 9 |
| SWG3 | sentence_packed | `fuse_min` | grounded | - | 0.9250 | 1.0000 | 0.0000 | False | 0.9888 | 9 |
| SWU3 | sentence_packed | `gte_only` | ungrounded | negation_flip | 0.9832 | 1.0000 | 0.0000 | False | 0.9979 | 9 |
| SWU3 | sentence_packed | `code_only` | ungrounded | negation_flip | 0.9219 | 1.0000 | 0.0000 | False | 0.9868 | 9 |
| SWU3 | sentence_packed | `fuse_mean` | ungrounded | negation_flip | 0.9525 | 1.0000 | 0.0000 | False | 0.9923 | 9 |
| SWU3 | sentence_packed | `fuse_max` | ungrounded | negation_flip | 0.9832 | 1.0000 | 0.0000 | False | 0.9979 | 9 |
| SWU3 | sentence_packed | `fuse_min` | ungrounded | negation_flip | 0.9219 | 1.0000 | 0.0000 | False | 0.9868 | 9 |
| SWG4 | sentence_packed | `gte_only` | grounded | - | 0.9840 | 1.0000 | 0.0000 | False | 0.9982 | 14 |
| SWG4 | sentence_packed | `code_only` | grounded | - | 0.9212 | 1.0000 | 0.0000 | False | 0.9830 | 14 |
| SWG4 | sentence_packed | `fuse_mean` | grounded | - | 0.9526 | 1.0000 | 0.0000 | False | 0.9906 | 14 |
| SWG4 | sentence_packed | `fuse_max` | grounded | - | 0.9840 | 1.0000 | 0.0000 | False | 0.9982 | 14 |
| SWG4 | sentence_packed | `fuse_min` | grounded | - | 0.9212 | 1.0000 | 0.0000 | False | 0.9830 | 14 |
| SWU4 | sentence_packed | `gte_only` | ungrounded | entity_swap | 0.9850 | 1.0000 | 0.0000 | False | 0.9979 | 14 |
| SWU4 | sentence_packed | `code_only` | ungrounded | entity_swap | 0.9097 | 1.0000 | 0.0000 | False | 1.0000 | 14 |
| SWU4 | sentence_packed | `fuse_mean` | ungrounded | entity_swap | 0.9473 | 1.0000 | 0.0000 | False | 0.9989 | 14 |
| SWU4 | sentence_packed | `fuse_max` | ungrounded | entity_swap | 0.9850 | 1.0000 | 0.0000 | False | 1.0000 | 14 |
| SWU4 | sentence_packed | `fuse_min` | ungrounded | entity_swap | 0.9097 | 1.0000 | 0.0000 | False | 0.9979 | 14 |
| SWG5 | sentence_packed | `gte_only` | grounded | - | 0.9802 | 1.0000 | 0.0000 | False | 0.9973 | 16 |
| SWG5 | sentence_packed | `code_only` | grounded | - | 0.8905 | 1.0000 | 0.0000 | False | 0.9357 | 16 |
| SWG5 | sentence_packed | `fuse_mean` | grounded | - | 0.9353 | 1.0000 | 0.0000 | False | 0.9665 | 16 |
| SWG5 | sentence_packed | `fuse_max` | grounded | - | 0.9802 | 1.0000 | 0.0000 | False | 0.9973 | 16 |
| SWG5 | sentence_packed | `fuse_min` | grounded | - | 0.8905 | 1.0000 | 0.0000 | False | 0.9357 | 16 |
| SWU5 | sentence_packed | `gte_only` | ungrounded | entity_swap | 0.9815 | 1.0000 | 0.0000 | False | 0.9980 | 16 |
| SWU5 | sentence_packed | `code_only` | ungrounded | entity_swap | 0.9052 | 1.0000 | 0.0000 | False | 0.9725 | 16 |
| SWU5 | sentence_packed | `fuse_mean` | ungrounded | entity_swap | 0.9433 | 1.0000 | 0.0000 | False | 0.9852 | 16 |
| SWU5 | sentence_packed | `fuse_max` | ungrounded | entity_swap | 0.9815 | 1.0000 | 0.0000 | False | 0.9980 | 16 |
| SWU5 | sentence_packed | `fuse_min` | ungrounded | entity_swap | 0.9052 | 1.0000 | 0.0000 | False | 0.9725 | 16 |
| SWG6 | sentence_packed | `gte_only` | grounded | - | 0.9815 | 1.0000 | 0.0000 | False | 0.9785 | 16 |
| SWG6 | sentence_packed | `code_only` | grounded | - | 0.9073 | 1.0000 | 0.0000 | False | 0.9607 | 16 |
| SWG6 | sentence_packed | `fuse_mean` | grounded | - | 0.9444 | 1.0000 | 0.0000 | False | 0.9696 | 16 |
| SWG6 | sentence_packed | `fuse_max` | grounded | - | 0.9815 | 1.0000 | 0.0000 | False | 0.9785 | 16 |
| SWG6 | sentence_packed | `fuse_min` | grounded | - | 0.9073 | 1.0000 | 0.0000 | False | 0.9607 | 16 |
| SWU6 | sentence_packed | `gte_only` | ungrounded | entity_swap | 0.9815 | 1.0000 | 0.0000 | False | 0.9786 | 16 |
| SWU6 | sentence_packed | `code_only` | ungrounded | entity_swap | 0.9082 | 1.0000 | 0.0000 | False | 0.9606 | 16 |
| SWU6 | sentence_packed | `fuse_mean` | ungrounded | entity_swap | 0.9449 | 1.0000 | 0.0000 | False | 0.9696 | 16 |
| SWU6 | sentence_packed | `fuse_max` | ungrounded | entity_swap | 0.9815 | 1.0000 | 0.0000 | False | 0.9786 | 16 |
| SWU6 | sentence_packed | `fuse_min` | ungrounded | entity_swap | 0.9082 | 1.0000 | 0.0000 | False | 0.9606 | 16 |
| SWG7 | sentence_packed | `gte_only` | grounded | - | 0.9869 | 1.0000 | 0.0000 | False | 0.9879 | 22 |
| SWG7 | sentence_packed | `code_only` | grounded | - | 0.9243 | 1.0000 | 0.0000 | False | 0.9546 | 22 |
| SWG7 | sentence_packed | `fuse_mean` | grounded | - | 0.9556 | 1.0000 | 0.0000 | False | 0.9712 | 22 |
| SWG7 | sentence_packed | `fuse_max` | grounded | - | 0.9869 | 1.0000 | 0.0000 | False | 0.9879 | 22 |
| SWG7 | sentence_packed | `fuse_min` | grounded | - | 0.9243 | 1.0000 | 0.0000 | False | 0.9546 | 22 |
| SWU7 | sentence_packed | `gte_only` | ungrounded | phantom_api | 0.9866 | 1.0000 | 0.0000 | False | 0.9879 | 22 |
| SWU7 | sentence_packed | `code_only` | ungrounded | phantom_api | 0.9204 | 1.0000 | 0.0000 | False | 0.9546 | 22 |
| SWU7 | sentence_packed | `fuse_mean` | ungrounded | phantom_api | 0.9535 | 1.0000 | 0.0000 | False | 0.9712 | 22 |
| SWU7 | sentence_packed | `fuse_max` | ungrounded | phantom_api | 0.9866 | 1.0000 | 0.0000 | False | 0.9879 | 22 |
| SWU7 | sentence_packed | `fuse_min` | ungrounded | phantom_api | 0.9204 | 1.0000 | 0.0000 | False | 0.9546 | 22 |
| SWG8 | sentence_packed | `gte_only` | grounded | - | 0.9896 | 1.0000 | 0.0000 | False | 0.9999 | 43 |
| SWG8 | sentence_packed | `code_only` | grounded | - | 0.9510 | 1.0000 | 0.0000 | False | 1.0000 | 43 |
| SWG8 | sentence_packed | `fuse_mean` | grounded | - | 0.9703 | 1.0000 | 0.0000 | False | 0.9999 | 43 |
| SWG8 | sentence_packed | `fuse_max` | grounded | - | 0.9896 | 1.0000 | 0.0000 | False | 1.0000 | 43 |
| SWG8 | sentence_packed | `fuse_min` | grounded | - | 0.9510 | 1.0000 | 0.0000 | False | 0.9999 | 43 |
| SWU8 | sentence_packed | `gte_only` | ungrounded | entity_swap | 0.9898 | 1.0000 | 0.0000 | False | 0.9999 | 43 |
| SWU8 | sentence_packed | `code_only` | ungrounded | entity_swap | 0.9512 | 1.0000 | 0.0000 | False | 1.0000 | 43 |
| SWU8 | sentence_packed | `fuse_mean` | ungrounded | entity_swap | 0.9705 | 1.0000 | 0.0000 | False | 0.9999 | 43 |
| SWU8 | sentence_packed | `fuse_max` | ungrounded | entity_swap | 0.9898 | 1.0000 | 0.0000 | False | 1.0000 | 43 |
| SWU8 | sentence_packed | `fuse_min` | ungrounded | entity_swap | 0.9512 | 1.0000 | 0.0000 | False | 0.9999 | 43 |
| SWG9 | sentence_packed | `gte_only` | grounded | - | 0.9818 | 1.0000 | 0.0000 | False | 0.9980 | 40 |
| SWG9 | sentence_packed | `code_only` | grounded | - | 0.9228 | 1.0000 | 0.0000 | False | 0.9854 | 40 |
| SWG9 | sentence_packed | `fuse_mean` | grounded | - | 0.9523 | 1.0000 | 0.0000 | False | 0.9917 | 40 |
| SWG9 | sentence_packed | `fuse_max` | grounded | - | 0.9818 | 1.0000 | 0.0000 | False | 0.9980 | 40 |
| SWG9 | sentence_packed | `fuse_min` | grounded | - | 0.9228 | 1.0000 | 0.0000 | False | 0.9854 | 40 |
| SWU9 | sentence_packed | `gte_only` | ungrounded | phantom_api | 0.9802 | 1.0000 | 0.0000 | False | 0.9987 | 40 |
| SWU9 | sentence_packed | `code_only` | ungrounded | phantom_api | 0.9157 | 1.0000 | 0.0000 | False | 0.9681 | 40 |
| SWU9 | sentence_packed | `fuse_mean` | ungrounded | phantom_api | 0.9479 | 1.0000 | 0.0000 | False | 0.9834 | 40 |
| SWU9 | sentence_packed | `fuse_max` | ungrounded | phantom_api | 0.9802 | 1.0000 | 0.0000 | False | 0.9987 | 40 |
| SWU9 | sentence_packed | `fuse_min` | ungrounded | phantom_api | 0.9157 | 1.0000 | 0.0000 | False | 0.9681 | 40 |
| SWG10 | sentence_packed | `gte_only` | grounded | - | 0.9878 | 1.0000 | 0.0000 | False | 0.9998 | 22 |
| SWG10 | sentence_packed | `code_only` | grounded | - | 0.9331 | 1.0000 | 0.0000 | False | 1.0000 | 22 |
| SWG10 | sentence_packed | `fuse_mean` | grounded | - | 0.9605 | 1.0000 | 0.0000 | False | 0.9999 | 22 |
| SWG10 | sentence_packed | `fuse_max` | grounded | - | 0.9878 | 1.0000 | 0.0000 | False | 1.0000 | 22 |
| SWG10 | sentence_packed | `fuse_min` | grounded | - | 0.9331 | 1.0000 | 0.0000 | False | 0.9998 | 22 |
| SWU10 | sentence_packed | `gte_only` | ungrounded | phantom_api | 0.9877 | 1.0000 | 0.0000 | False | 0.9998 | 22 |
| SWU10 | sentence_packed | `code_only` | ungrounded | phantom_api | 0.9317 | 1.0000 | 0.0000 | False | 1.0000 | 22 |
| SWU10 | sentence_packed | `fuse_mean` | ungrounded | phantom_api | 0.9597 | 1.0000 | 0.0000 | False | 0.9999 | 22 |
| SWU10 | sentence_packed | `fuse_max` | ungrounded | phantom_api | 0.9877 | 1.0000 | 0.0000 | False | 1.0000 | 22 |
| SWU10 | sentence_packed | `fuse_min` | ungrounded | phantom_api | 0.9317 | 1.0000 | 0.0000 | False | 0.9998 | 22 |
| SWG11 | sentence_packed | `gte_only` | grounded | - | 0.9888 | 1.0000 | 0.0000 | False | 0.9996 | 76 |
| SWG11 | sentence_packed | `code_only` | grounded | - | 0.9437 | 1.0000 | 0.0000 | False | 1.0000 | 76 |
| SWG11 | sentence_packed | `fuse_mean` | grounded | - | 0.9662 | 1.0000 | 0.0000 | False | 0.9998 | 76 |
| SWG11 | sentence_packed | `fuse_max` | grounded | - | 0.9888 | 1.0000 | 0.0000 | False | 1.0000 | 76 |
| SWG11 | sentence_packed | `fuse_min` | grounded | - | 0.9437 | 1.0000 | 0.0000 | False | 0.9996 | 76 |
| SWU11 | sentence_packed | `gte_only` | ungrounded | phantom_api | 0.9880 | 1.0000 | 0.0000 | False | 0.9996 | 76 |
| SWU11 | sentence_packed | `code_only` | ungrounded | phantom_api | 0.9421 | 1.0000 | 0.0000 | False | 1.0000 | 76 |
| SWU11 | sentence_packed | `fuse_mean` | ungrounded | phantom_api | 0.9650 | 1.0000 | 0.0000 | False | 0.9998 | 76 |
| SWU11 | sentence_packed | `fuse_max` | ungrounded | phantom_api | 0.9880 | 1.0000 | 0.0000 | False | 1.0000 | 76 |
| SWU11 | sentence_packed | `fuse_min` | ungrounded | phantom_api | 0.9421 | 1.0000 | 0.0000 | False | 0.9996 | 76 |
| SWG12 | sentence_packed | `gte_only` | grounded | - | 0.9844 | 1.0000 | 0.0000 | False | 0.9984 | 17 |
| SWG12 | sentence_packed | `code_only` | grounded | - | 0.9070 | 1.0000 | 0.0000 | False | 0.9922 | 17 |
| SWG12 | sentence_packed | `fuse_mean` | grounded | - | 0.9457 | 1.0000 | 0.0000 | False | 0.9953 | 17 |
| SWG12 | sentence_packed | `fuse_max` | grounded | - | 0.9844 | 1.0000 | 0.0000 | False | 0.9984 | 17 |
| SWG12 | sentence_packed | `fuse_min` | grounded | - | 0.9070 | 1.0000 | 0.0000 | False | 0.9922 | 17 |
| SWU12 | sentence_packed | `gte_only` | ungrounded | negation_flip | 0.9839 | 1.0000 | 0.0000 | False | 0.9984 | 17 |
| SWU12 | sentence_packed | `code_only` | ungrounded | negation_flip | 0.9057 | 1.0000 | 0.0000 | False | 0.9917 | 17 |
| SWU12 | sentence_packed | `fuse_mean` | ungrounded | negation_flip | 0.9448 | 1.0000 | 0.0000 | False | 0.9951 | 17 |
| SWU12 | sentence_packed | `fuse_max` | ungrounded | negation_flip | 0.9839 | 1.0000 | 0.0000 | False | 0.9984 | 17 |
| SWU12 | sentence_packed | `fuse_min` | ungrounded | negation_flip | 0.9057 | 1.0000 | 0.0000 | False | 0.9917 | 17 |
| SWG13 | sentence_packed | `gte_only` | grounded | - | 0.9825 | 1.0000 | 0.0000 | False | 0.9897 | 65 |
| SWG13 | sentence_packed | `code_only` | grounded | - | 0.9012 | 1.0000 | 0.0000 | False | 0.9561 | 65 |
| SWG13 | sentence_packed | `fuse_mean` | grounded | - | 0.9419 | 1.0000 | 0.0000 | False | 0.9729 | 65 |
| SWG13 | sentence_packed | `fuse_max` | grounded | - | 0.9825 | 1.0000 | 0.0000 | False | 0.9897 | 65 |
| SWG13 | sentence_packed | `fuse_min` | grounded | - | 0.9012 | 1.0000 | 0.0000 | False | 0.9561 | 65 |
| SWU13 | sentence_packed | `gte_only` | ungrounded | entity_swap | 0.9819 | 1.0000 | 0.0000 | False | 0.9999 | 65 |
| SWU13 | sentence_packed | `code_only` | ungrounded | entity_swap | 0.8997 | 1.0000 | 0.0000 | False | 0.9836 | 65 |
| SWU13 | sentence_packed | `fuse_mean` | ungrounded | entity_swap | 0.9408 | 1.0000 | 0.0000 | False | 0.9918 | 65 |
| SWU13 | sentence_packed | `fuse_max` | ungrounded | entity_swap | 0.9819 | 1.0000 | 0.0000 | False | 0.9999 | 65 |
| SWU13 | sentence_packed | `fuse_min` | ungrounded | entity_swap | 0.8997 | 1.0000 | 0.0000 | False | 0.9836 | 65 |
| SWG14 | sentence_packed | `gte_only` | grounded | - | 0.9876 | 1.0000 | 0.0000 | False | 0.9922 | 19 |
| SWG14 | sentence_packed | `code_only` | grounded | - | 0.9271 | 1.0000 | 0.0000 | False | 0.9589 | 19 |
| SWG14 | sentence_packed | `fuse_mean` | grounded | - | 0.9574 | 1.0000 | 0.0000 | False | 0.9755 | 19 |
| SWG14 | sentence_packed | `fuse_max` | grounded | - | 0.9876 | 1.0000 | 0.0000 | False | 0.9922 | 19 |
| SWG14 | sentence_packed | `fuse_min` | grounded | - | 0.9271 | 1.0000 | 0.0000 | False | 0.9589 | 19 |
| SWU14 | sentence_packed | `gte_only` | ungrounded | entity_swap | 0.9876 | 1.0000 | 0.0000 | False | 0.9922 | 19 |
| SWU14 | sentence_packed | `code_only` | ungrounded | entity_swap | 0.9266 | 1.0000 | 0.0000 | False | 0.9589 | 19 |
| SWU14 | sentence_packed | `fuse_mean` | ungrounded | entity_swap | 0.9571 | 1.0000 | 0.0000 | False | 0.9755 | 19 |
| SWU14 | sentence_packed | `fuse_max` | ungrounded | entity_swap | 0.9876 | 1.0000 | 0.0000 | False | 0.9922 | 19 |
| SWU14 | sentence_packed | `fuse_min` | ungrounded | entity_swap | 0.9266 | 1.0000 | 0.0000 | False | 0.9589 | 19 |
| SWG15 | sentence_packed | `gte_only` | grounded | - | 0.9890 | 1.0000 | 0.0000 | False | 0.9973 | 45 |
| SWG15 | sentence_packed | `code_only` | grounded | - | 0.9358 | 1.0000 | 0.0000 | False | 0.9884 | 45 |
| SWG15 | sentence_packed | `fuse_mean` | grounded | - | 0.9624 | 1.0000 | 0.0000 | False | 0.9929 | 45 |
| SWG15 | sentence_packed | `fuse_max` | grounded | - | 0.9890 | 1.0000 | 0.0000 | False | 0.9973 | 45 |
| SWG15 | sentence_packed | `fuse_min` | grounded | - | 0.9358 | 1.0000 | 0.0000 | False | 0.9884 | 45 |
| SWU15 | sentence_packed | `gte_only` | ungrounded | entity_swap | 0.9888 | 1.0000 | 0.0000 | False | 0.9973 | 45 |
| SWU15 | sentence_packed | `code_only` | ungrounded | entity_swap | 0.9341 | 1.0000 | 0.0000 | False | 0.9884 | 45 |
| SWU15 | sentence_packed | `fuse_mean` | ungrounded | entity_swap | 0.9615 | 1.0000 | 0.0000 | False | 0.9929 | 45 |
| SWU15 | sentence_packed | `fuse_max` | ungrounded | entity_swap | 0.9888 | 1.0000 | 0.0000 | False | 0.9973 | 45 |
| SWU15 | sentence_packed | `fuse_min` | ungrounded | entity_swap | 0.9341 | 1.0000 | 0.0000 | False | 0.9884 | 45 |
| CG1 | colgrep | `gte_only` | grounded | - | 0.9876 | 1.0000 | 0.0000 | False | 0.9980 | 6 |
| CG1 | colgrep | `code_only` | grounded | - | 0.9394 | 1.0000 | 0.0000 | False | 1.0000 | 6 |
| CG1 | colgrep | `fuse_mean` | grounded | - | 0.9635 | 1.0000 | 0.0000 | False | 0.9990 | 6 |
| CG1 | colgrep | `fuse_max` | grounded | - | 0.9876 | 1.0000 | 0.0000 | False | 1.0000 | 6 |
| CG1 | colgrep | `fuse_min` | grounded | - | 0.9394 | 1.0000 | 0.0000 | False | 0.9980 | 6 |
| CG2 | colgrep | `gte_only` | grounded | - | 0.9844 | 1.0000 | 0.0000 | False | 0.9991 | 8 |
| CG2 | colgrep | `code_only` | grounded | - | 0.9398 | 1.0000 | 0.0000 | False | 0.9999 | 8 |
| CG2 | colgrep | `fuse_mean` | grounded | - | 0.9621 | 1.0000 | 0.0000 | False | 0.9995 | 8 |
| CG2 | colgrep | `fuse_max` | grounded | - | 0.9844 | 1.0000 | 0.0000 | False | 0.9999 | 8 |
| CG2 | colgrep | `fuse_min` | grounded | - | 0.9398 | 1.0000 | 0.0000 | False | 0.9991 | 8 |
| CG3 | colgrep | `gte_only` | grounded | - | 0.9722 | 1.0000 | 0.0000 | False | 0.9982 | 4 |
| CG3 | colgrep | `code_only` | grounded | - | 0.8693 | 1.0000 | 0.0000 | False | 1.0000 | 4 |
| CG3 | colgrep | `fuse_mean` | grounded | - | 0.9208 | 1.0000 | 0.0000 | False | 0.9991 | 4 |
| CG3 | colgrep | `fuse_max` | grounded | - | 0.9722 | 1.0000 | 0.0000 | False | 1.0000 | 4 |
| CG3 | colgrep | `fuse_min` | grounded | - | 0.8693 | 1.0000 | 0.0000 | False | 0.9982 | 4 |
| CU1 | colgrep | `gte_only` | ungrounded | phantom_api | 0.9850 | 1.0000 | 0.0000 | False | 0.9998 | 5 |
| CU1 | colgrep | `code_only` | ungrounded | phantom_api | 0.9440 | 1.0000 | 0.0000 | False | 1.0000 | 5 |
| CU1 | colgrep | `fuse_mean` | ungrounded | phantom_api | 0.9645 | 1.0000 | 0.0000 | False | 0.9999 | 5 |
| CU1 | colgrep | `fuse_max` | ungrounded | phantom_api | 0.9850 | 1.0000 | 0.0000 | False | 1.0000 | 5 |
| CU1 | colgrep | `fuse_min` | ungrounded | phantom_api | 0.9440 | 1.0000 | 0.0000 | False | 0.9998 | 5 |
| CU2 | colgrep | `gte_only` | ungrounded | phantom_api | 0.9752 | 1.0000 | 0.0000 | False | 0.9976 | 4 |
| CU2 | colgrep | `code_only` | ungrounded | phantom_api | 0.8665 | 1.0000 | 0.0000 | False | 0.9998 | 4 |
| CU2 | colgrep | `fuse_mean` | ungrounded | phantom_api | 0.9208 | 1.0000 | 0.0000 | False | 0.9987 | 4 |
| CU2 | colgrep | `fuse_max` | ungrounded | phantom_api | 0.9752 | 1.0000 | 0.0000 | False | 0.9998 | 4 |
| CU2 | colgrep | `fuse_min` | ungrounded | phantom_api | 0.8665 | 1.0000 | 0.0000 | False | 0.9976 | 4 |
| CU3 | colgrep | `gte_only` | ungrounded | entity_swap | 0.9835 | 1.0000 | 0.0000 | False | 0.9974 | 4 |
| CU3 | colgrep | `code_only` | ungrounded | entity_swap | 0.9471 | 1.0000 | 0.0000 | False | 1.0000 | 4 |
| CU3 | colgrep | `fuse_mean` | ungrounded | entity_swap | 0.9653 | 1.0000 | 0.0000 | False | 0.9987 | 4 |
| CU3 | colgrep | `fuse_max` | ungrounded | entity_swap | 0.9835 | 1.0000 | 0.0000 | False | 1.0000 | 4 |
| CU3 | colgrep | `fuse_min` | ungrounded | entity_swap | 0.9471 | 1.0000 | 0.0000 | False | 0.9974 | 4 |
| CU4 | colgrep | `gte_only` | ungrounded | entity_swap | 0.9765 | 1.0000 | 0.0000 | False | 0.9999 | 4 |
| CU4 | colgrep | `code_only` | ungrounded | entity_swap | 0.9001 | 1.0000 | 0.0000 | False | 1.0000 | 4 |
| CU4 | colgrep | `fuse_mean` | ungrounded | entity_swap | 0.9383 | 1.0000 | 0.0000 | False | 0.9999 | 4 |
| CU4 | colgrep | `fuse_max` | ungrounded | entity_swap | 0.9765 | 1.0000 | 0.0000 | False | 1.0000 | 4 |
| CU4 | colgrep | `fuse_min` | ungrounded | entity_swap | 0.9001 | 1.0000 | 0.0000 | False | 0.9999 | 4 |
| CU5 | colgrep | `gte_only` | ungrounded | parametric | 0.9758 | 1.0000 | 0.0000 | False | 0.9948 | 4 |
| CU5 | colgrep | `code_only` | ungrounded | parametric | 0.9163 | 1.0000 | 0.0000 | False | 0.9998 | 4 |
| CU5 | colgrep | `fuse_mean` | ungrounded | parametric | 0.9461 | 1.0000 | 0.0000 | False | 0.9973 | 4 |
| CU5 | colgrep | `fuse_max` | ungrounded | parametric | 0.9758 | 1.0000 | 0.0000 | False | 0.9998 | 4 |
| CU5 | colgrep | `fuse_min` | ungrounded | parametric | 0.9163 | 1.0000 | 0.0000 | False | 0.9948 | 4 |
| CU6 | colgrep | `gte_only` | ungrounded | parametric | 0.9760 | 1.0000 | 0.0000 | False | 0.9973 | 5 |
| CU6 | colgrep | `code_only` | ungrounded | parametric | 0.9183 | 1.0000 | 0.0000 | False | 1.0000 | 5 |
| CU6 | colgrep | `fuse_mean` | ungrounded | parametric | 0.9472 | 1.0000 | 0.0000 | False | 0.9987 | 5 |
| CU6 | colgrep | `fuse_max` | ungrounded | parametric | 0.9760 | 1.0000 | 0.0000 | False | 1.0000 | 5 |
| CU6 | colgrep | `fuse_min` | ungrounded | parametric | 0.9183 | 1.0000 | 0.0000 | False | 0.9973 | 5 |
| CA1 | colgrep | `gte_only` | ambiguous | partial | 0.9811 | 1.0000 | 0.0000 | False | 0.9983 | 4 |
| CA1 | colgrep | `code_only` | ambiguous | partial | 0.9290 | 1.0000 | 0.0000 | False | 1.0000 | 4 |
| CA1 | colgrep | `fuse_mean` | ambiguous | partial | 0.9550 | 1.0000 | 0.0000 | False | 0.9991 | 4 |
| CA1 | colgrep | `fuse_max` | ambiguous | partial | 0.9811 | 1.0000 | 0.0000 | False | 1.0000 | 4 |
| CA1 | colgrep | `fuse_min` | ambiguous | partial | 0.9290 | 1.0000 | 0.0000 | False | 0.9983 | 4 |
| CA2 | colgrep | `gte_only` | ambiguous | partial | 0.9777 | 1.0000 | 0.0000 | False | 0.9978 | 4 |
| CA2 | colgrep | `code_only` | ambiguous | partial | 0.9271 | 1.0000 | 0.0000 | False | 1.0000 | 4 |
| CA2 | colgrep | `fuse_mean` | ambiguous | partial | 0.9524 | 1.0000 | 0.0000 | False | 0.9989 | 4 |
| CA2 | colgrep | `fuse_max` | ambiguous | partial | 0.9777 | 1.0000 | 0.0000 | False | 1.0000 | 4 |
| CA2 | colgrep | `fuse_min` | ambiguous | partial | 0.9271 | 1.0000 | 0.0000 | False | 0.9978 | 4 |
| CA3 | colgrep | `gte_only` | ambiguous | negation_flip | 0.9815 | 1.0000 | 0.0000 | False | 0.9967 | 5 |
| CA3 | colgrep | `code_only` | ambiguous | negation_flip | 0.9260 | 1.0000 | 0.0000 | False | 0.9999 | 5 |
| CA3 | colgrep | `fuse_mean` | ambiguous | negation_flip | 0.9537 | 1.0000 | 0.0000 | False | 0.9983 | 5 |
| CA3 | colgrep | `fuse_max` | ambiguous | negation_flip | 0.9815 | 1.0000 | 0.0000 | False | 0.9999 | 5 |
| CA3 | colgrep | `fuse_min` | ambiguous | negation_flip | 0.9260 | 1.0000 | 0.0000 | False | 0.9967 | 5 |
| SWG1 | colgrep | `gte_only` | grounded | - | 0.9671 | 1.0000 | 0.0000 | False | 0.9873 | 9 |
| SWG1 | colgrep | `code_only` | grounded | - | 0.8263 | 1.0000 | 0.0000 | False | 0.9438 | 9 |
| SWG1 | colgrep | `fuse_mean` | grounded | - | 0.8967 | 1.0000 | 0.0000 | False | 0.9656 | 9 |
| SWG1 | colgrep | `fuse_max` | grounded | - | 0.9671 | 1.0000 | 0.0000 | False | 0.9873 | 9 |
| SWG1 | colgrep | `fuse_min` | grounded | - | 0.8263 | 1.0000 | 0.0000 | False | 0.9438 | 9 |
| SWU1 | colgrep | `gte_only` | ungrounded | entity_swap | 0.9658 | 1.0000 | 0.0000 | False | 0.9870 | 9 |
| SWU1 | colgrep | `code_only` | ungrounded | entity_swap | 0.8277 | 1.0000 | 0.0000 | False | 0.9400 | 9 |
| SWU1 | colgrep | `fuse_mean` | ungrounded | entity_swap | 0.8967 | 1.0000 | 0.0000 | False | 0.9635 | 9 |
| SWU1 | colgrep | `fuse_max` | ungrounded | entity_swap | 0.9658 | 1.0000 | 0.0000 | False | 0.9870 | 9 |
| SWU1 | colgrep | `fuse_min` | ungrounded | entity_swap | 0.8277 | 1.0000 | 0.0000 | False | 0.9400 | 9 |
| SWG2 | colgrep | `gte_only` | grounded | - | 0.9644 | 1.0000 | 0.0000 | False | 0.9983 | 7 |
| SWG2 | colgrep | `code_only` | grounded | - | 0.8514 | 1.0000 | 0.0000 | False | 0.9927 | 7 |
| SWG2 | colgrep | `fuse_mean` | grounded | - | 0.9079 | 1.0000 | 0.0000 | False | 0.9955 | 7 |
| SWG2 | colgrep | `fuse_max` | grounded | - | 0.9644 | 1.0000 | 0.0000 | False | 0.9983 | 7 |
| SWG2 | colgrep | `fuse_min` | grounded | - | 0.8514 | 1.0000 | 0.0000 | False | 0.9927 | 7 |
| SWU2 | colgrep | `gte_only` | ungrounded | phantom_api | 0.9631 | 1.0000 | 0.0000 | False | 0.9983 | 7 |
| SWU2 | colgrep | `code_only` | ungrounded | phantom_api | 0.8445 | 1.0000 | 0.0000 | False | 0.9927 | 7 |
| SWU2 | colgrep | `fuse_mean` | ungrounded | phantom_api | 0.9038 | 1.0000 | 0.0000 | False | 0.9955 | 7 |
| SWU2 | colgrep | `fuse_max` | ungrounded | phantom_api | 0.9631 | 1.0000 | 0.0000 | False | 0.9983 | 7 |
| SWU2 | colgrep | `fuse_min` | ungrounded | phantom_api | 0.8445 | 1.0000 | 0.0000 | False | 0.9927 | 7 |
| SWG3 | colgrep | `gte_only` | grounded | - | 0.9853 | 1.0000 | 0.0000 | False | 0.9981 | 12 |
| SWG3 | colgrep | `code_only` | grounded | - | 0.9329 | 1.0000 | 0.0000 | False | 0.9924 | 12 |
| SWG3 | colgrep | `fuse_mean` | grounded | - | 0.9591 | 1.0000 | 0.0000 | False | 0.9953 | 12 |
| SWG3 | colgrep | `fuse_max` | grounded | - | 0.9853 | 1.0000 | 0.0000 | False | 0.9981 | 12 |
| SWG3 | colgrep | `fuse_min` | grounded | - | 0.9329 | 1.0000 | 0.0000 | False | 0.9924 | 12 |
| SWU3 | colgrep | `gte_only` | ungrounded | negation_flip | 0.9844 | 1.0000 | 0.0000 | False | 0.9981 | 12 |
| SWU3 | colgrep | `code_only` | ungrounded | negation_flip | 0.9306 | 1.0000 | 0.0000 | False | 0.9915 | 12 |
| SWU3 | colgrep | `fuse_mean` | ungrounded | negation_flip | 0.9575 | 1.0000 | 0.0000 | False | 0.9948 | 12 |
| SWU3 | colgrep | `fuse_max` | ungrounded | negation_flip | 0.9844 | 1.0000 | 0.0000 | False | 0.9981 | 12 |
| SWU3 | colgrep | `fuse_min` | ungrounded | negation_flip | 0.9306 | 1.0000 | 0.0000 | False | 0.9915 | 12 |
| SWG4 | colgrep | `gte_only` | grounded | - | 0.9883 | 1.0000 | 0.0000 | False | 0.9982 | 22 |
| SWG4 | colgrep | `code_only` | grounded | - | 0.9280 | 1.0000 | 0.0000 | False | 0.9841 | 22 |
| SWG4 | colgrep | `fuse_mean` | grounded | - | 0.9581 | 1.0000 | 0.0000 | False | 0.9911 | 22 |
| SWG4 | colgrep | `fuse_max` | grounded | - | 0.9883 | 1.0000 | 0.0000 | False | 0.9982 | 22 |
| SWG4 | colgrep | `fuse_min` | grounded | - | 0.9280 | 1.0000 | 0.0000 | False | 0.9841 | 22 |
| SWU4 | colgrep | `gte_only` | ungrounded | entity_swap | 0.9868 | 1.0000 | 0.0000 | False | 0.9972 | 22 |
| SWU4 | colgrep | `code_only` | ungrounded | entity_swap | 0.9176 | 1.0000 | 0.0000 | False | 0.9876 | 22 |
| SWU4 | colgrep | `fuse_mean` | ungrounded | entity_swap | 0.9522 | 1.0000 | 0.0000 | False | 0.9924 | 22 |
| SWU4 | colgrep | `fuse_max` | ungrounded | entity_swap | 0.9868 | 1.0000 | 0.0000 | False | 0.9972 | 22 |
| SWU4 | colgrep | `fuse_min` | ungrounded | entity_swap | 0.9176 | 1.0000 | 0.0000 | False | 0.9876 | 22 |
| SWG5 | colgrep | `gte_only` | grounded | - | 0.9804 | 1.0000 | 0.0000 | False | 0.9975 | 18 |
| SWG5 | colgrep | `code_only` | grounded | - | 0.8916 | 1.0000 | 0.0000 | False | 0.9721 | 18 |
| SWG5 | colgrep | `fuse_mean` | grounded | - | 0.9360 | 1.0000 | 0.0000 | False | 0.9848 | 18 |
| SWG5 | colgrep | `fuse_max` | grounded | - | 0.9804 | 1.0000 | 0.0000 | False | 0.9975 | 18 |
| SWG5 | colgrep | `fuse_min` | grounded | - | 0.8916 | 1.0000 | 0.0000 | False | 0.9721 | 18 |
| SWU5 | colgrep | `gte_only` | ungrounded | entity_swap | 0.9817 | 1.0000 | 0.0000 | False | 0.9980 | 18 |
| SWU5 | colgrep | `code_only` | ungrounded | entity_swap | 0.9051 | 1.0000 | 0.0000 | False | 0.9725 | 18 |
| SWU5 | colgrep | `fuse_mean` | ungrounded | entity_swap | 0.9434 | 1.0000 | 0.0000 | False | 0.9852 | 18 |
| SWU5 | colgrep | `fuse_max` | ungrounded | entity_swap | 0.9817 | 1.0000 | 0.0000 | False | 0.9980 | 18 |
| SWU5 | colgrep | `fuse_min` | ungrounded | entity_swap | 0.9051 | 1.0000 | 0.0000 | False | 0.9725 | 18 |
| SWG6 | colgrep | `gte_only` | grounded | - | 0.9715 | 1.0000 | 0.0000 | False | 0.9793 | 15 |
| SWG6 | colgrep | `code_only` | grounded | - | 0.8431 | 1.0000 | 0.0000 | False | 0.9576 | 15 |
| SWG6 | colgrep | `fuse_mean` | grounded | - | 0.9073 | 1.0000 | 0.0000 | False | 0.9684 | 15 |
| SWG6 | colgrep | `fuse_max` | grounded | - | 0.9715 | 1.0000 | 0.0000 | False | 0.9793 | 15 |
| SWG6 | colgrep | `fuse_min` | grounded | - | 0.8431 | 1.0000 | 0.0000 | False | 0.9576 | 15 |
| SWU6 | colgrep | `gte_only` | ungrounded | entity_swap | 0.9715 | 1.0000 | 0.0000 | False | 0.9792 | 15 |
| SWU6 | colgrep | `code_only` | ungrounded | entity_swap | 0.8438 | 1.0000 | 0.0000 | False | 0.9576 | 15 |
| SWU6 | colgrep | `fuse_mean` | ungrounded | entity_swap | 0.9077 | 1.0000 | 0.0000 | False | 0.9684 | 15 |
| SWU6 | colgrep | `fuse_max` | ungrounded | entity_swap | 0.9715 | 1.0000 | 0.0000 | False | 0.9792 | 15 |
| SWU6 | colgrep | `fuse_min` | ungrounded | entity_swap | 0.8438 | 1.0000 | 0.0000 | False | 0.9576 | 15 |
| SWG7 | colgrep | `gte_only` | grounded | - | 0.9869 | 1.0000 | 0.0000 | False | 0.9886 | 22 |
| SWG7 | colgrep | `code_only` | grounded | - | 0.9212 | 1.0000 | 0.0000 | False | 0.9516 | 22 |
| SWG7 | colgrep | `fuse_mean` | grounded | - | 0.9541 | 1.0000 | 0.0000 | False | 0.9701 | 22 |
| SWG7 | colgrep | `fuse_max` | grounded | - | 0.9869 | 1.0000 | 0.0000 | False | 0.9886 | 22 |
| SWG7 | colgrep | `fuse_min` | grounded | - | 0.9212 | 1.0000 | 0.0000 | False | 0.9516 | 22 |
| SWU7 | colgrep | `gte_only` | ungrounded | phantom_api | 0.9864 | 1.0000 | 0.0000 | False | 0.9886 | 22 |
| SWU7 | colgrep | `code_only` | ungrounded | phantom_api | 0.9172 | 1.0000 | 0.0000 | False | 0.9516 | 22 |
| SWU7 | colgrep | `fuse_mean` | ungrounded | phantom_api | 0.9518 | 1.0000 | 0.0000 | False | 0.9701 | 22 |
| SWU7 | colgrep | `fuse_max` | ungrounded | phantom_api | 0.9864 | 1.0000 | 0.0000 | False | 0.9886 | 22 |
| SWU7 | colgrep | `fuse_min` | ungrounded | phantom_api | 0.9172 | 1.0000 | 0.0000 | False | 0.9516 | 22 |
| SWG8 | colgrep | `gte_only` | grounded | - | 0.9904 | 1.0000 | 0.0000 | False | 0.9999 | 56 |
| SWG8 | colgrep | `code_only` | grounded | - | 0.9541 | 1.0000 | 0.0000 | False | 1.0000 | 56 |
| SWG8 | colgrep | `fuse_mean` | grounded | - | 0.9722 | 1.0000 | 0.0000 | False | 1.0000 | 56 |
| SWG8 | colgrep | `fuse_max` | grounded | - | 0.9904 | 1.0000 | 0.0000 | False | 1.0000 | 56 |
| SWG8 | colgrep | `fuse_min` | grounded | - | 0.9541 | 1.0000 | 0.0000 | False | 0.9999 | 56 |
| SWU8 | colgrep | `gte_only` | ungrounded | entity_swap | 0.9906 | 1.0000 | 0.0000 | False | 0.9999 | 56 |
| SWU8 | colgrep | `code_only` | ungrounded | entity_swap | 0.9542 | 1.0000 | 0.0000 | False | 1.0000 | 56 |
| SWU8 | colgrep | `fuse_mean` | ungrounded | entity_swap | 0.9724 | 1.0000 | 0.0000 | False | 1.0000 | 56 |
| SWU8 | colgrep | `fuse_max` | ungrounded | entity_swap | 0.9906 | 1.0000 | 0.0000 | False | 1.0000 | 56 |
| SWU8 | colgrep | `fuse_min` | ungrounded | entity_swap | 0.9542 | 1.0000 | 0.0000 | False | 0.9999 | 56 |
| SWG9 | colgrep | `gte_only` | grounded | - | 0.9812 | 1.0000 | 0.0000 | False | 0.9982 | 55 |
| SWG9 | colgrep | `code_only` | grounded | - | 0.9226 | 1.0000 | 0.0000 | False | 0.9854 | 55 |
| SWG9 | colgrep | `fuse_mean` | grounded | - | 0.9519 | 1.0000 | 0.0000 | False | 0.9918 | 55 |
| SWG9 | colgrep | `fuse_max` | grounded | - | 0.9812 | 1.0000 | 0.0000 | False | 0.9982 | 55 |
| SWG9 | colgrep | `fuse_min` | grounded | - | 0.9226 | 1.0000 | 0.0000 | False | 0.9854 | 55 |
| SWU9 | colgrep | `gte_only` | ungrounded | phantom_api | 0.9796 | 1.0000 | 0.0000 | False | 0.9987 | 55 |
| SWU9 | colgrep | `code_only` | ungrounded | phantom_api | 0.9153 | 1.0000 | 0.0000 | False | 0.9681 | 55 |
| SWU9 | colgrep | `fuse_mean` | ungrounded | phantom_api | 0.9474 | 1.0000 | 0.0000 | False | 0.9834 | 55 |
| SWU9 | colgrep | `fuse_max` | ungrounded | phantom_api | 0.9796 | 1.0000 | 0.0000 | False | 0.9987 | 55 |
| SWU9 | colgrep | `fuse_min` | ungrounded | phantom_api | 0.9153 | 1.0000 | 0.0000 | False | 0.9681 | 55 |
| SWG10 | colgrep | `gte_only` | grounded | - | 0.9879 | 1.0000 | 0.0000 | False | 0.9998 | 36 |
| SWG10 | colgrep | `code_only` | grounded | - | 0.9326 | 1.0000 | 0.0000 | False | 1.0000 | 36 |
| SWG10 | colgrep | `fuse_mean` | grounded | - | 0.9603 | 1.0000 | 0.0000 | False | 0.9999 | 36 |
| SWG10 | colgrep | `fuse_max` | grounded | - | 0.9879 | 1.0000 | 0.0000 | False | 1.0000 | 36 |
| SWG10 | colgrep | `fuse_min` | grounded | - | 0.9326 | 1.0000 | 0.0000 | False | 0.9998 | 36 |
| SWU10 | colgrep | `gte_only` | ungrounded | phantom_api | 0.9878 | 1.0000 | 0.0000 | False | 0.9998 | 36 |
| SWU10 | colgrep | `code_only` | ungrounded | phantom_api | 0.9314 | 1.0000 | 0.0000 | False | 1.0000 | 36 |
| SWU10 | colgrep | `fuse_mean` | ungrounded | phantom_api | 0.9596 | 1.0000 | 0.0000 | False | 0.9999 | 36 |
| SWU10 | colgrep | `fuse_max` | ungrounded | phantom_api | 0.9878 | 1.0000 | 0.0000 | False | 1.0000 | 36 |
| SWU10 | colgrep | `fuse_min` | ungrounded | phantom_api | 0.9314 | 1.0000 | 0.0000 | False | 0.9998 | 36 |
| SWG11 | colgrep | `gte_only` | grounded | - | 0.9876 | 1.0000 | 0.0000 | False | 0.9981 | 98 |
| SWG11 | colgrep | `code_only` | grounded | - | 0.9419 | 1.0000 | 0.0000 | False | 1.0000 | 98 |
| SWG11 | colgrep | `fuse_mean` | grounded | - | 0.9647 | 1.0000 | 0.0000 | False | 0.9990 | 98 |
| SWG11 | colgrep | `fuse_max` | grounded | - | 0.9876 | 1.0000 | 0.0000 | False | 1.0000 | 98 |
| SWG11 | colgrep | `fuse_min` | grounded | - | 0.9419 | 1.0000 | 0.0000 | False | 0.9981 | 98 |
| SWU11 | colgrep | `gte_only` | ungrounded | phantom_api | 0.9871 | 1.0000 | 0.0000 | False | 0.9987 | 98 |
| SWU11 | colgrep | `code_only` | ungrounded | phantom_api | 0.9416 | 1.0000 | 0.0000 | False | 1.0000 | 98 |
| SWU11 | colgrep | `fuse_mean` | ungrounded | phantom_api | 0.9643 | 1.0000 | 0.0000 | False | 0.9993 | 98 |
| SWU11 | colgrep | `fuse_max` | ungrounded | phantom_api | 0.9871 | 1.0000 | 0.0000 | False | 1.0000 | 98 |
| SWU11 | colgrep | `fuse_min` | ungrounded | phantom_api | 0.9416 | 1.0000 | 0.0000 | False | 0.9987 | 98 |
| SWG12 | colgrep | `gte_only` | grounded | - | 0.9788 | 1.0000 | 0.0000 | False | 0.9984 | 17 |
| SWG12 | colgrep | `code_only` | grounded | - | 0.8875 | 1.0000 | 0.0000 | False | 0.9922 | 17 |
| SWG12 | colgrep | `fuse_mean` | grounded | - | 0.9332 | 1.0000 | 0.0000 | False | 0.9953 | 17 |
| SWG12 | colgrep | `fuse_max` | grounded | - | 0.9788 | 1.0000 | 0.0000 | False | 0.9984 | 17 |
| SWG12 | colgrep | `fuse_min` | grounded | - | 0.8875 | 1.0000 | 0.0000 | False | 0.9922 | 17 |
| SWU12 | colgrep | `gte_only` | ungrounded | negation_flip | 0.9783 | 1.0000 | 0.0000 | False | 0.9984 | 17 |
| SWU12 | colgrep | `code_only` | ungrounded | negation_flip | 0.8857 | 1.0000 | 0.0000 | False | 0.9917 | 17 |
| SWU12 | colgrep | `fuse_mean` | ungrounded | negation_flip | 0.9320 | 1.0000 | 0.0000 | False | 0.9951 | 17 |
| SWU12 | colgrep | `fuse_max` | ungrounded | negation_flip | 0.9783 | 1.0000 | 0.0000 | False | 0.9984 | 17 |
| SWU12 | colgrep | `fuse_min` | ungrounded | negation_flip | 0.8857 | 1.0000 | 0.0000 | False | 0.9917 | 17 |
| SWG13 | colgrep | `gte_only` | grounded | - | 0.9824 | 1.0000 | 0.0000 | False | 0.9897 | 93 |
| SWG13 | colgrep | `code_only` | grounded | - | 0.9066 | 1.0000 | 0.0000 | False | 0.9629 | 93 |
| SWG13 | colgrep | `fuse_mean` | grounded | - | 0.9445 | 1.0000 | 0.0000 | False | 0.9763 | 93 |
| SWG13 | colgrep | `fuse_max` | grounded | - | 0.9824 | 1.0000 | 0.0000 | False | 0.9897 | 93 |
| SWG13 | colgrep | `fuse_min` | grounded | - | 0.9066 | 1.0000 | 0.0000 | False | 0.9629 | 93 |
| SWU13 | colgrep | `gte_only` | ungrounded | entity_swap | 0.9818 | 1.0000 | 0.0000 | False | 1.0000 | 93 |
| SWU13 | colgrep | `code_only` | ungrounded | entity_swap | 0.9044 | 1.0000 | 0.0000 | False | 0.9826 | 93 |
| SWU13 | colgrep | `fuse_mean` | ungrounded | entity_swap | 0.9431 | 1.0000 | 0.0000 | False | 0.9913 | 93 |
| SWU13 | colgrep | `fuse_max` | ungrounded | entity_swap | 0.9818 | 1.0000 | 0.0000 | False | 1.0000 | 93 |
| SWU13 | colgrep | `fuse_min` | ungrounded | entity_swap | 0.9044 | 1.0000 | 0.0000 | False | 0.9826 | 93 |
| SWG14 | colgrep | `gte_only` | grounded | - | 0.9895 | 1.0000 | 0.0000 | False | 0.9986 | 33 |
| SWG14 | colgrep | `code_only` | grounded | - | 0.9368 | 1.0000 | 0.0000 | False | 0.9934 | 33 |
| SWG14 | colgrep | `fuse_mean` | grounded | - | 0.9632 | 1.0000 | 0.0000 | False | 0.9960 | 33 |
| SWG14 | colgrep | `fuse_max` | grounded | - | 0.9895 | 1.0000 | 0.0000 | False | 0.9986 | 33 |
| SWG14 | colgrep | `fuse_min` | grounded | - | 0.9368 | 1.0000 | 0.0000 | False | 0.9934 | 33 |
| SWU14 | colgrep | `gte_only` | ungrounded | entity_swap | 0.9895 | 1.0000 | 0.0000 | False | 0.9986 | 33 |
| SWU14 | colgrep | `code_only` | ungrounded | entity_swap | 0.9364 | 1.0000 | 0.0000 | False | 0.9934 | 33 |
| SWU14 | colgrep | `fuse_mean` | ungrounded | entity_swap | 0.9629 | 1.0000 | 0.0000 | False | 0.9960 | 33 |
| SWU14 | colgrep | `fuse_max` | ungrounded | entity_swap | 0.9895 | 1.0000 | 0.0000 | False | 0.9986 | 33 |
| SWU14 | colgrep | `fuse_min` | ungrounded | entity_swap | 0.9364 | 1.0000 | 0.0000 | False | 0.9934 | 33 |
| SWG15 | colgrep | `gte_only` | grounded | - | 0.9899 | 1.0000 | 0.0000 | False | 0.9978 | 75 |
| SWG15 | colgrep | `code_only` | grounded | - | 0.9428 | 1.0000 | 0.0000 | False | 0.9884 | 75 |
| SWG15 | colgrep | `fuse_mean` | grounded | - | 0.9664 | 1.0000 | 0.0000 | False | 0.9931 | 75 |
| SWG15 | colgrep | `fuse_max` | grounded | - | 0.9899 | 1.0000 | 0.0000 | False | 0.9978 | 75 |
| SWG15 | colgrep | `fuse_min` | grounded | - | 0.9428 | 1.0000 | 0.0000 | False | 0.9884 | 75 |
| SWU15 | colgrep | `gte_only` | ungrounded | entity_swap | 0.9897 | 1.0000 | 0.0000 | False | 0.9978 | 75 |
| SWU15 | colgrep | `code_only` | ungrounded | entity_swap | 0.9413 | 1.0000 | 0.0000 | False | 0.9884 | 75 |
| SWU15 | colgrep | `fuse_mean` | ungrounded | entity_swap | 0.9655 | 1.0000 | 0.0000 | False | 0.9931 | 75 |
| SWU15 | colgrep | `fuse_max` | ungrounded | entity_swap | 0.9897 | 1.0000 | 0.0000 | False | 0.9978 | 75 |
| SWU15 | colgrep | `fuse_min` | ungrounded | entity_swap | 0.9413 | 1.0000 | 0.0000 | False | 0.9884 | 75 |
