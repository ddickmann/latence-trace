# Coding-Agent Groundedness: Scorer Stack x Chunker

> NO — the flagship colgrep x fuse_mean cell did not clear the RAG-style gate: reverse_context AUROC=0.574, phantom-API precision@0.35=0.00, ungrounded minus grounded unused-ratio delta=+0.0000. Stacking lift vs gte_only=-0.0265. Best fusion rule on colgrep is `fuse_max` (AUROC=0.601); consider flipping the flagship.

- run tag: `chunk_size_sweep_v1__b128_gte`
- primary scorer: `lightonai/GTE-ModernColBERT-v1`
- orthogonal scorer: `lightonai/LateOn-Code-edge` (available: `True`)
- shared chunk budget: `128` tokens (GTE tokenizer)
- max SWE-bench instances: `15`
- phantom threshold: `0.35`
- flagship cell: `colgrep` x `fuse_mean` (present: `True`)

## Headline matrix

| chunker | scorer | AUROC | grounded cov | cov delta | unused delta | phantom@thr | grounded held-out rate | p95 ms | support units |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sentence_packed | `gte_only` | 0.6032 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1192.34 | 9.9 |
| sentence_packed | `code_only` | 0.6058 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1092.33 | 9.9 |
| sentence_packed | `fuse_mean` | 0.6138 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1098.29 | 9.9 |
| sentence_packed | `fuse_max` | 0.6032 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1098.29 | 9.9 |
| sentence_packed | `fuse_min` | 0.6058 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1098.29 | 9.9 |
| colgrep | `gte_only` | 0.6005 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1299.54 | 16.2 |
| colgrep | `code_only` | 0.5741 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1207.99 | 16.2 |
| colgrep | `fuse_mean` (flagship) | 0.5741 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1293.10 | 16.2 |
| colgrep | `fuse_max` | 0.6005 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1293.10 | 16.2 |
| colgrep | `fuse_min` | 0.5741 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1293.10 | 16.2 |

`*` marks cells that pass `reverse_context AUROC >= 0.90`.

## Score distribution diagnostics (reverse_context)

| chunker | scorer | grounded mean | grounded p10-p90 | ungrounded mean | ungrounded p10-p90 | separation | saturation@0.95 |
|---|---|---:|---|---:|---|---:|---:|
| sentence_packed | `gte_only` | 0.9815 | 0.9743-0.9877 | 0.9791 | 0.9702-0.9868 | +0.0024 | 1.0000 |
| sentence_packed | `code_only` | 0.9294 | 0.9122-0.9477 | 0.9231 | 0.9016-0.9466 | +0.0063 | 0.0256 |
| sentence_packed | `fuse_mean` | 0.9555 | 0.9449-0.9678 | 0.9511 | 0.9347-0.9666 | +0.0044 | 0.6923 |
| sentence_packed | `fuse_max` | 0.9815 | 0.9743-0.9877 | 0.9791 | 0.9702-0.9868 | +0.0024 | 1.0000 |
| sentence_packed | `fuse_min` | 0.9294 | 0.9122-0.9477 | 0.9231 | 0.9016-0.9466 | +0.0063 | 0.0256 |
| colgrep | `gte_only` | 0.9814 | 0.9676-0.9879 | 0.9798 | 0.9699-0.9873 | +0.0016 | 1.0000 |
| colgrep | `code_only` | 0.9270 | 0.8788-0.9497 | 0.9214 | 0.8751-0.9480 | +0.0056 | 0.0513 |
| colgrep | `fuse_mean` | 0.9542 | 0.9231-0.9687 | 0.9506 | 0.9208-0.9683 | +0.0036 | 0.7179 |
| colgrep | `fuse_max` | 0.9814 | 0.9676-0.9879 | 0.9798 | 0.9699-0.9873 | +0.0016 | 1.0000 |
| colgrep | `fuse_min` | 0.9270 | 0.8788-0.9497 | 0.9214 | 0.8751-0.9480 | +0.0056 | 0.0513 |

`separation` is mean(grounded) − mean(ungrounded) on `reverse_context`. `saturation@0.95` is the fraction of anchor cases (grounded + ungrounded) whose score ≥ 0.95 — a ceiling effect indicator.

## Stacking lift (fuse_* minus gte_only, per chunker)

| chunker | rule | AUROC lift | unused-delta lift | phantom precision lift |
|---|---|---:|---:|---:|
| sentence_packed | fuse_mean | +0.0106 | +0.0000 | +0.0000 |
| sentence_packed | fuse_max | +0.0000 | +0.0000 | +0.0000 |
| sentence_packed | fuse_min | +0.0026 | +0.0000 | +0.0000 |
| colgrep | fuse_mean | -0.0265 | +0.0000 | +0.0000 |
| colgrep | fuse_max | +0.0000 | +0.0000 | +0.0000 |
| colgrep | fuse_min | -0.0265 | +0.0000 | +0.0000 |

## Chunker lift (colgrep minus sentence_packed, per scorer)

| scorer | AUROC lift | unused-delta lift | phantom precision lift |
|---|---:|---:|---:|
| `gte_only` | -0.0026 | +0.0000 | +0.0000 |
| `code_only` | -0.0317 | +0.0000 | +0.0000 |
| `fuse_mean` | -0.0397 | +0.0000 | +0.0000 |
| `fuse_max` | -0.0026 | +0.0000 | +0.0000 |
| `fuse_min` | -0.0317 | +0.0000 | +0.0000 |

## Subcategory snapshot (per cell)

### `sentence_packed` x `gte_only`

| subcategory | count | reverse_context | coverage | unused ratio | max evidence |
|---|---:|---:|---:|---:|---:|
| entity_swap | 10 | 0.9806 | 1.0000 | 0.0000 | 0.9979 |
| grounded | 18 | 0.9815 | 1.0000 | 0.0000 | 0.9982 |
| negation_flip | 3 | 0.9783 | 1.0000 | 0.0000 | 0.9972 |
| parametric | 2 | 0.9697 | 1.0000 | 0.0000 | 0.9961 |
| partial | 2 | 0.9696 | 1.0000 | 0.0000 | 0.9976 |
| phantom_api | 7 | 0.9790 | 1.0000 | 0.0000 | 0.9983 |

### `sentence_packed` x `code_only`

| subcategory | count | reverse_context | coverage | unused ratio | max evidence |
|---|---:|---:|---:|---:|---:|
| entity_swap | 10 | 0.9258 | 1.0000 | 0.0000 | 0.9897 |
| grounded | 18 | 0.9294 | 1.0000 | 0.0000 | 0.9931 |
| negation_flip | 3 | 0.9111 | 1.0000 | 0.0000 | 0.9876 |
| parametric | 2 | 0.9081 | 1.0000 | 0.0000 | 0.9863 |
| partial | 2 | 0.9193 | 1.0000 | 0.0000 | 0.9981 |
| phantom_api | 7 | 0.9235 | 1.0000 | 0.0000 | 0.9844 |

### `sentence_packed` x `fuse_mean`

| subcategory | count | reverse_context | coverage | unused ratio | max evidence |
|---|---:|---:|---:|---:|---:|
| entity_swap | 10 | 0.9532 | 1.0000 | 0.0000 | 0.9938 |
| grounded | 18 | 0.9555 | 1.0000 | 0.0000 | 0.9957 |
| negation_flip | 3 | 0.9447 | 1.0000 | 0.0000 | 0.9924 |
| parametric | 2 | 0.9389 | 1.0000 | 0.0000 | 0.9912 |
| partial | 2 | 0.9445 | 1.0000 | 0.0000 | 0.9978 |
| phantom_api | 7 | 0.9512 | 1.0000 | 0.0000 | 0.9913 |

### `sentence_packed` x `fuse_max`

| subcategory | count | reverse_context | coverage | unused ratio | max evidence |
|---|---:|---:|---:|---:|---:|
| entity_swap | 10 | 0.9806 | 1.0000 | 0.0000 | 0.9983 |
| grounded | 18 | 0.9815 | 1.0000 | 0.0000 | 0.9988 |
| negation_flip | 3 | 0.9783 | 1.0000 | 0.0000 | 0.9972 |
| parametric | 2 | 0.9697 | 1.0000 | 0.0000 | 0.9978 |
| partial | 2 | 0.9696 | 1.0000 | 0.0000 | 0.9991 |
| phantom_api | 7 | 0.9790 | 1.0000 | 0.0000 | 0.9983 |

### `sentence_packed` x `fuse_min`

| subcategory | count | reverse_context | coverage | unused ratio | max evidence |
|---|---:|---:|---:|---:|---:|
| entity_swap | 10 | 0.9258 | 1.0000 | 0.0000 | 0.9893 |
| grounded | 18 | 0.9294 | 1.0000 | 0.0000 | 0.9926 |
| negation_flip | 3 | 0.9111 | 1.0000 | 0.0000 | 0.9876 |
| parametric | 2 | 0.9081 | 1.0000 | 0.0000 | 0.9847 |
| partial | 2 | 0.9193 | 1.0000 | 0.0000 | 0.9966 |
| phantom_api | 7 | 0.9235 | 1.0000 | 0.0000 | 0.9844 |

### `colgrep` x `gte_only`

| subcategory | count | reverse_context | coverage | unused ratio | max evidence |
|---|---:|---:|---:|---:|---:|
| entity_swap | 10 | 0.9802 | 1.0000 | 0.0000 | 0.9978 |
| grounded | 18 | 0.9814 | 1.0000 | 0.0000 | 0.9984 |
| negation_flip | 3 | 0.9793 | 1.0000 | 0.0000 | 0.9982 |
| parametric | 2 | 0.9789 | 1.0000 | 0.0000 | 0.9986 |
| partial | 2 | 0.9793 | 1.0000 | 0.0000 | 0.9981 |
| phantom_api | 7 | 0.9799 | 1.0000 | 0.0000 | 0.9988 |

### `colgrep` x `code_only`

| subcategory | count | reverse_context | coverage | unused ratio | max evidence |
|---|---:|---:|---:|---:|---:|
| entity_swap | 10 | 0.9216 | 1.0000 | 0.0000 | 0.9913 |
| grounded | 18 | 0.9270 | 1.0000 | 0.0000 | 0.9928 |
| negation_flip | 3 | 0.9188 | 1.0000 | 0.0000 | 0.9947 |
| parametric | 2 | 0.9273 | 1.0000 | 0.0000 | 0.9999 |
| partial | 2 | 0.9385 | 1.0000 | 0.0000 | 0.9999 |
| phantom_api | 7 | 0.9206 | 1.0000 | 0.0000 | 0.9904 |

### `colgrep` x `fuse_mean`

| subcategory | count | reverse_context | coverage | unused ratio | max evidence |
|---|---:|---:|---:|---:|---:|
| entity_swap | 10 | 0.9509 | 1.0000 | 0.0000 | 0.9946 |
| grounded | 18 | 0.9542 | 1.0000 | 0.0000 | 0.9956 |
| negation_flip | 3 | 0.9490 | 1.0000 | 0.0000 | 0.9964 |
| parametric | 2 | 0.9531 | 1.0000 | 0.0000 | 0.9992 |
| partial | 2 | 0.9589 | 1.0000 | 0.0000 | 0.9990 |
| phantom_api | 7 | 0.9503 | 1.0000 | 0.0000 | 0.9946 |

### `colgrep` x `fuse_max`

| subcategory | count | reverse_context | coverage | unused ratio | max evidence |
|---|---:|---:|---:|---:|---:|
| entity_swap | 10 | 0.9802 | 1.0000 | 0.0000 | 0.9986 |
| grounded | 18 | 0.9814 | 1.0000 | 0.0000 | 0.9989 |
| negation_flip | 3 | 0.9793 | 1.0000 | 0.0000 | 0.9990 |
| parametric | 2 | 0.9789 | 1.0000 | 0.0000 | 0.9999 |
| partial | 2 | 0.9793 | 1.0000 | 0.0000 | 0.9999 |
| phantom_api | 7 | 0.9799 | 1.0000 | 0.0000 | 0.9991 |

### `colgrep` x `fuse_min`

| subcategory | count | reverse_context | coverage | unused ratio | max evidence |
|---|---:|---:|---:|---:|---:|
| entity_swap | 10 | 0.9216 | 1.0000 | 0.0000 | 0.9905 |
| grounded | 18 | 0.9270 | 1.0000 | 0.0000 | 0.9924 |
| negation_flip | 3 | 0.9188 | 1.0000 | 0.0000 | 0.9939 |
| parametric | 2 | 0.9273 | 1.0000 | 0.0000 | 0.9986 |
| partial | 2 | 0.9385 | 1.0000 | 0.0000 | 0.9981 |
| phantom_api | 7 | 0.9206 | 1.0000 | 0.0000 | 0.9900 |

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
| CG1 | sentence_packed | `gte_only` | grounded | - | 0.9792 | 1.0000 | 0.0000 | False | 0.9990 | 1 |
| CG1 | sentence_packed | `code_only` | grounded | - | 0.9246 | 1.0000 | 0.0000 | False | 1.0000 | 1 |
| CG1 | sentence_packed | `fuse_mean` | grounded | - | 0.9519 | 1.0000 | 0.0000 | False | 0.9995 | 1 |
| CG1 | sentence_packed | `fuse_max` | grounded | - | 0.9792 | 1.0000 | 0.0000 | False | 1.0000 | 1 |
| CG1 | sentence_packed | `fuse_min` | grounded | - | 0.9246 | 1.0000 | 0.0000 | False | 0.9990 | 1 |
| CG2 | sentence_packed | `gte_only` | grounded | - | 0.9746 | 1.0000 | 0.0000 | False | 0.9982 | 2 |
| CG2 | sentence_packed | `code_only` | grounded | - | 0.9316 | 1.0000 | 0.0000 | False | 0.9998 | 2 |
| CG2 | sentence_packed | `fuse_mean` | grounded | - | 0.9531 | 1.0000 | 0.0000 | False | 0.9990 | 2 |
| CG2 | sentence_packed | `fuse_max` | grounded | - | 0.9746 | 1.0000 | 0.0000 | False | 0.9998 | 2 |
| CG2 | sentence_packed | `fuse_min` | grounded | - | 0.9316 | 1.0000 | 0.0000 | False | 0.9982 | 2 |
| CG3 | sentence_packed | `gte_only` | grounded | - | 0.9707 | 1.0000 | 0.0000 | False | 0.9981 | 1 |
| CG3 | sentence_packed | `code_only` | grounded | - | 0.8725 | 1.0000 | 0.0000 | False | 1.0000 | 1 |
| CG3 | sentence_packed | `fuse_mean` | grounded | - | 0.9216 | 1.0000 | 0.0000 | False | 0.9991 | 1 |
| CG3 | sentence_packed | `fuse_max` | grounded | - | 0.9707 | 1.0000 | 0.0000 | False | 1.0000 | 1 |
| CG3 | sentence_packed | `fuse_min` | grounded | - | 0.8725 | 1.0000 | 0.0000 | False | 0.9981 | 1 |
| CU1 | sentence_packed | `gte_only` | ungrounded | phantom_api | 0.9731 | 1.0000 | 0.0000 | False | 0.9982 | 1 |
| CU1 | sentence_packed | `code_only` | ungrounded | phantom_api | 0.9088 | 1.0000 | 0.0000 | False | 0.9650 | 1 |
| CU1 | sentence_packed | `fuse_mean` | ungrounded | phantom_api | 0.9409 | 1.0000 | 0.0000 | False | 0.9816 | 1 |
| CU1 | sentence_packed | `fuse_max` | ungrounded | phantom_api | 0.9731 | 1.0000 | 0.0000 | False | 0.9982 | 1 |
| CU1 | sentence_packed | `fuse_min` | ungrounded | phantom_api | 0.9088 | 1.0000 | 0.0000 | False | 0.9650 | 1 |
| CU2 | sentence_packed | `gte_only` | ungrounded | phantom_api | 0.9660 | 1.0000 | 0.0000 | False | 0.9975 | 1 |
| CU2 | sentence_packed | `code_only` | ungrounded | phantom_api | 0.8696 | 1.0000 | 0.0000 | False | 0.9833 | 1 |
| CU2 | sentence_packed | `fuse_mean` | ungrounded | phantom_api | 0.9178 | 1.0000 | 0.0000 | False | 0.9904 | 1 |
| CU2 | sentence_packed | `fuse_max` | ungrounded | phantom_api | 0.9660 | 1.0000 | 0.0000 | False | 0.9975 | 1 |
| CU2 | sentence_packed | `fuse_min` | ungrounded | phantom_api | 0.8696 | 1.0000 | 0.0000 | False | 0.9833 | 1 |
| CU3 | sentence_packed | `gte_only` | ungrounded | entity_swap | 0.9760 | 1.0000 | 0.0000 | False | 0.9973 | 1 |
| CU3 | sentence_packed | `code_only` | ungrounded | entity_swap | 0.9264 | 1.0000 | 0.0000 | False | 1.0000 | 1 |
| CU3 | sentence_packed | `fuse_mean` | ungrounded | entity_swap | 0.9512 | 1.0000 | 0.0000 | False | 0.9986 | 1 |
| CU3 | sentence_packed | `fuse_max` | ungrounded | entity_swap | 0.9760 | 1.0000 | 0.0000 | False | 1.0000 | 1 |
| CU3 | sentence_packed | `fuse_min` | ungrounded | entity_swap | 0.9264 | 1.0000 | 0.0000 | False | 0.9973 | 1 |
| CU4 | sentence_packed | `gte_only` | ungrounded | entity_swap | 0.9702 | 1.0000 | 0.0000 | False | 0.9976 | 1 |
| CU4 | sentence_packed | `code_only` | ungrounded | entity_swap | 0.8886 | 1.0000 | 0.0000 | False | 0.9863 | 1 |
| CU4 | sentence_packed | `fuse_mean` | ungrounded | entity_swap | 0.9294 | 1.0000 | 0.0000 | False | 0.9920 | 1 |
| CU4 | sentence_packed | `fuse_max` | ungrounded | entity_swap | 0.9702 | 1.0000 | 0.0000 | False | 0.9976 | 1 |
| CU4 | sentence_packed | `fuse_min` | ungrounded | entity_swap | 0.8886 | 1.0000 | 0.0000 | False | 0.9863 | 1 |
| CU5 | sentence_packed | `gte_only` | ungrounded | parametric | 0.9678 | 1.0000 | 0.0000 | False | 0.9955 | 1 |
| CU5 | sentence_packed | `code_only` | ungrounded | parametric | 0.9016 | 1.0000 | 0.0000 | False | 0.9726 | 1 |
| CU5 | sentence_packed | `fuse_mean` | ungrounded | parametric | 0.9347 | 1.0000 | 0.0000 | False | 0.9841 | 1 |
| CU5 | sentence_packed | `fuse_max` | ungrounded | parametric | 0.9678 | 1.0000 | 0.0000 | False | 0.9955 | 1 |
| CU5 | sentence_packed | `fuse_min` | ungrounded | parametric | 0.9016 | 1.0000 | 0.0000 | False | 0.9726 | 1 |
| CU6 | sentence_packed | `gte_only` | ungrounded | parametric | 0.9717 | 1.0000 | 0.0000 | False | 0.9967 | 1 |
| CU6 | sentence_packed | `code_only` | ungrounded | parametric | 0.9146 | 1.0000 | 0.0000 | False | 1.0000 | 1 |
| CU6 | sentence_packed | `fuse_mean` | ungrounded | parametric | 0.9431 | 1.0000 | 0.0000 | False | 0.9983 | 1 |
| CU6 | sentence_packed | `fuse_max` | ungrounded | parametric | 0.9717 | 1.0000 | 0.0000 | False | 1.0000 | 1 |
| CU6 | sentence_packed | `fuse_min` | ungrounded | parametric | 0.9146 | 1.0000 | 0.0000 | False | 0.9967 | 1 |
| CA1 | sentence_packed | `gte_only` | ambiguous | partial | 0.9672 | 1.0000 | 0.0000 | False | 0.9981 | 1 |
| CA1 | sentence_packed | `code_only` | ambiguous | partial | 0.9100 | 1.0000 | 0.0000 | False | 0.9961 | 1 |
| CA1 | sentence_packed | `fuse_mean` | ambiguous | partial | 0.9386 | 1.0000 | 0.0000 | False | 0.9971 | 1 |
| CA1 | sentence_packed | `fuse_max` | ambiguous | partial | 0.9672 | 1.0000 | 0.0000 | False | 0.9981 | 1 |
| CA1 | sentence_packed | `fuse_min` | ambiguous | partial | 0.9100 | 1.0000 | 0.0000 | False | 0.9961 | 1 |
| CA2 | sentence_packed | `gte_only` | ambiguous | partial | 0.9721 | 1.0000 | 0.0000 | False | 0.9971 | 1 |
| CA2 | sentence_packed | `code_only` | ambiguous | partial | 0.9286 | 1.0000 | 0.0000 | False | 1.0000 | 1 |
| CA2 | sentence_packed | `fuse_mean` | ambiguous | partial | 0.9504 | 1.0000 | 0.0000 | False | 0.9986 | 1 |
| CA2 | sentence_packed | `fuse_max` | ambiguous | partial | 0.9721 | 1.0000 | 0.0000 | False | 1.0000 | 1 |
| CA2 | sentence_packed | `fuse_min` | ambiguous | partial | 0.9286 | 1.0000 | 0.0000 | False | 0.9971 | 1 |
| CA3 | sentence_packed | `gte_only` | ambiguous | negation_flip | 0.9719 | 1.0000 | 0.0000 | False | 0.9966 | 1 |
| CA3 | sentence_packed | `code_only` | ambiguous | negation_flip | 0.8871 | 1.0000 | 0.0000 | False | 0.9861 | 1 |
| CA3 | sentence_packed | `fuse_mean` | ambiguous | negation_flip | 0.9295 | 1.0000 | 0.0000 | False | 0.9914 | 1 |
| CA3 | sentence_packed | `fuse_max` | ambiguous | negation_flip | 0.9719 | 1.0000 | 0.0000 | False | 0.9966 | 1 |
| CA3 | sentence_packed | `fuse_min` | ambiguous | negation_flip | 0.8871 | 1.0000 | 0.0000 | False | 0.9861 | 1 |
| SWG1 | sentence_packed | `gte_only` | grounded | - | 0.9736 | 1.0000 | 0.0000 | False | 0.9970 | 5 |
| SWG1 | sentence_packed | `code_only` | grounded | - | 0.9106 | 1.0000 | 0.0000 | False | 0.9863 | 5 |
| SWG1 | sentence_packed | `fuse_mean` | grounded | - | 0.9421 | 1.0000 | 0.0000 | False | 0.9917 | 5 |
| SWG1 | sentence_packed | `fuse_max` | grounded | - | 0.9736 | 1.0000 | 0.0000 | False | 0.9970 | 5 |
| SWG1 | sentence_packed | `fuse_min` | grounded | - | 0.9106 | 1.0000 | 0.0000 | False | 0.9863 | 5 |
| SWU1 | sentence_packed | `gte_only` | ungrounded | entity_swap | 0.9765 | 1.0000 | 0.0000 | False | 0.9998 | 5 |
| SWU1 | sentence_packed | `code_only` | ungrounded | entity_swap | 0.9195 | 1.0000 | 0.0000 | False | 0.9883 | 5 |
| SWU1 | sentence_packed | `fuse_mean` | ungrounded | entity_swap | 0.9480 | 1.0000 | 0.0000 | False | 0.9940 | 5 |
| SWU1 | sentence_packed | `fuse_max` | ungrounded | entity_swap | 0.9765 | 1.0000 | 0.0000 | False | 0.9998 | 5 |
| SWU1 | sentence_packed | `fuse_min` | ungrounded | entity_swap | 0.9195 | 1.0000 | 0.0000 | False | 0.9883 | 5 |
| SWG2 | sentence_packed | `gte_only` | grounded | - | 0.9756 | 1.0000 | 0.0000 | False | 0.9994 | 4 |
| SWG2 | sentence_packed | `code_only` | grounded | - | 0.9191 | 1.0000 | 0.0000 | False | 0.9767 | 4 |
| SWG2 | sentence_packed | `fuse_mean` | grounded | - | 0.9473 | 1.0000 | 0.0000 | False | 0.9881 | 4 |
| SWG2 | sentence_packed | `fuse_max` | grounded | - | 0.9756 | 1.0000 | 0.0000 | False | 0.9994 | 4 |
| SWG2 | sentence_packed | `fuse_min` | grounded | - | 0.9191 | 1.0000 | 0.0000 | False | 0.9767 | 4 |
| SWU2 | sentence_packed | `gte_only` | ungrounded | phantom_api | 0.9741 | 1.0000 | 0.0000 | False | 0.9994 | 4 |
| SWU2 | sentence_packed | `code_only` | ungrounded | phantom_api | 0.9187 | 1.0000 | 0.0000 | False | 0.9718 | 4 |
| SWU2 | sentence_packed | `fuse_mean` | ungrounded | phantom_api | 0.9464 | 1.0000 | 0.0000 | False | 0.9856 | 4 |
| SWU2 | sentence_packed | `fuse_max` | ungrounded | phantom_api | 0.9741 | 1.0000 | 0.0000 | False | 0.9994 | 4 |
| SWU2 | sentence_packed | `fuse_min` | ungrounded | phantom_api | 0.9187 | 1.0000 | 0.0000 | False | 0.9718 | 4 |
| SWG3 | sentence_packed | `gte_only` | grounded | - | 0.9826 | 1.0000 | 0.0000 | False | 0.9978 | 4 |
| SWG3 | sentence_packed | `code_only` | grounded | - | 0.9311 | 1.0000 | 0.0000 | False | 0.9931 | 4 |
| SWG3 | sentence_packed | `fuse_mean` | grounded | - | 0.9568 | 1.0000 | 0.0000 | False | 0.9954 | 4 |
| SWG3 | sentence_packed | `fuse_max` | grounded | - | 0.9826 | 1.0000 | 0.0000 | False | 0.9978 | 4 |
| SWG3 | sentence_packed | `fuse_min` | grounded | - | 0.9311 | 1.0000 | 0.0000 | False | 0.9931 | 4 |
| SWU3 | sentence_packed | `gte_only` | ungrounded | negation_flip | 0.9827 | 1.0000 | 0.0000 | False | 0.9978 | 4 |
| SWU3 | sentence_packed | `code_only` | ungrounded | negation_flip | 0.9251 | 1.0000 | 0.0000 | False | 0.9931 | 4 |
| SWU3 | sentence_packed | `fuse_mean` | ungrounded | negation_flip | 0.9539 | 1.0000 | 0.0000 | False | 0.9954 | 4 |
| SWU3 | sentence_packed | `fuse_max` | ungrounded | negation_flip | 0.9827 | 1.0000 | 0.0000 | False | 0.9978 | 4 |
| SWU3 | sentence_packed | `fuse_min` | ungrounded | negation_flip | 0.9251 | 1.0000 | 0.0000 | False | 0.9931 | 4 |
| SWG4 | sentence_packed | `gte_only` | grounded | - | 0.9828 | 1.0000 | 0.0000 | False | 0.9963 | 7 |
| SWG4 | sentence_packed | `code_only` | grounded | - | 0.9371 | 1.0000 | 0.0000 | False | 1.0000 | 7 |
| SWG4 | sentence_packed | `fuse_mean` | grounded | - | 0.9599 | 1.0000 | 0.0000 | False | 0.9982 | 7 |
| SWG4 | sentence_packed | `fuse_max` | grounded | - | 0.9828 | 1.0000 | 0.0000 | False | 1.0000 | 7 |
| SWG4 | sentence_packed | `fuse_min` | grounded | - | 0.9371 | 1.0000 | 0.0000 | False | 0.9963 | 7 |
| SWU4 | sentence_packed | `gte_only` | ungrounded | entity_swap | 0.9824 | 1.0000 | 0.0000 | False | 0.9963 | 7 |
| SWU4 | sentence_packed | `code_only` | ungrounded | entity_swap | 0.9328 | 1.0000 | 0.0000 | False | 0.9759 | 7 |
| SWU4 | sentence_packed | `fuse_mean` | ungrounded | entity_swap | 0.9576 | 1.0000 | 0.0000 | False | 0.9861 | 7 |
| SWU4 | sentence_packed | `fuse_max` | ungrounded | entity_swap | 0.9824 | 1.0000 | 0.0000 | False | 0.9963 | 7 |
| SWU4 | sentence_packed | `fuse_min` | ungrounded | entity_swap | 0.9328 | 1.0000 | 0.0000 | False | 0.9759 | 7 |
| SWG5 | sentence_packed | `gte_only` | grounded | - | 0.9794 | 1.0000 | 0.0000 | False | 0.9987 | 8 |
| SWG5 | sentence_packed | `code_only` | grounded | - | 0.9129 | 1.0000 | 0.0000 | False | 1.0000 | 8 |
| SWG5 | sentence_packed | `fuse_mean` | grounded | - | 0.9461 | 1.0000 | 0.0000 | False | 0.9993 | 8 |
| SWG5 | sentence_packed | `fuse_max` | grounded | - | 0.9794 | 1.0000 | 0.0000 | False | 1.0000 | 8 |
| SWG5 | sentence_packed | `fuse_min` | grounded | - | 0.9129 | 1.0000 | 0.0000 | False | 0.9987 | 8 |
| SWU5 | sentence_packed | `gte_only` | ungrounded | entity_swap | 0.9787 | 1.0000 | 0.0000 | False | 0.9987 | 8 |
| SWU5 | sentence_packed | `code_only` | ungrounded | entity_swap | 0.9101 | 1.0000 | 0.0000 | False | 1.0000 | 8 |
| SWU5 | sentence_packed | `fuse_mean` | ungrounded | entity_swap | 0.9444 | 1.0000 | 0.0000 | False | 0.9993 | 8 |
| SWU5 | sentence_packed | `fuse_max` | ungrounded | entity_swap | 0.9787 | 1.0000 | 0.0000 | False | 1.0000 | 8 |
| SWU5 | sentence_packed | `fuse_min` | ungrounded | entity_swap | 0.9101 | 1.0000 | 0.0000 | False | 0.9987 | 8 |
| SWG6 | sentence_packed | `gte_only` | grounded | - | 0.9823 | 1.0000 | 0.0000 | False | 0.9984 | 8 |
| SWG6 | sentence_packed | `code_only` | grounded | - | 0.9400 | 1.0000 | 0.0000 | False | 0.9946 | 8 |
| SWG6 | sentence_packed | `fuse_mean` | grounded | - | 0.9611 | 1.0000 | 0.0000 | False | 0.9965 | 8 |
| SWG6 | sentence_packed | `fuse_max` | grounded | - | 0.9823 | 1.0000 | 0.0000 | False | 0.9984 | 8 |
| SWG6 | sentence_packed | `fuse_min` | grounded | - | 0.9400 | 1.0000 | 0.0000 | False | 0.9946 | 8 |
| SWU6 | sentence_packed | `gte_only` | ungrounded | entity_swap | 0.9822 | 1.0000 | 0.0000 | False | 0.9984 | 8 |
| SWU6 | sentence_packed | `code_only` | ungrounded | entity_swap | 0.9401 | 1.0000 | 0.0000 | False | 0.9945 | 8 |
| SWU6 | sentence_packed | `fuse_mean` | ungrounded | entity_swap | 0.9612 | 1.0000 | 0.0000 | False | 0.9964 | 8 |
| SWU6 | sentence_packed | `fuse_max` | ungrounded | entity_swap | 0.9822 | 1.0000 | 0.0000 | False | 0.9984 | 8 |
| SWU6 | sentence_packed | `fuse_min` | ungrounded | entity_swap | 0.9401 | 1.0000 | 0.0000 | False | 0.9945 | 8 |
| SWG7 | sentence_packed | `gte_only` | grounded | - | 0.9880 | 1.0000 | 0.0000 | False | 0.9985 | 10 |
| SWG7 | sentence_packed | `code_only` | grounded | - | 0.9518 | 1.0000 | 0.0000 | False | 0.9944 | 10 |
| SWG7 | sentence_packed | `fuse_mean` | grounded | - | 0.9699 | 1.0000 | 0.0000 | False | 0.9965 | 10 |
| SWG7 | sentence_packed | `fuse_max` | grounded | - | 0.9880 | 1.0000 | 0.0000 | False | 0.9985 | 10 |
| SWG7 | sentence_packed | `fuse_min` | grounded | - | 0.9518 | 1.0000 | 0.0000 | False | 0.9944 | 10 |
| SWU7 | sentence_packed | `gte_only` | ungrounded | phantom_api | 0.9866 | 1.0000 | 0.0000 | False | 0.9982 | 10 |
| SWU7 | sentence_packed | `code_only` | ungrounded | phantom_api | 0.9466 | 1.0000 | 0.0000 | False | 0.9905 | 10 |
| SWU7 | sentence_packed | `fuse_mean` | ungrounded | phantom_api | 0.9666 | 1.0000 | 0.0000 | False | 0.9944 | 10 |
| SWU7 | sentence_packed | `fuse_max` | ungrounded | phantom_api | 0.9866 | 1.0000 | 0.0000 | False | 0.9982 | 10 |
| SWU7 | sentence_packed | `fuse_min` | ungrounded | phantom_api | 0.9466 | 1.0000 | 0.0000 | False | 0.9905 | 10 |
| SWG8 | sentence_packed | `gte_only` | grounded | - | 0.9867 | 1.0000 | 0.0000 | False | 0.9998 | 20 |
| SWG8 | sentence_packed | `code_only` | grounded | - | 0.9487 | 1.0000 | 0.0000 | False | 0.9984 | 20 |
| SWG8 | sentence_packed | `fuse_mean` | grounded | - | 0.9677 | 1.0000 | 0.0000 | False | 0.9991 | 20 |
| SWG8 | sentence_packed | `fuse_max` | grounded | - | 0.9867 | 1.0000 | 0.0000 | False | 0.9998 | 20 |
| SWG8 | sentence_packed | `fuse_min` | grounded | - | 0.9487 | 1.0000 | 0.0000 | False | 0.9984 | 20 |
| SWU8 | sentence_packed | `gte_only` | ungrounded | entity_swap | 0.9868 | 1.0000 | 0.0000 | False | 0.9998 | 20 |
| SWU8 | sentence_packed | `code_only` | ungrounded | entity_swap | 0.9489 | 1.0000 | 0.0000 | False | 0.9984 | 20 |
| SWU8 | sentence_packed | `fuse_mean` | ungrounded | entity_swap | 0.9679 | 1.0000 | 0.0000 | False | 0.9991 | 20 |
| SWU8 | sentence_packed | `fuse_max` | ungrounded | entity_swap | 0.9868 | 1.0000 | 0.0000 | False | 0.9998 | 20 |
| SWU8 | sentence_packed | `fuse_min` | ungrounded | entity_swap | 0.9489 | 1.0000 | 0.0000 | False | 0.9984 | 20 |
| SWG9 | sentence_packed | `gte_only` | grounded | - | 0.9822 | 1.0000 | 0.0000 | False | 0.9980 | 18 |
| SWG9 | sentence_packed | `code_only` | grounded | - | 0.9375 | 1.0000 | 0.0000 | False | 0.9949 | 18 |
| SWG9 | sentence_packed | `fuse_mean` | grounded | - | 0.9599 | 1.0000 | 0.0000 | False | 0.9965 | 18 |
| SWG9 | sentence_packed | `fuse_max` | grounded | - | 0.9822 | 1.0000 | 0.0000 | False | 0.9980 | 18 |
| SWG9 | sentence_packed | `fuse_min` | grounded | - | 0.9375 | 1.0000 | 0.0000 | False | 0.9949 | 18 |
| SWU9 | sentence_packed | `gte_only` | ungrounded | phantom_api | 0.9798 | 1.0000 | 0.0000 | False | 0.9980 | 18 |
| SWU9 | sentence_packed | `code_only` | ungrounded | phantom_api | 0.9284 | 1.0000 | 0.0000 | False | 0.9949 | 18 |
| SWU9 | sentence_packed | `fuse_mean` | ungrounded | phantom_api | 0.9541 | 1.0000 | 0.0000 | False | 0.9965 | 18 |
| SWU9 | sentence_packed | `fuse_max` | ungrounded | phantom_api | 0.9798 | 1.0000 | 0.0000 | False | 0.9980 | 18 |
| SWU9 | sentence_packed | `fuse_min` | ungrounded | phantom_api | 0.9284 | 1.0000 | 0.0000 | False | 0.9949 | 18 |
| SWG10 | sentence_packed | `gte_only` | grounded | - | 0.9858 | 1.0000 | 0.0000 | False | 0.9992 | 11 |
| SWG10 | sentence_packed | `code_only` | grounded | - | 0.9440 | 1.0000 | 0.0000 | False | 0.9950 | 11 |
| SWG10 | sentence_packed | `fuse_mean` | grounded | - | 0.9649 | 1.0000 | 0.0000 | False | 0.9971 | 11 |
| SWG10 | sentence_packed | `fuse_max` | grounded | - | 0.9858 | 1.0000 | 0.0000 | False | 0.9992 | 11 |
| SWG10 | sentence_packed | `fuse_min` | grounded | - | 0.9440 | 1.0000 | 0.0000 | False | 0.9950 | 11 |
| SWU10 | sentence_packed | `gte_only` | ungrounded | phantom_api | 0.9862 | 1.0000 | 0.0000 | False | 0.9992 | 11 |
| SWU10 | sentence_packed | `code_only` | ungrounded | phantom_api | 0.9447 | 1.0000 | 0.0000 | False | 0.9950 | 11 |
| SWU10 | sentence_packed | `fuse_mean` | ungrounded | phantom_api | 0.9654 | 1.0000 | 0.0000 | False | 0.9971 | 11 |
| SWU10 | sentence_packed | `fuse_max` | ungrounded | phantom_api | 0.9862 | 1.0000 | 0.0000 | False | 0.9992 | 11 |
| SWU10 | sentence_packed | `fuse_min` | ungrounded | phantom_api | 0.9447 | 1.0000 | 0.0000 | False | 0.9950 | 11 |
| SWG11 | sentence_packed | `gte_only` | grounded | - | 0.9875 | 1.0000 | 0.0000 | False | 0.9973 | 36 |
| SWG11 | sentence_packed | `code_only` | grounded | - | 0.9465 | 1.0000 | 0.0000 | False | 0.9966 | 36 |
| SWG11 | sentence_packed | `fuse_mean` | grounded | - | 0.9670 | 1.0000 | 0.0000 | False | 0.9970 | 36 |
| SWG11 | sentence_packed | `fuse_max` | grounded | - | 0.9875 | 1.0000 | 0.0000 | False | 0.9973 | 36 |
| SWG11 | sentence_packed | `fuse_min` | grounded | - | 0.9465 | 1.0000 | 0.0000 | False | 0.9966 | 36 |
| SWU11 | sentence_packed | `gte_only` | ungrounded | phantom_api | 0.9870 | 1.0000 | 0.0000 | False | 0.9973 | 36 |
| SWU11 | sentence_packed | `code_only` | ungrounded | phantom_api | 0.9474 | 1.0000 | 0.0000 | False | 0.9903 | 36 |
| SWU11 | sentence_packed | `fuse_mean` | ungrounded | phantom_api | 0.9672 | 1.0000 | 0.0000 | False | 0.9938 | 36 |
| SWU11 | sentence_packed | `fuse_max` | ungrounded | phantom_api | 0.9870 | 1.0000 | 0.0000 | False | 0.9973 | 36 |
| SWU11 | sentence_packed | `fuse_min` | ungrounded | phantom_api | 0.9474 | 1.0000 | 0.0000 | False | 0.9903 | 36 |
| SWG12 | sentence_packed | `gte_only` | grounded | - | 0.9817 | 1.0000 | 0.0000 | False | 0.9973 | 9 |
| SWG12 | sentence_packed | `code_only` | grounded | - | 0.9237 | 1.0000 | 0.0000 | False | 0.9836 | 9 |
| SWG12 | sentence_packed | `fuse_mean` | grounded | - | 0.9527 | 1.0000 | 0.0000 | False | 0.9905 | 9 |
| SWG12 | sentence_packed | `fuse_max` | grounded | - | 0.9817 | 1.0000 | 0.0000 | False | 0.9973 | 9 |
| SWG12 | sentence_packed | `fuse_min` | grounded | - | 0.9237 | 1.0000 | 0.0000 | False | 0.9836 | 9 |
| SWU12 | sentence_packed | `gte_only` | ungrounded | negation_flip | 0.9802 | 1.0000 | 0.0000 | False | 0.9973 | 9 |
| SWU12 | sentence_packed | `code_only` | ungrounded | negation_flip | 0.9211 | 1.0000 | 0.0000 | False | 0.9836 | 9 |
| SWU12 | sentence_packed | `fuse_mean` | ungrounded | negation_flip | 0.9507 | 1.0000 | 0.0000 | False | 0.9905 | 9 |
| SWU12 | sentence_packed | `fuse_max` | ungrounded | negation_flip | 0.9802 | 1.0000 | 0.0000 | False | 0.9973 | 9 |
| SWU12 | sentence_packed | `fuse_min` | ungrounded | negation_flip | 0.9211 | 1.0000 | 0.0000 | False | 0.9836 | 9 |
| SWG13 | sentence_packed | `gte_only` | grounded | - | 0.9815 | 1.0000 | 0.0000 | False | 0.9991 | 30 |
| SWG13 | sentence_packed | `code_only` | grounded | - | 0.9209 | 1.0000 | 0.0000 | False | 0.9931 | 30 |
| SWG13 | sentence_packed | `fuse_mean` | grounded | - | 0.9512 | 1.0000 | 0.0000 | False | 0.9961 | 30 |
| SWG13 | sentence_packed | `fuse_max` | grounded | - | 0.9815 | 1.0000 | 0.0000 | False | 0.9991 | 30 |
| SWG13 | sentence_packed | `fuse_min` | grounded | - | 0.9209 | 1.0000 | 0.0000 | False | 0.9931 | 30 |
| SWU13 | sentence_packed | `gte_only` | ungrounded | entity_swap | 0.9811 | 1.0000 | 0.0000 | False | 0.9977 | 30 |
| SWU13 | sentence_packed | `code_only` | ungrounded | entity_swap | 0.9194 | 1.0000 | 0.0000 | False | 0.9839 | 30 |
| SWU13 | sentence_packed | `fuse_mean` | ungrounded | entity_swap | 0.9503 | 1.0000 | 0.0000 | False | 0.9908 | 30 |
| SWU13 | sentence_packed | `fuse_max` | ungrounded | entity_swap | 0.9811 | 1.0000 | 0.0000 | False | 0.9977 | 30 |
| SWU13 | sentence_packed | `fuse_min` | ungrounded | entity_swap | 0.9194 | 1.0000 | 0.0000 | False | 0.9839 | 30 |
| SWG14 | sentence_packed | `gte_only` | grounded | - | 0.9843 | 1.0000 | 0.0000 | False | 0.9984 | 9 |
| SWG14 | sentence_packed | `code_only` | grounded | - | 0.9296 | 1.0000 | 0.0000 | False | 0.9804 | 9 |
| SWG14 | sentence_packed | `fuse_mean` | grounded | - | 0.9569 | 1.0000 | 0.0000 | False | 0.9894 | 9 |
| SWG14 | sentence_packed | `fuse_max` | grounded | - | 0.9843 | 1.0000 | 0.0000 | False | 0.9984 | 9 |
| SWG14 | sentence_packed | `fuse_min` | grounded | - | 0.9296 | 1.0000 | 0.0000 | False | 0.9804 | 9 |
| SWU14 | sentence_packed | `gte_only` | ungrounded | entity_swap | 0.9843 | 1.0000 | 0.0000 | False | 0.9984 | 9 |
| SWU14 | sentence_packed | `code_only` | ungrounded | entity_swap | 0.9292 | 1.0000 | 0.0000 | False | 0.9804 | 9 |
| SWU14 | sentence_packed | `fuse_mean` | ungrounded | entity_swap | 0.9568 | 1.0000 | 0.0000 | False | 0.9894 | 9 |
| SWU14 | sentence_packed | `fuse_max` | ungrounded | entity_swap | 0.9843 | 1.0000 | 0.0000 | False | 0.9984 | 9 |
| SWU14 | sentence_packed | `fuse_min` | ungrounded | entity_swap | 0.9292 | 1.0000 | 0.0000 | False | 0.9804 | 9 |
| SWG15 | sentence_packed | `gte_only` | grounded | - | 0.9889 | 1.0000 | 0.0000 | False | 0.9975 | 22 |
| SWG15 | sentence_packed | `code_only` | grounded | - | 0.9473 | 1.0000 | 0.0000 | False | 0.9889 | 22 |
| SWG15 | sentence_packed | `fuse_mean` | grounded | - | 0.9681 | 1.0000 | 0.0000 | False | 0.9932 | 22 |
| SWG15 | sentence_packed | `fuse_max` | grounded | - | 0.9889 | 1.0000 | 0.0000 | False | 0.9975 | 22 |
| SWG15 | sentence_packed | `fuse_min` | grounded | - | 0.9473 | 1.0000 | 0.0000 | False | 0.9889 | 22 |
| SWU15 | sentence_packed | `gte_only` | ungrounded | entity_swap | 0.9878 | 1.0000 | 0.0000 | False | 0.9949 | 22 |
| SWU15 | sentence_packed | `code_only` | ungrounded | entity_swap | 0.9431 | 1.0000 | 0.0000 | False | 0.9889 | 22 |
| SWU15 | sentence_packed | `fuse_mean` | ungrounded | entity_swap | 0.9654 | 1.0000 | 0.0000 | False | 0.9919 | 22 |
| SWU15 | sentence_packed | `fuse_max` | ungrounded | entity_swap | 0.9878 | 1.0000 | 0.0000 | False | 0.9949 | 22 |
| SWU15 | sentence_packed | `fuse_min` | ungrounded | entity_swap | 0.9431 | 1.0000 | 0.0000 | False | 0.9889 | 22 |
| CG1 | colgrep | `gte_only` | grounded | - | 0.9866 | 1.0000 | 0.0000 | False | 0.9995 | 4 |
| CG1 | colgrep | `code_only` | grounded | - | 0.9384 | 1.0000 | 0.0000 | False | 1.0000 | 4 |
| CG1 | colgrep | `fuse_mean` | grounded | - | 0.9625 | 1.0000 | 0.0000 | False | 0.9998 | 4 |
| CG1 | colgrep | `fuse_max` | grounded | - | 0.9866 | 1.0000 | 0.0000 | False | 1.0000 | 4 |
| CG1 | colgrep | `fuse_min` | grounded | - | 0.9384 | 1.0000 | 0.0000 | False | 0.9995 | 4 |
| CG2 | colgrep | `gte_only` | grounded | - | 0.9829 | 1.0000 | 0.0000 | False | 0.9994 | 5 |
| CG2 | colgrep | `code_only` | grounded | - | 0.9461 | 1.0000 | 0.0000 | False | 0.9999 | 5 |
| CG2 | colgrep | `fuse_mean` | grounded | - | 0.9645 | 1.0000 | 0.0000 | False | 0.9997 | 5 |
| CG2 | colgrep | `fuse_max` | grounded | - | 0.9829 | 1.0000 | 0.0000 | False | 0.9999 | 5 |
| CG2 | colgrep | `fuse_min` | grounded | - | 0.9461 | 1.0000 | 0.0000 | False | 0.9994 | 5 |
| CG3 | colgrep | `gte_only` | grounded | - | 0.9854 | 1.0000 | 0.0000 | False | 0.9982 | 2 |
| CG3 | colgrep | `code_only` | grounded | - | 0.9423 | 1.0000 | 0.0000 | False | 1.0000 | 2 |
| CG3 | colgrep | `fuse_mean` | grounded | - | 0.9638 | 1.0000 | 0.0000 | False | 0.9991 | 2 |
| CG3 | colgrep | `fuse_max` | grounded | - | 0.9854 | 1.0000 | 0.0000 | False | 1.0000 | 2 |
| CG3 | colgrep | `fuse_min` | grounded | - | 0.9423 | 1.0000 | 0.0000 | False | 0.9982 | 2 |
| CU1 | colgrep | `gte_only` | ungrounded | phantom_api | 0.9827 | 1.0000 | 0.0000 | False | 0.9999 | 3 |
| CU1 | colgrep | `code_only` | ungrounded | phantom_api | 0.9470 | 1.0000 | 0.0000 | False | 1.0000 | 3 |
| CU1 | colgrep | `fuse_mean` | ungrounded | phantom_api | 0.9649 | 1.0000 | 0.0000 | False | 0.9999 | 3 |
| CU1 | colgrep | `fuse_max` | ungrounded | phantom_api | 0.9827 | 1.0000 | 0.0000 | False | 1.0000 | 3 |
| CU1 | colgrep | `fuse_min` | ungrounded | phantom_api | 0.9470 | 1.0000 | 0.0000 | False | 0.9999 | 3 |
| CU2 | colgrep | `gte_only` | ungrounded | phantom_api | 0.9752 | 1.0000 | 0.0000 | False | 0.9976 | 4 |
| CU2 | colgrep | `code_only` | ungrounded | phantom_api | 0.8665 | 1.0000 | 0.0000 | False | 0.9998 | 4 |
| CU2 | colgrep | `fuse_mean` | ungrounded | phantom_api | 0.9208 | 1.0000 | 0.0000 | False | 0.9987 | 4 |
| CU2 | colgrep | `fuse_max` | ungrounded | phantom_api | 0.9752 | 1.0000 | 0.0000 | False | 0.9998 | 4 |
| CU2 | colgrep | `fuse_min` | ungrounded | phantom_api | 0.8665 | 1.0000 | 0.0000 | False | 0.9976 | 4 |
| CU3 | colgrep | `gte_only` | ungrounded | entity_swap | 0.9828 | 1.0000 | 0.0000 | False | 0.9979 | 2 |
| CU3 | colgrep | `code_only` | ungrounded | entity_swap | 0.9478 | 1.0000 | 0.0000 | False | 1.0000 | 2 |
| CU3 | colgrep | `fuse_mean` | ungrounded | entity_swap | 0.9653 | 1.0000 | 0.0000 | False | 0.9989 | 2 |
| CU3 | colgrep | `fuse_max` | ungrounded | entity_swap | 0.9828 | 1.0000 | 0.0000 | False | 1.0000 | 2 |
| CU3 | colgrep | `fuse_min` | ungrounded | entity_swap | 0.9478 | 1.0000 | 0.0000 | False | 0.9979 | 2 |
| CU4 | colgrep | `gte_only` | ungrounded | entity_swap | 0.9758 | 1.0000 | 0.0000 | False | 0.9973 | 2 |
| CU4 | colgrep | `code_only` | ungrounded | entity_swap | 0.9047 | 1.0000 | 0.0000 | False | 0.9999 | 2 |
| CU4 | colgrep | `fuse_mean` | ungrounded | entity_swap | 0.9403 | 1.0000 | 0.0000 | False | 0.9986 | 2 |
| CU4 | colgrep | `fuse_max` | ungrounded | entity_swap | 0.9758 | 1.0000 | 0.0000 | False | 0.9999 | 2 |
| CU4 | colgrep | `fuse_min` | ungrounded | entity_swap | 0.9047 | 1.0000 | 0.0000 | False | 0.9973 | 2 |
| CU5 | colgrep | `gte_only` | ungrounded | parametric | 0.9795 | 1.0000 | 0.0000 | False | 0.9996 | 3 |
| CU5 | colgrep | `code_only` | ungrounded | parametric | 0.9230 | 1.0000 | 0.0000 | False | 0.9998 | 3 |
| CU5 | colgrep | `fuse_mean` | ungrounded | parametric | 0.9513 | 1.0000 | 0.0000 | False | 0.9997 | 3 |
| CU5 | colgrep | `fuse_max` | ungrounded | parametric | 0.9795 | 1.0000 | 0.0000 | False | 0.9998 | 3 |
| CU5 | colgrep | `fuse_min` | ungrounded | parametric | 0.9230 | 1.0000 | 0.0000 | False | 0.9996 | 3 |
| CU6 | colgrep | `gte_only` | ungrounded | parametric | 0.9783 | 1.0000 | 0.0000 | False | 0.9975 | 4 |
| CU6 | colgrep | `code_only` | ungrounded | parametric | 0.9316 | 1.0000 | 0.0000 | False | 1.0000 | 4 |
| CU6 | colgrep | `fuse_mean` | ungrounded | parametric | 0.9550 | 1.0000 | 0.0000 | False | 0.9988 | 4 |
| CU6 | colgrep | `fuse_max` | ungrounded | parametric | 0.9783 | 1.0000 | 0.0000 | False | 1.0000 | 4 |
| CU6 | colgrep | `fuse_min` | ungrounded | parametric | 0.9316 | 1.0000 | 0.0000 | False | 0.9975 | 4 |
| CA1 | colgrep | `gte_only` | ambiguous | partial | 0.9818 | 1.0000 | 0.0000 | False | 0.9985 | 2 |
| CA1 | colgrep | `code_only` | ambiguous | partial | 0.9427 | 1.0000 | 0.0000 | False | 0.9999 | 2 |
| CA1 | colgrep | `fuse_mean` | ambiguous | partial | 0.9623 | 1.0000 | 0.0000 | False | 0.9992 | 2 |
| CA1 | colgrep | `fuse_max` | ambiguous | partial | 0.9818 | 1.0000 | 0.0000 | False | 0.9999 | 2 |
| CA1 | colgrep | `fuse_min` | ambiguous | partial | 0.9427 | 1.0000 | 0.0000 | False | 0.9985 | 2 |
| CA2 | colgrep | `gte_only` | ambiguous | partial | 0.9768 | 1.0000 | 0.0000 | False | 0.9977 | 2 |
| CA2 | colgrep | `code_only` | ambiguous | partial | 0.9342 | 1.0000 | 0.0000 | False | 0.9999 | 2 |
| CA2 | colgrep | `fuse_mean` | ambiguous | partial | 0.9555 | 1.0000 | 0.0000 | False | 0.9988 | 2 |
| CA2 | colgrep | `fuse_max` | ambiguous | partial | 0.9768 | 1.0000 | 0.0000 | False | 0.9999 | 2 |
| CA2 | colgrep | `fuse_min` | ambiguous | partial | 0.9342 | 1.0000 | 0.0000 | False | 0.9977 | 2 |
| CA3 | colgrep | `gte_only` | ambiguous | negation_flip | 0.9806 | 1.0000 | 0.0000 | False | 0.9977 | 3 |
| CA3 | colgrep | `code_only` | ambiguous | negation_flip | 0.9217 | 1.0000 | 0.0000 | False | 0.9999 | 3 |
| CA3 | colgrep | `fuse_mean` | ambiguous | negation_flip | 0.9511 | 1.0000 | 0.0000 | False | 0.9988 | 3 |
| CA3 | colgrep | `fuse_max` | ambiguous | negation_flip | 0.9806 | 1.0000 | 0.0000 | False | 0.9999 | 3 |
| CA3 | colgrep | `fuse_min` | ambiguous | negation_flip | 0.9217 | 1.0000 | 0.0000 | False | 0.9977 | 3 |
| SWG1 | colgrep | `gte_only` | grounded | - | 0.9616 | 1.0000 | 0.0000 | False | 0.9965 | 5 |
| SWG1 | colgrep | `code_only` | grounded | - | 0.8671 | 1.0000 | 0.0000 | False | 0.9863 | 5 |
| SWG1 | colgrep | `fuse_mean` | grounded | - | 0.9143 | 1.0000 | 0.0000 | False | 0.9914 | 5 |
| SWG1 | colgrep | `fuse_max` | grounded | - | 0.9616 | 1.0000 | 0.0000 | False | 0.9965 | 5 |
| SWG1 | colgrep | `fuse_min` | grounded | - | 0.8671 | 1.0000 | 0.0000 | False | 0.9863 | 5 |
| SWU1 | colgrep | `gte_only` | ungrounded | entity_swap | 0.9647 | 1.0000 | 0.0000 | False | 0.9998 | 5 |
| SWU1 | colgrep | `code_only` | ungrounded | entity_swap | 0.8751 | 1.0000 | 0.0000 | False | 0.9883 | 5 |
| SWU1 | colgrep | `fuse_mean` | ungrounded | entity_swap | 0.9199 | 1.0000 | 0.0000 | False | 0.9940 | 5 |
| SWU1 | colgrep | `fuse_max` | ungrounded | entity_swap | 0.9647 | 1.0000 | 0.0000 | False | 0.9998 | 5 |
| SWU1 | colgrep | `fuse_min` | ungrounded | entity_swap | 0.8751 | 1.0000 | 0.0000 | False | 0.9883 | 5 |
| SWG2 | colgrep | `gte_only` | grounded | - | 0.9622 | 1.0000 | 0.0000 | False | 0.9997 | 3 |
| SWG2 | colgrep | `code_only` | grounded | - | 0.8641 | 1.0000 | 0.0000 | False | 0.9790 | 3 |
| SWG2 | colgrep | `fuse_mean` | grounded | - | 0.9132 | 1.0000 | 0.0000 | False | 0.9894 | 3 |
| SWG2 | colgrep | `fuse_max` | grounded | - | 0.9622 | 1.0000 | 0.0000 | False | 0.9997 | 3 |
| SWG2 | colgrep | `fuse_min` | grounded | - | 0.8641 | 1.0000 | 0.0000 | False | 0.9790 | 3 |
| SWU2 | colgrep | `gte_only` | ungrounded | phantom_api | 0.9603 | 1.0000 | 0.0000 | False | 0.9997 | 3 |
| SWU2 | colgrep | `code_only` | ungrounded | phantom_api | 0.8635 | 1.0000 | 0.0000 | False | 0.9739 | 3 |
| SWU2 | colgrep | `fuse_mean` | ungrounded | phantom_api | 0.9119 | 1.0000 | 0.0000 | False | 0.9868 | 3 |
| SWU2 | colgrep | `fuse_max` | ungrounded | phantom_api | 0.9603 | 1.0000 | 0.0000 | False | 0.9997 | 3 |
| SWU2 | colgrep | `fuse_min` | ungrounded | phantom_api | 0.8635 | 1.0000 | 0.0000 | False | 0.9739 | 3 |
| SWG3 | colgrep | `gte_only` | grounded | - | 0.9828 | 1.0000 | 0.0000 | False | 0.9982 | 6 |
| SWG3 | colgrep | `code_only` | grounded | - | 0.9310 | 1.0000 | 0.0000 | False | 0.9912 | 6 |
| SWG3 | colgrep | `fuse_mean` | grounded | - | 0.9569 | 1.0000 | 0.0000 | False | 0.9947 | 6 |
| SWG3 | colgrep | `fuse_max` | grounded | - | 0.9828 | 1.0000 | 0.0000 | False | 0.9982 | 6 |
| SWG3 | colgrep | `fuse_min` | grounded | - | 0.9310 | 1.0000 | 0.0000 | False | 0.9912 | 6 |
| SWU3 | colgrep | `gte_only` | ungrounded | negation_flip | 0.9819 | 1.0000 | 0.0000 | False | 0.9982 | 6 |
| SWU3 | colgrep | `code_only` | ungrounded | negation_flip | 0.9265 | 1.0000 | 0.0000 | False | 0.9912 | 6 |
| SWU3 | colgrep | `fuse_mean` | ungrounded | negation_flip | 0.9542 | 1.0000 | 0.0000 | False | 0.9947 | 6 |
| SWU3 | colgrep | `fuse_max` | ungrounded | negation_flip | 0.9819 | 1.0000 | 0.0000 | False | 0.9982 | 6 |
| SWU3 | colgrep | `fuse_min` | ungrounded | negation_flip | 0.9265 | 1.0000 | 0.0000 | False | 0.9912 | 6 |
| SWG4 | colgrep | `gte_only` | grounded | - | 0.9868 | 1.0000 | 0.0000 | False | 0.9977 | 12 |
| SWG4 | colgrep | `code_only` | grounded | - | 0.9344 | 1.0000 | 0.0000 | False | 1.0000 | 12 |
| SWG4 | colgrep | `fuse_mean` | grounded | - | 0.9606 | 1.0000 | 0.0000 | False | 0.9988 | 12 |
| SWG4 | colgrep | `fuse_max` | grounded | - | 0.9868 | 1.0000 | 0.0000 | False | 1.0000 | 12 |
| SWG4 | colgrep | `fuse_min` | grounded | - | 0.9344 | 1.0000 | 0.0000 | False | 0.9977 | 12 |
| SWU4 | colgrep | `gte_only` | ungrounded | entity_swap | 0.9863 | 1.0000 | 0.0000 | False | 0.9977 | 12 |
| SWU4 | colgrep | `code_only` | ungrounded | entity_swap | 0.9312 | 1.0000 | 0.0000 | False | 0.9731 | 12 |
| SWU4 | colgrep | `fuse_mean` | ungrounded | entity_swap | 0.9587 | 1.0000 | 0.0000 | False | 0.9854 | 12 |
| SWU4 | colgrep | `fuse_max` | ungrounded | entity_swap | 0.9863 | 1.0000 | 0.0000 | False | 0.9977 | 12 |
| SWU4 | colgrep | `fuse_min` | ungrounded | entity_swap | 0.9312 | 1.0000 | 0.0000 | False | 0.9731 | 12 |
| SWG5 | colgrep | `gte_only` | grounded | - | 0.9789 | 1.0000 | 0.0000 | False | 0.9971 | 8 |
| SWG5 | colgrep | `code_only` | grounded | - | 0.9221 | 1.0000 | 0.0000 | False | 1.0000 | 8 |
| SWG5 | colgrep | `fuse_mean` | grounded | - | 0.9505 | 1.0000 | 0.0000 | False | 0.9985 | 8 |
| SWG5 | colgrep | `fuse_max` | grounded | - | 0.9789 | 1.0000 | 0.0000 | False | 1.0000 | 8 |
| SWG5 | colgrep | `fuse_min` | grounded | - | 0.9221 | 1.0000 | 0.0000 | False | 0.9971 | 8 |
| SWU5 | colgrep | `gte_only` | ungrounded | entity_swap | 0.9777 | 1.0000 | 0.0000 | False | 0.9971 | 8 |
| SWU5 | colgrep | `code_only` | ungrounded | entity_swap | 0.9164 | 1.0000 | 0.0000 | False | 1.0000 | 8 |
| SWU5 | colgrep | `fuse_mean` | ungrounded | entity_swap | 0.9471 | 1.0000 | 0.0000 | False | 0.9985 | 8 |
| SWU5 | colgrep | `fuse_max` | ungrounded | entity_swap | 0.9777 | 1.0000 | 0.0000 | False | 1.0000 | 8 |
| SWU5 | colgrep | `fuse_min` | ungrounded | entity_swap | 0.9164 | 1.0000 | 0.0000 | False | 0.9971 | 8 |
| SWG6 | colgrep | `gte_only` | grounded | - | 0.9699 | 1.0000 | 0.0000 | False | 0.9968 | 8 |
| SWG6 | colgrep | `code_only` | grounded | - | 0.8838 | 1.0000 | 0.0000 | False | 0.9870 | 8 |
| SWG6 | colgrep | `fuse_mean` | grounded | - | 0.9269 | 1.0000 | 0.0000 | False | 0.9919 | 8 |
| SWG6 | colgrep | `fuse_max` | grounded | - | 0.9699 | 1.0000 | 0.0000 | False | 0.9968 | 8 |
| SWG6 | colgrep | `fuse_min` | grounded | - | 0.8838 | 1.0000 | 0.0000 | False | 0.9870 | 8 |
| SWU6 | colgrep | `gte_only` | ungrounded | entity_swap | 0.9699 | 1.0000 | 0.0000 | False | 0.9968 | 8 |
| SWU6 | colgrep | `code_only` | ungrounded | entity_swap | 0.8826 | 1.0000 | 0.0000 | False | 0.9865 | 8 |
| SWU6 | colgrep | `fuse_mean` | ungrounded | entity_swap | 0.9262 | 1.0000 | 0.0000 | False | 0.9916 | 8 |
| SWU6 | colgrep | `fuse_max` | ungrounded | entity_swap | 0.9699 | 1.0000 | 0.0000 | False | 0.9968 | 8 |
| SWU6 | colgrep | `fuse_min` | ungrounded | entity_swap | 0.8826 | 1.0000 | 0.0000 | False | 0.9865 | 8 |
| SWG7 | colgrep | `gte_only` | grounded | - | 0.9873 | 1.0000 | 0.0000 | False | 1.0000 | 11 |
| SWG7 | colgrep | `code_only` | grounded | - | 0.9488 | 1.0000 | 0.0000 | False | 0.9944 | 11 |
| SWG7 | colgrep | `fuse_mean` | grounded | - | 0.9681 | 1.0000 | 0.0000 | False | 0.9972 | 11 |
| SWG7 | colgrep | `fuse_max` | grounded | - | 0.9873 | 1.0000 | 0.0000 | False | 1.0000 | 11 |
| SWG7 | colgrep | `fuse_min` | grounded | - | 0.9488 | 1.0000 | 0.0000 | False | 0.9944 | 11 |
| SWU7 | colgrep | `gte_only` | ungrounded | phantom_api | 0.9857 | 1.0000 | 0.0000 | False | 0.9999 | 11 |
| SWU7 | colgrep | `code_only` | ungrounded | phantom_api | 0.9432 | 1.0000 | 0.0000 | False | 0.9905 | 11 |
| SWU7 | colgrep | `fuse_mean` | ungrounded | phantom_api | 0.9644 | 1.0000 | 0.0000 | False | 0.9952 | 11 |
| SWU7 | colgrep | `fuse_max` | ungrounded | phantom_api | 0.9857 | 1.0000 | 0.0000 | False | 0.9999 | 11 |
| SWU7 | colgrep | `fuse_min` | ungrounded | phantom_api | 0.9432 | 1.0000 | 0.0000 | False | 0.9905 | 11 |
| SWG8 | colgrep | `gte_only` | grounded | - | 0.9870 | 1.0000 | 0.0000 | False | 0.9997 | 31 |
| SWG8 | colgrep | `code_only` | grounded | - | 0.9496 | 1.0000 | 0.0000 | False | 0.9984 | 31 |
| SWG8 | colgrep | `fuse_mean` | grounded | - | 0.9683 | 1.0000 | 0.0000 | False | 0.9991 | 31 |
| SWG8 | colgrep | `fuse_max` | grounded | - | 0.9870 | 1.0000 | 0.0000 | False | 0.9997 | 31 |
| SWG8 | colgrep | `fuse_min` | grounded | - | 0.9496 | 1.0000 | 0.0000 | False | 0.9984 | 31 |
| SWU8 | colgrep | `gte_only` | ungrounded | entity_swap | 0.9872 | 1.0000 | 0.0000 | False | 0.9997 | 31 |
| SWU8 | colgrep | `code_only` | ungrounded | entity_swap | 0.9496 | 1.0000 | 0.0000 | False | 0.9984 | 31 |
| SWU8 | colgrep | `fuse_mean` | ungrounded | entity_swap | 0.9684 | 1.0000 | 0.0000 | False | 0.9991 | 31 |
| SWU8 | colgrep | `fuse_max` | ungrounded | entity_swap | 0.9872 | 1.0000 | 0.0000 | False | 0.9997 | 31 |
| SWU8 | colgrep | `fuse_min` | ungrounded | entity_swap | 0.9496 | 1.0000 | 0.0000 | False | 0.9984 | 31 |
| SWG9 | colgrep | `gte_only` | grounded | - | 0.9816 | 1.0000 | 0.0000 | False | 0.9980 | 30 |
| SWG9 | colgrep | `code_only` | grounded | - | 0.9358 | 1.0000 | 0.0000 | False | 0.9949 | 30 |
| SWG9 | colgrep | `fuse_mean` | grounded | - | 0.9587 | 1.0000 | 0.0000 | False | 0.9965 | 30 |
| SWG9 | colgrep | `fuse_max` | grounded | - | 0.9816 | 1.0000 | 0.0000 | False | 0.9980 | 30 |
| SWG9 | colgrep | `fuse_min` | grounded | - | 0.9358 | 1.0000 | 0.0000 | False | 0.9949 | 30 |
| SWU9 | colgrep | `gte_only` | ungrounded | phantom_api | 0.9794 | 1.0000 | 0.0000 | False | 0.9980 | 30 |
| SWU9 | colgrep | `code_only` | ungrounded | phantom_api | 0.9271 | 1.0000 | 0.0000 | False | 0.9949 | 30 |
| SWU9 | colgrep | `fuse_mean` | ungrounded | phantom_api | 0.9532 | 1.0000 | 0.0000 | False | 0.9965 | 30 |
| SWU9 | colgrep | `fuse_max` | ungrounded | phantom_api | 0.9794 | 1.0000 | 0.0000 | False | 0.9980 | 30 |
| SWU9 | colgrep | `fuse_min` | ungrounded | phantom_api | 0.9271 | 1.0000 | 0.0000 | False | 0.9949 | 30 |
| SWG10 | colgrep | `gte_only` | grounded | - | 0.9870 | 1.0000 | 0.0000 | False | 0.9991 | 19 |
| SWG10 | colgrep | `code_only` | grounded | - | 0.9467 | 1.0000 | 0.0000 | False | 0.9835 | 19 |
| SWG10 | colgrep | `fuse_mean` | grounded | - | 0.9668 | 1.0000 | 0.0000 | False | 0.9913 | 19 |
| SWG10 | colgrep | `fuse_max` | grounded | - | 0.9870 | 1.0000 | 0.0000 | False | 0.9991 | 19 |
| SWG10 | colgrep | `fuse_min` | grounded | - | 0.9467 | 1.0000 | 0.0000 | False | 0.9835 | 19 |
| SWU10 | colgrep | `gte_only` | ungrounded | phantom_api | 0.9873 | 1.0000 | 0.0000 | False | 0.9983 | 19 |
| SWU10 | colgrep | `code_only` | ungrounded | phantom_api | 0.9469 | 1.0000 | 0.0000 | False | 0.9835 | 19 |
| SWU10 | colgrep | `fuse_mean` | ungrounded | phantom_api | 0.9671 | 1.0000 | 0.0000 | False | 0.9909 | 19 |
| SWU10 | colgrep | `fuse_max` | ungrounded | phantom_api | 0.9873 | 1.0000 | 0.0000 | False | 0.9983 | 19 |
| SWU10 | colgrep | `fuse_min` | ungrounded | phantom_api | 0.9469 | 1.0000 | 0.0000 | False | 0.9835 | 19 |
| SWG11 | colgrep | `gte_only` | grounded | - | 0.9891 | 1.0000 | 0.0000 | False | 0.9980 | 55 |
| SWG11 | colgrep | `code_only` | grounded | - | 0.9499 | 1.0000 | 0.0000 | False | 0.9964 | 55 |
| SWG11 | colgrep | `fuse_mean` | grounded | - | 0.9695 | 1.0000 | 0.0000 | False | 0.9972 | 55 |
| SWG11 | colgrep | `fuse_max` | grounded | - | 0.9891 | 1.0000 | 0.0000 | False | 0.9980 | 55 |
| SWG11 | colgrep | `fuse_min` | grounded | - | 0.9499 | 1.0000 | 0.0000 | False | 0.9964 | 55 |
| SWU11 | colgrep | `gte_only` | ungrounded | phantom_api | 0.9888 | 1.0000 | 0.0000 | False | 0.9980 | 55 |
| SWU11 | colgrep | `code_only` | ungrounded | phantom_api | 0.9501 | 1.0000 | 0.0000 | False | 0.9898 | 55 |
| SWU11 | colgrep | `fuse_mean` | ungrounded | phantom_api | 0.9694 | 1.0000 | 0.0000 | False | 0.9939 | 55 |
| SWU11 | colgrep | `fuse_max` | ungrounded | phantom_api | 0.9888 | 1.0000 | 0.0000 | False | 0.9980 | 55 |
| SWU11 | colgrep | `fuse_min` | ungrounded | phantom_api | 0.9501 | 1.0000 | 0.0000 | False | 0.9898 | 55 |
| SWG12 | colgrep | `gte_only` | grounded | - | 0.9768 | 1.0000 | 0.0000 | False | 0.9988 | 8 |
| SWG12 | colgrep | `code_only` | grounded | - | 0.9108 | 1.0000 | 0.0000 | False | 0.9929 | 8 |
| SWG12 | colgrep | `fuse_mean` | grounded | - | 0.9438 | 1.0000 | 0.0000 | False | 0.9958 | 8 |
| SWG12 | colgrep | `fuse_max` | grounded | - | 0.9768 | 1.0000 | 0.0000 | False | 0.9988 | 8 |
| SWG12 | colgrep | `fuse_min` | grounded | - | 0.9108 | 1.0000 | 0.0000 | False | 0.9929 | 8 |
| SWU12 | colgrep | `gte_only` | ungrounded | negation_flip | 0.9753 | 1.0000 | 0.0000 | False | 0.9988 | 8 |
| SWU12 | colgrep | `code_only` | ungrounded | negation_flip | 0.9081 | 1.0000 | 0.0000 | False | 0.9929 | 8 |
| SWU12 | colgrep | `fuse_mean` | ungrounded | negation_flip | 0.9417 | 1.0000 | 0.0000 | False | 0.9958 | 8 |
| SWU12 | colgrep | `fuse_max` | ungrounded | negation_flip | 0.9753 | 1.0000 | 0.0000 | False | 0.9988 | 8 |
| SWU12 | colgrep | `fuse_min` | ungrounded | negation_flip | 0.9081 | 1.0000 | 0.0000 | False | 0.9929 | 8 |
| SWG13 | colgrep | `gte_only` | grounded | - | 0.9823 | 1.0000 | 0.0000 | False | 0.9988 | 59 |
| SWG13 | colgrep | `code_only` | grounded | - | 0.9245 | 1.0000 | 0.0000 | False | 0.9939 | 59 |
| SWG13 | colgrep | `fuse_mean` | grounded | - | 0.9534 | 1.0000 | 0.0000 | False | 0.9964 | 59 |
| SWG13 | colgrep | `fuse_max` | grounded | - | 0.9823 | 1.0000 | 0.0000 | False | 0.9988 | 59 |
| SWG13 | colgrep | `fuse_min` | grounded | - | 0.9245 | 1.0000 | 0.0000 | False | 0.9939 | 59 |
| SWU13 | colgrep | `gte_only` | ungrounded | entity_swap | 0.9819 | 1.0000 | 0.0000 | False | 0.9984 | 59 |
| SWU13 | colgrep | `code_only` | ungrounded | entity_swap | 0.9226 | 1.0000 | 0.0000 | False | 0.9939 | 59 |
| SWU13 | colgrep | `fuse_mean` | ungrounded | entity_swap | 0.9523 | 1.0000 | 0.0000 | False | 0.9961 | 59 |
| SWU13 | colgrep | `fuse_max` | ungrounded | entity_swap | 0.9819 | 1.0000 | 0.0000 | False | 0.9984 | 59 |
| SWU13 | colgrep | `fuse_min` | ungrounded | entity_swap | 0.9226 | 1.0000 | 0.0000 | False | 0.9939 | 59 |
| SWG14 | colgrep | `gte_only` | grounded | - | 0.9868 | 1.0000 | 0.0000 | False | 0.9981 | 18 |
| SWG14 | colgrep | `code_only` | grounded | - | 0.9391 | 1.0000 | 0.0000 | False | 0.9845 | 18 |
| SWG14 | colgrep | `fuse_mean` | grounded | - | 0.9630 | 1.0000 | 0.0000 | False | 0.9913 | 18 |
| SWG14 | colgrep | `fuse_max` | grounded | - | 0.9868 | 1.0000 | 0.0000 | False | 0.9981 | 18 |
| SWG14 | colgrep | `fuse_min` | grounded | - | 0.9391 | 1.0000 | 0.0000 | False | 0.9845 | 18 |
| SWU14 | colgrep | `gte_only` | ungrounded | entity_swap | 0.9867 | 1.0000 | 0.0000 | False | 0.9981 | 18 |
| SWU14 | colgrep | `code_only` | ungrounded | entity_swap | 0.9383 | 1.0000 | 0.0000 | False | 0.9838 | 18 |
| SWU14 | colgrep | `fuse_mean` | ungrounded | entity_swap | 0.9625 | 1.0000 | 0.0000 | False | 0.9910 | 18 |
| SWU14 | colgrep | `fuse_max` | ungrounded | entity_swap | 0.9867 | 1.0000 | 0.0000 | False | 0.9981 | 18 |
| SWU14 | colgrep | `fuse_min` | ungrounded | entity_swap | 0.9383 | 1.0000 | 0.0000 | False | 0.9838 | 18 |
| SWG15 | colgrep | `gte_only` | grounded | - | 0.9897 | 1.0000 | 0.0000 | False | 0.9980 | 50 |
| SWG15 | colgrep | `code_only` | grounded | - | 0.9517 | 1.0000 | 0.0000 | False | 0.9889 | 50 |
| SWG15 | colgrep | `fuse_mean` | grounded | - | 0.9707 | 1.0000 | 0.0000 | False | 0.9934 | 50 |
| SWG15 | colgrep | `fuse_max` | grounded | - | 0.9897 | 1.0000 | 0.0000 | False | 0.9980 | 50 |
| SWG15 | colgrep | `fuse_min` | grounded | - | 0.9517 | 1.0000 | 0.0000 | False | 0.9889 | 50 |
| SWU15 | colgrep | `gte_only` | ungrounded | entity_swap | 0.9886 | 1.0000 | 0.0000 | False | 0.9955 | 50 |
| SWU15 | colgrep | `code_only` | ungrounded | entity_swap | 0.9480 | 1.0000 | 0.0000 | False | 0.9889 | 50 |
| SWU15 | colgrep | `fuse_mean` | ungrounded | entity_swap | 0.9683 | 1.0000 | 0.0000 | False | 0.9922 | 50 |
| SWU15 | colgrep | `fuse_max` | ungrounded | entity_swap | 0.9886 | 1.0000 | 0.0000 | False | 0.9955 | 50 |
| SWU15 | colgrep | `fuse_min` | ungrounded | entity_swap | 0.9480 | 1.0000 | 0.0000 | False | 0.9889 | 50 |
