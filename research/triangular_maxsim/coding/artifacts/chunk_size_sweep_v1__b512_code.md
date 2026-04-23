# Coding-Agent Groundedness: Scorer Stack x Chunker

> NO — the flagship colgrep x fuse_mean cell did not clear the RAG-style gate: reverse_context AUROC=0.569, phantom-API precision@0.35=0.00, ungrounded minus grounded unused-ratio delta=+0.0000. Stacking lift vs gte_only=-0.0291. Best fusion rule on colgrep is `fuse_max` (AUROC=0.598); consider flipping the flagship.

- run tag: `chunk_size_sweep_v1__b512_code`
- primary scorer: `lightonai/GTE-ModernColBERT-v1`
- orthogonal scorer: `lightonai/LateOn-Code-edge` (available: `True`)
- shared chunk budget: `512` tokens (GTE tokenizer)
- max SWE-bench instances: `15`
- phantom threshold: `0.35`
- flagship cell: `colgrep` x `fuse_mean` (present: `True`)

## Headline matrix

| chunker | scorer | AUROC | grounded cov | cov delta | unused delta | phantom@thr | grounded held-out rate | p95 ms | support units |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sentence_packed | `gte_only` | 0.5688 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 292.17 | 2.9 |
| sentence_packed | `code_only` | 0.5926 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 383.24 | 2.9 |
| sentence_packed | `fuse_mean` | 0.5952 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 304.38 | 2.9 |
| sentence_packed | `fuse_max` | 0.5688 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 304.38 | 2.9 |
| sentence_packed | `fuse_min` | 0.5926 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 304.38 | 2.9 |
| colgrep | `gte_only` | 0.5979 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 387.70 | 8.0 |
| colgrep | `code_only` | 0.5714 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 396.11 | 8.0 |
| colgrep | `fuse_mean` (flagship) | 0.5688 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 395.79 | 8.0 |
| colgrep | `fuse_max` | 0.5979 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 395.79 | 8.0 |
| colgrep | `fuse_min` | 0.5714 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 395.79 | 8.0 |

`*` marks cells that pass `reverse_context AUROC >= 0.90`.

## Score distribution diagnostics (reverse_context)

| chunker | scorer | grounded mean | grounded p10-p90 | ungrounded mean | ungrounded p10-p90 | separation | saturation@0.95 |
|---|---|---:|---|---:|---|---:|---:|
| sentence_packed | `gte_only` | 0.9777 | 0.9714-0.9828 | 0.9762 | 0.9700-0.9826 | +0.0015 | 1.0000 |
| sentence_packed | `code_only` | 0.9405 | 0.9206-0.9596 | 0.9333 | 0.9016-0.9567 | +0.0072 | 0.3590 |
| sentence_packed | `fuse_mean` | 0.9591 | 0.9458-0.9707 | 0.9547 | 0.9347-0.9690 | +0.0044 | 0.7436 |
| sentence_packed | `fuse_max` | 0.9777 | 0.9714-0.9828 | 0.9762 | 0.9700-0.9826 | +0.0015 | 1.0000 |
| sentence_packed | `fuse_min` | 0.9405 | 0.9206-0.9596 | 0.9333 | 0.9016-0.9567 | +0.0072 | 0.3590 |
| colgrep | `gte_only` | 0.9789 | 0.9570-0.9877 | 0.9778 | 0.9572-0.9870 | +0.0011 | 1.0000 |
| colgrep | `code_only` | 0.9397 | 0.8977-0.9614 | 0.9331 | 0.8877-0.9591 | +0.0066 | 0.4103 |
| colgrep | `fuse_mean` | 0.9593 | 0.9273-0.9734 | 0.9554 | 0.9221-0.9726 | +0.0038 | 0.7436 |
| colgrep | `fuse_max` | 0.9789 | 0.9570-0.9877 | 0.9778 | 0.9572-0.9870 | +0.0011 | 1.0000 |
| colgrep | `fuse_min` | 0.9397 | 0.8977-0.9614 | 0.9331 | 0.8877-0.9591 | +0.0066 | 0.4103 |

`separation` is mean(grounded) − mean(ungrounded) on `reverse_context`. `saturation@0.95` is the fraction of anchor cases (grounded + ungrounded) whose score ≥ 0.95 — a ceiling effect indicator.

## Stacking lift (fuse_* minus gte_only, per chunker)

| chunker | rule | AUROC lift | unused-delta lift | phantom precision lift |
|---|---|---:|---:|---:|
| sentence_packed | fuse_mean | +0.0265 | +0.0000 | +0.0000 |
| sentence_packed | fuse_max | +0.0000 | +0.0000 | +0.0000 |
| sentence_packed | fuse_min | +0.0238 | +0.0000 | +0.0000 |
| colgrep | fuse_mean | -0.0291 | +0.0000 | +0.0000 |
| colgrep | fuse_max | +0.0000 | +0.0000 | +0.0000 |
| colgrep | fuse_min | -0.0265 | +0.0000 | +0.0000 |

## Chunker lift (colgrep minus sentence_packed, per scorer)

| scorer | AUROC lift | unused-delta lift | phantom precision lift |
|---|---:|---:|---:|
| `gte_only` | +0.0291 | +0.0000 | +0.0000 |
| `code_only` | -0.0212 | +0.0000 | +0.0000 |
| `fuse_mean` | -0.0265 | +0.0000 | +0.0000 |
| `fuse_max` | +0.0291 | +0.0000 | +0.0000 |
| `fuse_min` | -0.0212 | +0.0000 | +0.0000 |

## Subcategory snapshot (per cell)

### `sentence_packed` x `gte_only`

| subcategory | count | reverse_context | coverage | unused ratio | max evidence |
|---|---:|---:|---:|---:|---:|
| entity_swap | 10 | 0.9772 | 1.0000 | 0.0000 | 0.9970 |
| grounded | 18 | 0.9777 | 1.0000 | 0.0000 | 0.9979 |
| negation_flip | 3 | 0.9757 | 1.0000 | 0.0000 | 0.9974 |
| parametric | 2 | 0.9697 | 1.0000 | 0.0000 | 0.9961 |
| partial | 2 | 0.9696 | 1.0000 | 0.0000 | 0.9976 |
| phantom_api | 7 | 0.9763 | 1.0000 | 0.0000 | 0.9987 |

### `sentence_packed` x `code_only`

| subcategory | count | reverse_context | coverage | unused ratio | max evidence |
|---|---:|---:|---:|---:|---:|
| entity_swap | 10 | 0.9374 | 1.0000 | 0.0000 | 0.9867 |
| grounded | 18 | 0.9405 | 1.0000 | 0.0000 | 0.9900 |
| negation_flip | 3 | 0.9219 | 1.0000 | 0.0000 | 0.9810 |
| parametric | 2 | 0.9081 | 1.0000 | 0.0000 | 0.9863 |
| partial | 2 | 0.9193 | 1.0000 | 0.0000 | 0.9981 |
| phantom_api | 7 | 0.9327 | 1.0000 | 0.0000 | 0.9845 |

### `sentence_packed` x `fuse_mean`

| subcategory | count | reverse_context | coverage | unused ratio | max evidence |
|---|---:|---:|---:|---:|---:|
| entity_swap | 10 | 0.9573 | 1.0000 | 0.0000 | 0.9918 |
| grounded | 18 | 0.9591 | 1.0000 | 0.0000 | 0.9940 |
| negation_flip | 3 | 0.9488 | 1.0000 | 0.0000 | 0.9892 |
| parametric | 2 | 0.9389 | 1.0000 | 0.0000 | 0.9912 |
| partial | 2 | 0.9445 | 1.0000 | 0.0000 | 0.9978 |
| phantom_api | 7 | 0.9545 | 1.0000 | 0.0000 | 0.9916 |

### `sentence_packed` x `fuse_max`

| subcategory | count | reverse_context | coverage | unused ratio | max evidence |
|---|---:|---:|---:|---:|---:|
| entity_swap | 10 | 0.9772 | 1.0000 | 0.0000 | 0.9973 |
| grounded | 18 | 0.9777 | 1.0000 | 0.0000 | 0.9984 |
| negation_flip | 3 | 0.9757 | 1.0000 | 0.0000 | 0.9974 |
| parametric | 2 | 0.9697 | 1.0000 | 0.0000 | 0.9978 |
| partial | 2 | 0.9696 | 1.0000 | 0.0000 | 0.9991 |
| phantom_api | 7 | 0.9763 | 1.0000 | 0.0000 | 0.9987 |

### `sentence_packed` x `fuse_min`

| subcategory | count | reverse_context | coverage | unused ratio | max evidence |
|---|---:|---:|---:|---:|---:|
| entity_swap | 10 | 0.9374 | 1.0000 | 0.0000 | 0.9864 |
| grounded | 18 | 0.9405 | 1.0000 | 0.0000 | 0.9895 |
| negation_flip | 3 | 0.9219 | 1.0000 | 0.0000 | 0.9810 |
| parametric | 2 | 0.9081 | 1.0000 | 0.0000 | 0.9847 |
| partial | 2 | 0.9193 | 1.0000 | 0.0000 | 0.9966 |
| phantom_api | 7 | 0.9327 | 1.0000 | 0.0000 | 0.9845 |

### `colgrep` x `gte_only`

| subcategory | count | reverse_context | coverage | unused ratio | max evidence |
|---|---:|---:|---:|---:|---:|
| entity_swap | 10 | 0.9779 | 1.0000 | 0.0000 | 0.9973 |
| grounded | 18 | 0.9789 | 1.0000 | 0.0000 | 0.9982 |
| negation_flip | 3 | 0.9760 | 1.0000 | 0.0000 | 0.9979 |
| parametric | 2 | 0.9789 | 1.0000 | 0.0000 | 0.9986 |
| partial | 2 | 0.9793 | 1.0000 | 0.0000 | 0.9981 |
| phantom_api | 7 | 0.9783 | 1.0000 | 0.0000 | 0.9992 |

### `colgrep` x `code_only`

| subcategory | count | reverse_context | coverage | unused ratio | max evidence |
|---|---:|---:|---:|---:|---:|
| entity_swap | 10 | 0.9367 | 1.0000 | 0.0000 | 0.9918 |
| grounded | 18 | 0.9397 | 1.0000 | 0.0000 | 0.9916 |
| negation_flip | 3 | 0.9272 | 1.0000 | 0.0000 | 0.9852 |
| parametric | 2 | 0.9273 | 1.0000 | 0.0000 | 0.9999 |
| partial | 2 | 0.9385 | 1.0000 | 0.0000 | 0.9999 |
| phantom_api | 7 | 0.9305 | 1.0000 | 0.0000 | 0.9929 |

### `colgrep` x `fuse_mean`

| subcategory | count | reverse_context | coverage | unused ratio | max evidence |
|---|---:|---:|---:|---:|---:|
| entity_swap | 10 | 0.9573 | 1.0000 | 0.0000 | 0.9946 |
| grounded | 18 | 0.9593 | 1.0000 | 0.0000 | 0.9949 |
| negation_flip | 3 | 0.9516 | 1.0000 | 0.0000 | 0.9915 |
| parametric | 2 | 0.9531 | 1.0000 | 0.0000 | 0.9992 |
| partial | 2 | 0.9589 | 1.0000 | 0.0000 | 0.9990 |
| phantom_api | 7 | 0.9544 | 1.0000 | 0.0000 | 0.9961 |

### `colgrep` x `fuse_max`

| subcategory | count | reverse_context | coverage | unused ratio | max evidence |
|---|---:|---:|---:|---:|---:|
| entity_swap | 10 | 0.9779 | 1.0000 | 0.0000 | 0.9979 |
| grounded | 18 | 0.9789 | 1.0000 | 0.0000 | 0.9986 |
| negation_flip | 3 | 0.9760 | 1.0000 | 0.0000 | 0.9987 |
| parametric | 2 | 0.9789 | 1.0000 | 0.0000 | 0.9999 |
| partial | 2 | 0.9793 | 1.0000 | 0.0000 | 0.9999 |
| phantom_api | 7 | 0.9783 | 1.0000 | 0.0000 | 0.9995 |

### `colgrep` x `fuse_min`

| subcategory | count | reverse_context | coverage | unused ratio | max evidence |
|---|---:|---:|---:|---:|---:|
| entity_swap | 10 | 0.9367 | 1.0000 | 0.0000 | 0.9912 |
| grounded | 18 | 0.9397 | 1.0000 | 0.0000 | 0.9912 |
| negation_flip | 3 | 0.9272 | 1.0000 | 0.0000 | 0.9844 |
| parametric | 2 | 0.9273 | 1.0000 | 0.0000 | 0.9986 |
| partial | 2 | 0.9385 | 1.0000 | 0.0000 | 0.9981 |
| phantom_api | 7 | 0.9305 | 1.0000 | 0.0000 | 0.9926 |

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
| CG2 | sentence_packed | `gte_only` | grounded | - | 0.9739 | 1.0000 | 0.0000 | False | 0.9994 | 1 |
| CG2 | sentence_packed | `code_only` | grounded | - | 0.9304 | 1.0000 | 0.0000 | False | 0.9910 | 1 |
| CG2 | sentence_packed | `fuse_mean` | grounded | - | 0.9522 | 1.0000 | 0.0000 | False | 0.9952 | 1 |
| CG2 | sentence_packed | `fuse_max` | grounded | - | 0.9739 | 1.0000 | 0.0000 | False | 0.9994 | 1 |
| CG2 | sentence_packed | `fuse_min` | grounded | - | 0.9304 | 1.0000 | 0.0000 | False | 0.9910 | 1 |
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
| SWG1 | sentence_packed | `gte_only` | grounded | - | 0.9694 | 1.0000 | 0.0000 | False | 0.9962 | 2 |
| SWG1 | sentence_packed | `code_only` | grounded | - | 0.9167 | 1.0000 | 0.0000 | False | 0.9692 | 2 |
| SWG1 | sentence_packed | `fuse_mean` | grounded | - | 0.9431 | 1.0000 | 0.0000 | False | 0.9827 | 2 |
| SWG1 | sentence_packed | `fuse_max` | grounded | - | 0.9694 | 1.0000 | 0.0000 | False | 0.9962 | 2 |
| SWG1 | sentence_packed | `fuse_min` | grounded | - | 0.9167 | 1.0000 | 0.0000 | False | 0.9692 | 2 |
| SWU1 | sentence_packed | `gte_only` | ungrounded | entity_swap | 0.9700 | 1.0000 | 0.0000 | False | 0.9977 | 2 |
| SWU1 | sentence_packed | `code_only` | ungrounded | entity_swap | 0.9177 | 1.0000 | 0.0000 | False | 0.9733 | 2 |
| SWU1 | sentence_packed | `fuse_mean` | ungrounded | entity_swap | 0.9439 | 1.0000 | 0.0000 | False | 0.9855 | 2 |
| SWU1 | sentence_packed | `fuse_max` | ungrounded | entity_swap | 0.9700 | 1.0000 | 0.0000 | False | 0.9977 | 2 |
| SWU1 | sentence_packed | `fuse_min` | ungrounded | entity_swap | 0.9177 | 1.0000 | 0.0000 | False | 0.9733 | 2 |
| SWG2 | sentence_packed | `gte_only` | grounded | - | 0.9717 | 1.0000 | 0.0000 | False | 0.9999 | 1 |
| SWG2 | sentence_packed | `code_only` | grounded | - | 0.9223 | 1.0000 | 0.0000 | False | 0.9809 | 1 |
| SWG2 | sentence_packed | `fuse_mean` | grounded | - | 0.9470 | 1.0000 | 0.0000 | False | 0.9904 | 1 |
| SWG2 | sentence_packed | `fuse_max` | grounded | - | 0.9717 | 1.0000 | 0.0000 | False | 0.9999 | 1 |
| SWG2 | sentence_packed | `fuse_min` | grounded | - | 0.9223 | 1.0000 | 0.0000 | False | 0.9809 | 1 |
| SWU2 | sentence_packed | `gte_only` | ungrounded | phantom_api | 0.9711 | 1.0000 | 0.0000 | False | 0.9996 | 1 |
| SWU2 | sentence_packed | `code_only` | ungrounded | phantom_api | 0.9233 | 1.0000 | 0.0000 | False | 0.9731 | 1 |
| SWU2 | sentence_packed | `fuse_mean` | ungrounded | phantom_api | 0.9472 | 1.0000 | 0.0000 | False | 0.9864 | 1 |
| SWU2 | sentence_packed | `fuse_max` | ungrounded | phantom_api | 0.9711 | 1.0000 | 0.0000 | False | 0.9996 | 1 |
| SWU2 | sentence_packed | `fuse_min` | ungrounded | phantom_api | 0.9233 | 1.0000 | 0.0000 | False | 0.9731 | 1 |
| SWG3 | sentence_packed | `gte_only` | grounded | - | 0.9783 | 1.0000 | 0.0000 | False | 0.9970 | 1 |
| SWG3 | sentence_packed | `code_only` | grounded | - | 0.9352 | 1.0000 | 0.0000 | False | 0.9829 | 1 |
| SWG3 | sentence_packed | `fuse_mean` | grounded | - | 0.9568 | 1.0000 | 0.0000 | False | 0.9899 | 1 |
| SWG3 | sentence_packed | `fuse_max` | grounded | - | 0.9783 | 1.0000 | 0.0000 | False | 0.9970 | 1 |
| SWG3 | sentence_packed | `fuse_min` | grounded | - | 0.9352 | 1.0000 | 0.0000 | False | 0.9829 | 1 |
| SWU3 | sentence_packed | `gte_only` | ungrounded | negation_flip | 0.9788 | 1.0000 | 0.0000 | False | 0.9969 | 1 |
| SWU3 | sentence_packed | `code_only` | ungrounded | negation_flip | 0.9326 | 1.0000 | 0.0000 | False | 0.9657 | 1 |
| SWU3 | sentence_packed | `fuse_mean` | ungrounded | negation_flip | 0.9557 | 1.0000 | 0.0000 | False | 0.9813 | 1 |
| SWU3 | sentence_packed | `fuse_max` | ungrounded | negation_flip | 0.9788 | 1.0000 | 0.0000 | False | 0.9969 | 1 |
| SWU3 | sentence_packed | `fuse_min` | ungrounded | negation_flip | 0.9326 | 1.0000 | 0.0000 | False | 0.9657 | 1 |
| SWG4 | sentence_packed | `gte_only` | grounded | - | 0.9775 | 1.0000 | 0.0000 | False | 0.9941 | 2 |
| SWG4 | sentence_packed | `code_only` | grounded | - | 0.9509 | 1.0000 | 0.0000 | False | 0.9968 | 2 |
| SWG4 | sentence_packed | `fuse_mean` | grounded | - | 0.9642 | 1.0000 | 0.0000 | False | 0.9955 | 2 |
| SWG4 | sentence_packed | `fuse_max` | grounded | - | 0.9775 | 1.0000 | 0.0000 | False | 0.9968 | 2 |
| SWG4 | sentence_packed | `fuse_min` | grounded | - | 0.9509 | 1.0000 | 0.0000 | False | 0.9941 | 2 |
| SWU4 | sentence_packed | `gte_only` | ungrounded | entity_swap | 0.9777 | 1.0000 | 0.0000 | False | 0.9962 | 2 |
| SWU4 | sentence_packed | `code_only` | ungrounded | entity_swap | 0.9473 | 1.0000 | 0.0000 | False | 0.9969 | 2 |
| SWU4 | sentence_packed | `fuse_mean` | ungrounded | entity_swap | 0.9625 | 1.0000 | 0.0000 | False | 0.9965 | 2 |
| SWU4 | sentence_packed | `fuse_max` | ungrounded | entity_swap | 0.9777 | 1.0000 | 0.0000 | False | 0.9969 | 2 |
| SWU4 | sentence_packed | `fuse_min` | ungrounded | entity_swap | 0.9473 | 1.0000 | 0.0000 | False | 0.9962 | 2 |
| SWG5 | sentence_packed | `gte_only` | grounded | - | 0.9792 | 1.0000 | 0.0000 | False | 0.9979 | 2 |
| SWG5 | sentence_packed | `code_only` | grounded | - | 0.9434 | 1.0000 | 0.0000 | False | 0.9861 | 2 |
| SWG5 | sentence_packed | `fuse_mean` | grounded | - | 0.9613 | 1.0000 | 0.0000 | False | 0.9920 | 2 |
| SWG5 | sentence_packed | `fuse_max` | grounded | - | 0.9792 | 1.0000 | 0.0000 | False | 0.9979 | 2 |
| SWG5 | sentence_packed | `fuse_min` | grounded | - | 0.9434 | 1.0000 | 0.0000 | False | 0.9861 | 2 |
| SWU5 | sentence_packed | `gte_only` | ungrounded | entity_swap | 0.9803 | 1.0000 | 0.0000 | False | 0.9979 | 2 |
| SWU5 | sentence_packed | `code_only` | ungrounded | entity_swap | 0.9482 | 1.0000 | 0.0000 | False | 0.9858 | 2 |
| SWU5 | sentence_packed | `fuse_mean` | ungrounded | entity_swap | 0.9643 | 1.0000 | 0.0000 | False | 0.9918 | 2 |
| SWU5 | sentence_packed | `fuse_max` | ungrounded | entity_swap | 0.9803 | 1.0000 | 0.0000 | False | 0.9979 | 2 |
| SWU5 | sentence_packed | `fuse_min` | ungrounded | entity_swap | 0.9482 | 1.0000 | 0.0000 | False | 0.9858 | 2 |
| SWG6 | sentence_packed | `gte_only` | grounded | - | 0.9724 | 1.0000 | 0.0000 | False | 0.9967 | 2 |
| SWG6 | sentence_packed | `code_only` | grounded | - | 0.9371 | 1.0000 | 0.0000 | False | 0.9723 | 2 |
| SWG6 | sentence_packed | `fuse_mean` | grounded | - | 0.9547 | 1.0000 | 0.0000 | False | 0.9845 | 2 |
| SWG6 | sentence_packed | `fuse_max` | grounded | - | 0.9724 | 1.0000 | 0.0000 | False | 0.9967 | 2 |
| SWG6 | sentence_packed | `fuse_min` | grounded | - | 0.9371 | 1.0000 | 0.0000 | False | 0.9723 | 2 |
| SWU6 | sentence_packed | `gte_only` | ungrounded | entity_swap | 0.9724 | 1.0000 | 0.0000 | False | 0.9966 | 2 |
| SWU6 | sentence_packed | `code_only` | ungrounded | entity_swap | 0.9353 | 1.0000 | 0.0000 | False | 0.9726 | 2 |
| SWU6 | sentence_packed | `fuse_mean` | ungrounded | entity_swap | 0.9538 | 1.0000 | 0.0000 | False | 0.9846 | 2 |
| SWU6 | sentence_packed | `fuse_max` | ungrounded | entity_swap | 0.9724 | 1.0000 | 0.0000 | False | 0.9966 | 2 |
| SWU6 | sentence_packed | `fuse_min` | ungrounded | entity_swap | 0.9353 | 1.0000 | 0.0000 | False | 0.9726 | 2 |
| SWG7 | sentence_packed | `gte_only` | grounded | - | 0.9847 | 1.0000 | 0.0000 | False | 0.9997 | 3 |
| SWG7 | sentence_packed | `code_only` | grounded | - | 0.9582 | 1.0000 | 0.0000 | False | 0.9890 | 3 |
| SWG7 | sentence_packed | `fuse_mean` | grounded | - | 0.9714 | 1.0000 | 0.0000 | False | 0.9944 | 3 |
| SWG7 | sentence_packed | `fuse_max` | grounded | - | 0.9847 | 1.0000 | 0.0000 | False | 0.9997 | 3 |
| SWG7 | sentence_packed | `fuse_min` | grounded | - | 0.9582 | 1.0000 | 0.0000 | False | 0.9890 | 3 |
| SWU7 | sentence_packed | `gte_only` | ungrounded | phantom_api | 0.9836 | 1.0000 | 0.0000 | False | 0.9998 | 3 |
| SWU7 | sentence_packed | `code_only` | ungrounded | phantom_api | 0.9544 | 1.0000 | 0.0000 | False | 0.9924 | 3 |
| SWU7 | sentence_packed | `fuse_mean` | ungrounded | phantom_api | 0.9690 | 1.0000 | 0.0000 | False | 0.9961 | 3 |
| SWU7 | sentence_packed | `fuse_max` | ungrounded | phantom_api | 0.9836 | 1.0000 | 0.0000 | False | 0.9998 | 3 |
| SWU7 | sentence_packed | `fuse_min` | ungrounded | phantom_api | 0.9544 | 1.0000 | 0.0000 | False | 0.9924 | 3 |
| SWG8 | sentence_packed | `gte_only` | grounded | - | 0.9817 | 1.0000 | 0.0000 | False | 0.9981 | 5 |
| SWG8 | sentence_packed | `code_only` | grounded | - | 0.9600 | 1.0000 | 0.0000 | False | 0.9980 | 5 |
| SWG8 | sentence_packed | `fuse_mean` | grounded | - | 0.9708 | 1.0000 | 0.0000 | False | 0.9981 | 5 |
| SWG8 | sentence_packed | `fuse_max` | grounded | - | 0.9817 | 1.0000 | 0.0000 | False | 0.9981 | 5 |
| SWG8 | sentence_packed | `fuse_min` | grounded | - | 0.9600 | 1.0000 | 0.0000 | False | 0.9980 | 5 |
| SWU8 | sentence_packed | `gte_only` | ungrounded | entity_swap | 0.9817 | 1.0000 | 0.0000 | False | 0.9981 | 5 |
| SWU8 | sentence_packed | `code_only` | ungrounded | entity_swap | 0.9593 | 1.0000 | 0.0000 | False | 0.9977 | 5 |
| SWU8 | sentence_packed | `fuse_mean` | ungrounded | entity_swap | 0.9705 | 1.0000 | 0.0000 | False | 0.9979 | 5 |
| SWU8 | sentence_packed | `fuse_max` | ungrounded | entity_swap | 0.9817 | 1.0000 | 0.0000 | False | 0.9981 | 5 |
| SWU8 | sentence_packed | `fuse_min` | ungrounded | entity_swap | 0.9593 | 1.0000 | 0.0000 | False | 0.9977 | 5 |
| SWG9 | sentence_packed | `gte_only` | grounded | - | 0.9826 | 1.0000 | 0.0000 | False | 0.9991 | 5 |
| SWG9 | sentence_packed | `code_only` | grounded | - | 0.9562 | 1.0000 | 0.0000 | False | 0.9923 | 5 |
| SWG9 | sentence_packed | `fuse_mean` | grounded | - | 0.9694 | 1.0000 | 0.0000 | False | 0.9957 | 5 |
| SWG9 | sentence_packed | `fuse_max` | grounded | - | 0.9826 | 1.0000 | 0.0000 | False | 0.9991 | 5 |
| SWG9 | sentence_packed | `fuse_min` | grounded | - | 0.9562 | 1.0000 | 0.0000 | False | 0.9923 | 5 |
| SWU9 | sentence_packed | `gte_only` | ungrounded | phantom_api | 0.9822 | 1.0000 | 0.0000 | False | 0.9994 | 5 |
| SWU9 | sentence_packed | `code_only` | ungrounded | phantom_api | 0.9528 | 1.0000 | 0.0000 | False | 0.9906 | 5 |
| SWU9 | sentence_packed | `fuse_mean` | ungrounded | phantom_api | 0.9675 | 1.0000 | 0.0000 | False | 0.9950 | 5 |
| SWU9 | sentence_packed | `fuse_max` | ungrounded | phantom_api | 0.9822 | 1.0000 | 0.0000 | False | 0.9994 | 5 |
| SWU9 | sentence_packed | `fuse_min` | ungrounded | phantom_api | 0.9528 | 1.0000 | 0.0000 | False | 0.9906 | 5 |
| SWG10 | sentence_packed | `gte_only` | grounded | - | 0.9773 | 1.0000 | 0.0000 | False | 0.9998 | 3 |
| SWG10 | sentence_packed | `code_only` | grounded | - | 0.9640 | 1.0000 | 0.0000 | False | 0.9962 | 3 |
| SWG10 | sentence_packed | `fuse_mean` | grounded | - | 0.9707 | 1.0000 | 0.0000 | False | 0.9980 | 3 |
| SWG10 | sentence_packed | `fuse_max` | grounded | - | 0.9773 | 1.0000 | 0.0000 | False | 0.9998 | 3 |
| SWG10 | sentence_packed | `fuse_min` | grounded | - | 0.9640 | 1.0000 | 0.0000 | False | 0.9962 | 3 |
| SWU10 | sentence_packed | `gte_only` | ungrounded | phantom_api | 0.9767 | 1.0000 | 0.0000 | False | 0.9998 | 3 |
| SWU10 | sentence_packed | `code_only` | ungrounded | phantom_api | 0.9635 | 1.0000 | 0.0000 | False | 0.9962 | 3 |
| SWU10 | sentence_packed | `fuse_mean` | ungrounded | phantom_api | 0.9701 | 1.0000 | 0.0000 | False | 0.9980 | 3 |
| SWU10 | sentence_packed | `fuse_max` | ungrounded | phantom_api | 0.9767 | 1.0000 | 0.0000 | False | 0.9998 | 3 |
| SWU10 | sentence_packed | `fuse_min` | ungrounded | phantom_api | 0.9635 | 1.0000 | 0.0000 | False | 0.9962 | 3 |
| SWG11 | sentence_packed | `gte_only` | grounded | - | 0.9817 | 1.0000 | 0.0000 | False | 0.9966 | 9 |
| SWG11 | sentence_packed | `code_only` | grounded | - | 0.9574 | 1.0000 | 0.0000 | False | 0.9943 | 9 |
| SWG11 | sentence_packed | `fuse_mean` | grounded | - | 0.9695 | 1.0000 | 0.0000 | False | 0.9955 | 9 |
| SWG11 | sentence_packed | `fuse_max` | grounded | - | 0.9817 | 1.0000 | 0.0000 | False | 0.9966 | 9 |
| SWG11 | sentence_packed | `fuse_min` | grounded | - | 0.9574 | 1.0000 | 0.0000 | False | 0.9943 | 9 |
| SWU11 | sentence_packed | `gte_only` | ungrounded | phantom_api | 0.9813 | 1.0000 | 0.0000 | False | 0.9967 | 9 |
| SWU11 | sentence_packed | `code_only` | ungrounded | phantom_api | 0.9567 | 1.0000 | 0.0000 | False | 0.9910 | 9 |
| SWU11 | sentence_packed | `fuse_mean` | ungrounded | phantom_api | 0.9690 | 1.0000 | 0.0000 | False | 0.9939 | 9 |
| SWU11 | sentence_packed | `fuse_max` | ungrounded | phantom_api | 0.9813 | 1.0000 | 0.0000 | False | 0.9967 | 9 |
| SWU11 | sentence_packed | `fuse_min` | ungrounded | phantom_api | 0.9567 | 1.0000 | 0.0000 | False | 0.9910 | 9 |
| SWG12 | sentence_packed | `gte_only` | grounded | - | 0.9783 | 1.0000 | 0.0000 | False | 0.9967 | 2 |
| SWG12 | sentence_packed | `code_only` | grounded | - | 0.9473 | 1.0000 | 0.0000 | False | 1.0000 | 2 |
| SWG12 | sentence_packed | `fuse_mean` | grounded | - | 0.9628 | 1.0000 | 0.0000 | False | 0.9983 | 2 |
| SWG12 | sentence_packed | `fuse_max` | grounded | - | 0.9783 | 1.0000 | 0.0000 | False | 1.0000 | 2 |
| SWG12 | sentence_packed | `fuse_min` | grounded | - | 0.9473 | 1.0000 | 0.0000 | False | 0.9967 | 2 |
| SWU12 | sentence_packed | `gte_only` | ungrounded | negation_flip | 0.9765 | 1.0000 | 0.0000 | False | 0.9987 | 2 |
| SWU12 | sentence_packed | `code_only` | ungrounded | negation_flip | 0.9461 | 1.0000 | 0.0000 | False | 0.9911 | 2 |
| SWU12 | sentence_packed | `fuse_mean` | ungrounded | negation_flip | 0.9613 | 1.0000 | 0.0000 | False | 0.9949 | 2 |
| SWU12 | sentence_packed | `fuse_max` | ungrounded | negation_flip | 0.9765 | 1.0000 | 0.0000 | False | 0.9987 | 2 |
| SWU12 | sentence_packed | `fuse_min` | ungrounded | negation_flip | 0.9461 | 1.0000 | 0.0000 | False | 0.9911 | 2 |
| SWG13 | sentence_packed | `gte_only` | grounded | - | 0.9811 | 1.0000 | 0.0000 | False | 0.9999 | 8 |
| SWG13 | sentence_packed | `code_only` | grounded | - | 0.9480 | 1.0000 | 0.0000 | False | 0.9940 | 8 |
| SWG13 | sentence_packed | `fuse_mean` | grounded | - | 0.9645 | 1.0000 | 0.0000 | False | 0.9970 | 8 |
| SWG13 | sentence_packed | `fuse_max` | grounded | - | 0.9811 | 1.0000 | 0.0000 | False | 0.9999 | 8 |
| SWG13 | sentence_packed | `fuse_min` | grounded | - | 0.9480 | 1.0000 | 0.0000 | False | 0.9940 | 8 |
| SWU13 | sentence_packed | `gte_only` | ungrounded | entity_swap | 0.9826 | 1.0000 | 0.0000 | False | 1.0000 | 8 |
| SWU13 | sentence_packed | `code_only` | ungrounded | entity_swap | 0.9532 | 1.0000 | 0.0000 | False | 0.9929 | 8 |
| SWU13 | sentence_packed | `fuse_mean` | ungrounded | entity_swap | 0.9679 | 1.0000 | 0.0000 | False | 0.9964 | 8 |
| SWU13 | sentence_packed | `fuse_max` | ungrounded | entity_swap | 0.9826 | 1.0000 | 0.0000 | False | 1.0000 | 8 |
| SWU13 | sentence_packed | `fuse_min` | ungrounded | entity_swap | 0.9532 | 1.0000 | 0.0000 | False | 0.9929 | 8 |
| SWG14 | sentence_packed | `gte_only` | grounded | - | 0.9833 | 1.0000 | 0.0000 | False | 0.9968 | 3 |
| SWG14 | sentence_packed | `code_only` | grounded | - | 0.9449 | 1.0000 | 0.0000 | False | 0.9861 | 3 |
| SWG14 | sentence_packed | `fuse_mean` | grounded | - | 0.9641 | 1.0000 | 0.0000 | False | 0.9914 | 3 |
| SWG14 | sentence_packed | `fuse_max` | grounded | - | 0.9833 | 1.0000 | 0.0000 | False | 0.9968 | 3 |
| SWG14 | sentence_packed | `fuse_min` | grounded | - | 0.9449 | 1.0000 | 0.0000 | False | 0.9861 | 3 |
| SWU14 | sentence_packed | `gte_only` | ungrounded | entity_swap | 0.9833 | 1.0000 | 0.0000 | False | 0.9968 | 3 |
| SWU14 | sentence_packed | `code_only` | ungrounded | entity_swap | 0.9449 | 1.0000 | 0.0000 | False | 0.9861 | 3 |
| SWU14 | sentence_packed | `fuse_mean` | ungrounded | entity_swap | 0.9641 | 1.0000 | 0.0000 | False | 0.9915 | 3 |
| SWU14 | sentence_packed | `fuse_max` | ungrounded | entity_swap | 0.9833 | 1.0000 | 0.0000 | False | 0.9968 | 3 |
| SWU14 | sentence_packed | `fuse_min` | ungrounded | entity_swap | 0.9449 | 1.0000 | 0.0000 | False | 0.9861 | 3 |
| SWG15 | sentence_packed | `gte_only` | grounded | - | 0.9761 | 1.0000 | 0.0000 | False | 0.9975 | 6 |
| SWG15 | sentence_packed | `code_only` | grounded | - | 0.9594 | 1.0000 | 0.0000 | False | 0.9913 | 6 |
| SWG15 | sentence_packed | `fuse_mean` | grounded | - | 0.9678 | 1.0000 | 0.0000 | False | 0.9944 | 6 |
| SWG15 | sentence_packed | `fuse_max` | grounded | - | 0.9761 | 1.0000 | 0.0000 | False | 0.9975 | 6 |
| SWG15 | sentence_packed | `fuse_min` | grounded | - | 0.9594 | 1.0000 | 0.0000 | False | 0.9913 | 6 |
| SWU15 | sentence_packed | `gte_only` | ungrounded | entity_swap | 0.9778 | 1.0000 | 0.0000 | False | 0.9914 | 6 |
| SWU15 | sentence_packed | `code_only` | ungrounded | entity_swap | 0.9536 | 1.0000 | 0.0000 | False | 0.9755 | 6 |
| SWU15 | sentence_packed | `fuse_mean` | ungrounded | entity_swap | 0.9657 | 1.0000 | 0.0000 | False | 0.9834 | 6 |
| SWU15 | sentence_packed | `fuse_max` | ungrounded | entity_swap | 0.9778 | 1.0000 | 0.0000 | False | 0.9914 | 6 |
| SWU15 | sentence_packed | `fuse_min` | ungrounded | entity_swap | 0.9536 | 1.0000 | 0.0000 | False | 0.9755 | 6 |
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
| CU1 | colgrep | `gte_only` | ungrounded | phantom_api | 0.9820 | 1.0000 | 0.0000 | False | 0.9998 | 2 |
| CU1 | colgrep | `code_only` | ungrounded | phantom_api | 0.9360 | 1.0000 | 0.0000 | False | 0.9998 | 2 |
| CU1 | colgrep | `fuse_mean` | ungrounded | phantom_api | 0.9590 | 1.0000 | 0.0000 | False | 0.9998 | 2 |
| CU1 | colgrep | `fuse_max` | ungrounded | phantom_api | 0.9820 | 1.0000 | 0.0000 | False | 0.9998 | 2 |
| CU1 | colgrep | `fuse_min` | ungrounded | phantom_api | 0.9360 | 1.0000 | 0.0000 | False | 0.9998 | 2 |
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
| SWG1 | colgrep | `gte_only` | grounded | - | 0.9544 | 1.0000 | 0.0000 | False | 0.9955 | 2 |
| SWG1 | colgrep | `code_only` | grounded | - | 0.8793 | 1.0000 | 0.0000 | False | 0.9714 | 2 |
| SWG1 | colgrep | `fuse_mean` | grounded | - | 0.9169 | 1.0000 | 0.0000 | False | 0.9835 | 2 |
| SWG1 | colgrep | `fuse_max` | grounded | - | 0.9544 | 1.0000 | 0.0000 | False | 0.9955 | 2 |
| SWG1 | colgrep | `fuse_min` | grounded | - | 0.8793 | 1.0000 | 0.0000 | False | 0.9714 | 2 |
| SWU1 | colgrep | `gte_only` | ungrounded | entity_swap | 0.9538 | 1.0000 | 0.0000 | False | 0.9954 | 2 |
| SWU1 | colgrep | `code_only` | ungrounded | entity_swap | 0.8870 | 1.0000 | 0.0000 | False | 0.9779 | 2 |
| SWU1 | colgrep | `fuse_mean` | ungrounded | entity_swap | 0.9204 | 1.0000 | 0.0000 | False | 0.9866 | 2 |
| SWU1 | colgrep | `fuse_max` | ungrounded | entity_swap | 0.9538 | 1.0000 | 0.0000 | False | 0.9954 | 2 |
| SWU1 | colgrep | `fuse_min` | ungrounded | entity_swap | 0.8870 | 1.0000 | 0.0000 | False | 0.9779 | 2 |
| SWG2 | colgrep | `gte_only` | grounded | - | 0.9553 | 1.0000 | 0.0000 | False | 1.0000 | 1 |
| SWG2 | colgrep | `code_only` | grounded | - | 0.8892 | 1.0000 | 0.0000 | False | 0.9804 | 1 |
| SWG2 | colgrep | `fuse_mean` | grounded | - | 0.9223 | 1.0000 | 0.0000 | False | 0.9902 | 1 |
| SWG2 | colgrep | `fuse_max` | grounded | - | 0.9553 | 1.0000 | 0.0000 | False | 1.0000 | 1 |
| SWG2 | colgrep | `fuse_min` | grounded | - | 0.8892 | 1.0000 | 0.0000 | False | 0.9804 | 1 |
| SWU2 | colgrep | `gte_only` | ungrounded | phantom_api | 0.9566 | 1.0000 | 0.0000 | False | 0.9998 | 1 |
| SWU2 | colgrep | `code_only` | ungrounded | phantom_api | 0.8877 | 1.0000 | 0.0000 | False | 0.9735 | 1 |
| SWU2 | colgrep | `fuse_mean` | ungrounded | phantom_api | 0.9221 | 1.0000 | 0.0000 | False | 0.9867 | 1 |
| SWU2 | colgrep | `fuse_max` | ungrounded | phantom_api | 0.9566 | 1.0000 | 0.0000 | False | 0.9998 | 1 |
| SWU2 | colgrep | `fuse_min` | ungrounded | phantom_api | 0.8877 | 1.0000 | 0.0000 | False | 0.9735 | 1 |
| SWG3 | colgrep | `gte_only` | grounded | - | 0.9790 | 1.0000 | 0.0000 | False | 0.9973 | 3 |
| SWG3 | colgrep | `code_only` | grounded | - | 0.9390 | 1.0000 | 0.0000 | False | 0.9864 | 3 |
| SWG3 | colgrep | `fuse_mean` | grounded | - | 0.9590 | 1.0000 | 0.0000 | False | 0.9918 | 3 |
| SWG3 | colgrep | `fuse_max` | grounded | - | 0.9790 | 1.0000 | 0.0000 | False | 0.9973 | 3 |
| SWG3 | colgrep | `fuse_min` | grounded | - | 0.9390 | 1.0000 | 0.0000 | False | 0.9864 | 3 |
| SWU3 | colgrep | `gte_only` | ungrounded | negation_flip | 0.9799 | 1.0000 | 0.0000 | False | 0.9973 | 3 |
| SWU3 | colgrep | `code_only` | ungrounded | negation_flip | 0.9333 | 1.0000 | 0.0000 | False | 0.9620 | 3 |
| SWU3 | colgrep | `fuse_mean` | ungrounded | negation_flip | 0.9566 | 1.0000 | 0.0000 | False | 0.9796 | 3 |
| SWU3 | colgrep | `fuse_max` | ungrounded | negation_flip | 0.9799 | 1.0000 | 0.0000 | False | 0.9973 | 3 |
| SWU3 | colgrep | `fuse_min` | ungrounded | negation_flip | 0.9333 | 1.0000 | 0.0000 | False | 0.9620 | 3 |
| SWG4 | colgrep | `gte_only` | grounded | - | 0.9889 | 1.0000 | 0.0000 | False | 0.9967 | 6 |
| SWG4 | colgrep | `code_only` | grounded | - | 0.9566 | 1.0000 | 0.0000 | False | 0.9967 | 6 |
| SWG4 | colgrep | `fuse_mean` | grounded | - | 0.9727 | 1.0000 | 0.0000 | False | 0.9967 | 6 |
| SWG4 | colgrep | `fuse_max` | grounded | - | 0.9889 | 1.0000 | 0.0000 | False | 0.9967 | 6 |
| SWG4 | colgrep | `fuse_min` | grounded | - | 0.9566 | 1.0000 | 0.0000 | False | 0.9967 | 6 |
| SWU4 | colgrep | `gte_only` | ungrounded | entity_swap | 0.9870 | 1.0000 | 0.0000 | False | 0.9968 | 6 |
| SWU4 | colgrep | `code_only` | ungrounded | entity_swap | 0.9524 | 1.0000 | 0.0000 | False | 0.9978 | 6 |
| SWU4 | colgrep | `fuse_mean` | ungrounded | entity_swap | 0.9697 | 1.0000 | 0.0000 | False | 0.9973 | 6 |
| SWU4 | colgrep | `fuse_max` | ungrounded | entity_swap | 0.9870 | 1.0000 | 0.0000 | False | 0.9978 | 6 |
| SWU4 | colgrep | `fuse_min` | ungrounded | entity_swap | 0.9524 | 1.0000 | 0.0000 | False | 0.9968 | 6 |
| SWG5 | colgrep | `gte_only` | grounded | - | 0.9833 | 1.0000 | 0.0000 | False | 0.9998 | 3 |
| SWG5 | colgrep | `code_only` | grounded | - | 0.9447 | 1.0000 | 0.0000 | False | 0.9955 | 3 |
| SWG5 | colgrep | `fuse_mean` | grounded | - | 0.9640 | 1.0000 | 0.0000 | False | 0.9977 | 3 |
| SWG5 | colgrep | `fuse_max` | grounded | - | 0.9833 | 1.0000 | 0.0000 | False | 0.9998 | 3 |
| SWG5 | colgrep | `fuse_min` | grounded | - | 0.9447 | 1.0000 | 0.0000 | False | 0.9955 | 3 |
| SWU5 | colgrep | `gte_only` | ungrounded | entity_swap | 0.9838 | 1.0000 | 0.0000 | False | 0.9991 | 3 |
| SWU5 | colgrep | `code_only` | ungrounded | entity_swap | 0.9486 | 1.0000 | 0.0000 | False | 0.9957 | 3 |
| SWU5 | colgrep | `fuse_mean` | ungrounded | entity_swap | 0.9662 | 1.0000 | 0.0000 | False | 0.9974 | 3 |
| SWU5 | colgrep | `fuse_max` | ungrounded | entity_swap | 0.9838 | 1.0000 | 0.0000 | False | 0.9991 | 3 |
| SWU5 | colgrep | `fuse_min` | ungrounded | entity_swap | 0.9486 | 1.0000 | 0.0000 | False | 0.9957 | 3 |
| SWG6 | colgrep | `gte_only` | grounded | - | 0.9577 | 1.0000 | 0.0000 | False | 0.9964 | 2 |
| SWG6 | colgrep | `code_only` | grounded | - | 0.9013 | 1.0000 | 0.0000 | False | 0.9720 | 2 |
| SWG6 | colgrep | `fuse_mean` | grounded | - | 0.9295 | 1.0000 | 0.0000 | False | 0.9842 | 2 |
| SWG6 | colgrep | `fuse_max` | grounded | - | 0.9577 | 1.0000 | 0.0000 | False | 0.9964 | 2 |
| SWG6 | colgrep | `fuse_min` | grounded | - | 0.9013 | 1.0000 | 0.0000 | False | 0.9720 | 2 |
| SWU6 | colgrep | `gte_only` | ungrounded | entity_swap | 0.9572 | 1.0000 | 0.0000 | False | 0.9964 | 2 |
| SWU6 | colgrep | `code_only` | ungrounded | entity_swap | 0.9015 | 1.0000 | 0.0000 | False | 0.9744 | 2 |
| SWU6 | colgrep | `fuse_mean` | ungrounded | entity_swap | 0.9294 | 1.0000 | 0.0000 | False | 0.9854 | 2 |
| SWU6 | colgrep | `fuse_max` | ungrounded | entity_swap | 0.9572 | 1.0000 | 0.0000 | False | 0.9964 | 2 |
| SWU6 | colgrep | `fuse_min` | ungrounded | entity_swap | 0.9015 | 1.0000 | 0.0000 | False | 0.9744 | 2 |
| SWG7 | colgrep | `gte_only` | grounded | - | 0.9845 | 1.0000 | 0.0000 | False | 0.9995 | 3 |
| SWG7 | colgrep | `code_only` | grounded | - | 0.9527 | 1.0000 | 0.0000 | False | 0.9918 | 3 |
| SWG7 | colgrep | `fuse_mean` | grounded | - | 0.9686 | 1.0000 | 0.0000 | False | 0.9957 | 3 |
| SWG7 | colgrep | `fuse_max` | grounded | - | 0.9845 | 1.0000 | 0.0000 | False | 0.9995 | 3 |
| SWG7 | colgrep | `fuse_min` | grounded | - | 0.9527 | 1.0000 | 0.0000 | False | 0.9918 | 3 |
| SWU7 | colgrep | `gte_only` | ungrounded | phantom_api | 0.9837 | 1.0000 | 0.0000 | False | 0.9998 | 3 |
| SWU7 | colgrep | `code_only` | ungrounded | phantom_api | 0.9505 | 1.0000 | 0.0000 | False | 0.9945 | 3 |
| SWU7 | colgrep | `fuse_mean` | ungrounded | phantom_api | 0.9671 | 1.0000 | 0.0000 | False | 0.9971 | 3 |
| SWU7 | colgrep | `fuse_max` | ungrounded | phantom_api | 0.9837 | 1.0000 | 0.0000 | False | 0.9998 | 3 |
| SWU7 | colgrep | `fuse_min` | ungrounded | phantom_api | 0.9505 | 1.0000 | 0.0000 | False | 0.9945 | 3 |
| SWG8 | colgrep | `gte_only` | grounded | - | 0.9827 | 1.0000 | 0.0000 | False | 0.9978 | 12 |
| SWG8 | colgrep | `code_only` | grounded | - | 0.9608 | 1.0000 | 0.0000 | False | 0.9980 | 12 |
| SWG8 | colgrep | `fuse_mean` | grounded | - | 0.9718 | 1.0000 | 0.0000 | False | 0.9979 | 12 |
| SWG8 | colgrep | `fuse_max` | grounded | - | 0.9827 | 1.0000 | 0.0000 | False | 0.9980 | 12 |
| SWG8 | colgrep | `fuse_min` | grounded | - | 0.9608 | 1.0000 | 0.0000 | False | 0.9978 | 12 |
| SWU8 | colgrep | `gte_only` | ungrounded | entity_swap | 0.9828 | 1.0000 | 0.0000 | False | 0.9978 | 12 |
| SWU8 | colgrep | `code_only` | ungrounded | entity_swap | 0.9602 | 1.0000 | 0.0000 | False | 0.9978 | 12 |
| SWU8 | colgrep | `fuse_mean` | ungrounded | entity_swap | 0.9715 | 1.0000 | 0.0000 | False | 0.9978 | 12 |
| SWU8 | colgrep | `fuse_max` | ungrounded | entity_swap | 0.9828 | 1.0000 | 0.0000 | False | 0.9978 | 12 |
| SWU8 | colgrep | `fuse_min` | ungrounded | entity_swap | 0.9602 | 1.0000 | 0.0000 | False | 0.9978 | 12 |
| SWG9 | colgrep | `gte_only` | grounded | - | 0.9812 | 1.0000 | 0.0000 | False | 0.9989 | 11 |
| SWG9 | colgrep | `code_only` | grounded | - | 0.9508 | 1.0000 | 0.0000 | False | 0.9910 | 11 |
| SWG9 | colgrep | `fuse_mean` | grounded | - | 0.9660 | 1.0000 | 0.0000 | False | 0.9950 | 11 |
| SWG9 | colgrep | `fuse_max` | grounded | - | 0.9812 | 1.0000 | 0.0000 | False | 0.9989 | 11 |
| SWG9 | colgrep | `fuse_min` | grounded | - | 0.9508 | 1.0000 | 0.0000 | False | 0.9910 | 11 |
| SWU9 | colgrep | `gte_only` | ungrounded | phantom_api | 0.9806 | 1.0000 | 0.0000 | False | 0.9998 | 11 |
| SWU9 | colgrep | `code_only` | ungrounded | phantom_api | 0.9484 | 1.0000 | 0.0000 | False | 0.9916 | 11 |
| SWU9 | colgrep | `fuse_mean` | ungrounded | phantom_api | 0.9645 | 1.0000 | 0.0000 | False | 0.9957 | 11 |
| SWU9 | colgrep | `fuse_max` | ungrounded | phantom_api | 0.9806 | 1.0000 | 0.0000 | False | 0.9998 | 11 |
| SWU9 | colgrep | `fuse_min` | ungrounded | phantom_api | 0.9484 | 1.0000 | 0.0000 | False | 0.9916 | 11 |
| SWG10 | colgrep | `gte_only` | grounded | - | 0.9852 | 1.0000 | 0.0000 | False | 0.9999 | 6 |
| SWG10 | colgrep | `code_only` | grounded | - | 0.9644 | 1.0000 | 0.0000 | False | 0.9963 | 6 |
| SWG10 | colgrep | `fuse_mean` | grounded | - | 0.9748 | 1.0000 | 0.0000 | False | 0.9981 | 6 |
| SWG10 | colgrep | `fuse_max` | grounded | - | 0.9852 | 1.0000 | 0.0000 | False | 0.9999 | 6 |
| SWG10 | colgrep | `fuse_min` | grounded | - | 0.9644 | 1.0000 | 0.0000 | False | 0.9963 | 6 |
| SWU10 | colgrep | `gte_only` | ungrounded | phantom_api | 0.9842 | 1.0000 | 0.0000 | False | 0.9999 | 6 |
| SWU10 | colgrep | `code_only` | ungrounded | phantom_api | 0.9653 | 1.0000 | 0.0000 | False | 0.9963 | 6 |
| SWU10 | colgrep | `fuse_mean` | ungrounded | phantom_api | 0.9748 | 1.0000 | 0.0000 | False | 0.9981 | 6 |
| SWU10 | colgrep | `fuse_max` | ungrounded | phantom_api | 0.9842 | 1.0000 | 0.0000 | False | 0.9999 | 6 |
| SWU10 | colgrep | `fuse_min` | ungrounded | phantom_api | 0.9653 | 1.0000 | 0.0000 | False | 0.9963 | 6 |
| SWG11 | colgrep | `gte_only` | grounded | - | 0.9865 | 1.0000 | 0.0000 | False | 0.9976 | 22 |
| SWG11 | colgrep | `code_only` | grounded | - | 0.9586 | 1.0000 | 0.0000 | False | 0.9952 | 22 |
| SWG11 | colgrep | `fuse_mean` | grounded | - | 0.9725 | 1.0000 | 0.0000 | False | 0.9964 | 22 |
| SWG11 | colgrep | `fuse_max` | grounded | - | 0.9865 | 1.0000 | 0.0000 | False | 0.9976 | 22 |
| SWG11 | colgrep | `fuse_min` | grounded | - | 0.9586 | 1.0000 | 0.0000 | False | 0.9952 | 22 |
| SWU11 | colgrep | `gte_only` | ungrounded | phantom_api | 0.9861 | 1.0000 | 0.0000 | False | 0.9977 | 22 |
| SWU11 | colgrep | `code_only` | ungrounded | phantom_api | 0.9591 | 1.0000 | 0.0000 | False | 0.9947 | 22 |
| SWU11 | colgrep | `fuse_mean` | ungrounded | phantom_api | 0.9726 | 1.0000 | 0.0000 | False | 0.9962 | 22 |
| SWU11 | colgrep | `fuse_max` | ungrounded | phantom_api | 0.9861 | 1.0000 | 0.0000 | False | 0.9977 | 22 |
| SWU11 | colgrep | `fuse_min` | ungrounded | phantom_api | 0.9591 | 1.0000 | 0.0000 | False | 0.9947 | 22 |
| SWG12 | colgrep | `gte_only` | grounded | - | 0.9687 | 1.0000 | 0.0000 | False | 0.9964 | 2 |
| SWG12 | colgrep | `code_only` | grounded | - | 0.9252 | 1.0000 | 0.0000 | False | 1.0000 | 2 |
| SWG12 | colgrep | `fuse_mean` | grounded | - | 0.9470 | 1.0000 | 0.0000 | False | 0.9982 | 2 |
| SWG12 | colgrep | `fuse_max` | grounded | - | 0.9687 | 1.0000 | 0.0000 | False | 1.0000 | 2 |
| SWG12 | colgrep | `fuse_min` | grounded | - | 0.9252 | 1.0000 | 0.0000 | False | 0.9964 | 2 |
| SWU12 | colgrep | `gte_only` | ungrounded | negation_flip | 0.9674 | 1.0000 | 0.0000 | False | 0.9988 | 2 |
| SWU12 | colgrep | `code_only` | ungrounded | negation_flip | 0.9265 | 1.0000 | 0.0000 | False | 0.9936 | 2 |
| SWU12 | colgrep | `fuse_mean` | ungrounded | negation_flip | 0.9470 | 1.0000 | 0.0000 | False | 0.9962 | 2 |
| SWU12 | colgrep | `fuse_max` | ungrounded | negation_flip | 0.9674 | 1.0000 | 0.0000 | False | 0.9988 | 2 |
| SWU12 | colgrep | `fuse_min` | ungrounded | negation_flip | 0.9265 | 1.0000 | 0.0000 | False | 0.9936 | 2 |
| SWG13 | colgrep | `gte_only` | grounded | - | 0.9813 | 1.0000 | 0.0000 | False | 1.0000 | 36 |
| SWG13 | colgrep | `code_only` | grounded | - | 0.9487 | 1.0000 | 0.0000 | False | 0.9921 | 36 |
| SWG13 | colgrep | `fuse_mean` | grounded | - | 0.9650 | 1.0000 | 0.0000 | False | 0.9960 | 36 |
| SWG13 | colgrep | `fuse_max` | grounded | - | 0.9813 | 1.0000 | 0.0000 | False | 1.0000 | 36 |
| SWG13 | colgrep | `fuse_min` | grounded | - | 0.9487 | 1.0000 | 0.0000 | False | 0.9921 | 36 |
| SWU13 | colgrep | `gte_only` | ungrounded | entity_swap | 0.9817 | 1.0000 | 0.0000 | False | 0.9995 | 36 |
| SWU13 | colgrep | `code_only` | ungrounded | entity_swap | 0.9537 | 1.0000 | 0.0000 | False | 0.9911 | 36 |
| SWU13 | colgrep | `fuse_mean` | ungrounded | entity_swap | 0.9677 | 1.0000 | 0.0000 | False | 0.9953 | 36 |
| SWU13 | colgrep | `fuse_max` | ungrounded | entity_swap | 0.9817 | 1.0000 | 0.0000 | False | 0.9995 | 36 |
| SWU13 | colgrep | `fuse_min` | ungrounded | entity_swap | 0.9537 | 1.0000 | 0.0000 | False | 0.9911 | 36 |
| SWG14 | colgrep | `gte_only` | grounded | - | 0.9873 | 1.0000 | 0.0000 | False | 0.9975 | 8 |
| SWG14 | colgrep | `code_only` | grounded | - | 0.9528 | 1.0000 | 0.0000 | False | 0.9912 | 8 |
| SWG14 | colgrep | `fuse_mean` | grounded | - | 0.9701 | 1.0000 | 0.0000 | False | 0.9943 | 8 |
| SWG14 | colgrep | `fuse_max` | grounded | - | 0.9873 | 1.0000 | 0.0000 | False | 0.9975 | 8 |
| SWG14 | colgrep | `fuse_min` | grounded | - | 0.9528 | 1.0000 | 0.0000 | False | 0.9912 | 8 |
| SWU14 | colgrep | `gte_only` | ungrounded | entity_swap | 0.9873 | 1.0000 | 0.0000 | False | 0.9975 | 8 |
| SWU14 | colgrep | `code_only` | ungrounded | entity_swap | 0.9524 | 1.0000 | 0.0000 | False | 0.9912 | 8 |
| SWU14 | colgrep | `fuse_mean` | ungrounded | entity_swap | 0.9698 | 1.0000 | 0.0000 | False | 0.9944 | 8 |
| SWU14 | colgrep | `fuse_max` | ungrounded | entity_swap | 0.9873 | 1.0000 | 0.0000 | False | 0.9975 | 8 |
| SWU14 | colgrep | `fuse_min` | ungrounded | entity_swap | 0.9524 | 1.0000 | 0.0000 | False | 0.9912 | 8 |
| SWG15 | colgrep | `gte_only` | grounded | - | 0.9887 | 1.0000 | 0.0000 | False | 0.9975 | 33 |
| SWG15 | colgrep | `code_only` | grounded | - | 0.9627 | 1.0000 | 0.0000 | False | 0.9901 | 33 |
| SWG15 | colgrep | `fuse_mean` | grounded | - | 0.9757 | 1.0000 | 0.0000 | False | 0.9938 | 33 |
| SWG15 | colgrep | `fuse_max` | grounded | - | 0.9887 | 1.0000 | 0.0000 | False | 0.9975 | 33 |
| SWG15 | colgrep | `fuse_min` | grounded | - | 0.9627 | 1.0000 | 0.0000 | False | 0.9901 | 33 |
| SWU15 | colgrep | `gte_only` | ungrounded | entity_swap | 0.9871 | 1.0000 | 0.0000 | False | 0.9957 | 33 |
| SWU15 | colgrep | `code_only` | ungrounded | entity_swap | 0.9590 | 1.0000 | 0.0000 | False | 0.9919 | 33 |
| SWU15 | colgrep | `fuse_mean` | ungrounded | entity_swap | 0.9730 | 1.0000 | 0.0000 | False | 0.9938 | 33 |
| SWU15 | colgrep | `fuse_max` | ungrounded | entity_swap | 0.9871 | 1.0000 | 0.0000 | False | 0.9957 | 33 |
| SWU15 | colgrep | `fuse_min` | ungrounded | entity_swap | 0.9590 | 1.0000 | 0.0000 | False | 0.9919 | 33 |
