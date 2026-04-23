# Coding-Agent Groundedness: Scorer Stack x Chunker

> NO — the flagship colgrep x fuse_mean cell did not clear the RAG-style gate: reverse_context AUROC=0.585, phantom-API precision@0.35=0.00, ungrounded minus grounded unused-ratio delta=+0.0000. Stacking lift vs gte_only=+0.0437.

- run tag: `chunk_size_sweep_v1__b1024_code`
- primary scorer: `lightonai/GTE-ModernColBERT-v1`
- orthogonal scorer: `lightonai/LateOn-Code-edge` (available: `True`)
- shared chunk budget: `1024` tokens (GTE tokenizer)
- max SWE-bench instances: `15`
- phantom threshold: `0.35`
- flagship cell: `colgrep` x `fuse_mean` (present: `True`)

## Headline matrix

| chunker | scorer | AUROC | grounded cov | cov delta | unused delta | phantom@thr | grounded held-out rate | p95 ms | support units |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sentence_packed | `gte_only` | 0.5569 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 193.89 | 1.8 |
| sentence_packed | `code_only` | 0.5926 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 206.43 | 1.8 |
| sentence_packed | `fuse_mean` | 0.6058 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 205.28 | 1.8 |
| sentence_packed | `fuse_max` | 0.5688 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 205.28 | 1.8 |
| sentence_packed | `fuse_min` | 0.5913 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 205.28 | 1.8 |
| colgrep | `gte_only` | 0.5410 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 293.99 | 7.3 |
| colgrep | `code_only` | 0.5794 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 301.77 | 7.3 |
| colgrep | `fuse_mean` (flagship) | 0.5847 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 301.10 | 7.3 |
| colgrep | `fuse_max` | 0.5410 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 301.10 | 7.3 |
| colgrep | `fuse_min` | 0.5794 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 301.10 | 7.3 |

`*` marks cells that pass `reverse_context AUROC >= 0.90`.

## Score distribution diagnostics (reverse_context)

| chunker | scorer | grounded mean | grounded p10-p90 | ungrounded mean | ungrounded p10-p90 | separation | saturation@0.95 |
|---|---|---:|---|---:|---|---:|---:|
| sentence_packed | `gte_only` | 0.9715 | 0.9616-0.9791 | 0.9708 | 0.9642-0.9785 | +0.0007 | 1.0000 |
| sentence_packed | `code_only` | 0.9394 | 0.9198-0.9614 | 0.9322 | 0.9016-0.9567 | +0.0072 | 0.2564 |
| sentence_packed | `fuse_mean` | 0.9554 | 0.9442-0.9666 | 0.9515 | 0.9347-0.9661 | +0.0039 | 0.6410 |
| sentence_packed | `fuse_max` | 0.9721 | 0.9640-0.9791 | 0.9712 | 0.9651-0.9785 | +0.0009 | 1.0000 |
| sentence_packed | `fuse_min` | 0.9388 | 0.9198-0.9596 | 0.9318 | 0.9016-0.9567 | +0.0070 | 0.2564 |
| colgrep | `gte_only` | 0.9735 | 0.9525-0.9872 | 0.9733 | 0.9534-0.9852 | +0.0002 | 0.8974 |
| colgrep | `code_only` | 0.9392 | 0.8925-0.9639 | 0.9320 | 0.8877-0.9590 | +0.0072 | 0.3846 |
| colgrep | `fuse_mean` | 0.9564 | 0.9234-0.9728 | 0.9527 | 0.9221-0.9720 | +0.0037 | 0.7436 |
| colgrep | `fuse_max` | 0.9737 | 0.9525-0.9872 | 0.9734 | 0.9534-0.9852 | +0.0002 | 0.8974 |
| colgrep | `fuse_min` | 0.9390 | 0.8925-0.9631 | 0.9319 | 0.8877-0.9590 | +0.0071 | 0.3846 |

`separation` is mean(grounded) − mean(ungrounded) on `reverse_context`. `saturation@0.95` is the fraction of anchor cases (grounded + ungrounded) whose score ≥ 0.95 — a ceiling effect indicator.

## Stacking lift (fuse_* minus gte_only, per chunker)

| chunker | rule | AUROC lift | unused-delta lift | phantom precision lift |
|---|---|---:|---:|---:|
| sentence_packed | fuse_mean | +0.0489 | +0.0000 | +0.0000 |
| sentence_packed | fuse_max | +0.0119 | +0.0000 | +0.0000 |
| sentence_packed | fuse_min | +0.0344 | +0.0000 | +0.0000 |
| colgrep | fuse_mean | +0.0437 | +0.0000 | +0.0000 |
| colgrep | fuse_max | +0.0000 | +0.0000 | +0.0000 |
| colgrep | fuse_min | +0.0384 | +0.0000 | +0.0000 |

## Chunker lift (colgrep minus sentence_packed, per scorer)

| scorer | AUROC lift | unused-delta lift | phantom precision lift |
|---|---:|---:|---:|
| `gte_only` | -0.0159 | +0.0000 | +0.0000 |
| `code_only` | -0.0132 | +0.0000 | +0.0000 |
| `fuse_mean` | -0.0212 | +0.0000 | +0.0000 |
| `fuse_max` | -0.0278 | +0.0000 | +0.0000 |
| `fuse_min` | -0.0119 | +0.0000 | +0.0000 |

## Subcategory snapshot (per cell)

### `sentence_packed` x `gte_only`

| subcategory | count | reverse_context | coverage | unused ratio | max evidence |
|---|---:|---:|---:|---:|---:|
| entity_swap | 10 | 0.9721 | 1.0000 | 0.0000 | 0.9965 |
| grounded | 18 | 0.9715 | 1.0000 | 0.0000 | 0.9973 |
| negation_flip | 3 | 0.9682 | 1.0000 | 0.0000 | 0.9974 |
| parametric | 2 | 0.9697 | 1.0000 | 0.0000 | 0.9961 |
| partial | 2 | 0.9696 | 1.0000 | 0.0000 | 0.9976 |
| phantom_api | 7 | 0.9704 | 1.0000 | 0.0000 | 0.9984 |

### `sentence_packed` x `code_only`

| subcategory | count | reverse_context | coverage | unused ratio | max evidence |
|---|---:|---:|---:|---:|---:|
| entity_swap | 10 | 0.9364 | 1.0000 | 0.0000 | 0.9871 |
| grounded | 18 | 0.9394 | 1.0000 | 0.0000 | 0.9902 |
| negation_flip | 3 | 0.9197 | 1.0000 | 0.0000 | 0.9825 |
| parametric | 2 | 0.9081 | 1.0000 | 0.0000 | 0.9863 |
| partial | 2 | 0.9193 | 1.0000 | 0.0000 | 0.9981 |
| phantom_api | 7 | 0.9320 | 1.0000 | 0.0000 | 0.9848 |

### `sentence_packed` x `fuse_mean`

| subcategory | count | reverse_context | coverage | unused ratio | max evidence |
|---|---:|---:|---:|---:|---:|
| entity_swap | 10 | 0.9543 | 1.0000 | 0.0000 | 0.9918 |
| grounded | 18 | 0.9554 | 1.0000 | 0.0000 | 0.9938 |
| negation_flip | 3 | 0.9439 | 1.0000 | 0.0000 | 0.9900 |
| parametric | 2 | 0.9389 | 1.0000 | 0.0000 | 0.9912 |
| partial | 2 | 0.9445 | 1.0000 | 0.0000 | 0.9978 |
| phantom_api | 7 | 0.9512 | 1.0000 | 0.0000 | 0.9916 |

### `sentence_packed` x `fuse_max`

| subcategory | count | reverse_context | coverage | unused ratio | max evidence |
|---|---:|---:|---:|---:|---:|
| entity_swap | 10 | 0.9727 | 1.0000 | 0.0000 | 0.9970 |
| grounded | 18 | 0.9721 | 1.0000 | 0.0000 | 0.9981 |
| negation_flip | 3 | 0.9682 | 1.0000 | 0.0000 | 0.9974 |
| parametric | 2 | 0.9697 | 1.0000 | 0.0000 | 0.9978 |
| partial | 2 | 0.9696 | 1.0000 | 0.0000 | 0.9991 |
| phantom_api | 7 | 0.9709 | 1.0000 | 0.0000 | 0.9984 |

### `sentence_packed` x `fuse_min`

| subcategory | count | reverse_context | coverage | unused ratio | max evidence |
|---|---:|---:|---:|---:|---:|
| entity_swap | 10 | 0.9358 | 1.0000 | 0.0000 | 0.9866 |
| grounded | 18 | 0.9388 | 1.0000 | 0.0000 | 0.9895 |
| negation_flip | 3 | 0.9197 | 1.0000 | 0.0000 | 0.9825 |
| parametric | 2 | 0.9081 | 1.0000 | 0.0000 | 0.9847 |
| partial | 2 | 0.9193 | 1.0000 | 0.0000 | 0.9966 |
| phantom_api | 7 | 0.9315 | 1.0000 | 0.0000 | 0.9848 |

### `colgrep` x `gte_only`

| subcategory | count | reverse_context | coverage | unused ratio | max evidence |
|---|---:|---:|---:|---:|---:|
| entity_swap | 10 | 0.9741 | 1.0000 | 0.0000 | 0.9969 |
| grounded | 18 | 0.9735 | 1.0000 | 0.0000 | 0.9977 |
| negation_flip | 3 | 0.9691 | 1.0000 | 0.0000 | 0.9979 |
| parametric | 2 | 0.9789 | 1.0000 | 0.0000 | 0.9986 |
| partial | 2 | 0.9793 | 1.0000 | 0.0000 | 0.9981 |
| phantom_api | 7 | 0.9734 | 1.0000 | 0.0000 | 0.9992 |

### `colgrep` x `code_only`

| subcategory | count | reverse_context | coverage | unused ratio | max evidence |
|---|---:|---:|---:|---:|---:|
| entity_swap | 10 | 0.9353 | 1.0000 | 0.0000 | 0.9912 |
| grounded | 18 | 0.9392 | 1.0000 | 0.0000 | 0.9913 |
| negation_flip | 3 | 0.9245 | 1.0000 | 0.0000 | 0.9868 |
| parametric | 2 | 0.9273 | 1.0000 | 0.0000 | 0.9999 |
| partial | 2 | 0.9385 | 1.0000 | 0.0000 | 0.9999 |
| phantom_api | 7 | 0.9304 | 1.0000 | 0.0000 | 0.9927 |

### `colgrep` x `fuse_mean`

| subcategory | count | reverse_context | coverage | unused ratio | max evidence |
|---|---:|---:|---:|---:|---:|
| entity_swap | 10 | 0.9547 | 1.0000 | 0.0000 | 0.9940 |
| grounded | 18 | 0.9564 | 1.0000 | 0.0000 | 0.9945 |
| negation_flip | 3 | 0.9468 | 1.0000 | 0.0000 | 0.9924 |
| parametric | 2 | 0.9531 | 1.0000 | 0.0000 | 0.9992 |
| partial | 2 | 0.9589 | 1.0000 | 0.0000 | 0.9990 |
| phantom_api | 7 | 0.9519 | 1.0000 | 0.0000 | 0.9960 |

### `colgrep` x `fuse_max`

| subcategory | count | reverse_context | coverage | unused ratio | max evidence |
|---|---:|---:|---:|---:|---:|
| entity_swap | 10 | 0.9744 | 1.0000 | 0.0000 | 0.9977 |
| grounded | 18 | 0.9737 | 1.0000 | 0.0000 | 0.9982 |
| negation_flip | 3 | 0.9691 | 1.0000 | 0.0000 | 0.9986 |
| parametric | 2 | 0.9789 | 1.0000 | 0.0000 | 0.9999 |
| partial | 2 | 0.9793 | 1.0000 | 0.0000 | 0.9999 |
| phantom_api | 7 | 0.9734 | 1.0000 | 0.0000 | 0.9995 |

### `colgrep` x `fuse_min`

| subcategory | count | reverse_context | coverage | unused ratio | max evidence |
|---|---:|---:|---:|---:|---:|
| entity_swap | 10 | 0.9351 | 1.0000 | 0.0000 | 0.9904 |
| grounded | 18 | 0.9390 | 1.0000 | 0.0000 | 0.9907 |
| negation_flip | 3 | 0.9245 | 1.0000 | 0.0000 | 0.9861 |
| parametric | 2 | 0.9273 | 1.0000 | 0.0000 | 0.9986 |
| partial | 2 | 0.9385 | 1.0000 | 0.0000 | 0.9981 |
| phantom_api | 7 | 0.9304 | 1.0000 | 0.0000 | 0.9924 |

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
| SWG1 | sentence_packed | `gte_only` | grounded | - | 0.9617 | 1.0000 | 0.0000 | False | 0.9962 | 1 |
| SWG1 | sentence_packed | `code_only` | grounded | - | 0.9139 | 1.0000 | 0.0000 | False | 0.9765 | 1 |
| SWG1 | sentence_packed | `fuse_mean` | grounded | - | 0.9378 | 1.0000 | 0.0000 | False | 0.9864 | 1 |
| SWG1 | sentence_packed | `fuse_max` | grounded | - | 0.9617 | 1.0000 | 0.0000 | False | 0.9962 | 1 |
| SWG1 | sentence_packed | `fuse_min` | grounded | - | 0.9139 | 1.0000 | 0.0000 | False | 0.9765 | 1 |
| SWU1 | sentence_packed | `gte_only` | ungrounded | entity_swap | 0.9651 | 1.0000 | 0.0000 | False | 0.9977 | 1 |
| SWU1 | sentence_packed | `code_only` | ungrounded | entity_swap | 0.9157 | 1.0000 | 0.0000 | False | 0.9703 | 1 |
| SWU1 | sentence_packed | `fuse_mean` | ungrounded | entity_swap | 0.9404 | 1.0000 | 0.0000 | False | 0.9840 | 1 |
| SWU1 | sentence_packed | `fuse_max` | ungrounded | entity_swap | 0.9651 | 1.0000 | 0.0000 | False | 0.9977 | 1 |
| SWU1 | sentence_packed | `fuse_min` | ungrounded | entity_swap | 0.9157 | 1.0000 | 0.0000 | False | 0.9703 | 1 |
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
| SWG4 | sentence_packed | `gte_only` | grounded | - | 0.9765 | 1.0000 | 0.0000 | False | 0.9941 | 1 |
| SWG4 | sentence_packed | `code_only` | grounded | - | 0.9496 | 1.0000 | 0.0000 | False | 0.9966 | 1 |
| SWG4 | sentence_packed | `fuse_mean` | grounded | - | 0.9630 | 1.0000 | 0.0000 | False | 0.9954 | 1 |
| SWG4 | sentence_packed | `fuse_max` | grounded | - | 0.9765 | 1.0000 | 0.0000 | False | 0.9966 | 1 |
| SWG4 | sentence_packed | `fuse_min` | grounded | - | 0.9496 | 1.0000 | 0.0000 | False | 0.9941 | 1 |
| SWU4 | sentence_packed | `gte_only` | ungrounded | entity_swap | 0.9763 | 1.0000 | 0.0000 | False | 0.9962 | 1 |
| SWU4 | sentence_packed | `code_only` | ungrounded | entity_swap | 0.9458 | 1.0000 | 0.0000 | False | 0.9960 | 1 |
| SWU4 | sentence_packed | `fuse_mean` | ungrounded | entity_swap | 0.9611 | 1.0000 | 0.0000 | False | 0.9961 | 1 |
| SWU4 | sentence_packed | `fuse_max` | ungrounded | entity_swap | 0.9763 | 1.0000 | 0.0000 | False | 0.9962 | 1 |
| SWU4 | sentence_packed | `fuse_min` | ungrounded | entity_swap | 0.9458 | 1.0000 | 0.0000 | False | 0.9960 | 1 |
| SWG5 | sentence_packed | `gte_only` | grounded | - | 0.9720 | 1.0000 | 0.0000 | False | 0.9976 | 1 |
| SWG5 | sentence_packed | `code_only` | grounded | - | 0.9434 | 1.0000 | 0.0000 | False | 0.9874 | 1 |
| SWG5 | sentence_packed | `fuse_mean` | grounded | - | 0.9577 | 1.0000 | 0.0000 | False | 0.9925 | 1 |
| SWG5 | sentence_packed | `fuse_max` | grounded | - | 0.9720 | 1.0000 | 0.0000 | False | 0.9976 | 1 |
| SWG5 | sentence_packed | `fuse_min` | grounded | - | 0.9434 | 1.0000 | 0.0000 | False | 0.9874 | 1 |
| SWU5 | sentence_packed | `gte_only` | ungrounded | entity_swap | 0.9728 | 1.0000 | 0.0000 | False | 0.9976 | 1 |
| SWU5 | sentence_packed | `code_only` | ungrounded | entity_swap | 0.9466 | 1.0000 | 0.0000 | False | 0.9862 | 1 |
| SWU5 | sentence_packed | `fuse_mean` | ungrounded | entity_swap | 0.9597 | 1.0000 | 0.0000 | False | 0.9919 | 1 |
| SWU5 | sentence_packed | `fuse_max` | ungrounded | entity_swap | 0.9728 | 1.0000 | 0.0000 | False | 0.9976 | 1 |
| SWU5 | sentence_packed | `fuse_min` | ungrounded | entity_swap | 0.9466 | 1.0000 | 0.0000 | False | 0.9862 | 1 |
| SWG6 | sentence_packed | `gte_only` | grounded | - | 0.9687 | 1.0000 | 0.0000 | False | 0.9945 | 1 |
| SWG6 | sentence_packed | `code_only` | grounded | - | 0.9274 | 1.0000 | 0.0000 | False | 0.9662 | 1 |
| SWG6 | sentence_packed | `fuse_mean` | grounded | - | 0.9480 | 1.0000 | 0.0000 | False | 0.9803 | 1 |
| SWG6 | sentence_packed | `fuse_max` | grounded | - | 0.9687 | 1.0000 | 0.0000 | False | 0.9945 | 1 |
| SWG6 | sentence_packed | `fuse_min` | grounded | - | 0.9274 | 1.0000 | 0.0000 | False | 0.9662 | 1 |
| SWU6 | sentence_packed | `gte_only` | ungrounded | entity_swap | 0.9687 | 1.0000 | 0.0000 | False | 0.9944 | 1 |
| SWU6 | sentence_packed | `code_only` | ungrounded | entity_swap | 0.9256 | 1.0000 | 0.0000 | False | 0.9655 | 1 |
| SWU6 | sentence_packed | `fuse_mean` | ungrounded | entity_swap | 0.9472 | 1.0000 | 0.0000 | False | 0.9800 | 1 |
| SWU6 | sentence_packed | `fuse_max` | ungrounded | entity_swap | 0.9687 | 1.0000 | 0.0000 | False | 0.9944 | 1 |
| SWU6 | sentence_packed | `fuse_min` | ungrounded | entity_swap | 0.9256 | 1.0000 | 0.0000 | False | 0.9655 | 1 |
| SWG7 | sentence_packed | `gte_only` | grounded | - | 0.9790 | 1.0000 | 0.0000 | False | 0.9996 | 2 |
| SWG7 | sentence_packed | `code_only` | grounded | - | 0.9541 | 1.0000 | 0.0000 | False | 0.9882 | 2 |
| SWG7 | sentence_packed | `fuse_mean` | grounded | - | 0.9666 | 1.0000 | 0.0000 | False | 0.9939 | 2 |
| SWG7 | sentence_packed | `fuse_max` | grounded | - | 0.9790 | 1.0000 | 0.0000 | False | 0.9996 | 2 |
| SWG7 | sentence_packed | `fuse_min` | grounded | - | 0.9541 | 1.0000 | 0.0000 | False | 0.9882 | 2 |
| SWU7 | sentence_packed | `gte_only` | ungrounded | phantom_api | 0.9752 | 1.0000 | 0.0000 | False | 0.9992 | 2 |
| SWU7 | sentence_packed | `code_only` | ungrounded | phantom_api | 0.9497 | 1.0000 | 0.0000 | False | 0.9933 | 2 |
| SWU7 | sentence_packed | `fuse_mean` | ungrounded | phantom_api | 0.9624 | 1.0000 | 0.0000 | False | 0.9962 | 2 |
| SWU7 | sentence_packed | `fuse_max` | ungrounded | phantom_api | 0.9752 | 1.0000 | 0.0000 | False | 0.9992 | 2 |
| SWU7 | sentence_packed | `fuse_min` | ungrounded | phantom_api | 0.9497 | 1.0000 | 0.0000 | False | 0.9933 | 2 |
| SWG8 | sentence_packed | `gte_only` | grounded | - | 0.9616 | 1.0000 | 0.0000 | False | 0.9961 | 3 |
| SWG8 | sentence_packed | `code_only` | grounded | - | 0.9676 | 1.0000 | 0.0000 | False | 0.9991 | 3 |
| SWG8 | sentence_packed | `fuse_mean` | grounded | - | 0.9646 | 1.0000 | 0.0000 | False | 0.9976 | 3 |
| SWG8 | sentence_packed | `fuse_max` | grounded | - | 0.9676 | 1.0000 | 0.0000 | False | 0.9991 | 3 |
| SWG8 | sentence_packed | `fuse_min` | grounded | - | 0.9616 | 1.0000 | 0.0000 | False | 0.9961 | 3 |
| SWU8 | sentence_packed | `gte_only` | ungrounded | entity_swap | 0.9610 | 1.0000 | 0.0000 | False | 0.9963 | 3 |
| SWU8 | sentence_packed | `code_only` | ungrounded | entity_swap | 0.9670 | 1.0000 | 0.0000 | False | 0.9990 | 3 |
| SWU8 | sentence_packed | `fuse_mean` | ungrounded | entity_swap | 0.9640 | 1.0000 | 0.0000 | False | 0.9976 | 3 |
| SWU8 | sentence_packed | `fuse_max` | ungrounded | entity_swap | 0.9670 | 1.0000 | 0.0000 | False | 0.9990 | 3 |
| SWU8 | sentence_packed | `fuse_min` | ungrounded | entity_swap | 0.9610 | 1.0000 | 0.0000 | False | 0.9963 | 3 |
| SWG9 | sentence_packed | `gte_only` | grounded | - | 0.9650 | 1.0000 | 0.0000 | False | 0.9964 | 3 |
| SWG9 | sentence_packed | `code_only` | grounded | - | 0.9499 | 1.0000 | 0.0000 | False | 0.9933 | 3 |
| SWG9 | sentence_packed | `fuse_mean` | grounded | - | 0.9575 | 1.0000 | 0.0000 | False | 0.9949 | 3 |
| SWG9 | sentence_packed | `fuse_max` | grounded | - | 0.9650 | 1.0000 | 0.0000 | False | 0.9964 | 3 |
| SWG9 | sentence_packed | `fuse_min` | grounded | - | 0.9499 | 1.0000 | 0.0000 | False | 0.9933 | 3 |
| SWU9 | sentence_packed | `gte_only` | ungrounded | phantom_api | 0.9642 | 1.0000 | 0.0000 | False | 0.9999 | 3 |
| SWU9 | sentence_packed | `code_only` | ungrounded | phantom_api | 0.9483 | 1.0000 | 0.0000 | False | 0.9894 | 3 |
| SWU9 | sentence_packed | `fuse_mean` | ungrounded | phantom_api | 0.9562 | 1.0000 | 0.0000 | False | 0.9947 | 3 |
| SWU9 | sentence_packed | `fuse_max` | ungrounded | phantom_api | 0.9642 | 1.0000 | 0.0000 | False | 0.9999 | 3 |
| SWU9 | sentence_packed | `fuse_min` | ungrounded | phantom_api | 0.9483 | 1.0000 | 0.0000 | False | 0.9894 | 3 |
| SWG10 | sentence_packed | `gte_only` | grounded | - | 0.9643 | 1.0000 | 0.0000 | False | 0.9979 | 2 |
| SWG10 | sentence_packed | `code_only` | grounded | - | 0.9691 | 1.0000 | 0.0000 | False | 0.9977 | 2 |
| SWG10 | sentence_packed | `fuse_mean` | grounded | - | 0.9667 | 1.0000 | 0.0000 | False | 0.9978 | 2 |
| SWG10 | sentence_packed | `fuse_max` | grounded | - | 0.9691 | 1.0000 | 0.0000 | False | 0.9979 | 2 |
| SWG10 | sentence_packed | `fuse_min` | grounded | - | 0.9643 | 1.0000 | 0.0000 | False | 0.9977 | 2 |
| SWU10 | sentence_packed | `gte_only` | ungrounded | phantom_api | 0.9643 | 1.0000 | 0.0000 | False | 0.9979 | 2 |
| SWU10 | sentence_packed | `code_only` | ungrounded | phantom_api | 0.9679 | 1.0000 | 0.0000 | False | 0.9978 | 2 |
| SWU10 | sentence_packed | `fuse_mean` | ungrounded | phantom_api | 0.9661 | 1.0000 | 0.0000 | False | 0.9978 | 2 |
| SWU10 | sentence_packed | `fuse_max` | ungrounded | phantom_api | 0.9679 | 1.0000 | 0.0000 | False | 0.9979 | 2 |
| SWU10 | sentence_packed | `fuse_min` | ungrounded | phantom_api | 0.9643 | 1.0000 | 0.0000 | False | 0.9978 | 2 |
| SWG11 | sentence_packed | `gte_only` | grounded | - | 0.9788 | 1.0000 | 0.0000 | False | 0.9966 | 5 |
| SWG11 | sentence_packed | `code_only` | grounded | - | 0.9571 | 1.0000 | 0.0000 | False | 0.9927 | 5 |
| SWG11 | sentence_packed | `fuse_mean` | grounded | - | 0.9680 | 1.0000 | 0.0000 | False | 0.9947 | 5 |
| SWG11 | sentence_packed | `fuse_max` | grounded | - | 0.9788 | 1.0000 | 0.0000 | False | 0.9966 | 5 |
| SWG11 | sentence_packed | `fuse_min` | grounded | - | 0.9571 | 1.0000 | 0.0000 | False | 0.9927 | 5 |
| SWU11 | sentence_packed | `gte_only` | ungrounded | phantom_api | 0.9785 | 1.0000 | 0.0000 | False | 0.9967 | 5 |
| SWU11 | sentence_packed | `code_only` | ungrounded | phantom_api | 0.9567 | 1.0000 | 0.0000 | False | 0.9917 | 5 |
| SWU11 | sentence_packed | `fuse_mean` | ungrounded | phantom_api | 0.9676 | 1.0000 | 0.0000 | False | 0.9942 | 5 |
| SWU11 | sentence_packed | `fuse_max` | ungrounded | phantom_api | 0.9785 | 1.0000 | 0.0000 | False | 0.9967 | 5 |
| SWU11 | sentence_packed | `fuse_min` | ungrounded | phantom_api | 0.9567 | 1.0000 | 0.0000 | False | 0.9917 | 5 |
| SWG12 | sentence_packed | `gte_only` | grounded | - | 0.9542 | 1.0000 | 0.0000 | False | 0.9952 | 1 |
| SWG12 | sentence_packed | `code_only` | grounded | - | 0.9406 | 1.0000 | 0.0000 | False | 1.0000 | 1 |
| SWG12 | sentence_packed | `fuse_mean` | grounded | - | 0.9474 | 1.0000 | 0.0000 | False | 0.9976 | 1 |
| SWG12 | sentence_packed | `fuse_max` | grounded | - | 0.9542 | 1.0000 | 0.0000 | False | 1.0000 | 1 |
| SWG12 | sentence_packed | `fuse_min` | grounded | - | 0.9406 | 1.0000 | 0.0000 | False | 0.9952 | 1 |
| SWU12 | sentence_packed | `gte_only` | ungrounded | negation_flip | 0.9538 | 1.0000 | 0.0000 | False | 0.9987 | 1 |
| SWU12 | sentence_packed | `code_only` | ungrounded | negation_flip | 0.9395 | 1.0000 | 0.0000 | False | 0.9957 | 1 |
| SWU12 | sentence_packed | `fuse_mean` | ungrounded | negation_flip | 0.9466 | 1.0000 | 0.0000 | False | 0.9972 | 1 |
| SWU12 | sentence_packed | `fuse_max` | ungrounded | negation_flip | 0.9538 | 1.0000 | 0.0000 | False | 0.9987 | 1 |
| SWU12 | sentence_packed | `fuse_min` | ungrounded | negation_flip | 0.9395 | 1.0000 | 0.0000 | False | 0.9957 | 1 |
| SWG13 | sentence_packed | `gte_only` | grounded | - | 0.9814 | 1.0000 | 0.0000 | False | 0.9999 | 4 |
| SWG13 | sentence_packed | `code_only` | grounded | - | 0.9491 | 1.0000 | 0.0000 | False | 0.9936 | 4 |
| SWG13 | sentence_packed | `fuse_mean` | grounded | - | 0.9653 | 1.0000 | 0.0000 | False | 0.9968 | 4 |
| SWG13 | sentence_packed | `fuse_max` | grounded | - | 0.9814 | 1.0000 | 0.0000 | False | 0.9999 | 4 |
| SWG13 | sentence_packed | `fuse_min` | grounded | - | 0.9491 | 1.0000 | 0.0000 | False | 0.9936 | 4 |
| SWU13 | sentence_packed | `gte_only` | ungrounded | entity_swap | 0.9822 | 1.0000 | 0.0000 | False | 1.0000 | 4 |
| SWU13 | sentence_packed | `code_only` | ungrounded | entity_swap | 0.9533 | 1.0000 | 0.0000 | False | 0.9929 | 4 |
| SWU13 | sentence_packed | `fuse_mean` | ungrounded | entity_swap | 0.9678 | 1.0000 | 0.0000 | False | 0.9964 | 4 |
| SWU13 | sentence_packed | `fuse_max` | ungrounded | entity_swap | 0.9822 | 1.0000 | 0.0000 | False | 1.0000 | 4 |
| SWU13 | sentence_packed | `fuse_min` | ungrounded | entity_swap | 0.9533 | 1.0000 | 0.0000 | False | 0.9929 | 4 |
| SWG14 | sentence_packed | `gte_only` | grounded | - | 0.9771 | 1.0000 | 0.0000 | False | 0.9968 | 2 |
| SWG14 | sentence_packed | `code_only` | grounded | - | 0.9432 | 1.0000 | 0.0000 | False | 0.9861 | 2 |
| SWG14 | sentence_packed | `fuse_mean` | grounded | - | 0.9602 | 1.0000 | 0.0000 | False | 0.9914 | 2 |
| SWG14 | sentence_packed | `fuse_max` | grounded | - | 0.9771 | 1.0000 | 0.0000 | False | 0.9968 | 2 |
| SWG14 | sentence_packed | `fuse_min` | grounded | - | 0.9432 | 1.0000 | 0.0000 | False | 0.9861 | 2 |
| SWU14 | sentence_packed | `gte_only` | ungrounded | entity_swap | 0.9773 | 1.0000 | 0.0000 | False | 0.9968 | 2 |
| SWU14 | sentence_packed | `code_only` | ungrounded | entity_swap | 0.9429 | 1.0000 | 0.0000 | False | 0.9861 | 2 |
| SWU14 | sentence_packed | `fuse_mean` | ungrounded | entity_swap | 0.9601 | 1.0000 | 0.0000 | False | 0.9914 | 2 |
| SWU14 | sentence_packed | `fuse_max` | ungrounded | entity_swap | 0.9773 | 1.0000 | 0.0000 | False | 0.9968 | 2 |
| SWU14 | sentence_packed | `fuse_min` | ungrounded | entity_swap | 0.9429 | 1.0000 | 0.0000 | False | 0.9861 | 2 |
| SWG15 | sentence_packed | `gte_only` | grounded | - | 0.9731 | 1.0000 | 0.0000 | False | 0.9975 | 3 |
| SWG15 | sentence_packed | `code_only` | grounded | - | 0.9588 | 1.0000 | 0.0000 | False | 0.9915 | 3 |
| SWG15 | sentence_packed | `fuse_mean` | grounded | - | 0.9660 | 1.0000 | 0.0000 | False | 0.9945 | 3 |
| SWG15 | sentence_packed | `fuse_max` | grounded | - | 0.9731 | 1.0000 | 0.0000 | False | 0.9975 | 3 |
| SWG15 | sentence_packed | `fuse_min` | grounded | - | 0.9588 | 1.0000 | 0.0000 | False | 0.9915 | 3 |
| SWU15 | sentence_packed | `gte_only` | ungrounded | entity_swap | 0.9718 | 1.0000 | 0.0000 | False | 0.9912 | 3 |
| SWU15 | sentence_packed | `code_only` | ungrounded | entity_swap | 0.9526 | 1.0000 | 0.0000 | False | 0.9888 | 3 |
| SWU15 | sentence_packed | `fuse_mean` | ungrounded | entity_swap | 0.9622 | 1.0000 | 0.0000 | False | 0.9900 | 3 |
| SWU15 | sentence_packed | `fuse_max` | ungrounded | entity_swap | 0.9718 | 1.0000 | 0.0000 | False | 0.9912 | 3 |
| SWU15 | sentence_packed | `fuse_min` | ungrounded | entity_swap | 0.9526 | 1.0000 | 0.0000 | False | 0.9888 | 3 |
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
| SWG1 | colgrep | `gte_only` | grounded | - | 0.9471 | 1.0000 | 0.0000 | False | 0.9955 | 1 |
| SWG1 | colgrep | `code_only` | grounded | - | 0.8752 | 1.0000 | 0.0000 | False | 0.9777 | 1 |
| SWG1 | colgrep | `fuse_mean` | grounded | - | 0.9111 | 1.0000 | 0.0000 | False | 0.9866 | 1 |
| SWG1 | colgrep | `fuse_max` | grounded | - | 0.9471 | 1.0000 | 0.0000 | False | 0.9955 | 1 |
| SWG1 | colgrep | `fuse_min` | grounded | - | 0.8752 | 1.0000 | 0.0000 | False | 0.9777 | 1 |
| SWU1 | colgrep | `gte_only` | ungrounded | entity_swap | 0.9496 | 1.0000 | 0.0000 | False | 0.9954 | 1 |
| SWU1 | colgrep | `code_only` | ungrounded | entity_swap | 0.8759 | 1.0000 | 0.0000 | False | 0.9800 | 1 |
| SWU1 | colgrep | `fuse_mean` | ungrounded | entity_swap | 0.9127 | 1.0000 | 0.0000 | False | 0.9877 | 1 |
| SWU1 | colgrep | `fuse_max` | ungrounded | entity_swap | 0.9496 | 1.0000 | 0.0000 | False | 0.9954 | 1 |
| SWU1 | colgrep | `fuse_min` | ungrounded | entity_swap | 0.8759 | 1.0000 | 0.0000 | False | 0.9800 | 1 |
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
| SWG3 | colgrep | `gte_only` | grounded | - | 0.9765 | 1.0000 | 0.0000 | False | 0.9973 | 2 |
| SWG3 | colgrep | `code_only` | grounded | - | 0.9387 | 1.0000 | 0.0000 | False | 0.9860 | 2 |
| SWG3 | colgrep | `fuse_mean` | grounded | - | 0.9576 | 1.0000 | 0.0000 | False | 0.9917 | 2 |
| SWG3 | colgrep | `fuse_max` | grounded | - | 0.9765 | 1.0000 | 0.0000 | False | 0.9973 | 2 |
| SWG3 | colgrep | `fuse_min` | grounded | - | 0.9387 | 1.0000 | 0.0000 | False | 0.9860 | 2 |
| SWU3 | colgrep | `gte_only` | ungrounded | negation_flip | 0.9783 | 1.0000 | 0.0000 | False | 0.9973 | 2 |
| SWU3 | colgrep | `code_only` | ungrounded | negation_flip | 0.9329 | 1.0000 | 0.0000 | False | 0.9648 | 2 |
| SWU3 | colgrep | `fuse_mean` | ungrounded | negation_flip | 0.9556 | 1.0000 | 0.0000 | False | 0.9810 | 2 |
| SWU3 | colgrep | `fuse_max` | ungrounded | negation_flip | 0.9783 | 1.0000 | 0.0000 | False | 0.9973 | 2 |
| SWU3 | colgrep | `fuse_min` | ungrounded | negation_flip | 0.9329 | 1.0000 | 0.0000 | False | 0.9648 | 2 |
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
| SWG5 | colgrep | `gte_only` | grounded | - | 0.9746 | 1.0000 | 0.0000 | False | 0.9973 | 2 |
| SWG5 | colgrep | `code_only` | grounded | - | 0.9420 | 1.0000 | 0.0000 | False | 0.9967 | 2 |
| SWG5 | colgrep | `fuse_mean` | grounded | - | 0.9583 | 1.0000 | 0.0000 | False | 0.9970 | 2 |
| SWG5 | colgrep | `fuse_max` | grounded | - | 0.9746 | 1.0000 | 0.0000 | False | 0.9973 | 2 |
| SWG5 | colgrep | `fuse_min` | grounded | - | 0.9420 | 1.0000 | 0.0000 | False | 0.9967 | 2 |
| SWU5 | colgrep | `gte_only` | ungrounded | entity_swap | 0.9751 | 1.0000 | 0.0000 | False | 0.9973 | 2 |
| SWU5 | colgrep | `code_only` | ungrounded | entity_swap | 0.9469 | 1.0000 | 0.0000 | False | 0.9963 | 2 |
| SWU5 | colgrep | `fuse_mean` | ungrounded | entity_swap | 0.9610 | 1.0000 | 0.0000 | False | 0.9968 | 2 |
| SWU5 | colgrep | `fuse_max` | ungrounded | entity_swap | 0.9751 | 1.0000 | 0.0000 | False | 0.9973 | 2 |
| SWU5 | colgrep | `fuse_min` | ungrounded | entity_swap | 0.9469 | 1.0000 | 0.0000 | False | 0.9963 | 2 |
| SWG6 | colgrep | `gte_only` | grounded | - | 0.9539 | 1.0000 | 0.0000 | False | 0.9954 | 1 |
| SWG6 | colgrep | `code_only` | grounded | - | 0.8940 | 1.0000 | 0.0000 | False | 0.9658 | 1 |
| SWG6 | colgrep | `fuse_mean` | grounded | - | 0.9239 | 1.0000 | 0.0000 | False | 0.9806 | 1 |
| SWG6 | colgrep | `fuse_max` | grounded | - | 0.9539 | 1.0000 | 0.0000 | False | 0.9954 | 1 |
| SWG6 | colgrep | `fuse_min` | grounded | - | 0.8940 | 1.0000 | 0.0000 | False | 0.9658 | 1 |
| SWU6 | colgrep | `gte_only` | ungrounded | entity_swap | 0.9534 | 1.0000 | 0.0000 | False | 0.9953 | 1 |
| SWU6 | colgrep | `code_only` | ungrounded | entity_swap | 0.8942 | 1.0000 | 0.0000 | False | 0.9651 | 1 |
| SWU6 | colgrep | `fuse_mean` | ungrounded | entity_swap | 0.9238 | 1.0000 | 0.0000 | False | 0.9802 | 1 |
| SWU6 | colgrep | `fuse_max` | ungrounded | entity_swap | 0.9534 | 1.0000 | 0.0000 | False | 0.9953 | 1 |
| SWU6 | colgrep | `fuse_min` | ungrounded | entity_swap | 0.8942 | 1.0000 | 0.0000 | False | 0.9651 | 1 |
| SWG7 | colgrep | `gte_only` | grounded | - | 0.9773 | 1.0000 | 0.0000 | False | 0.9995 | 2 |
| SWG7 | colgrep | `code_only` | grounded | - | 0.9520 | 1.0000 | 0.0000 | False | 0.9877 | 2 |
| SWG7 | colgrep | `fuse_mean` | grounded | - | 0.9646 | 1.0000 | 0.0000 | False | 0.9936 | 2 |
| SWG7 | colgrep | `fuse_max` | grounded | - | 0.9773 | 1.0000 | 0.0000 | False | 0.9995 | 2 |
| SWG7 | colgrep | `fuse_min` | grounded | - | 0.9520 | 1.0000 | 0.0000 | False | 0.9877 | 2 |
| SWU7 | colgrep | `gte_only` | ungrounded | phantom_api | 0.9748 | 1.0000 | 0.0000 | False | 0.9998 | 2 |
| SWU7 | colgrep | `code_only` | ungrounded | phantom_api | 0.9486 | 1.0000 | 0.0000 | False | 0.9945 | 2 |
| SWU7 | colgrep | `fuse_mean` | ungrounded | phantom_api | 0.9617 | 1.0000 | 0.0000 | False | 0.9971 | 2 |
| SWU7 | colgrep | `fuse_max` | ungrounded | phantom_api | 0.9748 | 1.0000 | 0.0000 | False | 0.9998 | 2 |
| SWU7 | colgrep | `fuse_min` | ungrounded | phantom_api | 0.9486 | 1.0000 | 0.0000 | False | 0.9945 | 2 |
| SWG8 | colgrep | `gte_only` | grounded | - | 0.9640 | 1.0000 | 0.0000 | False | 0.9961 | 11 |
| SWG8 | colgrep | `code_only` | grounded | - | 0.9669 | 1.0000 | 0.0000 | False | 0.9982 | 11 |
| SWG8 | colgrep | `fuse_mean` | grounded | - | 0.9654 | 1.0000 | 0.0000 | False | 0.9971 | 11 |
| SWG8 | colgrep | `fuse_max` | grounded | - | 0.9669 | 1.0000 | 0.0000 | False | 0.9982 | 11 |
| SWG8 | colgrep | `fuse_min` | grounded | - | 0.9640 | 1.0000 | 0.0000 | False | 0.9961 | 11 |
| SWU8 | colgrep | `gte_only` | ungrounded | entity_swap | 0.9636 | 1.0000 | 0.0000 | False | 0.9963 | 11 |
| SWU8 | colgrep | `code_only` | ungrounded | entity_swap | 0.9663 | 1.0000 | 0.0000 | False | 0.9983 | 11 |
| SWU8 | colgrep | `fuse_mean` | ungrounded | entity_swap | 0.9649 | 1.0000 | 0.0000 | False | 0.9973 | 11 |
| SWU8 | colgrep | `fuse_max` | ungrounded | entity_swap | 0.9663 | 1.0000 | 0.0000 | False | 0.9983 | 11 |
| SWU8 | colgrep | `fuse_min` | ungrounded | entity_swap | 0.9636 | 1.0000 | 0.0000 | False | 0.9963 | 11 |
| SWG9 | colgrep | `gte_only` | grounded | - | 0.9647 | 1.0000 | 0.0000 | False | 0.9958 | 10 |
| SWG9 | colgrep | `code_only` | grounded | - | 0.9515 | 1.0000 | 0.0000 | False | 0.9900 | 10 |
| SWG9 | colgrep | `fuse_mean` | grounded | - | 0.9581 | 1.0000 | 0.0000 | False | 0.9929 | 10 |
| SWG9 | colgrep | `fuse_max` | grounded | - | 0.9647 | 1.0000 | 0.0000 | False | 0.9958 | 10 |
| SWG9 | colgrep | `fuse_min` | grounded | - | 0.9515 | 1.0000 | 0.0000 | False | 0.9900 | 10 |
| SWU9 | colgrep | `gte_only` | ungrounded | phantom_api | 0.9643 | 1.0000 | 0.0000 | False | 0.9998 | 10 |
| SWU9 | colgrep | `code_only` | ungrounded | phantom_api | 0.9461 | 1.0000 | 0.0000 | False | 0.9915 | 10 |
| SWU9 | colgrep | `fuse_mean` | ungrounded | phantom_api | 0.9552 | 1.0000 | 0.0000 | False | 0.9957 | 10 |
| SWU9 | colgrep | `fuse_max` | ungrounded | phantom_api | 0.9643 | 1.0000 | 0.0000 | False | 0.9998 | 10 |
| SWU9 | colgrep | `fuse_min` | ungrounded | phantom_api | 0.9461 | 1.0000 | 0.0000 | False | 0.9915 | 10 |
| SWG10 | colgrep | `gte_only` | grounded | - | 0.9760 | 1.0000 | 0.0000 | False | 0.9999 | 5 |
| SWG10 | colgrep | `code_only` | grounded | - | 0.9699 | 1.0000 | 0.0000 | False | 0.9936 | 5 |
| SWG10 | colgrep | `fuse_mean` | grounded | - | 0.9729 | 1.0000 | 0.0000 | False | 0.9968 | 5 |
| SWG10 | colgrep | `fuse_max` | grounded | - | 0.9760 | 1.0000 | 0.0000 | False | 0.9999 | 5 |
| SWG10 | colgrep | `fuse_min` | grounded | - | 0.9699 | 1.0000 | 0.0000 | False | 0.9936 | 5 |
| SWU10 | colgrep | `gte_only` | ungrounded | phantom_api | 0.9760 | 1.0000 | 0.0000 | False | 0.9999 | 5 |
| SWU10 | colgrep | `code_only` | ungrounded | phantom_api | 0.9690 | 1.0000 | 0.0000 | False | 0.9937 | 5 |
| SWU10 | colgrep | `fuse_mean` | ungrounded | phantom_api | 0.9725 | 1.0000 | 0.0000 | False | 0.9968 | 5 |
| SWU10 | colgrep | `fuse_max` | ungrounded | phantom_api | 0.9760 | 1.0000 | 0.0000 | False | 0.9999 | 5 |
| SWU10 | colgrep | `fuse_min` | ungrounded | phantom_api | 0.9690 | 1.0000 | 0.0000 | False | 0.9937 | 5 |
| SWG11 | colgrep | `gte_only` | grounded | - | 0.9857 | 1.0000 | 0.0000 | False | 0.9976 | 20 |
| SWG11 | colgrep | `code_only` | grounded | - | 0.9593 | 1.0000 | 0.0000 | False | 0.9965 | 20 |
| SWG11 | colgrep | `fuse_mean` | grounded | - | 0.9725 | 1.0000 | 0.0000 | False | 0.9971 | 20 |
| SWG11 | colgrep | `fuse_max` | grounded | - | 0.9857 | 1.0000 | 0.0000 | False | 0.9976 | 20 |
| SWG11 | colgrep | `fuse_min` | grounded | - | 0.9593 | 1.0000 | 0.0000 | False | 0.9965 | 20 |
| SWU11 | colgrep | `gte_only` | ungrounded | phantom_api | 0.9852 | 1.0000 | 0.0000 | False | 0.9977 | 20 |
| SWU11 | colgrep | `code_only` | ungrounded | phantom_api | 0.9587 | 1.0000 | 0.0000 | False | 0.9961 | 20 |
| SWU11 | colgrep | `fuse_mean` | ungrounded | phantom_api | 0.9720 | 1.0000 | 0.0000 | False | 0.9969 | 20 |
| SWU11 | colgrep | `fuse_max` | ungrounded | phantom_api | 0.9852 | 1.0000 | 0.0000 | False | 0.9977 | 20 |
| SWU11 | colgrep | `fuse_min` | ungrounded | phantom_api | 0.9587 | 1.0000 | 0.0000 | False | 0.9961 | 20 |
| SWG12 | colgrep | `gte_only` | grounded | - | 0.9494 | 1.0000 | 0.0000 | False | 0.9946 | 1 |
| SWG12 | colgrep | `code_only` | grounded | - | 0.9195 | 1.0000 | 0.0000 | False | 1.0000 | 1 |
| SWG12 | colgrep | `fuse_mean` | grounded | - | 0.9344 | 1.0000 | 0.0000 | False | 0.9973 | 1 |
| SWG12 | colgrep | `fuse_max` | grounded | - | 0.9494 | 1.0000 | 0.0000 | False | 1.0000 | 1 |
| SWG12 | colgrep | `fuse_min` | grounded | - | 0.9195 | 1.0000 | 0.0000 | False | 0.9946 | 1 |
| SWU12 | colgrep | `gte_only` | ungrounded | negation_flip | 0.9484 | 1.0000 | 0.0000 | False | 0.9988 | 1 |
| SWU12 | colgrep | `code_only` | ungrounded | negation_flip | 0.9190 | 1.0000 | 0.0000 | False | 0.9957 | 1 |
| SWU12 | colgrep | `fuse_mean` | ungrounded | negation_flip | 0.9337 | 1.0000 | 0.0000 | False | 0.9973 | 1 |
| SWU12 | colgrep | `fuse_max` | ungrounded | negation_flip | 0.9484 | 1.0000 | 0.0000 | False | 0.9988 | 1 |
| SWU12 | colgrep | `fuse_min` | ungrounded | negation_flip | 0.9190 | 1.0000 | 0.0000 | False | 0.9957 | 1 |
| SWG13 | colgrep | `gte_only` | grounded | - | 0.9812 | 1.0000 | 0.0000 | False | 1.0000 | 34 |
| SWG13 | colgrep | `code_only` | grounded | - | 0.9490 | 1.0000 | 0.0000 | False | 0.9931 | 34 |
| SWG13 | colgrep | `fuse_mean` | grounded | - | 0.9651 | 1.0000 | 0.0000 | False | 0.9965 | 34 |
| SWG13 | colgrep | `fuse_max` | grounded | - | 0.9812 | 1.0000 | 0.0000 | False | 1.0000 | 34 |
| SWG13 | colgrep | `fuse_min` | grounded | - | 0.9490 | 1.0000 | 0.0000 | False | 0.9931 | 34 |
| SWU13 | colgrep | `gte_only` | ungrounded | entity_swap | 0.9816 | 1.0000 | 0.0000 | False | 0.9995 | 34 |
| SWU13 | colgrep | `code_only` | ungrounded | entity_swap | 0.9543 | 1.0000 | 0.0000 | False | 0.9912 | 34 |
| SWU13 | colgrep | `fuse_mean` | ungrounded | entity_swap | 0.9679 | 1.0000 | 0.0000 | False | 0.9953 | 34 |
| SWU13 | colgrep | `fuse_max` | ungrounded | entity_swap | 0.9816 | 1.0000 | 0.0000 | False | 0.9995 | 34 |
| SWU13 | colgrep | `fuse_min` | ungrounded | entity_swap | 0.9543 | 1.0000 | 0.0000 | False | 0.9912 | 34 |
| SWG14 | colgrep | `gte_only` | grounded | - | 0.9852 | 1.0000 | 0.0000 | False | 0.9975 | 7 |
| SWG14 | colgrep | `code_only` | grounded | - | 0.9524 | 1.0000 | 0.0000 | False | 0.9912 | 7 |
| SWG14 | colgrep | `fuse_mean` | grounded | - | 0.9688 | 1.0000 | 0.0000 | False | 0.9943 | 7 |
| SWG14 | colgrep | `fuse_max` | grounded | - | 0.9852 | 1.0000 | 0.0000 | False | 0.9975 | 7 |
| SWG14 | colgrep | `fuse_min` | grounded | - | 0.9524 | 1.0000 | 0.0000 | False | 0.9912 | 7 |
| SWU14 | colgrep | `gte_only` | ungrounded | entity_swap | 0.9852 | 1.0000 | 0.0000 | False | 0.9975 | 7 |
| SWU14 | colgrep | `code_only` | ungrounded | entity_swap | 0.9520 | 1.0000 | 0.0000 | False | 0.9912 | 7 |
| SWU14 | colgrep | `fuse_mean` | ungrounded | entity_swap | 0.9686 | 1.0000 | 0.0000 | False | 0.9944 | 7 |
| SWU14 | colgrep | `fuse_max` | ungrounded | entity_swap | 0.9852 | 1.0000 | 0.0000 | False | 0.9975 | 7 |
| SWU14 | colgrep | `fuse_min` | ungrounded | entity_swap | 0.9520 | 1.0000 | 0.0000 | False | 0.9912 | 7 |
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
