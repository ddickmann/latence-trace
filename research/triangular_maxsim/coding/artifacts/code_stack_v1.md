# Coding-Agent Groundedness: Scorer Stack x Chunker

> NO — the flagship colgrep x fuse_mean cell did not clear the RAG-style gate: reverse_context AUROC=0.603, phantom-API precision@0.35=0.00, ungrounded minus grounded unused-ratio delta=+0.0000. Stacking lift vs gte_only=-0.0212. Best fusion rule on colgrep is `fuse_max` (AUROC=0.624); consider flipping the flagship.

- run tag: `code_stack_v1`
- primary scorer: `lightonai/GTE-ModernColBERT-v1`
- orthogonal scorer: `lightonai/LateOn-Code-edge` (available: `True`)
- shared chunk budget: `256` tokens (GTE tokenizer)
- max SWE-bench instances: `15`
- phantom threshold: `0.35`
- flagship cell: `colgrep` x `fuse_mean` (present: `True`)

## Headline matrix

| chunker | scorer | AUROC | grounded cov | cov delta | unused delta | phantom@thr | grounded held-out rate | p95 ms | support units |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sentence_packed | `gte_only` | 0.6058 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1382.18 | 5.1 |
| sentence_packed | `code_only` | 0.6111 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1792.88 | 5.1 |
| sentence_packed | `fuse_mean` | 0.6296 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1772.43 | 5.1 |
| sentence_packed | `fuse_max` | 0.6058 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1772.43 | 5.1 |
| sentence_packed | `fuse_min` | 0.6111 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1772.43 | 5.1 |
| colgrep | `gte_only` | 0.6243 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 798.04 | 10.5 |
| colgrep | `code_only` | 0.6005 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 707.84 | 10.5 |
| colgrep | `fuse_mean` (flagship) | 0.6032 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 776.96 | 10.5 |
| colgrep | `fuse_max` | 0.6243 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 776.96 | 10.5 |
| colgrep | `fuse_min` | 0.6005 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 776.96 | 10.5 |

`*` marks cells that pass `reverse_context AUROC >= 0.90`.

## Stacking lift (fuse_* minus gte_only, per chunker)

| chunker | rule | AUROC lift | unused-delta lift | phantom precision lift |
|---|---|---:|---:|---:|
| sentence_packed | fuse_mean | +0.0238 | +0.0000 | +0.0000 |
| sentence_packed | fuse_max | +0.0000 | +0.0000 | +0.0000 |
| sentence_packed | fuse_min | +0.0053 | +0.0000 | +0.0000 |
| colgrep | fuse_mean | -0.0212 | +0.0000 | +0.0000 |
| colgrep | fuse_max | +0.0000 | +0.0000 | +0.0000 |
| colgrep | fuse_min | -0.0238 | +0.0000 | +0.0000 |

## Chunker lift (colgrep minus sentence_packed, per scorer)

| scorer | AUROC lift | unused-delta lift | phantom precision lift |
|---|---:|---:|---:|
| `gte_only` | +0.0185 | +0.0000 | +0.0000 |
| `code_only` | -0.0106 | +0.0000 | +0.0000 |
| `fuse_mean` | -0.0265 | +0.0000 | +0.0000 |
| `fuse_max` | +0.0185 | +0.0000 | +0.0000 |
| `fuse_min` | -0.0106 | +0.0000 | +0.0000 |

## Subcategory snapshot (per cell)

### `sentence_packed` x `gte_only`

| subcategory | count | reverse_context | coverage | unused ratio | max evidence |
|---|---:|---:|---:|---:|---:|
| entity_swap | 10 | 0.9805 | 1.0000 | 0.0000 | 0.9971 |
| grounded | 18 | 0.9817 | 1.0000 | 0.0000 | 0.9983 |
| negation_flip | 3 | 0.9793 | 1.0000 | 0.0000 | 0.9972 |
| parametric | 2 | 0.9697 | 1.0000 | 0.0000 | 0.9961 |
| partial | 2 | 0.9696 | 1.0000 | 0.0000 | 0.9976 |
| phantom_api | 7 | 0.9789 | 1.0000 | 0.0000 | 0.9988 |

### `sentence_packed` x `code_only`

| subcategory | count | reverse_context | coverage | unused ratio | max evidence |
|---|---:|---:|---:|---:|---:|
| entity_swap | 10 | 0.9326 | 1.0000 | 0.0000 | 0.9879 |
| grounded | 18 | 0.9373 | 1.0000 | 0.0000 | 0.9905 |
| negation_flip | 3 | 0.9196 | 1.0000 | 0.0000 | 0.9849 |
| parametric | 2 | 0.9081 | 1.0000 | 0.0000 | 0.9863 |
| partial | 2 | 0.9193 | 1.0000 | 0.0000 | 0.9981 |
| phantom_api | 7 | 0.9301 | 1.0000 | 0.0000 | 0.9841 |

### `sentence_packed` x `fuse_mean`

| subcategory | count | reverse_context | coverage | unused ratio | max evidence |
|---|---:|---:|---:|---:|---:|
| entity_swap | 10 | 0.9565 | 1.0000 | 0.0000 | 0.9925 |
| grounded | 18 | 0.9595 | 1.0000 | 0.0000 | 0.9944 |
| negation_flip | 3 | 0.9495 | 1.0000 | 0.0000 | 0.9911 |
| parametric | 2 | 0.9389 | 1.0000 | 0.0000 | 0.9912 |
| partial | 2 | 0.9445 | 1.0000 | 0.0000 | 0.9978 |
| phantom_api | 7 | 0.9545 | 1.0000 | 0.0000 | 0.9914 |

### `sentence_packed` x `fuse_max`

| subcategory | count | reverse_context | coverage | unused ratio | max evidence |
|---|---:|---:|---:|---:|---:|
| entity_swap | 10 | 0.9805 | 1.0000 | 0.0000 | 0.9974 |
| grounded | 18 | 0.9817 | 1.0000 | 0.0000 | 0.9985 |
| negation_flip | 3 | 0.9793 | 1.0000 | 0.0000 | 0.9979 |
| parametric | 2 | 0.9697 | 1.0000 | 0.0000 | 0.9978 |
| partial | 2 | 0.9696 | 1.0000 | 0.0000 | 0.9991 |
| phantom_api | 7 | 0.9789 | 1.0000 | 0.0000 | 0.9988 |

### `sentence_packed` x `fuse_min`

| subcategory | count | reverse_context | coverage | unused ratio | max evidence |
|---|---:|---:|---:|---:|---:|
| entity_swap | 10 | 0.9326 | 1.0000 | 0.0000 | 0.9876 |
| grounded | 18 | 0.9373 | 1.0000 | 0.0000 | 0.9903 |
| negation_flip | 3 | 0.9196 | 1.0000 | 0.0000 | 0.9842 |
| parametric | 2 | 0.9081 | 1.0000 | 0.0000 | 0.9847 |
| partial | 2 | 0.9193 | 1.0000 | 0.0000 | 0.9966 |
| phantom_api | 7 | 0.9301 | 1.0000 | 0.0000 | 0.9841 |

### `colgrep` x `gte_only`

| subcategory | count | reverse_context | coverage | unused ratio | max evidence |
|---|---:|---:|---:|---:|---:|
| entity_swap | 10 | 0.9795 | 1.0000 | 0.0000 | 0.9976 |
| grounded | 18 | 0.9812 | 1.0000 | 0.0000 | 0.9985 |
| negation_flip | 3 | 0.9796 | 1.0000 | 0.0000 | 0.9971 |
| parametric | 2 | 0.9789 | 1.0000 | 0.0000 | 0.9986 |
| partial | 2 | 0.9793 | 1.0000 | 0.0000 | 0.9981 |
| phantom_api | 7 | 0.9798 | 1.0000 | 0.0000 | 0.9992 |

### `colgrep` x `code_only`

| subcategory | count | reverse_context | coverage | unused ratio | max evidence |
|---|---:|---:|---:|---:|---:|
| entity_swap | 10 | 0.9319 | 1.0000 | 0.0000 | 0.9904 |
| grounded | 18 | 0.9365 | 1.0000 | 0.0000 | 0.9913 |
| negation_flip | 3 | 0.9260 | 1.0000 | 0.0000 | 0.9883 |
| parametric | 2 | 0.9273 | 1.0000 | 0.0000 | 0.9999 |
| partial | 2 | 0.9385 | 1.0000 | 0.0000 | 0.9999 |
| phantom_api | 7 | 0.9278 | 1.0000 | 0.0000 | 0.9918 |

### `colgrep` x `fuse_mean`

| subcategory | count | reverse_context | coverage | unused ratio | max evidence |
|---|---:|---:|---:|---:|---:|
| entity_swap | 10 | 0.9557 | 1.0000 | 0.0000 | 0.9940 |
| grounded | 18 | 0.9588 | 1.0000 | 0.0000 | 0.9949 |
| negation_flip | 3 | 0.9528 | 1.0000 | 0.0000 | 0.9927 |
| parametric | 2 | 0.9531 | 1.0000 | 0.0000 | 0.9992 |
| partial | 2 | 0.9589 | 1.0000 | 0.0000 | 0.9990 |
| phantom_api | 7 | 0.9538 | 1.0000 | 0.0000 | 0.9955 |

### `colgrep` x `fuse_max`

| subcategory | count | reverse_context | coverage | unused ratio | max evidence |
|---|---:|---:|---:|---:|---:|
| entity_swap | 10 | 0.9795 | 1.0000 | 0.0000 | 0.9981 |
| grounded | 18 | 0.9812 | 1.0000 | 0.0000 | 0.9987 |
| negation_flip | 3 | 0.9796 | 1.0000 | 0.0000 | 0.9991 |
| parametric | 2 | 0.9789 | 1.0000 | 0.0000 | 0.9999 |
| partial | 2 | 0.9793 | 1.0000 | 0.0000 | 0.9999 |
| phantom_api | 7 | 0.9798 | 1.0000 | 0.0000 | 0.9995 |

### `colgrep` x `fuse_min`

| subcategory | count | reverse_context | coverage | unused ratio | max evidence |
|---|---:|---:|---:|---:|---:|
| entity_swap | 10 | 0.9319 | 1.0000 | 0.0000 | 0.9900 |
| grounded | 18 | 0.9365 | 1.0000 | 0.0000 | 0.9911 |
| negation_flip | 3 | 0.9260 | 1.0000 | 0.0000 | 0.9863 |
| parametric | 2 | 0.9273 | 1.0000 | 0.0000 | 0.9986 |
| partial | 2 | 0.9385 | 1.0000 | 0.0000 | 0.9981 |
| phantom_api | 7 | 0.9278 | 1.0000 | 0.0000 | 0.9915 |

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
| SWG1 | sentence_packed | `gte_only` | grounded | - | 0.9754 | 1.0000 | 0.0000 | False | 0.9974 | 3 |
| SWG1 | sentence_packed | `code_only` | grounded | - | 0.9180 | 1.0000 | 0.0000 | False | 0.9787 | 3 |
| SWG1 | sentence_packed | `fuse_mean` | grounded | - | 0.9467 | 1.0000 | 0.0000 | False | 0.9881 | 3 |
| SWG1 | sentence_packed | `fuse_max` | grounded | - | 0.9754 | 1.0000 | 0.0000 | False | 0.9974 | 3 |
| SWG1 | sentence_packed | `fuse_min` | grounded | - | 0.9180 | 1.0000 | 0.0000 | False | 0.9787 | 3 |
| SWU1 | sentence_packed | `gte_only` | ungrounded | entity_swap | 0.9752 | 1.0000 | 0.0000 | False | 0.9991 | 3 |
| SWU1 | sentence_packed | `code_only` | ungrounded | entity_swap | 0.9187 | 1.0000 | 0.0000 | False | 0.9758 | 3 |
| SWU1 | sentence_packed | `fuse_mean` | ungrounded | entity_swap | 0.9469 | 1.0000 | 0.0000 | False | 0.9874 | 3 |
| SWU1 | sentence_packed | `fuse_max` | ungrounded | entity_swap | 0.9752 | 1.0000 | 0.0000 | False | 0.9991 | 3 |
| SWU1 | sentence_packed | `fuse_min` | ungrounded | entity_swap | 0.9187 | 1.0000 | 0.0000 | False | 0.9758 | 3 |
| SWG2 | sentence_packed | `gte_only` | grounded | - | 0.9754 | 1.0000 | 0.0000 | False | 0.9999 | 2 |
| SWG2 | sentence_packed | `code_only` | grounded | - | 0.9251 | 1.0000 | 0.0000 | False | 0.9814 | 2 |
| SWG2 | sentence_packed | `fuse_mean` | grounded | - | 0.9503 | 1.0000 | 0.0000 | False | 0.9907 | 2 |
| SWG2 | sentence_packed | `fuse_max` | grounded | - | 0.9754 | 1.0000 | 0.0000 | False | 0.9999 | 2 |
| SWG2 | sentence_packed | `fuse_min` | grounded | - | 0.9251 | 1.0000 | 0.0000 | False | 0.9814 | 2 |
| SWU2 | sentence_packed | `gte_only` | ungrounded | phantom_api | 0.9742 | 1.0000 | 0.0000 | False | 1.0000 | 2 |
| SWU2 | sentence_packed | `code_only` | ungrounded | phantom_api | 0.9241 | 1.0000 | 0.0000 | False | 0.9768 | 2 |
| SWU2 | sentence_packed | `fuse_mean` | ungrounded | phantom_api | 0.9491 | 1.0000 | 0.0000 | False | 0.9884 | 2 |
| SWU2 | sentence_packed | `fuse_max` | ungrounded | phantom_api | 0.9742 | 1.0000 | 0.0000 | False | 1.0000 | 2 |
| SWU2 | sentence_packed | `fuse_min` | ungrounded | phantom_api | 0.9241 | 1.0000 | 0.0000 | False | 0.9768 | 2 |
| SWG3 | sentence_packed | `gte_only` | grounded | - | 0.9829 | 1.0000 | 0.0000 | False | 0.9972 | 2 |
| SWG3 | sentence_packed | `code_only` | grounded | - | 0.9407 | 1.0000 | 0.0000 | False | 0.9925 | 2 |
| SWG3 | sentence_packed | `fuse_mean` | grounded | - | 0.9618 | 1.0000 | 0.0000 | False | 0.9948 | 2 |
| SWG3 | sentence_packed | `fuse_max` | grounded | - | 0.9829 | 1.0000 | 0.0000 | False | 0.9972 | 2 |
| SWG3 | sentence_packed | `fuse_min` | grounded | - | 0.9407 | 1.0000 | 0.0000 | False | 0.9925 | 2 |
| SWU3 | sentence_packed | `gte_only` | ungrounded | negation_flip | 0.9831 | 1.0000 | 0.0000 | False | 0.9971 | 2 |
| SWU3 | sentence_packed | `code_only` | ungrounded | negation_flip | 0.9341 | 1.0000 | 0.0000 | False | 0.9687 | 2 |
| SWU3 | sentence_packed | `fuse_mean` | ungrounded | negation_flip | 0.9586 | 1.0000 | 0.0000 | False | 0.9829 | 2 |
| SWU3 | sentence_packed | `fuse_max` | ungrounded | negation_flip | 0.9831 | 1.0000 | 0.0000 | False | 0.9971 | 2 |
| SWU3 | sentence_packed | `fuse_min` | ungrounded | negation_flip | 0.9341 | 1.0000 | 0.0000 | False | 0.9687 | 2 |
| SWG4 | sentence_packed | `gte_only` | grounded | - | 0.9840 | 1.0000 | 0.0000 | False | 0.9973 | 3 |
| SWG4 | sentence_packed | `code_only` | grounded | - | 0.9429 | 1.0000 | 0.0000 | False | 0.9934 | 3 |
| SWG4 | sentence_packed | `fuse_mean` | grounded | - | 0.9635 | 1.0000 | 0.0000 | False | 0.9953 | 3 |
| SWG4 | sentence_packed | `fuse_max` | grounded | - | 0.9840 | 1.0000 | 0.0000 | False | 0.9973 | 3 |
| SWG4 | sentence_packed | `fuse_min` | grounded | - | 0.9429 | 1.0000 | 0.0000 | False | 0.9934 | 3 |
| SWU4 | sentence_packed | `gte_only` | ungrounded | entity_swap | 0.9828 | 1.0000 | 0.0000 | False | 0.9962 | 3 |
| SWU4 | sentence_packed | `code_only` | ungrounded | entity_swap | 0.9372 | 1.0000 | 0.0000 | False | 0.9938 | 3 |
| SWU4 | sentence_packed | `fuse_mean` | ungrounded | entity_swap | 0.9600 | 1.0000 | 0.0000 | False | 0.9950 | 3 |
| SWU4 | sentence_packed | `fuse_max` | ungrounded | entity_swap | 0.9828 | 1.0000 | 0.0000 | False | 0.9962 | 3 |
| SWU4 | sentence_packed | `fuse_min` | ungrounded | entity_swap | 0.9372 | 1.0000 | 0.0000 | False | 0.9938 | 3 |
| SWG5 | sentence_packed | `gte_only` | grounded | - | 0.9793 | 1.0000 | 0.0000 | False | 0.9976 | 4 |
| SWG5 | sentence_packed | `code_only` | grounded | - | 0.9294 | 1.0000 | 0.0000 | False | 0.9941 | 4 |
| SWG5 | sentence_packed | `fuse_mean` | grounded | - | 0.9543 | 1.0000 | 0.0000 | False | 0.9958 | 4 |
| SWG5 | sentence_packed | `fuse_max` | grounded | - | 0.9793 | 1.0000 | 0.0000 | False | 0.9976 | 4 |
| SWG5 | sentence_packed | `fuse_min` | grounded | - | 0.9294 | 1.0000 | 0.0000 | False | 0.9941 | 4 |
| SWU5 | sentence_packed | `gte_only` | ungrounded | entity_swap | 0.9793 | 1.0000 | 0.0000 | False | 0.9975 | 4 |
| SWU5 | sentence_packed | `code_only` | ungrounded | entity_swap | 0.9293 | 1.0000 | 0.0000 | False | 0.9942 | 4 |
| SWU5 | sentence_packed | `fuse_mean` | ungrounded | entity_swap | 0.9543 | 1.0000 | 0.0000 | False | 0.9958 | 4 |
| SWU5 | sentence_packed | `fuse_max` | ungrounded | entity_swap | 0.9793 | 1.0000 | 0.0000 | False | 0.9975 | 4 |
| SWU5 | sentence_packed | `fuse_min` | ungrounded | entity_swap | 0.9293 | 1.0000 | 0.0000 | False | 0.9942 | 4 |
| SWG6 | sentence_packed | `gte_only` | grounded | - | 0.9819 | 1.0000 | 0.0000 | False | 0.9966 | 4 |
| SWG6 | sentence_packed | `code_only` | grounded | - | 0.9462 | 1.0000 | 0.0000 | False | 0.9740 | 4 |
| SWG6 | sentence_packed | `fuse_mean` | grounded | - | 0.9640 | 1.0000 | 0.0000 | False | 0.9853 | 4 |
| SWG6 | sentence_packed | `fuse_max` | grounded | - | 0.9819 | 1.0000 | 0.0000 | False | 0.9966 | 4 |
| SWG6 | sentence_packed | `fuse_min` | grounded | - | 0.9462 | 1.0000 | 0.0000 | False | 0.9740 | 4 |
| SWU6 | sentence_packed | `gte_only` | ungrounded | entity_swap | 0.9819 | 1.0000 | 0.0000 | False | 0.9966 | 4 |
| SWU6 | sentence_packed | `code_only` | ungrounded | entity_swap | 0.9452 | 1.0000 | 0.0000 | False | 0.9745 | 4 |
| SWU6 | sentence_packed | `fuse_mean` | ungrounded | entity_swap | 0.9635 | 1.0000 | 0.0000 | False | 0.9855 | 4 |
| SWU6 | sentence_packed | `fuse_max` | ungrounded | entity_swap | 0.9819 | 1.0000 | 0.0000 | False | 0.9966 | 4 |
| SWU6 | sentence_packed | `fuse_min` | ungrounded | entity_swap | 0.9452 | 1.0000 | 0.0000 | False | 0.9745 | 4 |
| SWG7 | sentence_packed | `gte_only` | grounded | - | 0.9871 | 1.0000 | 0.0000 | False | 0.9998 | 5 |
| SWG7 | sentence_packed | `code_only` | grounded | - | 0.9585 | 1.0000 | 0.0000 | False | 0.9947 | 5 |
| SWG7 | sentence_packed | `fuse_mean` | grounded | - | 0.9728 | 1.0000 | 0.0000 | False | 0.9973 | 5 |
| SWG7 | sentence_packed | `fuse_max` | grounded | - | 0.9871 | 1.0000 | 0.0000 | False | 0.9998 | 5 |
| SWG7 | sentence_packed | `fuse_min` | grounded | - | 0.9585 | 1.0000 | 0.0000 | False | 0.9947 | 5 |
| SWU7 | sentence_packed | `gte_only` | ungrounded | phantom_api | 0.9857 | 1.0000 | 0.0000 | False | 0.9998 | 5 |
| SWU7 | sentence_packed | `code_only` | ungrounded | phantom_api | 0.9542 | 1.0000 | 0.0000 | False | 0.9947 | 5 |
| SWU7 | sentence_packed | `fuse_mean` | ungrounded | phantom_api | 0.9700 | 1.0000 | 0.0000 | False | 0.9973 | 5 |
| SWU7 | sentence_packed | `fuse_max` | ungrounded | phantom_api | 0.9857 | 1.0000 | 0.0000 | False | 0.9998 | 5 |
| SWU7 | sentence_packed | `fuse_min` | ungrounded | phantom_api | 0.9542 | 1.0000 | 0.0000 | False | 0.9947 | 5 |
| SWG8 | sentence_packed | `gte_only` | grounded | - | 0.9855 | 1.0000 | 0.0000 | False | 0.9982 | 10 |
| SWG8 | sentence_packed | `code_only` | grounded | - | 0.9557 | 1.0000 | 0.0000 | False | 0.9977 | 10 |
| SWG8 | sentence_packed | `fuse_mean` | grounded | - | 0.9706 | 1.0000 | 0.0000 | False | 0.9979 | 10 |
| SWG8 | sentence_packed | `fuse_max` | grounded | - | 0.9855 | 1.0000 | 0.0000 | False | 0.9982 | 10 |
| SWG8 | sentence_packed | `fuse_min` | grounded | - | 0.9557 | 1.0000 | 0.0000 | False | 0.9977 | 10 |
| SWU8 | sentence_packed | `gte_only` | ungrounded | entity_swap | 0.9852 | 1.0000 | 0.0000 | False | 0.9983 | 10 |
| SWU8 | sentence_packed | `code_only` | ungrounded | entity_swap | 0.9560 | 1.0000 | 0.0000 | False | 0.9946 | 10 |
| SWU8 | sentence_packed | `fuse_mean` | ungrounded | entity_swap | 0.9706 | 1.0000 | 0.0000 | False | 0.9965 | 10 |
| SWU8 | sentence_packed | `fuse_max` | ungrounded | entity_swap | 0.9852 | 1.0000 | 0.0000 | False | 0.9983 | 10 |
| SWU8 | sentence_packed | `fuse_min` | ungrounded | entity_swap | 0.9560 | 1.0000 | 0.0000 | False | 0.9946 | 10 |
| SWG9 | sentence_packed | `gte_only` | grounded | - | 0.9829 | 1.0000 | 0.0000 | False | 0.9992 | 9 |
| SWG9 | sentence_packed | `code_only` | grounded | - | 0.9490 | 1.0000 | 0.0000 | False | 0.9887 | 9 |
| SWG9 | sentence_packed | `fuse_mean` | grounded | - | 0.9660 | 1.0000 | 0.0000 | False | 0.9939 | 9 |
| SWG9 | sentence_packed | `fuse_max` | grounded | - | 0.9829 | 1.0000 | 0.0000 | False | 0.9992 | 9 |
| SWG9 | sentence_packed | `fuse_min` | grounded | - | 0.9490 | 1.0000 | 0.0000 | False | 0.9887 | 9 |
| SWU9 | sentence_packed | `gte_only` | ungrounded | phantom_api | 0.9823 | 1.0000 | 0.0000 | False | 0.9992 | 9 |
| SWU9 | sentence_packed | `code_only` | ungrounded | phantom_api | 0.9448 | 1.0000 | 0.0000 | False | 0.9882 | 9 |
| SWU9 | sentence_packed | `fuse_mean` | ungrounded | phantom_api | 0.9636 | 1.0000 | 0.0000 | False | 0.9937 | 9 |
| SWU9 | sentence_packed | `fuse_max` | ungrounded | phantom_api | 0.9823 | 1.0000 | 0.0000 | False | 0.9992 | 9 |
| SWU9 | sentence_packed | `fuse_min` | ungrounded | phantom_api | 0.9448 | 1.0000 | 0.0000 | False | 0.9882 | 9 |
| SWG10 | sentence_packed | `gte_only` | grounded | - | 0.9868 | 1.0000 | 0.0000 | False | 0.9997 | 6 |
| SWG10 | sentence_packed | `code_only` | grounded | - | 0.9560 | 1.0000 | 0.0000 | False | 0.9943 | 6 |
| SWG10 | sentence_packed | `fuse_mean` | grounded | - | 0.9714 | 1.0000 | 0.0000 | False | 0.9970 | 6 |
| SWG10 | sentence_packed | `fuse_max` | grounded | - | 0.9868 | 1.0000 | 0.0000 | False | 0.9997 | 6 |
| SWG10 | sentence_packed | `fuse_min` | grounded | - | 0.9560 | 1.0000 | 0.0000 | False | 0.9943 | 6 |
| SWU10 | sentence_packed | `gte_only` | ungrounded | phantom_api | 0.9855 | 1.0000 | 0.0000 | False | 0.9997 | 6 |
| SWU10 | sentence_packed | `code_only` | ungrounded | phantom_api | 0.9536 | 1.0000 | 0.0000 | False | 0.9943 | 6 |
| SWU10 | sentence_packed | `fuse_mean` | ungrounded | phantom_api | 0.9695 | 1.0000 | 0.0000 | False | 0.9970 | 6 |
| SWU10 | sentence_packed | `fuse_max` | ungrounded | phantom_api | 0.9855 | 1.0000 | 0.0000 | False | 0.9997 | 6 |
| SWU10 | sentence_packed | `fuse_min` | ungrounded | phantom_api | 0.9536 | 1.0000 | 0.0000 | False | 0.9943 | 6 |
| SWG11 | sentence_packed | `gte_only` | grounded | - | 0.9868 | 1.0000 | 0.0000 | False | 0.9981 | 18 |
| SWG11 | sentence_packed | `code_only` | grounded | - | 0.9552 | 1.0000 | 0.0000 | False | 0.9950 | 18 |
| SWG11 | sentence_packed | `fuse_mean` | grounded | - | 0.9710 | 1.0000 | 0.0000 | False | 0.9965 | 18 |
| SWG11 | sentence_packed | `fuse_max` | grounded | - | 0.9868 | 1.0000 | 0.0000 | False | 0.9981 | 18 |
| SWG11 | sentence_packed | `fuse_min` | grounded | - | 0.9552 | 1.0000 | 0.0000 | False | 0.9950 | 18 |
| SWU11 | sentence_packed | `gte_only` | ungrounded | phantom_api | 0.9859 | 1.0000 | 0.0000 | False | 0.9969 | 18 |
| SWU11 | sentence_packed | `code_only` | ungrounded | phantom_api | 0.9558 | 1.0000 | 0.0000 | False | 0.9865 | 18 |
| SWU11 | sentence_packed | `fuse_mean` | ungrounded | phantom_api | 0.9708 | 1.0000 | 0.0000 | False | 0.9917 | 18 |
| SWU11 | sentence_packed | `fuse_max` | ungrounded | phantom_api | 0.9859 | 1.0000 | 0.0000 | False | 0.9969 | 18 |
| SWU11 | sentence_packed | `fuse_min` | ungrounded | phantom_api | 0.9558 | 1.0000 | 0.0000 | False | 0.9865 | 18 |
| SWG12 | sentence_packed | `gte_only` | grounded | - | 0.9834 | 1.0000 | 0.0000 | False | 0.9977 | 4 |
| SWG12 | sentence_packed | `code_only` | grounded | - | 0.9373 | 1.0000 | 0.0000 | False | 0.9891 | 4 |
| SWG12 | sentence_packed | `fuse_mean` | grounded | - | 0.9604 | 1.0000 | 0.0000 | False | 0.9934 | 4 |
| SWG12 | sentence_packed | `fuse_max` | grounded | - | 0.9834 | 1.0000 | 0.0000 | False | 0.9977 | 4 |
| SWG12 | sentence_packed | `fuse_min` | grounded | - | 0.9373 | 1.0000 | 0.0000 | False | 0.9891 | 4 |
| SWU12 | sentence_packed | `gte_only` | ungrounded | negation_flip | 0.9830 | 1.0000 | 0.0000 | False | 0.9978 | 4 |
| SWU12 | sentence_packed | `code_only` | ungrounded | negation_flip | 0.9378 | 1.0000 | 0.0000 | False | 1.0000 | 4 |
| SWU12 | sentence_packed | `fuse_mean` | ungrounded | negation_flip | 0.9604 | 1.0000 | 0.0000 | False | 0.9989 | 4 |
| SWU12 | sentence_packed | `fuse_max` | ungrounded | negation_flip | 0.9830 | 1.0000 | 0.0000 | False | 1.0000 | 4 |
| SWU12 | sentence_packed | `fuse_min` | ungrounded | negation_flip | 0.9378 | 1.0000 | 0.0000 | False | 0.9978 | 4 |
| SWG13 | sentence_packed | `gte_only` | grounded | - | 0.9817 | 1.0000 | 0.0000 | False | 0.9998 | 15 |
| SWG13 | sentence_packed | `code_only` | grounded | - | 0.9425 | 1.0000 | 0.0000 | False | 0.9926 | 15 |
| SWG13 | sentence_packed | `fuse_mean` | grounded | - | 0.9621 | 1.0000 | 0.0000 | False | 0.9962 | 15 |
| SWG13 | sentence_packed | `fuse_max` | grounded | - | 0.9817 | 1.0000 | 0.0000 | False | 0.9998 | 15 |
| SWG13 | sentence_packed | `fuse_min` | grounded | - | 0.9425 | 1.0000 | 0.0000 | False | 0.9926 | 15 |
| SWU13 | sentence_packed | `gte_only` | ungrounded | entity_swap | 0.9818 | 1.0000 | 0.0000 | False | 0.9977 | 15 |
| SWU13 | sentence_packed | `code_only` | ungrounded | entity_swap | 0.9422 | 1.0000 | 0.0000 | False | 0.9855 | 15 |
| SWU13 | sentence_packed | `fuse_mean` | ungrounded | entity_swap | 0.9620 | 1.0000 | 0.0000 | False | 0.9916 | 15 |
| SWU13 | sentence_packed | `fuse_max` | ungrounded | entity_swap | 0.9818 | 1.0000 | 0.0000 | False | 0.9977 | 15 |
| SWU13 | sentence_packed | `fuse_min` | ungrounded | entity_swap | 0.9422 | 1.0000 | 0.0000 | False | 0.9855 | 15 |
| SWG14 | sentence_packed | `gte_only` | grounded | - | 0.9846 | 1.0000 | 0.0000 | False | 0.9969 | 5 |
| SWG14 | sentence_packed | `code_only` | grounded | - | 0.9332 | 1.0000 | 0.0000 | False | 0.9802 | 5 |
| SWG14 | sentence_packed | `fuse_mean` | grounded | - | 0.9589 | 1.0000 | 0.0000 | False | 0.9885 | 5 |
| SWG14 | sentence_packed | `fuse_max` | grounded | - | 0.9846 | 1.0000 | 0.0000 | False | 0.9969 | 5 |
| SWG14 | sentence_packed | `fuse_min` | grounded | - | 0.9332 | 1.0000 | 0.0000 | False | 0.9802 | 5 |
| SWU14 | sentence_packed | `gte_only` | ungrounded | entity_swap | 0.9846 | 1.0000 | 0.0000 | False | 0.9969 | 5 |
| SWU14 | sentence_packed | `code_only` | ungrounded | entity_swap | 0.9329 | 1.0000 | 0.0000 | False | 0.9799 | 5 |
| SWU14 | sentence_packed | `fuse_mean` | ungrounded | entity_swap | 0.9587 | 1.0000 | 0.0000 | False | 0.9884 | 5 |
| SWU14 | sentence_packed | `fuse_max` | ungrounded | entity_swap | 0.9846 | 1.0000 | 0.0000 | False | 0.9969 | 5 |
| SWU14 | sentence_packed | `fuse_min` | ungrounded | entity_swap | 0.9329 | 1.0000 | 0.0000 | False | 0.9799 | 5 |
| SWG15 | sentence_packed | `gte_only` | grounded | - | 0.9888 | 1.0000 | 0.0000 | False | 0.9974 | 11 |
| SWG15 | sentence_packed | `code_only` | grounded | - | 0.9541 | 1.0000 | 0.0000 | False | 0.9915 | 11 |
| SWG15 | sentence_packed | `fuse_mean` | grounded | - | 0.9714 | 1.0000 | 0.0000 | False | 0.9945 | 11 |
| SWG15 | sentence_packed | `fuse_max` | grounded | - | 0.9888 | 1.0000 | 0.0000 | False | 0.9974 | 11 |
| SWG15 | sentence_packed | `fuse_min` | grounded | - | 0.9541 | 1.0000 | 0.0000 | False | 0.9915 | 11 |
| SWU15 | sentence_packed | `gte_only` | ungrounded | entity_swap | 0.9878 | 1.0000 | 0.0000 | False | 0.9942 | 11 |
| SWU15 | sentence_packed | `code_only` | ungrounded | entity_swap | 0.9492 | 1.0000 | 0.0000 | False | 0.9944 | 11 |
| SWU15 | sentence_packed | `fuse_mean` | ungrounded | entity_swap | 0.9685 | 1.0000 | 0.0000 | False | 0.9943 | 11 |
| SWU15 | sentence_packed | `fuse_max` | ungrounded | entity_swap | 0.9878 | 1.0000 | 0.0000 | False | 0.9944 | 11 |
| SWU15 | sentence_packed | `fuse_min` | ungrounded | entity_swap | 0.9492 | 1.0000 | 0.0000 | False | 0.9942 | 11 |
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
| SWG1 | colgrep | `gte_only` | grounded | - | 0.9617 | 1.0000 | 0.0000 | False | 0.9976 | 3 |
| SWG1 | colgrep | `code_only` | grounded | - | 0.8787 | 1.0000 | 0.0000 | False | 0.9747 | 3 |
| SWG1 | colgrep | `fuse_mean` | grounded | - | 0.9202 | 1.0000 | 0.0000 | False | 0.9861 | 3 |
| SWG1 | colgrep | `fuse_max` | grounded | - | 0.9617 | 1.0000 | 0.0000 | False | 0.9976 | 3 |
| SWG1 | colgrep | `fuse_min` | grounded | - | 0.8787 | 1.0000 | 0.0000 | False | 0.9747 | 3 |
| SWU1 | colgrep | `gte_only` | ungrounded | entity_swap | 0.9614 | 1.0000 | 0.0000 | False | 0.9963 | 3 |
| SWU1 | colgrep | `code_only` | ungrounded | entity_swap | 0.8890 | 1.0000 | 0.0000 | False | 0.9804 | 3 |
| SWU1 | colgrep | `fuse_mean` | ungrounded | entity_swap | 0.9252 | 1.0000 | 0.0000 | False | 0.9884 | 3 |
| SWU1 | colgrep | `fuse_max` | ungrounded | entity_swap | 0.9614 | 1.0000 | 0.0000 | False | 0.9963 | 3 |
| SWU1 | colgrep | `fuse_min` | ungrounded | entity_swap | 0.8890 | 1.0000 | 0.0000 | False | 0.9804 | 3 |
| SWG2 | colgrep | `gte_only` | grounded | - | 0.9633 | 1.0000 | 0.0000 | False | 1.0000 | 2 |
| SWG2 | colgrep | `code_only` | grounded | - | 0.8877 | 1.0000 | 0.0000 | False | 0.9817 | 2 |
| SWG2 | colgrep | `fuse_mean` | grounded | - | 0.9255 | 1.0000 | 0.0000 | False | 0.9908 | 2 |
| SWG2 | colgrep | `fuse_max` | grounded | - | 0.9633 | 1.0000 | 0.0000 | False | 1.0000 | 2 |
| SWG2 | colgrep | `fuse_min` | grounded | - | 0.8877 | 1.0000 | 0.0000 | False | 0.9817 | 2 |
| SWU2 | colgrep | `gte_only` | ungrounded | phantom_api | 0.9620 | 1.0000 | 0.0000 | False | 0.9999 | 2 |
| SWU2 | colgrep | `code_only` | ungrounded | phantom_api | 0.8848 | 1.0000 | 0.0000 | False | 0.9767 | 2 |
| SWU2 | colgrep | `fuse_mean` | ungrounded | phantom_api | 0.9234 | 1.0000 | 0.0000 | False | 0.9883 | 2 |
| SWU2 | colgrep | `fuse_max` | ungrounded | phantom_api | 0.9620 | 1.0000 | 0.0000 | False | 0.9999 | 2 |
| SWU2 | colgrep | `fuse_min` | ungrounded | phantom_api | 0.8848 | 1.0000 | 0.0000 | False | 0.9767 | 2 |
| SWG3 | colgrep | `gte_only` | grounded | - | 0.9829 | 1.0000 | 0.0000 | False | 0.9974 | 4 |
| SWG3 | colgrep | `code_only` | grounded | - | 0.9414 | 1.0000 | 0.0000 | False | 0.9840 | 4 |
| SWG3 | colgrep | `fuse_mean` | grounded | - | 0.9621 | 1.0000 | 0.0000 | False | 0.9907 | 4 |
| SWG3 | colgrep | `fuse_max` | grounded | - | 0.9829 | 1.0000 | 0.0000 | False | 0.9974 | 4 |
| SWG3 | colgrep | `fuse_min` | grounded | - | 0.9414 | 1.0000 | 0.0000 | False | 0.9840 | 4 |
| SWU3 | colgrep | `gte_only` | ungrounded | negation_flip | 0.9827 | 1.0000 | 0.0000 | False | 0.9974 | 4 |
| SWU3 | colgrep | `code_only` | ungrounded | negation_flip | 0.9376 | 1.0000 | 0.0000 | False | 0.9651 | 4 |
| SWU3 | colgrep | `fuse_mean` | ungrounded | negation_flip | 0.9601 | 1.0000 | 0.0000 | False | 0.9812 | 4 |
| SWU3 | colgrep | `fuse_max` | ungrounded | negation_flip | 0.9827 | 1.0000 | 0.0000 | False | 0.9974 | 4 |
| SWU3 | colgrep | `fuse_min` | ungrounded | negation_flip | 0.9376 | 1.0000 | 0.0000 | False | 0.9651 | 4 |
| SWG4 | colgrep | `gte_only` | grounded | - | 0.9872 | 1.0000 | 0.0000 | False | 0.9972 | 8 |
| SWG4 | colgrep | `code_only` | grounded | - | 0.9486 | 1.0000 | 0.0000 | False | 0.9978 | 8 |
| SWG4 | colgrep | `fuse_mean` | grounded | - | 0.9679 | 1.0000 | 0.0000 | False | 0.9975 | 8 |
| SWG4 | colgrep | `fuse_max` | grounded | - | 0.9872 | 1.0000 | 0.0000 | False | 0.9978 | 8 |
| SWG4 | colgrep | `fuse_min` | grounded | - | 0.9486 | 1.0000 | 0.0000 | False | 0.9972 | 8 |
| SWU4 | colgrep | `gte_only` | ungrounded | entity_swap | 0.9859 | 1.0000 | 0.0000 | False | 0.9974 | 8 |
| SWU4 | colgrep | `code_only` | ungrounded | entity_swap | 0.9456 | 1.0000 | 0.0000 | False | 0.9963 | 8 |
| SWU4 | colgrep | `fuse_mean` | ungrounded | entity_swap | 0.9658 | 1.0000 | 0.0000 | False | 0.9969 | 8 |
| SWU4 | colgrep | `fuse_max` | ungrounded | entity_swap | 0.9859 | 1.0000 | 0.0000 | False | 0.9974 | 8 |
| SWU4 | colgrep | `fuse_min` | ungrounded | entity_swap | 0.9456 | 1.0000 | 0.0000 | False | 0.9963 | 8 |
| SWG5 | colgrep | `gte_only` | grounded | - | 0.9785 | 1.0000 | 0.0000 | False | 0.9982 | 4 |
| SWG5 | colgrep | `code_only` | grounded | - | 0.9335 | 1.0000 | 0.0000 | False | 0.9955 | 4 |
| SWG5 | colgrep | `fuse_mean` | grounded | - | 0.9560 | 1.0000 | 0.0000 | False | 0.9968 | 4 |
| SWG5 | colgrep | `fuse_max` | grounded | - | 0.9785 | 1.0000 | 0.0000 | False | 0.9982 | 4 |
| SWG5 | colgrep | `fuse_min` | grounded | - | 0.9335 | 1.0000 | 0.0000 | False | 0.9955 | 4 |
| SWU5 | colgrep | `gte_only` | ungrounded | entity_swap | 0.9784 | 1.0000 | 0.0000 | False | 0.9981 | 4 |
| SWU5 | colgrep | `code_only` | ungrounded | entity_swap | 0.9333 | 1.0000 | 0.0000 | False | 0.9950 | 4 |
| SWU5 | colgrep | `fuse_mean` | ungrounded | entity_swap | 0.9559 | 1.0000 | 0.0000 | False | 0.9966 | 4 |
| SWU5 | colgrep | `fuse_max` | ungrounded | entity_swap | 0.9784 | 1.0000 | 0.0000 | False | 0.9981 | 4 |
| SWU5 | colgrep | `fuse_min` | ungrounded | entity_swap | 0.9333 | 1.0000 | 0.0000 | False | 0.9950 | 4 |
| SWG6 | colgrep | `gte_only` | grounded | - | 0.9690 | 1.0000 | 0.0000 | False | 0.9981 | 4 |
| SWG6 | colgrep | `code_only` | grounded | - | 0.9053 | 1.0000 | 0.0000 | False | 0.9736 | 4 |
| SWG6 | colgrep | `fuse_mean` | grounded | - | 0.9371 | 1.0000 | 0.0000 | False | 0.9859 | 4 |
| SWG6 | colgrep | `fuse_max` | grounded | - | 0.9690 | 1.0000 | 0.0000 | False | 0.9981 | 4 |
| SWG6 | colgrep | `fuse_min` | grounded | - | 0.9053 | 1.0000 | 0.0000 | False | 0.9736 | 4 |
| SWU6 | colgrep | `gte_only` | ungrounded | entity_swap | 0.9689 | 1.0000 | 0.0000 | False | 0.9981 | 4 |
| SWU6 | colgrep | `code_only` | ungrounded | entity_swap | 0.9052 | 1.0000 | 0.0000 | False | 0.9731 | 4 |
| SWU6 | colgrep | `fuse_mean` | ungrounded | entity_swap | 0.9371 | 1.0000 | 0.0000 | False | 0.9856 | 4 |
| SWU6 | colgrep | `fuse_max` | ungrounded | entity_swap | 0.9689 | 1.0000 | 0.0000 | False | 0.9981 | 4 |
| SWU6 | colgrep | `fuse_min` | ungrounded | entity_swap | 0.9052 | 1.0000 | 0.0000 | False | 0.9731 | 4 |
| SWG7 | colgrep | `gte_only` | grounded | - | 0.9860 | 1.0000 | 0.0000 | False | 0.9998 | 6 |
| SWG7 | colgrep | `code_only` | grounded | - | 0.9532 | 1.0000 | 0.0000 | False | 0.9947 | 6 |
| SWG7 | colgrep | `fuse_mean` | grounded | - | 0.9696 | 1.0000 | 0.0000 | False | 0.9973 | 6 |
| SWG7 | colgrep | `fuse_max` | grounded | - | 0.9860 | 1.0000 | 0.0000 | False | 0.9998 | 6 |
| SWG7 | colgrep | `fuse_min` | grounded | - | 0.9532 | 1.0000 | 0.0000 | False | 0.9947 | 6 |
| SWU7 | colgrep | `gte_only` | ungrounded | phantom_api | 0.9846 | 1.0000 | 0.0000 | False | 0.9999 | 6 |
| SWU7 | colgrep | `code_only` | ungrounded | phantom_api | 0.9496 | 1.0000 | 0.0000 | False | 0.9947 | 6 |
| SWU7 | colgrep | `fuse_mean` | ungrounded | phantom_api | 0.9671 | 1.0000 | 0.0000 | False | 0.9973 | 6 |
| SWU7 | colgrep | `fuse_max` | ungrounded | phantom_api | 0.9846 | 1.0000 | 0.0000 | False | 0.9999 | 6 |
| SWU7 | colgrep | `fuse_min` | ungrounded | phantom_api | 0.9496 | 1.0000 | 0.0000 | False | 0.9947 | 6 |
| SWG8 | colgrep | `gte_only` | grounded | - | 0.9859 | 1.0000 | 0.0000 | False | 0.9982 | 17 |
| SWG8 | colgrep | `code_only` | grounded | - | 0.9572 | 1.0000 | 0.0000 | False | 0.9977 | 17 |
| SWG8 | colgrep | `fuse_mean` | grounded | - | 0.9715 | 1.0000 | 0.0000 | False | 0.9979 | 17 |
| SWG8 | colgrep | `fuse_max` | grounded | - | 0.9859 | 1.0000 | 0.0000 | False | 0.9982 | 17 |
| SWG8 | colgrep | `fuse_min` | grounded | - | 0.9572 | 1.0000 | 0.0000 | False | 0.9977 | 17 |
| SWU8 | colgrep | `gte_only` | ungrounded | entity_swap | 0.9856 | 1.0000 | 0.0000 | False | 0.9983 | 17 |
| SWU8 | colgrep | `code_only` | ungrounded | entity_swap | 0.9578 | 1.0000 | 0.0000 | False | 0.9946 | 17 |
| SWU8 | colgrep | `fuse_mean` | ungrounded | entity_swap | 0.9717 | 1.0000 | 0.0000 | False | 0.9965 | 17 |
| SWU8 | colgrep | `fuse_max` | ungrounded | entity_swap | 0.9856 | 1.0000 | 0.0000 | False | 0.9983 | 17 |
| SWU8 | colgrep | `fuse_min` | ungrounded | entity_swap | 0.9578 | 1.0000 | 0.0000 | False | 0.9946 | 17 |
| SWG9 | colgrep | `gte_only` | grounded | - | 0.9820 | 1.0000 | 0.0000 | False | 0.9992 | 17 |
| SWG9 | colgrep | `code_only` | grounded | - | 0.9494 | 1.0000 | 0.0000 | False | 0.9887 | 17 |
| SWG9 | colgrep | `fuse_mean` | grounded | - | 0.9657 | 1.0000 | 0.0000 | False | 0.9939 | 17 |
| SWG9 | colgrep | `fuse_max` | grounded | - | 0.9820 | 1.0000 | 0.0000 | False | 0.9992 | 17 |
| SWG9 | colgrep | `fuse_min` | grounded | - | 0.9494 | 1.0000 | 0.0000 | False | 0.9887 | 17 |
| SWU9 | colgrep | `gte_only` | ungrounded | phantom_api | 0.9811 | 1.0000 | 0.0000 | False | 0.9992 | 17 |
| SWU9 | colgrep | `code_only` | ungrounded | phantom_api | 0.9449 | 1.0000 | 0.0000 | False | 0.9882 | 17 |
| SWU9 | colgrep | `fuse_mean` | ungrounded | phantom_api | 0.9630 | 1.0000 | 0.0000 | False | 0.9937 | 17 |
| SWU9 | colgrep | `fuse_max` | ungrounded | phantom_api | 0.9811 | 1.0000 | 0.0000 | False | 0.9992 | 17 |
| SWU9 | colgrep | `fuse_min` | ungrounded | phantom_api | 0.9449 | 1.0000 | 0.0000 | False | 0.9882 | 17 |
| SWG10 | colgrep | `gte_only` | grounded | - | 0.9884 | 1.0000 | 0.0000 | False | 0.9999 | 10 |
| SWG10 | colgrep | `code_only` | grounded | - | 0.9599 | 1.0000 | 0.0000 | False | 0.9940 | 10 |
| SWG10 | colgrep | `fuse_mean` | grounded | - | 0.9742 | 1.0000 | 0.0000 | False | 0.9970 | 10 |
| SWG10 | colgrep | `fuse_max` | grounded | - | 0.9884 | 1.0000 | 0.0000 | False | 0.9999 | 10 |
| SWG10 | colgrep | `fuse_min` | grounded | - | 0.9599 | 1.0000 | 0.0000 | False | 0.9940 | 10 |
| SWU10 | colgrep | `gte_only` | ungrounded | phantom_api | 0.9875 | 1.0000 | 0.0000 | False | 0.9999 | 10 |
| SWU10 | colgrep | `code_only` | ungrounded | phantom_api | 0.9587 | 1.0000 | 0.0000 | False | 0.9940 | 10 |
| SWU10 | colgrep | `fuse_mean` | ungrounded | phantom_api | 0.9731 | 1.0000 | 0.0000 | False | 0.9970 | 10 |
| SWU10 | colgrep | `fuse_max` | ungrounded | phantom_api | 0.9875 | 1.0000 | 0.0000 | False | 0.9999 | 10 |
| SWU10 | colgrep | `fuse_min` | ungrounded | phantom_api | 0.9587 | 1.0000 | 0.0000 | False | 0.9940 | 10 |
| SWG11 | colgrep | `gte_only` | grounded | - | 0.9876 | 1.0000 | 0.0000 | False | 0.9987 | 32 |
| SWG11 | colgrep | `code_only` | grounded | - | 0.9560 | 1.0000 | 0.0000 | False | 0.9964 | 32 |
| SWG11 | colgrep | `fuse_mean` | grounded | - | 0.9718 | 1.0000 | 0.0000 | False | 0.9975 | 32 |
| SWG11 | colgrep | `fuse_max` | grounded | - | 0.9876 | 1.0000 | 0.0000 | False | 0.9987 | 32 |
| SWG11 | colgrep | `fuse_min` | grounded | - | 0.9560 | 1.0000 | 0.0000 | False | 0.9964 | 32 |
| SWU11 | colgrep | `gte_only` | ungrounded | phantom_api | 0.9864 | 1.0000 | 0.0000 | False | 0.9976 | 32 |
| SWU11 | colgrep | `code_only` | ungrounded | phantom_api | 0.9546 | 1.0000 | 0.0000 | False | 0.9893 | 32 |
| SWU11 | colgrep | `fuse_mean` | ungrounded | phantom_api | 0.9705 | 1.0000 | 0.0000 | False | 0.9935 | 32 |
| SWU11 | colgrep | `fuse_max` | ungrounded | phantom_api | 0.9864 | 1.0000 | 0.0000 | False | 0.9976 | 32 |
| SWU11 | colgrep | `fuse_min` | ungrounded | phantom_api | 0.9546 | 1.0000 | 0.0000 | False | 0.9893 | 32 |
| SWG12 | colgrep | `gte_only` | grounded | - | 0.9760 | 1.0000 | 0.0000 | False | 0.9964 | 4 |
| SWG12 | colgrep | `code_only` | grounded | - | 0.9181 | 1.0000 | 0.0000 | False | 0.9889 | 4 |
| SWG12 | colgrep | `fuse_mean` | grounded | - | 0.9471 | 1.0000 | 0.0000 | False | 0.9927 | 4 |
| SWG12 | colgrep | `fuse_max` | grounded | - | 0.9760 | 1.0000 | 0.0000 | False | 0.9964 | 4 |
| SWG12 | colgrep | `fuse_min` | grounded | - | 0.9181 | 1.0000 | 0.0000 | False | 0.9889 | 4 |
| SWU12 | colgrep | `gte_only` | ungrounded | negation_flip | 0.9756 | 1.0000 | 0.0000 | False | 0.9962 | 4 |
| SWU12 | colgrep | `code_only` | ungrounded | negation_flip | 0.9187 | 1.0000 | 0.0000 | False | 1.0000 | 4 |
| SWU12 | colgrep | `fuse_mean` | ungrounded | negation_flip | 0.9471 | 1.0000 | 0.0000 | False | 0.9981 | 4 |
| SWU12 | colgrep | `fuse_max` | ungrounded | negation_flip | 0.9756 | 1.0000 | 0.0000 | False | 1.0000 | 4 |
| SWU12 | colgrep | `fuse_min` | ungrounded | negation_flip | 0.9187 | 1.0000 | 0.0000 | False | 0.9962 | 4 |
| SWG13 | colgrep | `gte_only` | grounded | - | 0.9815 | 1.0000 | 0.0000 | False | 0.9999 | 44 |
| SWG13 | colgrep | `code_only` | grounded | - | 0.9420 | 1.0000 | 0.0000 | False | 0.9936 | 44 |
| SWG13 | colgrep | `fuse_mean` | grounded | - | 0.9617 | 1.0000 | 0.0000 | False | 0.9968 | 44 |
| SWG13 | colgrep | `fuse_max` | grounded | - | 0.9815 | 1.0000 | 0.0000 | False | 0.9999 | 44 |
| SWG13 | colgrep | `fuse_min` | grounded | - | 0.9420 | 1.0000 | 0.0000 | False | 0.9936 | 44 |
| SWU13 | colgrep | `gte_only` | ungrounded | entity_swap | 0.9812 | 1.0000 | 0.0000 | False | 0.9996 | 44 |
| SWU13 | colgrep | `code_only` | ungrounded | entity_swap | 0.9420 | 1.0000 | 0.0000 | False | 0.9838 | 44 |
| SWU13 | colgrep | `fuse_mean` | ungrounded | entity_swap | 0.9616 | 1.0000 | 0.0000 | False | 0.9917 | 44 |
| SWU13 | colgrep | `fuse_max` | ungrounded | entity_swap | 0.9812 | 1.0000 | 0.0000 | False | 0.9996 | 44 |
| SWU13 | colgrep | `fuse_min` | ungrounded | entity_swap | 0.9420 | 1.0000 | 0.0000 | False | 0.9838 | 44 |
| SWG14 | colgrep | `gte_only` | grounded | - | 0.9859 | 1.0000 | 0.0000 | False | 0.9977 | 11 |
| SWG14 | colgrep | `code_only` | grounded | - | 0.9391 | 1.0000 | 0.0000 | False | 0.9925 | 11 |
| SWG14 | colgrep | `fuse_mean` | grounded | - | 0.9625 | 1.0000 | 0.0000 | False | 0.9951 | 11 |
| SWG14 | colgrep | `fuse_max` | grounded | - | 0.9859 | 1.0000 | 0.0000 | False | 0.9977 | 11 |
| SWG14 | colgrep | `fuse_min` | grounded | - | 0.9391 | 1.0000 | 0.0000 | False | 0.9925 | 11 |
| SWU14 | colgrep | `gte_only` | ungrounded | entity_swap | 0.9859 | 1.0000 | 0.0000 | False | 0.9977 | 11 |
| SWU14 | colgrep | `code_only` | ungrounded | entity_swap | 0.9387 | 1.0000 | 0.0000 | False | 0.9921 | 11 |
| SWU14 | colgrep | `fuse_mean` | ungrounded | entity_swap | 0.9623 | 1.0000 | 0.0000 | False | 0.9949 | 11 |
| SWU14 | colgrep | `fuse_max` | ungrounded | entity_swap | 0.9859 | 1.0000 | 0.0000 | False | 0.9977 | 11 |
| SWU14 | colgrep | `fuse_min` | ungrounded | entity_swap | 0.9387 | 1.0000 | 0.0000 | False | 0.9921 | 11 |
| SWG15 | colgrep | `gte_only` | grounded | - | 0.9906 | 1.0000 | 0.0000 | False | 0.9977 | 38 |
| SWG15 | colgrep | `code_only` | grounded | - | 0.9598 | 1.0000 | 0.0000 | False | 0.9890 | 38 |
| SWG15 | colgrep | `fuse_mean` | grounded | - | 0.9752 | 1.0000 | 0.0000 | False | 0.9933 | 38 |
| SWG15 | colgrep | `fuse_max` | grounded | - | 0.9906 | 1.0000 | 0.0000 | False | 0.9977 | 38 |
| SWG15 | colgrep | `fuse_min` | grounded | - | 0.9598 | 1.0000 | 0.0000 | False | 0.9890 | 38 |
| SWU15 | colgrep | `gte_only` | ungrounded | entity_swap | 0.9889 | 1.0000 | 0.0000 | False | 0.9955 | 38 |
| SWU15 | colgrep | `code_only` | ungrounded | entity_swap | 0.9545 | 1.0000 | 0.0000 | False | 0.9890 | 38 |
| SWU15 | colgrep | `fuse_mean` | ungrounded | entity_swap | 0.9717 | 1.0000 | 0.0000 | False | 0.9922 | 38 |
| SWU15 | colgrep | `fuse_max` | ungrounded | entity_swap | 0.9889 | 1.0000 | 0.0000 | False | 0.9955 | 38 |
| SWU15 | colgrep | `fuse_min` | ungrounded | entity_swap | 0.9545 | 1.0000 | 0.0000 | False | 0.9890 | 38 |
