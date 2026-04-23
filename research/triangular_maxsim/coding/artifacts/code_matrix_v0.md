# Coding-Agent Groundedness: Encoder x Chunker Matrix

> NO — the flagship colgrep x LateOn-Code-edge cell did not clear the RAG-style gate: reverse_context AUROC=0.579, phantom-API precision@0.35=0.00, ungrounded minus grounded unused-ratio delta=+0.0000.

- run tag: `code_matrix_v0`
- max SWE-bench instances: `15`
- phantom threshold: `0.35`
- flagship cell (colgrep x LateOn-Code-edge) present: `True`

## Cells evaluated

- `lightonai/GTE-ModernColBERT-v1` x `sentence_packed` (ctx=256, resp=256, encoder limit=300, transport=`local_pylate`)
- `lightonai/GTE-ModernColBERT-v1` x `colgrep` (ctx=256, resp=256, encoder limit=300, transport=`local_pylate`)
- `lightonai/LateOn-Code-edge` x `sentence_packed` (ctx=1024, resp=1024, encoder limit=2048, transport=`local_pylate`)
- `lightonai/LateOn-Code-edge` x `colgrep` (ctx=1024, resp=1024, encoder limit=2048, transport=`local_pylate`)

## Headline matrix

| encoder | chunker | AUROC | grounded cov | cov delta | unused delta | phantom@thr | grounded held-out rate | p95 ms | support units |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `lightonai/GTE-ModernColBERT-v1` | sentence_packed | 0.6058 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 602.45 | 5.1 |
| `lightonai/GTE-ModernColBERT-v1` | colgrep | 0.6243 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 720.29 | 10.5 |
| `lightonai/LateOn-Code-edge` | sentence_packed | 0.5926 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 265.65 | 1.8 |
| `lightonai/LateOn-Code-edge` | colgrep | 0.5794 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 301.92 | 7.3 |

`*` marks cells that pass `reverse_context AUROC >= 0.90`.

## Per-encoder chunker deltas (colgrep - sentence_packed)

| encoder | AUROC delta | coverage-delta delta | unused-delta delta | phantom precision delta |
|---|---:|---:|---:|---:|
| `lightonai/GTE-ModernColBERT-v1` | +0.0185 | +0.0000 | +0.0000 | +0.0000 |
| `lightonai/LateOn-Code-edge` | -0.0132 | +0.0000 | +0.0000 | +0.0000 |

## Cross-encoder deltas (LateOn-Code-edge - GTE-ModernColBERT-v1)

| chunker | AUROC delta | unused-delta delta |
|---|---:|---:|
| sentence_packed | -0.0132 | +0.0000 |
| colgrep | -0.0450 | +0.0000 |

## Subcategory snapshot (per cell)

### `lightonai/GTE-ModernColBERT-v1` x `sentence_packed`

| subcategory | count | reverse_context | coverage | unused ratio | max evidence |
|---|---:|---:|---:|---:|---:|
| entity_swap | 10 | 0.9805 | 1.0000 | 0.0000 | 0.9971 |
| grounded | 18 | 0.9817 | 1.0000 | 0.0000 | 0.9983 |
| negation_flip | 3 | 0.9793 | 1.0000 | 0.0000 | 0.9972 |
| parametric | 2 | 0.9697 | 1.0000 | 0.0000 | 0.9961 |
| partial | 2 | 0.9696 | 1.0000 | 0.0000 | 0.9976 |
| phantom_api | 7 | 0.9789 | 1.0000 | 0.0000 | 0.9988 |

### `lightonai/GTE-ModernColBERT-v1` x `colgrep`

| subcategory | count | reverse_context | coverage | unused ratio | max evidence |
|---|---:|---:|---:|---:|---:|
| entity_swap | 10 | 0.9795 | 1.0000 | 0.0000 | 0.9976 |
| grounded | 18 | 0.9812 | 1.0000 | 0.0000 | 0.9985 |
| negation_flip | 3 | 0.9796 | 1.0000 | 0.0000 | 0.9971 |
| parametric | 2 | 0.9789 | 1.0000 | 0.0000 | 0.9986 |
| partial | 2 | 0.9793 | 1.0000 | 0.0000 | 0.9981 |
| phantom_api | 7 | 0.9798 | 1.0000 | 0.0000 | 0.9992 |

### `lightonai/LateOn-Code-edge` x `sentence_packed`

| subcategory | count | reverse_context | coverage | unused ratio | max evidence |
|---|---:|---:|---:|---:|---:|
| entity_swap | 10 | 0.9364 | 1.0000 | 0.0000 | 0.9871 |
| grounded | 18 | 0.9394 | 1.0000 | 0.0000 | 0.9902 |
| negation_flip | 3 | 0.9197 | 1.0000 | 0.0000 | 0.9825 |
| parametric | 2 | 0.9081 | 1.0000 | 0.0000 | 0.9863 |
| partial | 2 | 0.9193 | 1.0000 | 0.0000 | 0.9981 |
| phantom_api | 7 | 0.9320 | 1.0000 | 0.0000 | 0.9848 |

### `lightonai/LateOn-Code-edge` x `colgrep`

| subcategory | count | reverse_context | coverage | unused ratio | max evidence |
|---|---:|---:|---:|---:|---:|
| entity_swap | 10 | 0.9353 | 1.0000 | 0.0000 | 0.9912 |
| grounded | 18 | 0.9392 | 1.0000 | 0.0000 | 0.9913 |
| negation_flip | 3 | 0.9245 | 1.0000 | 0.0000 | 0.9868 |
| parametric | 2 | 0.9273 | 1.0000 | 0.0000 | 0.9999 |
| partial | 2 | 0.9385 | 1.0000 | 0.0000 | 0.9999 |
| phantom_api | 7 | 0.9304 | 1.0000 | 0.0000 | 0.9927 |

## Held-out support units (flagship cell)

Flagship cell: `lightonai/LateOn-Code-edge` x `colgrep`. IDs with `usage_state == 'unused'` are the support units the core scorer classified as held-out for the response.

| case | label | subcategory | held-out count | uncertain count | held-out ids |
|---|---|---|---:|---:|---|
| CG1 | grounded | - | 0 | 2 | - |
| CG2 | grounded | - | 0 | 4 | - |
| CG3 | grounded | - | 0 | 1 | - |
| CU1 | ungrounded | phantom_api | 0 | 2 | - |
| CU2 | ungrounded | phantom_api | 0 | 3 | - |
| CU3 | ungrounded | entity_swap | 0 | 1 | - |
| CU4 | ungrounded | entity_swap | 0 | 1 | - |
| CU5 | ungrounded | parametric | 0 | 2 | - |
| CU6 | ungrounded | parametric | 0 | 3 | - |
| CA1 | ambiguous | partial | 0 | 1 | - |
| CA2 | ambiguous | partial | 0 | 1 | - |
| CA3 | ambiguous | negation_flip | 0 | 3 | - |
| SWG1 | grounded | - | 0 | 1 | - |
| SWU1 | ungrounded | entity_swap | 0 | 1 | - |
| SWG2 | grounded | - | 0 | 0 | - |
| SWU2 | ungrounded | phantom_api | 0 | 0 | - |
| SWG3 | grounded | - | 0 | 2 | - |
| SWU3 | ungrounded | negation_flip | 0 | 2 | - |
| SWG4 | grounded | - | 0 | 4 | - |
| SWU4 | ungrounded | entity_swap | 0 | 4 | - |
| SWG5 | grounded | - | 0 | 1 | - |
| SWU5 | ungrounded | entity_swap | 0 | 1 | - |
| SWG6 | grounded | - | 0 | 1 | - |
| SWU6 | ungrounded | entity_swap | 0 | 1 | - |
| SWG7 | grounded | - | 0 | 2 | - |
| SWU7 | ungrounded | phantom_api | 0 | 2 | - |
| SWG8 | grounded | - | 0 | 10 | - |
| SWU8 | ungrounded | entity_swap | 0 | 10 | - |
| SWG9 | grounded | - | 0 | 10 | - |
| SWU9 | ungrounded | phantom_api | 0 | 9 | - |
| SWG10 | grounded | - | 0 | 3 | - |
| SWU10 | ungrounded | phantom_api | 0 | 3 | - |
| SWG11 | grounded | - | 0 | 19 | - |
| SWU11 | ungrounded | phantom_api | 0 | 19 | - |
| SWG12 | grounded | - | 0 | 1 | - |
| SWU12 | ungrounded | negation_flip | 0 | 1 | - |
| SWG13 | grounded | - | 0 | 33 | - |
| SWU13 | ungrounded | entity_swap | 0 | 33 | - |
| SWG14 | grounded | - | 0 | 6 | - |
| SWU14 | ungrounded | entity_swap | 0 | 6 | - |
| SWG15 | grounded | - | 0 | 32 | - |
| SWU15 | ungrounded | entity_swap | 0 | 32 | - |

## Per-case scores

| id | encoder | chunker | label | subcategory | reverse_context | coverage | unused | phantom | max evidence | units |
|---|---|---|---|---|---:|---:|---:|---|---:|---:|
| CG1 | `lightonai/GTE-ModernColBERT-v1` | sentence_packed | grounded | - | 0.9792 | 1.0000 | 0.0000 | False | 0.9990 | 1 |
| CG2 | `lightonai/GTE-ModernColBERT-v1` | sentence_packed | grounded | - | 0.9739 | 1.0000 | 0.0000 | False | 0.9994 | 1 |
| CG3 | `lightonai/GTE-ModernColBERT-v1` | sentence_packed | grounded | - | 0.9707 | 1.0000 | 0.0000 | False | 0.9981 | 1 |
| CU1 | `lightonai/GTE-ModernColBERT-v1` | sentence_packed | ungrounded | phantom_api | 0.9731 | 1.0000 | 0.0000 | False | 0.9982 | 1 |
| CU2 | `lightonai/GTE-ModernColBERT-v1` | sentence_packed | ungrounded | phantom_api | 0.9660 | 1.0000 | 0.0000 | False | 0.9975 | 1 |
| CU3 | `lightonai/GTE-ModernColBERT-v1` | sentence_packed | ungrounded | entity_swap | 0.9760 | 1.0000 | 0.0000 | False | 0.9973 | 1 |
| CU4 | `lightonai/GTE-ModernColBERT-v1` | sentence_packed | ungrounded | entity_swap | 0.9702 | 1.0000 | 0.0000 | False | 0.9976 | 1 |
| CU5 | `lightonai/GTE-ModernColBERT-v1` | sentence_packed | ungrounded | parametric | 0.9678 | 1.0000 | 0.0000 | False | 0.9955 | 1 |
| CU6 | `lightonai/GTE-ModernColBERT-v1` | sentence_packed | ungrounded | parametric | 0.9717 | 1.0000 | 0.0000 | False | 0.9967 | 1 |
| CA1 | `lightonai/GTE-ModernColBERT-v1` | sentence_packed | ambiguous | partial | 0.9672 | 1.0000 | 0.0000 | False | 0.9981 | 1 |
| CA2 | `lightonai/GTE-ModernColBERT-v1` | sentence_packed | ambiguous | partial | 0.9721 | 1.0000 | 0.0000 | False | 0.9971 | 1 |
| CA3 | `lightonai/GTE-ModernColBERT-v1` | sentence_packed | ambiguous | negation_flip | 0.9719 | 1.0000 | 0.0000 | False | 0.9966 | 1 |
| SWG1 | `lightonai/GTE-ModernColBERT-v1` | sentence_packed | grounded | - | 0.9754 | 1.0000 | 0.0000 | False | 0.9974 | 3 |
| SWU1 | `lightonai/GTE-ModernColBERT-v1` | sentence_packed | ungrounded | entity_swap | 0.9752 | 1.0000 | 0.0000 | False | 0.9991 | 3 |
| SWG2 | `lightonai/GTE-ModernColBERT-v1` | sentence_packed | grounded | - | 0.9754 | 1.0000 | 0.0000 | False | 0.9999 | 2 |
| SWU2 | `lightonai/GTE-ModernColBERT-v1` | sentence_packed | ungrounded | phantom_api | 0.9742 | 1.0000 | 0.0000 | False | 1.0000 | 2 |
| SWG3 | `lightonai/GTE-ModernColBERT-v1` | sentence_packed | grounded | - | 0.9829 | 1.0000 | 0.0000 | False | 0.9972 | 2 |
| SWU3 | `lightonai/GTE-ModernColBERT-v1` | sentence_packed | ungrounded | negation_flip | 0.9831 | 1.0000 | 0.0000 | False | 0.9971 | 2 |
| SWG4 | `lightonai/GTE-ModernColBERT-v1` | sentence_packed | grounded | - | 0.9840 | 1.0000 | 0.0000 | False | 0.9973 | 3 |
| SWU4 | `lightonai/GTE-ModernColBERT-v1` | sentence_packed | ungrounded | entity_swap | 0.9828 | 1.0000 | 0.0000 | False | 0.9962 | 3 |
| SWG5 | `lightonai/GTE-ModernColBERT-v1` | sentence_packed | grounded | - | 0.9793 | 1.0000 | 0.0000 | False | 0.9976 | 4 |
| SWU5 | `lightonai/GTE-ModernColBERT-v1` | sentence_packed | ungrounded | entity_swap | 0.9793 | 1.0000 | 0.0000 | False | 0.9975 | 4 |
| SWG6 | `lightonai/GTE-ModernColBERT-v1` | sentence_packed | grounded | - | 0.9819 | 1.0000 | 0.0000 | False | 0.9966 | 4 |
| SWU6 | `lightonai/GTE-ModernColBERT-v1` | sentence_packed | ungrounded | entity_swap | 0.9819 | 1.0000 | 0.0000 | False | 0.9966 | 4 |
| SWG7 | `lightonai/GTE-ModernColBERT-v1` | sentence_packed | grounded | - | 0.9871 | 1.0000 | 0.0000 | False | 0.9998 | 5 |
| SWU7 | `lightonai/GTE-ModernColBERT-v1` | sentence_packed | ungrounded | phantom_api | 0.9857 | 1.0000 | 0.0000 | False | 0.9998 | 5 |
| SWG8 | `lightonai/GTE-ModernColBERT-v1` | sentence_packed | grounded | - | 0.9855 | 1.0000 | 0.0000 | False | 0.9982 | 10 |
| SWU8 | `lightonai/GTE-ModernColBERT-v1` | sentence_packed | ungrounded | entity_swap | 0.9852 | 1.0000 | 0.0000 | False | 0.9983 | 10 |
| SWG9 | `lightonai/GTE-ModernColBERT-v1` | sentence_packed | grounded | - | 0.9829 | 1.0000 | 0.0000 | False | 0.9992 | 9 |
| SWU9 | `lightonai/GTE-ModernColBERT-v1` | sentence_packed | ungrounded | phantom_api | 0.9823 | 1.0000 | 0.0000 | False | 0.9992 | 9 |
| SWG10 | `lightonai/GTE-ModernColBERT-v1` | sentence_packed | grounded | - | 0.9868 | 1.0000 | 0.0000 | False | 0.9997 | 6 |
| SWU10 | `lightonai/GTE-ModernColBERT-v1` | sentence_packed | ungrounded | phantom_api | 0.9855 | 1.0000 | 0.0000 | False | 0.9997 | 6 |
| SWG11 | `lightonai/GTE-ModernColBERT-v1` | sentence_packed | grounded | - | 0.9868 | 1.0000 | 0.0000 | False | 0.9981 | 18 |
| SWU11 | `lightonai/GTE-ModernColBERT-v1` | sentence_packed | ungrounded | phantom_api | 0.9859 | 1.0000 | 0.0000 | False | 0.9969 | 18 |
| SWG12 | `lightonai/GTE-ModernColBERT-v1` | sentence_packed | grounded | - | 0.9834 | 1.0000 | 0.0000 | False | 0.9977 | 4 |
| SWU12 | `lightonai/GTE-ModernColBERT-v1` | sentence_packed | ungrounded | negation_flip | 0.9830 | 1.0000 | 0.0000 | False | 0.9978 | 4 |
| SWG13 | `lightonai/GTE-ModernColBERT-v1` | sentence_packed | grounded | - | 0.9817 | 1.0000 | 0.0000 | False | 0.9998 | 15 |
| SWU13 | `lightonai/GTE-ModernColBERT-v1` | sentence_packed | ungrounded | entity_swap | 0.9818 | 1.0000 | 0.0000 | False | 0.9977 | 15 |
| SWG14 | `lightonai/GTE-ModernColBERT-v1` | sentence_packed | grounded | - | 0.9846 | 1.0000 | 0.0000 | False | 0.9969 | 5 |
| SWU14 | `lightonai/GTE-ModernColBERT-v1` | sentence_packed | ungrounded | entity_swap | 0.9846 | 1.0000 | 0.0000 | False | 0.9969 | 5 |
| SWG15 | `lightonai/GTE-ModernColBERT-v1` | sentence_packed | grounded | - | 0.9888 | 1.0000 | 0.0000 | False | 0.9974 | 11 |
| SWU15 | `lightonai/GTE-ModernColBERT-v1` | sentence_packed | ungrounded | entity_swap | 0.9878 | 1.0000 | 0.0000 | False | 0.9942 | 11 |
| CG1 | `lightonai/GTE-ModernColBERT-v1` | colgrep | grounded | - | 0.9866 | 1.0000 | 0.0000 | False | 0.9995 | 4 |
| CG2 | `lightonai/GTE-ModernColBERT-v1` | colgrep | grounded | - | 0.9829 | 1.0000 | 0.0000 | False | 0.9994 | 5 |
| CG3 | `lightonai/GTE-ModernColBERT-v1` | colgrep | grounded | - | 0.9854 | 1.0000 | 0.0000 | False | 0.9982 | 2 |
| CU1 | `lightonai/GTE-ModernColBERT-v1` | colgrep | ungrounded | phantom_api | 0.9820 | 1.0000 | 0.0000 | False | 0.9998 | 2 |
| CU2 | `lightonai/GTE-ModernColBERT-v1` | colgrep | ungrounded | phantom_api | 0.9752 | 1.0000 | 0.0000 | False | 0.9976 | 4 |
| CU3 | `lightonai/GTE-ModernColBERT-v1` | colgrep | ungrounded | entity_swap | 0.9828 | 1.0000 | 0.0000 | False | 0.9979 | 2 |
| CU4 | `lightonai/GTE-ModernColBERT-v1` | colgrep | ungrounded | entity_swap | 0.9758 | 1.0000 | 0.0000 | False | 0.9973 | 2 |
| CU5 | `lightonai/GTE-ModernColBERT-v1` | colgrep | ungrounded | parametric | 0.9795 | 1.0000 | 0.0000 | False | 0.9996 | 3 |
| CU6 | `lightonai/GTE-ModernColBERT-v1` | colgrep | ungrounded | parametric | 0.9783 | 1.0000 | 0.0000 | False | 0.9975 | 4 |
| CA1 | `lightonai/GTE-ModernColBERT-v1` | colgrep | ambiguous | partial | 0.9818 | 1.0000 | 0.0000 | False | 0.9985 | 2 |
| CA2 | `lightonai/GTE-ModernColBERT-v1` | colgrep | ambiguous | partial | 0.9768 | 1.0000 | 0.0000 | False | 0.9977 | 2 |
| CA3 | `lightonai/GTE-ModernColBERT-v1` | colgrep | ambiguous | negation_flip | 0.9806 | 1.0000 | 0.0000 | False | 0.9977 | 3 |
| SWG1 | `lightonai/GTE-ModernColBERT-v1` | colgrep | grounded | - | 0.9617 | 1.0000 | 0.0000 | False | 0.9976 | 3 |
| SWU1 | `lightonai/GTE-ModernColBERT-v1` | colgrep | ungrounded | entity_swap | 0.9614 | 1.0000 | 0.0000 | False | 0.9963 | 3 |
| SWG2 | `lightonai/GTE-ModernColBERT-v1` | colgrep | grounded | - | 0.9633 | 1.0000 | 0.0000 | False | 1.0000 | 2 |
| SWU2 | `lightonai/GTE-ModernColBERT-v1` | colgrep | ungrounded | phantom_api | 0.9620 | 1.0000 | 0.0000 | False | 0.9999 | 2 |
| SWG3 | `lightonai/GTE-ModernColBERT-v1` | colgrep | grounded | - | 0.9829 | 1.0000 | 0.0000 | False | 0.9974 | 4 |
| SWU3 | `lightonai/GTE-ModernColBERT-v1` | colgrep | ungrounded | negation_flip | 0.9827 | 1.0000 | 0.0000 | False | 0.9974 | 4 |
| SWG4 | `lightonai/GTE-ModernColBERT-v1` | colgrep | grounded | - | 0.9872 | 1.0000 | 0.0000 | False | 0.9972 | 8 |
| SWU4 | `lightonai/GTE-ModernColBERT-v1` | colgrep | ungrounded | entity_swap | 0.9859 | 1.0000 | 0.0000 | False | 0.9974 | 8 |
| SWG5 | `lightonai/GTE-ModernColBERT-v1` | colgrep | grounded | - | 0.9785 | 1.0000 | 0.0000 | False | 0.9982 | 4 |
| SWU5 | `lightonai/GTE-ModernColBERT-v1` | colgrep | ungrounded | entity_swap | 0.9784 | 1.0000 | 0.0000 | False | 0.9981 | 4 |
| SWG6 | `lightonai/GTE-ModernColBERT-v1` | colgrep | grounded | - | 0.9690 | 1.0000 | 0.0000 | False | 0.9981 | 4 |
| SWU6 | `lightonai/GTE-ModernColBERT-v1` | colgrep | ungrounded | entity_swap | 0.9689 | 1.0000 | 0.0000 | False | 0.9981 | 4 |
| SWG7 | `lightonai/GTE-ModernColBERT-v1` | colgrep | grounded | - | 0.9860 | 1.0000 | 0.0000 | False | 0.9998 | 6 |
| SWU7 | `lightonai/GTE-ModernColBERT-v1` | colgrep | ungrounded | phantom_api | 0.9846 | 1.0000 | 0.0000 | False | 0.9999 | 6 |
| SWG8 | `lightonai/GTE-ModernColBERT-v1` | colgrep | grounded | - | 0.9859 | 1.0000 | 0.0000 | False | 0.9982 | 17 |
| SWU8 | `lightonai/GTE-ModernColBERT-v1` | colgrep | ungrounded | entity_swap | 0.9856 | 1.0000 | 0.0000 | False | 0.9983 | 17 |
| SWG9 | `lightonai/GTE-ModernColBERT-v1` | colgrep | grounded | - | 0.9820 | 1.0000 | 0.0000 | False | 0.9992 | 17 |
| SWU9 | `lightonai/GTE-ModernColBERT-v1` | colgrep | ungrounded | phantom_api | 0.9811 | 1.0000 | 0.0000 | False | 0.9992 | 17 |
| SWG10 | `lightonai/GTE-ModernColBERT-v1` | colgrep | grounded | - | 0.9884 | 1.0000 | 0.0000 | False | 0.9999 | 10 |
| SWU10 | `lightonai/GTE-ModernColBERT-v1` | colgrep | ungrounded | phantom_api | 0.9875 | 1.0000 | 0.0000 | False | 0.9999 | 10 |
| SWG11 | `lightonai/GTE-ModernColBERT-v1` | colgrep | grounded | - | 0.9876 | 1.0000 | 0.0000 | False | 0.9987 | 32 |
| SWU11 | `lightonai/GTE-ModernColBERT-v1` | colgrep | ungrounded | phantom_api | 0.9864 | 1.0000 | 0.0000 | False | 0.9976 | 32 |
| SWG12 | `lightonai/GTE-ModernColBERT-v1` | colgrep | grounded | - | 0.9760 | 1.0000 | 0.0000 | False | 0.9964 | 4 |
| SWU12 | `lightonai/GTE-ModernColBERT-v1` | colgrep | ungrounded | negation_flip | 0.9756 | 1.0000 | 0.0000 | False | 0.9962 | 4 |
| SWG13 | `lightonai/GTE-ModernColBERT-v1` | colgrep | grounded | - | 0.9815 | 1.0000 | 0.0000 | False | 0.9999 | 44 |
| SWU13 | `lightonai/GTE-ModernColBERT-v1` | colgrep | ungrounded | entity_swap | 0.9812 | 1.0000 | 0.0000 | False | 0.9996 | 44 |
| SWG14 | `lightonai/GTE-ModernColBERT-v1` | colgrep | grounded | - | 0.9859 | 1.0000 | 0.0000 | False | 0.9977 | 11 |
| SWU14 | `lightonai/GTE-ModernColBERT-v1` | colgrep | ungrounded | entity_swap | 0.9859 | 1.0000 | 0.0000 | False | 0.9977 | 11 |
| SWG15 | `lightonai/GTE-ModernColBERT-v1` | colgrep | grounded | - | 0.9906 | 1.0000 | 0.0000 | False | 0.9977 | 38 |
| SWU15 | `lightonai/GTE-ModernColBERT-v1` | colgrep | ungrounded | entity_swap | 0.9889 | 1.0000 | 0.0000 | False | 0.9955 | 38 |
| CG1 | `lightonai/LateOn-Code-edge` | sentence_packed | grounded | - | 0.9246 | 1.0000 | 0.0000 | False | 1.0000 | 1 |
| CG2 | `lightonai/LateOn-Code-edge` | sentence_packed | grounded | - | 0.9304 | 1.0000 | 0.0000 | False | 0.9910 | 1 |
| CG3 | `lightonai/LateOn-Code-edge` | sentence_packed | grounded | - | 0.8725 | 1.0000 | 0.0000 | False | 1.0000 | 1 |
| CU1 | `lightonai/LateOn-Code-edge` | sentence_packed | ungrounded | phantom_api | 0.9088 | 1.0000 | 0.0000 | False | 0.9650 | 1 |
| CU2 | `lightonai/LateOn-Code-edge` | sentence_packed | ungrounded | phantom_api | 0.8696 | 1.0000 | 0.0000 | False | 0.9833 | 1 |
| CU3 | `lightonai/LateOn-Code-edge` | sentence_packed | ungrounded | entity_swap | 0.9264 | 1.0000 | 0.0000 | False | 1.0000 | 1 |
| CU4 | `lightonai/LateOn-Code-edge` | sentence_packed | ungrounded | entity_swap | 0.8886 | 1.0000 | 0.0000 | False | 0.9863 | 1 |
| CU5 | `lightonai/LateOn-Code-edge` | sentence_packed | ungrounded | parametric | 0.9016 | 1.0000 | 0.0000 | False | 0.9726 | 1 |
| CU6 | `lightonai/LateOn-Code-edge` | sentence_packed | ungrounded | parametric | 0.9146 | 1.0000 | 0.0000 | False | 1.0000 | 1 |
| CA1 | `lightonai/LateOn-Code-edge` | sentence_packed | ambiguous | partial | 0.9100 | 1.0000 | 0.0000 | False | 0.9961 | 1 |
| CA2 | `lightonai/LateOn-Code-edge` | sentence_packed | ambiguous | partial | 0.9286 | 1.0000 | 0.0000 | False | 1.0000 | 1 |
| CA3 | `lightonai/LateOn-Code-edge` | sentence_packed | ambiguous | negation_flip | 0.8871 | 1.0000 | 0.0000 | False | 0.9861 | 1 |
| SWG1 | `lightonai/LateOn-Code-edge` | sentence_packed | grounded | - | 0.9139 | 1.0000 | 0.0000 | False | 0.9765 | 1 |
| SWU1 | `lightonai/LateOn-Code-edge` | sentence_packed | ungrounded | entity_swap | 0.9157 | 1.0000 | 0.0000 | False | 0.9703 | 1 |
| SWG2 | `lightonai/LateOn-Code-edge` | sentence_packed | grounded | - | 0.9223 | 1.0000 | 0.0000 | False | 0.9809 | 1 |
| SWU2 | `lightonai/LateOn-Code-edge` | sentence_packed | ungrounded | phantom_api | 0.9233 | 1.0000 | 0.0000 | False | 0.9731 | 1 |
| SWG3 | `lightonai/LateOn-Code-edge` | sentence_packed | grounded | - | 0.9352 | 1.0000 | 0.0000 | False | 0.9829 | 1 |
| SWU3 | `lightonai/LateOn-Code-edge` | sentence_packed | ungrounded | negation_flip | 0.9326 | 1.0000 | 0.0000 | False | 0.9657 | 1 |
| SWG4 | `lightonai/LateOn-Code-edge` | sentence_packed | grounded | - | 0.9496 | 1.0000 | 0.0000 | False | 0.9966 | 1 |
| SWU4 | `lightonai/LateOn-Code-edge` | sentence_packed | ungrounded | entity_swap | 0.9458 | 1.0000 | 0.0000 | False | 0.9960 | 1 |
| SWG5 | `lightonai/LateOn-Code-edge` | sentence_packed | grounded | - | 0.9434 | 1.0000 | 0.0000 | False | 0.9874 | 1 |
| SWU5 | `lightonai/LateOn-Code-edge` | sentence_packed | ungrounded | entity_swap | 0.9466 | 1.0000 | 0.0000 | False | 0.9862 | 1 |
| SWG6 | `lightonai/LateOn-Code-edge` | sentence_packed | grounded | - | 0.9274 | 1.0000 | 0.0000 | False | 0.9662 | 1 |
| SWU6 | `lightonai/LateOn-Code-edge` | sentence_packed | ungrounded | entity_swap | 0.9256 | 1.0000 | 0.0000 | False | 0.9655 | 1 |
| SWG7 | `lightonai/LateOn-Code-edge` | sentence_packed | grounded | - | 0.9541 | 1.0000 | 0.0000 | False | 0.9882 | 2 |
| SWU7 | `lightonai/LateOn-Code-edge` | sentence_packed | ungrounded | phantom_api | 0.9497 | 1.0000 | 0.0000 | False | 0.9933 | 2 |
| SWG8 | `lightonai/LateOn-Code-edge` | sentence_packed | grounded | - | 0.9676 | 1.0000 | 0.0000 | False | 0.9991 | 3 |
| SWU8 | `lightonai/LateOn-Code-edge` | sentence_packed | ungrounded | entity_swap | 0.9670 | 1.0000 | 0.0000 | False | 0.9990 | 3 |
| SWG9 | `lightonai/LateOn-Code-edge` | sentence_packed | grounded | - | 0.9499 | 1.0000 | 0.0000 | False | 0.9933 | 3 |
| SWU9 | `lightonai/LateOn-Code-edge` | sentence_packed | ungrounded | phantom_api | 0.9483 | 1.0000 | 0.0000 | False | 0.9894 | 3 |
| SWG10 | `lightonai/LateOn-Code-edge` | sentence_packed | grounded | - | 0.9691 | 1.0000 | 0.0000 | False | 0.9977 | 2 |
| SWU10 | `lightonai/LateOn-Code-edge` | sentence_packed | ungrounded | phantom_api | 0.9679 | 1.0000 | 0.0000 | False | 0.9978 | 2 |
| SWG11 | `lightonai/LateOn-Code-edge` | sentence_packed | grounded | - | 0.9571 | 1.0000 | 0.0000 | False | 0.9927 | 5 |
| SWU11 | `lightonai/LateOn-Code-edge` | sentence_packed | ungrounded | phantom_api | 0.9567 | 1.0000 | 0.0000 | False | 0.9917 | 5 |
| SWG12 | `lightonai/LateOn-Code-edge` | sentence_packed | grounded | - | 0.9406 | 1.0000 | 0.0000 | False | 1.0000 | 1 |
| SWU12 | `lightonai/LateOn-Code-edge` | sentence_packed | ungrounded | negation_flip | 0.9395 | 1.0000 | 0.0000 | False | 0.9957 | 1 |
| SWG13 | `lightonai/LateOn-Code-edge` | sentence_packed | grounded | - | 0.9491 | 1.0000 | 0.0000 | False | 0.9936 | 4 |
| SWU13 | `lightonai/LateOn-Code-edge` | sentence_packed | ungrounded | entity_swap | 0.9533 | 1.0000 | 0.0000 | False | 0.9929 | 4 |
| SWG14 | `lightonai/LateOn-Code-edge` | sentence_packed | grounded | - | 0.9432 | 1.0000 | 0.0000 | False | 0.9861 | 2 |
| SWU14 | `lightonai/LateOn-Code-edge` | sentence_packed | ungrounded | entity_swap | 0.9429 | 1.0000 | 0.0000 | False | 0.9861 | 2 |
| SWG15 | `lightonai/LateOn-Code-edge` | sentence_packed | grounded | - | 0.9588 | 1.0000 | 0.0000 | False | 0.9915 | 3 |
| SWU15 | `lightonai/LateOn-Code-edge` | sentence_packed | ungrounded | entity_swap | 0.9526 | 1.0000 | 0.0000 | False | 0.9888 | 3 |
| CG1 | `lightonai/LateOn-Code-edge` | colgrep | grounded | - | 0.9384 | 1.0000 | 0.0000 | False | 1.0000 | 4 |
| CG2 | `lightonai/LateOn-Code-edge` | colgrep | grounded | - | 0.9461 | 1.0000 | 0.0000 | False | 0.9999 | 5 |
| CG3 | `lightonai/LateOn-Code-edge` | colgrep | grounded | - | 0.9423 | 1.0000 | 0.0000 | False | 1.0000 | 2 |
| CU1 | `lightonai/LateOn-Code-edge` | colgrep | ungrounded | phantom_api | 0.9360 | 1.0000 | 0.0000 | False | 0.9998 | 2 |
| CU2 | `lightonai/LateOn-Code-edge` | colgrep | ungrounded | phantom_api | 0.8665 | 1.0000 | 0.0000 | False | 0.9998 | 4 |
| CU3 | `lightonai/LateOn-Code-edge` | colgrep | ungrounded | entity_swap | 0.9478 | 1.0000 | 0.0000 | False | 1.0000 | 2 |
| CU4 | `lightonai/LateOn-Code-edge` | colgrep | ungrounded | entity_swap | 0.9047 | 1.0000 | 0.0000 | False | 0.9999 | 2 |
| CU5 | `lightonai/LateOn-Code-edge` | colgrep | ungrounded | parametric | 0.9230 | 1.0000 | 0.0000 | False | 0.9998 | 3 |
| CU6 | `lightonai/LateOn-Code-edge` | colgrep | ungrounded | parametric | 0.9316 | 1.0000 | 0.0000 | False | 1.0000 | 4 |
| CA1 | `lightonai/LateOn-Code-edge` | colgrep | ambiguous | partial | 0.9427 | 1.0000 | 0.0000 | False | 0.9999 | 2 |
| CA2 | `lightonai/LateOn-Code-edge` | colgrep | ambiguous | partial | 0.9342 | 1.0000 | 0.0000 | False | 0.9999 | 2 |
| CA3 | `lightonai/LateOn-Code-edge` | colgrep | ambiguous | negation_flip | 0.9217 | 1.0000 | 0.0000 | False | 0.9999 | 3 |
| SWG1 | `lightonai/LateOn-Code-edge` | colgrep | grounded | - | 0.8752 | 1.0000 | 0.0000 | False | 0.9777 | 1 |
| SWU1 | `lightonai/LateOn-Code-edge` | colgrep | ungrounded | entity_swap | 0.8759 | 1.0000 | 0.0000 | False | 0.9800 | 1 |
| SWG2 | `lightonai/LateOn-Code-edge` | colgrep | grounded | - | 0.8892 | 1.0000 | 0.0000 | False | 0.9804 | 1 |
| SWU2 | `lightonai/LateOn-Code-edge` | colgrep | ungrounded | phantom_api | 0.8877 | 1.0000 | 0.0000 | False | 0.9735 | 1 |
| SWG3 | `lightonai/LateOn-Code-edge` | colgrep | grounded | - | 0.9387 | 1.0000 | 0.0000 | False | 0.9860 | 2 |
| SWU3 | `lightonai/LateOn-Code-edge` | colgrep | ungrounded | negation_flip | 0.9329 | 1.0000 | 0.0000 | False | 0.9648 | 2 |
| SWG4 | `lightonai/LateOn-Code-edge` | colgrep | grounded | - | 0.9566 | 1.0000 | 0.0000 | False | 0.9967 | 6 |
| SWU4 | `lightonai/LateOn-Code-edge` | colgrep | ungrounded | entity_swap | 0.9524 | 1.0000 | 0.0000 | False | 0.9978 | 6 |
| SWG5 | `lightonai/LateOn-Code-edge` | colgrep | grounded | - | 0.9420 | 1.0000 | 0.0000 | False | 0.9967 | 2 |
| SWU5 | `lightonai/LateOn-Code-edge` | colgrep | ungrounded | entity_swap | 0.9469 | 1.0000 | 0.0000 | False | 0.9963 | 2 |
| SWG6 | `lightonai/LateOn-Code-edge` | colgrep | grounded | - | 0.8940 | 1.0000 | 0.0000 | False | 0.9658 | 1 |
| SWU6 | `lightonai/LateOn-Code-edge` | colgrep | ungrounded | entity_swap | 0.8942 | 1.0000 | 0.0000 | False | 0.9651 | 1 |
| SWG7 | `lightonai/LateOn-Code-edge` | colgrep | grounded | - | 0.9520 | 1.0000 | 0.0000 | False | 0.9877 | 2 |
| SWU7 | `lightonai/LateOn-Code-edge` | colgrep | ungrounded | phantom_api | 0.9486 | 1.0000 | 0.0000 | False | 0.9945 | 2 |
| SWG8 | `lightonai/LateOn-Code-edge` | colgrep | grounded | - | 0.9669 | 1.0000 | 0.0000 | False | 0.9982 | 11 |
| SWU8 | `lightonai/LateOn-Code-edge` | colgrep | ungrounded | entity_swap | 0.9663 | 1.0000 | 0.0000 | False | 0.9983 | 11 |
| SWG9 | `lightonai/LateOn-Code-edge` | colgrep | grounded | - | 0.9515 | 1.0000 | 0.0000 | False | 0.9900 | 10 |
| SWU9 | `lightonai/LateOn-Code-edge` | colgrep | ungrounded | phantom_api | 0.9461 | 1.0000 | 0.0000 | False | 0.9915 | 10 |
| SWG10 | `lightonai/LateOn-Code-edge` | colgrep | grounded | - | 0.9699 | 1.0000 | 0.0000 | False | 0.9936 | 5 |
| SWU10 | `lightonai/LateOn-Code-edge` | colgrep | ungrounded | phantom_api | 0.9690 | 1.0000 | 0.0000 | False | 0.9937 | 5 |
| SWG11 | `lightonai/LateOn-Code-edge` | colgrep | grounded | - | 0.9593 | 1.0000 | 0.0000 | False | 0.9965 | 20 |
| SWU11 | `lightonai/LateOn-Code-edge` | colgrep | ungrounded | phantom_api | 0.9587 | 1.0000 | 0.0000 | False | 0.9961 | 20 |
| SWG12 | `lightonai/LateOn-Code-edge` | colgrep | grounded | - | 0.9195 | 1.0000 | 0.0000 | False | 1.0000 | 1 |
| SWU12 | `lightonai/LateOn-Code-edge` | colgrep | ungrounded | negation_flip | 0.9190 | 1.0000 | 0.0000 | False | 0.9957 | 1 |
| SWG13 | `lightonai/LateOn-Code-edge` | colgrep | grounded | - | 0.9490 | 1.0000 | 0.0000 | False | 0.9931 | 34 |
| SWU13 | `lightonai/LateOn-Code-edge` | colgrep | ungrounded | entity_swap | 0.9543 | 1.0000 | 0.0000 | False | 0.9912 | 34 |
| SWG14 | `lightonai/LateOn-Code-edge` | colgrep | grounded | - | 0.9524 | 1.0000 | 0.0000 | False | 0.9912 | 7 |
| SWU14 | `lightonai/LateOn-Code-edge` | colgrep | ungrounded | entity_swap | 0.9520 | 1.0000 | 0.0000 | False | 0.9912 | 7 |
| SWG15 | `lightonai/LateOn-Code-edge` | colgrep | grounded | - | 0.9627 | 1.0000 | 0.0000 | False | 0.9901 | 33 |
| SWU15 | `lightonai/LateOn-Code-edge` | colgrep | ungrounded | entity_swap | 0.9590 | 1.0000 | 0.0000 | False | 0.9919 | 33 |
