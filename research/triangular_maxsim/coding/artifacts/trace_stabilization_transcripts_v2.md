# Coding-Agent Groundedness: Scorer Stack x Chunker

> NO — the flagship colgrep x fuse_mean cell did not clear the RAG-style gate: reverse_context AUROC=0.697, phantom-API precision@0.35=n/a, ungrounded minus grounded unused-ratio delta=+0.0000. Stacking lift vs gte_only=-0.0644. Best fusion rule on colgrep is `fuse_max` (AUROC=0.761); consider flipping the flagship.

- run tag: `trace_stabilization_transcripts_v2`
- case bank: `transcripts_v2`
- primary scorer: `lightonai/GTE-ModernColBERT-v1`
- orthogonal scorer: `lightonai/LateOn-Code-edge` (available: `True`)
- shared chunk budget: `256` tokens (GTE tokenizer)
- max SWE-bench instances: `15` (ignored for `transcripts_v1`)
- phantom threshold: `0.35`
- flagship cell: `colgrep` x `fuse_mean` (present: `True`)

## Headline matrix

| chunker | scorer | AUROC | grounded cov | cov delta | unused delta | phantom@thr | grounded held-out rate | p95 ms | support units |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sentence_packed | `gte_only` | 0.7917 | 1.0000 | 0.0000 | 0.0000 | n/a | 0.0000 | 2606.78 | 108.3 |
| sentence_packed | `code_only` | 0.6689 | 1.0000 | 0.0000 | 0.0000 | n/a | 0.0000 | 2398.62 | 108.3 |
| sentence_packed | `fuse_mean` | 0.7128 | 1.0000 | 0.0000 | 0.0000 | n/a | 0.0000 | 2593.10 | 108.3 |
| sentence_packed | `fuse_max` | 0.7917 | 1.0000 | 0.0000 | 0.0000 | n/a | 0.0000 | 2593.10 | 108.3 |
| sentence_packed | `fuse_min` | 0.6689 | 1.0000 | 0.0000 | 0.0000 | n/a | 0.0000 | 2593.10 | 108.3 |
| colgrep | `gte_only` | 0.7614 | 1.0000 | 0.0000 | 0.0000 | n/a | 0.0000 | 2019.17 | 91.7 |
| colgrep | `code_only` | 0.6508 | 1.0000 | 0.0000 | 0.0000 | n/a | 0.0000 | 1899.35 | 91.7 |
| colgrep | `fuse_mean` (flagship) | 0.6969 | 1.0000 | 0.0000 | 0.0000 | n/a | 0.0000 | 1993.57 | 91.7 |
| colgrep | `fuse_max` | 0.7614 | 1.0000 | 0.0000 | 0.0000 | n/a | 0.0000 | 1993.57 | 91.7 |
| colgrep | `fuse_min` | 0.6508 | 1.0000 | 0.0000 | 0.0000 | n/a | 0.0000 | 1993.57 | 91.7 |

`*` marks cells that pass `reverse_context AUROC >= 0.90`.

## Score distribution diagnostics (reverse_context)

| chunker | scorer | grounded mean | grounded p10-p90 | ungrounded mean | ungrounded p10-p90 | separation | saturation@0.95 |
|---|---|---:|---|---:|---|---:|---:|
| sentence_packed | `gte_only` | 0.9780 | 0.9723-0.9850 | 0.9712 | 0.9638-0.9811 | +0.0067 | 0.9917 |
| sentence_packed | `code_only` | 0.9357 | 0.9152-0.9535 | 0.9272 | 0.9069-0.9459 | +0.0086 | 0.1417 |
| sentence_packed | `fuse_mean` | 0.9569 | 0.9438-0.9690 | 0.9492 | 0.9360-0.9627 | +0.0076 | 0.6667 |
| sentence_packed | `fuse_max` | 0.9780 | 0.9723-0.9850 | 0.9712 | 0.9638-0.9811 | +0.0067 | 0.9917 |
| sentence_packed | `fuse_min` | 0.9357 | 0.9152-0.9535 | 0.9272 | 0.9069-0.9459 | +0.0086 | 0.1417 |
| colgrep | `gte_only` | 0.9760 | 0.9657-0.9840 | 0.9692 | 0.9570-0.9791 | +0.0068 | 0.9833 |
| colgrep | `code_only` | 0.9308 | 0.9117-0.9521 | 0.9224 | 0.9034-0.9434 | +0.0084 | 0.1000 |
| colgrep | `fuse_mean` | 0.9534 | 0.9402-0.9680 | 0.9458 | 0.9314-0.9581 | +0.0076 | 0.5167 |
| colgrep | `fuse_max` | 0.9760 | 0.9657-0.9840 | 0.9692 | 0.9570-0.9791 | +0.0068 | 0.9833 |
| colgrep | `fuse_min` | 0.9308 | 0.9117-0.9521 | 0.9224 | 0.9034-0.9434 | +0.0084 | 0.1000 |

`separation` is mean(grounded) − mean(ungrounded) on `reverse_context`. `saturation@0.95` is the fraction of anchor cases (grounded + ungrounded) whose score ≥ 0.95 — a ceiling effect indicator.

## Tier metrics (transcripts_v1: correct / ambiguous / wrong)

`AUROC c-vs-w` = AUROC on `tier=correct` vs `tier=wrong`. `AUROC c-vs-a` = AUROC on `correct` vs `ambiguous` (hard bucket). `monotonicity` = fraction of base scenarios where `correct > ambiguous > wrong` holds on `reverse_context`. `partial mono` = fraction where `correct > wrong` holds (ignores ambiguous).

| chunker | scorer | AUROC c-vs-w | AUROC c-vs-a | AUROC c-vs-wa | monotonicity | partial mono | correct mean | ambiguous mean | wrong mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sentence_packed | `gte_only` | 0.7917 | 0.5503 | 0.6710 | 0.7500 | 0.9667 | 0.9780 | 0.9771 | 0.9712 |
| sentence_packed | `code_only` | 0.6689 | 0.5306 | 0.5997 | 0.5000 | 0.8833 | 0.9357 | 0.9344 | 0.9272 |
| sentence_packed | `fuse_mean` | 0.7128 | 0.5447 | 0.6288 | 0.5667 | 0.9333 | 0.9569 | 0.9557 | 0.9492 |
| sentence_packed | `fuse_max` | 0.7917 | 0.5503 | 0.6710 | 0.7500 | 0.9667 | 0.9780 | 0.9771 | 0.9712 |
| sentence_packed | `fuse_min` | 0.6689 | 0.5306 | 0.5997 | 0.5000 | 0.8833 | 0.9357 | 0.9344 | 0.9272 |
| colgrep | `gte_only` | 0.7614 | 0.5494 | 0.6554 | 0.7667 | 0.9500 | 0.9760 | 0.9750 | 0.9692 |
| colgrep | `code_only` | 0.6508 | 0.5314 | 0.5911 | 0.5000 | 0.8500 | 0.9308 | 0.9294 | 0.9224 |
| colgrep | `fuse_mean` | 0.6969 | 0.5408 | 0.6189 | 0.5667 | 0.9333 | 0.9534 | 0.9522 | 0.9458 |
| colgrep | `fuse_max` | 0.7614 | 0.5494 | 0.6554 | 0.7667 | 0.9500 | 0.9760 | 0.9750 | 0.9692 |
| colgrep | `fuse_min` | 0.6508 | 0.5314 | 0.5911 | 0.5000 | 0.8500 | 0.9308 | 0.9294 | 0.9224 |

## Stacking lift (fuse_* minus gte_only, per chunker)

| chunker | rule | AUROC lift | unused-delta lift | phantom precision lift |
|---|---|---:|---:|---:|
| sentence_packed | fuse_mean | -0.0789 | +0.0000 | n/a |
| sentence_packed | fuse_max | +0.0000 | +0.0000 | n/a |
| sentence_packed | fuse_min | -0.1228 | +0.0000 | n/a |
| colgrep | fuse_mean | -0.0644 | +0.0000 | n/a |
| colgrep | fuse_max | +0.0000 | +0.0000 | n/a |
| colgrep | fuse_min | -0.1106 | +0.0000 | n/a |

## Chunker lift (colgrep minus sentence_packed, per scorer)

| scorer | AUROC lift | unused-delta lift | phantom precision lift |
|---|---:|---:|---:|
| `gte_only` | -0.0303 | +0.0000 | n/a |
| `code_only` | -0.0181 | +0.0000 | n/a |
| `fuse_mean` | -0.0158 | +0.0000 | n/a |
| `fuse_max` | -0.0303 | +0.0000 | n/a |
| `fuse_min` | -0.0181 | +0.0000 | n/a |

## Subcategory snapshot (per cell)

### `sentence_packed` x `gte_only`

| subcategory | count | reverse_context | coverage | unused ratio | max evidence |
|---|---:|---:|---:|---:|---:|
| ambiguous | 60 | 0.9771 | 1.0000 | 0.0000 | 0.9974 |
| correct | 60 | 0.9780 | 1.0000 | 0.0000 | 0.9979 |
| wrong | 60 | 0.9712 | 1.0000 | 0.0000 | 0.9979 |

### `sentence_packed` x `code_only`

| subcategory | count | reverse_context | coverage | unused ratio | max evidence |
|---|---:|---:|---:|---:|---:|
| ambiguous | 60 | 0.9344 | 1.0000 | 0.0000 | 0.9894 |
| correct | 60 | 0.9357 | 1.0000 | 0.0000 | 0.9913 |
| wrong | 60 | 0.9272 | 1.0000 | 0.0000 | 0.9917 |

### `sentence_packed` x `fuse_mean`

| subcategory | count | reverse_context | coverage | unused ratio | max evidence |
|---|---:|---:|---:|---:|---:|
| ambiguous | 60 | 0.9557 | 1.0000 | 0.0000 | 0.9934 |
| correct | 60 | 0.9569 | 1.0000 | 0.0000 | 0.9946 |
| wrong | 60 | 0.9492 | 1.0000 | 0.0000 | 0.9948 |

### `sentence_packed` x `fuse_max`

| subcategory | count | reverse_context | coverage | unused ratio | max evidence |
|---|---:|---:|---:|---:|---:|
| ambiguous | 60 | 0.9771 | 1.0000 | 0.0000 | 0.9978 |
| correct | 60 | 0.9780 | 1.0000 | 0.0000 | 0.9983 |
| wrong | 60 | 0.9712 | 1.0000 | 0.0000 | 0.9983 |

### `sentence_packed` x `fuse_min`

| subcategory | count | reverse_context | coverage | unused ratio | max evidence |
|---|---:|---:|---:|---:|---:|
| ambiguous | 60 | 0.9344 | 1.0000 | 0.0000 | 0.9890 |
| correct | 60 | 0.9357 | 1.0000 | 0.0000 | 0.9909 |
| wrong | 60 | 0.9272 | 1.0000 | 0.0000 | 0.9913 |

### `colgrep` x `gte_only`

| subcategory | count | reverse_context | coverage | unused ratio | max evidence |
|---|---:|---:|---:|---:|---:|
| ambiguous | 60 | 0.9750 | 1.0000 | 0.0000 | 0.9972 |
| correct | 60 | 0.9760 | 1.0000 | 0.0000 | 0.9977 |
| wrong | 60 | 0.9692 | 1.0000 | 0.0000 | 0.9978 |

### `colgrep` x `code_only`

| subcategory | count | reverse_context | coverage | unused ratio | max evidence |
|---|---:|---:|---:|---:|---:|
| ambiguous | 60 | 0.9294 | 1.0000 | 0.0000 | 0.9888 |
| correct | 60 | 0.9308 | 1.0000 | 0.0000 | 0.9905 |
| wrong | 60 | 0.9224 | 1.0000 | 0.0000 | 0.9908 |

### `colgrep` x `fuse_mean`

| subcategory | count | reverse_context | coverage | unused ratio | max evidence |
|---|---:|---:|---:|---:|---:|
| ambiguous | 60 | 0.9522 | 1.0000 | 0.0000 | 0.9930 |
| correct | 60 | 0.9534 | 1.0000 | 0.0000 | 0.9941 |
| wrong | 60 | 0.9458 | 1.0000 | 0.0000 | 0.9943 |

### `colgrep` x `fuse_max`

| subcategory | count | reverse_context | coverage | unused ratio | max evidence |
|---|---:|---:|---:|---:|---:|
| ambiguous | 60 | 0.9750 | 1.0000 | 0.0000 | 0.9977 |
| correct | 60 | 0.9760 | 1.0000 | 0.0000 | 0.9981 |
| wrong | 60 | 0.9692 | 1.0000 | 0.0000 | 0.9981 |

### `colgrep` x `fuse_min`

| subcategory | count | reverse_context | coverage | unused ratio | max evidence |
|---|---:|---:|---:|---:|---:|
| ambiguous | 60 | 0.9294 | 1.0000 | 0.0000 | 0.9884 |
| correct | 60 | 0.9308 | 1.0000 | 0.0000 | 0.9901 |
| wrong | 60 | 0.9224 | 1.0000 | 0.0000 | 0.9905 |

## Held-out support units (flagship cell)

Flagship cell: `colgrep` x `fuse_mean`. Consensus held-out ids are the intersection of GTE's and LateOn-Code-edge's independently computed `usage_state == 'unused'` sets. `gte` / `code` columns show each encoder's raw unused set.

| case | label | subcategory | consensus held-out | gte unused | code unused | consensus held-out ids |
|---|---|---|---:|---:|---:|---|
| base_01_f0afeaf2_t952_f1__correct | grounded | correct | 0 | 0 | 0 | - |
| base_01_f0afeaf2_t952_f1__wrong | ungrounded | wrong | 0 | 0 | 0 | - |
| base_01_f0afeaf2_t952_f1__ambiguous | ambiguous | ambiguous | 0 | 0 | 0 | - |
| base_02_61e1f5a1_t258_f0__correct | grounded | correct | 0 | 0 | 0 | - |
| base_02_61e1f5a1_t258_f0__wrong | ungrounded | wrong | 0 | 0 | 0 | - |
| base_02_61e1f5a1_t258_f0__ambiguous | ambiguous | ambiguous | 0 | 0 | 0 | - |
| base_03_f0afeaf2_t2082_f0__correct | grounded | correct | 0 | 0 | 0 | - |
| base_03_f0afeaf2_t2082_f0__wrong | ungrounded | wrong | 0 | 0 | 0 | - |
| base_03_f0afeaf2_t2082_f0__ambiguous | ambiguous | ambiguous | 0 | 0 | 0 | - |
| base_04_8a0f6f68_t521_f3__correct | grounded | correct | 0 | 0 | 0 | - |
| base_04_8a0f6f68_t521_f3__wrong | ungrounded | wrong | 0 | 0 | 0 | - |
| base_04_8a0f6f68_t521_f3__ambiguous | ambiguous | ambiguous | 0 | 0 | 0 | - |
| base_05_126e38fc_t1159_f4__correct | grounded | correct | 0 | 0 | 0 | - |
| base_05_126e38fc_t1159_f4__wrong | ungrounded | wrong | 0 | 0 | 0 | - |
| base_05_126e38fc_t1159_f4__ambiguous | ambiguous | ambiguous | 0 | 0 | 0 | - |
| base_06_f0afeaf2_t1771_f0__correct | grounded | correct | 0 | 0 | 0 | - |
| base_06_f0afeaf2_t1771_f0__wrong | ungrounded | wrong | 0 | 0 | 0 | - |
| base_06_f0afeaf2_t1771_f0__ambiguous | ambiguous | ambiguous | 0 | 0 | 0 | - |
| base_07_7b7fb621_t328_f1__correct | grounded | correct | 0 | 0 | 0 | - |
| base_07_7b7fb621_t328_f1__wrong | ungrounded | wrong | 0 | 0 | 0 | - |
| base_07_7b7fb621_t328_f1__ambiguous | ambiguous | ambiguous | 0 | 0 | 0 | - |
| base_08_126e38fc_t1785_f0__correct | grounded | correct | 0 | 0 | 0 | - |
| base_08_126e38fc_t1785_f0__wrong | ungrounded | wrong | 0 | 0 | 0 | - |
| base_08_126e38fc_t1785_f0__ambiguous | ambiguous | ambiguous | 0 | 0 | 0 | - |
| base_09_0b1422bb_t571_f0__correct | grounded | correct | 0 | 0 | 0 | - |
| base_09_0b1422bb_t571_f0__wrong | ungrounded | wrong | 0 | 0 | 0 | - |
| base_09_0b1422bb_t571_f0__ambiguous | ambiguous | ambiguous | 0 | 0 | 0 | - |
| base_10_d7fd7127_t1579_f0__correct | grounded | correct | 0 | 0 | 0 | - |
| base_10_d7fd7127_t1579_f0__wrong | ungrounded | wrong | 0 | 0 | 0 | - |
| base_10_d7fd7127_t1579_f0__ambiguous | ambiguous | ambiguous | 0 | 0 | 0 | - |
| base_11_f0afeaf2_t1418_f1__correct | grounded | correct | 0 | 0 | 0 | - |
| base_11_f0afeaf2_t1418_f1__wrong | ungrounded | wrong | 0 | 0 | 0 | - |
| base_11_f0afeaf2_t1418_f1__ambiguous | ambiguous | ambiguous | 0 | 0 | 0 | - |
| base_12_f0afeaf2_t955_f2__correct | grounded | correct | 0 | 0 | 0 | - |
| base_12_f0afeaf2_t955_f2__wrong | ungrounded | wrong | 0 | 0 | 0 | - |
| base_12_f0afeaf2_t955_f2__ambiguous | ambiguous | ambiguous | 0 | 0 | 0 | - |
| base_13_d7fd7127_t1553_f0__correct | grounded | correct | 0 | 0 | 0 | - |
| base_13_d7fd7127_t1553_f0__wrong | ungrounded | wrong | 0 | 0 | 0 | - |
| base_13_d7fd7127_t1553_f0__ambiguous | ambiguous | ambiguous | 0 | 0 | 0 | - |
| base_14_f0afeaf2_t2825_f0__correct | grounded | correct | 0 | 0 | 0 | - |
| base_14_f0afeaf2_t2825_f0__wrong | ungrounded | wrong | 0 | 0 | 0 | - |
| base_14_f0afeaf2_t2825_f0__ambiguous | ambiguous | ambiguous | 0 | 0 | 0 | - |
| base_15_91cf0ec9_t419_f2__correct | grounded | correct | 0 | 0 | 0 | - |
| base_15_91cf0ec9_t419_f2__wrong | ungrounded | wrong | 0 | 0 | 0 | - |
| base_15_91cf0ec9_t419_f2__ambiguous | ambiguous | ambiguous | 0 | 0 | 0 | - |
| base_16_8a0f6f68_t1002_f0__correct | grounded | correct | 0 | 0 | 0 | - |
| base_16_8a0f6f68_t1002_f0__wrong | ungrounded | wrong | 0 | 0 | 0 | - |
| base_16_8a0f6f68_t1002_f0__ambiguous | ambiguous | ambiguous | 0 | 0 | 0 | - |
| base_17_126e38fc_t712_f0__correct | grounded | correct | 0 | 0 | 0 | - |
| base_17_126e38fc_t712_f0__wrong | ungrounded | wrong | 0 | 0 | 0 | - |
| base_17_126e38fc_t712_f0__ambiguous | ambiguous | ambiguous | 0 | 0 | 0 | - |
| base_18_f0afeaf2_t457_f0__correct | grounded | correct | 0 | 0 | 0 | - |
| base_18_f0afeaf2_t457_f0__wrong | ungrounded | wrong | 0 | 0 | 0 | - |
| base_18_f0afeaf2_t457_f0__ambiguous | ambiguous | ambiguous | 0 | 0 | 0 | - |
| base_19_5e875b7e_t1360_f0__correct | grounded | correct | 0 | 0 | 0 | - |
| base_19_5e875b7e_t1360_f0__wrong | ungrounded | wrong | 0 | 0 | 0 | - |
| base_19_5e875b7e_t1360_f0__ambiguous | ambiguous | ambiguous | 0 | 0 | 0 | - |
| base_20_d7fd7127_t1617_f0__correct | grounded | correct | 0 | 0 | 0 | - |
| base_20_d7fd7127_t1617_f0__wrong | ungrounded | wrong | 0 | 0 | 0 | - |
| base_20_d7fd7127_t1617_f0__ambiguous | ambiguous | ambiguous | 0 | 0 | 0 | - |
| base_21_a4d14a56_t732_f0__correct | grounded | correct | 0 | 0 | 0 | - |
| base_21_a4d14a56_t732_f0__wrong | ungrounded | wrong | 0 | 0 | 0 | - |
| base_21_a4d14a56_t732_f0__ambiguous | ambiguous | ambiguous | 0 | 0 | 0 | - |
| base_22_d7fd7127_t1209_f1__correct | grounded | correct | 0 | 0 | 0 | - |
| base_22_d7fd7127_t1209_f1__wrong | ungrounded | wrong | 0 | 0 | 0 | - |
| base_22_d7fd7127_t1209_f1__ambiguous | ambiguous | ambiguous | 0 | 0 | 0 | - |
| base_23_a4d14a56_t579_f0__correct | grounded | correct | 0 | 0 | 0 | - |
| base_23_a4d14a56_t579_f0__wrong | ungrounded | wrong | 0 | 0 | 0 | - |
| base_23_a4d14a56_t579_f0__ambiguous | ambiguous | ambiguous | 0 | 0 | 0 | - |
| base_24_126e38fc_t1774_f0__correct | grounded | correct | 0 | 0 | 0 | - |
| base_24_126e38fc_t1774_f0__wrong | ungrounded | wrong | 0 | 0 | 0 | - |
| base_24_126e38fc_t1774_f0__ambiguous | ambiguous | ambiguous | 0 | 0 | 0 | - |
| base_25_8a0f6f68_t1261_f1__correct | grounded | correct | 0 | 0 | 0 | - |
| base_25_8a0f6f68_t1261_f1__wrong | ungrounded | wrong | 0 | 0 | 0 | - |
| base_25_8a0f6f68_t1261_f1__ambiguous | ambiguous | ambiguous | 0 | 0 | 0 | - |
| base_26_f0afeaf2_t1530_f1__correct | grounded | correct | 0 | 0 | 0 | - |
| base_26_f0afeaf2_t1530_f1__wrong | ungrounded | wrong | 0 | 0 | 0 | - |
| base_26_f0afeaf2_t1530_f1__ambiguous | ambiguous | ambiguous | 0 | 0 | 0 | - |
| base_27_8a0f6f68_t1163_f0__correct | grounded | correct | 0 | 0 | 0 | - |
| base_27_8a0f6f68_t1163_f0__wrong | ungrounded | wrong | 0 | 0 | 0 | - |
| base_27_8a0f6f68_t1163_f0__ambiguous | ambiguous | ambiguous | 0 | 0 | 0 | - |
| base_28_126e38fc_t1892_f0__correct | grounded | correct | 0 | 0 | 0 | - |
| base_28_126e38fc_t1892_f0__wrong | ungrounded | wrong | 0 | 0 | 0 | - |
| base_28_126e38fc_t1892_f0__ambiguous | ambiguous | ambiguous | 0 | 0 | 0 | - |
| base_29_8a0f6f68_t1278_f0__correct | grounded | correct | 0 | 0 | 0 | - |
| base_29_8a0f6f68_t1278_f0__wrong | ungrounded | wrong | 0 | 0 | 0 | - |
| base_29_8a0f6f68_t1278_f0__ambiguous | ambiguous | ambiguous | 0 | 0 | 0 | - |
| base_30_d7fd7127_t807_f1__correct | grounded | correct | 0 | 0 | 0 | - |
| base_30_d7fd7127_t807_f1__wrong | ungrounded | wrong | 0 | 0 | 0 | - |
| base_30_d7fd7127_t807_f1__ambiguous | ambiguous | ambiguous | 0 | 0 | 0 | - |
| base_31_8a0f6f68_t1301_f0__correct | grounded | correct | 0 | 0 | 0 | - |
| base_31_8a0f6f68_t1301_f0__wrong | ungrounded | wrong | 0 | 0 | 0 | - |
| base_31_8a0f6f68_t1301_f0__ambiguous | ambiguous | ambiguous | 0 | 0 | 0 | - |
| base_32_126e38fc_t1007_f1__correct | grounded | correct | 0 | 0 | 0 | - |
| base_32_126e38fc_t1007_f1__wrong | ungrounded | wrong | 0 | 0 | 0 | - |
| base_32_126e38fc_t1007_f1__ambiguous | ambiguous | ambiguous | 0 | 0 | 0 | - |
| base_33_d7fd7127_t1397_f0__correct | grounded | correct | 0 | 0 | 0 | - |
| base_33_d7fd7127_t1397_f0__wrong | ungrounded | wrong | 0 | 0 | 0 | - |
| base_33_d7fd7127_t1397_f0__ambiguous | ambiguous | ambiguous | 0 | 0 | 0 | - |
| base_34_5e875b7e_t1637_f0__correct | grounded | correct | 0 | 0 | 0 | - |
| base_34_5e875b7e_t1637_f0__wrong | ungrounded | wrong | 0 | 0 | 0 | - |
| base_34_5e875b7e_t1637_f0__ambiguous | ambiguous | ambiguous | 0 | 0 | 0 | - |
| base_35_a4d14a56_t1005_f0__correct | grounded | correct | 0 | 0 | 0 | - |
| base_35_a4d14a56_t1005_f0__wrong | ungrounded | wrong | 0 | 0 | 0 | - |
| base_35_a4d14a56_t1005_f0__ambiguous | ambiguous | ambiguous | 0 | 0 | 0 | - |
| base_36_a4d14a56_t693_f0__correct | grounded | correct | 0 | 0 | 0 | - |
| base_36_a4d14a56_t693_f0__wrong | ungrounded | wrong | 0 | 0 | 0 | - |
| base_36_a4d14a56_t693_f0__ambiguous | ambiguous | ambiguous | 0 | 0 | 0 | - |
| base_37_126e38fc_t1936_f0__correct | grounded | correct | 0 | 0 | 0 | - |
| base_37_126e38fc_t1936_f0__wrong | ungrounded | wrong | 0 | 0 | 0 | - |
| base_37_126e38fc_t1936_f0__ambiguous | ambiguous | ambiguous | 0 | 0 | 0 | - |
| base_38_a4d14a56_t869_f0__correct | grounded | correct | 0 | 0 | 0 | - |
| base_38_a4d14a56_t869_f0__wrong | ungrounded | wrong | 0 | 0 | 0 | - |
| base_38_a4d14a56_t869_f0__ambiguous | ambiguous | ambiguous | 0 | 0 | 0 | - |
| base_39_5e875b7e_t2278_f0__correct | grounded | correct | 0 | 0 | 0 | - |
| base_39_5e875b7e_t2278_f0__wrong | ungrounded | wrong | 0 | 0 | 0 | - |
| base_39_5e875b7e_t2278_f0__ambiguous | ambiguous | ambiguous | 0 | 0 | 0 | - |
| base_40_ed69b0cd_t395_f0__correct | grounded | correct | 0 | 0 | 0 | - |
| base_40_ed69b0cd_t395_f0__wrong | ungrounded | wrong | 0 | 0 | 0 | - |
| base_40_ed69b0cd_t395_f0__ambiguous | ambiguous | ambiguous | 0 | 0 | 0 | - |
| base_41_a4d14a56_t994_f1__correct | grounded | correct | 0 | 0 | 0 | - |
| base_41_a4d14a56_t994_f1__wrong | ungrounded | wrong | 0 | 0 | 0 | - |
| base_41_a4d14a56_t994_f1__ambiguous | ambiguous | ambiguous | 0 | 0 | 0 | - |
| base_42_a4d14a56_t531_f0__correct | grounded | correct | 0 | 0 | 0 | - |
| base_42_a4d14a56_t531_f0__wrong | ungrounded | wrong | 0 | 0 | 0 | - |
| base_42_a4d14a56_t531_f0__ambiguous | ambiguous | ambiguous | 0 | 0 | 0 | - |
| base_43_ed69b0cd_t364_f0__correct | grounded | correct | 0 | 0 | 0 | - |
| base_43_ed69b0cd_t364_f0__wrong | ungrounded | wrong | 0 | 0 | 0 | - |
| base_43_ed69b0cd_t364_f0__ambiguous | ambiguous | ambiguous | 0 | 0 | 0 | - |
| base_44_a4d14a56_t568_f0__correct | grounded | correct | 0 | 0 | 0 | - |
| base_44_a4d14a56_t568_f0__wrong | ungrounded | wrong | 0 | 0 | 0 | - |
| base_44_a4d14a56_t568_f0__ambiguous | ambiguous | ambiguous | 0 | 0 | 0 | - |
| base_45_ed69b0cd_t367_f0__correct | grounded | correct | 0 | 0 | 0 | - |
| base_45_ed69b0cd_t367_f0__wrong | ungrounded | wrong | 0 | 0 | 0 | - |
| base_45_ed69b0cd_t367_f0__ambiguous | ambiguous | ambiguous | 0 | 0 | 0 | - |
| base_46_d7fd7127_t317_f0__correct | grounded | correct | 0 | 0 | 0 | - |
| base_46_d7fd7127_t317_f0__wrong | ungrounded | wrong | 0 | 0 | 0 | - |
| base_46_d7fd7127_t317_f0__ambiguous | ambiguous | ambiguous | 0 | 0 | 0 | - |
| base_47_0b1422bb_t847_f0__correct | grounded | correct | 0 | 0 | 0 | - |
| base_47_0b1422bb_t847_f0__wrong | ungrounded | wrong | 0 | 0 | 0 | - |
| base_47_0b1422bb_t847_f0__ambiguous | ambiguous | ambiguous | 0 | 0 | 0 | - |
| base_48_ed69b0cd_t10_f2__correct | grounded | correct | 0 | 0 | 0 | - |
| base_48_ed69b0cd_t10_f2__wrong | ungrounded | wrong | 0 | 0 | 0 | - |
| base_48_ed69b0cd_t10_f2__ambiguous | ambiguous | ambiguous | 0 | 0 | 0 | - |
| base_49_fa052757_t213_f0__correct | grounded | correct | 0 | 0 | 0 | - |
| base_49_fa052757_t213_f0__wrong | ungrounded | wrong | 0 | 0 | 0 | - |
| base_49_fa052757_t213_f0__ambiguous | ambiguous | ambiguous | 0 | 0 | 0 | - |
| base_50_0b1422bb_t839_f0__correct | grounded | correct | 0 | 0 | 0 | - |
| base_50_0b1422bb_t839_f0__wrong | ungrounded | wrong | 0 | 0 | 0 | - |
| base_50_0b1422bb_t839_f0__ambiguous | ambiguous | ambiguous | 0 | 0 | 0 | - |
| base_51_5e875b7e_t1929_f0__correct | grounded | correct | 0 | 0 | 0 | - |
| base_51_5e875b7e_t1929_f0__wrong | ungrounded | wrong | 0 | 0 | 0 | - |
| base_51_5e875b7e_t1929_f0__ambiguous | ambiguous | ambiguous | 0 | 0 | 0 | - |
| base_52_7b7fb621_t310_f3__correct | grounded | correct | 0 | 0 | 0 | - |
| base_52_7b7fb621_t310_f3__wrong | ungrounded | wrong | 0 | 0 | 0 | - |
| base_52_7b7fb621_t310_f3__ambiguous | ambiguous | ambiguous | 0 | 0 | 0 | - |
| base_53_7b7fb621_t347_f0__correct | grounded | correct | 0 | 0 | 0 | - |
| base_53_7b7fb621_t347_f0__wrong | ungrounded | wrong | 0 | 0 | 0 | - |
| base_53_7b7fb621_t347_f0__ambiguous | ambiguous | ambiguous | 0 | 0 | 0 | - |
| base_54_5e875b7e_t630_f0__correct | grounded | correct | 0 | 0 | 0 | - |
| base_54_5e875b7e_t630_f0__wrong | ungrounded | wrong | 0 | 0 | 0 | - |
| base_54_5e875b7e_t630_f0__ambiguous | ambiguous | ambiguous | 0 | 0 | 0 | - |
| base_55_0b1422bb_t652_f0__correct | grounded | correct | 0 | 0 | 0 | - |
| base_55_0b1422bb_t652_f0__wrong | ungrounded | wrong | 0 | 0 | 0 | - |
| base_55_0b1422bb_t652_f0__ambiguous | ambiguous | ambiguous | 0 | 0 | 0 | - |
| base_56_7b7fb621_t406_f0__correct | grounded | correct | 0 | 0 | 0 | - |
| base_56_7b7fb621_t406_f0__wrong | ungrounded | wrong | 0 | 0 | 0 | - |
| base_56_7b7fb621_t406_f0__ambiguous | ambiguous | ambiguous | 0 | 0 | 0 | - |
| base_57_91cf0ec9_t84_f10__correct | grounded | correct | 0 | 0 | 0 | - |
| base_57_91cf0ec9_t84_f10__wrong | ungrounded | wrong | 0 | 0 | 0 | - |
| base_57_91cf0ec9_t84_f10__ambiguous | ambiguous | ambiguous | 0 | 0 | 0 | - |
| base_58_7b7fb621_t237_f0__correct | grounded | correct | 0 | 0 | 0 | - |
| base_58_7b7fb621_t237_f0__wrong | ungrounded | wrong | 0 | 0 | 0 | - |
| base_58_7b7fb621_t237_f0__ambiguous | ambiguous | ambiguous | 0 | 0 | 0 | - |
| base_59_ee0ad405_t137_f0__correct | grounded | correct | 0 | 0 | 0 | - |
| base_59_ee0ad405_t137_f0__wrong | ungrounded | wrong | 0 | 0 | 0 | - |
| base_59_ee0ad405_t137_f0__ambiguous | ambiguous | ambiguous | 0 | 0 | 0 | - |
| base_60_7b7fb621_t178_f1__correct | grounded | correct | 0 | 0 | 0 | - |
| base_60_7b7fb621_t178_f1__wrong | ungrounded | wrong | 0 | 0 | 0 | - |
| base_60_7b7fb621_t178_f1__ambiguous | ambiguous | ambiguous | 0 | 0 | 0 | - |

## Per-case scores

| id | chunker | scorer | label | subcategory | reverse_context | coverage | unused | phantom | max evidence | units |
|---|---|---|---|---|---:|---:|---:|---|---:|---:|
| base_01_f0afeaf2_t952_f1__correct | sentence_packed | `gte_only` | grounded | correct | 0.9843 | 1.0000 | 0.0000 | False | 0.9981 | 145 |
| base_01_f0afeaf2_t952_f1__correct | sentence_packed | `code_only` | grounded | correct | 0.9569 | 1.0000 | 0.0000 | False | 0.9858 | 145 |
| base_01_f0afeaf2_t952_f1__correct | sentence_packed | `fuse_mean` | grounded | correct | 0.9706 | 1.0000 | 0.0000 | False | 0.9919 | 145 |
| base_01_f0afeaf2_t952_f1__correct | sentence_packed | `fuse_max` | grounded | correct | 0.9843 | 1.0000 | 0.0000 | False | 0.9981 | 145 |
| base_01_f0afeaf2_t952_f1__correct | sentence_packed | `fuse_min` | grounded | correct | 0.9569 | 1.0000 | 0.0000 | False | 0.9858 | 145 |
| base_01_f0afeaf2_t952_f1__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9820 | 1.0000 | 0.0000 | False | 0.9999 | 145 |
| base_01_f0afeaf2_t952_f1__wrong | sentence_packed | `code_only` | ungrounded | wrong | 0.9512 | 1.0000 | 0.0000 | False | 0.9927 | 145 |
| base_01_f0afeaf2_t952_f1__wrong | sentence_packed | `fuse_mean` | ungrounded | wrong | 0.9666 | 1.0000 | 0.0000 | False | 0.9963 | 145 |
| base_01_f0afeaf2_t952_f1__wrong | sentence_packed | `fuse_max` | ungrounded | wrong | 0.9820 | 1.0000 | 0.0000 | False | 0.9999 | 145 |
| base_01_f0afeaf2_t952_f1__wrong | sentence_packed | `fuse_min` | ungrounded | wrong | 0.9512 | 1.0000 | 0.0000 | False | 0.9927 | 145 |
| base_01_f0afeaf2_t952_f1__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9844 | 1.0000 | 0.0000 | False | 0.9981 | 145 |
| base_01_f0afeaf2_t952_f1__ambiguous | sentence_packed | `code_only` | ambiguous | ambiguous | 0.9577 | 1.0000 | 0.0000 | False | 0.9861 | 145 |
| base_01_f0afeaf2_t952_f1__ambiguous | sentence_packed | `fuse_mean` | ambiguous | ambiguous | 0.9710 | 1.0000 | 0.0000 | False | 0.9921 | 145 |
| base_01_f0afeaf2_t952_f1__ambiguous | sentence_packed | `fuse_max` | ambiguous | ambiguous | 0.9844 | 1.0000 | 0.0000 | False | 0.9981 | 145 |
| base_01_f0afeaf2_t952_f1__ambiguous | sentence_packed | `fuse_min` | ambiguous | ambiguous | 0.9577 | 1.0000 | 0.0000 | False | 0.9861 | 145 |
| base_02_61e1f5a1_t258_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9725 | 1.0000 | 0.0000 | False | 0.9988 | 88 |
| base_02_61e1f5a1_t258_f0__correct | sentence_packed | `code_only` | grounded | correct | 0.9203 | 1.0000 | 0.0000 | False | 1.0000 | 88 |
| base_02_61e1f5a1_t258_f0__correct | sentence_packed | `fuse_mean` | grounded | correct | 0.9464 | 1.0000 | 0.0000 | False | 0.9994 | 88 |
| base_02_61e1f5a1_t258_f0__correct | sentence_packed | `fuse_max` | grounded | correct | 0.9725 | 1.0000 | 0.0000 | False | 1.0000 | 88 |
| base_02_61e1f5a1_t258_f0__correct | sentence_packed | `fuse_min` | grounded | correct | 0.9203 | 1.0000 | 0.0000 | False | 0.9988 | 88 |
| base_02_61e1f5a1_t258_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9685 | 1.0000 | 0.0000 | False | 0.9981 | 88 |
| base_02_61e1f5a1_t258_f0__wrong | sentence_packed | `code_only` | ungrounded | wrong | 0.9079 | 1.0000 | 0.0000 | False | 1.0000 | 88 |
| base_02_61e1f5a1_t258_f0__wrong | sentence_packed | `fuse_mean` | ungrounded | wrong | 0.9382 | 1.0000 | 0.0000 | False | 0.9990 | 88 |
| base_02_61e1f5a1_t258_f0__wrong | sentence_packed | `fuse_max` | ungrounded | wrong | 0.9685 | 1.0000 | 0.0000 | False | 1.0000 | 88 |
| base_02_61e1f5a1_t258_f0__wrong | sentence_packed | `fuse_min` | ungrounded | wrong | 0.9079 | 1.0000 | 0.0000 | False | 0.9981 | 88 |
| base_02_61e1f5a1_t258_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9724 | 1.0000 | 0.0000 | False | 0.9972 | 88 |
| base_02_61e1f5a1_t258_f0__ambiguous | sentence_packed | `code_only` | ambiguous | ambiguous | 0.9192 | 1.0000 | 0.0000 | False | 0.9840 | 88 |
| base_02_61e1f5a1_t258_f0__ambiguous | sentence_packed | `fuse_mean` | ambiguous | ambiguous | 0.9458 | 1.0000 | 0.0000 | False | 0.9906 | 88 |
| base_02_61e1f5a1_t258_f0__ambiguous | sentence_packed | `fuse_max` | ambiguous | ambiguous | 0.9724 | 1.0000 | 0.0000 | False | 0.9972 | 88 |
| base_02_61e1f5a1_t258_f0__ambiguous | sentence_packed | `fuse_min` | ambiguous | ambiguous | 0.9192 | 1.0000 | 0.0000 | False | 0.9840 | 88 |
| base_03_f0afeaf2_t2082_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9774 | 1.0000 | 0.0000 | False | 0.9979 | 198 |
| base_03_f0afeaf2_t2082_f0__correct | sentence_packed | `code_only` | grounded | correct | 0.9487 | 1.0000 | 0.0000 | False | 0.9856 | 198 |
| base_03_f0afeaf2_t2082_f0__correct | sentence_packed | `fuse_mean` | grounded | correct | 0.9631 | 1.0000 | 0.0000 | False | 0.9917 | 198 |
| base_03_f0afeaf2_t2082_f0__correct | sentence_packed | `fuse_max` | grounded | correct | 0.9774 | 1.0000 | 0.0000 | False | 0.9979 | 198 |
| base_03_f0afeaf2_t2082_f0__correct | sentence_packed | `fuse_min` | grounded | correct | 0.9487 | 1.0000 | 0.0000 | False | 0.9856 | 198 |
| base_03_f0afeaf2_t2082_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9705 | 1.0000 | 0.0000 | False | 0.9955 | 198 |
| base_03_f0afeaf2_t2082_f0__wrong | sentence_packed | `code_only` | ungrounded | wrong | 0.9411 | 1.0000 | 0.0000 | False | 0.9951 | 198 |
| base_03_f0afeaf2_t2082_f0__wrong | sentence_packed | `fuse_mean` | ungrounded | wrong | 0.9558 | 1.0000 | 0.0000 | False | 0.9953 | 198 |
| base_03_f0afeaf2_t2082_f0__wrong | sentence_packed | `fuse_max` | ungrounded | wrong | 0.9705 | 1.0000 | 0.0000 | False | 0.9955 | 198 |
| base_03_f0afeaf2_t2082_f0__wrong | sentence_packed | `fuse_min` | ungrounded | wrong | 0.9411 | 1.0000 | 0.0000 | False | 0.9951 | 198 |
| base_03_f0afeaf2_t2082_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9776 | 1.0000 | 0.0000 | False | 0.9974 | 198 |
| base_03_f0afeaf2_t2082_f0__ambiguous | sentence_packed | `code_only` | ambiguous | ambiguous | 0.9490 | 1.0000 | 0.0000 | False | 1.0000 | 198 |
| base_03_f0afeaf2_t2082_f0__ambiguous | sentence_packed | `fuse_mean` | ambiguous | ambiguous | 0.9633 | 1.0000 | 0.0000 | False | 0.9987 | 198 |
| base_03_f0afeaf2_t2082_f0__ambiguous | sentence_packed | `fuse_max` | ambiguous | ambiguous | 0.9776 | 1.0000 | 0.0000 | False | 1.0000 | 198 |
| base_03_f0afeaf2_t2082_f0__ambiguous | sentence_packed | `fuse_min` | ambiguous | ambiguous | 0.9490 | 1.0000 | 0.0000 | False | 0.9974 | 198 |
| base_04_8a0f6f68_t521_f3__correct | sentence_packed | `gte_only` | grounded | correct | 0.9850 | 1.0000 | 0.0000 | False | 0.9975 | 71 |
| base_04_8a0f6f68_t521_f3__correct | sentence_packed | `code_only` | grounded | correct | 0.9614 | 1.0000 | 0.0000 | False | 0.9870 | 71 |
| base_04_8a0f6f68_t521_f3__correct | sentence_packed | `fuse_mean` | grounded | correct | 0.9732 | 1.0000 | 0.0000 | False | 0.9922 | 71 |
| base_04_8a0f6f68_t521_f3__correct | sentence_packed | `fuse_max` | grounded | correct | 0.9850 | 1.0000 | 0.0000 | False | 0.9975 | 71 |
| base_04_8a0f6f68_t521_f3__correct | sentence_packed | `fuse_min` | grounded | correct | 0.9614 | 1.0000 | 0.0000 | False | 0.9870 | 71 |
| base_04_8a0f6f68_t521_f3__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9811 | 1.0000 | 0.0000 | False | 0.9975 | 71 |
| base_04_8a0f6f68_t521_f3__wrong | sentence_packed | `code_only` | ungrounded | wrong | 0.9523 | 1.0000 | 0.0000 | False | 0.9918 | 71 |
| base_04_8a0f6f68_t521_f3__wrong | sentence_packed | `fuse_mean` | ungrounded | wrong | 0.9667 | 1.0000 | 0.0000 | False | 0.9947 | 71 |
| base_04_8a0f6f68_t521_f3__wrong | sentence_packed | `fuse_max` | ungrounded | wrong | 0.9811 | 1.0000 | 0.0000 | False | 0.9975 | 71 |
| base_04_8a0f6f68_t521_f3__wrong | sentence_packed | `fuse_min` | ungrounded | wrong | 0.9523 | 1.0000 | 0.0000 | False | 0.9918 | 71 |
| base_04_8a0f6f68_t521_f3__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9851 | 1.0000 | 0.0000 | False | 0.9975 | 71 |
| base_04_8a0f6f68_t521_f3__ambiguous | sentence_packed | `code_only` | ambiguous | ambiguous | 0.9596 | 1.0000 | 0.0000 | False | 0.9870 | 71 |
| base_04_8a0f6f68_t521_f3__ambiguous | sentence_packed | `fuse_mean` | ambiguous | ambiguous | 0.9723 | 1.0000 | 0.0000 | False | 0.9922 | 71 |
| base_04_8a0f6f68_t521_f3__ambiguous | sentence_packed | `fuse_max` | ambiguous | ambiguous | 0.9851 | 1.0000 | 0.0000 | False | 0.9975 | 71 |
| base_04_8a0f6f68_t521_f3__ambiguous | sentence_packed | `fuse_min` | ambiguous | ambiguous | 0.9596 | 1.0000 | 0.0000 | False | 0.9870 | 71 |
| base_05_126e38fc_t1159_f4__correct | sentence_packed | `gte_only` | grounded | correct | 0.9748 | 1.0000 | 0.0000 | False | 1.0000 | 130 |
| base_05_126e38fc_t1159_f4__correct | sentence_packed | `code_only` | grounded | correct | 0.9257 | 1.0000 | 0.0000 | False | 0.9998 | 130 |
| base_05_126e38fc_t1159_f4__correct | sentence_packed | `fuse_mean` | grounded | correct | 0.9503 | 1.0000 | 0.0000 | False | 0.9999 | 130 |
| base_05_126e38fc_t1159_f4__correct | sentence_packed | `fuse_max` | grounded | correct | 0.9748 | 1.0000 | 0.0000 | False | 1.0000 | 130 |
| base_05_126e38fc_t1159_f4__correct | sentence_packed | `fuse_min` | grounded | correct | 0.9257 | 1.0000 | 0.0000 | False | 0.9998 | 130 |
| base_05_126e38fc_t1159_f4__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9726 | 1.0000 | 0.0000 | False | 1.0000 | 130 |
| base_05_126e38fc_t1159_f4__wrong | sentence_packed | `code_only` | ungrounded | wrong | 0.9220 | 1.0000 | 0.0000 | False | 1.0000 | 130 |
| base_05_126e38fc_t1159_f4__wrong | sentence_packed | `fuse_mean` | ungrounded | wrong | 0.9473 | 1.0000 | 0.0000 | False | 1.0000 | 130 |
| base_05_126e38fc_t1159_f4__wrong | sentence_packed | `fuse_max` | ungrounded | wrong | 0.9726 | 1.0000 | 0.0000 | False | 1.0000 | 130 |
| base_05_126e38fc_t1159_f4__wrong | sentence_packed | `fuse_min` | ungrounded | wrong | 0.9220 | 1.0000 | 0.0000 | False | 1.0000 | 130 |
| base_05_126e38fc_t1159_f4__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9745 | 1.0000 | 0.0000 | False | 0.9896 | 130 |
| base_05_126e38fc_t1159_f4__ambiguous | sentence_packed | `code_only` | ambiguous | ambiguous | 0.9254 | 1.0000 | 0.0000 | False | 0.9265 | 130 |
| base_05_126e38fc_t1159_f4__ambiguous | sentence_packed | `fuse_mean` | ambiguous | ambiguous | 0.9500 | 1.0000 | 0.0000 | False | 0.9581 | 130 |
| base_05_126e38fc_t1159_f4__ambiguous | sentence_packed | `fuse_max` | ambiguous | ambiguous | 0.9745 | 1.0000 | 0.0000 | False | 0.9896 | 130 |
| base_05_126e38fc_t1159_f4__ambiguous | sentence_packed | `fuse_min` | ambiguous | ambiguous | 0.9254 | 1.0000 | 0.0000 | False | 0.9265 | 130 |
| base_06_f0afeaf2_t1771_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9755 | 1.0000 | 0.0000 | False | 0.9929 | 199 |
| base_06_f0afeaf2_t1771_f0__correct | sentence_packed | `code_only` | grounded | correct | 0.9338 | 1.0000 | 0.0000 | False | 0.9863 | 199 |
| base_06_f0afeaf2_t1771_f0__correct | sentence_packed | `fuse_mean` | grounded | correct | 0.9547 | 1.0000 | 0.0000 | False | 0.9896 | 199 |
| base_06_f0afeaf2_t1771_f0__correct | sentence_packed | `fuse_max` | grounded | correct | 0.9755 | 1.0000 | 0.0000 | False | 0.9929 | 199 |
| base_06_f0afeaf2_t1771_f0__correct | sentence_packed | `fuse_min` | grounded | correct | 0.9338 | 1.0000 | 0.0000 | False | 0.9863 | 199 |
| base_06_f0afeaf2_t1771_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9700 | 1.0000 | 0.0000 | False | 0.9705 | 199 |
| base_06_f0afeaf2_t1771_f0__wrong | sentence_packed | `code_only` | ungrounded | wrong | 0.9245 | 1.0000 | 0.0000 | False | 0.9815 | 199 |
| base_06_f0afeaf2_t1771_f0__wrong | sentence_packed | `fuse_mean` | ungrounded | wrong | 0.9472 | 1.0000 | 0.0000 | False | 0.9760 | 199 |
| base_06_f0afeaf2_t1771_f0__wrong | sentence_packed | `fuse_max` | ungrounded | wrong | 0.9700 | 1.0000 | 0.0000 | False | 0.9815 | 199 |
| base_06_f0afeaf2_t1771_f0__wrong | sentence_packed | `fuse_min` | ungrounded | wrong | 0.9245 | 1.0000 | 0.0000 | False | 0.9705 | 199 |
| base_06_f0afeaf2_t1771_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9709 | 1.0000 | 0.0000 | False | 0.9674 | 199 |
| base_06_f0afeaf2_t1771_f0__ambiguous | sentence_packed | `code_only` | ambiguous | ambiguous | 0.9244 | 1.0000 | 0.0000 | False | 0.9815 | 199 |
| base_06_f0afeaf2_t1771_f0__ambiguous | sentence_packed | `fuse_mean` | ambiguous | ambiguous | 0.9477 | 1.0000 | 0.0000 | False | 0.9745 | 199 |
| base_06_f0afeaf2_t1771_f0__ambiguous | sentence_packed | `fuse_max` | ambiguous | ambiguous | 0.9709 | 1.0000 | 0.0000 | False | 0.9815 | 199 |
| base_06_f0afeaf2_t1771_f0__ambiguous | sentence_packed | `fuse_min` | ambiguous | ambiguous | 0.9244 | 1.0000 | 0.0000 | False | 0.9674 | 199 |
| base_07_7b7fb621_t328_f1__correct | sentence_packed | `gte_only` | grounded | correct | 0.9776 | 1.0000 | 0.0000 | False | 0.9999 | 36 |
| base_07_7b7fb621_t328_f1__correct | sentence_packed | `code_only` | grounded | correct | 0.9373 | 1.0000 | 0.0000 | False | 1.0000 | 36 |
| base_07_7b7fb621_t328_f1__correct | sentence_packed | `fuse_mean` | grounded | correct | 0.9574 | 1.0000 | 0.0000 | False | 0.9999 | 36 |
| base_07_7b7fb621_t328_f1__correct | sentence_packed | `fuse_max` | grounded | correct | 0.9776 | 1.0000 | 0.0000 | False | 1.0000 | 36 |
| base_07_7b7fb621_t328_f1__correct | sentence_packed | `fuse_min` | grounded | correct | 0.9373 | 1.0000 | 0.0000 | False | 0.9999 | 36 |
| base_07_7b7fb621_t328_f1__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9703 | 1.0000 | 0.0000 | False | 0.9999 | 36 |
| base_07_7b7fb621_t328_f1__wrong | sentence_packed | `code_only` | ungrounded | wrong | 0.9277 | 1.0000 | 0.0000 | False | 1.0000 | 36 |
| base_07_7b7fb621_t328_f1__wrong | sentence_packed | `fuse_mean` | ungrounded | wrong | 0.9490 | 1.0000 | 0.0000 | False | 0.9999 | 36 |
| base_07_7b7fb621_t328_f1__wrong | sentence_packed | `fuse_max` | ungrounded | wrong | 0.9703 | 1.0000 | 0.0000 | False | 1.0000 | 36 |
| base_07_7b7fb621_t328_f1__wrong | sentence_packed | `fuse_min` | ungrounded | wrong | 0.9277 | 1.0000 | 0.0000 | False | 0.9999 | 36 |
| base_07_7b7fb621_t328_f1__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9777 | 1.0000 | 0.0000 | False | 1.0000 | 36 |
| base_07_7b7fb621_t328_f1__ambiguous | sentence_packed | `code_only` | ambiguous | ambiguous | 0.9359 | 1.0000 | 0.0000 | False | 1.0000 | 36 |
| base_07_7b7fb621_t328_f1__ambiguous | sentence_packed | `fuse_mean` | ambiguous | ambiguous | 0.9568 | 1.0000 | 0.0000 | False | 1.0000 | 36 |
| base_07_7b7fb621_t328_f1__ambiguous | sentence_packed | `fuse_max` | ambiguous | ambiguous | 0.9777 | 1.0000 | 0.0000 | False | 1.0000 | 36 |
| base_07_7b7fb621_t328_f1__ambiguous | sentence_packed | `fuse_min` | ambiguous | ambiguous | 0.9359 | 1.0000 | 0.0000 | False | 1.0000 | 36 |
| base_08_126e38fc_t1785_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9807 | 1.0000 | 0.0000 | False | 0.9983 | 210 |
| base_08_126e38fc_t1785_f0__correct | sentence_packed | `code_only` | grounded | correct | 0.9475 | 1.0000 | 0.0000 | False | 0.9885 | 210 |
| base_08_126e38fc_t1785_f0__correct | sentence_packed | `fuse_mean` | grounded | correct | 0.9641 | 1.0000 | 0.0000 | False | 0.9934 | 210 |
| base_08_126e38fc_t1785_f0__correct | sentence_packed | `fuse_max` | grounded | correct | 0.9807 | 1.0000 | 0.0000 | False | 0.9983 | 210 |
| base_08_126e38fc_t1785_f0__correct | sentence_packed | `fuse_min` | grounded | correct | 0.9475 | 1.0000 | 0.0000 | False | 0.9885 | 210 |
| base_08_126e38fc_t1785_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9792 | 1.0000 | 0.0000 | False | 0.9982 | 210 |
| base_08_126e38fc_t1785_f0__wrong | sentence_packed | `code_only` | ungrounded | wrong | 0.9500 | 1.0000 | 0.0000 | False | 0.9888 | 210 |
| base_08_126e38fc_t1785_f0__wrong | sentence_packed | `fuse_mean` | ungrounded | wrong | 0.9646 | 1.0000 | 0.0000 | False | 0.9935 | 210 |
| base_08_126e38fc_t1785_f0__wrong | sentence_packed | `fuse_max` | ungrounded | wrong | 0.9792 | 1.0000 | 0.0000 | False | 0.9982 | 210 |
| base_08_126e38fc_t1785_f0__wrong | sentence_packed | `fuse_min` | ungrounded | wrong | 0.9500 | 1.0000 | 0.0000 | False | 0.9888 | 210 |
| base_08_126e38fc_t1785_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9806 | 1.0000 | 0.0000 | False | 0.9983 | 210 |
| base_08_126e38fc_t1785_f0__ambiguous | sentence_packed | `code_only` | ambiguous | ambiguous | 0.9470 | 1.0000 | 0.0000 | False | 0.9884 | 210 |
| base_08_126e38fc_t1785_f0__ambiguous | sentence_packed | `fuse_mean` | ambiguous | ambiguous | 0.9638 | 1.0000 | 0.0000 | False | 0.9933 | 210 |
| base_08_126e38fc_t1785_f0__ambiguous | sentence_packed | `fuse_max` | ambiguous | ambiguous | 0.9806 | 1.0000 | 0.0000 | False | 0.9983 | 210 |
| base_08_126e38fc_t1785_f0__ambiguous | sentence_packed | `fuse_min` | ambiguous | ambiguous | 0.9470 | 1.0000 | 0.0000 | False | 0.9884 | 210 |
| base_09_0b1422bb_t571_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9584 | 1.0000 | 0.0000 | False | 0.9980 | 20 |
| base_09_0b1422bb_t571_f0__correct | sentence_packed | `code_only` | grounded | correct | 0.8929 | 1.0000 | 0.0000 | False | 0.9831 | 20 |
| base_09_0b1422bb_t571_f0__correct | sentence_packed | `fuse_mean` | grounded | correct | 0.9256 | 1.0000 | 0.0000 | False | 0.9906 | 20 |
| base_09_0b1422bb_t571_f0__correct | sentence_packed | `fuse_max` | grounded | correct | 0.9584 | 1.0000 | 0.0000 | False | 0.9980 | 20 |
| base_09_0b1422bb_t571_f0__correct | sentence_packed | `fuse_min` | grounded | correct | 0.8929 | 1.0000 | 0.0000 | False | 0.9831 | 20 |
| base_09_0b1422bb_t571_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9565 | 1.0000 | 0.0000 | False | 0.9979 | 20 |
| base_09_0b1422bb_t571_f0__wrong | sentence_packed | `code_only` | ungrounded | wrong | 0.8940 | 1.0000 | 0.0000 | False | 0.9897 | 20 |
| base_09_0b1422bb_t571_f0__wrong | sentence_packed | `fuse_mean` | ungrounded | wrong | 0.9253 | 1.0000 | 0.0000 | False | 0.9938 | 20 |
| base_09_0b1422bb_t571_f0__wrong | sentence_packed | `fuse_max` | ungrounded | wrong | 0.9565 | 1.0000 | 0.0000 | False | 0.9979 | 20 |
| base_09_0b1422bb_t571_f0__wrong | sentence_packed | `fuse_min` | ungrounded | wrong | 0.8940 | 1.0000 | 0.0000 | False | 0.9897 | 20 |
| base_09_0b1422bb_t571_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9582 | 1.0000 | 0.0000 | False | 0.9980 | 20 |
| base_09_0b1422bb_t571_f0__ambiguous | sentence_packed | `code_only` | ambiguous | ambiguous | 0.8937 | 1.0000 | 0.0000 | False | 0.9831 | 20 |
| base_09_0b1422bb_t571_f0__ambiguous | sentence_packed | `fuse_mean` | ambiguous | ambiguous | 0.9260 | 1.0000 | 0.0000 | False | 0.9906 | 20 |
| base_09_0b1422bb_t571_f0__ambiguous | sentence_packed | `fuse_max` | ambiguous | ambiguous | 0.9582 | 1.0000 | 0.0000 | False | 0.9980 | 20 |
| base_09_0b1422bb_t571_f0__ambiguous | sentence_packed | `fuse_min` | ambiguous | ambiguous | 0.8937 | 1.0000 | 0.0000 | False | 0.9831 | 20 |
| base_10_d7fd7127_t1579_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9830 | 1.0000 | 0.0000 | False | 0.9951 | 218 |
| base_10_d7fd7127_t1579_f0__correct | sentence_packed | `code_only` | grounded | correct | 0.9413 | 1.0000 | 0.0000 | False | 0.9860 | 218 |
| base_10_d7fd7127_t1579_f0__correct | sentence_packed | `fuse_mean` | grounded | correct | 0.9621 | 1.0000 | 0.0000 | False | 0.9905 | 218 |
| base_10_d7fd7127_t1579_f0__correct | sentence_packed | `fuse_max` | grounded | correct | 0.9830 | 1.0000 | 0.0000 | False | 0.9951 | 218 |
| base_10_d7fd7127_t1579_f0__correct | sentence_packed | `fuse_min` | grounded | correct | 0.9413 | 1.0000 | 0.0000 | False | 0.9860 | 218 |
| base_10_d7fd7127_t1579_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9729 | 1.0000 | 0.0000 | False | 0.9948 | 218 |
| base_10_d7fd7127_t1579_f0__wrong | sentence_packed | `code_only` | ungrounded | wrong | 0.9316 | 1.0000 | 0.0000 | False | 0.9838 | 218 |
| base_10_d7fd7127_t1579_f0__wrong | sentence_packed | `fuse_mean` | ungrounded | wrong | 0.9523 | 1.0000 | 0.0000 | False | 0.9893 | 218 |
| base_10_d7fd7127_t1579_f0__wrong | sentence_packed | `fuse_max` | ungrounded | wrong | 0.9729 | 1.0000 | 0.0000 | False | 0.9948 | 218 |
| base_10_d7fd7127_t1579_f0__wrong | sentence_packed | `fuse_min` | ungrounded | wrong | 0.9316 | 1.0000 | 0.0000 | False | 0.9838 | 218 |
| base_10_d7fd7127_t1579_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9828 | 1.0000 | 0.0000 | False | 0.9951 | 218 |
| base_10_d7fd7127_t1579_f0__ambiguous | sentence_packed | `code_only` | ambiguous | ambiguous | 0.9415 | 1.0000 | 0.0000 | False | 0.9859 | 218 |
| base_10_d7fd7127_t1579_f0__ambiguous | sentence_packed | `fuse_mean` | ambiguous | ambiguous | 0.9621 | 1.0000 | 0.0000 | False | 0.9905 | 218 |
| base_10_d7fd7127_t1579_f0__ambiguous | sentence_packed | `fuse_max` | ambiguous | ambiguous | 0.9828 | 1.0000 | 0.0000 | False | 0.9951 | 218 |
| base_10_d7fd7127_t1579_f0__ambiguous | sentence_packed | `fuse_min` | ambiguous | ambiguous | 0.9415 | 1.0000 | 0.0000 | False | 0.9859 | 218 |
| base_11_f0afeaf2_t1418_f1__correct | sentence_packed | `gte_only` | grounded | correct | 0.9850 | 1.0000 | 0.0000 | False | 0.9960 | 195 |
| base_11_f0afeaf2_t1418_f1__correct | sentence_packed | `code_only` | grounded | correct | 0.9516 | 1.0000 | 0.0000 | False | 0.9879 | 195 |
| base_11_f0afeaf2_t1418_f1__correct | sentence_packed | `fuse_mean` | grounded | correct | 0.9683 | 1.0000 | 0.0000 | False | 0.9919 | 195 |
| base_11_f0afeaf2_t1418_f1__correct | sentence_packed | `fuse_max` | grounded | correct | 0.9850 | 1.0000 | 0.0000 | False | 0.9960 | 195 |
| base_11_f0afeaf2_t1418_f1__correct | sentence_packed | `fuse_min` | grounded | correct | 0.9516 | 1.0000 | 0.0000 | False | 0.9879 | 195 |
| base_11_f0afeaf2_t1418_f1__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9731 | 1.0000 | 0.0000 | False | 0.9964 | 195 |
| base_11_f0afeaf2_t1418_f1__wrong | sentence_packed | `code_only` | ungrounded | wrong | 0.9271 | 1.0000 | 0.0000 | False | 0.9799 | 195 |
| base_11_f0afeaf2_t1418_f1__wrong | sentence_packed | `fuse_mean` | ungrounded | wrong | 0.9501 | 1.0000 | 0.0000 | False | 0.9882 | 195 |
| base_11_f0afeaf2_t1418_f1__wrong | sentence_packed | `fuse_max` | ungrounded | wrong | 0.9731 | 1.0000 | 0.0000 | False | 0.9964 | 195 |
| base_11_f0afeaf2_t1418_f1__wrong | sentence_packed | `fuse_min` | ungrounded | wrong | 0.9271 | 1.0000 | 0.0000 | False | 0.9799 | 195 |
| base_11_f0afeaf2_t1418_f1__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9848 | 1.0000 | 0.0000 | False | 0.9961 | 195 |
| base_11_f0afeaf2_t1418_f1__ambiguous | sentence_packed | `code_only` | ambiguous | ambiguous | 0.9494 | 1.0000 | 0.0000 | False | 0.9881 | 195 |
| base_11_f0afeaf2_t1418_f1__ambiguous | sentence_packed | `fuse_mean` | ambiguous | ambiguous | 0.9671 | 1.0000 | 0.0000 | False | 0.9921 | 195 |
| base_11_f0afeaf2_t1418_f1__ambiguous | sentence_packed | `fuse_max` | ambiguous | ambiguous | 0.9848 | 1.0000 | 0.0000 | False | 0.9961 | 195 |
| base_11_f0afeaf2_t1418_f1__ambiguous | sentence_packed | `fuse_min` | ambiguous | ambiguous | 0.9494 | 1.0000 | 0.0000 | False | 0.9881 | 195 |
| base_12_f0afeaf2_t955_f2__correct | sentence_packed | `gte_only` | grounded | correct | 0.9757 | 1.0000 | 0.0000 | False | 0.9946 | 145 |
| base_12_f0afeaf2_t955_f2__correct | sentence_packed | `code_only` | grounded | correct | 0.9124 | 1.0000 | 0.0000 | False | 0.9894 | 145 |
| base_12_f0afeaf2_t955_f2__correct | sentence_packed | `fuse_mean` | grounded | correct | 0.9440 | 1.0000 | 0.0000 | False | 0.9920 | 145 |
| base_12_f0afeaf2_t955_f2__correct | sentence_packed | `fuse_max` | grounded | correct | 0.9757 | 1.0000 | 0.0000 | False | 0.9946 | 145 |
| base_12_f0afeaf2_t955_f2__correct | sentence_packed | `fuse_min` | grounded | correct | 0.9124 | 1.0000 | 0.0000 | False | 0.9894 | 145 |
| base_12_f0afeaf2_t955_f2__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9673 | 1.0000 | 0.0000 | False | 1.0000 | 145 |
| base_12_f0afeaf2_t955_f2__wrong | sentence_packed | `code_only` | ungrounded | wrong | 0.9165 | 1.0000 | 0.0000 | False | 1.0000 | 145 |
| base_12_f0afeaf2_t955_f2__wrong | sentence_packed | `fuse_mean` | ungrounded | wrong | 0.9419 | 1.0000 | 0.0000 | False | 1.0000 | 145 |
| base_12_f0afeaf2_t955_f2__wrong | sentence_packed | `fuse_max` | ungrounded | wrong | 0.9673 | 1.0000 | 0.0000 | False | 1.0000 | 145 |
| base_12_f0afeaf2_t955_f2__wrong | sentence_packed | `fuse_min` | ungrounded | wrong | 0.9165 | 1.0000 | 0.0000 | False | 1.0000 | 145 |
| base_12_f0afeaf2_t955_f2__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9781 | 1.0000 | 0.0000 | False | 0.9946 | 145 |
| base_12_f0afeaf2_t955_f2__ambiguous | sentence_packed | `code_only` | ambiguous | ambiguous | 0.9088 | 1.0000 | 0.0000 | False | 0.9823 | 145 |
| base_12_f0afeaf2_t955_f2__ambiguous | sentence_packed | `fuse_mean` | ambiguous | ambiguous | 0.9434 | 1.0000 | 0.0000 | False | 0.9884 | 145 |
| base_12_f0afeaf2_t955_f2__ambiguous | sentence_packed | `fuse_max` | ambiguous | ambiguous | 0.9781 | 1.0000 | 0.0000 | False | 0.9946 | 145 |
| base_12_f0afeaf2_t955_f2__ambiguous | sentence_packed | `fuse_min` | ambiguous | ambiguous | 0.9088 | 1.0000 | 0.0000 | False | 0.9823 | 145 |
| base_13_d7fd7127_t1553_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9851 | 1.0000 | 0.0000 | False | 1.0000 | 218 |
| base_13_d7fd7127_t1553_f0__correct | sentence_packed | `code_only` | grounded | correct | 0.9589 | 1.0000 | 0.0000 | False | 0.9920 | 218 |
| base_13_d7fd7127_t1553_f0__correct | sentence_packed | `fuse_mean` | grounded | correct | 0.9720 | 1.0000 | 0.0000 | False | 0.9960 | 218 |
| base_13_d7fd7127_t1553_f0__correct | sentence_packed | `fuse_max` | grounded | correct | 0.9851 | 1.0000 | 0.0000 | False | 1.0000 | 218 |
| base_13_d7fd7127_t1553_f0__correct | sentence_packed | `fuse_min` | grounded | correct | 0.9589 | 1.0000 | 0.0000 | False | 0.9920 | 218 |
| base_13_d7fd7127_t1553_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9720 | 1.0000 | 0.0000 | False | 1.0000 | 218 |
| base_13_d7fd7127_t1553_f0__wrong | sentence_packed | `code_only` | ungrounded | wrong | 0.9373 | 1.0000 | 0.0000 | False | 0.9868 | 218 |
| base_13_d7fd7127_t1553_f0__wrong | sentence_packed | `fuse_mean` | ungrounded | wrong | 0.9547 | 1.0000 | 0.0000 | False | 0.9934 | 218 |
| base_13_d7fd7127_t1553_f0__wrong | sentence_packed | `fuse_max` | ungrounded | wrong | 0.9720 | 1.0000 | 0.0000 | False | 1.0000 | 218 |
| base_13_d7fd7127_t1553_f0__wrong | sentence_packed | `fuse_min` | ungrounded | wrong | 0.9373 | 1.0000 | 0.0000 | False | 0.9868 | 218 |
| base_13_d7fd7127_t1553_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9851 | 1.0000 | 0.0000 | False | 1.0000 | 218 |
| base_13_d7fd7127_t1553_f0__ambiguous | sentence_packed | `code_only` | ambiguous | ambiguous | 0.9586 | 1.0000 | 0.0000 | False | 0.9920 | 218 |
| base_13_d7fd7127_t1553_f0__ambiguous | sentence_packed | `fuse_mean` | ambiguous | ambiguous | 0.9718 | 1.0000 | 0.0000 | False | 0.9960 | 218 |
| base_13_d7fd7127_t1553_f0__ambiguous | sentence_packed | `fuse_max` | ambiguous | ambiguous | 0.9851 | 1.0000 | 0.0000 | False | 1.0000 | 218 |
| base_13_d7fd7127_t1553_f0__ambiguous | sentence_packed | `fuse_min` | ambiguous | ambiguous | 0.9586 | 1.0000 | 0.0000 | False | 0.9920 | 218 |
| base_14_f0afeaf2_t2825_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9842 | 1.0000 | 0.0000 | False | 1.0000 | 200 |
| base_14_f0afeaf2_t2825_f0__correct | sentence_packed | `code_only` | grounded | correct | 0.9520 | 1.0000 | 0.0000 | False | 1.0000 | 200 |
| base_14_f0afeaf2_t2825_f0__correct | sentence_packed | `fuse_mean` | grounded | correct | 0.9681 | 1.0000 | 0.0000 | False | 1.0000 | 200 |
| base_14_f0afeaf2_t2825_f0__correct | sentence_packed | `fuse_max` | grounded | correct | 0.9842 | 1.0000 | 0.0000 | False | 1.0000 | 200 |
| base_14_f0afeaf2_t2825_f0__correct | sentence_packed | `fuse_min` | grounded | correct | 0.9520 | 1.0000 | 0.0000 | False | 1.0000 | 200 |
| base_14_f0afeaf2_t2825_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9756 | 1.0000 | 0.0000 | False | 1.0000 | 200 |
| base_14_f0afeaf2_t2825_f0__wrong | sentence_packed | `code_only` | ungrounded | wrong | 0.9445 | 1.0000 | 0.0000 | False | 0.9999 | 200 |
| base_14_f0afeaf2_t2825_f0__wrong | sentence_packed | `fuse_mean` | ungrounded | wrong | 0.9600 | 1.0000 | 0.0000 | False | 1.0000 | 200 |
| base_14_f0afeaf2_t2825_f0__wrong | sentence_packed | `fuse_max` | ungrounded | wrong | 0.9756 | 1.0000 | 0.0000 | False | 1.0000 | 200 |
| base_14_f0afeaf2_t2825_f0__wrong | sentence_packed | `fuse_min` | ungrounded | wrong | 0.9445 | 1.0000 | 0.0000 | False | 0.9999 | 200 |
| base_14_f0afeaf2_t2825_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9839 | 1.0000 | 0.0000 | False | 0.9992 | 200 |
| base_14_f0afeaf2_t2825_f0__ambiguous | sentence_packed | `code_only` | ambiguous | ambiguous | 0.9530 | 1.0000 | 0.0000 | False | 0.9999 | 200 |
| base_14_f0afeaf2_t2825_f0__ambiguous | sentence_packed | `fuse_mean` | ambiguous | ambiguous | 0.9684 | 1.0000 | 0.0000 | False | 0.9996 | 200 |
| base_14_f0afeaf2_t2825_f0__ambiguous | sentence_packed | `fuse_max` | ambiguous | ambiguous | 0.9839 | 1.0000 | 0.0000 | False | 0.9999 | 200 |
| base_14_f0afeaf2_t2825_f0__ambiguous | sentence_packed | `fuse_min` | ambiguous | ambiguous | 0.9530 | 1.0000 | 0.0000 | False | 0.9992 | 200 |
| base_15_91cf0ec9_t419_f2__correct | sentence_packed | `gte_only` | grounded | correct | 0.9825 | 1.0000 | 0.0000 | False | 0.9995 | 110 |
| base_15_91cf0ec9_t419_f2__correct | sentence_packed | `code_only` | grounded | correct | 0.9299 | 1.0000 | 0.0000 | False | 0.9844 | 110 |
| base_15_91cf0ec9_t419_f2__correct | sentence_packed | `fuse_mean` | grounded | correct | 0.9562 | 1.0000 | 0.0000 | False | 0.9920 | 110 |
| base_15_91cf0ec9_t419_f2__correct | sentence_packed | `fuse_max` | grounded | correct | 0.9825 | 1.0000 | 0.0000 | False | 0.9995 | 110 |
| base_15_91cf0ec9_t419_f2__correct | sentence_packed | `fuse_min` | grounded | correct | 0.9299 | 1.0000 | 0.0000 | False | 0.9844 | 110 |
| base_15_91cf0ec9_t419_f2__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9779 | 1.0000 | 0.0000 | False | 0.9998 | 110 |
| base_15_91cf0ec9_t419_f2__wrong | sentence_packed | `code_only` | ungrounded | wrong | 0.9276 | 1.0000 | 0.0000 | False | 1.0000 | 110 |
| base_15_91cf0ec9_t419_f2__wrong | sentence_packed | `fuse_mean` | ungrounded | wrong | 0.9527 | 1.0000 | 0.0000 | False | 0.9999 | 110 |
| base_15_91cf0ec9_t419_f2__wrong | sentence_packed | `fuse_max` | ungrounded | wrong | 0.9779 | 1.0000 | 0.0000 | False | 1.0000 | 110 |
| base_15_91cf0ec9_t419_f2__wrong | sentence_packed | `fuse_min` | ungrounded | wrong | 0.9276 | 1.0000 | 0.0000 | False | 0.9998 | 110 |
| base_15_91cf0ec9_t419_f2__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9821 | 1.0000 | 0.0000 | False | 0.9995 | 110 |
| base_15_91cf0ec9_t419_f2__ambiguous | sentence_packed | `code_only` | ambiguous | ambiguous | 0.9308 | 1.0000 | 0.0000 | False | 0.9844 | 110 |
| base_15_91cf0ec9_t419_f2__ambiguous | sentence_packed | `fuse_mean` | ambiguous | ambiguous | 0.9565 | 1.0000 | 0.0000 | False | 0.9920 | 110 |
| base_15_91cf0ec9_t419_f2__ambiguous | sentence_packed | `fuse_max` | ambiguous | ambiguous | 0.9821 | 1.0000 | 0.0000 | False | 0.9995 | 110 |
| base_15_91cf0ec9_t419_f2__ambiguous | sentence_packed | `fuse_min` | ambiguous | ambiguous | 0.9308 | 1.0000 | 0.0000 | False | 0.9844 | 110 |
| base_16_8a0f6f68_t1002_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9810 | 1.0000 | 0.0000 | False | 0.9956 | 93 |
| base_16_8a0f6f68_t1002_f0__correct | sentence_packed | `code_only` | grounded | correct | 0.9482 | 1.0000 | 0.0000 | False | 0.9647 | 93 |
| base_16_8a0f6f68_t1002_f0__correct | sentence_packed | `fuse_mean` | grounded | correct | 0.9646 | 1.0000 | 0.0000 | False | 0.9802 | 93 |
| base_16_8a0f6f68_t1002_f0__correct | sentence_packed | `fuse_max` | grounded | correct | 0.9810 | 1.0000 | 0.0000 | False | 0.9956 | 93 |
| base_16_8a0f6f68_t1002_f0__correct | sentence_packed | `fuse_min` | grounded | correct | 0.9482 | 1.0000 | 0.0000 | False | 0.9647 | 93 |
| base_16_8a0f6f68_t1002_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9738 | 1.0000 | 0.0000 | False | 1.0000 | 93 |
| base_16_8a0f6f68_t1002_f0__wrong | sentence_packed | `code_only` | ungrounded | wrong | 0.9382 | 1.0000 | 0.0000 | False | 1.0000 | 93 |
| base_16_8a0f6f68_t1002_f0__wrong | sentence_packed | `fuse_mean` | ungrounded | wrong | 0.9560 | 1.0000 | 0.0000 | False | 1.0000 | 93 |
| base_16_8a0f6f68_t1002_f0__wrong | sentence_packed | `fuse_max` | ungrounded | wrong | 0.9738 | 1.0000 | 0.0000 | False | 1.0000 | 93 |
| base_16_8a0f6f68_t1002_f0__wrong | sentence_packed | `fuse_min` | ungrounded | wrong | 0.9382 | 1.0000 | 0.0000 | False | 1.0000 | 93 |
| base_16_8a0f6f68_t1002_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9808 | 1.0000 | 0.0000 | False | 0.9971 | 93 |
| base_16_8a0f6f68_t1002_f0__ambiguous | sentence_packed | `code_only` | ambiguous | ambiguous | 0.9481 | 1.0000 | 0.0000 | False | 0.9854 | 93 |
| base_16_8a0f6f68_t1002_f0__ambiguous | sentence_packed | `fuse_mean` | ambiguous | ambiguous | 0.9644 | 1.0000 | 0.0000 | False | 0.9912 | 93 |
| base_16_8a0f6f68_t1002_f0__ambiguous | sentence_packed | `fuse_max` | ambiguous | ambiguous | 0.9808 | 1.0000 | 0.0000 | False | 0.9971 | 93 |
| base_16_8a0f6f68_t1002_f0__ambiguous | sentence_packed | `fuse_min` | ambiguous | ambiguous | 0.9481 | 1.0000 | 0.0000 | False | 0.9854 | 93 |
| base_17_126e38fc_t712_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9797 | 1.0000 | 0.0000 | False | 0.9959 | 58 |
| base_17_126e38fc_t712_f0__correct | sentence_packed | `code_only` | grounded | correct | 0.9451 | 1.0000 | 0.0000 | False | 0.9837 | 58 |
| base_17_126e38fc_t712_f0__correct | sentence_packed | `fuse_mean` | grounded | correct | 0.9624 | 1.0000 | 0.0000 | False | 0.9898 | 58 |
| base_17_126e38fc_t712_f0__correct | sentence_packed | `fuse_max` | grounded | correct | 0.9797 | 1.0000 | 0.0000 | False | 0.9959 | 58 |
| base_17_126e38fc_t712_f0__correct | sentence_packed | `fuse_min` | grounded | correct | 0.9451 | 1.0000 | 0.0000 | False | 0.9837 | 58 |
| base_17_126e38fc_t712_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9762 | 1.0000 | 0.0000 | False | 0.9945 | 58 |
| base_17_126e38fc_t712_f0__wrong | sentence_packed | `code_only` | ungrounded | wrong | 0.9385 | 1.0000 | 0.0000 | False | 0.9854 | 58 |
| base_17_126e38fc_t712_f0__wrong | sentence_packed | `fuse_mean` | ungrounded | wrong | 0.9573 | 1.0000 | 0.0000 | False | 0.9900 | 58 |
| base_17_126e38fc_t712_f0__wrong | sentence_packed | `fuse_max` | ungrounded | wrong | 0.9762 | 1.0000 | 0.0000 | False | 0.9945 | 58 |
| base_17_126e38fc_t712_f0__wrong | sentence_packed | `fuse_min` | ungrounded | wrong | 0.9385 | 1.0000 | 0.0000 | False | 0.9854 | 58 |
| base_17_126e38fc_t712_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9797 | 1.0000 | 0.0000 | False | 0.9959 | 58 |
| base_17_126e38fc_t712_f0__ambiguous | sentence_packed | `code_only` | ambiguous | ambiguous | 0.9446 | 1.0000 | 0.0000 | False | 0.9840 | 58 |
| base_17_126e38fc_t712_f0__ambiguous | sentence_packed | `fuse_mean` | ambiguous | ambiguous | 0.9622 | 1.0000 | 0.0000 | False | 0.9900 | 58 |
| base_17_126e38fc_t712_f0__ambiguous | sentence_packed | `fuse_max` | ambiguous | ambiguous | 0.9797 | 1.0000 | 0.0000 | False | 0.9959 | 58 |
| base_17_126e38fc_t712_f0__ambiguous | sentence_packed | `fuse_min` | ambiguous | ambiguous | 0.9446 | 1.0000 | 0.0000 | False | 0.9840 | 58 |
| base_18_f0afeaf2_t457_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9789 | 1.0000 | 0.0000 | False | 1.0000 | 89 |
| base_18_f0afeaf2_t457_f0__correct | sentence_packed | `code_only` | grounded | correct | 0.9240 | 1.0000 | 0.0000 | False | 0.9879 | 89 |
| base_18_f0afeaf2_t457_f0__correct | sentence_packed | `fuse_mean` | grounded | correct | 0.9515 | 1.0000 | 0.0000 | False | 0.9939 | 89 |
| base_18_f0afeaf2_t457_f0__correct | sentence_packed | `fuse_max` | grounded | correct | 0.9789 | 1.0000 | 0.0000 | False | 1.0000 | 89 |
| base_18_f0afeaf2_t457_f0__correct | sentence_packed | `fuse_min` | grounded | correct | 0.9240 | 1.0000 | 0.0000 | False | 0.9879 | 89 |
| base_18_f0afeaf2_t457_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9697 | 1.0000 | 0.0000 | False | 0.9963 | 89 |
| base_18_f0afeaf2_t457_f0__wrong | sentence_packed | `code_only` | ungrounded | wrong | 0.9103 | 1.0000 | 0.0000 | False | 0.9853 | 89 |
| base_18_f0afeaf2_t457_f0__wrong | sentence_packed | `fuse_mean` | ungrounded | wrong | 0.9400 | 1.0000 | 0.0000 | False | 0.9908 | 89 |
| base_18_f0afeaf2_t457_f0__wrong | sentence_packed | `fuse_max` | ungrounded | wrong | 0.9697 | 1.0000 | 0.0000 | False | 0.9963 | 89 |
| base_18_f0afeaf2_t457_f0__wrong | sentence_packed | `fuse_min` | ungrounded | wrong | 0.9103 | 1.0000 | 0.0000 | False | 0.9853 | 89 |
| base_18_f0afeaf2_t457_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9800 | 1.0000 | 0.0000 | False | 0.9956 | 89 |
| base_18_f0afeaf2_t457_f0__ambiguous | sentence_packed | `code_only` | ambiguous | ambiguous | 0.9257 | 1.0000 | 0.0000 | False | 0.9860 | 89 |
| base_18_f0afeaf2_t457_f0__ambiguous | sentence_packed | `fuse_mean` | ambiguous | ambiguous | 0.9528 | 1.0000 | 0.0000 | False | 0.9908 | 89 |
| base_18_f0afeaf2_t457_f0__ambiguous | sentence_packed | `fuse_max` | ambiguous | ambiguous | 0.9800 | 1.0000 | 0.0000 | False | 0.9956 | 89 |
| base_18_f0afeaf2_t457_f0__ambiguous | sentence_packed | `fuse_min` | ambiguous | ambiguous | 0.9257 | 1.0000 | 0.0000 | False | 0.9860 | 89 |
| base_19_5e875b7e_t1360_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9783 | 1.0000 | 0.0000 | False | 0.9948 | 60 |
| base_19_5e875b7e_t1360_f0__correct | sentence_packed | `code_only` | grounded | correct | 0.9499 | 1.0000 | 0.0000 | False | 0.9887 | 60 |
| base_19_5e875b7e_t1360_f0__correct | sentence_packed | `fuse_mean` | grounded | correct | 0.9641 | 1.0000 | 0.0000 | False | 0.9918 | 60 |
| base_19_5e875b7e_t1360_f0__correct | sentence_packed | `fuse_max` | grounded | correct | 0.9783 | 1.0000 | 0.0000 | False | 0.9948 | 60 |
| base_19_5e875b7e_t1360_f0__correct | sentence_packed | `fuse_min` | grounded | correct | 0.9499 | 1.0000 | 0.0000 | False | 0.9887 | 60 |
| base_19_5e875b7e_t1360_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9740 | 1.0000 | 0.0000 | False | 0.9971 | 60 |
| base_19_5e875b7e_t1360_f0__wrong | sentence_packed | `code_only` | ungrounded | wrong | 0.9379 | 1.0000 | 0.0000 | False | 0.9865 | 60 |
| base_19_5e875b7e_t1360_f0__wrong | sentence_packed | `fuse_mean` | ungrounded | wrong | 0.9559 | 1.0000 | 0.0000 | False | 0.9918 | 60 |
| base_19_5e875b7e_t1360_f0__wrong | sentence_packed | `fuse_max` | ungrounded | wrong | 0.9740 | 1.0000 | 0.0000 | False | 0.9971 | 60 |
| base_19_5e875b7e_t1360_f0__wrong | sentence_packed | `fuse_min` | ungrounded | wrong | 0.9379 | 1.0000 | 0.0000 | False | 0.9865 | 60 |
| base_19_5e875b7e_t1360_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9782 | 1.0000 | 0.0000 | False | 0.9969 | 60 |
| base_19_5e875b7e_t1360_f0__ambiguous | sentence_packed | `code_only` | ambiguous | ambiguous | 0.9511 | 1.0000 | 0.0000 | False | 0.9887 | 60 |
| base_19_5e875b7e_t1360_f0__ambiguous | sentence_packed | `fuse_mean` | ambiguous | ambiguous | 0.9646 | 1.0000 | 0.0000 | False | 0.9928 | 60 |
| base_19_5e875b7e_t1360_f0__ambiguous | sentence_packed | `fuse_max` | ambiguous | ambiguous | 0.9782 | 1.0000 | 0.0000 | False | 0.9969 | 60 |
| base_19_5e875b7e_t1360_f0__ambiguous | sentence_packed | `fuse_min` | ambiguous | ambiguous | 0.9511 | 1.0000 | 0.0000 | False | 0.9887 | 60 |
| base_20_d7fd7127_t1617_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9876 | 1.0000 | 0.0000 | False | 0.9988 | 226 |
| base_20_d7fd7127_t1617_f0__correct | sentence_packed | `code_only` | grounded | correct | 0.9535 | 1.0000 | 0.0000 | False | 0.9896 | 226 |
| base_20_d7fd7127_t1617_f0__correct | sentence_packed | `fuse_mean` | grounded | correct | 0.9705 | 1.0000 | 0.0000 | False | 0.9942 | 226 |
| base_20_d7fd7127_t1617_f0__correct | sentence_packed | `fuse_max` | grounded | correct | 0.9876 | 1.0000 | 0.0000 | False | 0.9988 | 226 |
| base_20_d7fd7127_t1617_f0__correct | sentence_packed | `fuse_min` | grounded | correct | 0.9535 | 1.0000 | 0.0000 | False | 0.9896 | 226 |
| base_20_d7fd7127_t1617_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9870 | 1.0000 | 0.0000 | False | 0.9988 | 226 |
| base_20_d7fd7127_t1617_f0__wrong | sentence_packed | `code_only` | ungrounded | wrong | 0.9506 | 1.0000 | 0.0000 | False | 0.9892 | 226 |
| base_20_d7fd7127_t1617_f0__wrong | sentence_packed | `fuse_mean` | ungrounded | wrong | 0.9688 | 1.0000 | 0.0000 | False | 0.9940 | 226 |
| base_20_d7fd7127_t1617_f0__wrong | sentence_packed | `fuse_max` | ungrounded | wrong | 0.9870 | 1.0000 | 0.0000 | False | 0.9988 | 226 |
| base_20_d7fd7127_t1617_f0__wrong | sentence_packed | `fuse_min` | ungrounded | wrong | 0.9506 | 1.0000 | 0.0000 | False | 0.9892 | 226 |
| base_20_d7fd7127_t1617_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9874 | 1.0000 | 0.0000 | False | 0.9988 | 226 |
| base_20_d7fd7127_t1617_f0__ambiguous | sentence_packed | `code_only` | ambiguous | ambiguous | 0.9545 | 1.0000 | 0.0000 | False | 0.9904 | 226 |
| base_20_d7fd7127_t1617_f0__ambiguous | sentence_packed | `fuse_mean` | ambiguous | ambiguous | 0.9709 | 1.0000 | 0.0000 | False | 0.9946 | 226 |
| base_20_d7fd7127_t1617_f0__ambiguous | sentence_packed | `fuse_max` | ambiguous | ambiguous | 0.9874 | 1.0000 | 0.0000 | False | 0.9988 | 226 |
| base_20_d7fd7127_t1617_f0__ambiguous | sentence_packed | `fuse_min` | ambiguous | ambiguous | 0.9545 | 1.0000 | 0.0000 | False | 0.9904 | 226 |
| base_21_a4d14a56_t732_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9824 | 1.0000 | 0.0000 | False | 1.0000 | 93 |
| base_21_a4d14a56_t732_f0__correct | sentence_packed | `code_only` | grounded | correct | 0.9425 | 1.0000 | 0.0000 | False | 0.9999 | 93 |
| base_21_a4d14a56_t732_f0__correct | sentence_packed | `fuse_mean` | grounded | correct | 0.9625 | 1.0000 | 0.0000 | False | 0.9999 | 93 |
| base_21_a4d14a56_t732_f0__correct | sentence_packed | `fuse_max` | grounded | correct | 0.9824 | 1.0000 | 0.0000 | False | 1.0000 | 93 |
| base_21_a4d14a56_t732_f0__correct | sentence_packed | `fuse_min` | grounded | correct | 0.9425 | 1.0000 | 0.0000 | False | 0.9999 | 93 |
| base_21_a4d14a56_t732_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9723 | 1.0000 | 0.0000 | False | 1.0000 | 93 |
| base_21_a4d14a56_t732_f0__wrong | sentence_packed | `code_only` | ungrounded | wrong | 0.9266 | 1.0000 | 0.0000 | False | 0.9999 | 93 |
| base_21_a4d14a56_t732_f0__wrong | sentence_packed | `fuse_mean` | ungrounded | wrong | 0.9495 | 1.0000 | 0.0000 | False | 1.0000 | 93 |
| base_21_a4d14a56_t732_f0__wrong | sentence_packed | `fuse_max` | ungrounded | wrong | 0.9723 | 1.0000 | 0.0000 | False | 1.0000 | 93 |
| base_21_a4d14a56_t732_f0__wrong | sentence_packed | `fuse_min` | ungrounded | wrong | 0.9266 | 1.0000 | 0.0000 | False | 0.9999 | 93 |
| base_21_a4d14a56_t732_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9778 | 1.0000 | 0.0000 | False | 1.0000 | 93 |
| base_21_a4d14a56_t732_f0__ambiguous | sentence_packed | `code_only` | ambiguous | ambiguous | 0.9323 | 1.0000 | 0.0000 | False | 0.9999 | 93 |
| base_21_a4d14a56_t732_f0__ambiguous | sentence_packed | `fuse_mean` | ambiguous | ambiguous | 0.9551 | 1.0000 | 0.0000 | False | 1.0000 | 93 |
| base_21_a4d14a56_t732_f0__ambiguous | sentence_packed | `fuse_max` | ambiguous | ambiguous | 0.9778 | 1.0000 | 0.0000 | False | 1.0000 | 93 |
| base_21_a4d14a56_t732_f0__ambiguous | sentence_packed | `fuse_min` | ambiguous | ambiguous | 0.9323 | 1.0000 | 0.0000 | False | 0.9999 | 93 |
| base_22_d7fd7127_t1209_f1__correct | sentence_packed | `gte_only` | grounded | correct | 0.9744 | 1.0000 | 0.0000 | False | 1.0000 | 165 |
| base_22_d7fd7127_t1209_f1__correct | sentence_packed | `code_only` | grounded | correct | 0.9291 | 1.0000 | 0.0000 | False | 0.9773 | 165 |
| base_22_d7fd7127_t1209_f1__correct | sentence_packed | `fuse_mean` | grounded | correct | 0.9518 | 1.0000 | 0.0000 | False | 0.9886 | 165 |
| base_22_d7fd7127_t1209_f1__correct | sentence_packed | `fuse_max` | grounded | correct | 0.9744 | 1.0000 | 0.0000 | False | 1.0000 | 165 |
| base_22_d7fd7127_t1209_f1__correct | sentence_packed | `fuse_min` | grounded | correct | 0.9291 | 1.0000 | 0.0000 | False | 0.9773 | 165 |
| base_22_d7fd7127_t1209_f1__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9748 | 1.0000 | 0.0000 | False | 1.0000 | 165 |
| base_22_d7fd7127_t1209_f1__wrong | sentence_packed | `code_only` | ungrounded | wrong | 0.9302 | 1.0000 | 0.0000 | False | 0.9818 | 165 |
| base_22_d7fd7127_t1209_f1__wrong | sentence_packed | `fuse_mean` | ungrounded | wrong | 0.9525 | 1.0000 | 0.0000 | False | 0.9909 | 165 |
| base_22_d7fd7127_t1209_f1__wrong | sentence_packed | `fuse_max` | ungrounded | wrong | 0.9748 | 1.0000 | 0.0000 | False | 1.0000 | 165 |
| base_22_d7fd7127_t1209_f1__wrong | sentence_packed | `fuse_min` | ungrounded | wrong | 0.9302 | 1.0000 | 0.0000 | False | 0.9818 | 165 |
| base_22_d7fd7127_t1209_f1__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9746 | 1.0000 | 0.0000 | False | 1.0000 | 165 |
| base_22_d7fd7127_t1209_f1__ambiguous | sentence_packed | `code_only` | ambiguous | ambiguous | 0.9310 | 1.0000 | 0.0000 | False | 0.9776 | 165 |
| base_22_d7fd7127_t1209_f1__ambiguous | sentence_packed | `fuse_mean` | ambiguous | ambiguous | 0.9528 | 1.0000 | 0.0000 | False | 0.9888 | 165 |
| base_22_d7fd7127_t1209_f1__ambiguous | sentence_packed | `fuse_max` | ambiguous | ambiguous | 0.9746 | 1.0000 | 0.0000 | False | 1.0000 | 165 |
| base_22_d7fd7127_t1209_f1__ambiguous | sentence_packed | `fuse_min` | ambiguous | ambiguous | 0.9310 | 1.0000 | 0.0000 | False | 0.9776 | 165 |
| base_23_a4d14a56_t579_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9790 | 1.0000 | 0.0000 | False | 0.9975 | 68 |
| base_23_a4d14a56_t579_f0__correct | sentence_packed | `code_only` | grounded | correct | 0.9276 | 1.0000 | 0.0000 | False | 0.9998 | 68 |
| base_23_a4d14a56_t579_f0__correct | sentence_packed | `fuse_mean` | grounded | correct | 0.9533 | 1.0000 | 0.0000 | False | 0.9987 | 68 |
| base_23_a4d14a56_t579_f0__correct | sentence_packed | `fuse_max` | grounded | correct | 0.9790 | 1.0000 | 0.0000 | False | 0.9998 | 68 |
| base_23_a4d14a56_t579_f0__correct | sentence_packed | `fuse_min` | grounded | correct | 0.9276 | 1.0000 | 0.0000 | False | 0.9975 | 68 |
| base_23_a4d14a56_t579_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9672 | 1.0000 | 0.0000 | False | 0.9981 | 68 |
| base_23_a4d14a56_t579_f0__wrong | sentence_packed | `code_only` | ungrounded | wrong | 0.9126 | 1.0000 | 0.0000 | False | 0.9871 | 68 |
| base_23_a4d14a56_t579_f0__wrong | sentence_packed | `fuse_mean` | ungrounded | wrong | 0.9399 | 1.0000 | 0.0000 | False | 0.9926 | 68 |
| base_23_a4d14a56_t579_f0__wrong | sentence_packed | `fuse_max` | ungrounded | wrong | 0.9672 | 1.0000 | 0.0000 | False | 0.9981 | 68 |
| base_23_a4d14a56_t579_f0__wrong | sentence_packed | `fuse_min` | ungrounded | wrong | 0.9126 | 1.0000 | 0.0000 | False | 0.9871 | 68 |
| base_23_a4d14a56_t579_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9788 | 1.0000 | 0.0000 | False | 0.9976 | 68 |
| base_23_a4d14a56_t579_f0__ambiguous | sentence_packed | `code_only` | ambiguous | ambiguous | 0.9291 | 1.0000 | 0.0000 | False | 0.9998 | 68 |
| base_23_a4d14a56_t579_f0__ambiguous | sentence_packed | `fuse_mean` | ambiguous | ambiguous | 0.9539 | 1.0000 | 0.0000 | False | 0.9987 | 68 |
| base_23_a4d14a56_t579_f0__ambiguous | sentence_packed | `fuse_max` | ambiguous | ambiguous | 0.9788 | 1.0000 | 0.0000 | False | 0.9998 | 68 |
| base_23_a4d14a56_t579_f0__ambiguous | sentence_packed | `fuse_min` | ambiguous | ambiguous | 0.9291 | 1.0000 | 0.0000 | False | 0.9976 | 68 |
| base_24_126e38fc_t1774_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9788 | 1.0000 | 0.0000 | False | 0.9999 | 210 |
| base_24_126e38fc_t1774_f0__correct | sentence_packed | `code_only` | grounded | correct | 0.9241 | 1.0000 | 0.0000 | False | 0.9832 | 210 |
| base_24_126e38fc_t1774_f0__correct | sentence_packed | `fuse_mean` | grounded | correct | 0.9515 | 1.0000 | 0.0000 | False | 0.9916 | 210 |
| base_24_126e38fc_t1774_f0__correct | sentence_packed | `fuse_max` | grounded | correct | 0.9788 | 1.0000 | 0.0000 | False | 0.9999 | 210 |
| base_24_126e38fc_t1774_f0__correct | sentence_packed | `fuse_min` | grounded | correct | 0.9241 | 1.0000 | 0.0000 | False | 0.9832 | 210 |
| base_24_126e38fc_t1774_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9706 | 1.0000 | 0.0000 | False | 1.0000 | 210 |
| base_24_126e38fc_t1774_f0__wrong | sentence_packed | `code_only` | ungrounded | wrong | 0.9161 | 1.0000 | 0.0000 | False | 0.9846 | 210 |
| base_24_126e38fc_t1774_f0__wrong | sentence_packed | `fuse_mean` | ungrounded | wrong | 0.9434 | 1.0000 | 0.0000 | False | 0.9923 | 210 |
| base_24_126e38fc_t1774_f0__wrong | sentence_packed | `fuse_max` | ungrounded | wrong | 0.9706 | 1.0000 | 0.0000 | False | 1.0000 | 210 |
| base_24_126e38fc_t1774_f0__wrong | sentence_packed | `fuse_min` | ungrounded | wrong | 0.9161 | 1.0000 | 0.0000 | False | 0.9846 | 210 |
| base_24_126e38fc_t1774_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9789 | 1.0000 | 0.0000 | False | 1.0000 | 210 |
| base_24_126e38fc_t1774_f0__ambiguous | sentence_packed | `code_only` | ambiguous | ambiguous | 0.9245 | 1.0000 | 0.0000 | False | 0.9833 | 210 |
| base_24_126e38fc_t1774_f0__ambiguous | sentence_packed | `fuse_mean` | ambiguous | ambiguous | 0.9517 | 1.0000 | 0.0000 | False | 0.9916 | 210 |
| base_24_126e38fc_t1774_f0__ambiguous | sentence_packed | `fuse_max` | ambiguous | ambiguous | 0.9789 | 1.0000 | 0.0000 | False | 1.0000 | 210 |
| base_24_126e38fc_t1774_f0__ambiguous | sentence_packed | `fuse_min` | ambiguous | ambiguous | 0.9245 | 1.0000 | 0.0000 | False | 0.9833 | 210 |
| base_25_8a0f6f68_t1261_f1__correct | sentence_packed | `gte_only` | grounded | correct | 0.9757 | 1.0000 | 0.0000 | False | 1.0000 | 120 |
| base_25_8a0f6f68_t1261_f1__correct | sentence_packed | `code_only` | grounded | correct | 0.9506 | 1.0000 | 0.0000 | False | 1.0000 | 120 |
| base_25_8a0f6f68_t1261_f1__correct | sentence_packed | `fuse_mean` | grounded | correct | 0.9631 | 1.0000 | 0.0000 | False | 1.0000 | 120 |
| base_25_8a0f6f68_t1261_f1__correct | sentence_packed | `fuse_max` | grounded | correct | 0.9757 | 1.0000 | 0.0000 | False | 1.0000 | 120 |
| base_25_8a0f6f68_t1261_f1__correct | sentence_packed | `fuse_min` | grounded | correct | 0.9506 | 1.0000 | 0.0000 | False | 1.0000 | 120 |
| base_25_8a0f6f68_t1261_f1__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9672 | 1.0000 | 0.0000 | False | 1.0000 | 120 |
| base_25_8a0f6f68_t1261_f1__wrong | sentence_packed | `code_only` | ungrounded | wrong | 0.9458 | 1.0000 | 0.0000 | False | 1.0000 | 120 |
| base_25_8a0f6f68_t1261_f1__wrong | sentence_packed | `fuse_mean` | ungrounded | wrong | 0.9565 | 1.0000 | 0.0000 | False | 1.0000 | 120 |
| base_25_8a0f6f68_t1261_f1__wrong | sentence_packed | `fuse_max` | ungrounded | wrong | 0.9672 | 1.0000 | 0.0000 | False | 1.0000 | 120 |
| base_25_8a0f6f68_t1261_f1__wrong | sentence_packed | `fuse_min` | ungrounded | wrong | 0.9458 | 1.0000 | 0.0000 | False | 1.0000 | 120 |
| base_25_8a0f6f68_t1261_f1__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9740 | 1.0000 | 0.0000 | False | 1.0000 | 120 |
| base_25_8a0f6f68_t1261_f1__ambiguous | sentence_packed | `code_only` | ambiguous | ambiguous | 0.9491 | 1.0000 | 0.0000 | False | 1.0000 | 120 |
| base_25_8a0f6f68_t1261_f1__ambiguous | sentence_packed | `fuse_mean` | ambiguous | ambiguous | 0.9615 | 1.0000 | 0.0000 | False | 1.0000 | 120 |
| base_25_8a0f6f68_t1261_f1__ambiguous | sentence_packed | `fuse_max` | ambiguous | ambiguous | 0.9740 | 1.0000 | 0.0000 | False | 1.0000 | 120 |
| base_25_8a0f6f68_t1261_f1__ambiguous | sentence_packed | `fuse_min` | ambiguous | ambiguous | 0.9491 | 1.0000 | 0.0000 | False | 1.0000 | 120 |
| base_26_f0afeaf2_t1530_f1__correct | sentence_packed | `gte_only` | grounded | correct | 0.9802 | 1.0000 | 0.0000 | False | 0.9989 | 195 |
| base_26_f0afeaf2_t1530_f1__correct | sentence_packed | `code_only` | grounded | correct | 0.9451 | 1.0000 | 0.0000 | False | 1.0000 | 195 |
| base_26_f0afeaf2_t1530_f1__correct | sentence_packed | `fuse_mean` | grounded | correct | 0.9626 | 1.0000 | 0.0000 | False | 0.9994 | 195 |
| base_26_f0afeaf2_t1530_f1__correct | sentence_packed | `fuse_max` | grounded | correct | 0.9802 | 1.0000 | 0.0000 | False | 1.0000 | 195 |
| base_26_f0afeaf2_t1530_f1__correct | sentence_packed | `fuse_min` | grounded | correct | 0.9451 | 1.0000 | 0.0000 | False | 0.9989 | 195 |
| base_26_f0afeaf2_t1530_f1__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9804 | 1.0000 | 0.0000 | False | 0.9993 | 195 |
| base_26_f0afeaf2_t1530_f1__wrong | sentence_packed | `code_only` | ungrounded | wrong | 0.9445 | 1.0000 | 0.0000 | False | 1.0000 | 195 |
| base_26_f0afeaf2_t1530_f1__wrong | sentence_packed | `fuse_mean` | ungrounded | wrong | 0.9625 | 1.0000 | 0.0000 | False | 0.9997 | 195 |
| base_26_f0afeaf2_t1530_f1__wrong | sentence_packed | `fuse_max` | ungrounded | wrong | 0.9804 | 1.0000 | 0.0000 | False | 1.0000 | 195 |
| base_26_f0afeaf2_t1530_f1__wrong | sentence_packed | `fuse_min` | ungrounded | wrong | 0.9445 | 1.0000 | 0.0000 | False | 0.9993 | 195 |
| base_26_f0afeaf2_t1530_f1__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9802 | 1.0000 | 0.0000 | False | 0.9989 | 195 |
| base_26_f0afeaf2_t1530_f1__ambiguous | sentence_packed | `code_only` | ambiguous | ambiguous | 0.9404 | 1.0000 | 0.0000 | False | 1.0000 | 195 |
| base_26_f0afeaf2_t1530_f1__ambiguous | sentence_packed | `fuse_mean` | ambiguous | ambiguous | 0.9603 | 1.0000 | 0.0000 | False | 0.9994 | 195 |
| base_26_f0afeaf2_t1530_f1__ambiguous | sentence_packed | `fuse_max` | ambiguous | ambiguous | 0.9802 | 1.0000 | 0.0000 | False | 1.0000 | 195 |
| base_26_f0afeaf2_t1530_f1__ambiguous | sentence_packed | `fuse_min` | ambiguous | ambiguous | 0.9404 | 1.0000 | 0.0000 | False | 0.9989 | 195 |
| base_27_8a0f6f68_t1163_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9793 | 1.0000 | 0.0000 | False | 1.0000 | 112 |
| base_27_8a0f6f68_t1163_f0__correct | sentence_packed | `code_only` | grounded | correct | 0.9432 | 1.0000 | 0.0000 | False | 0.9999 | 112 |
| base_27_8a0f6f68_t1163_f0__correct | sentence_packed | `fuse_mean` | grounded | correct | 0.9612 | 1.0000 | 0.0000 | False | 0.9999 | 112 |
| base_27_8a0f6f68_t1163_f0__correct | sentence_packed | `fuse_max` | grounded | correct | 0.9793 | 1.0000 | 0.0000 | False | 1.0000 | 112 |
| base_27_8a0f6f68_t1163_f0__correct | sentence_packed | `fuse_min` | grounded | correct | 0.9432 | 1.0000 | 0.0000 | False | 0.9999 | 112 |
| base_27_8a0f6f68_t1163_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9710 | 1.0000 | 0.0000 | False | 1.0000 | 112 |
| base_27_8a0f6f68_t1163_f0__wrong | sentence_packed | `code_only` | ungrounded | wrong | 0.9187 | 1.0000 | 0.0000 | False | 0.9999 | 112 |
| base_27_8a0f6f68_t1163_f0__wrong | sentence_packed | `fuse_mean` | ungrounded | wrong | 0.9448 | 1.0000 | 0.0000 | False | 0.9999 | 112 |
| base_27_8a0f6f68_t1163_f0__wrong | sentence_packed | `fuse_max` | ungrounded | wrong | 0.9710 | 1.0000 | 0.0000 | False | 1.0000 | 112 |
| base_27_8a0f6f68_t1163_f0__wrong | sentence_packed | `fuse_min` | ungrounded | wrong | 0.9187 | 1.0000 | 0.0000 | False | 0.9999 | 112 |
| base_27_8a0f6f68_t1163_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9754 | 1.0000 | 0.0000 | False | 0.9997 | 112 |
| base_27_8a0f6f68_t1163_f0__ambiguous | sentence_packed | `code_only` | ambiguous | ambiguous | 0.9302 | 1.0000 | 0.0000 | False | 0.9999 | 112 |
| base_27_8a0f6f68_t1163_f0__ambiguous | sentence_packed | `fuse_mean` | ambiguous | ambiguous | 0.9528 | 1.0000 | 0.0000 | False | 0.9998 | 112 |
| base_27_8a0f6f68_t1163_f0__ambiguous | sentence_packed | `fuse_max` | ambiguous | ambiguous | 0.9754 | 1.0000 | 0.0000 | False | 0.9999 | 112 |
| base_27_8a0f6f68_t1163_f0__ambiguous | sentence_packed | `fuse_min` | ambiguous | ambiguous | 0.9302 | 1.0000 | 0.0000 | False | 0.9997 | 112 |
| base_28_126e38fc_t1892_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9877 | 1.0000 | 0.0000 | False | 0.9978 | 223 |
| base_28_126e38fc_t1892_f0__correct | sentence_packed | `code_only` | grounded | correct | 0.9644 | 1.0000 | 0.0000 | False | 0.9942 | 223 |
| base_28_126e38fc_t1892_f0__correct | sentence_packed | `fuse_mean` | grounded | correct | 0.9760 | 1.0000 | 0.0000 | False | 0.9960 | 223 |
| base_28_126e38fc_t1892_f0__correct | sentence_packed | `fuse_max` | grounded | correct | 0.9877 | 1.0000 | 0.0000 | False | 0.9978 | 223 |
| base_28_126e38fc_t1892_f0__correct | sentence_packed | `fuse_min` | grounded | correct | 0.9644 | 1.0000 | 0.0000 | False | 0.9942 | 223 |
| base_28_126e38fc_t1892_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9816 | 1.0000 | 0.0000 | False | 0.9976 | 223 |
| base_28_126e38fc_t1892_f0__wrong | sentence_packed | `code_only` | ungrounded | wrong | 0.9559 | 1.0000 | 0.0000 | False | 0.9936 | 223 |
| base_28_126e38fc_t1892_f0__wrong | sentence_packed | `fuse_mean` | ungrounded | wrong | 0.9687 | 1.0000 | 0.0000 | False | 0.9956 | 223 |
| base_28_126e38fc_t1892_f0__wrong | sentence_packed | `fuse_max` | ungrounded | wrong | 0.9816 | 1.0000 | 0.0000 | False | 0.9976 | 223 |
| base_28_126e38fc_t1892_f0__wrong | sentence_packed | `fuse_min` | ungrounded | wrong | 0.9559 | 1.0000 | 0.0000 | False | 0.9936 | 223 |
| base_28_126e38fc_t1892_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9878 | 1.0000 | 0.0000 | False | 0.9978 | 223 |
| base_28_126e38fc_t1892_f0__ambiguous | sentence_packed | `code_only` | ambiguous | ambiguous | 0.9653 | 1.0000 | 0.0000 | False | 0.9943 | 223 |
| base_28_126e38fc_t1892_f0__ambiguous | sentence_packed | `fuse_mean` | ambiguous | ambiguous | 0.9765 | 1.0000 | 0.0000 | False | 0.9961 | 223 |
| base_28_126e38fc_t1892_f0__ambiguous | sentence_packed | `fuse_max` | ambiguous | ambiguous | 0.9878 | 1.0000 | 0.0000 | False | 0.9978 | 223 |
| base_28_126e38fc_t1892_f0__ambiguous | sentence_packed | `fuse_min` | ambiguous | ambiguous | 0.9653 | 1.0000 | 0.0000 | False | 0.9943 | 223 |
| base_29_8a0f6f68_t1278_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9763 | 1.0000 | 0.0000 | False | 0.9999 | 126 |
| base_29_8a0f6f68_t1278_f0__correct | sentence_packed | `code_only` | grounded | correct | 0.9374 | 1.0000 | 0.0000 | False | 1.0000 | 126 |
| base_29_8a0f6f68_t1278_f0__correct | sentence_packed | `fuse_mean` | grounded | correct | 0.9569 | 1.0000 | 0.0000 | False | 1.0000 | 126 |
| base_29_8a0f6f68_t1278_f0__correct | sentence_packed | `fuse_max` | grounded | correct | 0.9763 | 1.0000 | 0.0000 | False | 1.0000 | 126 |
| base_29_8a0f6f68_t1278_f0__correct | sentence_packed | `fuse_min` | grounded | correct | 0.9374 | 1.0000 | 0.0000 | False | 0.9999 | 126 |
| base_29_8a0f6f68_t1278_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9673 | 1.0000 | 0.0000 | False | 1.0000 | 126 |
| base_29_8a0f6f68_t1278_f0__wrong | sentence_packed | `code_only` | ungrounded | wrong | 0.9186 | 1.0000 | 0.0000 | False | 1.0000 | 126 |
| base_29_8a0f6f68_t1278_f0__wrong | sentence_packed | `fuse_mean` | ungrounded | wrong | 0.9430 | 1.0000 | 0.0000 | False | 1.0000 | 126 |
| base_29_8a0f6f68_t1278_f0__wrong | sentence_packed | `fuse_max` | ungrounded | wrong | 0.9673 | 1.0000 | 0.0000 | False | 1.0000 | 126 |
| base_29_8a0f6f68_t1278_f0__wrong | sentence_packed | `fuse_min` | ungrounded | wrong | 0.9186 | 1.0000 | 0.0000 | False | 1.0000 | 126 |
| base_29_8a0f6f68_t1278_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9711 | 1.0000 | 0.0000 | False | 1.0000 | 126 |
| base_29_8a0f6f68_t1278_f0__ambiguous | sentence_packed | `code_only` | ambiguous | ambiguous | 0.9238 | 1.0000 | 0.0000 | False | 0.9999 | 126 |
| base_29_8a0f6f68_t1278_f0__ambiguous | sentence_packed | `fuse_mean` | ambiguous | ambiguous | 0.9474 | 1.0000 | 0.0000 | False | 0.9999 | 126 |
| base_29_8a0f6f68_t1278_f0__ambiguous | sentence_packed | `fuse_max` | ambiguous | ambiguous | 0.9711 | 1.0000 | 0.0000 | False | 1.0000 | 126 |
| base_29_8a0f6f68_t1278_f0__ambiguous | sentence_packed | `fuse_min` | ambiguous | ambiguous | 0.9238 | 1.0000 | 0.0000 | False | 0.9999 | 126 |
| base_30_d7fd7127_t807_f1__correct | sentence_packed | `gte_only` | grounded | correct | 0.9816 | 1.0000 | 0.0000 | False | 0.9988 | 113 |
| base_30_d7fd7127_t807_f1__correct | sentence_packed | `code_only` | grounded | correct | 0.9392 | 1.0000 | 0.0000 | False | 0.9833 | 113 |
| base_30_d7fd7127_t807_f1__correct | sentence_packed | `fuse_mean` | grounded | correct | 0.9604 | 1.0000 | 0.0000 | False | 0.9910 | 113 |
| base_30_d7fd7127_t807_f1__correct | sentence_packed | `fuse_max` | grounded | correct | 0.9816 | 1.0000 | 0.0000 | False | 0.9988 | 113 |
| base_30_d7fd7127_t807_f1__correct | sentence_packed | `fuse_min` | grounded | correct | 0.9392 | 1.0000 | 0.0000 | False | 0.9833 | 113 |
| base_30_d7fd7127_t807_f1__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9816 | 1.0000 | 0.0000 | False | 0.9988 | 113 |
| base_30_d7fd7127_t807_f1__wrong | sentence_packed | `code_only` | ungrounded | wrong | 0.9377 | 1.0000 | 0.0000 | False | 0.9837 | 113 |
| base_30_d7fd7127_t807_f1__wrong | sentence_packed | `fuse_mean` | ungrounded | wrong | 0.9596 | 1.0000 | 0.0000 | False | 0.9912 | 113 |
| base_30_d7fd7127_t807_f1__wrong | sentence_packed | `fuse_max` | ungrounded | wrong | 0.9816 | 1.0000 | 0.0000 | False | 0.9988 | 113 |
| base_30_d7fd7127_t807_f1__wrong | sentence_packed | `fuse_min` | ungrounded | wrong | 0.9377 | 1.0000 | 0.0000 | False | 0.9837 | 113 |
| base_30_d7fd7127_t807_f1__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9796 | 1.0000 | 0.0000 | False | 1.0000 | 113 |
| base_30_d7fd7127_t807_f1__ambiguous | sentence_packed | `code_only` | ambiguous | ambiguous | 0.9379 | 1.0000 | 0.0000 | False | 0.9863 | 113 |
| base_30_d7fd7127_t807_f1__ambiguous | sentence_packed | `fuse_mean` | ambiguous | ambiguous | 0.9587 | 1.0000 | 0.0000 | False | 0.9931 | 113 |
| base_30_d7fd7127_t807_f1__ambiguous | sentence_packed | `fuse_max` | ambiguous | ambiguous | 0.9796 | 1.0000 | 0.0000 | False | 1.0000 | 113 |
| base_30_d7fd7127_t807_f1__ambiguous | sentence_packed | `fuse_min` | ambiguous | ambiguous | 0.9379 | 1.0000 | 0.0000 | False | 0.9863 | 113 |
| base_31_8a0f6f68_t1301_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9738 | 1.0000 | 0.0000 | False | 1.0000 | 125 |
| base_31_8a0f6f68_t1301_f0__correct | sentence_packed | `code_only` | grounded | correct | 0.9285 | 1.0000 | 0.0000 | False | 1.0000 | 125 |
| base_31_8a0f6f68_t1301_f0__correct | sentence_packed | `fuse_mean` | grounded | correct | 0.9512 | 1.0000 | 0.0000 | False | 1.0000 | 125 |
| base_31_8a0f6f68_t1301_f0__correct | sentence_packed | `fuse_max` | grounded | correct | 0.9738 | 1.0000 | 0.0000 | False | 1.0000 | 125 |
| base_31_8a0f6f68_t1301_f0__correct | sentence_packed | `fuse_min` | grounded | correct | 0.9285 | 1.0000 | 0.0000 | False | 1.0000 | 125 |
| base_31_8a0f6f68_t1301_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9666 | 1.0000 | 0.0000 | False | 1.0000 | 125 |
| base_31_8a0f6f68_t1301_f0__wrong | sentence_packed | `code_only` | ungrounded | wrong | 0.9240 | 1.0000 | 0.0000 | False | 1.0000 | 125 |
| base_31_8a0f6f68_t1301_f0__wrong | sentence_packed | `fuse_mean` | ungrounded | wrong | 0.9453 | 1.0000 | 0.0000 | False | 1.0000 | 125 |
| base_31_8a0f6f68_t1301_f0__wrong | sentence_packed | `fuse_max` | ungrounded | wrong | 0.9666 | 1.0000 | 0.0000 | False | 1.0000 | 125 |
| base_31_8a0f6f68_t1301_f0__wrong | sentence_packed | `fuse_min` | ungrounded | wrong | 0.9240 | 1.0000 | 0.0000 | False | 1.0000 | 125 |
| base_31_8a0f6f68_t1301_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9720 | 1.0000 | 0.0000 | False | 1.0000 | 125 |
| base_31_8a0f6f68_t1301_f0__ambiguous | sentence_packed | `code_only` | ambiguous | ambiguous | 0.9294 | 1.0000 | 0.0000 | False | 1.0000 | 125 |
| base_31_8a0f6f68_t1301_f0__ambiguous | sentence_packed | `fuse_mean` | ambiguous | ambiguous | 0.9507 | 1.0000 | 0.0000 | False | 1.0000 | 125 |
| base_31_8a0f6f68_t1301_f0__ambiguous | sentence_packed | `fuse_max` | ambiguous | ambiguous | 0.9720 | 1.0000 | 0.0000 | False | 1.0000 | 125 |
| base_31_8a0f6f68_t1301_f0__ambiguous | sentence_packed | `fuse_min` | ambiguous | ambiguous | 0.9294 | 1.0000 | 0.0000 | False | 1.0000 | 125 |
| base_32_126e38fc_t1007_f1__correct | sentence_packed | `gte_only` | grounded | correct | 0.9856 | 1.0000 | 0.0000 | False | 1.0000 | 102 |
| base_32_126e38fc_t1007_f1__correct | sentence_packed | `code_only` | grounded | correct | 0.9381 | 1.0000 | 0.0000 | False | 1.0000 | 102 |
| base_32_126e38fc_t1007_f1__correct | sentence_packed | `fuse_mean` | grounded | correct | 0.9618 | 1.0000 | 0.0000 | False | 1.0000 | 102 |
| base_32_126e38fc_t1007_f1__correct | sentence_packed | `fuse_max` | grounded | correct | 0.9856 | 1.0000 | 0.0000 | False | 1.0000 | 102 |
| base_32_126e38fc_t1007_f1__correct | sentence_packed | `fuse_min` | grounded | correct | 0.9381 | 1.0000 | 0.0000 | False | 1.0000 | 102 |
| base_32_126e38fc_t1007_f1__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9739 | 1.0000 | 0.0000 | False | 1.0000 | 102 |
| base_32_126e38fc_t1007_f1__wrong | sentence_packed | `code_only` | ungrounded | wrong | 0.9293 | 1.0000 | 0.0000 | False | 1.0000 | 102 |
| base_32_126e38fc_t1007_f1__wrong | sentence_packed | `fuse_mean` | ungrounded | wrong | 0.9516 | 1.0000 | 0.0000 | False | 1.0000 | 102 |
| base_32_126e38fc_t1007_f1__wrong | sentence_packed | `fuse_max` | ungrounded | wrong | 0.9739 | 1.0000 | 0.0000 | False | 1.0000 | 102 |
| base_32_126e38fc_t1007_f1__wrong | sentence_packed | `fuse_min` | ungrounded | wrong | 0.9293 | 1.0000 | 0.0000 | False | 1.0000 | 102 |
| base_32_126e38fc_t1007_f1__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9847 | 1.0000 | 0.0000 | False | 0.9999 | 102 |
| base_32_126e38fc_t1007_f1__ambiguous | sentence_packed | `code_only` | ambiguous | ambiguous | 0.9395 | 1.0000 | 0.0000 | False | 1.0000 | 102 |
| base_32_126e38fc_t1007_f1__ambiguous | sentence_packed | `fuse_mean` | ambiguous | ambiguous | 0.9621 | 1.0000 | 0.0000 | False | 1.0000 | 102 |
| base_32_126e38fc_t1007_f1__ambiguous | sentence_packed | `fuse_max` | ambiguous | ambiguous | 0.9847 | 1.0000 | 0.0000 | False | 1.0000 | 102 |
| base_32_126e38fc_t1007_f1__ambiguous | sentence_packed | `fuse_min` | ambiguous | ambiguous | 0.9395 | 1.0000 | 0.0000 | False | 0.9999 | 102 |
| base_33_d7fd7127_t1397_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9850 | 1.0000 | 0.0000 | False | 0.9972 | 198 |
| base_33_d7fd7127_t1397_f0__correct | sentence_packed | `code_only` | grounded | correct | 0.9448 | 1.0000 | 0.0000 | False | 0.9816 | 198 |
| base_33_d7fd7127_t1397_f0__correct | sentence_packed | `fuse_mean` | grounded | correct | 0.9649 | 1.0000 | 0.0000 | False | 0.9894 | 198 |
| base_33_d7fd7127_t1397_f0__correct | sentence_packed | `fuse_max` | grounded | correct | 0.9850 | 1.0000 | 0.0000 | False | 0.9972 | 198 |
| base_33_d7fd7127_t1397_f0__correct | sentence_packed | `fuse_min` | grounded | correct | 0.9448 | 1.0000 | 0.0000 | False | 0.9816 | 198 |
| base_33_d7fd7127_t1397_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9812 | 1.0000 | 0.0000 | False | 0.9973 | 198 |
| base_33_d7fd7127_t1397_f0__wrong | sentence_packed | `code_only` | ungrounded | wrong | 0.9349 | 1.0000 | 0.0000 | False | 0.9688 | 198 |
| base_33_d7fd7127_t1397_f0__wrong | sentence_packed | `fuse_mean` | ungrounded | wrong | 0.9581 | 1.0000 | 0.0000 | False | 0.9831 | 198 |
| base_33_d7fd7127_t1397_f0__wrong | sentence_packed | `fuse_max` | ungrounded | wrong | 0.9812 | 1.0000 | 0.0000 | False | 0.9973 | 198 |
| base_33_d7fd7127_t1397_f0__wrong | sentence_packed | `fuse_min` | ungrounded | wrong | 0.9349 | 1.0000 | 0.0000 | False | 0.9688 | 198 |
| base_33_d7fd7127_t1397_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9822 | 1.0000 | 0.0000 | False | 0.9974 | 198 |
| base_33_d7fd7127_t1397_f0__ambiguous | sentence_packed | `code_only` | ambiguous | ambiguous | 0.9429 | 1.0000 | 0.0000 | False | 0.9842 | 198 |
| base_33_d7fd7127_t1397_f0__ambiguous | sentence_packed | `fuse_mean` | ambiguous | ambiguous | 0.9626 | 1.0000 | 0.0000 | False | 0.9908 | 198 |
| base_33_d7fd7127_t1397_f0__ambiguous | sentence_packed | `fuse_max` | ambiguous | ambiguous | 0.9822 | 1.0000 | 0.0000 | False | 0.9974 | 198 |
| base_33_d7fd7127_t1397_f0__ambiguous | sentence_packed | `fuse_min` | ambiguous | ambiguous | 0.9429 | 1.0000 | 0.0000 | False | 0.9842 | 198 |
| base_34_5e875b7e_t1637_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9779 | 1.0000 | 0.0000 | False | 1.0000 | 61 |
| base_34_5e875b7e_t1637_f0__correct | sentence_packed | `code_only` | grounded | correct | 0.9388 | 1.0000 | 0.0000 | False | 1.0000 | 61 |
| base_34_5e875b7e_t1637_f0__correct | sentence_packed | `fuse_mean` | grounded | correct | 0.9584 | 1.0000 | 0.0000 | False | 1.0000 | 61 |
| base_34_5e875b7e_t1637_f0__correct | sentence_packed | `fuse_max` | grounded | correct | 0.9779 | 1.0000 | 0.0000 | False | 1.0000 | 61 |
| base_34_5e875b7e_t1637_f0__correct | sentence_packed | `fuse_min` | grounded | correct | 0.9388 | 1.0000 | 0.0000 | False | 1.0000 | 61 |
| base_34_5e875b7e_t1637_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9743 | 1.0000 | 0.0000 | False | 1.0000 | 61 |
| base_34_5e875b7e_t1637_f0__wrong | sentence_packed | `code_only` | ungrounded | wrong | 0.9325 | 1.0000 | 0.0000 | False | 1.0000 | 61 |
| base_34_5e875b7e_t1637_f0__wrong | sentence_packed | `fuse_mean` | ungrounded | wrong | 0.9534 | 1.0000 | 0.0000 | False | 1.0000 | 61 |
| base_34_5e875b7e_t1637_f0__wrong | sentence_packed | `fuse_max` | ungrounded | wrong | 0.9743 | 1.0000 | 0.0000 | False | 1.0000 | 61 |
| base_34_5e875b7e_t1637_f0__wrong | sentence_packed | `fuse_min` | ungrounded | wrong | 0.9325 | 1.0000 | 0.0000 | False | 1.0000 | 61 |
| base_34_5e875b7e_t1637_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9772 | 1.0000 | 0.0000 | False | 1.0000 | 61 |
| base_34_5e875b7e_t1637_f0__ambiguous | sentence_packed | `code_only` | ambiguous | ambiguous | 0.9360 | 1.0000 | 0.0000 | False | 1.0000 | 61 |
| base_34_5e875b7e_t1637_f0__ambiguous | sentence_packed | `fuse_mean` | ambiguous | ambiguous | 0.9566 | 1.0000 | 0.0000 | False | 1.0000 | 61 |
| base_34_5e875b7e_t1637_f0__ambiguous | sentence_packed | `fuse_max` | ambiguous | ambiguous | 0.9772 | 1.0000 | 0.0000 | False | 1.0000 | 61 |
| base_34_5e875b7e_t1637_f0__ambiguous | sentence_packed | `fuse_min` | ambiguous | ambiguous | 0.9360 | 1.0000 | 0.0000 | False | 1.0000 | 61 |
| base_35_a4d14a56_t1005_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9767 | 1.0000 | 0.0000 | False | 0.9987 | 184 |
| base_35_a4d14a56_t1005_f0__correct | sentence_packed | `code_only` | grounded | correct | 0.9389 | 1.0000 | 0.0000 | False | 0.9875 | 184 |
| base_35_a4d14a56_t1005_f0__correct | sentence_packed | `fuse_mean` | grounded | correct | 0.9578 | 1.0000 | 0.0000 | False | 0.9931 | 184 |
| base_35_a4d14a56_t1005_f0__correct | sentence_packed | `fuse_max` | grounded | correct | 0.9767 | 1.0000 | 0.0000 | False | 0.9987 | 184 |
| base_35_a4d14a56_t1005_f0__correct | sentence_packed | `fuse_min` | grounded | correct | 0.9389 | 1.0000 | 0.0000 | False | 0.9875 | 184 |
| base_35_a4d14a56_t1005_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9738 | 1.0000 | 0.0000 | False | 0.9976 | 184 |
| base_35_a4d14a56_t1005_f0__wrong | sentence_packed | `code_only` | ungrounded | wrong | 0.9361 | 1.0000 | 0.0000 | False | 0.9873 | 184 |
| base_35_a4d14a56_t1005_f0__wrong | sentence_packed | `fuse_mean` | ungrounded | wrong | 0.9549 | 1.0000 | 0.0000 | False | 0.9924 | 184 |
| base_35_a4d14a56_t1005_f0__wrong | sentence_packed | `fuse_max` | ungrounded | wrong | 0.9738 | 1.0000 | 0.0000 | False | 0.9976 | 184 |
| base_35_a4d14a56_t1005_f0__wrong | sentence_packed | `fuse_min` | ungrounded | wrong | 0.9361 | 1.0000 | 0.0000 | False | 0.9873 | 184 |
| base_35_a4d14a56_t1005_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9763 | 1.0000 | 0.0000 | False | 0.9987 | 184 |
| base_35_a4d14a56_t1005_f0__ambiguous | sentence_packed | `code_only` | ambiguous | ambiguous | 0.9382 | 1.0000 | 0.0000 | False | 0.9868 | 184 |
| base_35_a4d14a56_t1005_f0__ambiguous | sentence_packed | `fuse_mean` | ambiguous | ambiguous | 0.9573 | 1.0000 | 0.0000 | False | 0.9927 | 184 |
| base_35_a4d14a56_t1005_f0__ambiguous | sentence_packed | `fuse_max` | ambiguous | ambiguous | 0.9763 | 1.0000 | 0.0000 | False | 0.9987 | 184 |
| base_35_a4d14a56_t1005_f0__ambiguous | sentence_packed | `fuse_min` | ambiguous | ambiguous | 0.9382 | 1.0000 | 0.0000 | False | 0.9868 | 184 |
| base_36_a4d14a56_t693_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9826 | 1.0000 | 0.0000 | False | 1.0000 | 93 |
| base_36_a4d14a56_t693_f0__correct | sentence_packed | `code_only` | grounded | correct | 0.9528 | 1.0000 | 0.0000 | False | 1.0000 | 93 |
| base_36_a4d14a56_t693_f0__correct | sentence_packed | `fuse_mean` | grounded | correct | 0.9677 | 1.0000 | 0.0000 | False | 1.0000 | 93 |
| base_36_a4d14a56_t693_f0__correct | sentence_packed | `fuse_max` | grounded | correct | 0.9826 | 1.0000 | 0.0000 | False | 1.0000 | 93 |
| base_36_a4d14a56_t693_f0__correct | sentence_packed | `fuse_min` | grounded | correct | 0.9528 | 1.0000 | 0.0000 | False | 1.0000 | 93 |
| base_36_a4d14a56_t693_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9697 | 1.0000 | 0.0000 | False | 1.0000 | 93 |
| base_36_a4d14a56_t693_f0__wrong | sentence_packed | `code_only` | ungrounded | wrong | 0.9346 | 1.0000 | 0.0000 | False | 1.0000 | 93 |
| base_36_a4d14a56_t693_f0__wrong | sentence_packed | `fuse_mean` | ungrounded | wrong | 0.9521 | 1.0000 | 0.0000 | False | 1.0000 | 93 |
| base_36_a4d14a56_t693_f0__wrong | sentence_packed | `fuse_max` | ungrounded | wrong | 0.9697 | 1.0000 | 0.0000 | False | 1.0000 | 93 |
| base_36_a4d14a56_t693_f0__wrong | sentence_packed | `fuse_min` | ungrounded | wrong | 0.9346 | 1.0000 | 0.0000 | False | 1.0000 | 93 |
| base_36_a4d14a56_t693_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9785 | 1.0000 | 0.0000 | False | 1.0000 | 93 |
| base_36_a4d14a56_t693_f0__ambiguous | sentence_packed | `code_only` | ambiguous | ambiguous | 0.9424 | 1.0000 | 0.0000 | False | 1.0000 | 93 |
| base_36_a4d14a56_t693_f0__ambiguous | sentence_packed | `fuse_mean` | ambiguous | ambiguous | 0.9604 | 1.0000 | 0.0000 | False | 1.0000 | 93 |
| base_36_a4d14a56_t693_f0__ambiguous | sentence_packed | `fuse_max` | ambiguous | ambiguous | 0.9785 | 1.0000 | 0.0000 | False | 1.0000 | 93 |
| base_36_a4d14a56_t693_f0__ambiguous | sentence_packed | `fuse_min` | ambiguous | ambiguous | 0.9424 | 1.0000 | 0.0000 | False | 1.0000 | 93 |
| base_37_126e38fc_t1936_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9819 | 1.0000 | 0.0000 | False | 0.9999 | 216 |
| base_37_126e38fc_t1936_f0__correct | sentence_packed | `code_only` | grounded | correct | 0.9404 | 1.0000 | 0.0000 | False | 1.0000 | 216 |
| base_37_126e38fc_t1936_f0__correct | sentence_packed | `fuse_mean` | grounded | correct | 0.9611 | 1.0000 | 0.0000 | False | 1.0000 | 216 |
| base_37_126e38fc_t1936_f0__correct | sentence_packed | `fuse_max` | grounded | correct | 0.9819 | 1.0000 | 0.0000 | False | 1.0000 | 216 |
| base_37_126e38fc_t1936_f0__correct | sentence_packed | `fuse_min` | grounded | correct | 0.9404 | 1.0000 | 0.0000 | False | 0.9999 | 216 |
| base_37_126e38fc_t1936_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9698 | 1.0000 | 0.0000 | False | 1.0000 | 216 |
| base_37_126e38fc_t1936_f0__wrong | sentence_packed | `code_only` | ungrounded | wrong | 0.9037 | 1.0000 | 0.0000 | False | 1.0000 | 216 |
| base_37_126e38fc_t1936_f0__wrong | sentence_packed | `fuse_mean` | ungrounded | wrong | 0.9368 | 1.0000 | 0.0000 | False | 1.0000 | 216 |
| base_37_126e38fc_t1936_f0__wrong | sentence_packed | `fuse_max` | ungrounded | wrong | 0.9698 | 1.0000 | 0.0000 | False | 1.0000 | 216 |
| base_37_126e38fc_t1936_f0__wrong | sentence_packed | `fuse_min` | ungrounded | wrong | 0.9037 | 1.0000 | 0.0000 | False | 1.0000 | 216 |
| base_37_126e38fc_t1936_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9798 | 1.0000 | 0.0000 | False | 0.9999 | 216 |
| base_37_126e38fc_t1936_f0__ambiguous | sentence_packed | `code_only` | ambiguous | ambiguous | 0.9321 | 1.0000 | 0.0000 | False | 0.9999 | 216 |
| base_37_126e38fc_t1936_f0__ambiguous | sentence_packed | `fuse_mean` | ambiguous | ambiguous | 0.9559 | 1.0000 | 0.0000 | False | 0.9999 | 216 |
| base_37_126e38fc_t1936_f0__ambiguous | sentence_packed | `fuse_max` | ambiguous | ambiguous | 0.9798 | 1.0000 | 0.0000 | False | 0.9999 | 216 |
| base_37_126e38fc_t1936_f0__ambiguous | sentence_packed | `fuse_min` | ambiguous | ambiguous | 0.9321 | 1.0000 | 0.0000 | False | 0.9999 | 216 |
| base_38_a4d14a56_t869_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9727 | 1.0000 | 0.0000 | False | 1.0000 | 184 |
| base_38_a4d14a56_t869_f0__correct | sentence_packed | `code_only` | grounded | correct | 0.9284 | 1.0000 | 0.0000 | False | 1.0000 | 184 |
| base_38_a4d14a56_t869_f0__correct | sentence_packed | `fuse_mean` | grounded | correct | 0.9505 | 1.0000 | 0.0000 | False | 1.0000 | 184 |
| base_38_a4d14a56_t869_f0__correct | sentence_packed | `fuse_max` | grounded | correct | 0.9727 | 1.0000 | 0.0000 | False | 1.0000 | 184 |
| base_38_a4d14a56_t869_f0__correct | sentence_packed | `fuse_min` | grounded | correct | 0.9284 | 1.0000 | 0.0000 | False | 1.0000 | 184 |
| base_38_a4d14a56_t869_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9642 | 1.0000 | 0.0000 | False | 1.0000 | 184 |
| base_38_a4d14a56_t869_f0__wrong | sentence_packed | `code_only` | ungrounded | wrong | 0.9273 | 1.0000 | 0.0000 | False | 1.0000 | 184 |
| base_38_a4d14a56_t869_f0__wrong | sentence_packed | `fuse_mean` | ungrounded | wrong | 0.9457 | 1.0000 | 0.0000 | False | 1.0000 | 184 |
| base_38_a4d14a56_t869_f0__wrong | sentence_packed | `fuse_max` | ungrounded | wrong | 0.9642 | 1.0000 | 0.0000 | False | 1.0000 | 184 |
| base_38_a4d14a56_t869_f0__wrong | sentence_packed | `fuse_min` | ungrounded | wrong | 0.9273 | 1.0000 | 0.0000 | False | 1.0000 | 184 |
| base_38_a4d14a56_t869_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9703 | 1.0000 | 0.0000 | False | 1.0000 | 184 |
| base_38_a4d14a56_t869_f0__ambiguous | sentence_packed | `code_only` | ambiguous | ambiguous | 0.9319 | 1.0000 | 0.0000 | False | 1.0000 | 184 |
| base_38_a4d14a56_t869_f0__ambiguous | sentence_packed | `fuse_mean` | ambiguous | ambiguous | 0.9511 | 1.0000 | 0.0000 | False | 1.0000 | 184 |
| base_38_a4d14a56_t869_f0__ambiguous | sentence_packed | `fuse_max` | ambiguous | ambiguous | 0.9703 | 1.0000 | 0.0000 | False | 1.0000 | 184 |
| base_38_a4d14a56_t869_f0__ambiguous | sentence_packed | `fuse_min` | ambiguous | ambiguous | 0.9319 | 1.0000 | 0.0000 | False | 1.0000 | 184 |
| base_39_5e875b7e_t2278_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9681 | 1.0000 | 0.0000 | False | 0.9951 | 68 |
| base_39_5e875b7e_t2278_f0__correct | sentence_packed | `code_only` | grounded | correct | 0.9255 | 1.0000 | 0.0000 | False | 0.9894 | 68 |
| base_39_5e875b7e_t2278_f0__correct | sentence_packed | `fuse_mean` | grounded | correct | 0.9468 | 1.0000 | 0.0000 | False | 0.9922 | 68 |
| base_39_5e875b7e_t2278_f0__correct | sentence_packed | `fuse_max` | grounded | correct | 0.9681 | 1.0000 | 0.0000 | False | 0.9951 | 68 |
| base_39_5e875b7e_t2278_f0__correct | sentence_packed | `fuse_min` | grounded | correct | 0.9255 | 1.0000 | 0.0000 | False | 0.9894 | 68 |
| base_39_5e875b7e_t2278_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9677 | 1.0000 | 0.0000 | False | 0.9949 | 68 |
| base_39_5e875b7e_t2278_f0__wrong | sentence_packed | `code_only` | ungrounded | wrong | 0.9250 | 1.0000 | 0.0000 | False | 0.9798 | 68 |
| base_39_5e875b7e_t2278_f0__wrong | sentence_packed | `fuse_mean` | ungrounded | wrong | 0.9464 | 1.0000 | 0.0000 | False | 0.9874 | 68 |
| base_39_5e875b7e_t2278_f0__wrong | sentence_packed | `fuse_max` | ungrounded | wrong | 0.9677 | 1.0000 | 0.0000 | False | 0.9949 | 68 |
| base_39_5e875b7e_t2278_f0__wrong | sentence_packed | `fuse_min` | ungrounded | wrong | 0.9250 | 1.0000 | 0.0000 | False | 0.9798 | 68 |
| base_39_5e875b7e_t2278_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9682 | 1.0000 | 0.0000 | False | 0.9950 | 68 |
| base_39_5e875b7e_t2278_f0__ambiguous | sentence_packed | `code_only` | ambiguous | ambiguous | 0.9304 | 1.0000 | 0.0000 | False | 0.9893 | 68 |
| base_39_5e875b7e_t2278_f0__ambiguous | sentence_packed | `fuse_mean` | ambiguous | ambiguous | 0.9493 | 1.0000 | 0.0000 | False | 0.9922 | 68 |
| base_39_5e875b7e_t2278_f0__ambiguous | sentence_packed | `fuse_max` | ambiguous | ambiguous | 0.9682 | 1.0000 | 0.0000 | False | 0.9950 | 68 |
| base_39_5e875b7e_t2278_f0__ambiguous | sentence_packed | `fuse_min` | ambiguous | ambiguous | 0.9304 | 1.0000 | 0.0000 | False | 0.9893 | 68 |
| base_40_ed69b0cd_t395_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9857 | 1.0000 | 0.0000 | False | 1.0000 | 81 |
| base_40_ed69b0cd_t395_f0__correct | sentence_packed | `code_only` | grounded | correct | 0.9581 | 1.0000 | 0.0000 | False | 0.9999 | 81 |
| base_40_ed69b0cd_t395_f0__correct | sentence_packed | `fuse_mean` | grounded | correct | 0.9719 | 1.0000 | 0.0000 | False | 0.9999 | 81 |
| base_40_ed69b0cd_t395_f0__correct | sentence_packed | `fuse_max` | grounded | correct | 0.9857 | 1.0000 | 0.0000 | False | 1.0000 | 81 |
| base_40_ed69b0cd_t395_f0__correct | sentence_packed | `fuse_min` | grounded | correct | 0.9581 | 1.0000 | 0.0000 | False | 0.9999 | 81 |
| base_40_ed69b0cd_t395_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9674 | 1.0000 | 0.0000 | False | 1.0000 | 81 |
| base_40_ed69b0cd_t395_f0__wrong | sentence_packed | `code_only` | ungrounded | wrong | 0.9400 | 1.0000 | 0.0000 | False | 1.0000 | 81 |
| base_40_ed69b0cd_t395_f0__wrong | sentence_packed | `fuse_mean` | ungrounded | wrong | 0.9537 | 1.0000 | 0.0000 | False | 1.0000 | 81 |
| base_40_ed69b0cd_t395_f0__wrong | sentence_packed | `fuse_max` | ungrounded | wrong | 0.9674 | 1.0000 | 0.0000 | False | 1.0000 | 81 |
| base_40_ed69b0cd_t395_f0__wrong | sentence_packed | `fuse_min` | ungrounded | wrong | 0.9400 | 1.0000 | 0.0000 | False | 1.0000 | 81 |
| base_40_ed69b0cd_t395_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9852 | 1.0000 | 0.0000 | False | 1.0000 | 81 |
| base_40_ed69b0cd_t395_f0__ambiguous | sentence_packed | `code_only` | ambiguous | ambiguous | 0.9569 | 1.0000 | 0.0000 | False | 0.9999 | 81 |
| base_40_ed69b0cd_t395_f0__ambiguous | sentence_packed | `fuse_mean` | ambiguous | ambiguous | 0.9710 | 1.0000 | 0.0000 | False | 0.9999 | 81 |
| base_40_ed69b0cd_t395_f0__ambiguous | sentence_packed | `fuse_max` | ambiguous | ambiguous | 0.9852 | 1.0000 | 0.0000 | False | 1.0000 | 81 |
| base_40_ed69b0cd_t395_f0__ambiguous | sentence_packed | `fuse_min` | ambiguous | ambiguous | 0.9569 | 1.0000 | 0.0000 | False | 0.9999 | 81 |
| base_41_a4d14a56_t994_f1__correct | sentence_packed | `gte_only` | grounded | correct | 0.9859 | 1.0000 | 0.0000 | False | 0.9908 | 184 |
| base_41_a4d14a56_t994_f1__correct | sentence_packed | `code_only` | grounded | correct | 0.9507 | 1.0000 | 0.0000 | False | 0.9779 | 184 |
| base_41_a4d14a56_t994_f1__correct | sentence_packed | `fuse_mean` | grounded | correct | 0.9683 | 1.0000 | 0.0000 | False | 0.9843 | 184 |
| base_41_a4d14a56_t994_f1__correct | sentence_packed | `fuse_max` | grounded | correct | 0.9859 | 1.0000 | 0.0000 | False | 0.9908 | 184 |
| base_41_a4d14a56_t994_f1__correct | sentence_packed | `fuse_min` | grounded | correct | 0.9507 | 1.0000 | 0.0000 | False | 0.9779 | 184 |
| base_41_a4d14a56_t994_f1__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9829 | 1.0000 | 0.0000 | False | 0.9969 | 184 |
| base_41_a4d14a56_t994_f1__wrong | sentence_packed | `code_only` | ungrounded | wrong | 0.9465 | 1.0000 | 0.0000 | False | 0.9763 | 184 |
| base_41_a4d14a56_t994_f1__wrong | sentence_packed | `fuse_mean` | ungrounded | wrong | 0.9647 | 1.0000 | 0.0000 | False | 0.9866 | 184 |
| base_41_a4d14a56_t994_f1__wrong | sentence_packed | `fuse_max` | ungrounded | wrong | 0.9829 | 1.0000 | 0.0000 | False | 0.9969 | 184 |
| base_41_a4d14a56_t994_f1__wrong | sentence_packed | `fuse_min` | ungrounded | wrong | 0.9465 | 1.0000 | 0.0000 | False | 0.9763 | 184 |
| base_41_a4d14a56_t994_f1__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9855 | 1.0000 | 0.0000 | False | 0.9907 | 184 |
| base_41_a4d14a56_t994_f1__ambiguous | sentence_packed | `code_only` | ambiguous | ambiguous | 0.9507 | 1.0000 | 0.0000 | False | 0.9783 | 184 |
| base_41_a4d14a56_t994_f1__ambiguous | sentence_packed | `fuse_mean` | ambiguous | ambiguous | 0.9681 | 1.0000 | 0.0000 | False | 0.9845 | 184 |
| base_41_a4d14a56_t994_f1__ambiguous | sentence_packed | `fuse_max` | ambiguous | ambiguous | 0.9855 | 1.0000 | 0.0000 | False | 0.9907 | 184 |
| base_41_a4d14a56_t994_f1__ambiguous | sentence_packed | `fuse_min` | ambiguous | ambiguous | 0.9507 | 1.0000 | 0.0000 | False | 0.9783 | 184 |
| base_42_a4d14a56_t531_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9811 | 1.0000 | 0.0000 | False | 0.9965 | 67 |
| base_42_a4d14a56_t531_f0__correct | sentence_packed | `code_only` | grounded | correct | 0.9385 | 1.0000 | 0.0000 | False | 0.9865 | 67 |
| base_42_a4d14a56_t531_f0__correct | sentence_packed | `fuse_mean` | grounded | correct | 0.9598 | 1.0000 | 0.0000 | False | 0.9915 | 67 |
| base_42_a4d14a56_t531_f0__correct | sentence_packed | `fuse_max` | grounded | correct | 0.9811 | 1.0000 | 0.0000 | False | 0.9965 | 67 |
| base_42_a4d14a56_t531_f0__correct | sentence_packed | `fuse_min` | grounded | correct | 0.9385 | 1.0000 | 0.0000 | False | 0.9865 | 67 |
| base_42_a4d14a56_t531_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9728 | 1.0000 | 0.0000 | False | 0.9967 | 67 |
| base_42_a4d14a56_t531_f0__wrong | sentence_packed | `code_only` | ungrounded | wrong | 0.9318 | 1.0000 | 0.0000 | False | 0.9827 | 67 |
| base_42_a4d14a56_t531_f0__wrong | sentence_packed | `fuse_mean` | ungrounded | wrong | 0.9523 | 1.0000 | 0.0000 | False | 0.9897 | 67 |
| base_42_a4d14a56_t531_f0__wrong | sentence_packed | `fuse_max` | ungrounded | wrong | 0.9728 | 1.0000 | 0.0000 | False | 0.9967 | 67 |
| base_42_a4d14a56_t531_f0__wrong | sentence_packed | `fuse_min` | ungrounded | wrong | 0.9318 | 1.0000 | 0.0000 | False | 0.9827 | 67 |
| base_42_a4d14a56_t531_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9804 | 1.0000 | 0.0000 | False | 0.9967 | 67 |
| base_42_a4d14a56_t531_f0__ambiguous | sentence_packed | `code_only` | ambiguous | ambiguous | 0.9396 | 1.0000 | 0.0000 | False | 0.9871 | 67 |
| base_42_a4d14a56_t531_f0__ambiguous | sentence_packed | `fuse_mean` | ambiguous | ambiguous | 0.9600 | 1.0000 | 0.0000 | False | 0.9919 | 67 |
| base_42_a4d14a56_t531_f0__ambiguous | sentence_packed | `fuse_max` | ambiguous | ambiguous | 0.9804 | 1.0000 | 0.0000 | False | 0.9967 | 67 |
| base_42_a4d14a56_t531_f0__ambiguous | sentence_packed | `fuse_min` | ambiguous | ambiguous | 0.9396 | 1.0000 | 0.0000 | False | 0.9871 | 67 |
| base_43_ed69b0cd_t364_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9832 | 1.0000 | 0.0000 | False | 0.9970 | 81 |
| base_43_ed69b0cd_t364_f0__correct | sentence_packed | `code_only` | grounded | correct | 0.9488 | 1.0000 | 0.0000 | False | 0.9896 | 81 |
| base_43_ed69b0cd_t364_f0__correct | sentence_packed | `fuse_mean` | grounded | correct | 0.9660 | 1.0000 | 0.0000 | False | 0.9933 | 81 |
| base_43_ed69b0cd_t364_f0__correct | sentence_packed | `fuse_max` | grounded | correct | 0.9832 | 1.0000 | 0.0000 | False | 0.9970 | 81 |
| base_43_ed69b0cd_t364_f0__correct | sentence_packed | `fuse_min` | grounded | correct | 0.9488 | 1.0000 | 0.0000 | False | 0.9896 | 81 |
| base_43_ed69b0cd_t364_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9739 | 1.0000 | 0.0000 | False | 0.9982 | 81 |
| base_43_ed69b0cd_t364_f0__wrong | sentence_packed | `code_only` | ungrounded | wrong | 0.9324 | 1.0000 | 0.0000 | False | 0.9761 | 81 |
| base_43_ed69b0cd_t364_f0__wrong | sentence_packed | `fuse_mean` | ungrounded | wrong | 0.9532 | 1.0000 | 0.0000 | False | 0.9871 | 81 |
| base_43_ed69b0cd_t364_f0__wrong | sentence_packed | `fuse_max` | ungrounded | wrong | 0.9739 | 1.0000 | 0.0000 | False | 0.9982 | 81 |
| base_43_ed69b0cd_t364_f0__wrong | sentence_packed | `fuse_min` | ungrounded | wrong | 0.9324 | 1.0000 | 0.0000 | False | 0.9761 | 81 |
| base_43_ed69b0cd_t364_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9778 | 1.0000 | 0.0000 | False | 0.9940 | 81 |
| base_43_ed69b0cd_t364_f0__ambiguous | sentence_packed | `code_only` | ambiguous | ambiguous | 0.9342 | 1.0000 | 0.0000 | False | 0.9686 | 81 |
| base_43_ed69b0cd_t364_f0__ambiguous | sentence_packed | `fuse_mean` | ambiguous | ambiguous | 0.9560 | 1.0000 | 0.0000 | False | 0.9813 | 81 |
| base_43_ed69b0cd_t364_f0__ambiguous | sentence_packed | `fuse_max` | ambiguous | ambiguous | 0.9778 | 1.0000 | 0.0000 | False | 0.9940 | 81 |
| base_43_ed69b0cd_t364_f0__ambiguous | sentence_packed | `fuse_min` | ambiguous | ambiguous | 0.9342 | 1.0000 | 0.0000 | False | 0.9686 | 81 |
| base_44_a4d14a56_t568_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9825 | 1.0000 | 0.0000 | False | 0.9975 | 67 |
| base_44_a4d14a56_t568_f0__correct | sentence_packed | `code_only` | grounded | correct | 0.9375 | 1.0000 | 0.0000 | False | 0.9807 | 67 |
| base_44_a4d14a56_t568_f0__correct | sentence_packed | `fuse_mean` | grounded | correct | 0.9600 | 1.0000 | 0.0000 | False | 0.9891 | 67 |
| base_44_a4d14a56_t568_f0__correct | sentence_packed | `fuse_max` | grounded | correct | 0.9825 | 1.0000 | 0.0000 | False | 0.9975 | 67 |
| base_44_a4d14a56_t568_f0__correct | sentence_packed | `fuse_min` | grounded | correct | 0.9375 | 1.0000 | 0.0000 | False | 0.9807 | 67 |
| base_44_a4d14a56_t568_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9655 | 1.0000 | 0.0000 | False | 0.9982 | 67 |
| base_44_a4d14a56_t568_f0__wrong | sentence_packed | `code_only` | ungrounded | wrong | 0.9236 | 1.0000 | 0.0000 | False | 0.9824 | 67 |
| base_44_a4d14a56_t568_f0__wrong | sentence_packed | `fuse_mean` | ungrounded | wrong | 0.9445 | 1.0000 | 0.0000 | False | 0.9903 | 67 |
| base_44_a4d14a56_t568_f0__wrong | sentence_packed | `fuse_max` | ungrounded | wrong | 0.9655 | 1.0000 | 0.0000 | False | 0.9982 | 67 |
| base_44_a4d14a56_t568_f0__wrong | sentence_packed | `fuse_min` | ungrounded | wrong | 0.9236 | 1.0000 | 0.0000 | False | 0.9824 | 67 |
| base_44_a4d14a56_t568_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9804 | 1.0000 | 0.0000 | False | 0.9976 | 67 |
| base_44_a4d14a56_t568_f0__ambiguous | sentence_packed | `code_only` | ambiguous | ambiguous | 0.9388 | 1.0000 | 0.0000 | False | 0.9806 | 67 |
| base_44_a4d14a56_t568_f0__ambiguous | sentence_packed | `fuse_mean` | ambiguous | ambiguous | 0.9596 | 1.0000 | 0.0000 | False | 0.9891 | 67 |
| base_44_a4d14a56_t568_f0__ambiguous | sentence_packed | `fuse_max` | ambiguous | ambiguous | 0.9804 | 1.0000 | 0.0000 | False | 0.9976 | 67 |
| base_44_a4d14a56_t568_f0__ambiguous | sentence_packed | `fuse_min` | ambiguous | ambiguous | 0.9388 | 1.0000 | 0.0000 | False | 0.9806 | 67 |
| base_45_ed69b0cd_t367_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9804 | 1.0000 | 0.0000 | False | 1.0000 | 81 |
| base_45_ed69b0cd_t367_f0__correct | sentence_packed | `code_only` | grounded | correct | 0.9328 | 1.0000 | 0.0000 | False | 1.0000 | 81 |
| base_45_ed69b0cd_t367_f0__correct | sentence_packed | `fuse_mean` | grounded | correct | 0.9566 | 1.0000 | 0.0000 | False | 1.0000 | 81 |
| base_45_ed69b0cd_t367_f0__correct | sentence_packed | `fuse_max` | grounded | correct | 0.9804 | 1.0000 | 0.0000 | False | 1.0000 | 81 |
| base_45_ed69b0cd_t367_f0__correct | sentence_packed | `fuse_min` | grounded | correct | 0.9328 | 1.0000 | 0.0000 | False | 1.0000 | 81 |
| base_45_ed69b0cd_t367_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9732 | 1.0000 | 0.0000 | False | 1.0000 | 81 |
| base_45_ed69b0cd_t367_f0__wrong | sentence_packed | `code_only` | ungrounded | wrong | 0.9216 | 1.0000 | 0.0000 | False | 1.0000 | 81 |
| base_45_ed69b0cd_t367_f0__wrong | sentence_packed | `fuse_mean` | ungrounded | wrong | 0.9474 | 1.0000 | 0.0000 | False | 1.0000 | 81 |
| base_45_ed69b0cd_t367_f0__wrong | sentence_packed | `fuse_max` | ungrounded | wrong | 0.9732 | 1.0000 | 0.0000 | False | 1.0000 | 81 |
| base_45_ed69b0cd_t367_f0__wrong | sentence_packed | `fuse_min` | ungrounded | wrong | 0.9216 | 1.0000 | 0.0000 | False | 1.0000 | 81 |
| base_45_ed69b0cd_t367_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9794 | 1.0000 | 0.0000 | False | 1.0000 | 81 |
| base_45_ed69b0cd_t367_f0__ambiguous | sentence_packed | `code_only` | ambiguous | ambiguous | 0.9309 | 1.0000 | 0.0000 | False | 1.0000 | 81 |
| base_45_ed69b0cd_t367_f0__ambiguous | sentence_packed | `fuse_mean` | ambiguous | ambiguous | 0.9552 | 1.0000 | 0.0000 | False | 1.0000 | 81 |
| base_45_ed69b0cd_t367_f0__ambiguous | sentence_packed | `fuse_max` | ambiguous | ambiguous | 0.9794 | 1.0000 | 0.0000 | False | 1.0000 | 81 |
| base_45_ed69b0cd_t367_f0__ambiguous | sentence_packed | `fuse_min` | ambiguous | ambiguous | 0.9309 | 1.0000 | 0.0000 | False | 1.0000 | 81 |
| base_46_d7fd7127_t317_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9788 | 1.0000 | 0.0000 | False | 0.9939 | 81 |
| base_46_d7fd7127_t317_f0__correct | sentence_packed | `code_only` | grounded | correct | 0.9461 | 1.0000 | 0.0000 | False | 0.9936 | 81 |
| base_46_d7fd7127_t317_f0__correct | sentence_packed | `fuse_mean` | grounded | correct | 0.9624 | 1.0000 | 0.0000 | False | 0.9937 | 81 |
| base_46_d7fd7127_t317_f0__correct | sentence_packed | `fuse_max` | grounded | correct | 0.9788 | 1.0000 | 0.0000 | False | 0.9939 | 81 |
| base_46_d7fd7127_t317_f0__correct | sentence_packed | `fuse_min` | grounded | correct | 0.9461 | 1.0000 | 0.0000 | False | 0.9936 | 81 |
| base_46_d7fd7127_t317_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9684 | 1.0000 | 0.0000 | False | 0.9919 | 81 |
| base_46_d7fd7127_t317_f0__wrong | sentence_packed | `code_only` | ungrounded | wrong | 0.9377 | 1.0000 | 0.0000 | False | 0.9836 | 81 |
| base_46_d7fd7127_t317_f0__wrong | sentence_packed | `fuse_mean` | ungrounded | wrong | 0.9531 | 1.0000 | 0.0000 | False | 0.9878 | 81 |
| base_46_d7fd7127_t317_f0__wrong | sentence_packed | `fuse_max` | ungrounded | wrong | 0.9684 | 1.0000 | 0.0000 | False | 0.9919 | 81 |
| base_46_d7fd7127_t317_f0__wrong | sentence_packed | `fuse_min` | ungrounded | wrong | 0.9377 | 1.0000 | 0.0000 | False | 0.9836 | 81 |
| base_46_d7fd7127_t317_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9773 | 1.0000 | 0.0000 | False | 0.9939 | 81 |
| base_46_d7fd7127_t317_f0__ambiguous | sentence_packed | `code_only` | ambiguous | ambiguous | 0.9452 | 1.0000 | 0.0000 | False | 0.9946 | 81 |
| base_46_d7fd7127_t317_f0__ambiguous | sentence_packed | `fuse_mean` | ambiguous | ambiguous | 0.9613 | 1.0000 | 0.0000 | False | 0.9942 | 81 |
| base_46_d7fd7127_t317_f0__ambiguous | sentence_packed | `fuse_max` | ambiguous | ambiguous | 0.9773 | 1.0000 | 0.0000 | False | 0.9946 | 81 |
| base_46_d7fd7127_t317_f0__ambiguous | sentence_packed | `fuse_min` | ambiguous | ambiguous | 0.9452 | 1.0000 | 0.0000 | False | 0.9939 | 81 |
| base_47_0b1422bb_t847_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9507 | 1.0000 | 0.0000 | False | 0.9999 | 25 |
| base_47_0b1422bb_t847_f0__correct | sentence_packed | `code_only` | grounded | correct | 0.8896 | 1.0000 | 0.0000 | False | 0.9781 | 25 |
| base_47_0b1422bb_t847_f0__correct | sentence_packed | `fuse_mean` | grounded | correct | 0.9201 | 1.0000 | 0.0000 | False | 0.9890 | 25 |
| base_47_0b1422bb_t847_f0__correct | sentence_packed | `fuse_max` | grounded | correct | 0.9507 | 1.0000 | 0.0000 | False | 0.9999 | 25 |
| base_47_0b1422bb_t847_f0__correct | sentence_packed | `fuse_min` | grounded | correct | 0.8896 | 1.0000 | 0.0000 | False | 0.9781 | 25 |
| base_47_0b1422bb_t847_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9463 | 1.0000 | 0.0000 | False | 0.9966 | 25 |
| base_47_0b1422bb_t847_f0__wrong | sentence_packed | `code_only` | ungrounded | wrong | 0.8991 | 1.0000 | 0.0000 | False | 0.9784 | 25 |
| base_47_0b1422bb_t847_f0__wrong | sentence_packed | `fuse_mean` | ungrounded | wrong | 0.9227 | 1.0000 | 0.0000 | False | 0.9875 | 25 |
| base_47_0b1422bb_t847_f0__wrong | sentence_packed | `fuse_max` | ungrounded | wrong | 0.9463 | 1.0000 | 0.0000 | False | 0.9966 | 25 |
| base_47_0b1422bb_t847_f0__wrong | sentence_packed | `fuse_min` | ungrounded | wrong | 0.8991 | 1.0000 | 0.0000 | False | 0.9784 | 25 |
| base_47_0b1422bb_t847_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9502 | 1.0000 | 0.0000 | False | 0.9997 | 25 |
| base_47_0b1422bb_t847_f0__ambiguous | sentence_packed | `code_only` | ambiguous | ambiguous | 0.8903 | 1.0000 | 0.0000 | False | 0.9766 | 25 |
| base_47_0b1422bb_t847_f0__ambiguous | sentence_packed | `fuse_mean` | ambiguous | ambiguous | 0.9203 | 1.0000 | 0.0000 | False | 0.9881 | 25 |
| base_47_0b1422bb_t847_f0__ambiguous | sentence_packed | `fuse_max` | ambiguous | ambiguous | 0.9502 | 1.0000 | 0.0000 | False | 0.9997 | 25 |
| base_47_0b1422bb_t847_f0__ambiguous | sentence_packed | `fuse_min` | ambiguous | ambiguous | 0.8903 | 1.0000 | 0.0000 | False | 0.9766 | 25 |
| base_48_ed69b0cd_t10_f2__correct | sentence_packed | `gte_only` | grounded | correct | 0.9600 | 1.0000 | 0.0000 | False | 0.9890 | 19 |
| base_48_ed69b0cd_t10_f2__correct | sentence_packed | `code_only` | grounded | correct | 0.9155 | 1.0000 | 0.0000 | False | 0.9852 | 19 |
| base_48_ed69b0cd_t10_f2__correct | sentence_packed | `fuse_mean` | grounded | correct | 0.9378 | 1.0000 | 0.0000 | False | 0.9871 | 19 |
| base_48_ed69b0cd_t10_f2__correct | sentence_packed | `fuse_max` | grounded | correct | 0.9600 | 1.0000 | 0.0000 | False | 0.9890 | 19 |
| base_48_ed69b0cd_t10_f2__correct | sentence_packed | `fuse_min` | grounded | correct | 0.9155 | 1.0000 | 0.0000 | False | 0.9852 | 19 |
| base_48_ed69b0cd_t10_f2__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9524 | 1.0000 | 0.0000 | False | 0.9902 | 19 |
| base_48_ed69b0cd_t10_f2__wrong | sentence_packed | `code_only` | ungrounded | wrong | 0.9048 | 1.0000 | 0.0000 | False | 0.9842 | 19 |
| base_48_ed69b0cd_t10_f2__wrong | sentence_packed | `fuse_mean` | ungrounded | wrong | 0.9286 | 1.0000 | 0.0000 | False | 0.9872 | 19 |
| base_48_ed69b0cd_t10_f2__wrong | sentence_packed | `fuse_max` | ungrounded | wrong | 0.9524 | 1.0000 | 0.0000 | False | 0.9902 | 19 |
| base_48_ed69b0cd_t10_f2__wrong | sentence_packed | `fuse_min` | ungrounded | wrong | 0.9048 | 1.0000 | 0.0000 | False | 0.9842 | 19 |
| base_48_ed69b0cd_t10_f2__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9598 | 1.0000 | 0.0000 | False | 0.9892 | 19 |
| base_48_ed69b0cd_t10_f2__ambiguous | sentence_packed | `code_only` | ambiguous | ambiguous | 0.9160 | 1.0000 | 0.0000 | False | 0.9849 | 19 |
| base_48_ed69b0cd_t10_f2__ambiguous | sentence_packed | `fuse_mean` | ambiguous | ambiguous | 0.9379 | 1.0000 | 0.0000 | False | 0.9871 | 19 |
| base_48_ed69b0cd_t10_f2__ambiguous | sentence_packed | `fuse_max` | ambiguous | ambiguous | 0.9598 | 1.0000 | 0.0000 | False | 0.9892 | 19 |
| base_48_ed69b0cd_t10_f2__ambiguous | sentence_packed | `fuse_min` | ambiguous | ambiguous | 0.9160 | 1.0000 | 0.0000 | False | 0.9849 | 19 |
| base_49_fa052757_t213_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9709 | 1.0000 | 0.0000 | False | 0.9988 | 44 |
| base_49_fa052757_t213_f0__correct | sentence_packed | `code_only` | grounded | correct | 0.9127 | 1.0000 | 0.0000 | False | 0.9799 | 44 |
| base_49_fa052757_t213_f0__correct | sentence_packed | `fuse_mean` | grounded | correct | 0.9418 | 1.0000 | 0.0000 | False | 0.9893 | 44 |
| base_49_fa052757_t213_f0__correct | sentence_packed | `fuse_max` | grounded | correct | 0.9709 | 1.0000 | 0.0000 | False | 0.9988 | 44 |
| base_49_fa052757_t213_f0__correct | sentence_packed | `fuse_min` | grounded | correct | 0.9127 | 1.0000 | 0.0000 | False | 0.9799 | 44 |
| base_49_fa052757_t213_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9597 | 1.0000 | 0.0000 | False | 0.9989 | 44 |
| base_49_fa052757_t213_f0__wrong | sentence_packed | `code_only` | ungrounded | wrong | 0.9071 | 1.0000 | 0.0000 | False | 0.9682 | 44 |
| base_49_fa052757_t213_f0__wrong | sentence_packed | `fuse_mean` | ungrounded | wrong | 0.9334 | 1.0000 | 0.0000 | False | 0.9835 | 44 |
| base_49_fa052757_t213_f0__wrong | sentence_packed | `fuse_max` | ungrounded | wrong | 0.9597 | 1.0000 | 0.0000 | False | 0.9989 | 44 |
| base_49_fa052757_t213_f0__wrong | sentence_packed | `fuse_min` | ungrounded | wrong | 0.9071 | 1.0000 | 0.0000 | False | 0.9682 | 44 |
| base_49_fa052757_t213_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9706 | 1.0000 | 0.0000 | False | 0.9988 | 44 |
| base_49_fa052757_t213_f0__ambiguous | sentence_packed | `code_only` | ambiguous | ambiguous | 0.9137 | 1.0000 | 0.0000 | False | 0.9677 | 44 |
| base_49_fa052757_t213_f0__ambiguous | sentence_packed | `fuse_mean` | ambiguous | ambiguous | 0.9421 | 1.0000 | 0.0000 | False | 0.9833 | 44 |
| base_49_fa052757_t213_f0__ambiguous | sentence_packed | `fuse_max` | ambiguous | ambiguous | 0.9706 | 1.0000 | 0.0000 | False | 0.9988 | 44 |
| base_49_fa052757_t213_f0__ambiguous | sentence_packed | `fuse_min` | ambiguous | ambiguous | 0.9137 | 1.0000 | 0.0000 | False | 0.9677 | 44 |
| base_50_0b1422bb_t839_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9615 | 1.0000 | 0.0000 | False | 0.9999 | 25 |
| base_50_0b1422bb_t839_f0__correct | sentence_packed | `code_only` | grounded | correct | 0.8975 | 1.0000 | 0.0000 | False | 0.9818 | 25 |
| base_50_0b1422bb_t839_f0__correct | sentence_packed | `fuse_mean` | grounded | correct | 0.9295 | 1.0000 | 0.0000 | False | 0.9909 | 25 |
| base_50_0b1422bb_t839_f0__correct | sentence_packed | `fuse_max` | grounded | correct | 0.9615 | 1.0000 | 0.0000 | False | 0.9999 | 25 |
| base_50_0b1422bb_t839_f0__correct | sentence_packed | `fuse_min` | grounded | correct | 0.8975 | 1.0000 | 0.0000 | False | 0.9818 | 25 |
| base_50_0b1422bb_t839_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9578 | 1.0000 | 0.0000 | False | 0.9999 | 25 |
| base_50_0b1422bb_t839_f0__wrong | sentence_packed | `code_only` | ungrounded | wrong | 0.9045 | 1.0000 | 0.0000 | False | 0.9905 | 25 |
| base_50_0b1422bb_t839_f0__wrong | sentence_packed | `fuse_mean` | ungrounded | wrong | 0.9312 | 1.0000 | 0.0000 | False | 0.9952 | 25 |
| base_50_0b1422bb_t839_f0__wrong | sentence_packed | `fuse_max` | ungrounded | wrong | 0.9578 | 1.0000 | 0.0000 | False | 0.9999 | 25 |
| base_50_0b1422bb_t839_f0__wrong | sentence_packed | `fuse_min` | ungrounded | wrong | 0.9045 | 1.0000 | 0.0000 | False | 0.9905 | 25 |
| base_50_0b1422bb_t839_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9587 | 1.0000 | 0.0000 | False | 0.9999 | 25 |
| base_50_0b1422bb_t839_f0__ambiguous | sentence_packed | `code_only` | ambiguous | ambiguous | 0.9059 | 1.0000 | 0.0000 | False | 0.9843 | 25 |
| base_50_0b1422bb_t839_f0__ambiguous | sentence_packed | `fuse_mean` | ambiguous | ambiguous | 0.9323 | 1.0000 | 0.0000 | False | 0.9921 | 25 |
| base_50_0b1422bb_t839_f0__ambiguous | sentence_packed | `fuse_max` | ambiguous | ambiguous | 0.9587 | 1.0000 | 0.0000 | False | 0.9999 | 25 |
| base_50_0b1422bb_t839_f0__ambiguous | sentence_packed | `fuse_min` | ambiguous | ambiguous | 0.9059 | 1.0000 | 0.0000 | False | 0.9843 | 25 |
| base_51_5e875b7e_t1929_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9741 | 1.0000 | 0.0000 | False | 0.9999 | 61 |
| base_51_5e875b7e_t1929_f0__correct | sentence_packed | `code_only` | grounded | correct | 0.9386 | 1.0000 | 0.0000 | False | 1.0000 | 61 |
| base_51_5e875b7e_t1929_f0__correct | sentence_packed | `fuse_mean` | grounded | correct | 0.9563 | 1.0000 | 0.0000 | False | 1.0000 | 61 |
| base_51_5e875b7e_t1929_f0__correct | sentence_packed | `fuse_max` | grounded | correct | 0.9741 | 1.0000 | 0.0000 | False | 1.0000 | 61 |
| base_51_5e875b7e_t1929_f0__correct | sentence_packed | `fuse_min` | grounded | correct | 0.9386 | 1.0000 | 0.0000 | False | 0.9999 | 61 |
| base_51_5e875b7e_t1929_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9608 | 1.0000 | 0.0000 | False | 1.0000 | 61 |
| base_51_5e875b7e_t1929_f0__wrong | sentence_packed | `code_only` | ungrounded | wrong | 0.9186 | 1.0000 | 0.0000 | False | 1.0000 | 61 |
| base_51_5e875b7e_t1929_f0__wrong | sentence_packed | `fuse_mean` | ungrounded | wrong | 0.9397 | 1.0000 | 0.0000 | False | 1.0000 | 61 |
| base_51_5e875b7e_t1929_f0__wrong | sentence_packed | `fuse_max` | ungrounded | wrong | 0.9608 | 1.0000 | 0.0000 | False | 1.0000 | 61 |
| base_51_5e875b7e_t1929_f0__wrong | sentence_packed | `fuse_min` | ungrounded | wrong | 0.9186 | 1.0000 | 0.0000 | False | 1.0000 | 61 |
| base_51_5e875b7e_t1929_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9727 | 1.0000 | 0.0000 | False | 1.0000 | 61 |
| base_51_5e875b7e_t1929_f0__ambiguous | sentence_packed | `code_only` | ambiguous | ambiguous | 0.9327 | 1.0000 | 0.0000 | False | 1.0000 | 61 |
| base_51_5e875b7e_t1929_f0__ambiguous | sentence_packed | `fuse_mean` | ambiguous | ambiguous | 0.9527 | 1.0000 | 0.0000 | False | 1.0000 | 61 |
| base_51_5e875b7e_t1929_f0__ambiguous | sentence_packed | `fuse_max` | ambiguous | ambiguous | 0.9727 | 1.0000 | 0.0000 | False | 1.0000 | 61 |
| base_51_5e875b7e_t1929_f0__ambiguous | sentence_packed | `fuse_min` | ambiguous | ambiguous | 0.9327 | 1.0000 | 0.0000 | False | 1.0000 | 61 |
| base_52_7b7fb621_t310_f3__correct | sentence_packed | `gte_only` | grounded | correct | 0.9835 | 1.0000 | 0.0000 | False | 0.9989 | 37 |
| base_52_7b7fb621_t310_f3__correct | sentence_packed | `code_only` | grounded | correct | 0.9541 | 1.0000 | 0.0000 | False | 0.9915 | 37 |
| base_52_7b7fb621_t310_f3__correct | sentence_packed | `fuse_mean` | grounded | correct | 0.9688 | 1.0000 | 0.0000 | False | 0.9952 | 37 |
| base_52_7b7fb621_t310_f3__correct | sentence_packed | `fuse_max` | grounded | correct | 0.9835 | 1.0000 | 0.0000 | False | 0.9989 | 37 |
| base_52_7b7fb621_t310_f3__correct | sentence_packed | `fuse_min` | grounded | correct | 0.9541 | 1.0000 | 0.0000 | False | 0.9915 | 37 |
| base_52_7b7fb621_t310_f3__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9754 | 1.0000 | 0.0000 | False | 0.9984 | 37 |
| base_52_7b7fb621_t310_f3__wrong | sentence_packed | `code_only` | ungrounded | wrong | 0.9403 | 1.0000 | 0.0000 | False | 0.9909 | 37 |
| base_52_7b7fb621_t310_f3__wrong | sentence_packed | `fuse_mean` | ungrounded | wrong | 0.9579 | 1.0000 | 0.0000 | False | 0.9946 | 37 |
| base_52_7b7fb621_t310_f3__wrong | sentence_packed | `fuse_max` | ungrounded | wrong | 0.9754 | 1.0000 | 0.0000 | False | 0.9984 | 37 |
| base_52_7b7fb621_t310_f3__wrong | sentence_packed | `fuse_min` | ungrounded | wrong | 0.9403 | 1.0000 | 0.0000 | False | 0.9909 | 37 |
| base_52_7b7fb621_t310_f3__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9850 | 1.0000 | 0.0000 | False | 1.0000 | 37 |
| base_52_7b7fb621_t310_f3__ambiguous | sentence_packed | `code_only` | ambiguous | ambiguous | 0.9487 | 1.0000 | 0.0000 | False | 1.0000 | 37 |
| base_52_7b7fb621_t310_f3__ambiguous | sentence_packed | `fuse_mean` | ambiguous | ambiguous | 0.9669 | 1.0000 | 0.0000 | False | 1.0000 | 37 |
| base_52_7b7fb621_t310_f3__ambiguous | sentence_packed | `fuse_max` | ambiguous | ambiguous | 0.9850 | 1.0000 | 0.0000 | False | 1.0000 | 37 |
| base_52_7b7fb621_t310_f3__ambiguous | sentence_packed | `fuse_min` | ambiguous | ambiguous | 0.9487 | 1.0000 | 0.0000 | False | 1.0000 | 37 |
| base_53_7b7fb621_t347_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9785 | 1.0000 | 0.0000 | False | 0.9985 | 36 |
| base_53_7b7fb621_t347_f0__correct | sentence_packed | `code_only` | grounded | correct | 0.9264 | 1.0000 | 0.0000 | False | 1.0000 | 36 |
| base_53_7b7fb621_t347_f0__correct | sentence_packed | `fuse_mean` | grounded | correct | 0.9524 | 1.0000 | 0.0000 | False | 0.9992 | 36 |
| base_53_7b7fb621_t347_f0__correct | sentence_packed | `fuse_max` | grounded | correct | 0.9785 | 1.0000 | 0.0000 | False | 1.0000 | 36 |
| base_53_7b7fb621_t347_f0__correct | sentence_packed | `fuse_min` | grounded | correct | 0.9264 | 1.0000 | 0.0000 | False | 0.9985 | 36 |
| base_53_7b7fb621_t347_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9738 | 1.0000 | 0.0000 | False | 0.9993 | 36 |
| base_53_7b7fb621_t347_f0__wrong | sentence_packed | `code_only` | ungrounded | wrong | 0.9283 | 1.0000 | 0.0000 | False | 1.0000 | 36 |
| base_53_7b7fb621_t347_f0__wrong | sentence_packed | `fuse_mean` | ungrounded | wrong | 0.9510 | 1.0000 | 0.0000 | False | 0.9997 | 36 |
| base_53_7b7fb621_t347_f0__wrong | sentence_packed | `fuse_max` | ungrounded | wrong | 0.9738 | 1.0000 | 0.0000 | False | 1.0000 | 36 |
| base_53_7b7fb621_t347_f0__wrong | sentence_packed | `fuse_min` | ungrounded | wrong | 0.9283 | 1.0000 | 0.0000 | False | 0.9993 | 36 |
| base_53_7b7fb621_t347_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9788 | 1.0000 | 0.0000 | False | 0.9991 | 36 |
| base_53_7b7fb621_t347_f0__ambiguous | sentence_packed | `code_only` | ambiguous | ambiguous | 0.9311 | 1.0000 | 0.0000 | False | 1.0000 | 36 |
| base_53_7b7fb621_t347_f0__ambiguous | sentence_packed | `fuse_mean` | ambiguous | ambiguous | 0.9549 | 1.0000 | 0.0000 | False | 0.9995 | 36 |
| base_53_7b7fb621_t347_f0__ambiguous | sentence_packed | `fuse_max` | ambiguous | ambiguous | 0.9788 | 1.0000 | 0.0000 | False | 1.0000 | 36 |
| base_53_7b7fb621_t347_f0__ambiguous | sentence_packed | `fuse_min` | ambiguous | ambiguous | 0.9311 | 1.0000 | 0.0000 | False | 0.9991 | 36 |
| base_54_5e875b7e_t630_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9725 | 1.0000 | 0.0000 | False | 0.9932 | 59 |
| base_54_5e875b7e_t630_f0__correct | sentence_packed | `code_only` | grounded | correct | 0.9027 | 1.0000 | 0.0000 | False | 0.9734 | 59 |
| base_54_5e875b7e_t630_f0__correct | sentence_packed | `fuse_mean` | grounded | correct | 0.9376 | 1.0000 | 0.0000 | False | 0.9833 | 59 |
| base_54_5e875b7e_t630_f0__correct | sentence_packed | `fuse_max` | grounded | correct | 0.9725 | 1.0000 | 0.0000 | False | 0.9932 | 59 |
| base_54_5e875b7e_t630_f0__correct | sentence_packed | `fuse_min` | grounded | correct | 0.9027 | 1.0000 | 0.0000 | False | 0.9734 | 59 |
| base_54_5e875b7e_t630_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9710 | 1.0000 | 0.0000 | False | 0.9946 | 59 |
| base_54_5e875b7e_t630_f0__wrong | sentence_packed | `code_only` | ungrounded | wrong | 0.9004 | 1.0000 | 0.0000 | False | 1.0000 | 59 |
| base_54_5e875b7e_t630_f0__wrong | sentence_packed | `fuse_mean` | ungrounded | wrong | 0.9357 | 1.0000 | 0.0000 | False | 0.9973 | 59 |
| base_54_5e875b7e_t630_f0__wrong | sentence_packed | `fuse_max` | ungrounded | wrong | 0.9710 | 1.0000 | 0.0000 | False | 1.0000 | 59 |
| base_54_5e875b7e_t630_f0__wrong | sentence_packed | `fuse_min` | ungrounded | wrong | 0.9004 | 1.0000 | 0.0000 | False | 0.9946 | 59 |
| base_54_5e875b7e_t630_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9713 | 1.0000 | 0.0000 | False | 0.9924 | 59 |
| base_54_5e875b7e_t630_f0__ambiguous | sentence_packed | `code_only` | ambiguous | ambiguous | 0.9047 | 1.0000 | 0.0000 | False | 0.9709 | 59 |
| base_54_5e875b7e_t630_f0__ambiguous | sentence_packed | `fuse_mean` | ambiguous | ambiguous | 0.9380 | 1.0000 | 0.0000 | False | 0.9817 | 59 |
| base_54_5e875b7e_t630_f0__ambiguous | sentence_packed | `fuse_max` | ambiguous | ambiguous | 0.9713 | 1.0000 | 0.0000 | False | 0.9924 | 59 |
| base_54_5e875b7e_t630_f0__ambiguous | sentence_packed | `fuse_min` | ambiguous | ambiguous | 0.9047 | 1.0000 | 0.0000 | False | 0.9709 | 59 |
| base_55_0b1422bb_t652_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9731 | 1.0000 | 0.0000 | False | 0.9977 | 23 |
| base_55_0b1422bb_t652_f0__correct | sentence_packed | `code_only` | grounded | correct | 0.9165 | 1.0000 | 0.0000 | False | 1.0000 | 23 |
| base_55_0b1422bb_t652_f0__correct | sentence_packed | `fuse_mean` | grounded | correct | 0.9448 | 1.0000 | 0.0000 | False | 0.9989 | 23 |
| base_55_0b1422bb_t652_f0__correct | sentence_packed | `fuse_max` | grounded | correct | 0.9731 | 1.0000 | 0.0000 | False | 1.0000 | 23 |
| base_55_0b1422bb_t652_f0__correct | sentence_packed | `fuse_min` | grounded | correct | 0.9165 | 1.0000 | 0.0000 | False | 0.9977 | 23 |
| base_55_0b1422bb_t652_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9645 | 1.0000 | 0.0000 | False | 0.9975 | 23 |
| base_55_0b1422bb_t652_f0__wrong | sentence_packed | `code_only` | ungrounded | wrong | 0.9075 | 1.0000 | 0.0000 | False | 0.9940 | 23 |
| base_55_0b1422bb_t652_f0__wrong | sentence_packed | `fuse_mean` | ungrounded | wrong | 0.9360 | 1.0000 | 0.0000 | False | 0.9957 | 23 |
| base_55_0b1422bb_t652_f0__wrong | sentence_packed | `fuse_max` | ungrounded | wrong | 0.9645 | 1.0000 | 0.0000 | False | 0.9975 | 23 |
| base_55_0b1422bb_t652_f0__wrong | sentence_packed | `fuse_min` | ungrounded | wrong | 0.9075 | 1.0000 | 0.0000 | False | 0.9940 | 23 |
| base_55_0b1422bb_t652_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9731 | 1.0000 | 0.0000 | False | 0.9978 | 23 |
| base_55_0b1422bb_t652_f0__ambiguous | sentence_packed | `code_only` | ambiguous | ambiguous | 0.9165 | 1.0000 | 0.0000 | False | 1.0000 | 23 |
| base_55_0b1422bb_t652_f0__ambiguous | sentence_packed | `fuse_mean` | ambiguous | ambiguous | 0.9448 | 1.0000 | 0.0000 | False | 0.9989 | 23 |
| base_55_0b1422bb_t652_f0__ambiguous | sentence_packed | `fuse_max` | ambiguous | ambiguous | 0.9731 | 1.0000 | 0.0000 | False | 1.0000 | 23 |
| base_55_0b1422bb_t652_f0__ambiguous | sentence_packed | `fuse_min` | ambiguous | ambiguous | 0.9165 | 1.0000 | 0.0000 | False | 0.9978 | 23 |
| base_56_7b7fb621_t406_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9745 | 1.0000 | 0.0000 | False | 1.0000 | 47 |
| base_56_7b7fb621_t406_f0__correct | sentence_packed | `code_only` | grounded | correct | 0.9269 | 1.0000 | 0.0000 | False | 0.9999 | 47 |
| base_56_7b7fb621_t406_f0__correct | sentence_packed | `fuse_mean` | grounded | correct | 0.9507 | 1.0000 | 0.0000 | False | 0.9999 | 47 |
| base_56_7b7fb621_t406_f0__correct | sentence_packed | `fuse_max` | grounded | correct | 0.9745 | 1.0000 | 0.0000 | False | 1.0000 | 47 |
| base_56_7b7fb621_t406_f0__correct | sentence_packed | `fuse_min` | grounded | correct | 0.9269 | 1.0000 | 0.0000 | False | 0.9999 | 47 |
| base_56_7b7fb621_t406_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9720 | 1.0000 | 0.0000 | False | 0.9999 | 47 |
| base_56_7b7fb621_t406_f0__wrong | sentence_packed | `code_only` | ungrounded | wrong | 0.9216 | 1.0000 | 0.0000 | False | 0.9999 | 47 |
| base_56_7b7fb621_t406_f0__wrong | sentence_packed | `fuse_mean` | ungrounded | wrong | 0.9468 | 1.0000 | 0.0000 | False | 0.9999 | 47 |
| base_56_7b7fb621_t406_f0__wrong | sentence_packed | `fuse_max` | ungrounded | wrong | 0.9720 | 1.0000 | 0.0000 | False | 0.9999 | 47 |
| base_56_7b7fb621_t406_f0__wrong | sentence_packed | `fuse_min` | ungrounded | wrong | 0.9216 | 1.0000 | 0.0000 | False | 0.9999 | 47 |
| base_56_7b7fb621_t406_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9742 | 1.0000 | 0.0000 | False | 1.0000 | 47 |
| base_56_7b7fb621_t406_f0__ambiguous | sentence_packed | `code_only` | ambiguous | ambiguous | 0.9261 | 1.0000 | 0.0000 | False | 0.9999 | 47 |
| base_56_7b7fb621_t406_f0__ambiguous | sentence_packed | `fuse_mean` | ambiguous | ambiguous | 0.9502 | 1.0000 | 0.0000 | False | 0.9999 | 47 |
| base_56_7b7fb621_t406_f0__ambiguous | sentence_packed | `fuse_max` | ambiguous | ambiguous | 0.9742 | 1.0000 | 0.0000 | False | 1.0000 | 47 |
| base_56_7b7fb621_t406_f0__ambiguous | sentence_packed | `fuse_min` | ambiguous | ambiguous | 0.9261 | 1.0000 | 0.0000 | False | 0.9999 | 47 |
| base_57_91cf0ec9_t84_f10__correct | sentence_packed | `gte_only` | grounded | correct | 0.9789 | 1.0000 | 0.0000 | False | 0.9874 | 40 |
| base_57_91cf0ec9_t84_f10__correct | sentence_packed | `code_only` | grounded | correct | 0.9306 | 1.0000 | 0.0000 | False | 0.9999 | 40 |
| base_57_91cf0ec9_t84_f10__correct | sentence_packed | `fuse_mean` | grounded | correct | 0.9547 | 1.0000 | 0.0000 | False | 0.9936 | 40 |
| base_57_91cf0ec9_t84_f10__correct | sentence_packed | `fuse_max` | grounded | correct | 0.9789 | 1.0000 | 0.0000 | False | 0.9999 | 40 |
| base_57_91cf0ec9_t84_f10__correct | sentence_packed | `fuse_min` | grounded | correct | 0.9306 | 1.0000 | 0.0000 | False | 0.9874 | 40 |
| base_57_91cf0ec9_t84_f10__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9728 | 1.0000 | 0.0000 | False | 0.9983 | 40 |
| base_57_91cf0ec9_t84_f10__wrong | sentence_packed | `code_only` | ungrounded | wrong | 0.9240 | 1.0000 | 0.0000 | False | 0.9999 | 40 |
| base_57_91cf0ec9_t84_f10__wrong | sentence_packed | `fuse_mean` | ungrounded | wrong | 0.9484 | 1.0000 | 0.0000 | False | 0.9991 | 40 |
| base_57_91cf0ec9_t84_f10__wrong | sentence_packed | `fuse_max` | ungrounded | wrong | 0.9728 | 1.0000 | 0.0000 | False | 0.9999 | 40 |
| base_57_91cf0ec9_t84_f10__wrong | sentence_packed | `fuse_min` | ungrounded | wrong | 0.9240 | 1.0000 | 0.0000 | False | 0.9983 | 40 |
| base_57_91cf0ec9_t84_f10__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9787 | 1.0000 | 0.0000 | False | 0.9961 | 40 |
| base_57_91cf0ec9_t84_f10__ambiguous | sentence_packed | `code_only` | ambiguous | ambiguous | 0.9301 | 1.0000 | 0.0000 | False | 0.9747 | 40 |
| base_57_91cf0ec9_t84_f10__ambiguous | sentence_packed | `fuse_mean` | ambiguous | ambiguous | 0.9544 | 1.0000 | 0.0000 | False | 0.9854 | 40 |
| base_57_91cf0ec9_t84_f10__ambiguous | sentence_packed | `fuse_max` | ambiguous | ambiguous | 0.9787 | 1.0000 | 0.0000 | False | 0.9961 | 40 |
| base_57_91cf0ec9_t84_f10__ambiguous | sentence_packed | `fuse_min` | ambiguous | ambiguous | 0.9301 | 1.0000 | 0.0000 | False | 0.9747 | 40 |
| base_58_7b7fb621_t237_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9779 | 1.0000 | 0.0000 | False | 0.9997 | 23 |
| base_58_7b7fb621_t237_f0__correct | sentence_packed | `code_only` | grounded | correct | 0.9370 | 1.0000 | 0.0000 | False | 1.0000 | 23 |
| base_58_7b7fb621_t237_f0__correct | sentence_packed | `fuse_mean` | grounded | correct | 0.9575 | 1.0000 | 0.0000 | False | 0.9998 | 23 |
| base_58_7b7fb621_t237_f0__correct | sentence_packed | `fuse_max` | grounded | correct | 0.9779 | 1.0000 | 0.0000 | False | 1.0000 | 23 |
| base_58_7b7fb621_t237_f0__correct | sentence_packed | `fuse_min` | grounded | correct | 0.9370 | 1.0000 | 0.0000 | False | 0.9997 | 23 |
| base_58_7b7fb621_t237_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9690 | 1.0000 | 0.0000 | False | 0.9999 | 23 |
| base_58_7b7fb621_t237_f0__wrong | sentence_packed | `code_only` | ungrounded | wrong | 0.9165 | 1.0000 | 0.0000 | False | 0.9910 | 23 |
| base_58_7b7fb621_t237_f0__wrong | sentence_packed | `fuse_mean` | ungrounded | wrong | 0.9428 | 1.0000 | 0.0000 | False | 0.9955 | 23 |
| base_58_7b7fb621_t237_f0__wrong | sentence_packed | `fuse_max` | ungrounded | wrong | 0.9690 | 1.0000 | 0.0000 | False | 0.9999 | 23 |
| base_58_7b7fb621_t237_f0__wrong | sentence_packed | `fuse_min` | ungrounded | wrong | 0.9165 | 1.0000 | 0.0000 | False | 0.9910 | 23 |
| base_58_7b7fb621_t237_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9772 | 1.0000 | 0.0000 | False | 0.9996 | 23 |
| base_58_7b7fb621_t237_f0__ambiguous | sentence_packed | `code_only` | ambiguous | ambiguous | 0.9352 | 1.0000 | 0.0000 | False | 1.0000 | 23 |
| base_58_7b7fb621_t237_f0__ambiguous | sentence_packed | `fuse_mean` | ambiguous | ambiguous | 0.9562 | 1.0000 | 0.0000 | False | 0.9998 | 23 |
| base_58_7b7fb621_t237_f0__ambiguous | sentence_packed | `fuse_max` | ambiguous | ambiguous | 0.9772 | 1.0000 | 0.0000 | False | 1.0000 | 23 |
| base_58_7b7fb621_t237_f0__ambiguous | sentence_packed | `fuse_min` | ambiguous | ambiguous | 0.9352 | 1.0000 | 0.0000 | False | 0.9996 | 23 |
| base_59_ee0ad405_t137_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9761 | 1.0000 | 0.0000 | False | 0.9999 | 33 |
| base_59_ee0ad405_t137_f0__correct | sentence_packed | `code_only` | grounded | correct | 0.9238 | 1.0000 | 0.0000 | False | 0.9999 | 33 |
| base_59_ee0ad405_t137_f0__correct | sentence_packed | `fuse_mean` | grounded | correct | 0.9500 | 1.0000 | 0.0000 | False | 0.9999 | 33 |
| base_59_ee0ad405_t137_f0__correct | sentence_packed | `fuse_max` | grounded | correct | 0.9761 | 1.0000 | 0.0000 | False | 0.9999 | 33 |
| base_59_ee0ad405_t137_f0__correct | sentence_packed | `fuse_min` | grounded | correct | 0.9238 | 1.0000 | 0.0000 | False | 0.9999 | 33 |
| base_59_ee0ad405_t137_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9712 | 1.0000 | 0.0000 | False | 0.9999 | 33 |
| base_59_ee0ad405_t137_f0__wrong | sentence_packed | `code_only` | ungrounded | wrong | 0.9109 | 1.0000 | 0.0000 | False | 1.0000 | 33 |
| base_59_ee0ad405_t137_f0__wrong | sentence_packed | `fuse_mean` | ungrounded | wrong | 0.9411 | 1.0000 | 0.0000 | False | 1.0000 | 33 |
| base_59_ee0ad405_t137_f0__wrong | sentence_packed | `fuse_max` | ungrounded | wrong | 0.9712 | 1.0000 | 0.0000 | False | 1.0000 | 33 |
| base_59_ee0ad405_t137_f0__wrong | sentence_packed | `fuse_min` | ungrounded | wrong | 0.9109 | 1.0000 | 0.0000 | False | 0.9999 | 33 |
| base_59_ee0ad405_t137_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9759 | 1.0000 | 0.0000 | False | 1.0000 | 33 |
| base_59_ee0ad405_t137_f0__ambiguous | sentence_packed | `code_only` | ambiguous | ambiguous | 0.9232 | 1.0000 | 0.0000 | False | 0.9999 | 33 |
| base_59_ee0ad405_t137_f0__ambiguous | sentence_packed | `fuse_mean` | ambiguous | ambiguous | 0.9496 | 1.0000 | 0.0000 | False | 0.9999 | 33 |
| base_59_ee0ad405_t137_f0__ambiguous | sentence_packed | `fuse_max` | ambiguous | ambiguous | 0.9759 | 1.0000 | 0.0000 | False | 1.0000 | 33 |
| base_59_ee0ad405_t137_f0__ambiguous | sentence_packed | `fuse_min` | ambiguous | ambiguous | 0.9232 | 1.0000 | 0.0000 | False | 0.9999 | 33 |
| base_60_7b7fb621_t178_f1__correct | sentence_packed | `gte_only` | grounded | correct | 0.9808 | 1.0000 | 0.0000 | False | 0.9999 | 21 |
| base_60_7b7fb621_t178_f1__correct | sentence_packed | `code_only` | grounded | correct | 0.9298 | 1.0000 | 0.0000 | False | 0.9930 | 21 |
| base_60_7b7fb621_t178_f1__correct | sentence_packed | `fuse_mean` | grounded | correct | 0.9553 | 1.0000 | 0.0000 | False | 0.9965 | 21 |
| base_60_7b7fb621_t178_f1__correct | sentence_packed | `fuse_max` | grounded | correct | 0.9808 | 1.0000 | 0.0000 | False | 0.9999 | 21 |
| base_60_7b7fb621_t178_f1__correct | sentence_packed | `fuse_min` | grounded | correct | 0.9298 | 1.0000 | 0.0000 | False | 0.9930 | 21 |
| base_60_7b7fb621_t178_f1__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9786 | 1.0000 | 0.0000 | False | 0.9999 | 21 |
| base_60_7b7fb621_t178_f1__wrong | sentence_packed | `code_only` | ungrounded | wrong | 0.9281 | 1.0000 | 0.0000 | False | 1.0000 | 21 |
| base_60_7b7fb621_t178_f1__wrong | sentence_packed | `fuse_mean` | ungrounded | wrong | 0.9533 | 1.0000 | 0.0000 | False | 1.0000 | 21 |
| base_60_7b7fb621_t178_f1__wrong | sentence_packed | `fuse_max` | ungrounded | wrong | 0.9786 | 1.0000 | 0.0000 | False | 1.0000 | 21 |
| base_60_7b7fb621_t178_f1__wrong | sentence_packed | `fuse_min` | ungrounded | wrong | 0.9281 | 1.0000 | 0.0000 | False | 0.9999 | 21 |
| base_60_7b7fb621_t178_f1__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9807 | 1.0000 | 0.0000 | False | 0.9999 | 21 |
| base_60_7b7fb621_t178_f1__ambiguous | sentence_packed | `code_only` | ambiguous | ambiguous | 0.9294 | 1.0000 | 0.0000 | False | 0.9930 | 21 |
| base_60_7b7fb621_t178_f1__ambiguous | sentence_packed | `fuse_mean` | ambiguous | ambiguous | 0.9550 | 1.0000 | 0.0000 | False | 0.9965 | 21 |
| base_60_7b7fb621_t178_f1__ambiguous | sentence_packed | `fuse_max` | ambiguous | ambiguous | 0.9807 | 1.0000 | 0.0000 | False | 0.9999 | 21 |
| base_60_7b7fb621_t178_f1__ambiguous | sentence_packed | `fuse_min` | ambiguous | ambiguous | 0.9294 | 1.0000 | 0.0000 | False | 0.9930 | 21 |
| base_01_f0afeaf2_t952_f1__correct | colgrep | `gte_only` | grounded | correct | 0.9829 | 1.0000 | 0.0000 | False | 0.9979 | 126 |
| base_01_f0afeaf2_t952_f1__correct | colgrep | `code_only` | grounded | correct | 0.9536 | 1.0000 | 0.0000 | False | 0.9845 | 126 |
| base_01_f0afeaf2_t952_f1__correct | colgrep | `fuse_mean` | grounded | correct | 0.9682 | 1.0000 | 0.0000 | False | 0.9912 | 126 |
| base_01_f0afeaf2_t952_f1__correct | colgrep | `fuse_max` | grounded | correct | 0.9829 | 1.0000 | 0.0000 | False | 0.9979 | 126 |
| base_01_f0afeaf2_t952_f1__correct | colgrep | `fuse_min` | grounded | correct | 0.9536 | 1.0000 | 0.0000 | False | 0.9845 | 126 |
| base_01_f0afeaf2_t952_f1__wrong | colgrep | `gte_only` | ungrounded | wrong | 0.9807 | 1.0000 | 0.0000 | False | 0.9997 | 126 |
| base_01_f0afeaf2_t952_f1__wrong | colgrep | `code_only` | ungrounded | wrong | 0.9488 | 1.0000 | 0.0000 | False | 0.9915 | 126 |
| base_01_f0afeaf2_t952_f1__wrong | colgrep | `fuse_mean` | ungrounded | wrong | 0.9648 | 1.0000 | 0.0000 | False | 0.9956 | 126 |
| base_01_f0afeaf2_t952_f1__wrong | colgrep | `fuse_max` | ungrounded | wrong | 0.9807 | 1.0000 | 0.0000 | False | 0.9997 | 126 |
| base_01_f0afeaf2_t952_f1__wrong | colgrep | `fuse_min` | ungrounded | wrong | 0.9488 | 1.0000 | 0.0000 | False | 0.9915 | 126 |
| base_01_f0afeaf2_t952_f1__ambiguous | colgrep | `gte_only` | ambiguous | ambiguous | 0.9828 | 1.0000 | 0.0000 | False | 0.9979 | 126 |
| base_01_f0afeaf2_t952_f1__ambiguous | colgrep | `code_only` | ambiguous | ambiguous | 0.9542 | 1.0000 | 0.0000 | False | 0.9861 | 126 |
| base_01_f0afeaf2_t952_f1__ambiguous | colgrep | `fuse_mean` | ambiguous | ambiguous | 0.9685 | 1.0000 | 0.0000 | False | 0.9920 | 126 |
| base_01_f0afeaf2_t952_f1__ambiguous | colgrep | `fuse_max` | ambiguous | ambiguous | 0.9828 | 1.0000 | 0.0000 | False | 0.9979 | 126 |
| base_01_f0afeaf2_t952_f1__ambiguous | colgrep | `fuse_min` | ambiguous | ambiguous | 0.9542 | 1.0000 | 0.0000 | False | 0.9861 | 126 |
| base_02_61e1f5a1_t258_f0__correct | colgrep | `gte_only` | grounded | correct | 0.9728 | 1.0000 | 0.0000 | False | 0.9987 | 92 |
| base_02_61e1f5a1_t258_f0__correct | colgrep | `code_only` | grounded | correct | 0.9215 | 1.0000 | 0.0000 | False | 1.0000 | 92 |
| base_02_61e1f5a1_t258_f0__correct | colgrep | `fuse_mean` | grounded | correct | 0.9471 | 1.0000 | 0.0000 | False | 0.9994 | 92 |
| base_02_61e1f5a1_t258_f0__correct | colgrep | `fuse_max` | grounded | correct | 0.9728 | 1.0000 | 0.0000 | False | 1.0000 | 92 |
| base_02_61e1f5a1_t258_f0__correct | colgrep | `fuse_min` | grounded | correct | 0.9215 | 1.0000 | 0.0000 | False | 0.9987 | 92 |
| base_02_61e1f5a1_t258_f0__wrong | colgrep | `gte_only` | ungrounded | wrong | 0.9684 | 1.0000 | 0.0000 | False | 0.9982 | 92 |
| base_02_61e1f5a1_t258_f0__wrong | colgrep | `code_only` | ungrounded | wrong | 0.9090 | 1.0000 | 0.0000 | False | 1.0000 | 92 |
| base_02_61e1f5a1_t258_f0__wrong | colgrep | `fuse_mean` | ungrounded | wrong | 0.9387 | 1.0000 | 0.0000 | False | 0.9991 | 92 |
| base_02_61e1f5a1_t258_f0__wrong | colgrep | `fuse_max` | ungrounded | wrong | 0.9684 | 1.0000 | 0.0000 | False | 1.0000 | 92 |
| base_02_61e1f5a1_t258_f0__wrong | colgrep | `fuse_min` | ungrounded | wrong | 0.9090 | 1.0000 | 0.0000 | False | 0.9982 | 92 |
| base_02_61e1f5a1_t258_f0__ambiguous | colgrep | `gte_only` | ambiguous | ambiguous | 0.9725 | 1.0000 | 0.0000 | False | 0.9968 | 92 |
| base_02_61e1f5a1_t258_f0__ambiguous | colgrep | `code_only` | ambiguous | ambiguous | 0.9207 | 1.0000 | 0.0000 | False | 0.9847 | 92 |
| base_02_61e1f5a1_t258_f0__ambiguous | colgrep | `fuse_mean` | ambiguous | ambiguous | 0.9466 | 1.0000 | 0.0000 | False | 0.9907 | 92 |
| base_02_61e1f5a1_t258_f0__ambiguous | colgrep | `fuse_max` | ambiguous | ambiguous | 0.9725 | 1.0000 | 0.0000 | False | 0.9968 | 92 |
| base_02_61e1f5a1_t258_f0__ambiguous | colgrep | `fuse_min` | ambiguous | ambiguous | 0.9207 | 1.0000 | 0.0000 | False | 0.9847 | 92 |
| base_03_f0afeaf2_t2082_f0__correct | colgrep | `gte_only` | grounded | correct | 0.9741 | 1.0000 | 0.0000 | False | 0.9977 | 107 |
| base_03_f0afeaf2_t2082_f0__correct | colgrep | `code_only` | grounded | correct | 0.9404 | 1.0000 | 0.0000 | False | 0.9856 | 107 |
| base_03_f0afeaf2_t2082_f0__correct | colgrep | `fuse_mean` | grounded | correct | 0.9572 | 1.0000 | 0.0000 | False | 0.9916 | 107 |
| base_03_f0afeaf2_t2082_f0__correct | colgrep | `fuse_max` | grounded | correct | 0.9741 | 1.0000 | 0.0000 | False | 0.9977 | 107 |
| base_03_f0afeaf2_t2082_f0__correct | colgrep | `fuse_min` | grounded | correct | 0.9404 | 1.0000 | 0.0000 | False | 0.9856 | 107 |
| base_03_f0afeaf2_t2082_f0__wrong | colgrep | `gte_only` | ungrounded | wrong | 0.9675 | 1.0000 | 0.0000 | False | 0.9956 | 107 |
| base_03_f0afeaf2_t2082_f0__wrong | colgrep | `code_only` | ungrounded | wrong | 0.9324 | 1.0000 | 0.0000 | False | 0.9920 | 107 |
| base_03_f0afeaf2_t2082_f0__wrong | colgrep | `fuse_mean` | ungrounded | wrong | 0.9499 | 1.0000 | 0.0000 | False | 0.9938 | 107 |
| base_03_f0afeaf2_t2082_f0__wrong | colgrep | `fuse_max` | ungrounded | wrong | 0.9675 | 1.0000 | 0.0000 | False | 0.9956 | 107 |
| base_03_f0afeaf2_t2082_f0__wrong | colgrep | `fuse_min` | ungrounded | wrong | 0.9324 | 1.0000 | 0.0000 | False | 0.9920 | 107 |
| base_03_f0afeaf2_t2082_f0__ambiguous | colgrep | `gte_only` | ambiguous | ambiguous | 0.9742 | 1.0000 | 0.0000 | False | 0.9973 | 107 |
| base_03_f0afeaf2_t2082_f0__ambiguous | colgrep | `code_only` | ambiguous | ambiguous | 0.9404 | 1.0000 | 0.0000 | False | 1.0000 | 107 |
| base_03_f0afeaf2_t2082_f0__ambiguous | colgrep | `fuse_mean` | ambiguous | ambiguous | 0.9573 | 1.0000 | 0.0000 | False | 0.9986 | 107 |
| base_03_f0afeaf2_t2082_f0__ambiguous | colgrep | `fuse_max` | ambiguous | ambiguous | 0.9742 | 1.0000 | 0.0000 | False | 1.0000 | 107 |
| base_03_f0afeaf2_t2082_f0__ambiguous | colgrep | `fuse_min` | ambiguous | ambiguous | 0.9404 | 1.0000 | 0.0000 | False | 0.9973 | 107 |
| base_04_8a0f6f68_t521_f3__correct | colgrep | `gte_only` | grounded | correct | 0.9849 | 1.0000 | 0.0000 | False | 0.9974 | 91 |
| base_04_8a0f6f68_t521_f3__correct | colgrep | `code_only` | grounded | correct | 0.9598 | 1.0000 | 0.0000 | False | 0.9870 | 91 |
| base_04_8a0f6f68_t521_f3__correct | colgrep | `fuse_mean` | grounded | correct | 0.9723 | 1.0000 | 0.0000 | False | 0.9922 | 91 |
| base_04_8a0f6f68_t521_f3__correct | colgrep | `fuse_max` | grounded | correct | 0.9849 | 1.0000 | 0.0000 | False | 0.9974 | 91 |
| base_04_8a0f6f68_t521_f3__correct | colgrep | `fuse_min` | grounded | correct | 0.9598 | 1.0000 | 0.0000 | False | 0.9870 | 91 |
| base_04_8a0f6f68_t521_f3__wrong | colgrep | `gte_only` | ungrounded | wrong | 0.9809 | 1.0000 | 0.0000 | False | 0.9974 | 91 |
| base_04_8a0f6f68_t521_f3__wrong | colgrep | `code_only` | ungrounded | wrong | 0.9512 | 1.0000 | 0.0000 | False | 0.9911 | 91 |
| base_04_8a0f6f68_t521_f3__wrong | colgrep | `fuse_mean` | ungrounded | wrong | 0.9661 | 1.0000 | 0.0000 | False | 0.9942 | 91 |
| base_04_8a0f6f68_t521_f3__wrong | colgrep | `fuse_max` | ungrounded | wrong | 0.9809 | 1.0000 | 0.0000 | False | 0.9974 | 91 |
| base_04_8a0f6f68_t521_f3__wrong | colgrep | `fuse_min` | ungrounded | wrong | 0.9512 | 1.0000 | 0.0000 | False | 0.9911 | 91 |
| base_04_8a0f6f68_t521_f3__ambiguous | colgrep | `gte_only` | ambiguous | ambiguous | 0.9850 | 1.0000 | 0.0000 | False | 0.9974 | 91 |
| base_04_8a0f6f68_t521_f3__ambiguous | colgrep | `code_only` | ambiguous | ambiguous | 0.9579 | 1.0000 | 0.0000 | False | 0.9870 | 91 |
| base_04_8a0f6f68_t521_f3__ambiguous | colgrep | `fuse_mean` | ambiguous | ambiguous | 0.9714 | 1.0000 | 0.0000 | False | 0.9922 | 91 |
| base_04_8a0f6f68_t521_f3__ambiguous | colgrep | `fuse_max` | ambiguous | ambiguous | 0.9850 | 1.0000 | 0.0000 | False | 0.9974 | 91 |
| base_04_8a0f6f68_t521_f3__ambiguous | colgrep | `fuse_min` | ambiguous | ambiguous | 0.9579 | 1.0000 | 0.0000 | False | 0.9870 | 91 |
| base_05_126e38fc_t1159_f4__correct | colgrep | `gte_only` | grounded | correct | 0.9740 | 1.0000 | 0.0000 | False | 1.0000 | 146 |
| base_05_126e38fc_t1159_f4__correct | colgrep | `code_only` | grounded | correct | 0.9253 | 1.0000 | 0.0000 | False | 0.9998 | 146 |
| base_05_126e38fc_t1159_f4__correct | colgrep | `fuse_mean` | grounded | correct | 0.9497 | 1.0000 | 0.0000 | False | 0.9999 | 146 |
| base_05_126e38fc_t1159_f4__correct | colgrep | `fuse_max` | grounded | correct | 0.9740 | 1.0000 | 0.0000 | False | 1.0000 | 146 |
| base_05_126e38fc_t1159_f4__correct | colgrep | `fuse_min` | grounded | correct | 0.9253 | 1.0000 | 0.0000 | False | 0.9998 | 146 |
| base_05_126e38fc_t1159_f4__wrong | colgrep | `gte_only` | ungrounded | wrong | 0.9719 | 1.0000 | 0.0000 | False | 1.0000 | 146 |
| base_05_126e38fc_t1159_f4__wrong | colgrep | `code_only` | ungrounded | wrong | 0.9242 | 1.0000 | 0.0000 | False | 1.0000 | 146 |
| base_05_126e38fc_t1159_f4__wrong | colgrep | `fuse_mean` | ungrounded | wrong | 0.9480 | 1.0000 | 0.0000 | False | 1.0000 | 146 |
| base_05_126e38fc_t1159_f4__wrong | colgrep | `fuse_max` | ungrounded | wrong | 0.9719 | 1.0000 | 0.0000 | False | 1.0000 | 146 |
| base_05_126e38fc_t1159_f4__wrong | colgrep | `fuse_min` | ungrounded | wrong | 0.9242 | 1.0000 | 0.0000 | False | 1.0000 | 146 |
| base_05_126e38fc_t1159_f4__ambiguous | colgrep | `gte_only` | ambiguous | ambiguous | 0.9737 | 1.0000 | 0.0000 | False | 0.9902 | 146 |
| base_05_126e38fc_t1159_f4__ambiguous | colgrep | `code_only` | ambiguous | ambiguous | 0.9248 | 1.0000 | 0.0000 | False | 0.9370 | 146 |
| base_05_126e38fc_t1159_f4__ambiguous | colgrep | `fuse_mean` | ambiguous | ambiguous | 0.9493 | 1.0000 | 0.0000 | False | 0.9636 | 146 |
| base_05_126e38fc_t1159_f4__ambiguous | colgrep | `fuse_max` | ambiguous | ambiguous | 0.9737 | 1.0000 | 0.0000 | False | 0.9902 | 146 |
| base_05_126e38fc_t1159_f4__ambiguous | colgrep | `fuse_min` | ambiguous | ambiguous | 0.9248 | 1.0000 | 0.0000 | False | 0.9370 | 146 |
| base_06_f0afeaf2_t1771_f0__correct | colgrep | `gte_only` | grounded | correct | 0.9747 | 1.0000 | 0.0000 | False | 0.9928 | 154 |
| base_06_f0afeaf2_t1771_f0__correct | colgrep | `code_only` | grounded | correct | 0.9287 | 1.0000 | 0.0000 | False | 0.9824 | 154 |
| base_06_f0afeaf2_t1771_f0__correct | colgrep | `fuse_mean` | grounded | correct | 0.9517 | 1.0000 | 0.0000 | False | 0.9876 | 154 |
| base_06_f0afeaf2_t1771_f0__correct | colgrep | `fuse_max` | grounded | correct | 0.9747 | 1.0000 | 0.0000 | False | 0.9928 | 154 |
| base_06_f0afeaf2_t1771_f0__correct | colgrep | `fuse_min` | grounded | correct | 0.9287 | 1.0000 | 0.0000 | False | 0.9824 | 154 |
| base_06_f0afeaf2_t1771_f0__wrong | colgrep | `gte_only` | ungrounded | wrong | 0.9683 | 1.0000 | 0.0000 | False | 0.9695 | 154 |
| base_06_f0afeaf2_t1771_f0__wrong | colgrep | `code_only` | ungrounded | wrong | 0.9178 | 1.0000 | 0.0000 | False | 0.9789 | 154 |
| base_06_f0afeaf2_t1771_f0__wrong | colgrep | `fuse_mean` | ungrounded | wrong | 0.9431 | 1.0000 | 0.0000 | False | 0.9742 | 154 |
| base_06_f0afeaf2_t1771_f0__wrong | colgrep | `fuse_max` | ungrounded | wrong | 0.9683 | 1.0000 | 0.0000 | False | 0.9789 | 154 |
| base_06_f0afeaf2_t1771_f0__wrong | colgrep | `fuse_min` | ungrounded | wrong | 0.9178 | 1.0000 | 0.0000 | False | 0.9695 | 154 |
| base_06_f0afeaf2_t1771_f0__ambiguous | colgrep | `gte_only` | ambiguous | ambiguous | 0.9692 | 1.0000 | 0.0000 | False | 0.9626 | 154 |
| base_06_f0afeaf2_t1771_f0__ambiguous | colgrep | `code_only` | ambiguous | ambiguous | 0.9191 | 1.0000 | 0.0000 | False | 0.9789 | 154 |
| base_06_f0afeaf2_t1771_f0__ambiguous | colgrep | `fuse_mean` | ambiguous | ambiguous | 0.9441 | 1.0000 | 0.0000 | False | 0.9707 | 154 |
| base_06_f0afeaf2_t1771_f0__ambiguous | colgrep | `fuse_max` | ambiguous | ambiguous | 0.9692 | 1.0000 | 0.0000 | False | 0.9789 | 154 |
| base_06_f0afeaf2_t1771_f0__ambiguous | colgrep | `fuse_min` | ambiguous | ambiguous | 0.9191 | 1.0000 | 0.0000 | False | 0.9626 | 154 |
| base_07_7b7fb621_t328_f1__correct | colgrep | `gte_only` | grounded | correct | 0.9776 | 1.0000 | 0.0000 | False | 0.9999 | 45 |
| base_07_7b7fb621_t328_f1__correct | colgrep | `code_only` | grounded | correct | 0.9378 | 1.0000 | 0.0000 | False | 1.0000 | 45 |
| base_07_7b7fb621_t328_f1__correct | colgrep | `fuse_mean` | grounded | correct | 0.9577 | 1.0000 | 0.0000 | False | 0.9999 | 45 |
| base_07_7b7fb621_t328_f1__correct | colgrep | `fuse_max` | grounded | correct | 0.9776 | 1.0000 | 0.0000 | False | 1.0000 | 45 |
| base_07_7b7fb621_t328_f1__correct | colgrep | `fuse_min` | grounded | correct | 0.9378 | 1.0000 | 0.0000 | False | 0.9999 | 45 |
| base_07_7b7fb621_t328_f1__wrong | colgrep | `gte_only` | ungrounded | wrong | 0.9703 | 1.0000 | 0.0000 | False | 0.9998 | 45 |
| base_07_7b7fb621_t328_f1__wrong | colgrep | `code_only` | ungrounded | wrong | 0.9285 | 1.0000 | 0.0000 | False | 1.0000 | 45 |
| base_07_7b7fb621_t328_f1__wrong | colgrep | `fuse_mean` | ungrounded | wrong | 0.9494 | 1.0000 | 0.0000 | False | 0.9999 | 45 |
| base_07_7b7fb621_t328_f1__wrong | colgrep | `fuse_max` | ungrounded | wrong | 0.9703 | 1.0000 | 0.0000 | False | 1.0000 | 45 |
| base_07_7b7fb621_t328_f1__wrong | colgrep | `fuse_min` | ungrounded | wrong | 0.9285 | 1.0000 | 0.0000 | False | 0.9998 | 45 |
| base_07_7b7fb621_t328_f1__ambiguous | colgrep | `gte_only` | ambiguous | ambiguous | 0.9777 | 1.0000 | 0.0000 | False | 1.0000 | 45 |
| base_07_7b7fb621_t328_f1__ambiguous | colgrep | `code_only` | ambiguous | ambiguous | 0.9365 | 1.0000 | 0.0000 | False | 1.0000 | 45 |
| base_07_7b7fb621_t328_f1__ambiguous | colgrep | `fuse_mean` | ambiguous | ambiguous | 0.9571 | 1.0000 | 0.0000 | False | 1.0000 | 45 |
| base_07_7b7fb621_t328_f1__ambiguous | colgrep | `fuse_max` | ambiguous | ambiguous | 0.9777 | 1.0000 | 0.0000 | False | 1.0000 | 45 |
| base_07_7b7fb621_t328_f1__ambiguous | colgrep | `fuse_min` | ambiguous | ambiguous | 0.9365 | 1.0000 | 0.0000 | False | 1.0000 | 45 |
| base_08_126e38fc_t1785_f0__correct | colgrep | `gte_only` | grounded | correct | 0.9799 | 1.0000 | 0.0000 | False | 0.9984 | 216 |
| base_08_126e38fc_t1785_f0__correct | colgrep | `code_only` | grounded | correct | 0.9466 | 1.0000 | 0.0000 | False | 0.9888 | 216 |
| base_08_126e38fc_t1785_f0__correct | colgrep | `fuse_mean` | grounded | correct | 0.9633 | 1.0000 | 0.0000 | False | 0.9936 | 216 |
| base_08_126e38fc_t1785_f0__correct | colgrep | `fuse_max` | grounded | correct | 0.9799 | 1.0000 | 0.0000 | False | 0.9984 | 216 |
| base_08_126e38fc_t1785_f0__correct | colgrep | `fuse_min` | grounded | correct | 0.9466 | 1.0000 | 0.0000 | False | 0.9888 | 216 |
| base_08_126e38fc_t1785_f0__wrong | colgrep | `gte_only` | ungrounded | wrong | 0.9790 | 1.0000 | 0.0000 | False | 0.9982 | 216 |
| base_08_126e38fc_t1785_f0__wrong | colgrep | `code_only` | ungrounded | wrong | 0.9486 | 1.0000 | 0.0000 | False | 0.9899 | 216 |
| base_08_126e38fc_t1785_f0__wrong | colgrep | `fuse_mean` | ungrounded | wrong | 0.9638 | 1.0000 | 0.0000 | False | 0.9941 | 216 |
| base_08_126e38fc_t1785_f0__wrong | colgrep | `fuse_max` | ungrounded | wrong | 0.9790 | 1.0000 | 0.0000 | False | 0.9982 | 216 |
| base_08_126e38fc_t1785_f0__wrong | colgrep | `fuse_min` | ungrounded | wrong | 0.9486 | 1.0000 | 0.0000 | False | 0.9899 | 216 |
| base_08_126e38fc_t1785_f0__ambiguous | colgrep | `gte_only` | ambiguous | ambiguous | 0.9798 | 1.0000 | 0.0000 | False | 0.9984 | 216 |
| base_08_126e38fc_t1785_f0__ambiguous | colgrep | `code_only` | ambiguous | ambiguous | 0.9463 | 1.0000 | 0.0000 | False | 0.9896 | 216 |
| base_08_126e38fc_t1785_f0__ambiguous | colgrep | `fuse_mean` | ambiguous | ambiguous | 0.9631 | 1.0000 | 0.0000 | False | 0.9940 | 216 |
| base_08_126e38fc_t1785_f0__ambiguous | colgrep | `fuse_max` | ambiguous | ambiguous | 0.9798 | 1.0000 | 0.0000 | False | 0.9984 | 216 |
| base_08_126e38fc_t1785_f0__ambiguous | colgrep | `fuse_min` | ambiguous | ambiguous | 0.9463 | 1.0000 | 0.0000 | False | 0.9896 | 216 |
| base_09_0b1422bb_t571_f0__correct | colgrep | `gte_only` | grounded | correct | 0.9567 | 1.0000 | 0.0000 | False | 0.9978 | 32 |
| base_09_0b1422bb_t571_f0__correct | colgrep | `code_only` | grounded | correct | 0.8895 | 1.0000 | 0.0000 | False | 0.9846 | 32 |
| base_09_0b1422bb_t571_f0__correct | colgrep | `fuse_mean` | grounded | correct | 0.9231 | 1.0000 | 0.0000 | False | 0.9912 | 32 |
| base_09_0b1422bb_t571_f0__correct | colgrep | `fuse_max` | grounded | correct | 0.9567 | 1.0000 | 0.0000 | False | 0.9978 | 32 |
| base_09_0b1422bb_t571_f0__correct | colgrep | `fuse_min` | grounded | correct | 0.8895 | 1.0000 | 0.0000 | False | 0.9846 | 32 |
| base_09_0b1422bb_t571_f0__wrong | colgrep | `gte_only` | ungrounded | wrong | 0.9546 | 1.0000 | 0.0000 | False | 0.9979 | 32 |
| base_09_0b1422bb_t571_f0__wrong | colgrep | `code_only` | ungrounded | wrong | 0.8903 | 1.0000 | 0.0000 | False | 0.9892 | 32 |
| base_09_0b1422bb_t571_f0__wrong | colgrep | `fuse_mean` | ungrounded | wrong | 0.9225 | 1.0000 | 0.0000 | False | 0.9936 | 32 |
| base_09_0b1422bb_t571_f0__wrong | colgrep | `fuse_max` | ungrounded | wrong | 0.9546 | 1.0000 | 0.0000 | False | 0.9979 | 32 |
| base_09_0b1422bb_t571_f0__wrong | colgrep | `fuse_min` | ungrounded | wrong | 0.8903 | 1.0000 | 0.0000 | False | 0.9892 | 32 |
| base_09_0b1422bb_t571_f0__ambiguous | colgrep | `gte_only` | ambiguous | ambiguous | 0.9566 | 1.0000 | 0.0000 | False | 0.9978 | 32 |
| base_09_0b1422bb_t571_f0__ambiguous | colgrep | `code_only` | ambiguous | ambiguous | 0.8903 | 1.0000 | 0.0000 | False | 0.9846 | 32 |
| base_09_0b1422bb_t571_f0__ambiguous | colgrep | `fuse_mean` | ambiguous | ambiguous | 0.9235 | 1.0000 | 0.0000 | False | 0.9912 | 32 |
| base_09_0b1422bb_t571_f0__ambiguous | colgrep | `fuse_max` | ambiguous | ambiguous | 0.9566 | 1.0000 | 0.0000 | False | 0.9978 | 32 |
| base_09_0b1422bb_t571_f0__ambiguous | colgrep | `fuse_min` | ambiguous | ambiguous | 0.8903 | 1.0000 | 0.0000 | False | 0.9846 | 32 |
| base_10_d7fd7127_t1579_f0__correct | colgrep | `gte_only` | grounded | correct | 0.9799 | 1.0000 | 0.0000 | False | 0.9948 | 153 |
| base_10_d7fd7127_t1579_f0__correct | colgrep | `code_only` | grounded | correct | 0.9325 | 1.0000 | 0.0000 | False | 0.9727 | 153 |
| base_10_d7fd7127_t1579_f0__correct | colgrep | `fuse_mean` | grounded | correct | 0.9562 | 1.0000 | 0.0000 | False | 0.9837 | 153 |
| base_10_d7fd7127_t1579_f0__correct | colgrep | `fuse_max` | grounded | correct | 0.9799 | 1.0000 | 0.0000 | False | 0.9948 | 153 |
| base_10_d7fd7127_t1579_f0__correct | colgrep | `fuse_min` | grounded | correct | 0.9325 | 1.0000 | 0.0000 | False | 0.9727 | 153 |
| base_10_d7fd7127_t1579_f0__wrong | colgrep | `gte_only` | ungrounded | wrong | 0.9669 | 1.0000 | 0.0000 | False | 0.9942 | 153 |
| base_10_d7fd7127_t1579_f0__wrong | colgrep | `code_only` | ungrounded | wrong | 0.9223 | 1.0000 | 0.0000 | False | 0.9783 | 153 |
| base_10_d7fd7127_t1579_f0__wrong | colgrep | `fuse_mean` | ungrounded | wrong | 0.9446 | 1.0000 | 0.0000 | False | 0.9862 | 153 |
| base_10_d7fd7127_t1579_f0__wrong | colgrep | `fuse_max` | ungrounded | wrong | 0.9669 | 1.0000 | 0.0000 | False | 0.9942 | 153 |
| base_10_d7fd7127_t1579_f0__wrong | colgrep | `fuse_min` | ungrounded | wrong | 0.9223 | 1.0000 | 0.0000 | False | 0.9783 | 153 |
| base_10_d7fd7127_t1579_f0__ambiguous | colgrep | `gte_only` | ambiguous | ambiguous | 0.9797 | 1.0000 | 0.0000 | False | 0.9948 | 153 |
| base_10_d7fd7127_t1579_f0__ambiguous | colgrep | `code_only` | ambiguous | ambiguous | 0.9328 | 1.0000 | 0.0000 | False | 0.9719 | 153 |
| base_10_d7fd7127_t1579_f0__ambiguous | colgrep | `fuse_mean` | ambiguous | ambiguous | 0.9563 | 1.0000 | 0.0000 | False | 0.9833 | 153 |
| base_10_d7fd7127_t1579_f0__ambiguous | colgrep | `fuse_max` | ambiguous | ambiguous | 0.9797 | 1.0000 | 0.0000 | False | 0.9948 | 153 |
| base_10_d7fd7127_t1579_f0__ambiguous | colgrep | `fuse_min` | ambiguous | ambiguous | 0.9328 | 1.0000 | 0.0000 | False | 0.9719 | 153 |
| base_11_f0afeaf2_t1418_f1__correct | colgrep | `gte_only` | grounded | correct | 0.9846 | 1.0000 | 0.0000 | False | 0.9960 | 152 |
| base_11_f0afeaf2_t1418_f1__correct | colgrep | `code_only` | grounded | correct | 0.9500 | 1.0000 | 0.0000 | False | 0.9868 | 152 |
| base_11_f0afeaf2_t1418_f1__correct | colgrep | `fuse_mean` | grounded | correct | 0.9673 | 1.0000 | 0.0000 | False | 0.9914 | 152 |
| base_11_f0afeaf2_t1418_f1__correct | colgrep | `fuse_max` | grounded | correct | 0.9846 | 1.0000 | 0.0000 | False | 0.9960 | 152 |
| base_11_f0afeaf2_t1418_f1__correct | colgrep | `fuse_min` | grounded | correct | 0.9500 | 1.0000 | 0.0000 | False | 0.9868 | 152 |
| base_11_f0afeaf2_t1418_f1__wrong | colgrep | `gte_only` | ungrounded | wrong | 0.9705 | 1.0000 | 0.0000 | False | 0.9962 | 152 |
| base_11_f0afeaf2_t1418_f1__wrong | colgrep | `code_only` | ungrounded | wrong | 0.9240 | 1.0000 | 0.0000 | False | 0.9806 | 152 |
| base_11_f0afeaf2_t1418_f1__wrong | colgrep | `fuse_mean` | ungrounded | wrong | 0.9472 | 1.0000 | 0.0000 | False | 0.9884 | 152 |
| base_11_f0afeaf2_t1418_f1__wrong | colgrep | `fuse_max` | ungrounded | wrong | 0.9705 | 1.0000 | 0.0000 | False | 0.9962 | 152 |
| base_11_f0afeaf2_t1418_f1__wrong | colgrep | `fuse_min` | ungrounded | wrong | 0.9240 | 1.0000 | 0.0000 | False | 0.9806 | 152 |
| base_11_f0afeaf2_t1418_f1__ambiguous | colgrep | `gte_only` | ambiguous | ambiguous | 0.9843 | 1.0000 | 0.0000 | False | 0.9961 | 152 |
| base_11_f0afeaf2_t1418_f1__ambiguous | colgrep | `code_only` | ambiguous | ambiguous | 0.9477 | 1.0000 | 0.0000 | False | 0.9870 | 152 |
| base_11_f0afeaf2_t1418_f1__ambiguous | colgrep | `fuse_mean` | ambiguous | ambiguous | 0.9660 | 1.0000 | 0.0000 | False | 0.9916 | 152 |
| base_11_f0afeaf2_t1418_f1__ambiguous | colgrep | `fuse_max` | ambiguous | ambiguous | 0.9843 | 1.0000 | 0.0000 | False | 0.9961 | 152 |
| base_11_f0afeaf2_t1418_f1__ambiguous | colgrep | `fuse_min` | ambiguous | ambiguous | 0.9477 | 1.0000 | 0.0000 | False | 0.9870 | 152 |
| base_12_f0afeaf2_t955_f2__correct | colgrep | `gte_only` | grounded | correct | 0.9760 | 1.0000 | 0.0000 | False | 0.9928 | 126 |
| base_12_f0afeaf2_t955_f2__correct | colgrep | `code_only` | grounded | correct | 0.9108 | 1.0000 | 0.0000 | False | 0.9894 | 126 |
| base_12_f0afeaf2_t955_f2__correct | colgrep | `fuse_mean` | grounded | correct | 0.9434 | 1.0000 | 0.0000 | False | 0.9911 | 126 |
| base_12_f0afeaf2_t955_f2__correct | colgrep | `fuse_max` | grounded | correct | 0.9760 | 1.0000 | 0.0000 | False | 0.9928 | 126 |
| base_12_f0afeaf2_t955_f2__correct | colgrep | `fuse_min` | grounded | correct | 0.9108 | 1.0000 | 0.0000 | False | 0.9894 | 126 |
| base_12_f0afeaf2_t955_f2__wrong | colgrep | `gte_only` | ungrounded | wrong | 0.9669 | 1.0000 | 0.0000 | False | 1.0000 | 126 |
| base_12_f0afeaf2_t955_f2__wrong | colgrep | `code_only` | ungrounded | wrong | 0.9131 | 1.0000 | 0.0000 | False | 1.0000 | 126 |
| base_12_f0afeaf2_t955_f2__wrong | colgrep | `fuse_mean` | ungrounded | wrong | 0.9400 | 1.0000 | 0.0000 | False | 1.0000 | 126 |
| base_12_f0afeaf2_t955_f2__wrong | colgrep | `fuse_max` | ungrounded | wrong | 0.9669 | 1.0000 | 0.0000 | False | 1.0000 | 126 |
| base_12_f0afeaf2_t955_f2__wrong | colgrep | `fuse_min` | ungrounded | wrong | 0.9131 | 1.0000 | 0.0000 | False | 1.0000 | 126 |
| base_12_f0afeaf2_t955_f2__ambiguous | colgrep | `gte_only` | ambiguous | ambiguous | 0.9775 | 1.0000 | 0.0000 | False | 0.9945 | 126 |
| base_12_f0afeaf2_t955_f2__ambiguous | colgrep | `code_only` | ambiguous | ambiguous | 0.9050 | 1.0000 | 0.0000 | False | 0.9779 | 126 |
| base_12_f0afeaf2_t955_f2__ambiguous | colgrep | `fuse_mean` | ambiguous | ambiguous | 0.9413 | 1.0000 | 0.0000 | False | 0.9862 | 126 |
| base_12_f0afeaf2_t955_f2__ambiguous | colgrep | `fuse_max` | ambiguous | ambiguous | 0.9775 | 1.0000 | 0.0000 | False | 0.9945 | 126 |
| base_12_f0afeaf2_t955_f2__ambiguous | colgrep | `fuse_min` | ambiguous | ambiguous | 0.9050 | 1.0000 | 0.0000 | False | 0.9779 | 126 |
| base_13_d7fd7127_t1553_f0__correct | colgrep | `gte_only` | grounded | correct | 0.9837 | 1.0000 | 0.0000 | False | 0.9999 | 153 |
| base_13_d7fd7127_t1553_f0__correct | colgrep | `code_only` | grounded | correct | 0.9553 | 1.0000 | 0.0000 | False | 0.9920 | 153 |
| base_13_d7fd7127_t1553_f0__correct | colgrep | `fuse_mean` | grounded | correct | 0.9695 | 1.0000 | 0.0000 | False | 0.9959 | 153 |
| base_13_d7fd7127_t1553_f0__correct | colgrep | `fuse_max` | grounded | correct | 0.9837 | 1.0000 | 0.0000 | False | 0.9999 | 153 |
| base_13_d7fd7127_t1553_f0__correct | colgrep | `fuse_min` | grounded | correct | 0.9553 | 1.0000 | 0.0000 | False | 0.9920 | 153 |
| base_13_d7fd7127_t1553_f0__wrong | colgrep | `gte_only` | ungrounded | wrong | 0.9676 | 1.0000 | 0.0000 | False | 1.0000 | 153 |
| base_13_d7fd7127_t1553_f0__wrong | colgrep | `code_only` | ungrounded | wrong | 0.9345 | 1.0000 | 0.0000 | False | 0.9830 | 153 |
| base_13_d7fd7127_t1553_f0__wrong | colgrep | `fuse_mean` | ungrounded | wrong | 0.9511 | 1.0000 | 0.0000 | False | 0.9915 | 153 |
| base_13_d7fd7127_t1553_f0__wrong | colgrep | `fuse_max` | ungrounded | wrong | 0.9676 | 1.0000 | 0.0000 | False | 1.0000 | 153 |
| base_13_d7fd7127_t1553_f0__wrong | colgrep | `fuse_min` | ungrounded | wrong | 0.9345 | 1.0000 | 0.0000 | False | 0.9830 | 153 |
| base_13_d7fd7127_t1553_f0__ambiguous | colgrep | `gte_only` | ambiguous | ambiguous | 0.9837 | 1.0000 | 0.0000 | False | 1.0000 | 153 |
| base_13_d7fd7127_t1553_f0__ambiguous | colgrep | `code_only` | ambiguous | ambiguous | 0.9550 | 1.0000 | 0.0000 | False | 0.9920 | 153 |
| base_13_d7fd7127_t1553_f0__ambiguous | colgrep | `fuse_mean` | ambiguous | ambiguous | 0.9694 | 1.0000 | 0.0000 | False | 0.9960 | 153 |
| base_13_d7fd7127_t1553_f0__ambiguous | colgrep | `fuse_max` | ambiguous | ambiguous | 0.9837 | 1.0000 | 0.0000 | False | 1.0000 | 153 |
| base_13_d7fd7127_t1553_f0__ambiguous | colgrep | `fuse_min` | ambiguous | ambiguous | 0.9550 | 1.0000 | 0.0000 | False | 0.9920 | 153 |
| base_14_f0afeaf2_t2825_f0__correct | colgrep | `gte_only` | grounded | correct | 0.9809 | 1.0000 | 0.0000 | False | 1.0000 | 107 |
| base_14_f0afeaf2_t2825_f0__correct | colgrep | `code_only` | grounded | correct | 0.9452 | 1.0000 | 0.0000 | False | 1.0000 | 107 |
| base_14_f0afeaf2_t2825_f0__correct | colgrep | `fuse_mean` | grounded | correct | 0.9630 | 1.0000 | 0.0000 | False | 1.0000 | 107 |
| base_14_f0afeaf2_t2825_f0__correct | colgrep | `fuse_max` | grounded | correct | 0.9809 | 1.0000 | 0.0000 | False | 1.0000 | 107 |
| base_14_f0afeaf2_t2825_f0__correct | colgrep | `fuse_min` | grounded | correct | 0.9452 | 1.0000 | 0.0000 | False | 1.0000 | 107 |
| base_14_f0afeaf2_t2825_f0__wrong | colgrep | `gte_only` | ungrounded | wrong | 0.9716 | 1.0000 | 0.0000 | False | 1.0000 | 107 |
| base_14_f0afeaf2_t2825_f0__wrong | colgrep | `code_only` | ungrounded | wrong | 0.9358 | 1.0000 | 0.0000 | False | 0.9999 | 107 |
| base_14_f0afeaf2_t2825_f0__wrong | colgrep | `fuse_mean` | ungrounded | wrong | 0.9537 | 1.0000 | 0.0000 | False | 1.0000 | 107 |
| base_14_f0afeaf2_t2825_f0__wrong | colgrep | `fuse_max` | ungrounded | wrong | 0.9716 | 1.0000 | 0.0000 | False | 1.0000 | 107 |
| base_14_f0afeaf2_t2825_f0__wrong | colgrep | `fuse_min` | ungrounded | wrong | 0.9358 | 1.0000 | 0.0000 | False | 0.9999 | 107 |
| base_14_f0afeaf2_t2825_f0__ambiguous | colgrep | `gte_only` | ambiguous | ambiguous | 0.9806 | 1.0000 | 0.0000 | False | 0.9988 | 107 |
| base_14_f0afeaf2_t2825_f0__ambiguous | colgrep | `code_only` | ambiguous | ambiguous | 0.9454 | 1.0000 | 0.0000 | False | 0.9999 | 107 |
| base_14_f0afeaf2_t2825_f0__ambiguous | colgrep | `fuse_mean` | ambiguous | ambiguous | 0.9630 | 1.0000 | 0.0000 | False | 0.9994 | 107 |
| base_14_f0afeaf2_t2825_f0__ambiguous | colgrep | `fuse_max` | ambiguous | ambiguous | 0.9806 | 1.0000 | 0.0000 | False | 0.9999 | 107 |
| base_14_f0afeaf2_t2825_f0__ambiguous | colgrep | `fuse_min` | ambiguous | ambiguous | 0.9454 | 1.0000 | 0.0000 | False | 0.9988 | 107 |
| base_15_91cf0ec9_t419_f2__correct | colgrep | `gte_only` | grounded | correct | 0.9829 | 1.0000 | 0.0000 | False | 0.9992 | 94 |
| base_15_91cf0ec9_t419_f2__correct | colgrep | `code_only` | grounded | correct | 0.9268 | 1.0000 | 0.0000 | False | 0.9881 | 94 |
| base_15_91cf0ec9_t419_f2__correct | colgrep | `fuse_mean` | grounded | correct | 0.9549 | 1.0000 | 0.0000 | False | 0.9936 | 94 |
| base_15_91cf0ec9_t419_f2__correct | colgrep | `fuse_max` | grounded | correct | 0.9829 | 1.0000 | 0.0000 | False | 0.9992 | 94 |
| base_15_91cf0ec9_t419_f2__correct | colgrep | `fuse_min` | grounded | correct | 0.9268 | 1.0000 | 0.0000 | False | 0.9881 | 94 |
| base_15_91cf0ec9_t419_f2__wrong | colgrep | `gte_only` | ungrounded | wrong | 0.9781 | 1.0000 | 0.0000 | False | 0.9999 | 94 |
| base_15_91cf0ec9_t419_f2__wrong | colgrep | `code_only` | ungrounded | wrong | 0.9259 | 1.0000 | 0.0000 | False | 1.0000 | 94 |
| base_15_91cf0ec9_t419_f2__wrong | colgrep | `fuse_mean` | ungrounded | wrong | 0.9520 | 1.0000 | 0.0000 | False | 0.9999 | 94 |
| base_15_91cf0ec9_t419_f2__wrong | colgrep | `fuse_max` | ungrounded | wrong | 0.9781 | 1.0000 | 0.0000 | False | 1.0000 | 94 |
| base_15_91cf0ec9_t419_f2__wrong | colgrep | `fuse_min` | ungrounded | wrong | 0.9259 | 1.0000 | 0.0000 | False | 0.9999 | 94 |
| base_15_91cf0ec9_t419_f2__ambiguous | colgrep | `gte_only` | ambiguous | ambiguous | 0.9824 | 1.0000 | 0.0000 | False | 0.9992 | 94 |
| base_15_91cf0ec9_t419_f2__ambiguous | colgrep | `code_only` | ambiguous | ambiguous | 0.9276 | 1.0000 | 0.0000 | False | 0.9881 | 94 |
| base_15_91cf0ec9_t419_f2__ambiguous | colgrep | `fuse_mean` | ambiguous | ambiguous | 0.9550 | 1.0000 | 0.0000 | False | 0.9936 | 94 |
| base_15_91cf0ec9_t419_f2__ambiguous | colgrep | `fuse_max` | ambiguous | ambiguous | 0.9824 | 1.0000 | 0.0000 | False | 0.9992 | 94 |
| base_15_91cf0ec9_t419_f2__ambiguous | colgrep | `fuse_min` | ambiguous | ambiguous | 0.9276 | 1.0000 | 0.0000 | False | 0.9881 | 94 |
| base_16_8a0f6f68_t1002_f0__correct | colgrep | `gte_only` | grounded | correct | 0.9802 | 1.0000 | 0.0000 | False | 0.9936 | 117 |
| base_16_8a0f6f68_t1002_f0__correct | colgrep | `code_only` | grounded | correct | 0.9473 | 1.0000 | 0.0000 | False | 0.9488 | 117 |
| base_16_8a0f6f68_t1002_f0__correct | colgrep | `fuse_mean` | grounded | correct | 0.9638 | 1.0000 | 0.0000 | False | 0.9712 | 117 |
| base_16_8a0f6f68_t1002_f0__correct | colgrep | `fuse_max` | grounded | correct | 0.9802 | 1.0000 | 0.0000 | False | 0.9936 | 117 |
| base_16_8a0f6f68_t1002_f0__correct | colgrep | `fuse_min` | grounded | correct | 0.9473 | 1.0000 | 0.0000 | False | 0.9488 | 117 |
| base_16_8a0f6f68_t1002_f0__wrong | colgrep | `gte_only` | ungrounded | wrong | 0.9736 | 1.0000 | 0.0000 | False | 1.0000 | 117 |
| base_16_8a0f6f68_t1002_f0__wrong | colgrep | `code_only` | ungrounded | wrong | 0.9375 | 1.0000 | 0.0000 | False | 1.0000 | 117 |
| base_16_8a0f6f68_t1002_f0__wrong | colgrep | `fuse_mean` | ungrounded | wrong | 0.9555 | 1.0000 | 0.0000 | False | 1.0000 | 117 |
| base_16_8a0f6f68_t1002_f0__wrong | colgrep | `fuse_max` | ungrounded | wrong | 0.9736 | 1.0000 | 0.0000 | False | 1.0000 | 117 |
| base_16_8a0f6f68_t1002_f0__wrong | colgrep | `fuse_min` | ungrounded | wrong | 0.9375 | 1.0000 | 0.0000 | False | 1.0000 | 117 |
| base_16_8a0f6f68_t1002_f0__ambiguous | colgrep | `gte_only` | ambiguous | ambiguous | 0.9799 | 1.0000 | 0.0000 | False | 0.9964 | 117 |
| base_16_8a0f6f68_t1002_f0__ambiguous | colgrep | `code_only` | ambiguous | ambiguous | 0.9471 | 1.0000 | 0.0000 | False | 0.9854 | 117 |
| base_16_8a0f6f68_t1002_f0__ambiguous | colgrep | `fuse_mean` | ambiguous | ambiguous | 0.9635 | 1.0000 | 0.0000 | False | 0.9909 | 117 |
| base_16_8a0f6f68_t1002_f0__ambiguous | colgrep | `fuse_max` | ambiguous | ambiguous | 0.9799 | 1.0000 | 0.0000 | False | 0.9964 | 117 |
| base_16_8a0f6f68_t1002_f0__ambiguous | colgrep | `fuse_min` | ambiguous | ambiguous | 0.9471 | 1.0000 | 0.0000 | False | 0.9854 | 117 |
| base_17_126e38fc_t712_f0__correct | colgrep | `gte_only` | grounded | correct | 0.9799 | 1.0000 | 0.0000 | False | 0.9969 | 75 |
| base_17_126e38fc_t712_f0__correct | colgrep | `code_only` | grounded | correct | 0.9439 | 1.0000 | 0.0000 | False | 0.9852 | 75 |
| base_17_126e38fc_t712_f0__correct | colgrep | `fuse_mean` | grounded | correct | 0.9619 | 1.0000 | 0.0000 | False | 0.9911 | 75 |
| base_17_126e38fc_t712_f0__correct | colgrep | `fuse_max` | grounded | correct | 0.9799 | 1.0000 | 0.0000 | False | 0.9969 | 75 |
| base_17_126e38fc_t712_f0__correct | colgrep | `fuse_min` | grounded | correct | 0.9439 | 1.0000 | 0.0000 | False | 0.9852 | 75 |
| base_17_126e38fc_t712_f0__wrong | colgrep | `gte_only` | ungrounded | wrong | 0.9762 | 1.0000 | 0.0000 | False | 0.9958 | 75 |
| base_17_126e38fc_t712_f0__wrong | colgrep | `code_only` | ungrounded | wrong | 0.9393 | 1.0000 | 0.0000 | False | 0.9837 | 75 |
| base_17_126e38fc_t712_f0__wrong | colgrep | `fuse_mean` | ungrounded | wrong | 0.9577 | 1.0000 | 0.0000 | False | 0.9897 | 75 |
| base_17_126e38fc_t712_f0__wrong | colgrep | `fuse_max` | ungrounded | wrong | 0.9762 | 1.0000 | 0.0000 | False | 0.9958 | 75 |
| base_17_126e38fc_t712_f0__wrong | colgrep | `fuse_min` | ungrounded | wrong | 0.9393 | 1.0000 | 0.0000 | False | 0.9837 | 75 |
| base_17_126e38fc_t712_f0__ambiguous | colgrep | `gte_only` | ambiguous | ambiguous | 0.9799 | 1.0000 | 0.0000 | False | 0.9969 | 75 |
| base_17_126e38fc_t712_f0__ambiguous | colgrep | `code_only` | ambiguous | ambiguous | 0.9436 | 1.0000 | 0.0000 | False | 0.9848 | 75 |
| base_17_126e38fc_t712_f0__ambiguous | colgrep | `fuse_mean` | ambiguous | ambiguous | 0.9617 | 1.0000 | 0.0000 | False | 0.9909 | 75 |
| base_17_126e38fc_t712_f0__ambiguous | colgrep | `fuse_max` | ambiguous | ambiguous | 0.9799 | 1.0000 | 0.0000 | False | 0.9969 | 75 |
| base_17_126e38fc_t712_f0__ambiguous | colgrep | `fuse_min` | ambiguous | ambiguous | 0.9436 | 1.0000 | 0.0000 | False | 0.9848 | 75 |
| base_18_f0afeaf2_t457_f0__correct | colgrep | `gte_only` | grounded | correct | 0.9734 | 1.0000 | 0.0000 | False | 0.9999 | 92 |
| base_18_f0afeaf2_t457_f0__correct | colgrep | `code_only` | grounded | correct | 0.9168 | 1.0000 | 0.0000 | False | 0.9863 | 92 |
| base_18_f0afeaf2_t457_f0__correct | colgrep | `fuse_mean` | grounded | correct | 0.9451 | 1.0000 | 0.0000 | False | 0.9931 | 92 |
| base_18_f0afeaf2_t457_f0__correct | colgrep | `fuse_max` | grounded | correct | 0.9734 | 1.0000 | 0.0000 | False | 0.9999 | 92 |
| base_18_f0afeaf2_t457_f0__correct | colgrep | `fuse_min` | grounded | correct | 0.9168 | 1.0000 | 0.0000 | False | 0.9863 | 92 |
| base_18_f0afeaf2_t457_f0__wrong | colgrep | `gte_only` | ungrounded | wrong | 0.9656 | 1.0000 | 0.0000 | False | 0.9965 | 92 |
| base_18_f0afeaf2_t457_f0__wrong | colgrep | `code_only` | ungrounded | wrong | 0.9029 | 1.0000 | 0.0000 | False | 0.9859 | 92 |
| base_18_f0afeaf2_t457_f0__wrong | colgrep | `fuse_mean` | ungrounded | wrong | 0.9342 | 1.0000 | 0.0000 | False | 0.9912 | 92 |
| base_18_f0afeaf2_t457_f0__wrong | colgrep | `fuse_max` | ungrounded | wrong | 0.9656 | 1.0000 | 0.0000 | False | 0.9965 | 92 |
| base_18_f0afeaf2_t457_f0__wrong | colgrep | `fuse_min` | ungrounded | wrong | 0.9029 | 1.0000 | 0.0000 | False | 0.9859 | 92 |
| base_18_f0afeaf2_t457_f0__ambiguous | colgrep | `gte_only` | ambiguous | ambiguous | 0.9752 | 1.0000 | 0.0000 | False | 0.9958 | 92 |
| base_18_f0afeaf2_t457_f0__ambiguous | colgrep | `code_only` | ambiguous | ambiguous | 0.9152 | 1.0000 | 0.0000 | False | 0.9814 | 92 |
| base_18_f0afeaf2_t457_f0__ambiguous | colgrep | `fuse_mean` | ambiguous | ambiguous | 0.9452 | 1.0000 | 0.0000 | False | 0.9886 | 92 |
| base_18_f0afeaf2_t457_f0__ambiguous | colgrep | `fuse_max` | ambiguous | ambiguous | 0.9752 | 1.0000 | 0.0000 | False | 0.9958 | 92 |
| base_18_f0afeaf2_t457_f0__ambiguous | colgrep | `fuse_min` | ambiguous | ambiguous | 0.9152 | 1.0000 | 0.0000 | False | 0.9814 | 92 |
| base_19_5e875b7e_t1360_f0__correct | colgrep | `gte_only` | grounded | correct | 0.9695 | 1.0000 | 0.0000 | False | 0.9942 | 23 |
| base_19_5e875b7e_t1360_f0__correct | colgrep | `code_only` | grounded | correct | 0.9311 | 1.0000 | 0.0000 | False | 0.9885 | 23 |
| base_19_5e875b7e_t1360_f0__correct | colgrep | `fuse_mean` | grounded | correct | 0.9503 | 1.0000 | 0.0000 | False | 0.9914 | 23 |
| base_19_5e875b7e_t1360_f0__correct | colgrep | `fuse_max` | grounded | correct | 0.9695 | 1.0000 | 0.0000 | False | 0.9942 | 23 |
| base_19_5e875b7e_t1360_f0__correct | colgrep | `fuse_min` | grounded | correct | 0.9311 | 1.0000 | 0.0000 | False | 0.9885 | 23 |
| base_19_5e875b7e_t1360_f0__wrong | colgrep | `gte_only` | ungrounded | wrong | 0.9659 | 1.0000 | 0.0000 | False | 0.9973 | 23 |
| base_19_5e875b7e_t1360_f0__wrong | colgrep | `code_only` | ungrounded | wrong | 0.9192 | 1.0000 | 0.0000 | False | 0.9860 | 23 |
| base_19_5e875b7e_t1360_f0__wrong | colgrep | `fuse_mean` | ungrounded | wrong | 0.9426 | 1.0000 | 0.0000 | False | 0.9917 | 23 |
| base_19_5e875b7e_t1360_f0__wrong | colgrep | `fuse_max` | ungrounded | wrong | 0.9659 | 1.0000 | 0.0000 | False | 0.9973 | 23 |
| base_19_5e875b7e_t1360_f0__wrong | colgrep | `fuse_min` | ungrounded | wrong | 0.9192 | 1.0000 | 0.0000 | False | 0.9860 | 23 |
| base_19_5e875b7e_t1360_f0__ambiguous | colgrep | `gte_only` | ambiguous | ambiguous | 0.9693 | 1.0000 | 0.0000 | False | 0.9966 | 23 |
| base_19_5e875b7e_t1360_f0__ambiguous | colgrep | `code_only` | ambiguous | ambiguous | 0.9324 | 1.0000 | 0.0000 | False | 0.9885 | 23 |
| base_19_5e875b7e_t1360_f0__ambiguous | colgrep | `fuse_mean` | ambiguous | ambiguous | 0.9508 | 1.0000 | 0.0000 | False | 0.9926 | 23 |
| base_19_5e875b7e_t1360_f0__ambiguous | colgrep | `fuse_max` | ambiguous | ambiguous | 0.9693 | 1.0000 | 0.0000 | False | 0.9966 | 23 |
| base_19_5e875b7e_t1360_f0__ambiguous | colgrep | `fuse_min` | ambiguous | ambiguous | 0.9324 | 1.0000 | 0.0000 | False | 0.9885 | 23 |
| base_20_d7fd7127_t1617_f0__correct | colgrep | `gte_only` | grounded | correct | 0.9868 | 1.0000 | 0.0000 | False | 0.9989 | 160 |
| base_20_d7fd7127_t1617_f0__correct | colgrep | `code_only` | grounded | correct | 0.9531 | 1.0000 | 0.0000 | False | 0.9915 | 160 |
| base_20_d7fd7127_t1617_f0__correct | colgrep | `fuse_mean` | grounded | correct | 0.9700 | 1.0000 | 0.0000 | False | 0.9952 | 160 |
| base_20_d7fd7127_t1617_f0__correct | colgrep | `fuse_max` | grounded | correct | 0.9868 | 1.0000 | 0.0000 | False | 0.9989 | 160 |
| base_20_d7fd7127_t1617_f0__correct | colgrep | `fuse_min` | grounded | correct | 0.9531 | 1.0000 | 0.0000 | False | 0.9915 | 160 |
| base_20_d7fd7127_t1617_f0__wrong | colgrep | `gte_only` | ungrounded | wrong | 0.9862 | 1.0000 | 0.0000 | False | 0.9990 | 160 |
| base_20_d7fd7127_t1617_f0__wrong | colgrep | `code_only` | ungrounded | wrong | 0.9496 | 1.0000 | 0.0000 | False | 0.9906 | 160 |
| base_20_d7fd7127_t1617_f0__wrong | colgrep | `fuse_mean` | ungrounded | wrong | 0.9679 | 1.0000 | 0.0000 | False | 0.9948 | 160 |
| base_20_d7fd7127_t1617_f0__wrong | colgrep | `fuse_max` | ungrounded | wrong | 0.9862 | 1.0000 | 0.0000 | False | 0.9990 | 160 |
| base_20_d7fd7127_t1617_f0__wrong | colgrep | `fuse_min` | ungrounded | wrong | 0.9496 | 1.0000 | 0.0000 | False | 0.9906 | 160 |
| base_20_d7fd7127_t1617_f0__ambiguous | colgrep | `gte_only` | ambiguous | ambiguous | 0.9865 | 1.0000 | 0.0000 | False | 0.9989 | 160 |
| base_20_d7fd7127_t1617_f0__ambiguous | colgrep | `code_only` | ambiguous | ambiguous | 0.9538 | 1.0000 | 0.0000 | False | 0.9914 | 160 |
| base_20_d7fd7127_t1617_f0__ambiguous | colgrep | `fuse_mean` | ambiguous | ambiguous | 0.9701 | 1.0000 | 0.0000 | False | 0.9952 | 160 |
| base_20_d7fd7127_t1617_f0__ambiguous | colgrep | `fuse_max` | ambiguous | ambiguous | 0.9865 | 1.0000 | 0.0000 | False | 0.9989 | 160 |
| base_20_d7fd7127_t1617_f0__ambiguous | colgrep | `fuse_min` | ambiguous | ambiguous | 0.9538 | 1.0000 | 0.0000 | False | 0.9914 | 160 |
| base_21_a4d14a56_t732_f0__correct | colgrep | `gte_only` | grounded | correct | 0.9807 | 1.0000 | 0.0000 | False | 1.0000 | 98 |
| base_21_a4d14a56_t732_f0__correct | colgrep | `code_only` | grounded | correct | 0.9345 | 1.0000 | 0.0000 | False | 0.9999 | 98 |
| base_21_a4d14a56_t732_f0__correct | colgrep | `fuse_mean` | grounded | correct | 0.9576 | 1.0000 | 0.0000 | False | 1.0000 | 98 |
| base_21_a4d14a56_t732_f0__correct | colgrep | `fuse_max` | grounded | correct | 0.9807 | 1.0000 | 0.0000 | False | 1.0000 | 98 |
| base_21_a4d14a56_t732_f0__correct | colgrep | `fuse_min` | grounded | correct | 0.9345 | 1.0000 | 0.0000 | False | 0.9999 | 98 |
| base_21_a4d14a56_t732_f0__wrong | colgrep | `gte_only` | ungrounded | wrong | 0.9711 | 1.0000 | 0.0000 | False | 1.0000 | 98 |
| base_21_a4d14a56_t732_f0__wrong | colgrep | `code_only` | ungrounded | wrong | 0.9216 | 1.0000 | 0.0000 | False | 0.9999 | 98 |
| base_21_a4d14a56_t732_f0__wrong | colgrep | `fuse_mean` | ungrounded | wrong | 0.9463 | 1.0000 | 0.0000 | False | 1.0000 | 98 |
| base_21_a4d14a56_t732_f0__wrong | colgrep | `fuse_max` | ungrounded | wrong | 0.9711 | 1.0000 | 0.0000 | False | 1.0000 | 98 |
| base_21_a4d14a56_t732_f0__wrong | colgrep | `fuse_min` | ungrounded | wrong | 0.9216 | 1.0000 | 0.0000 | False | 0.9999 | 98 |
| base_21_a4d14a56_t732_f0__ambiguous | colgrep | `gte_only` | ambiguous | ambiguous | 0.9742 | 1.0000 | 0.0000 | False | 1.0000 | 98 |
| base_21_a4d14a56_t732_f0__ambiguous | colgrep | `code_only` | ambiguous | ambiguous | 0.9256 | 1.0000 | 0.0000 | False | 0.9999 | 98 |
| base_21_a4d14a56_t732_f0__ambiguous | colgrep | `fuse_mean` | ambiguous | ambiguous | 0.9499 | 1.0000 | 0.0000 | False | 1.0000 | 98 |
| base_21_a4d14a56_t732_f0__ambiguous | colgrep | `fuse_max` | ambiguous | ambiguous | 0.9742 | 1.0000 | 0.0000 | False | 1.0000 | 98 |
| base_21_a4d14a56_t732_f0__ambiguous | colgrep | `fuse_min` | ambiguous | ambiguous | 0.9256 | 1.0000 | 0.0000 | False | 0.9999 | 98 |
| base_22_d7fd7127_t1209_f1__correct | colgrep | `gte_only` | grounded | correct | 0.9733 | 1.0000 | 0.0000 | False | 1.0000 | 105 |
| base_22_d7fd7127_t1209_f1__correct | colgrep | `code_only` | grounded | correct | 0.9218 | 1.0000 | 0.0000 | False | 0.9772 | 105 |
| base_22_d7fd7127_t1209_f1__correct | colgrep | `fuse_mean` | grounded | correct | 0.9475 | 1.0000 | 0.0000 | False | 0.9886 | 105 |
| base_22_d7fd7127_t1209_f1__correct | colgrep | `fuse_max` | grounded | correct | 0.9733 | 1.0000 | 0.0000 | False | 1.0000 | 105 |
| base_22_d7fd7127_t1209_f1__correct | colgrep | `fuse_min` | grounded | correct | 0.9218 | 1.0000 | 0.0000 | False | 0.9772 | 105 |
| base_22_d7fd7127_t1209_f1__wrong | colgrep | `gte_only` | ungrounded | wrong | 0.9737 | 1.0000 | 0.0000 | False | 1.0000 | 105 |
| base_22_d7fd7127_t1209_f1__wrong | colgrep | `code_only` | ungrounded | wrong | 0.9240 | 1.0000 | 0.0000 | False | 0.9776 | 105 |
| base_22_d7fd7127_t1209_f1__wrong | colgrep | `fuse_mean` | ungrounded | wrong | 0.9489 | 1.0000 | 0.0000 | False | 0.9888 | 105 |
| base_22_d7fd7127_t1209_f1__wrong | colgrep | `fuse_max` | ungrounded | wrong | 0.9737 | 1.0000 | 0.0000 | False | 1.0000 | 105 |
| base_22_d7fd7127_t1209_f1__wrong | colgrep | `fuse_min` | ungrounded | wrong | 0.9240 | 1.0000 | 0.0000 | False | 0.9776 | 105 |
| base_22_d7fd7127_t1209_f1__ambiguous | colgrep | `gte_only` | ambiguous | ambiguous | 0.9734 | 1.0000 | 0.0000 | False | 1.0000 | 105 |
| base_22_d7fd7127_t1209_f1__ambiguous | colgrep | `code_only` | ambiguous | ambiguous | 0.9239 | 1.0000 | 0.0000 | False | 0.9777 | 105 |
| base_22_d7fd7127_t1209_f1__ambiguous | colgrep | `fuse_mean` | ambiguous | ambiguous | 0.9487 | 1.0000 | 0.0000 | False | 0.9888 | 105 |
| base_22_d7fd7127_t1209_f1__ambiguous | colgrep | `fuse_max` | ambiguous | ambiguous | 0.9734 | 1.0000 | 0.0000 | False | 1.0000 | 105 |
| base_22_d7fd7127_t1209_f1__ambiguous | colgrep | `fuse_min` | ambiguous | ambiguous | 0.9239 | 1.0000 | 0.0000 | False | 0.9777 | 105 |
| base_23_a4d14a56_t579_f0__correct | colgrep | `gte_only` | grounded | correct | 0.9792 | 1.0000 | 0.0000 | False | 0.9975 | 84 |
| base_23_a4d14a56_t579_f0__correct | colgrep | `code_only` | grounded | correct | 0.9276 | 1.0000 | 0.0000 | False | 0.9999 | 84 |
| base_23_a4d14a56_t579_f0__correct | colgrep | `fuse_mean` | grounded | correct | 0.9534 | 1.0000 | 0.0000 | False | 0.9987 | 84 |
| base_23_a4d14a56_t579_f0__correct | colgrep | `fuse_max` | grounded | correct | 0.9792 | 1.0000 | 0.0000 | False | 0.9999 | 84 |
| base_23_a4d14a56_t579_f0__correct | colgrep | `fuse_min` | grounded | correct | 0.9276 | 1.0000 | 0.0000 | False | 0.9975 | 84 |
| base_23_a4d14a56_t579_f0__wrong | colgrep | `gte_only` | ungrounded | wrong | 0.9674 | 1.0000 | 0.0000 | False | 0.9983 | 84 |
| base_23_a4d14a56_t579_f0__wrong | colgrep | `code_only` | ungrounded | wrong | 0.9152 | 1.0000 | 0.0000 | False | 0.9871 | 84 |
| base_23_a4d14a56_t579_f0__wrong | colgrep | `fuse_mean` | ungrounded | wrong | 0.9413 | 1.0000 | 0.0000 | False | 0.9927 | 84 |
| base_23_a4d14a56_t579_f0__wrong | colgrep | `fuse_max` | ungrounded | wrong | 0.9674 | 1.0000 | 0.0000 | False | 0.9983 | 84 |
| base_23_a4d14a56_t579_f0__wrong | colgrep | `fuse_min` | ungrounded | wrong | 0.9152 | 1.0000 | 0.0000 | False | 0.9871 | 84 |
| base_23_a4d14a56_t579_f0__ambiguous | colgrep | `gte_only` | ambiguous | ambiguous | 0.9792 | 1.0000 | 0.0000 | False | 0.9975 | 84 |
| base_23_a4d14a56_t579_f0__ambiguous | colgrep | `code_only` | ambiguous | ambiguous | 0.9288 | 1.0000 | 0.0000 | False | 0.9999 | 84 |
| base_23_a4d14a56_t579_f0__ambiguous | colgrep | `fuse_mean` | ambiguous | ambiguous | 0.9540 | 1.0000 | 0.0000 | False | 0.9987 | 84 |
| base_23_a4d14a56_t579_f0__ambiguous | colgrep | `fuse_max` | ambiguous | ambiguous | 0.9792 | 1.0000 | 0.0000 | False | 0.9999 | 84 |
| base_23_a4d14a56_t579_f0__ambiguous | colgrep | `fuse_min` | ambiguous | ambiguous | 0.9288 | 1.0000 | 0.0000 | False | 0.9975 | 84 |
| base_24_126e38fc_t1774_f0__correct | colgrep | `gte_only` | grounded | correct | 0.9785 | 1.0000 | 0.0000 | False | 1.0000 | 216 |
| base_24_126e38fc_t1774_f0__correct | colgrep | `code_only` | grounded | correct | 0.9239 | 1.0000 | 0.0000 | False | 0.9848 | 216 |
| base_24_126e38fc_t1774_f0__correct | colgrep | `fuse_mean` | grounded | correct | 0.9512 | 1.0000 | 0.0000 | False | 0.9924 | 216 |
| base_24_126e38fc_t1774_f0__correct | colgrep | `fuse_max` | grounded | correct | 0.9785 | 1.0000 | 0.0000 | False | 1.0000 | 216 |
| base_24_126e38fc_t1774_f0__correct | colgrep | `fuse_min` | grounded | correct | 0.9239 | 1.0000 | 0.0000 | False | 0.9848 | 216 |
| base_24_126e38fc_t1774_f0__wrong | colgrep | `gte_only` | ungrounded | wrong | 0.9704 | 1.0000 | 0.0000 | False | 1.0000 | 216 |
| base_24_126e38fc_t1774_f0__wrong | colgrep | `code_only` | ungrounded | wrong | 0.9157 | 1.0000 | 0.0000 | False | 0.9858 | 216 |
| base_24_126e38fc_t1774_f0__wrong | colgrep | `fuse_mean` | ungrounded | wrong | 0.9430 | 1.0000 | 0.0000 | False | 0.9929 | 216 |
| base_24_126e38fc_t1774_f0__wrong | colgrep | `fuse_max` | ungrounded | wrong | 0.9704 | 1.0000 | 0.0000 | False | 1.0000 | 216 |
| base_24_126e38fc_t1774_f0__wrong | colgrep | `fuse_min` | ungrounded | wrong | 0.9157 | 1.0000 | 0.0000 | False | 0.9858 | 216 |
| base_24_126e38fc_t1774_f0__ambiguous | colgrep | `gte_only` | ambiguous | ambiguous | 0.9785 | 1.0000 | 0.0000 | False | 0.9999 | 216 |
| base_24_126e38fc_t1774_f0__ambiguous | colgrep | `code_only` | ambiguous | ambiguous | 0.9242 | 1.0000 | 0.0000 | False | 0.9858 | 216 |
| base_24_126e38fc_t1774_f0__ambiguous | colgrep | `fuse_mean` | ambiguous | ambiguous | 0.9514 | 1.0000 | 0.0000 | False | 0.9929 | 216 |
| base_24_126e38fc_t1774_f0__ambiguous | colgrep | `fuse_max` | ambiguous | ambiguous | 0.9785 | 1.0000 | 0.0000 | False | 0.9999 | 216 |
| base_24_126e38fc_t1774_f0__ambiguous | colgrep | `fuse_min` | ambiguous | ambiguous | 0.9242 | 1.0000 | 0.0000 | False | 0.9858 | 216 |
| base_25_8a0f6f68_t1261_f1__correct | colgrep | `gte_only` | grounded | correct | 0.9757 | 1.0000 | 0.0000 | False | 1.0000 | 145 |
| base_25_8a0f6f68_t1261_f1__correct | colgrep | `code_only` | grounded | correct | 0.9504 | 1.0000 | 0.0000 | False | 1.0000 | 145 |
| base_25_8a0f6f68_t1261_f1__correct | colgrep | `fuse_mean` | grounded | correct | 0.9630 | 1.0000 | 0.0000 | False | 1.0000 | 145 |
| base_25_8a0f6f68_t1261_f1__correct | colgrep | `fuse_max` | grounded | correct | 0.9757 | 1.0000 | 0.0000 | False | 1.0000 | 145 |
| base_25_8a0f6f68_t1261_f1__correct | colgrep | `fuse_min` | grounded | correct | 0.9504 | 1.0000 | 0.0000 | False | 1.0000 | 145 |
| base_25_8a0f6f68_t1261_f1__wrong | colgrep | `gte_only` | ungrounded | wrong | 0.9671 | 1.0000 | 0.0000 | False | 1.0000 | 145 |
| base_25_8a0f6f68_t1261_f1__wrong | colgrep | `code_only` | ungrounded | wrong | 0.9458 | 1.0000 | 0.0000 | False | 1.0000 | 145 |
| base_25_8a0f6f68_t1261_f1__wrong | colgrep | `fuse_mean` | ungrounded | wrong | 0.9565 | 1.0000 | 0.0000 | False | 1.0000 | 145 |
| base_25_8a0f6f68_t1261_f1__wrong | colgrep | `fuse_max` | ungrounded | wrong | 0.9671 | 1.0000 | 0.0000 | False | 1.0000 | 145 |
| base_25_8a0f6f68_t1261_f1__wrong | colgrep | `fuse_min` | ungrounded | wrong | 0.9458 | 1.0000 | 0.0000 | False | 1.0000 | 145 |
| base_25_8a0f6f68_t1261_f1__ambiguous | colgrep | `gte_only` | ambiguous | ambiguous | 0.9740 | 1.0000 | 0.0000 | False | 1.0000 | 145 |
| base_25_8a0f6f68_t1261_f1__ambiguous | colgrep | `code_only` | ambiguous | ambiguous | 0.9489 | 1.0000 | 0.0000 | False | 1.0000 | 145 |
| base_25_8a0f6f68_t1261_f1__ambiguous | colgrep | `fuse_mean` | ambiguous | ambiguous | 0.9615 | 1.0000 | 0.0000 | False | 1.0000 | 145 |
| base_25_8a0f6f68_t1261_f1__ambiguous | colgrep | `fuse_max` | ambiguous | ambiguous | 0.9740 | 1.0000 | 0.0000 | False | 1.0000 | 145 |
| base_25_8a0f6f68_t1261_f1__ambiguous | colgrep | `fuse_min` | ambiguous | ambiguous | 0.9489 | 1.0000 | 0.0000 | False | 1.0000 | 145 |
| base_26_f0afeaf2_t1530_f1__correct | colgrep | `gte_only` | grounded | correct | 0.9803 | 1.0000 | 0.0000 | False | 0.9990 | 152 |
| base_26_f0afeaf2_t1530_f1__correct | colgrep | `code_only` | grounded | correct | 0.9440 | 1.0000 | 0.0000 | False | 1.0000 | 152 |
| base_26_f0afeaf2_t1530_f1__correct | colgrep | `fuse_mean` | grounded | correct | 0.9621 | 1.0000 | 0.0000 | False | 0.9995 | 152 |
| base_26_f0afeaf2_t1530_f1__correct | colgrep | `fuse_max` | grounded | correct | 0.9803 | 1.0000 | 0.0000 | False | 1.0000 | 152 |
| base_26_f0afeaf2_t1530_f1__correct | colgrep | `fuse_min` | grounded | correct | 0.9440 | 1.0000 | 0.0000 | False | 0.9990 | 152 |
| base_26_f0afeaf2_t1530_f1__wrong | colgrep | `gte_only` | ungrounded | wrong | 0.9804 | 1.0000 | 0.0000 | False | 0.9995 | 152 |
| base_26_f0afeaf2_t1530_f1__wrong | colgrep | `code_only` | ungrounded | wrong | 0.9431 | 1.0000 | 0.0000 | False | 1.0000 | 152 |
| base_26_f0afeaf2_t1530_f1__wrong | colgrep | `fuse_mean` | ungrounded | wrong | 0.9618 | 1.0000 | 0.0000 | False | 0.9997 | 152 |
| base_26_f0afeaf2_t1530_f1__wrong | colgrep | `fuse_max` | ungrounded | wrong | 0.9804 | 1.0000 | 0.0000 | False | 1.0000 | 152 |
| base_26_f0afeaf2_t1530_f1__wrong | colgrep | `fuse_min` | ungrounded | wrong | 0.9431 | 1.0000 | 0.0000 | False | 0.9995 | 152 |
| base_26_f0afeaf2_t1530_f1__ambiguous | colgrep | `gte_only` | ambiguous | ambiguous | 0.9804 | 1.0000 | 0.0000 | False | 0.9991 | 152 |
| base_26_f0afeaf2_t1530_f1__ambiguous | colgrep | `code_only` | ambiguous | ambiguous | 0.9388 | 1.0000 | 0.0000 | False | 1.0000 | 152 |
| base_26_f0afeaf2_t1530_f1__ambiguous | colgrep | `fuse_mean` | ambiguous | ambiguous | 0.9596 | 1.0000 | 0.0000 | False | 0.9995 | 152 |
| base_26_f0afeaf2_t1530_f1__ambiguous | colgrep | `fuse_max` | ambiguous | ambiguous | 0.9804 | 1.0000 | 0.0000 | False | 1.0000 | 152 |
| base_26_f0afeaf2_t1530_f1__ambiguous | colgrep | `fuse_min` | ambiguous | ambiguous | 0.9388 | 1.0000 | 0.0000 | False | 0.9991 | 152 |
| base_27_8a0f6f68_t1163_f0__correct | colgrep | `gte_only` | grounded | correct | 0.9773 | 1.0000 | 0.0000 | False | 1.0000 | 138 |
| base_27_8a0f6f68_t1163_f0__correct | colgrep | `code_only` | grounded | correct | 0.9432 | 1.0000 | 0.0000 | False | 0.9999 | 138 |
| base_27_8a0f6f68_t1163_f0__correct | colgrep | `fuse_mean` | grounded | correct | 0.9603 | 1.0000 | 0.0000 | False | 0.9999 | 138 |
| base_27_8a0f6f68_t1163_f0__correct | colgrep | `fuse_max` | grounded | correct | 0.9773 | 1.0000 | 0.0000 | False | 1.0000 | 138 |
| base_27_8a0f6f68_t1163_f0__correct | colgrep | `fuse_min` | grounded | correct | 0.9432 | 1.0000 | 0.0000 | False | 0.9999 | 138 |
| base_27_8a0f6f68_t1163_f0__wrong | colgrep | `gte_only` | ungrounded | wrong | 0.9682 | 1.0000 | 0.0000 | False | 1.0000 | 138 |
| base_27_8a0f6f68_t1163_f0__wrong | colgrep | `code_only` | ungrounded | wrong | 0.9185 | 1.0000 | 0.0000 | False | 0.9999 | 138 |
| base_27_8a0f6f68_t1163_f0__wrong | colgrep | `fuse_mean` | ungrounded | wrong | 0.9434 | 1.0000 | 0.0000 | False | 0.9999 | 138 |
| base_27_8a0f6f68_t1163_f0__wrong | colgrep | `fuse_max` | ungrounded | wrong | 0.9682 | 1.0000 | 0.0000 | False | 1.0000 | 138 |
| base_27_8a0f6f68_t1163_f0__wrong | colgrep | `fuse_min` | ungrounded | wrong | 0.9185 | 1.0000 | 0.0000 | False | 0.9999 | 138 |
| base_27_8a0f6f68_t1163_f0__ambiguous | colgrep | `gte_only` | ambiguous | ambiguous | 0.9732 | 1.0000 | 0.0000 | False | 0.9998 | 138 |
| base_27_8a0f6f68_t1163_f0__ambiguous | colgrep | `code_only` | ambiguous | ambiguous | 0.9305 | 1.0000 | 0.0000 | False | 0.9999 | 138 |
| base_27_8a0f6f68_t1163_f0__ambiguous | colgrep | `fuse_mean` | ambiguous | ambiguous | 0.9518 | 1.0000 | 0.0000 | False | 0.9999 | 138 |
| base_27_8a0f6f68_t1163_f0__ambiguous | colgrep | `fuse_max` | ambiguous | ambiguous | 0.9732 | 1.0000 | 0.0000 | False | 0.9999 | 138 |
| base_27_8a0f6f68_t1163_f0__ambiguous | colgrep | `fuse_min` | ambiguous | ambiguous | 0.9305 | 1.0000 | 0.0000 | False | 0.9998 | 138 |
| base_28_126e38fc_t1892_f0__correct | colgrep | `gte_only` | grounded | correct | 0.9872 | 1.0000 | 0.0000 | False | 0.9976 | 230 |
| base_28_126e38fc_t1892_f0__correct | colgrep | `code_only` | grounded | correct | 0.9620 | 1.0000 | 0.0000 | False | 0.9955 | 230 |
| base_28_126e38fc_t1892_f0__correct | colgrep | `fuse_mean` | grounded | correct | 0.9746 | 1.0000 | 0.0000 | False | 0.9966 | 230 |
| base_28_126e38fc_t1892_f0__correct | colgrep | `fuse_max` | grounded | correct | 0.9872 | 1.0000 | 0.0000 | False | 0.9976 | 230 |
| base_28_126e38fc_t1892_f0__correct | colgrep | `fuse_min` | grounded | correct | 0.9620 | 1.0000 | 0.0000 | False | 0.9955 | 230 |
| base_28_126e38fc_t1892_f0__wrong | colgrep | `gte_only` | ungrounded | wrong | 0.9813 | 1.0000 | 0.0000 | False | 0.9972 | 230 |
| base_28_126e38fc_t1892_f0__wrong | colgrep | `code_only` | ungrounded | wrong | 0.9541 | 1.0000 | 0.0000 | False | 0.9953 | 230 |
| base_28_126e38fc_t1892_f0__wrong | colgrep | `fuse_mean` | ungrounded | wrong | 0.9677 | 1.0000 | 0.0000 | False | 0.9962 | 230 |
| base_28_126e38fc_t1892_f0__wrong | colgrep | `fuse_max` | ungrounded | wrong | 0.9813 | 1.0000 | 0.0000 | False | 0.9972 | 230 |
| base_28_126e38fc_t1892_f0__wrong | colgrep | `fuse_min` | ungrounded | wrong | 0.9541 | 1.0000 | 0.0000 | False | 0.9953 | 230 |
| base_28_126e38fc_t1892_f0__ambiguous | colgrep | `gte_only` | ambiguous | ambiguous | 0.9872 | 1.0000 | 0.0000 | False | 0.9979 | 230 |
| base_28_126e38fc_t1892_f0__ambiguous | colgrep | `code_only` | ambiguous | ambiguous | 0.9624 | 1.0000 | 0.0000 | False | 0.9955 | 230 |
| base_28_126e38fc_t1892_f0__ambiguous | colgrep | `fuse_mean` | ambiguous | ambiguous | 0.9748 | 1.0000 | 0.0000 | False | 0.9967 | 230 |
| base_28_126e38fc_t1892_f0__ambiguous | colgrep | `fuse_max` | ambiguous | ambiguous | 0.9872 | 1.0000 | 0.0000 | False | 0.9979 | 230 |
| base_28_126e38fc_t1892_f0__ambiguous | colgrep | `fuse_min` | ambiguous | ambiguous | 0.9624 | 1.0000 | 0.0000 | False | 0.9955 | 230 |
| base_29_8a0f6f68_t1278_f0__correct | colgrep | `gte_only` | grounded | correct | 0.9775 | 1.0000 | 0.0000 | False | 0.9999 | 150 |
| base_29_8a0f6f68_t1278_f0__correct | colgrep | `code_only` | grounded | correct | 0.9375 | 1.0000 | 0.0000 | False | 1.0000 | 150 |
| base_29_8a0f6f68_t1278_f0__correct | colgrep | `fuse_mean` | grounded | correct | 0.9575 | 1.0000 | 0.0000 | False | 1.0000 | 150 |
| base_29_8a0f6f68_t1278_f0__correct | colgrep | `fuse_max` | grounded | correct | 0.9775 | 1.0000 | 0.0000 | False | 1.0000 | 150 |
| base_29_8a0f6f68_t1278_f0__correct | colgrep | `fuse_min` | grounded | correct | 0.9375 | 1.0000 | 0.0000 | False | 0.9999 | 150 |
| base_29_8a0f6f68_t1278_f0__wrong | colgrep | `gte_only` | ungrounded | wrong | 0.9685 | 1.0000 | 0.0000 | False | 1.0000 | 150 |
| base_29_8a0f6f68_t1278_f0__wrong | colgrep | `code_only` | ungrounded | wrong | 0.9193 | 1.0000 | 0.0000 | False | 1.0000 | 150 |
| base_29_8a0f6f68_t1278_f0__wrong | colgrep | `fuse_mean` | ungrounded | wrong | 0.9439 | 1.0000 | 0.0000 | False | 1.0000 | 150 |
| base_29_8a0f6f68_t1278_f0__wrong | colgrep | `fuse_max` | ungrounded | wrong | 0.9685 | 1.0000 | 0.0000 | False | 1.0000 | 150 |
| base_29_8a0f6f68_t1278_f0__wrong | colgrep | `fuse_min` | ungrounded | wrong | 0.9193 | 1.0000 | 0.0000 | False | 1.0000 | 150 |
| base_29_8a0f6f68_t1278_f0__ambiguous | colgrep | `gte_only` | ambiguous | ambiguous | 0.9717 | 1.0000 | 0.0000 | False | 1.0000 | 150 |
| base_29_8a0f6f68_t1278_f0__ambiguous | colgrep | `code_only` | ambiguous | ambiguous | 0.9216 | 1.0000 | 0.0000 | False | 0.9999 | 150 |
| base_29_8a0f6f68_t1278_f0__ambiguous | colgrep | `fuse_mean` | ambiguous | ambiguous | 0.9466 | 1.0000 | 0.0000 | False | 0.9999 | 150 |
| base_29_8a0f6f68_t1278_f0__ambiguous | colgrep | `fuse_max` | ambiguous | ambiguous | 0.9717 | 1.0000 | 0.0000 | False | 1.0000 | 150 |
| base_29_8a0f6f68_t1278_f0__ambiguous | colgrep | `fuse_min` | ambiguous | ambiguous | 0.9216 | 1.0000 | 0.0000 | False | 0.9999 | 150 |
| base_30_d7fd7127_t807_f1__correct | colgrep | `gte_only` | grounded | correct | 0.9750 | 1.0000 | 0.0000 | False | 0.9988 | 70 |
| base_30_d7fd7127_t807_f1__correct | colgrep | `code_only` | grounded | correct | 0.9253 | 1.0000 | 0.0000 | False | 0.9861 | 70 |
| base_30_d7fd7127_t807_f1__correct | colgrep | `fuse_mean` | grounded | correct | 0.9502 | 1.0000 | 0.0000 | False | 0.9924 | 70 |
| base_30_d7fd7127_t807_f1__correct | colgrep | `fuse_max` | grounded | correct | 0.9750 | 1.0000 | 0.0000 | False | 0.9988 | 70 |
| base_30_d7fd7127_t807_f1__correct | colgrep | `fuse_min` | grounded | correct | 0.9253 | 1.0000 | 0.0000 | False | 0.9861 | 70 |
| base_30_d7fd7127_t807_f1__wrong | colgrep | `gte_only` | ungrounded | wrong | 0.9751 | 1.0000 | 0.0000 | False | 0.9988 | 70 |
| base_30_d7fd7127_t807_f1__wrong | colgrep | `code_only` | ungrounded | wrong | 0.9243 | 1.0000 | 0.0000 | False | 0.9863 | 70 |
| base_30_d7fd7127_t807_f1__wrong | colgrep | `fuse_mean` | ungrounded | wrong | 0.9497 | 1.0000 | 0.0000 | False | 0.9925 | 70 |
| base_30_d7fd7127_t807_f1__wrong | colgrep | `fuse_max` | ungrounded | wrong | 0.9751 | 1.0000 | 0.0000 | False | 0.9988 | 70 |
| base_30_d7fd7127_t807_f1__wrong | colgrep | `fuse_min` | ungrounded | wrong | 0.9243 | 1.0000 | 0.0000 | False | 0.9863 | 70 |
| base_30_d7fd7127_t807_f1__ambiguous | colgrep | `gte_only` | ambiguous | ambiguous | 0.9721 | 1.0000 | 0.0000 | False | 0.9999 | 70 |
| base_30_d7fd7127_t807_f1__ambiguous | colgrep | `code_only` | ambiguous | ambiguous | 0.9261 | 1.0000 | 0.0000 | False | 0.9847 | 70 |
| base_30_d7fd7127_t807_f1__ambiguous | colgrep | `fuse_mean` | ambiguous | ambiguous | 0.9491 | 1.0000 | 0.0000 | False | 0.9923 | 70 |
| base_30_d7fd7127_t807_f1__ambiguous | colgrep | `fuse_max` | ambiguous | ambiguous | 0.9721 | 1.0000 | 0.0000 | False | 0.9999 | 70 |
| base_30_d7fd7127_t807_f1__ambiguous | colgrep | `fuse_min` | ambiguous | ambiguous | 0.9261 | 1.0000 | 0.0000 | False | 0.9847 | 70 |
| base_31_8a0f6f68_t1301_f0__correct | colgrep | `gte_only` | grounded | correct | 0.9722 | 1.0000 | 0.0000 | False | 1.0000 | 149 |
| base_31_8a0f6f68_t1301_f0__correct | colgrep | `code_only` | grounded | correct | 0.9199 | 1.0000 | 0.0000 | False | 1.0000 | 149 |
| base_31_8a0f6f68_t1301_f0__correct | colgrep | `fuse_mean` | grounded | correct | 0.9460 | 1.0000 | 0.0000 | False | 1.0000 | 149 |
| base_31_8a0f6f68_t1301_f0__correct | colgrep | `fuse_max` | grounded | correct | 0.9722 | 1.0000 | 0.0000 | False | 1.0000 | 149 |
| base_31_8a0f6f68_t1301_f0__correct | colgrep | `fuse_min` | grounded | correct | 0.9199 | 1.0000 | 0.0000 | False | 1.0000 | 149 |
| base_31_8a0f6f68_t1301_f0__wrong | colgrep | `gte_only` | ungrounded | wrong | 0.9656 | 1.0000 | 0.0000 | False | 1.0000 | 149 |
| base_31_8a0f6f68_t1301_f0__wrong | colgrep | `code_only` | ungrounded | wrong | 0.9202 | 1.0000 | 0.0000 | False | 1.0000 | 149 |
| base_31_8a0f6f68_t1301_f0__wrong | colgrep | `fuse_mean` | ungrounded | wrong | 0.9429 | 1.0000 | 0.0000 | False | 1.0000 | 149 |
| base_31_8a0f6f68_t1301_f0__wrong | colgrep | `fuse_max` | ungrounded | wrong | 0.9656 | 1.0000 | 0.0000 | False | 1.0000 | 149 |
| base_31_8a0f6f68_t1301_f0__wrong | colgrep | `fuse_min` | ungrounded | wrong | 0.9202 | 1.0000 | 0.0000 | False | 1.0000 | 149 |
| base_31_8a0f6f68_t1301_f0__ambiguous | colgrep | `gte_only` | ambiguous | ambiguous | 0.9707 | 1.0000 | 0.0000 | False | 1.0000 | 149 |
| base_31_8a0f6f68_t1301_f0__ambiguous | colgrep | `code_only` | ambiguous | ambiguous | 0.9220 | 1.0000 | 0.0000 | False | 1.0000 | 149 |
| base_31_8a0f6f68_t1301_f0__ambiguous | colgrep | `fuse_mean` | ambiguous | ambiguous | 0.9464 | 1.0000 | 0.0000 | False | 1.0000 | 149 |
| base_31_8a0f6f68_t1301_f0__ambiguous | colgrep | `fuse_max` | ambiguous | ambiguous | 0.9707 | 1.0000 | 0.0000 | False | 1.0000 | 149 |
| base_31_8a0f6f68_t1301_f0__ambiguous | colgrep | `fuse_min` | ambiguous | ambiguous | 0.9220 | 1.0000 | 0.0000 | False | 1.0000 | 149 |
| base_32_126e38fc_t1007_f1__correct | colgrep | `gte_only` | grounded | correct | 0.9851 | 1.0000 | 0.0000 | False | 1.0000 | 126 |
| base_32_126e38fc_t1007_f1__correct | colgrep | `code_only` | grounded | correct | 0.9348 | 1.0000 | 0.0000 | False | 1.0000 | 126 |
| base_32_126e38fc_t1007_f1__correct | colgrep | `fuse_mean` | grounded | correct | 0.9600 | 1.0000 | 0.0000 | False | 1.0000 | 126 |
| base_32_126e38fc_t1007_f1__correct | colgrep | `fuse_max` | grounded | correct | 0.9851 | 1.0000 | 0.0000 | False | 1.0000 | 126 |
| base_32_126e38fc_t1007_f1__correct | colgrep | `fuse_min` | grounded | correct | 0.9348 | 1.0000 | 0.0000 | False | 1.0000 | 126 |
| base_32_126e38fc_t1007_f1__wrong | colgrep | `gte_only` | ungrounded | wrong | 0.9737 | 1.0000 | 0.0000 | False | 0.9999 | 126 |
| base_32_126e38fc_t1007_f1__wrong | colgrep | `code_only` | ungrounded | wrong | 0.9279 | 1.0000 | 0.0000 | False | 1.0000 | 126 |
| base_32_126e38fc_t1007_f1__wrong | colgrep | `fuse_mean` | ungrounded | wrong | 0.9508 | 1.0000 | 0.0000 | False | 1.0000 | 126 |
| base_32_126e38fc_t1007_f1__wrong | colgrep | `fuse_max` | ungrounded | wrong | 0.9737 | 1.0000 | 0.0000 | False | 1.0000 | 126 |
| base_32_126e38fc_t1007_f1__wrong | colgrep | `fuse_min` | ungrounded | wrong | 0.9279 | 1.0000 | 0.0000 | False | 0.9999 | 126 |
| base_32_126e38fc_t1007_f1__ambiguous | colgrep | `gte_only` | ambiguous | ambiguous | 0.9843 | 1.0000 | 0.0000 | False | 1.0000 | 126 |
| base_32_126e38fc_t1007_f1__ambiguous | colgrep | `code_only` | ambiguous | ambiguous | 0.9368 | 1.0000 | 0.0000 | False | 1.0000 | 126 |
| base_32_126e38fc_t1007_f1__ambiguous | colgrep | `fuse_mean` | ambiguous | ambiguous | 0.9605 | 1.0000 | 0.0000 | False | 1.0000 | 126 |
| base_32_126e38fc_t1007_f1__ambiguous | colgrep | `fuse_max` | ambiguous | ambiguous | 0.9843 | 1.0000 | 0.0000 | False | 1.0000 | 126 |
| base_32_126e38fc_t1007_f1__ambiguous | colgrep | `fuse_min` | ambiguous | ambiguous | 0.9368 | 1.0000 | 0.0000 | False | 1.0000 | 126 |
| base_33_d7fd7127_t1397_f0__correct | colgrep | `gte_only` | grounded | correct | 0.9839 | 1.0000 | 0.0000 | False | 0.9975 | 139 |
| base_33_d7fd7127_t1397_f0__correct | colgrep | `code_only` | grounded | correct | 0.9391 | 1.0000 | 0.0000 | False | 0.9779 | 139 |
| base_33_d7fd7127_t1397_f0__correct | colgrep | `fuse_mean` | grounded | correct | 0.9615 | 1.0000 | 0.0000 | False | 0.9877 | 139 |
| base_33_d7fd7127_t1397_f0__correct | colgrep | `fuse_max` | grounded | correct | 0.9839 | 1.0000 | 0.0000 | False | 0.9975 | 139 |
| base_33_d7fd7127_t1397_f0__correct | colgrep | `fuse_min` | grounded | correct | 0.9391 | 1.0000 | 0.0000 | False | 0.9779 | 139 |
| base_33_d7fd7127_t1397_f0__wrong | colgrep | `gte_only` | ungrounded | wrong | 0.9803 | 1.0000 | 0.0000 | False | 0.9973 | 139 |
| base_33_d7fd7127_t1397_f0__wrong | colgrep | `code_only` | ungrounded | wrong | 0.9300 | 1.0000 | 0.0000 | False | 0.9681 | 139 |
| base_33_d7fd7127_t1397_f0__wrong | colgrep | `fuse_mean` | ungrounded | wrong | 0.9551 | 1.0000 | 0.0000 | False | 0.9827 | 139 |
| base_33_d7fd7127_t1397_f0__wrong | colgrep | `fuse_max` | ungrounded | wrong | 0.9803 | 1.0000 | 0.0000 | False | 0.9973 | 139 |
| base_33_d7fd7127_t1397_f0__wrong | colgrep | `fuse_min` | ungrounded | wrong | 0.9300 | 1.0000 | 0.0000 | False | 0.9681 | 139 |
| base_33_d7fd7127_t1397_f0__ambiguous | colgrep | `gte_only` | ambiguous | ambiguous | 0.9809 | 1.0000 | 0.0000 | False | 0.9974 | 139 |
| base_33_d7fd7127_t1397_f0__ambiguous | colgrep | `code_only` | ambiguous | ambiguous | 0.9378 | 1.0000 | 0.0000 | False | 0.9800 | 139 |
| base_33_d7fd7127_t1397_f0__ambiguous | colgrep | `fuse_mean` | ambiguous | ambiguous | 0.9593 | 1.0000 | 0.0000 | False | 0.9887 | 139 |
| base_33_d7fd7127_t1397_f0__ambiguous | colgrep | `fuse_max` | ambiguous | ambiguous | 0.9809 | 1.0000 | 0.0000 | False | 0.9974 | 139 |
| base_33_d7fd7127_t1397_f0__ambiguous | colgrep | `fuse_min` | ambiguous | ambiguous | 0.9378 | 1.0000 | 0.0000 | False | 0.9800 | 139 |
| base_34_5e875b7e_t1637_f0__correct | colgrep | `gte_only` | grounded | correct | 0.9728 | 1.0000 | 0.0000 | False | 1.0000 | 26 |
| base_34_5e875b7e_t1637_f0__correct | colgrep | `code_only` | grounded | correct | 0.9267 | 1.0000 | 0.0000 | False | 1.0000 | 26 |
| base_34_5e875b7e_t1637_f0__correct | colgrep | `fuse_mean` | grounded | correct | 0.9497 | 1.0000 | 0.0000 | False | 1.0000 | 26 |
| base_34_5e875b7e_t1637_f0__correct | colgrep | `fuse_max` | grounded | correct | 0.9728 | 1.0000 | 0.0000 | False | 1.0000 | 26 |
| base_34_5e875b7e_t1637_f0__correct | colgrep | `fuse_min` | grounded | correct | 0.9267 | 1.0000 | 0.0000 | False | 1.0000 | 26 |
| base_34_5e875b7e_t1637_f0__wrong | colgrep | `gte_only` | ungrounded | wrong | 0.9684 | 1.0000 | 0.0000 | False | 1.0000 | 26 |
| base_34_5e875b7e_t1637_f0__wrong | colgrep | `code_only` | ungrounded | wrong | 0.9199 | 1.0000 | 0.0000 | False | 1.0000 | 26 |
| base_34_5e875b7e_t1637_f0__wrong | colgrep | `fuse_mean` | ungrounded | wrong | 0.9442 | 1.0000 | 0.0000 | False | 1.0000 | 26 |
| base_34_5e875b7e_t1637_f0__wrong | colgrep | `fuse_max` | ungrounded | wrong | 0.9684 | 1.0000 | 0.0000 | False | 1.0000 | 26 |
| base_34_5e875b7e_t1637_f0__wrong | colgrep | `fuse_min` | ungrounded | wrong | 0.9199 | 1.0000 | 0.0000 | False | 1.0000 | 26 |
| base_34_5e875b7e_t1637_f0__ambiguous | colgrep | `gte_only` | ambiguous | ambiguous | 0.9724 | 1.0000 | 0.0000 | False | 1.0000 | 26 |
| base_34_5e875b7e_t1637_f0__ambiguous | colgrep | `code_only` | ambiguous | ambiguous | 0.9220 | 1.0000 | 0.0000 | False | 1.0000 | 26 |
| base_34_5e875b7e_t1637_f0__ambiguous | colgrep | `fuse_mean` | ambiguous | ambiguous | 0.9472 | 1.0000 | 0.0000 | False | 1.0000 | 26 |
| base_34_5e875b7e_t1637_f0__ambiguous | colgrep | `fuse_max` | ambiguous | ambiguous | 0.9724 | 1.0000 | 0.0000 | False | 1.0000 | 26 |
| base_34_5e875b7e_t1637_f0__ambiguous | colgrep | `fuse_min` | ambiguous | ambiguous | 0.9220 | 1.0000 | 0.0000 | False | 1.0000 | 26 |
| base_35_a4d14a56_t1005_f0__correct | colgrep | `gte_only` | grounded | correct | 0.9670 | 1.0000 | 0.0000 | False | 0.9988 | 40 |
| base_35_a4d14a56_t1005_f0__correct | colgrep | `code_only` | grounded | correct | 0.9204 | 1.0000 | 0.0000 | False | 0.9860 | 40 |
| base_35_a4d14a56_t1005_f0__correct | colgrep | `fuse_mean` | grounded | correct | 0.9437 | 1.0000 | 0.0000 | False | 0.9924 | 40 |
| base_35_a4d14a56_t1005_f0__correct | colgrep | `fuse_max` | grounded | correct | 0.9670 | 1.0000 | 0.0000 | False | 0.9988 | 40 |
| base_35_a4d14a56_t1005_f0__correct | colgrep | `fuse_min` | grounded | correct | 0.9204 | 1.0000 | 0.0000 | False | 0.9860 | 40 |
| base_35_a4d14a56_t1005_f0__wrong | colgrep | `gte_only` | ungrounded | wrong | 0.9634 | 1.0000 | 0.0000 | False | 0.9975 | 40 |
| base_35_a4d14a56_t1005_f0__wrong | colgrep | `code_only` | ungrounded | wrong | 0.9148 | 1.0000 | 0.0000 | False | 0.9845 | 40 |
| base_35_a4d14a56_t1005_f0__wrong | colgrep | `fuse_mean` | ungrounded | wrong | 0.9391 | 1.0000 | 0.0000 | False | 0.9910 | 40 |
| base_35_a4d14a56_t1005_f0__wrong | colgrep | `fuse_max` | ungrounded | wrong | 0.9634 | 1.0000 | 0.0000 | False | 0.9975 | 40 |
| base_35_a4d14a56_t1005_f0__wrong | colgrep | `fuse_min` | ungrounded | wrong | 0.9148 | 1.0000 | 0.0000 | False | 0.9845 | 40 |
| base_35_a4d14a56_t1005_f0__ambiguous | colgrep | `gte_only` | ambiguous | ambiguous | 0.9666 | 1.0000 | 0.0000 | False | 0.9988 | 40 |
| base_35_a4d14a56_t1005_f0__ambiguous | colgrep | `code_only` | ambiguous | ambiguous | 0.9194 | 1.0000 | 0.0000 | False | 0.9858 | 40 |
| base_35_a4d14a56_t1005_f0__ambiguous | colgrep | `fuse_mean` | ambiguous | ambiguous | 0.9430 | 1.0000 | 0.0000 | False | 0.9923 | 40 |
| base_35_a4d14a56_t1005_f0__ambiguous | colgrep | `fuse_max` | ambiguous | ambiguous | 0.9666 | 1.0000 | 0.0000 | False | 0.9988 | 40 |
| base_35_a4d14a56_t1005_f0__ambiguous | colgrep | `fuse_min` | ambiguous | ambiguous | 0.9194 | 1.0000 | 0.0000 | False | 0.9858 | 40 |
| base_36_a4d14a56_t693_f0__correct | colgrep | `gte_only` | grounded | correct | 0.9823 | 1.0000 | 0.0000 | False | 1.0000 | 97 |
| base_36_a4d14a56_t693_f0__correct | colgrep | `code_only` | grounded | correct | 0.9506 | 1.0000 | 0.0000 | False | 1.0000 | 97 |
| base_36_a4d14a56_t693_f0__correct | colgrep | `fuse_mean` | grounded | correct | 0.9665 | 1.0000 | 0.0000 | False | 1.0000 | 97 |
| base_36_a4d14a56_t693_f0__correct | colgrep | `fuse_max` | grounded | correct | 0.9823 | 1.0000 | 0.0000 | False | 1.0000 | 97 |
| base_36_a4d14a56_t693_f0__correct | colgrep | `fuse_min` | grounded | correct | 0.9506 | 1.0000 | 0.0000 | False | 1.0000 | 97 |
| base_36_a4d14a56_t693_f0__wrong | colgrep | `gte_only` | ungrounded | wrong | 0.9699 | 1.0000 | 0.0000 | False | 1.0000 | 97 |
| base_36_a4d14a56_t693_f0__wrong | colgrep | `code_only` | ungrounded | wrong | 0.9343 | 1.0000 | 0.0000 | False | 1.0000 | 97 |
| base_36_a4d14a56_t693_f0__wrong | colgrep | `fuse_mean` | ungrounded | wrong | 0.9521 | 1.0000 | 0.0000 | False | 1.0000 | 97 |
| base_36_a4d14a56_t693_f0__wrong | colgrep | `fuse_max` | ungrounded | wrong | 0.9699 | 1.0000 | 0.0000 | False | 1.0000 | 97 |
| base_36_a4d14a56_t693_f0__wrong | colgrep | `fuse_min` | ungrounded | wrong | 0.9343 | 1.0000 | 0.0000 | False | 1.0000 | 97 |
| base_36_a4d14a56_t693_f0__ambiguous | colgrep | `gte_only` | ambiguous | ambiguous | 0.9764 | 1.0000 | 0.0000 | False | 1.0000 | 97 |
| base_36_a4d14a56_t693_f0__ambiguous | colgrep | `code_only` | ambiguous | ambiguous | 0.9396 | 1.0000 | 0.0000 | False | 1.0000 | 97 |
| base_36_a4d14a56_t693_f0__ambiguous | colgrep | `fuse_mean` | ambiguous | ambiguous | 0.9580 | 1.0000 | 0.0000 | False | 1.0000 | 97 |
| base_36_a4d14a56_t693_f0__ambiguous | colgrep | `fuse_max` | ambiguous | ambiguous | 0.9764 | 1.0000 | 0.0000 | False | 1.0000 | 97 |
| base_36_a4d14a56_t693_f0__ambiguous | colgrep | `fuse_min` | ambiguous | ambiguous | 0.9396 | 1.0000 | 0.0000 | False | 1.0000 | 97 |
| base_37_126e38fc_t1936_f0__correct | colgrep | `gte_only` | grounded | correct | 0.9818 | 1.0000 | 0.0000 | False | 0.9999 | 195 |
| base_37_126e38fc_t1936_f0__correct | colgrep | `code_only` | grounded | correct | 0.9403 | 1.0000 | 0.0000 | False | 1.0000 | 195 |
| base_37_126e38fc_t1936_f0__correct | colgrep | `fuse_mean` | grounded | correct | 0.9610 | 1.0000 | 0.0000 | False | 0.9999 | 195 |
| base_37_126e38fc_t1936_f0__correct | colgrep | `fuse_max` | grounded | correct | 0.9818 | 1.0000 | 0.0000 | False | 1.0000 | 195 |
| base_37_126e38fc_t1936_f0__correct | colgrep | `fuse_min` | grounded | correct | 0.9403 | 1.0000 | 0.0000 | False | 0.9999 | 195 |
| base_37_126e38fc_t1936_f0__wrong | colgrep | `gte_only` | ungrounded | wrong | 0.9704 | 1.0000 | 0.0000 | False | 1.0000 | 195 |
| base_37_126e38fc_t1936_f0__wrong | colgrep | `code_only` | ungrounded | wrong | 0.9038 | 1.0000 | 0.0000 | False | 1.0000 | 195 |
| base_37_126e38fc_t1936_f0__wrong | colgrep | `fuse_mean` | ungrounded | wrong | 0.9371 | 1.0000 | 0.0000 | False | 1.0000 | 195 |
| base_37_126e38fc_t1936_f0__wrong | colgrep | `fuse_max` | ungrounded | wrong | 0.9704 | 1.0000 | 0.0000 | False | 1.0000 | 195 |
| base_37_126e38fc_t1936_f0__wrong | colgrep | `fuse_min` | ungrounded | wrong | 0.9038 | 1.0000 | 0.0000 | False | 1.0000 | 195 |
| base_37_126e38fc_t1936_f0__ambiguous | colgrep | `gte_only` | ambiguous | ambiguous | 0.9807 | 1.0000 | 0.0000 | False | 0.9999 | 195 |
| base_37_126e38fc_t1936_f0__ambiguous | colgrep | `code_only` | ambiguous | ambiguous | 0.9311 | 1.0000 | 0.0000 | False | 0.9999 | 195 |
| base_37_126e38fc_t1936_f0__ambiguous | colgrep | `fuse_mean` | ambiguous | ambiguous | 0.9559 | 1.0000 | 0.0000 | False | 0.9999 | 195 |
| base_37_126e38fc_t1936_f0__ambiguous | colgrep | `fuse_max` | ambiguous | ambiguous | 0.9807 | 1.0000 | 0.0000 | False | 0.9999 | 195 |
| base_37_126e38fc_t1936_f0__ambiguous | colgrep | `fuse_min` | ambiguous | ambiguous | 0.9311 | 1.0000 | 0.0000 | False | 0.9999 | 195 |
| base_38_a4d14a56_t869_f0__correct | colgrep | `gte_only` | grounded | correct | 0.9687 | 1.0000 | 0.0000 | False | 1.0000 | 40 |
| base_38_a4d14a56_t869_f0__correct | colgrep | `code_only` | grounded | correct | 0.9142 | 1.0000 | 0.0000 | False | 1.0000 | 40 |
| base_38_a4d14a56_t869_f0__correct | colgrep | `fuse_mean` | grounded | correct | 0.9415 | 1.0000 | 0.0000 | False | 1.0000 | 40 |
| base_38_a4d14a56_t869_f0__correct | colgrep | `fuse_max` | grounded | correct | 0.9687 | 1.0000 | 0.0000 | False | 1.0000 | 40 |
| base_38_a4d14a56_t869_f0__correct | colgrep | `fuse_min` | grounded | correct | 0.9142 | 1.0000 | 0.0000 | False | 1.0000 | 40 |
| base_38_a4d14a56_t869_f0__wrong | colgrep | `gte_only` | ungrounded | wrong | 0.9567 | 1.0000 | 0.0000 | False | 1.0000 | 40 |
| base_38_a4d14a56_t869_f0__wrong | colgrep | `code_only` | ungrounded | wrong | 0.9089 | 1.0000 | 0.0000 | False | 1.0000 | 40 |
| base_38_a4d14a56_t869_f0__wrong | colgrep | `fuse_mean` | ungrounded | wrong | 0.9328 | 1.0000 | 0.0000 | False | 1.0000 | 40 |
| base_38_a4d14a56_t869_f0__wrong | colgrep | `fuse_max` | ungrounded | wrong | 0.9567 | 1.0000 | 0.0000 | False | 1.0000 | 40 |
| base_38_a4d14a56_t869_f0__wrong | colgrep | `fuse_min` | ungrounded | wrong | 0.9089 | 1.0000 | 0.0000 | False | 1.0000 | 40 |
| base_38_a4d14a56_t869_f0__ambiguous | colgrep | `gte_only` | ambiguous | ambiguous | 0.9659 | 1.0000 | 0.0000 | False | 1.0000 | 40 |
| base_38_a4d14a56_t869_f0__ambiguous | colgrep | `code_only` | ambiguous | ambiguous | 0.9202 | 1.0000 | 0.0000 | False | 1.0000 | 40 |
| base_38_a4d14a56_t869_f0__ambiguous | colgrep | `fuse_mean` | ambiguous | ambiguous | 0.9430 | 1.0000 | 0.0000 | False | 1.0000 | 40 |
| base_38_a4d14a56_t869_f0__ambiguous | colgrep | `fuse_max` | ambiguous | ambiguous | 0.9659 | 1.0000 | 0.0000 | False | 1.0000 | 40 |
| base_38_a4d14a56_t869_f0__ambiguous | colgrep | `fuse_min` | ambiguous | ambiguous | 0.9202 | 1.0000 | 0.0000 | False | 1.0000 | 40 |
| base_39_5e875b7e_t2278_f0__correct | colgrep | `gte_only` | grounded | correct | 0.9575 | 1.0000 | 0.0000 | False | 0.9947 | 43 |
| base_39_5e875b7e_t2278_f0__correct | colgrep | `code_only` | grounded | correct | 0.9061 | 1.0000 | 0.0000 | False | 0.9848 | 43 |
| base_39_5e875b7e_t2278_f0__correct | colgrep | `fuse_mean` | grounded | correct | 0.9318 | 1.0000 | 0.0000 | False | 0.9897 | 43 |
| base_39_5e875b7e_t2278_f0__correct | colgrep | `fuse_max` | grounded | correct | 0.9575 | 1.0000 | 0.0000 | False | 0.9947 | 43 |
| base_39_5e875b7e_t2278_f0__correct | colgrep | `fuse_min` | grounded | correct | 0.9061 | 1.0000 | 0.0000 | False | 0.9848 | 43 |
| base_39_5e875b7e_t2278_f0__wrong | colgrep | `gte_only` | ungrounded | wrong | 0.9570 | 1.0000 | 0.0000 | False | 0.9948 | 43 |
| base_39_5e875b7e_t2278_f0__wrong | colgrep | `code_only` | ungrounded | wrong | 0.9062 | 1.0000 | 0.0000 | False | 0.9773 | 43 |
| base_39_5e875b7e_t2278_f0__wrong | colgrep | `fuse_mean` | ungrounded | wrong | 0.9316 | 1.0000 | 0.0000 | False | 0.9861 | 43 |
| base_39_5e875b7e_t2278_f0__wrong | colgrep | `fuse_max` | ungrounded | wrong | 0.9570 | 1.0000 | 0.0000 | False | 0.9948 | 43 |
| base_39_5e875b7e_t2278_f0__wrong | colgrep | `fuse_min` | ungrounded | wrong | 0.9062 | 1.0000 | 0.0000 | False | 0.9773 | 43 |
| base_39_5e875b7e_t2278_f0__ambiguous | colgrep | `gte_only` | ambiguous | ambiguous | 0.9575 | 1.0000 | 0.0000 | False | 0.9945 | 43 |
| base_39_5e875b7e_t2278_f0__ambiguous | colgrep | `code_only` | ambiguous | ambiguous | 0.9093 | 1.0000 | 0.0000 | False | 0.9840 | 43 |
| base_39_5e875b7e_t2278_f0__ambiguous | colgrep | `fuse_mean` | ambiguous | ambiguous | 0.9334 | 1.0000 | 0.0000 | False | 0.9893 | 43 |
| base_39_5e875b7e_t2278_f0__ambiguous | colgrep | `fuse_max` | ambiguous | ambiguous | 0.9575 | 1.0000 | 0.0000 | False | 0.9945 | 43 |
| base_39_5e875b7e_t2278_f0__ambiguous | colgrep | `fuse_min` | ambiguous | ambiguous | 0.9093 | 1.0000 | 0.0000 | False | 0.9840 | 43 |
| base_40_ed69b0cd_t395_f0__correct | colgrep | `gte_only` | grounded | correct | 0.9848 | 1.0000 | 0.0000 | False | 1.0000 | 88 |
| base_40_ed69b0cd_t395_f0__correct | colgrep | `code_only` | grounded | correct | 0.9580 | 1.0000 | 0.0000 | False | 0.9999 | 88 |
| base_40_ed69b0cd_t395_f0__correct | colgrep | `fuse_mean` | grounded | correct | 0.9714 | 1.0000 | 0.0000 | False | 0.9999 | 88 |
| base_40_ed69b0cd_t395_f0__correct | colgrep | `fuse_max` | grounded | correct | 0.9848 | 1.0000 | 0.0000 | False | 1.0000 | 88 |
| base_40_ed69b0cd_t395_f0__correct | colgrep | `fuse_min` | grounded | correct | 0.9580 | 1.0000 | 0.0000 | False | 0.9999 | 88 |
| base_40_ed69b0cd_t395_f0__wrong | colgrep | `gte_only` | ungrounded | wrong | 0.9670 | 1.0000 | 0.0000 | False | 1.0000 | 88 |
| base_40_ed69b0cd_t395_f0__wrong | colgrep | `code_only` | ungrounded | wrong | 0.9387 | 1.0000 | 0.0000 | False | 1.0000 | 88 |
| base_40_ed69b0cd_t395_f0__wrong | colgrep | `fuse_mean` | ungrounded | wrong | 0.9528 | 1.0000 | 0.0000 | False | 1.0000 | 88 |
| base_40_ed69b0cd_t395_f0__wrong | colgrep | `fuse_max` | ungrounded | wrong | 0.9670 | 1.0000 | 0.0000 | False | 1.0000 | 88 |
| base_40_ed69b0cd_t395_f0__wrong | colgrep | `fuse_min` | ungrounded | wrong | 0.9387 | 1.0000 | 0.0000 | False | 1.0000 | 88 |
| base_40_ed69b0cd_t395_f0__ambiguous | colgrep | `gte_only` | ambiguous | ambiguous | 0.9843 | 1.0000 | 0.0000 | False | 1.0000 | 88 |
| base_40_ed69b0cd_t395_f0__ambiguous | colgrep | `code_only` | ambiguous | ambiguous | 0.9567 | 1.0000 | 0.0000 | False | 0.9999 | 88 |
| base_40_ed69b0cd_t395_f0__ambiguous | colgrep | `fuse_mean` | ambiguous | ambiguous | 0.9705 | 1.0000 | 0.0000 | False | 0.9999 | 88 |
| base_40_ed69b0cd_t395_f0__ambiguous | colgrep | `fuse_max` | ambiguous | ambiguous | 0.9843 | 1.0000 | 0.0000 | False | 1.0000 | 88 |
| base_40_ed69b0cd_t395_f0__ambiguous | colgrep | `fuse_min` | ambiguous | ambiguous | 0.9567 | 1.0000 | 0.0000 | False | 0.9999 | 88 |
| base_41_a4d14a56_t994_f1__correct | colgrep | `gte_only` | grounded | correct | 0.9789 | 1.0000 | 0.0000 | False | 0.9894 | 40 |
| base_41_a4d14a56_t994_f1__correct | colgrep | `code_only` | grounded | correct | 0.9288 | 1.0000 | 0.0000 | False | 0.9649 | 40 |
| base_41_a4d14a56_t994_f1__correct | colgrep | `fuse_mean` | grounded | correct | 0.9538 | 1.0000 | 0.0000 | False | 0.9772 | 40 |
| base_41_a4d14a56_t994_f1__correct | colgrep | `fuse_max` | grounded | correct | 0.9789 | 1.0000 | 0.0000 | False | 0.9894 | 40 |
| base_41_a4d14a56_t994_f1__correct | colgrep | `fuse_min` | grounded | correct | 0.9288 | 1.0000 | 0.0000 | False | 0.9649 | 40 |
| base_41_a4d14a56_t994_f1__wrong | colgrep | `gte_only` | ungrounded | wrong | 0.9755 | 1.0000 | 0.0000 | False | 0.9966 | 40 |
| base_41_a4d14a56_t994_f1__wrong | colgrep | `code_only` | ungrounded | wrong | 0.9256 | 1.0000 | 0.0000 | False | 0.9635 | 40 |
| base_41_a4d14a56_t994_f1__wrong | colgrep | `fuse_mean` | ungrounded | wrong | 0.9506 | 1.0000 | 0.0000 | False | 0.9800 | 40 |
| base_41_a4d14a56_t994_f1__wrong | colgrep | `fuse_max` | ungrounded | wrong | 0.9755 | 1.0000 | 0.0000 | False | 0.9966 | 40 |
| base_41_a4d14a56_t994_f1__wrong | colgrep | `fuse_min` | ungrounded | wrong | 0.9256 | 1.0000 | 0.0000 | False | 0.9635 | 40 |
| base_41_a4d14a56_t994_f1__ambiguous | colgrep | `gte_only` | ambiguous | ambiguous | 0.9783 | 1.0000 | 0.0000 | False | 0.9888 | 40 |
| base_41_a4d14a56_t994_f1__ambiguous | colgrep | `code_only` | ambiguous | ambiguous | 0.9276 | 1.0000 | 0.0000 | False | 0.9664 | 40 |
| base_41_a4d14a56_t994_f1__ambiguous | colgrep | `fuse_mean` | ambiguous | ambiguous | 0.9530 | 1.0000 | 0.0000 | False | 0.9776 | 40 |
| base_41_a4d14a56_t994_f1__ambiguous | colgrep | `fuse_max` | ambiguous | ambiguous | 0.9783 | 1.0000 | 0.0000 | False | 0.9888 | 40 |
| base_41_a4d14a56_t994_f1__ambiguous | colgrep | `fuse_min` | ambiguous | ambiguous | 0.9276 | 1.0000 | 0.0000 | False | 0.9664 | 40 |
| base_42_a4d14a56_t531_f0__correct | colgrep | `gte_only` | grounded | correct | 0.9789 | 1.0000 | 0.0000 | False | 0.9961 | 79 |
| base_42_a4d14a56_t531_f0__correct | colgrep | `code_only` | grounded | correct | 0.9375 | 1.0000 | 0.0000 | False | 0.9865 | 79 |
| base_42_a4d14a56_t531_f0__correct | colgrep | `fuse_mean` | grounded | correct | 0.9582 | 1.0000 | 0.0000 | False | 0.9913 | 79 |
| base_42_a4d14a56_t531_f0__correct | colgrep | `fuse_max` | grounded | correct | 0.9789 | 1.0000 | 0.0000 | False | 0.9961 | 79 |
| base_42_a4d14a56_t531_f0__correct | colgrep | `fuse_min` | grounded | correct | 0.9375 | 1.0000 | 0.0000 | False | 0.9865 | 79 |
| base_42_a4d14a56_t531_f0__wrong | colgrep | `gte_only` | ungrounded | wrong | 0.9706 | 1.0000 | 0.0000 | False | 0.9963 | 79 |
| base_42_a4d14a56_t531_f0__wrong | colgrep | `code_only` | ungrounded | wrong | 0.9294 | 1.0000 | 0.0000 | False | 0.9766 | 79 |
| base_42_a4d14a56_t531_f0__wrong | colgrep | `fuse_mean` | ungrounded | wrong | 0.9500 | 1.0000 | 0.0000 | False | 0.9864 | 79 |
| base_42_a4d14a56_t531_f0__wrong | colgrep | `fuse_max` | ungrounded | wrong | 0.9706 | 1.0000 | 0.0000 | False | 0.9963 | 79 |
| base_42_a4d14a56_t531_f0__wrong | colgrep | `fuse_min` | ungrounded | wrong | 0.9294 | 1.0000 | 0.0000 | False | 0.9766 | 79 |
| base_42_a4d14a56_t531_f0__ambiguous | colgrep | `gte_only` | ambiguous | ambiguous | 0.9784 | 1.0000 | 0.0000 | False | 0.9963 | 79 |
| base_42_a4d14a56_t531_f0__ambiguous | colgrep | `code_only` | ambiguous | ambiguous | 0.9383 | 1.0000 | 0.0000 | False | 0.9871 | 79 |
| base_42_a4d14a56_t531_f0__ambiguous | colgrep | `fuse_mean` | ambiguous | ambiguous | 0.9583 | 1.0000 | 0.0000 | False | 0.9917 | 79 |
| base_42_a4d14a56_t531_f0__ambiguous | colgrep | `fuse_max` | ambiguous | ambiguous | 0.9784 | 1.0000 | 0.0000 | False | 0.9963 | 79 |
| base_42_a4d14a56_t531_f0__ambiguous | colgrep | `fuse_min` | ambiguous | ambiguous | 0.9383 | 1.0000 | 0.0000 | False | 0.9871 | 79 |
| base_43_ed69b0cd_t364_f0__correct | colgrep | `gte_only` | grounded | correct | 0.9822 | 1.0000 | 0.0000 | False | 0.9970 | 88 |
| base_43_ed69b0cd_t364_f0__correct | colgrep | `code_only` | grounded | correct | 0.9473 | 1.0000 | 0.0000 | False | 0.9889 | 88 |
| base_43_ed69b0cd_t364_f0__correct | colgrep | `fuse_mean` | grounded | correct | 0.9648 | 1.0000 | 0.0000 | False | 0.9929 | 88 |
| base_43_ed69b0cd_t364_f0__correct | colgrep | `fuse_max` | grounded | correct | 0.9822 | 1.0000 | 0.0000 | False | 0.9970 | 88 |
| base_43_ed69b0cd_t364_f0__correct | colgrep | `fuse_min` | grounded | correct | 0.9473 | 1.0000 | 0.0000 | False | 0.9889 | 88 |
| base_43_ed69b0cd_t364_f0__wrong | colgrep | `gte_only` | ungrounded | wrong | 0.9728 | 1.0000 | 0.0000 | False | 0.9982 | 88 |
| base_43_ed69b0cd_t364_f0__wrong | colgrep | `code_only` | ungrounded | wrong | 0.9325 | 1.0000 | 0.0000 | False | 0.9782 | 88 |
| base_43_ed69b0cd_t364_f0__wrong | colgrep | `fuse_mean` | ungrounded | wrong | 0.9527 | 1.0000 | 0.0000 | False | 0.9882 | 88 |
| base_43_ed69b0cd_t364_f0__wrong | colgrep | `fuse_max` | ungrounded | wrong | 0.9728 | 1.0000 | 0.0000 | False | 0.9982 | 88 |
| base_43_ed69b0cd_t364_f0__wrong | colgrep | `fuse_min` | ungrounded | wrong | 0.9325 | 1.0000 | 0.0000 | False | 0.9782 | 88 |
| base_43_ed69b0cd_t364_f0__ambiguous | colgrep | `gte_only` | ambiguous | ambiguous | 0.9760 | 1.0000 | 0.0000 | False | 0.9936 | 88 |
| base_43_ed69b0cd_t364_f0__ambiguous | colgrep | `code_only` | ambiguous | ambiguous | 0.9324 | 1.0000 | 0.0000 | False | 0.9684 | 88 |
| base_43_ed69b0cd_t364_f0__ambiguous | colgrep | `fuse_mean` | ambiguous | ambiguous | 0.9542 | 1.0000 | 0.0000 | False | 0.9810 | 88 |
| base_43_ed69b0cd_t364_f0__ambiguous | colgrep | `fuse_max` | ambiguous | ambiguous | 0.9760 | 1.0000 | 0.0000 | False | 0.9936 | 88 |
| base_43_ed69b0cd_t364_f0__ambiguous | colgrep | `fuse_min` | ambiguous | ambiguous | 0.9324 | 1.0000 | 0.0000 | False | 0.9684 | 88 |
| base_44_a4d14a56_t568_f0__correct | colgrep | `gte_only` | grounded | correct | 0.9816 | 1.0000 | 0.0000 | False | 0.9977 | 80 |
| base_44_a4d14a56_t568_f0__correct | colgrep | `code_only` | grounded | correct | 0.9379 | 1.0000 | 0.0000 | False | 0.9828 | 80 |
| base_44_a4d14a56_t568_f0__correct | colgrep | `fuse_mean` | grounded | correct | 0.9598 | 1.0000 | 0.0000 | False | 0.9903 | 80 |
| base_44_a4d14a56_t568_f0__correct | colgrep | `fuse_max` | grounded | correct | 0.9816 | 1.0000 | 0.0000 | False | 0.9977 | 80 |
| base_44_a4d14a56_t568_f0__correct | colgrep | `fuse_min` | grounded | correct | 0.9379 | 1.0000 | 0.0000 | False | 0.9828 | 80 |
| base_44_a4d14a56_t568_f0__wrong | colgrep | `gte_only` | ungrounded | wrong | 0.9657 | 1.0000 | 0.0000 | False | 0.9987 | 80 |
| base_44_a4d14a56_t568_f0__wrong | colgrep | `code_only` | ungrounded | wrong | 0.9239 | 1.0000 | 0.0000 | False | 0.9824 | 80 |
| base_44_a4d14a56_t568_f0__wrong | colgrep | `fuse_mean` | ungrounded | wrong | 0.9448 | 1.0000 | 0.0000 | False | 0.9905 | 80 |
| base_44_a4d14a56_t568_f0__wrong | colgrep | `fuse_max` | ungrounded | wrong | 0.9657 | 1.0000 | 0.0000 | False | 0.9987 | 80 |
| base_44_a4d14a56_t568_f0__wrong | colgrep | `fuse_min` | ungrounded | wrong | 0.9239 | 1.0000 | 0.0000 | False | 0.9824 | 80 |
| base_44_a4d14a56_t568_f0__ambiguous | colgrep | `gte_only` | ambiguous | ambiguous | 0.9796 | 1.0000 | 0.0000 | False | 0.9977 | 80 |
| base_44_a4d14a56_t568_f0__ambiguous | colgrep | `code_only` | ambiguous | ambiguous | 0.9390 | 1.0000 | 0.0000 | False | 0.9806 | 80 |
| base_44_a4d14a56_t568_f0__ambiguous | colgrep | `fuse_mean` | ambiguous | ambiguous | 0.9593 | 1.0000 | 0.0000 | False | 0.9891 | 80 |
| base_44_a4d14a56_t568_f0__ambiguous | colgrep | `fuse_max` | ambiguous | ambiguous | 0.9796 | 1.0000 | 0.0000 | False | 0.9977 | 80 |
| base_44_a4d14a56_t568_f0__ambiguous | colgrep | `fuse_min` | ambiguous | ambiguous | 0.9390 | 1.0000 | 0.0000 | False | 0.9806 | 80 |
| base_45_ed69b0cd_t367_f0__correct | colgrep | `gte_only` | grounded | correct | 0.9806 | 1.0000 | 0.0000 | False | 1.0000 | 88 |
| base_45_ed69b0cd_t367_f0__correct | colgrep | `code_only` | grounded | correct | 0.9334 | 1.0000 | 0.0000 | False | 1.0000 | 88 |
| base_45_ed69b0cd_t367_f0__correct | colgrep | `fuse_mean` | grounded | correct | 0.9570 | 1.0000 | 0.0000 | False | 1.0000 | 88 |
| base_45_ed69b0cd_t367_f0__correct | colgrep | `fuse_max` | grounded | correct | 0.9806 | 1.0000 | 0.0000 | False | 1.0000 | 88 |
| base_45_ed69b0cd_t367_f0__correct | colgrep | `fuse_min` | grounded | correct | 0.9334 | 1.0000 | 0.0000 | False | 1.0000 | 88 |
| base_45_ed69b0cd_t367_f0__wrong | colgrep | `gte_only` | ungrounded | wrong | 0.9735 | 1.0000 | 0.0000 | False | 1.0000 | 88 |
| base_45_ed69b0cd_t367_f0__wrong | colgrep | `code_only` | ungrounded | wrong | 0.9213 | 1.0000 | 0.0000 | False | 1.0000 | 88 |
| base_45_ed69b0cd_t367_f0__wrong | colgrep | `fuse_mean` | ungrounded | wrong | 0.9474 | 1.0000 | 0.0000 | False | 1.0000 | 88 |
| base_45_ed69b0cd_t367_f0__wrong | colgrep | `fuse_max` | ungrounded | wrong | 0.9735 | 1.0000 | 0.0000 | False | 1.0000 | 88 |
| base_45_ed69b0cd_t367_f0__wrong | colgrep | `fuse_min` | ungrounded | wrong | 0.9213 | 1.0000 | 0.0000 | False | 1.0000 | 88 |
| base_45_ed69b0cd_t367_f0__ambiguous | colgrep | `gte_only` | ambiguous | ambiguous | 0.9795 | 1.0000 | 0.0000 | False | 1.0000 | 88 |
| base_45_ed69b0cd_t367_f0__ambiguous | colgrep | `code_only` | ambiguous | ambiguous | 0.9314 | 1.0000 | 0.0000 | False | 1.0000 | 88 |
| base_45_ed69b0cd_t367_f0__ambiguous | colgrep | `fuse_mean` | ambiguous | ambiguous | 0.9555 | 1.0000 | 0.0000 | False | 1.0000 | 88 |
| base_45_ed69b0cd_t367_f0__ambiguous | colgrep | `fuse_max` | ambiguous | ambiguous | 0.9795 | 1.0000 | 0.0000 | False | 1.0000 | 88 |
| base_45_ed69b0cd_t367_f0__ambiguous | colgrep | `fuse_min` | ambiguous | ambiguous | 0.9314 | 1.0000 | 0.0000 | False | 1.0000 | 88 |
| base_46_d7fd7127_t317_f0__correct | colgrep | `gte_only` | grounded | correct | 0.9755 | 1.0000 | 0.0000 | False | 0.9940 | 44 |
| base_46_d7fd7127_t317_f0__correct | colgrep | `code_only` | grounded | correct | 0.9378 | 1.0000 | 0.0000 | False | 0.9887 | 44 |
| base_46_d7fd7127_t317_f0__correct | colgrep | `fuse_mean` | grounded | correct | 0.9567 | 1.0000 | 0.0000 | False | 0.9914 | 44 |
| base_46_d7fd7127_t317_f0__correct | colgrep | `fuse_max` | grounded | correct | 0.9755 | 1.0000 | 0.0000 | False | 0.9940 | 44 |
| base_46_d7fd7127_t317_f0__correct | colgrep | `fuse_min` | grounded | correct | 0.9378 | 1.0000 | 0.0000 | False | 0.9887 | 44 |
| base_46_d7fd7127_t317_f0__wrong | colgrep | `gte_only` | ungrounded | wrong | 0.9662 | 1.0000 | 0.0000 | False | 0.9908 | 44 |
| base_46_d7fd7127_t317_f0__wrong | colgrep | `code_only` | ungrounded | wrong | 0.9289 | 1.0000 | 0.0000 | False | 0.9840 | 44 |
| base_46_d7fd7127_t317_f0__wrong | colgrep | `fuse_mean` | ungrounded | wrong | 0.9475 | 1.0000 | 0.0000 | False | 0.9874 | 44 |
| base_46_d7fd7127_t317_f0__wrong | colgrep | `fuse_max` | ungrounded | wrong | 0.9662 | 1.0000 | 0.0000 | False | 0.9908 | 44 |
| base_46_d7fd7127_t317_f0__wrong | colgrep | `fuse_min` | ungrounded | wrong | 0.9289 | 1.0000 | 0.0000 | False | 0.9840 | 44 |
| base_46_d7fd7127_t317_f0__ambiguous | colgrep | `gte_only` | ambiguous | ambiguous | 0.9740 | 1.0000 | 0.0000 | False | 0.9939 | 44 |
| base_46_d7fd7127_t317_f0__ambiguous | colgrep | `code_only` | ambiguous | ambiguous | 0.9379 | 1.0000 | 0.0000 | False | 0.9895 | 44 |
| base_46_d7fd7127_t317_f0__ambiguous | colgrep | `fuse_mean` | ambiguous | ambiguous | 0.9560 | 1.0000 | 0.0000 | False | 0.9917 | 44 |
| base_46_d7fd7127_t317_f0__ambiguous | colgrep | `fuse_max` | ambiguous | ambiguous | 0.9740 | 1.0000 | 0.0000 | False | 0.9939 | 44 |
| base_46_d7fd7127_t317_f0__ambiguous | colgrep | `fuse_min` | ambiguous | ambiguous | 0.9379 | 1.0000 | 0.0000 | False | 0.9895 | 44 |
| base_47_0b1422bb_t847_f0__correct | colgrep | `gte_only` | grounded | correct | 0.9497 | 1.0000 | 0.0000 | False | 0.9987 | 41 |
| base_47_0b1422bb_t847_f0__correct | colgrep | `code_only` | grounded | correct | 0.8875 | 1.0000 | 0.0000 | False | 0.9787 | 41 |
| base_47_0b1422bb_t847_f0__correct | colgrep | `fuse_mean` | grounded | correct | 0.9186 | 1.0000 | 0.0000 | False | 0.9887 | 41 |
| base_47_0b1422bb_t847_f0__correct | colgrep | `fuse_max` | grounded | correct | 0.9497 | 1.0000 | 0.0000 | False | 0.9987 | 41 |
| base_47_0b1422bb_t847_f0__correct | colgrep | `fuse_min` | grounded | correct | 0.8875 | 1.0000 | 0.0000 | False | 0.9787 | 41 |
| base_47_0b1422bb_t847_f0__wrong | colgrep | `gte_only` | ungrounded | wrong | 0.9454 | 1.0000 | 0.0000 | False | 0.9934 | 41 |
| base_47_0b1422bb_t847_f0__wrong | colgrep | `code_only` | ungrounded | wrong | 0.8965 | 1.0000 | 0.0000 | False | 0.9780 | 41 |
| base_47_0b1422bb_t847_f0__wrong | colgrep | `fuse_mean` | ungrounded | wrong | 0.9210 | 1.0000 | 0.0000 | False | 0.9857 | 41 |
| base_47_0b1422bb_t847_f0__wrong | colgrep | `fuse_max` | ungrounded | wrong | 0.9454 | 1.0000 | 0.0000 | False | 0.9934 | 41 |
| base_47_0b1422bb_t847_f0__wrong | colgrep | `fuse_min` | ungrounded | wrong | 0.8965 | 1.0000 | 0.0000 | False | 0.9780 | 41 |
| base_47_0b1422bb_t847_f0__ambiguous | colgrep | `gte_only` | ambiguous | ambiguous | 0.9494 | 1.0000 | 0.0000 | False | 0.9997 | 41 |
| base_47_0b1422bb_t847_f0__ambiguous | colgrep | `code_only` | ambiguous | ambiguous | 0.8874 | 1.0000 | 0.0000 | False | 0.9783 | 41 |
| base_47_0b1422bb_t847_f0__ambiguous | colgrep | `fuse_mean` | ambiguous | ambiguous | 0.9184 | 1.0000 | 0.0000 | False | 0.9890 | 41 |
| base_47_0b1422bb_t847_f0__ambiguous | colgrep | `fuse_max` | ambiguous | ambiguous | 0.9494 | 1.0000 | 0.0000 | False | 0.9997 | 41 |
| base_47_0b1422bb_t847_f0__ambiguous | colgrep | `fuse_min` | ambiguous | ambiguous | 0.8874 | 1.0000 | 0.0000 | False | 0.9783 | 41 |
| base_48_ed69b0cd_t10_f2__correct | colgrep | `gte_only` | grounded | correct | 0.9632 | 1.0000 | 0.0000 | False | 0.9891 | 11 |
| base_48_ed69b0cd_t10_f2__correct | colgrep | `code_only` | grounded | correct | 0.9180 | 1.0000 | 0.0000 | False | 0.9849 | 11 |
| base_48_ed69b0cd_t10_f2__correct | colgrep | `fuse_mean` | grounded | correct | 0.9406 | 1.0000 | 0.0000 | False | 0.9870 | 11 |
| base_48_ed69b0cd_t10_f2__correct | colgrep | `fuse_max` | grounded | correct | 0.9632 | 1.0000 | 0.0000 | False | 0.9891 | 11 |
| base_48_ed69b0cd_t10_f2__correct | colgrep | `fuse_min` | grounded | correct | 0.9180 | 1.0000 | 0.0000 | False | 0.9849 | 11 |
| base_48_ed69b0cd_t10_f2__wrong | colgrep | `gte_only` | ungrounded | wrong | 0.9516 | 1.0000 | 0.0000 | False | 0.9901 | 11 |
| base_48_ed69b0cd_t10_f2__wrong | colgrep | `code_only` | ungrounded | wrong | 0.9042 | 1.0000 | 0.0000 | False | 0.9772 | 11 |
| base_48_ed69b0cd_t10_f2__wrong | colgrep | `fuse_mean` | ungrounded | wrong | 0.9279 | 1.0000 | 0.0000 | False | 0.9837 | 11 |
| base_48_ed69b0cd_t10_f2__wrong | colgrep | `fuse_max` | ungrounded | wrong | 0.9516 | 1.0000 | 0.0000 | False | 0.9901 | 11 |
| base_48_ed69b0cd_t10_f2__wrong | colgrep | `fuse_min` | ungrounded | wrong | 0.9042 | 1.0000 | 0.0000 | False | 0.9772 | 11 |
| base_48_ed69b0cd_t10_f2__ambiguous | colgrep | `gte_only` | ambiguous | ambiguous | 0.9630 | 1.0000 | 0.0000 | False | 0.9892 | 11 |
| base_48_ed69b0cd_t10_f2__ambiguous | colgrep | `code_only` | ambiguous | ambiguous | 0.9185 | 1.0000 | 0.0000 | False | 0.9846 | 11 |
| base_48_ed69b0cd_t10_f2__ambiguous | colgrep | `fuse_mean` | ambiguous | ambiguous | 0.9408 | 1.0000 | 0.0000 | False | 0.9869 | 11 |
| base_48_ed69b0cd_t10_f2__ambiguous | colgrep | `fuse_max` | ambiguous | ambiguous | 0.9630 | 1.0000 | 0.0000 | False | 0.9892 | 11 |
| base_48_ed69b0cd_t10_f2__ambiguous | colgrep | `fuse_min` | ambiguous | ambiguous | 0.9185 | 1.0000 | 0.0000 | False | 0.9846 | 11 |
| base_49_fa052757_t213_f0__correct | colgrep | `gte_only` | grounded | correct | 0.9714 | 1.0000 | 0.0000 | False | 0.9989 | 57 |
| base_49_fa052757_t213_f0__correct | colgrep | `code_only` | grounded | correct | 0.9138 | 1.0000 | 0.0000 | False | 0.9809 | 57 |
| base_49_fa052757_t213_f0__correct | colgrep | `fuse_mean` | grounded | correct | 0.9426 | 1.0000 | 0.0000 | False | 0.9899 | 57 |
| base_49_fa052757_t213_f0__correct | colgrep | `fuse_max` | grounded | correct | 0.9714 | 1.0000 | 0.0000 | False | 0.9989 | 57 |
| base_49_fa052757_t213_f0__correct | colgrep | `fuse_min` | grounded | correct | 0.9138 | 1.0000 | 0.0000 | False | 0.9809 | 57 |
| base_49_fa052757_t213_f0__wrong | colgrep | `gte_only` | ungrounded | wrong | 0.9614 | 1.0000 | 0.0000 | False | 0.9990 | 57 |
| base_49_fa052757_t213_f0__wrong | colgrep | `code_only` | ungrounded | wrong | 0.9043 | 1.0000 | 0.0000 | False | 0.9682 | 57 |
| base_49_fa052757_t213_f0__wrong | colgrep | `fuse_mean` | ungrounded | wrong | 0.9328 | 1.0000 | 0.0000 | False | 0.9836 | 57 |
| base_49_fa052757_t213_f0__wrong | colgrep | `fuse_max` | ungrounded | wrong | 0.9614 | 1.0000 | 0.0000 | False | 0.9990 | 57 |
| base_49_fa052757_t213_f0__wrong | colgrep | `fuse_min` | ungrounded | wrong | 0.9043 | 1.0000 | 0.0000 | False | 0.9682 | 57 |
| base_49_fa052757_t213_f0__ambiguous | colgrep | `gte_only` | ambiguous | ambiguous | 0.9720 | 1.0000 | 0.0000 | False | 0.9989 | 57 |
| base_49_fa052757_t213_f0__ambiguous | colgrep | `code_only` | ambiguous | ambiguous | 0.9153 | 1.0000 | 0.0000 | False | 0.9677 | 57 |
| base_49_fa052757_t213_f0__ambiguous | colgrep | `fuse_mean` | ambiguous | ambiguous | 0.9437 | 1.0000 | 0.0000 | False | 0.9833 | 57 |
| base_49_fa052757_t213_f0__ambiguous | colgrep | `fuse_max` | ambiguous | ambiguous | 0.9720 | 1.0000 | 0.0000 | False | 0.9989 | 57 |
| base_49_fa052757_t213_f0__ambiguous | colgrep | `fuse_min` | ambiguous | ambiguous | 0.9153 | 1.0000 | 0.0000 | False | 0.9677 | 57 |
| base_50_0b1422bb_t839_f0__correct | colgrep | `gte_only` | grounded | correct | 0.9603 | 1.0000 | 0.0000 | False | 0.9999 | 41 |
| base_50_0b1422bb_t839_f0__correct | colgrep | `code_only` | grounded | correct | 0.8956 | 1.0000 | 0.0000 | False | 0.9836 | 41 |
| base_50_0b1422bb_t839_f0__correct | colgrep | `fuse_mean` | grounded | correct | 0.9279 | 1.0000 | 0.0000 | False | 0.9917 | 41 |
| base_50_0b1422bb_t839_f0__correct | colgrep | `fuse_max` | grounded | correct | 0.9603 | 1.0000 | 0.0000 | False | 0.9999 | 41 |
| base_50_0b1422bb_t839_f0__correct | colgrep | `fuse_min` | grounded | correct | 0.8956 | 1.0000 | 0.0000 | False | 0.9836 | 41 |
| base_50_0b1422bb_t839_f0__wrong | colgrep | `gte_only` | ungrounded | wrong | 0.9568 | 1.0000 | 0.0000 | False | 0.9999 | 41 |
| base_50_0b1422bb_t839_f0__wrong | colgrep | `code_only` | ungrounded | wrong | 0.9025 | 1.0000 | 0.0000 | False | 0.9907 | 41 |
| base_50_0b1422bb_t839_f0__wrong | colgrep | `fuse_mean` | ungrounded | wrong | 0.9296 | 1.0000 | 0.0000 | False | 0.9953 | 41 |
| base_50_0b1422bb_t839_f0__wrong | colgrep | `fuse_max` | ungrounded | wrong | 0.9568 | 1.0000 | 0.0000 | False | 0.9999 | 41 |
| base_50_0b1422bb_t839_f0__wrong | colgrep | `fuse_min` | ungrounded | wrong | 0.9025 | 1.0000 | 0.0000 | False | 0.9907 | 41 |
| base_50_0b1422bb_t839_f0__ambiguous | colgrep | `gte_only` | ambiguous | ambiguous | 0.9578 | 1.0000 | 0.0000 | False | 0.9999 | 41 |
| base_50_0b1422bb_t839_f0__ambiguous | colgrep | `code_only` | ambiguous | ambiguous | 0.9033 | 1.0000 | 0.0000 | False | 0.9834 | 41 |
| base_50_0b1422bb_t839_f0__ambiguous | colgrep | `fuse_mean` | ambiguous | ambiguous | 0.9305 | 1.0000 | 0.0000 | False | 0.9916 | 41 |
| base_50_0b1422bb_t839_f0__ambiguous | colgrep | `fuse_max` | ambiguous | ambiguous | 0.9578 | 1.0000 | 0.0000 | False | 0.9999 | 41 |
| base_50_0b1422bb_t839_f0__ambiguous | colgrep | `fuse_min` | ambiguous | ambiguous | 0.9033 | 1.0000 | 0.0000 | False | 0.9834 | 41 |
| base_51_5e875b7e_t1929_f0__correct | colgrep | `gte_only` | grounded | correct | 0.9610 | 1.0000 | 0.0000 | False | 0.9999 | 28 |
| base_51_5e875b7e_t1929_f0__correct | colgrep | `code_only` | grounded | correct | 0.9118 | 1.0000 | 0.0000 | False | 1.0000 | 28 |
| base_51_5e875b7e_t1929_f0__correct | colgrep | `fuse_mean` | grounded | correct | 0.9364 | 1.0000 | 0.0000 | False | 0.9999 | 28 |
| base_51_5e875b7e_t1929_f0__correct | colgrep | `fuse_max` | grounded | correct | 0.9610 | 1.0000 | 0.0000 | False | 1.0000 | 28 |
| base_51_5e875b7e_t1929_f0__correct | colgrep | `fuse_min` | grounded | correct | 0.9118 | 1.0000 | 0.0000 | False | 0.9999 | 28 |
| base_51_5e875b7e_t1929_f0__wrong | colgrep | `gte_only` | ungrounded | wrong | 0.9506 | 1.0000 | 0.0000 | False | 0.9999 | 28 |
| base_51_5e875b7e_t1929_f0__wrong | colgrep | `code_only` | ungrounded | wrong | 0.8931 | 1.0000 | 0.0000 | False | 1.0000 | 28 |
| base_51_5e875b7e_t1929_f0__wrong | colgrep | `fuse_mean` | ungrounded | wrong | 0.9219 | 1.0000 | 0.0000 | False | 1.0000 | 28 |
| base_51_5e875b7e_t1929_f0__wrong | colgrep | `fuse_max` | ungrounded | wrong | 0.9506 | 1.0000 | 0.0000 | False | 1.0000 | 28 |
| base_51_5e875b7e_t1929_f0__wrong | colgrep | `fuse_min` | ungrounded | wrong | 0.8931 | 1.0000 | 0.0000 | False | 0.9999 | 28 |
| base_51_5e875b7e_t1929_f0__ambiguous | colgrep | `gte_only` | ambiguous | ambiguous | 0.9597 | 1.0000 | 0.0000 | False | 0.9999 | 28 |
| base_51_5e875b7e_t1929_f0__ambiguous | colgrep | `code_only` | ambiguous | ambiguous | 0.9066 | 1.0000 | 0.0000 | False | 1.0000 | 28 |
| base_51_5e875b7e_t1929_f0__ambiguous | colgrep | `fuse_mean` | ambiguous | ambiguous | 0.9331 | 1.0000 | 0.0000 | False | 1.0000 | 28 |
| base_51_5e875b7e_t1929_f0__ambiguous | colgrep | `fuse_max` | ambiguous | ambiguous | 0.9597 | 1.0000 | 0.0000 | False | 1.0000 | 28 |
| base_51_5e875b7e_t1929_f0__ambiguous | colgrep | `fuse_min` | ambiguous | ambiguous | 0.9066 | 1.0000 | 0.0000 | False | 0.9999 | 28 |
| base_52_7b7fb621_t310_f3__correct | colgrep | `gte_only` | grounded | correct | 0.9838 | 1.0000 | 0.0000 | False | 0.9989 | 45 |
| base_52_7b7fb621_t310_f3__correct | colgrep | `code_only` | grounded | correct | 0.9520 | 1.0000 | 0.0000 | False | 0.9934 | 45 |
| base_52_7b7fb621_t310_f3__correct | colgrep | `fuse_mean` | grounded | correct | 0.9679 | 1.0000 | 0.0000 | False | 0.9961 | 45 |
| base_52_7b7fb621_t310_f3__correct | colgrep | `fuse_max` | grounded | correct | 0.9838 | 1.0000 | 0.0000 | False | 0.9989 | 45 |
| base_52_7b7fb621_t310_f3__correct | colgrep | `fuse_min` | grounded | correct | 0.9520 | 1.0000 | 0.0000 | False | 0.9934 | 45 |
| base_52_7b7fb621_t310_f3__wrong | colgrep | `gte_only` | ungrounded | wrong | 0.9756 | 1.0000 | 0.0000 | False | 0.9985 | 45 |
| base_52_7b7fb621_t310_f3__wrong | colgrep | `code_only` | ungrounded | wrong | 0.9389 | 1.0000 | 0.0000 | False | 0.9926 | 45 |
| base_52_7b7fb621_t310_f3__wrong | colgrep | `fuse_mean` | ungrounded | wrong | 0.9573 | 1.0000 | 0.0000 | False | 0.9956 | 45 |
| base_52_7b7fb621_t310_f3__wrong | colgrep | `fuse_max` | ungrounded | wrong | 0.9756 | 1.0000 | 0.0000 | False | 0.9985 | 45 |
| base_52_7b7fb621_t310_f3__wrong | colgrep | `fuse_min` | ungrounded | wrong | 0.9389 | 1.0000 | 0.0000 | False | 0.9926 | 45 |
| base_52_7b7fb621_t310_f3__ambiguous | colgrep | `gte_only` | ambiguous | ambiguous | 0.9841 | 1.0000 | 0.0000 | False | 1.0000 | 45 |
| base_52_7b7fb621_t310_f3__ambiguous | colgrep | `code_only` | ambiguous | ambiguous | 0.9467 | 1.0000 | 0.0000 | False | 1.0000 | 45 |
| base_52_7b7fb621_t310_f3__ambiguous | colgrep | `fuse_mean` | ambiguous | ambiguous | 0.9654 | 1.0000 | 0.0000 | False | 1.0000 | 45 |
| base_52_7b7fb621_t310_f3__ambiguous | colgrep | `fuse_max` | ambiguous | ambiguous | 0.9841 | 1.0000 | 0.0000 | False | 1.0000 | 45 |
| base_52_7b7fb621_t310_f3__ambiguous | colgrep | `fuse_min` | ambiguous | ambiguous | 0.9467 | 1.0000 | 0.0000 | False | 1.0000 | 45 |
| base_53_7b7fb621_t347_f0__correct | colgrep | `gte_only` | grounded | correct | 0.9782 | 1.0000 | 0.0000 | False | 0.9988 | 46 |
| base_53_7b7fb621_t347_f0__correct | colgrep | `code_only` | grounded | correct | 0.9281 | 1.0000 | 0.0000 | False | 1.0000 | 46 |
| base_53_7b7fb621_t347_f0__correct | colgrep | `fuse_mean` | grounded | correct | 0.9531 | 1.0000 | 0.0000 | False | 0.9994 | 46 |
| base_53_7b7fb621_t347_f0__correct | colgrep | `fuse_max` | grounded | correct | 0.9782 | 1.0000 | 0.0000 | False | 1.0000 | 46 |
| base_53_7b7fb621_t347_f0__correct | colgrep | `fuse_min` | grounded | correct | 0.9281 | 1.0000 | 0.0000 | False | 0.9988 | 46 |
| base_53_7b7fb621_t347_f0__wrong | colgrep | `gte_only` | ungrounded | wrong | 0.9737 | 1.0000 | 0.0000 | False | 0.9988 | 46 |
| base_53_7b7fb621_t347_f0__wrong | colgrep | `code_only` | ungrounded | wrong | 0.9299 | 1.0000 | 0.0000 | False | 1.0000 | 46 |
| base_53_7b7fb621_t347_f0__wrong | colgrep | `fuse_mean` | ungrounded | wrong | 0.9518 | 1.0000 | 0.0000 | False | 0.9994 | 46 |
| base_53_7b7fb621_t347_f0__wrong | colgrep | `fuse_max` | ungrounded | wrong | 0.9737 | 1.0000 | 0.0000 | False | 1.0000 | 46 |
| base_53_7b7fb621_t347_f0__wrong | colgrep | `fuse_min` | ungrounded | wrong | 0.9299 | 1.0000 | 0.0000 | False | 0.9988 | 46 |
| base_53_7b7fb621_t347_f0__ambiguous | colgrep | `gte_only` | ambiguous | ambiguous | 0.9785 | 1.0000 | 0.0000 | False | 0.9991 | 46 |
| base_53_7b7fb621_t347_f0__ambiguous | colgrep | `code_only` | ambiguous | ambiguous | 0.9329 | 1.0000 | 0.0000 | False | 1.0000 | 46 |
| base_53_7b7fb621_t347_f0__ambiguous | colgrep | `fuse_mean` | ambiguous | ambiguous | 0.9557 | 1.0000 | 0.0000 | False | 0.9995 | 46 |
| base_53_7b7fb621_t347_f0__ambiguous | colgrep | `fuse_max` | ambiguous | ambiguous | 0.9785 | 1.0000 | 0.0000 | False | 1.0000 | 46 |
| base_53_7b7fb621_t347_f0__ambiguous | colgrep | `fuse_min` | ambiguous | ambiguous | 0.9329 | 1.0000 | 0.0000 | False | 0.9991 | 46 |
| base_54_5e875b7e_t630_f0__correct | colgrep | `gte_only` | grounded | correct | 0.9659 | 1.0000 | 0.0000 | False | 0.9913 | 19 |
| base_54_5e875b7e_t630_f0__correct | colgrep | `code_only` | grounded | correct | 0.8796 | 1.0000 | 0.0000 | False | 0.9676 | 19 |
| base_54_5e875b7e_t630_f0__correct | colgrep | `fuse_mean` | grounded | correct | 0.9228 | 1.0000 | 0.0000 | False | 0.9795 | 19 |
| base_54_5e875b7e_t630_f0__correct | colgrep | `fuse_max` | grounded | correct | 0.9659 | 1.0000 | 0.0000 | False | 0.9913 | 19 |
| base_54_5e875b7e_t630_f0__correct | colgrep | `fuse_min` | grounded | correct | 0.8796 | 1.0000 | 0.0000 | False | 0.9676 | 19 |
| base_54_5e875b7e_t630_f0__wrong | colgrep | `gte_only` | ungrounded | wrong | 0.9642 | 1.0000 | 0.0000 | False | 0.9944 | 19 |
| base_54_5e875b7e_t630_f0__wrong | colgrep | `code_only` | ungrounded | wrong | 0.8767 | 1.0000 | 0.0000 | False | 1.0000 | 19 |
| base_54_5e875b7e_t630_f0__wrong | colgrep | `fuse_mean` | ungrounded | wrong | 0.9204 | 1.0000 | 0.0000 | False | 0.9972 | 19 |
| base_54_5e875b7e_t630_f0__wrong | colgrep | `fuse_max` | ungrounded | wrong | 0.9642 | 1.0000 | 0.0000 | False | 1.0000 | 19 |
| base_54_5e875b7e_t630_f0__wrong | colgrep | `fuse_min` | ungrounded | wrong | 0.8767 | 1.0000 | 0.0000 | False | 0.9944 | 19 |
| base_54_5e875b7e_t630_f0__ambiguous | colgrep | `gte_only` | ambiguous | ambiguous | 0.9650 | 1.0000 | 0.0000 | False | 0.9910 | 19 |
| base_54_5e875b7e_t630_f0__ambiguous | colgrep | `code_only` | ambiguous | ambiguous | 0.8836 | 1.0000 | 0.0000 | False | 0.9678 | 19 |
| base_54_5e875b7e_t630_f0__ambiguous | colgrep | `fuse_mean` | ambiguous | ambiguous | 0.9243 | 1.0000 | 0.0000 | False | 0.9794 | 19 |
| base_54_5e875b7e_t630_f0__ambiguous | colgrep | `fuse_max` | ambiguous | ambiguous | 0.9650 | 1.0000 | 0.0000 | False | 0.9910 | 19 |
| base_54_5e875b7e_t630_f0__ambiguous | colgrep | `fuse_min` | ambiguous | ambiguous | 0.8836 | 1.0000 | 0.0000 | False | 0.9678 | 19 |
| base_55_0b1422bb_t652_f0__correct | colgrep | `gte_only` | grounded | correct | 0.9707 | 1.0000 | 0.0000 | False | 0.9977 | 38 |
| base_55_0b1422bb_t652_f0__correct | colgrep | `code_only` | grounded | correct | 0.9129 | 1.0000 | 0.0000 | False | 1.0000 | 38 |
| base_55_0b1422bb_t652_f0__correct | colgrep | `fuse_mean` | grounded | correct | 0.9418 | 1.0000 | 0.0000 | False | 0.9989 | 38 |
| base_55_0b1422bb_t652_f0__correct | colgrep | `fuse_max` | grounded | correct | 0.9707 | 1.0000 | 0.0000 | False | 1.0000 | 38 |
| base_55_0b1422bb_t652_f0__correct | colgrep | `fuse_min` | grounded | correct | 0.9129 | 1.0000 | 0.0000 | False | 0.9977 | 38 |
| base_55_0b1422bb_t652_f0__wrong | colgrep | `gte_only` | ungrounded | wrong | 0.9625 | 1.0000 | 0.0000 | False | 0.9969 | 38 |
| base_55_0b1422bb_t652_f0__wrong | colgrep | `code_only` | ungrounded | wrong | 0.9034 | 1.0000 | 0.0000 | False | 0.9855 | 38 |
| base_55_0b1422bb_t652_f0__wrong | colgrep | `fuse_mean` | ungrounded | wrong | 0.9330 | 1.0000 | 0.0000 | False | 0.9912 | 38 |
| base_55_0b1422bb_t652_f0__wrong | colgrep | `fuse_max` | ungrounded | wrong | 0.9625 | 1.0000 | 0.0000 | False | 0.9969 | 38 |
| base_55_0b1422bb_t652_f0__wrong | colgrep | `fuse_min` | ungrounded | wrong | 0.9034 | 1.0000 | 0.0000 | False | 0.9855 | 38 |
| base_55_0b1422bb_t652_f0__ambiguous | colgrep | `gte_only` | ambiguous | ambiguous | 0.9705 | 1.0000 | 0.0000 | False | 0.9978 | 38 |
| base_55_0b1422bb_t652_f0__ambiguous | colgrep | `code_only` | ambiguous | ambiguous | 0.9128 | 1.0000 | 0.0000 | False | 1.0000 | 38 |
| base_55_0b1422bb_t652_f0__ambiguous | colgrep | `fuse_mean` | ambiguous | ambiguous | 0.9417 | 1.0000 | 0.0000 | False | 0.9989 | 38 |
| base_55_0b1422bb_t652_f0__ambiguous | colgrep | `fuse_max` | ambiguous | ambiguous | 0.9705 | 1.0000 | 0.0000 | False | 1.0000 | 38 |
| base_55_0b1422bb_t652_f0__ambiguous | colgrep | `fuse_min` | ambiguous | ambiguous | 0.9128 | 1.0000 | 0.0000 | False | 0.9978 | 38 |
| base_56_7b7fb621_t406_f0__correct | colgrep | `gte_only` | grounded | correct | 0.9729 | 1.0000 | 0.0000 | False | 0.9999 | 58 |
| base_56_7b7fb621_t406_f0__correct | colgrep | `code_only` | grounded | correct | 0.9242 | 1.0000 | 0.0000 | False | 0.9999 | 58 |
| base_56_7b7fb621_t406_f0__correct | colgrep | `fuse_mean` | grounded | correct | 0.9485 | 1.0000 | 0.0000 | False | 0.9999 | 58 |
| base_56_7b7fb621_t406_f0__correct | colgrep | `fuse_max` | grounded | correct | 0.9729 | 1.0000 | 0.0000 | False | 0.9999 | 58 |
| base_56_7b7fb621_t406_f0__correct | colgrep | `fuse_min` | grounded | correct | 0.9242 | 1.0000 | 0.0000 | False | 0.9999 | 58 |
| base_56_7b7fb621_t406_f0__wrong | colgrep | `gte_only` | ungrounded | wrong | 0.9706 | 1.0000 | 0.0000 | False | 0.9999 | 58 |
| base_56_7b7fb621_t406_f0__wrong | colgrep | `code_only` | ungrounded | wrong | 0.9202 | 1.0000 | 0.0000 | False | 0.9999 | 58 |
| base_56_7b7fb621_t406_f0__wrong | colgrep | `fuse_mean` | ungrounded | wrong | 0.9454 | 1.0000 | 0.0000 | False | 0.9999 | 58 |
| base_56_7b7fb621_t406_f0__wrong | colgrep | `fuse_max` | ungrounded | wrong | 0.9706 | 1.0000 | 0.0000 | False | 0.9999 | 58 |
| base_56_7b7fb621_t406_f0__wrong | colgrep | `fuse_min` | ungrounded | wrong | 0.9202 | 1.0000 | 0.0000 | False | 0.9999 | 58 |
| base_56_7b7fb621_t406_f0__ambiguous | colgrep | `gte_only` | ambiguous | ambiguous | 0.9727 | 1.0000 | 0.0000 | False | 0.9999 | 58 |
| base_56_7b7fb621_t406_f0__ambiguous | colgrep | `code_only` | ambiguous | ambiguous | 0.9236 | 1.0000 | 0.0000 | False | 0.9999 | 58 |
| base_56_7b7fb621_t406_f0__ambiguous | colgrep | `fuse_mean` | ambiguous | ambiguous | 0.9481 | 1.0000 | 0.0000 | False | 0.9999 | 58 |
| base_56_7b7fb621_t406_f0__ambiguous | colgrep | `fuse_max` | ambiguous | ambiguous | 0.9727 | 1.0000 | 0.0000 | False | 0.9999 | 58 |
| base_56_7b7fb621_t406_f0__ambiguous | colgrep | `fuse_min` | ambiguous | ambiguous | 0.9236 | 1.0000 | 0.0000 | False | 0.9999 | 58 |
| base_57_91cf0ec9_t84_f10__correct | colgrep | `gte_only` | grounded | correct | 0.9764 | 1.0000 | 0.0000 | False | 0.9878 | 23 |
| base_57_91cf0ec9_t84_f10__correct | colgrep | `code_only` | grounded | correct | 0.9239 | 1.0000 | 0.0000 | False | 0.9998 | 23 |
| base_57_91cf0ec9_t84_f10__correct | colgrep | `fuse_mean` | grounded | correct | 0.9502 | 1.0000 | 0.0000 | False | 0.9938 | 23 |
| base_57_91cf0ec9_t84_f10__correct | colgrep | `fuse_max` | grounded | correct | 0.9764 | 1.0000 | 0.0000 | False | 0.9998 | 23 |
| base_57_91cf0ec9_t84_f10__correct | colgrep | `fuse_min` | grounded | correct | 0.9239 | 1.0000 | 0.0000 | False | 0.9878 | 23 |
| base_57_91cf0ec9_t84_f10__wrong | colgrep | `gte_only` | ungrounded | wrong | 0.9704 | 1.0000 | 0.0000 | False | 0.9980 | 23 |
| base_57_91cf0ec9_t84_f10__wrong | colgrep | `code_only` | ungrounded | wrong | 0.9181 | 1.0000 | 0.0000 | False | 0.9999 | 23 |
| base_57_91cf0ec9_t84_f10__wrong | colgrep | `fuse_mean` | ungrounded | wrong | 0.9443 | 1.0000 | 0.0000 | False | 0.9989 | 23 |
| base_57_91cf0ec9_t84_f10__wrong | colgrep | `fuse_max` | ungrounded | wrong | 0.9704 | 1.0000 | 0.0000 | False | 0.9999 | 23 |
| base_57_91cf0ec9_t84_f10__wrong | colgrep | `fuse_min` | ungrounded | wrong | 0.9181 | 1.0000 | 0.0000 | False | 0.9980 | 23 |
| base_57_91cf0ec9_t84_f10__ambiguous | colgrep | `gte_only` | ambiguous | ambiguous | 0.9764 | 1.0000 | 0.0000 | False | 0.9959 | 23 |
| base_57_91cf0ec9_t84_f10__ambiguous | colgrep | `code_only` | ambiguous | ambiguous | 0.9236 | 1.0000 | 0.0000 | False | 0.9721 | 23 |
| base_57_91cf0ec9_t84_f10__ambiguous | colgrep | `fuse_mean` | ambiguous | ambiguous | 0.9500 | 1.0000 | 0.0000 | False | 0.9840 | 23 |
| base_57_91cf0ec9_t84_f10__ambiguous | colgrep | `fuse_max` | ambiguous | ambiguous | 0.9764 | 1.0000 | 0.0000 | False | 0.9959 | 23 |
| base_57_91cf0ec9_t84_f10__ambiguous | colgrep | `fuse_min` | ambiguous | ambiguous | 0.9236 | 1.0000 | 0.0000 | False | 0.9721 | 23 |
| base_58_7b7fb621_t237_f0__correct | colgrep | `gte_only` | grounded | correct | 0.9772 | 1.0000 | 0.0000 | False | 0.9999 | 29 |
| base_58_7b7fb621_t237_f0__correct | colgrep | `code_only` | grounded | correct | 0.9374 | 1.0000 | 0.0000 | False | 1.0000 | 29 |
| base_58_7b7fb621_t237_f0__correct | colgrep | `fuse_mean` | grounded | correct | 0.9573 | 1.0000 | 0.0000 | False | 1.0000 | 29 |
| base_58_7b7fb621_t237_f0__correct | colgrep | `fuse_max` | grounded | correct | 0.9772 | 1.0000 | 0.0000 | False | 1.0000 | 29 |
| base_58_7b7fb621_t237_f0__correct | colgrep | `fuse_min` | grounded | correct | 0.9374 | 1.0000 | 0.0000 | False | 0.9999 | 29 |
| base_58_7b7fb621_t237_f0__wrong | colgrep | `gte_only` | ungrounded | wrong | 0.9690 | 1.0000 | 0.0000 | False | 1.0000 | 29 |
| base_58_7b7fb621_t237_f0__wrong | colgrep | `code_only` | ungrounded | wrong | 0.9181 | 1.0000 | 0.0000 | False | 0.9903 | 29 |
| base_58_7b7fb621_t237_f0__wrong | colgrep | `fuse_mean` | ungrounded | wrong | 0.9435 | 1.0000 | 0.0000 | False | 0.9951 | 29 |
| base_58_7b7fb621_t237_f0__wrong | colgrep | `fuse_max` | ungrounded | wrong | 0.9690 | 1.0000 | 0.0000 | False | 1.0000 | 29 |
| base_58_7b7fb621_t237_f0__wrong | colgrep | `fuse_min` | ungrounded | wrong | 0.9181 | 1.0000 | 0.0000 | False | 0.9903 | 29 |
| base_58_7b7fb621_t237_f0__ambiguous | colgrep | `gte_only` | ambiguous | ambiguous | 0.9765 | 1.0000 | 0.0000 | False | 0.9999 | 29 |
| base_58_7b7fb621_t237_f0__ambiguous | colgrep | `code_only` | ambiguous | ambiguous | 0.9353 | 1.0000 | 0.0000 | False | 1.0000 | 29 |
| base_58_7b7fb621_t237_f0__ambiguous | colgrep | `fuse_mean` | ambiguous | ambiguous | 0.9559 | 1.0000 | 0.0000 | False | 1.0000 | 29 |
| base_58_7b7fb621_t237_f0__ambiguous | colgrep | `fuse_max` | ambiguous | ambiguous | 0.9765 | 1.0000 | 0.0000 | False | 1.0000 | 29 |
| base_58_7b7fb621_t237_f0__ambiguous | colgrep | `fuse_min` | ambiguous | ambiguous | 0.9353 | 1.0000 | 0.0000 | False | 0.9999 | 29 |
| base_59_ee0ad405_t137_f0__correct | colgrep | `gte_only` | grounded | correct | 0.9742 | 1.0000 | 0.0000 | False | 0.9999 | 31 |
| base_59_ee0ad405_t137_f0__correct | colgrep | `code_only` | grounded | correct | 0.9176 | 1.0000 | 0.0000 | False | 0.9999 | 31 |
| base_59_ee0ad405_t137_f0__correct | colgrep | `fuse_mean` | grounded | correct | 0.9459 | 1.0000 | 0.0000 | False | 0.9999 | 31 |
| base_59_ee0ad405_t137_f0__correct | colgrep | `fuse_max` | grounded | correct | 0.9742 | 1.0000 | 0.0000 | False | 0.9999 | 31 |
| base_59_ee0ad405_t137_f0__correct | colgrep | `fuse_min` | grounded | correct | 0.9176 | 1.0000 | 0.0000 | False | 0.9999 | 31 |
| base_59_ee0ad405_t137_f0__wrong | colgrep | `gte_only` | ungrounded | wrong | 0.9701 | 1.0000 | 0.0000 | False | 1.0000 | 31 |
| base_59_ee0ad405_t137_f0__wrong | colgrep | `code_only` | ungrounded | wrong | 0.9071 | 1.0000 | 0.0000 | False | 1.0000 | 31 |
| base_59_ee0ad405_t137_f0__wrong | colgrep | `fuse_mean` | ungrounded | wrong | 0.9386 | 1.0000 | 0.0000 | False | 1.0000 | 31 |
| base_59_ee0ad405_t137_f0__wrong | colgrep | `fuse_max` | ungrounded | wrong | 0.9701 | 1.0000 | 0.0000 | False | 1.0000 | 31 |
| base_59_ee0ad405_t137_f0__wrong | colgrep | `fuse_min` | ungrounded | wrong | 0.9071 | 1.0000 | 0.0000 | False | 1.0000 | 31 |
| base_59_ee0ad405_t137_f0__ambiguous | colgrep | `gte_only` | ambiguous | ambiguous | 0.9740 | 1.0000 | 0.0000 | False | 1.0000 | 31 |
| base_59_ee0ad405_t137_f0__ambiguous | colgrep | `code_only` | ambiguous | ambiguous | 0.9167 | 1.0000 | 0.0000 | False | 0.9999 | 31 |
| base_59_ee0ad405_t137_f0__ambiguous | colgrep | `fuse_mean` | ambiguous | ambiguous | 0.9454 | 1.0000 | 0.0000 | False | 0.9999 | 31 |
| base_59_ee0ad405_t137_f0__ambiguous | colgrep | `fuse_max` | ambiguous | ambiguous | 0.9740 | 1.0000 | 0.0000 | False | 1.0000 | 31 |
| base_59_ee0ad405_t137_f0__ambiguous | colgrep | `fuse_min` | ambiguous | ambiguous | 0.9167 | 1.0000 | 0.0000 | False | 0.9999 | 31 |
| base_60_7b7fb621_t178_f1__correct | colgrep | `gte_only` | grounded | correct | 0.9802 | 1.0000 | 0.0000 | False | 1.0000 | 24 |
| base_60_7b7fb621_t178_f1__correct | colgrep | `code_only` | grounded | correct | 0.9292 | 1.0000 | 0.0000 | False | 0.9930 | 24 |
| base_60_7b7fb621_t178_f1__correct | colgrep | `fuse_mean` | grounded | correct | 0.9547 | 1.0000 | 0.0000 | False | 0.9965 | 24 |
| base_60_7b7fb621_t178_f1__correct | colgrep | `fuse_max` | grounded | correct | 0.9802 | 1.0000 | 0.0000 | False | 1.0000 | 24 |
| base_60_7b7fb621_t178_f1__correct | colgrep | `fuse_min` | grounded | correct | 0.9292 | 1.0000 | 0.0000 | False | 0.9930 | 24 |
| base_60_7b7fb621_t178_f1__wrong | colgrep | `gte_only` | ungrounded | wrong | 0.9781 | 1.0000 | 0.0000 | False | 0.9999 | 24 |
| base_60_7b7fb621_t178_f1__wrong | colgrep | `code_only` | ungrounded | wrong | 0.9282 | 1.0000 | 0.0000 | False | 1.0000 | 24 |
| base_60_7b7fb621_t178_f1__wrong | colgrep | `fuse_mean` | ungrounded | wrong | 0.9531 | 1.0000 | 0.0000 | False | 1.0000 | 24 |
| base_60_7b7fb621_t178_f1__wrong | colgrep | `fuse_max` | ungrounded | wrong | 0.9781 | 1.0000 | 0.0000 | False | 1.0000 | 24 |
| base_60_7b7fb621_t178_f1__wrong | colgrep | `fuse_min` | ungrounded | wrong | 0.9282 | 1.0000 | 0.0000 | False | 0.9999 | 24 |
| base_60_7b7fb621_t178_f1__ambiguous | colgrep | `gte_only` | ambiguous | ambiguous | 0.9802 | 1.0000 | 0.0000 | False | 1.0000 | 24 |
| base_60_7b7fb621_t178_f1__ambiguous | colgrep | `code_only` | ambiguous | ambiguous | 0.9288 | 1.0000 | 0.0000 | False | 0.9930 | 24 |
| base_60_7b7fb621_t178_f1__ambiguous | colgrep | `fuse_mean` | ambiguous | ambiguous | 0.9545 | 1.0000 | 0.0000 | False | 0.9965 | 24 |
| base_60_7b7fb621_t178_f1__ambiguous | colgrep | `fuse_max` | ambiguous | ambiguous | 0.9802 | 1.0000 | 0.0000 | False | 1.0000 | 24 |
| base_60_7b7fb621_t178_f1__ambiguous | colgrep | `fuse_min` | ambiguous | ambiguous | 0.9288 | 1.0000 | 0.0000 | False | 0.9930 | 24 |
