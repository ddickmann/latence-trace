# Coding-Agent Groundedness: Scorer Stack x Chunker

> SKIP — the flagship cell (colgrep x fuse_mean) was not evaluated in this run (code encoder likely unavailable).

- run tag: `v2_pilot_encoder_chunker_ab`
- case bank: `transcripts_v2`
- primary scorer: `lightonai/GTE-ModernColBERT-v1`
- orthogonal scorer: `lightonai/LateOn-Code-edge` (available: `True`)
- shared chunk budget: `256` tokens (GTE tokenizer)
- max SWE-bench instances: `15` (ignored for `transcripts_v1`)
- phantom threshold: `0.35`
- flagship cell: `colgrep` x `fuse_mean` (present: `False`)

## Headline matrix

| chunker | scorer | AUROC | grounded cov | cov delta | unused delta | phantom@thr | grounded held-out rate | p95 ms | support units |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sentence_packed | `gte_only` | 0.7875 | 1.0000 | 0.0000 | 0.0000 | n/a | 0.0000 | 2801.29 | 135.4 |
| sentence_packed | `fuse_max` | 0.7875 | 1.0000 | 0.0000 | 0.0000 | n/a | 0.0000 | 2702.58 | 135.4 |
| colgrep | `gte_only` | 0.7700 | 1.0000 | 0.0000 | 0.0000 | n/a | 0.0000 | 1896.76 | 113.0 |
| colgrep | `fuse_max` | 0.7700 | 1.0000 | 0.0000 | 0.0000 | n/a | 0.0000 | 1890.83 | 113.0 |

`*` marks cells that pass `reverse_context AUROC >= 0.90`.

## Score distribution diagnostics (reverse_context)

| chunker | scorer | grounded mean | grounded p10-p90 | ungrounded mean | ungrounded p10-p90 | separation | saturation@0.95 |
|---|---|---:|---|---:|---|---:|---:|
| sentence_packed | `gte_only` | 0.9794 | 0.9746-0.9850 | 0.9735 | 0.9684-0.9812 | +0.0059 | 1.0000 |
| sentence_packed | `fuse_max` | 0.9794 | 0.9746-0.9850 | 0.9735 | 0.9684-0.9812 | +0.0059 | 1.0000 |
| colgrep | `gte_only` | 0.9778 | 0.9725-0.9846 | 0.9715 | 0.9659-0.9808 | +0.0062 | 1.0000 |
| colgrep | `fuse_max` | 0.9778 | 0.9725-0.9846 | 0.9715 | 0.9659-0.9808 | +0.0062 | 1.0000 |

`separation` is mean(grounded) − mean(ungrounded) on `reverse_context`. `saturation@0.95` is the fraction of anchor cases (grounded + ungrounded) whose score ≥ 0.95 — a ceiling effect indicator.

## Tier metrics (transcripts_v1: correct / ambiguous / wrong)

`AUROC c-vs-w` = AUROC on `tier=correct` vs `tier=wrong`. `AUROC c-vs-a` = AUROC on `correct` vs `ambiguous` (hard bucket). `monotonicity` = fraction of base scenarios where `correct > ambiguous > wrong` holds on `reverse_context`. `partial mono` = fraction where `correct > wrong` holds (ignores ambiguous).

| chunker | scorer | AUROC c-vs-w | AUROC c-vs-a | AUROC c-vs-wa | monotonicity | partial mono | correct mean | ambiguous mean | wrong mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sentence_packed | `gte_only` | 0.7875 | 0.5025 | 0.6450 | 0.7000 | 1.0000 | 0.9794 | 0.9792 | 0.9735 |
| sentence_packed | `fuse_max` | 0.7875 | 0.5025 | 0.6450 | 0.7000 | 1.0000 | 0.9794 | 0.9792 | 0.9735 |
| colgrep | `gte_only` | 0.7700 | 0.5275 | 0.6488 | 0.7500 | 1.0000 | 0.9778 | 0.9775 | 0.9715 |
| colgrep | `fuse_max` | 0.7700 | 0.5275 | 0.6488 | 0.7500 | 1.0000 | 0.9778 | 0.9775 | 0.9715 |

## Stacking lift (fuse_* minus gte_only, per chunker)

| chunker | rule | AUROC lift | unused-delta lift | phantom precision lift |
|---|---|---:|---:|---:|
| sentence_packed | fuse_mean | n/a | n/a | n/a |
| sentence_packed | fuse_max | +0.0000 | +0.0000 | n/a |
| sentence_packed | fuse_min | n/a | n/a | n/a |
| colgrep | fuse_mean | n/a | n/a | n/a |
| colgrep | fuse_max | +0.0000 | +0.0000 | n/a |
| colgrep | fuse_min | n/a | n/a | n/a |

## Chunker lift (colgrep minus sentence_packed, per scorer)

| scorer | AUROC lift | unused-delta lift | phantom precision lift |
|---|---:|---:|---:|
| `gte_only` | -0.0175 | +0.0000 | n/a |
| `fuse_max` | -0.0175 | +0.0000 | n/a |

## Subcategory snapshot (per cell)

### `sentence_packed` x `gte_only`

| subcategory | count | reverse_context | coverage | unused ratio | max evidence |
|---|---:|---:|---:|---:|---:|
| ambiguous | 20 | 0.9792 | 1.0000 | 0.0000 | 0.9956 |
| correct | 20 | 0.9794 | 1.0000 | 0.0000 | 0.9976 |
| wrong | 20 | 0.9735 | 1.0000 | 0.0000 | 0.9968 |

### `sentence_packed` x `fuse_max`

| subcategory | count | reverse_context | coverage | unused ratio | max evidence |
|---|---:|---:|---:|---:|---:|
| ambiguous | 20 | 0.9792 | 1.0000 | 0.0000 | 0.9965 |
| correct | 20 | 0.9794 | 1.0000 | 0.0000 | 0.9976 |
| wrong | 20 | 0.9735 | 1.0000 | 0.0000 | 0.9974 |

### `colgrep` x `gte_only`

| subcategory | count | reverse_context | coverage | unused ratio | max evidence |
|---|---:|---:|---:|---:|---:|
| ambiguous | 20 | 0.9775 | 1.0000 | 0.0000 | 0.9953 |
| correct | 20 | 0.9778 | 1.0000 | 0.0000 | 0.9973 |
| wrong | 20 | 0.9715 | 1.0000 | 0.0000 | 0.9967 |

### `colgrep` x `fuse_max`

| subcategory | count | reverse_context | coverage | unused ratio | max evidence |
|---|---:|---:|---:|---:|---:|
| ambiguous | 20 | 0.9775 | 1.0000 | 0.0000 | 0.9963 |
| correct | 20 | 0.9778 | 1.0000 | 0.0000 | 0.9974 |
| wrong | 20 | 0.9715 | 1.0000 | 0.0000 | 0.9973 |

## Per-case scores

| id | chunker | scorer | label | subcategory | reverse_context | coverage | unused | phantom | max evidence | units |
|---|---|---|---|---|---:|---:|---:|---|---:|---:|
| base_01_f0afeaf2_t952_f1__correct | sentence_packed | `gte_only` | grounded | correct | 0.9843 | 1.0000 | 0.0000 | False | 0.9981 | 145 |
| base_01_f0afeaf2_t952_f1__correct | sentence_packed | `fuse_max` | grounded | correct | 0.9843 | 1.0000 | 0.0000 | False | 0.9981 | 145 |
| base_01_f0afeaf2_t952_f1__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9820 | 1.0000 | 0.0000 | False | 0.9999 | 145 |
| base_01_f0afeaf2_t952_f1__wrong | sentence_packed | `fuse_max` | ungrounded | wrong | 0.9820 | 1.0000 | 0.0000 | False | 0.9999 | 145 |
| base_01_f0afeaf2_t952_f1__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9844 | 1.0000 | 0.0000 | False | 0.9981 | 145 |
| base_01_f0afeaf2_t952_f1__ambiguous | sentence_packed | `fuse_max` | ambiguous | ambiguous | 0.9844 | 1.0000 | 0.0000 | False | 0.9981 | 145 |
| base_02_61e1f5a1_t258_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9725 | 1.0000 | 0.0000 | False | 0.9988 | 88 |
| base_02_61e1f5a1_t258_f0__correct | sentence_packed | `fuse_max` | grounded | correct | 0.9725 | 1.0000 | 0.0000 | False | 1.0000 | 88 |
| base_02_61e1f5a1_t258_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9685 | 1.0000 | 0.0000 | False | 0.9981 | 88 |
| base_02_61e1f5a1_t258_f0__wrong | sentence_packed | `fuse_max` | ungrounded | wrong | 0.9685 | 1.0000 | 0.0000 | False | 1.0000 | 88 |
| base_02_61e1f5a1_t258_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9724 | 1.0000 | 0.0000 | False | 0.9972 | 88 |
| base_02_61e1f5a1_t258_f0__ambiguous | sentence_packed | `fuse_max` | ambiguous | ambiguous | 0.9724 | 1.0000 | 0.0000 | False | 0.9972 | 88 |
| base_03_f0afeaf2_t2082_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9774 | 1.0000 | 0.0000 | False | 0.9979 | 198 |
| base_03_f0afeaf2_t2082_f0__correct | sentence_packed | `fuse_max` | grounded | correct | 0.9774 | 1.0000 | 0.0000 | False | 0.9979 | 198 |
| base_03_f0afeaf2_t2082_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9705 | 1.0000 | 0.0000 | False | 0.9955 | 198 |
| base_03_f0afeaf2_t2082_f0__wrong | sentence_packed | `fuse_max` | ungrounded | wrong | 0.9705 | 1.0000 | 0.0000 | False | 0.9955 | 198 |
| base_03_f0afeaf2_t2082_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9776 | 1.0000 | 0.0000 | False | 0.9974 | 198 |
| base_03_f0afeaf2_t2082_f0__ambiguous | sentence_packed | `fuse_max` | ambiguous | ambiguous | 0.9776 | 1.0000 | 0.0000 | False | 1.0000 | 198 |
| base_04_8a0f6f68_t521_f3__correct | sentence_packed | `gte_only` | grounded | correct | 0.9850 | 1.0000 | 0.0000 | False | 0.9975 | 71 |
| base_04_8a0f6f68_t521_f3__correct | sentence_packed | `fuse_max` | grounded | correct | 0.9850 | 1.0000 | 0.0000 | False | 0.9975 | 71 |
| base_04_8a0f6f68_t521_f3__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9811 | 1.0000 | 0.0000 | False | 0.9975 | 71 |
| base_04_8a0f6f68_t521_f3__wrong | sentence_packed | `fuse_max` | ungrounded | wrong | 0.9811 | 1.0000 | 0.0000 | False | 0.9975 | 71 |
| base_04_8a0f6f68_t521_f3__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9851 | 1.0000 | 0.0000 | False | 0.9975 | 71 |
| base_04_8a0f6f68_t521_f3__ambiguous | sentence_packed | `fuse_max` | ambiguous | ambiguous | 0.9851 | 1.0000 | 0.0000 | False | 0.9975 | 71 |
| base_05_126e38fc_t1159_f4__correct | sentence_packed | `gte_only` | grounded | correct | 0.9748 | 1.0000 | 0.0000 | False | 1.0000 | 130 |
| base_05_126e38fc_t1159_f4__correct | sentence_packed | `fuse_max` | grounded | correct | 0.9748 | 1.0000 | 0.0000 | False | 1.0000 | 130 |
| base_05_126e38fc_t1159_f4__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9726 | 1.0000 | 0.0000 | False | 1.0000 | 130 |
| base_05_126e38fc_t1159_f4__wrong | sentence_packed | `fuse_max` | ungrounded | wrong | 0.9726 | 1.0000 | 0.0000 | False | 1.0000 | 130 |
| base_05_126e38fc_t1159_f4__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9745 | 1.0000 | 0.0000 | False | 0.9896 | 130 |
| base_05_126e38fc_t1159_f4__ambiguous | sentence_packed | `fuse_max` | ambiguous | ambiguous | 0.9745 | 1.0000 | 0.0000 | False | 0.9896 | 130 |
| base_06_f0afeaf2_t1771_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9755 | 1.0000 | 0.0000 | False | 0.9929 | 199 |
| base_06_f0afeaf2_t1771_f0__correct | sentence_packed | `fuse_max` | grounded | correct | 0.9755 | 1.0000 | 0.0000 | False | 0.9929 | 199 |
| base_06_f0afeaf2_t1771_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9700 | 1.0000 | 0.0000 | False | 0.9705 | 199 |
| base_06_f0afeaf2_t1771_f0__wrong | sentence_packed | `fuse_max` | ungrounded | wrong | 0.9700 | 1.0000 | 0.0000 | False | 0.9815 | 199 |
| base_06_f0afeaf2_t1771_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9709 | 1.0000 | 0.0000 | False | 0.9674 | 199 |
| base_06_f0afeaf2_t1771_f0__ambiguous | sentence_packed | `fuse_max` | ambiguous | ambiguous | 0.9709 | 1.0000 | 0.0000 | False | 0.9815 | 199 |
| base_07_7b7fb621_t328_f1__correct | sentence_packed | `gte_only` | grounded | correct | 0.9776 | 1.0000 | 0.0000 | False | 0.9999 | 36 |
| base_07_7b7fb621_t328_f1__correct | sentence_packed | `fuse_max` | grounded | correct | 0.9776 | 1.0000 | 0.0000 | False | 1.0000 | 36 |
| base_07_7b7fb621_t328_f1__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9703 | 1.0000 | 0.0000 | False | 0.9999 | 36 |
| base_07_7b7fb621_t328_f1__wrong | sentence_packed | `fuse_max` | ungrounded | wrong | 0.9703 | 1.0000 | 0.0000 | False | 1.0000 | 36 |
| base_07_7b7fb621_t328_f1__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9777 | 1.0000 | 0.0000 | False | 1.0000 | 36 |
| base_07_7b7fb621_t328_f1__ambiguous | sentence_packed | `fuse_max` | ambiguous | ambiguous | 0.9777 | 1.0000 | 0.0000 | False | 1.0000 | 36 |
| base_08_126e38fc_t1785_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9807 | 1.0000 | 0.0000 | False | 0.9983 | 210 |
| base_08_126e38fc_t1785_f0__correct | sentence_packed | `fuse_max` | grounded | correct | 0.9807 | 1.0000 | 0.0000 | False | 0.9983 | 210 |
| base_08_126e38fc_t1785_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9792 | 1.0000 | 0.0000 | False | 0.9982 | 210 |
| base_08_126e38fc_t1785_f0__wrong | sentence_packed | `fuse_max` | ungrounded | wrong | 0.9792 | 1.0000 | 0.0000 | False | 0.9982 | 210 |
| base_08_126e38fc_t1785_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9806 | 1.0000 | 0.0000 | False | 0.9983 | 210 |
| base_08_126e38fc_t1785_f0__ambiguous | sentence_packed | `fuse_max` | ambiguous | ambiguous | 0.9806 | 1.0000 | 0.0000 | False | 0.9983 | 210 |
| base_09_0b1422bb_t571_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9584 | 1.0000 | 0.0000 | False | 0.9980 | 20 |
| base_09_0b1422bb_t571_f0__correct | sentence_packed | `fuse_max` | grounded | correct | 0.9584 | 1.0000 | 0.0000 | False | 0.9980 | 20 |
| base_09_0b1422bb_t571_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9565 | 1.0000 | 0.0000 | False | 0.9979 | 20 |
| base_09_0b1422bb_t571_f0__wrong | sentence_packed | `fuse_max` | ungrounded | wrong | 0.9565 | 1.0000 | 0.0000 | False | 0.9979 | 20 |
| base_09_0b1422bb_t571_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9582 | 1.0000 | 0.0000 | False | 0.9980 | 20 |
| base_09_0b1422bb_t571_f0__ambiguous | sentence_packed | `fuse_max` | ambiguous | ambiguous | 0.9582 | 1.0000 | 0.0000 | False | 0.9980 | 20 |
| base_10_d7fd7127_t1579_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9830 | 1.0000 | 0.0000 | False | 0.9951 | 218 |
| base_10_d7fd7127_t1579_f0__correct | sentence_packed | `fuse_max` | grounded | correct | 0.9830 | 1.0000 | 0.0000 | False | 0.9951 | 218 |
| base_10_d7fd7127_t1579_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9729 | 1.0000 | 0.0000 | False | 0.9948 | 218 |
| base_10_d7fd7127_t1579_f0__wrong | sentence_packed | `fuse_max` | ungrounded | wrong | 0.9729 | 1.0000 | 0.0000 | False | 0.9948 | 218 |
| base_10_d7fd7127_t1579_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9828 | 1.0000 | 0.0000 | False | 0.9951 | 218 |
| base_10_d7fd7127_t1579_f0__ambiguous | sentence_packed | `fuse_max` | ambiguous | ambiguous | 0.9828 | 1.0000 | 0.0000 | False | 0.9951 | 218 |
| base_11_f0afeaf2_t1418_f1__correct | sentence_packed | `gte_only` | grounded | correct | 0.9850 | 1.0000 | 0.0000 | False | 0.9960 | 195 |
| base_11_f0afeaf2_t1418_f1__correct | sentence_packed | `fuse_max` | grounded | correct | 0.9850 | 1.0000 | 0.0000 | False | 0.9960 | 195 |
| base_11_f0afeaf2_t1418_f1__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9731 | 1.0000 | 0.0000 | False | 0.9964 | 195 |
| base_11_f0afeaf2_t1418_f1__wrong | sentence_packed | `fuse_max` | ungrounded | wrong | 0.9731 | 1.0000 | 0.0000 | False | 0.9964 | 195 |
| base_11_f0afeaf2_t1418_f1__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9848 | 1.0000 | 0.0000 | False | 0.9961 | 195 |
| base_11_f0afeaf2_t1418_f1__ambiguous | sentence_packed | `fuse_max` | ambiguous | ambiguous | 0.9848 | 1.0000 | 0.0000 | False | 0.9961 | 195 |
| base_12_f0afeaf2_t955_f2__correct | sentence_packed | `gte_only` | grounded | correct | 0.9757 | 1.0000 | 0.0000 | False | 0.9946 | 145 |
| base_12_f0afeaf2_t955_f2__correct | sentence_packed | `fuse_max` | grounded | correct | 0.9757 | 1.0000 | 0.0000 | False | 0.9946 | 145 |
| base_12_f0afeaf2_t955_f2__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9673 | 1.0000 | 0.0000 | False | 1.0000 | 145 |
| base_12_f0afeaf2_t955_f2__wrong | sentence_packed | `fuse_max` | ungrounded | wrong | 0.9673 | 1.0000 | 0.0000 | False | 1.0000 | 145 |
| base_12_f0afeaf2_t955_f2__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9781 | 1.0000 | 0.0000 | False | 0.9946 | 145 |
| base_12_f0afeaf2_t955_f2__ambiguous | sentence_packed | `fuse_max` | ambiguous | ambiguous | 0.9781 | 1.0000 | 0.0000 | False | 0.9946 | 145 |
| base_13_d7fd7127_t1553_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9851 | 1.0000 | 0.0000 | False | 1.0000 | 218 |
| base_13_d7fd7127_t1553_f0__correct | sentence_packed | `fuse_max` | grounded | correct | 0.9851 | 1.0000 | 0.0000 | False | 1.0000 | 218 |
| base_13_d7fd7127_t1553_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9720 | 1.0000 | 0.0000 | False | 1.0000 | 218 |
| base_13_d7fd7127_t1553_f0__wrong | sentence_packed | `fuse_max` | ungrounded | wrong | 0.9720 | 1.0000 | 0.0000 | False | 1.0000 | 218 |
| base_13_d7fd7127_t1553_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9851 | 1.0000 | 0.0000 | False | 1.0000 | 218 |
| base_13_d7fd7127_t1553_f0__ambiguous | sentence_packed | `fuse_max` | ambiguous | ambiguous | 0.9851 | 1.0000 | 0.0000 | False | 1.0000 | 218 |
| base_14_f0afeaf2_t2825_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9842 | 1.0000 | 0.0000 | False | 1.0000 | 200 |
| base_14_f0afeaf2_t2825_f0__correct | sentence_packed | `fuse_max` | grounded | correct | 0.9842 | 1.0000 | 0.0000 | False | 1.0000 | 200 |
| base_14_f0afeaf2_t2825_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9756 | 1.0000 | 0.0000 | False | 1.0000 | 200 |
| base_14_f0afeaf2_t2825_f0__wrong | sentence_packed | `fuse_max` | ungrounded | wrong | 0.9756 | 1.0000 | 0.0000 | False | 1.0000 | 200 |
| base_14_f0afeaf2_t2825_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9839 | 1.0000 | 0.0000 | False | 0.9992 | 200 |
| base_14_f0afeaf2_t2825_f0__ambiguous | sentence_packed | `fuse_max` | ambiguous | ambiguous | 0.9839 | 1.0000 | 0.0000 | False | 0.9999 | 200 |
| base_15_91cf0ec9_t419_f2__correct | sentence_packed | `gte_only` | grounded | correct | 0.9825 | 1.0000 | 0.0000 | False | 0.9995 | 110 |
| base_15_91cf0ec9_t419_f2__correct | sentence_packed | `fuse_max` | grounded | correct | 0.9825 | 1.0000 | 0.0000 | False | 0.9995 | 110 |
| base_15_91cf0ec9_t419_f2__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9779 | 1.0000 | 0.0000 | False | 0.9998 | 110 |
| base_15_91cf0ec9_t419_f2__wrong | sentence_packed | `fuse_max` | ungrounded | wrong | 0.9779 | 1.0000 | 0.0000 | False | 1.0000 | 110 |
| base_15_91cf0ec9_t419_f2__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9821 | 1.0000 | 0.0000 | False | 0.9995 | 110 |
| base_15_91cf0ec9_t419_f2__ambiguous | sentence_packed | `fuse_max` | ambiguous | ambiguous | 0.9821 | 1.0000 | 0.0000 | False | 0.9995 | 110 |
| base_16_8a0f6f68_t1002_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9810 | 1.0000 | 0.0000 | False | 0.9956 | 93 |
| base_16_8a0f6f68_t1002_f0__correct | sentence_packed | `fuse_max` | grounded | correct | 0.9810 | 1.0000 | 0.0000 | False | 0.9956 | 93 |
| base_16_8a0f6f68_t1002_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9738 | 1.0000 | 0.0000 | False | 1.0000 | 93 |
| base_16_8a0f6f68_t1002_f0__wrong | sentence_packed | `fuse_max` | ungrounded | wrong | 0.9738 | 1.0000 | 0.0000 | False | 1.0000 | 93 |
| base_16_8a0f6f68_t1002_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9808 | 1.0000 | 0.0000 | False | 0.9971 | 93 |
| base_16_8a0f6f68_t1002_f0__ambiguous | sentence_packed | `fuse_max` | ambiguous | ambiguous | 0.9808 | 1.0000 | 0.0000 | False | 0.9971 | 93 |
| base_17_126e38fc_t712_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9797 | 1.0000 | 0.0000 | False | 0.9959 | 58 |
| base_17_126e38fc_t712_f0__correct | sentence_packed | `fuse_max` | grounded | correct | 0.9797 | 1.0000 | 0.0000 | False | 0.9959 | 58 |
| base_17_126e38fc_t712_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9762 | 1.0000 | 0.0000 | False | 0.9945 | 58 |
| base_17_126e38fc_t712_f0__wrong | sentence_packed | `fuse_max` | ungrounded | wrong | 0.9762 | 1.0000 | 0.0000 | False | 0.9945 | 58 |
| base_17_126e38fc_t712_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9797 | 1.0000 | 0.0000 | False | 0.9959 | 58 |
| base_17_126e38fc_t712_f0__ambiguous | sentence_packed | `fuse_max` | ambiguous | ambiguous | 0.9797 | 1.0000 | 0.0000 | False | 0.9959 | 58 |
| base_18_f0afeaf2_t457_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9789 | 1.0000 | 0.0000 | False | 1.0000 | 89 |
| base_18_f0afeaf2_t457_f0__correct | sentence_packed | `fuse_max` | grounded | correct | 0.9789 | 1.0000 | 0.0000 | False | 1.0000 | 89 |
| base_18_f0afeaf2_t457_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9697 | 1.0000 | 0.0000 | False | 0.9963 | 89 |
| base_18_f0afeaf2_t457_f0__wrong | sentence_packed | `fuse_max` | ungrounded | wrong | 0.9697 | 1.0000 | 0.0000 | False | 0.9963 | 89 |
| base_18_f0afeaf2_t457_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9800 | 1.0000 | 0.0000 | False | 0.9956 | 89 |
| base_18_f0afeaf2_t457_f0__ambiguous | sentence_packed | `fuse_max` | ambiguous | ambiguous | 0.9800 | 1.0000 | 0.0000 | False | 0.9956 | 89 |
| base_19_5e875b7e_t1360_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9783 | 1.0000 | 0.0000 | False | 0.9948 | 60 |
| base_19_5e875b7e_t1360_f0__correct | sentence_packed | `fuse_max` | grounded | correct | 0.9783 | 1.0000 | 0.0000 | False | 0.9948 | 60 |
| base_19_5e875b7e_t1360_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9740 | 1.0000 | 0.0000 | False | 0.9971 | 60 |
| base_19_5e875b7e_t1360_f0__wrong | sentence_packed | `fuse_max` | ungrounded | wrong | 0.9740 | 1.0000 | 0.0000 | False | 0.9971 | 60 |
| base_19_5e875b7e_t1360_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9782 | 1.0000 | 0.0000 | False | 0.9969 | 60 |
| base_19_5e875b7e_t1360_f0__ambiguous | sentence_packed | `fuse_max` | ambiguous | ambiguous | 0.9782 | 1.0000 | 0.0000 | False | 0.9969 | 60 |
| base_20_d7fd7127_t1617_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9876 | 1.0000 | 0.0000 | False | 0.9988 | 226 |
| base_20_d7fd7127_t1617_f0__correct | sentence_packed | `fuse_max` | grounded | correct | 0.9876 | 1.0000 | 0.0000 | False | 0.9988 | 226 |
| base_20_d7fd7127_t1617_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9870 | 1.0000 | 0.0000 | False | 0.9988 | 226 |
| base_20_d7fd7127_t1617_f0__wrong | sentence_packed | `fuse_max` | ungrounded | wrong | 0.9870 | 1.0000 | 0.0000 | False | 0.9988 | 226 |
| base_20_d7fd7127_t1617_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9874 | 1.0000 | 0.0000 | False | 0.9988 | 226 |
| base_20_d7fd7127_t1617_f0__ambiguous | sentence_packed | `fuse_max` | ambiguous | ambiguous | 0.9874 | 1.0000 | 0.0000 | False | 0.9988 | 226 |
| base_01_f0afeaf2_t952_f1__correct | colgrep | `gte_only` | grounded | correct | 0.9829 | 1.0000 | 0.0000 | False | 0.9979 | 126 |
| base_01_f0afeaf2_t952_f1__correct | colgrep | `fuse_max` | grounded | correct | 0.9829 | 1.0000 | 0.0000 | False | 0.9979 | 126 |
| base_01_f0afeaf2_t952_f1__wrong | colgrep | `gte_only` | ungrounded | wrong | 0.9807 | 1.0000 | 0.0000 | False | 0.9997 | 126 |
| base_01_f0afeaf2_t952_f1__wrong | colgrep | `fuse_max` | ungrounded | wrong | 0.9807 | 1.0000 | 0.0000 | False | 0.9997 | 126 |
| base_01_f0afeaf2_t952_f1__ambiguous | colgrep | `gte_only` | ambiguous | ambiguous | 0.9828 | 1.0000 | 0.0000 | False | 0.9979 | 126 |
| base_01_f0afeaf2_t952_f1__ambiguous | colgrep | `fuse_max` | ambiguous | ambiguous | 0.9828 | 1.0000 | 0.0000 | False | 0.9979 | 126 |
| base_02_61e1f5a1_t258_f0__correct | colgrep | `gte_only` | grounded | correct | 0.9728 | 1.0000 | 0.0000 | False | 0.9987 | 92 |
| base_02_61e1f5a1_t258_f0__correct | colgrep | `fuse_max` | grounded | correct | 0.9728 | 1.0000 | 0.0000 | False | 1.0000 | 92 |
| base_02_61e1f5a1_t258_f0__wrong | colgrep | `gte_only` | ungrounded | wrong | 0.9684 | 1.0000 | 0.0000 | False | 0.9982 | 92 |
| base_02_61e1f5a1_t258_f0__wrong | colgrep | `fuse_max` | ungrounded | wrong | 0.9684 | 1.0000 | 0.0000 | False | 1.0000 | 92 |
| base_02_61e1f5a1_t258_f0__ambiguous | colgrep | `gte_only` | ambiguous | ambiguous | 0.9725 | 1.0000 | 0.0000 | False | 0.9968 | 92 |
| base_02_61e1f5a1_t258_f0__ambiguous | colgrep | `fuse_max` | ambiguous | ambiguous | 0.9725 | 1.0000 | 0.0000 | False | 0.9968 | 92 |
| base_03_f0afeaf2_t2082_f0__correct | colgrep | `gte_only` | grounded | correct | 0.9741 | 1.0000 | 0.0000 | False | 0.9977 | 107 |
| base_03_f0afeaf2_t2082_f0__correct | colgrep | `fuse_max` | grounded | correct | 0.9741 | 1.0000 | 0.0000 | False | 0.9977 | 107 |
| base_03_f0afeaf2_t2082_f0__wrong | colgrep | `gte_only` | ungrounded | wrong | 0.9675 | 1.0000 | 0.0000 | False | 0.9956 | 107 |
| base_03_f0afeaf2_t2082_f0__wrong | colgrep | `fuse_max` | ungrounded | wrong | 0.9675 | 1.0000 | 0.0000 | False | 0.9956 | 107 |
| base_03_f0afeaf2_t2082_f0__ambiguous | colgrep | `gte_only` | ambiguous | ambiguous | 0.9742 | 1.0000 | 0.0000 | False | 0.9973 | 107 |
| base_03_f0afeaf2_t2082_f0__ambiguous | colgrep | `fuse_max` | ambiguous | ambiguous | 0.9742 | 1.0000 | 0.0000 | False | 1.0000 | 107 |
| base_04_8a0f6f68_t521_f3__correct | colgrep | `gte_only` | grounded | correct | 0.9849 | 1.0000 | 0.0000 | False | 0.9974 | 91 |
| base_04_8a0f6f68_t521_f3__correct | colgrep | `fuse_max` | grounded | correct | 0.9849 | 1.0000 | 0.0000 | False | 0.9974 | 91 |
| base_04_8a0f6f68_t521_f3__wrong | colgrep | `gte_only` | ungrounded | wrong | 0.9809 | 1.0000 | 0.0000 | False | 0.9974 | 91 |
| base_04_8a0f6f68_t521_f3__wrong | colgrep | `fuse_max` | ungrounded | wrong | 0.9809 | 1.0000 | 0.0000 | False | 0.9974 | 91 |
| base_04_8a0f6f68_t521_f3__ambiguous | colgrep | `gte_only` | ambiguous | ambiguous | 0.9850 | 1.0000 | 0.0000 | False | 0.9974 | 91 |
| base_04_8a0f6f68_t521_f3__ambiguous | colgrep | `fuse_max` | ambiguous | ambiguous | 0.9850 | 1.0000 | 0.0000 | False | 0.9974 | 91 |
| base_05_126e38fc_t1159_f4__correct | colgrep | `gte_only` | grounded | correct | 0.9740 | 1.0000 | 0.0000 | False | 1.0000 | 146 |
| base_05_126e38fc_t1159_f4__correct | colgrep | `fuse_max` | grounded | correct | 0.9740 | 1.0000 | 0.0000 | False | 1.0000 | 146 |
| base_05_126e38fc_t1159_f4__wrong | colgrep | `gte_only` | ungrounded | wrong | 0.9719 | 1.0000 | 0.0000 | False | 1.0000 | 146 |
| base_05_126e38fc_t1159_f4__wrong | colgrep | `fuse_max` | ungrounded | wrong | 0.9719 | 1.0000 | 0.0000 | False | 1.0000 | 146 |
| base_05_126e38fc_t1159_f4__ambiguous | colgrep | `gte_only` | ambiguous | ambiguous | 0.9737 | 1.0000 | 0.0000 | False | 0.9902 | 146 |
| base_05_126e38fc_t1159_f4__ambiguous | colgrep | `fuse_max` | ambiguous | ambiguous | 0.9737 | 1.0000 | 0.0000 | False | 0.9902 | 146 |
| base_06_f0afeaf2_t1771_f0__correct | colgrep | `gte_only` | grounded | correct | 0.9747 | 1.0000 | 0.0000 | False | 0.9928 | 154 |
| base_06_f0afeaf2_t1771_f0__correct | colgrep | `fuse_max` | grounded | correct | 0.9747 | 1.0000 | 0.0000 | False | 0.9928 | 154 |
| base_06_f0afeaf2_t1771_f0__wrong | colgrep | `gte_only` | ungrounded | wrong | 0.9683 | 1.0000 | 0.0000 | False | 0.9695 | 154 |
| base_06_f0afeaf2_t1771_f0__wrong | colgrep | `fuse_max` | ungrounded | wrong | 0.9683 | 1.0000 | 0.0000 | False | 0.9789 | 154 |
| base_06_f0afeaf2_t1771_f0__ambiguous | colgrep | `gte_only` | ambiguous | ambiguous | 0.9692 | 1.0000 | 0.0000 | False | 0.9626 | 154 |
| base_06_f0afeaf2_t1771_f0__ambiguous | colgrep | `fuse_max` | ambiguous | ambiguous | 0.9692 | 1.0000 | 0.0000 | False | 0.9789 | 154 |
| base_07_7b7fb621_t328_f1__correct | colgrep | `gte_only` | grounded | correct | 0.9776 | 1.0000 | 0.0000 | False | 0.9999 | 45 |
| base_07_7b7fb621_t328_f1__correct | colgrep | `fuse_max` | grounded | correct | 0.9776 | 1.0000 | 0.0000 | False | 1.0000 | 45 |
| base_07_7b7fb621_t328_f1__wrong | colgrep | `gte_only` | ungrounded | wrong | 0.9703 | 1.0000 | 0.0000 | False | 0.9998 | 45 |
| base_07_7b7fb621_t328_f1__wrong | colgrep | `fuse_max` | ungrounded | wrong | 0.9703 | 1.0000 | 0.0000 | False | 1.0000 | 45 |
| base_07_7b7fb621_t328_f1__ambiguous | colgrep | `gte_only` | ambiguous | ambiguous | 0.9777 | 1.0000 | 0.0000 | False | 1.0000 | 45 |
| base_07_7b7fb621_t328_f1__ambiguous | colgrep | `fuse_max` | ambiguous | ambiguous | 0.9777 | 1.0000 | 0.0000 | False | 1.0000 | 45 |
| base_08_126e38fc_t1785_f0__correct | colgrep | `gte_only` | grounded | correct | 0.9799 | 1.0000 | 0.0000 | False | 0.9984 | 216 |
| base_08_126e38fc_t1785_f0__correct | colgrep | `fuse_max` | grounded | correct | 0.9799 | 1.0000 | 0.0000 | False | 0.9984 | 216 |
| base_08_126e38fc_t1785_f0__wrong | colgrep | `gte_only` | ungrounded | wrong | 0.9790 | 1.0000 | 0.0000 | False | 0.9982 | 216 |
| base_08_126e38fc_t1785_f0__wrong | colgrep | `fuse_max` | ungrounded | wrong | 0.9790 | 1.0000 | 0.0000 | False | 0.9982 | 216 |
| base_08_126e38fc_t1785_f0__ambiguous | colgrep | `gte_only` | ambiguous | ambiguous | 0.9798 | 1.0000 | 0.0000 | False | 0.9984 | 216 |
| base_08_126e38fc_t1785_f0__ambiguous | colgrep | `fuse_max` | ambiguous | ambiguous | 0.9798 | 1.0000 | 0.0000 | False | 0.9984 | 216 |
| base_09_0b1422bb_t571_f0__correct | colgrep | `gte_only` | grounded | correct | 0.9567 | 1.0000 | 0.0000 | False | 0.9978 | 32 |
| base_09_0b1422bb_t571_f0__correct | colgrep | `fuse_max` | grounded | correct | 0.9567 | 1.0000 | 0.0000 | False | 0.9978 | 32 |
| base_09_0b1422bb_t571_f0__wrong | colgrep | `gte_only` | ungrounded | wrong | 0.9546 | 1.0000 | 0.0000 | False | 0.9979 | 32 |
| base_09_0b1422bb_t571_f0__wrong | colgrep | `fuse_max` | ungrounded | wrong | 0.9546 | 1.0000 | 0.0000 | False | 0.9979 | 32 |
| base_09_0b1422bb_t571_f0__ambiguous | colgrep | `gte_only` | ambiguous | ambiguous | 0.9566 | 1.0000 | 0.0000 | False | 0.9978 | 32 |
| base_09_0b1422bb_t571_f0__ambiguous | colgrep | `fuse_max` | ambiguous | ambiguous | 0.9566 | 1.0000 | 0.0000 | False | 0.9978 | 32 |
| base_10_d7fd7127_t1579_f0__correct | colgrep | `gte_only` | grounded | correct | 0.9799 | 1.0000 | 0.0000 | False | 0.9948 | 153 |
| base_10_d7fd7127_t1579_f0__correct | colgrep | `fuse_max` | grounded | correct | 0.9799 | 1.0000 | 0.0000 | False | 0.9948 | 153 |
| base_10_d7fd7127_t1579_f0__wrong | colgrep | `gte_only` | ungrounded | wrong | 0.9669 | 1.0000 | 0.0000 | False | 0.9942 | 153 |
| base_10_d7fd7127_t1579_f0__wrong | colgrep | `fuse_max` | ungrounded | wrong | 0.9669 | 1.0000 | 0.0000 | False | 0.9942 | 153 |
| base_10_d7fd7127_t1579_f0__ambiguous | colgrep | `gte_only` | ambiguous | ambiguous | 0.9797 | 1.0000 | 0.0000 | False | 0.9948 | 153 |
| base_10_d7fd7127_t1579_f0__ambiguous | colgrep | `fuse_max` | ambiguous | ambiguous | 0.9797 | 1.0000 | 0.0000 | False | 0.9948 | 153 |
| base_11_f0afeaf2_t1418_f1__correct | colgrep | `gte_only` | grounded | correct | 0.9846 | 1.0000 | 0.0000 | False | 0.9960 | 152 |
| base_11_f0afeaf2_t1418_f1__correct | colgrep | `fuse_max` | grounded | correct | 0.9846 | 1.0000 | 0.0000 | False | 0.9960 | 152 |
| base_11_f0afeaf2_t1418_f1__wrong | colgrep | `gte_only` | ungrounded | wrong | 0.9705 | 1.0000 | 0.0000 | False | 0.9962 | 152 |
| base_11_f0afeaf2_t1418_f1__wrong | colgrep | `fuse_max` | ungrounded | wrong | 0.9705 | 1.0000 | 0.0000 | False | 0.9962 | 152 |
| base_11_f0afeaf2_t1418_f1__ambiguous | colgrep | `gte_only` | ambiguous | ambiguous | 0.9843 | 1.0000 | 0.0000 | False | 0.9961 | 152 |
| base_11_f0afeaf2_t1418_f1__ambiguous | colgrep | `fuse_max` | ambiguous | ambiguous | 0.9843 | 1.0000 | 0.0000 | False | 0.9961 | 152 |
| base_12_f0afeaf2_t955_f2__correct | colgrep | `gte_only` | grounded | correct | 0.9760 | 1.0000 | 0.0000 | False | 0.9928 | 126 |
| base_12_f0afeaf2_t955_f2__correct | colgrep | `fuse_max` | grounded | correct | 0.9760 | 1.0000 | 0.0000 | False | 0.9928 | 126 |
| base_12_f0afeaf2_t955_f2__wrong | colgrep | `gte_only` | ungrounded | wrong | 0.9669 | 1.0000 | 0.0000 | False | 1.0000 | 126 |
| base_12_f0afeaf2_t955_f2__wrong | colgrep | `fuse_max` | ungrounded | wrong | 0.9669 | 1.0000 | 0.0000 | False | 1.0000 | 126 |
| base_12_f0afeaf2_t955_f2__ambiguous | colgrep | `gte_only` | ambiguous | ambiguous | 0.9775 | 1.0000 | 0.0000 | False | 0.9945 | 126 |
| base_12_f0afeaf2_t955_f2__ambiguous | colgrep | `fuse_max` | ambiguous | ambiguous | 0.9775 | 1.0000 | 0.0000 | False | 0.9945 | 126 |
| base_13_d7fd7127_t1553_f0__correct | colgrep | `gte_only` | grounded | correct | 0.9837 | 1.0000 | 0.0000 | False | 0.9999 | 153 |
| base_13_d7fd7127_t1553_f0__correct | colgrep | `fuse_max` | grounded | correct | 0.9837 | 1.0000 | 0.0000 | False | 0.9999 | 153 |
| base_13_d7fd7127_t1553_f0__wrong | colgrep | `gte_only` | ungrounded | wrong | 0.9676 | 1.0000 | 0.0000 | False | 1.0000 | 153 |
| base_13_d7fd7127_t1553_f0__wrong | colgrep | `fuse_max` | ungrounded | wrong | 0.9676 | 1.0000 | 0.0000 | False | 1.0000 | 153 |
| base_13_d7fd7127_t1553_f0__ambiguous | colgrep | `gte_only` | ambiguous | ambiguous | 0.9837 | 1.0000 | 0.0000 | False | 1.0000 | 153 |
| base_13_d7fd7127_t1553_f0__ambiguous | colgrep | `fuse_max` | ambiguous | ambiguous | 0.9837 | 1.0000 | 0.0000 | False | 1.0000 | 153 |
| base_14_f0afeaf2_t2825_f0__correct | colgrep | `gte_only` | grounded | correct | 0.9809 | 1.0000 | 0.0000 | False | 1.0000 | 107 |
| base_14_f0afeaf2_t2825_f0__correct | colgrep | `fuse_max` | grounded | correct | 0.9809 | 1.0000 | 0.0000 | False | 1.0000 | 107 |
| base_14_f0afeaf2_t2825_f0__wrong | colgrep | `gte_only` | ungrounded | wrong | 0.9716 | 1.0000 | 0.0000 | False | 1.0000 | 107 |
| base_14_f0afeaf2_t2825_f0__wrong | colgrep | `fuse_max` | ungrounded | wrong | 0.9716 | 1.0000 | 0.0000 | False | 1.0000 | 107 |
| base_14_f0afeaf2_t2825_f0__ambiguous | colgrep | `gte_only` | ambiguous | ambiguous | 0.9806 | 1.0000 | 0.0000 | False | 0.9988 | 107 |
| base_14_f0afeaf2_t2825_f0__ambiguous | colgrep | `fuse_max` | ambiguous | ambiguous | 0.9806 | 1.0000 | 0.0000 | False | 0.9999 | 107 |
| base_15_91cf0ec9_t419_f2__correct | colgrep | `gte_only` | grounded | correct | 0.9829 | 1.0000 | 0.0000 | False | 0.9992 | 94 |
| base_15_91cf0ec9_t419_f2__correct | colgrep | `fuse_max` | grounded | correct | 0.9829 | 1.0000 | 0.0000 | False | 0.9992 | 94 |
| base_15_91cf0ec9_t419_f2__wrong | colgrep | `gte_only` | ungrounded | wrong | 0.9781 | 1.0000 | 0.0000 | False | 0.9999 | 94 |
| base_15_91cf0ec9_t419_f2__wrong | colgrep | `fuse_max` | ungrounded | wrong | 0.9781 | 1.0000 | 0.0000 | False | 1.0000 | 94 |
| base_15_91cf0ec9_t419_f2__ambiguous | colgrep | `gte_only` | ambiguous | ambiguous | 0.9824 | 1.0000 | 0.0000 | False | 0.9992 | 94 |
| base_15_91cf0ec9_t419_f2__ambiguous | colgrep | `fuse_max` | ambiguous | ambiguous | 0.9824 | 1.0000 | 0.0000 | False | 0.9992 | 94 |
| base_16_8a0f6f68_t1002_f0__correct | colgrep | `gte_only` | grounded | correct | 0.9802 | 1.0000 | 0.0000 | False | 0.9936 | 117 |
| base_16_8a0f6f68_t1002_f0__correct | colgrep | `fuse_max` | grounded | correct | 0.9802 | 1.0000 | 0.0000 | False | 0.9936 | 117 |
| base_16_8a0f6f68_t1002_f0__wrong | colgrep | `gte_only` | ungrounded | wrong | 0.9736 | 1.0000 | 0.0000 | False | 1.0000 | 117 |
| base_16_8a0f6f68_t1002_f0__wrong | colgrep | `fuse_max` | ungrounded | wrong | 0.9736 | 1.0000 | 0.0000 | False | 1.0000 | 117 |
| base_16_8a0f6f68_t1002_f0__ambiguous | colgrep | `gte_only` | ambiguous | ambiguous | 0.9799 | 1.0000 | 0.0000 | False | 0.9964 | 117 |
| base_16_8a0f6f68_t1002_f0__ambiguous | colgrep | `fuse_max` | ambiguous | ambiguous | 0.9799 | 1.0000 | 0.0000 | False | 0.9964 | 117 |
| base_17_126e38fc_t712_f0__correct | colgrep | `gte_only` | grounded | correct | 0.9799 | 1.0000 | 0.0000 | False | 0.9969 | 75 |
| base_17_126e38fc_t712_f0__correct | colgrep | `fuse_max` | grounded | correct | 0.9799 | 1.0000 | 0.0000 | False | 0.9969 | 75 |
| base_17_126e38fc_t712_f0__wrong | colgrep | `gte_only` | ungrounded | wrong | 0.9762 | 1.0000 | 0.0000 | False | 0.9958 | 75 |
| base_17_126e38fc_t712_f0__wrong | colgrep | `fuse_max` | ungrounded | wrong | 0.9762 | 1.0000 | 0.0000 | False | 0.9958 | 75 |
| base_17_126e38fc_t712_f0__ambiguous | colgrep | `gte_only` | ambiguous | ambiguous | 0.9799 | 1.0000 | 0.0000 | False | 0.9969 | 75 |
| base_17_126e38fc_t712_f0__ambiguous | colgrep | `fuse_max` | ambiguous | ambiguous | 0.9799 | 1.0000 | 0.0000 | False | 0.9969 | 75 |
| base_18_f0afeaf2_t457_f0__correct | colgrep | `gte_only` | grounded | correct | 0.9734 | 1.0000 | 0.0000 | False | 0.9999 | 92 |
| base_18_f0afeaf2_t457_f0__correct | colgrep | `fuse_max` | grounded | correct | 0.9734 | 1.0000 | 0.0000 | False | 0.9999 | 92 |
| base_18_f0afeaf2_t457_f0__wrong | colgrep | `gte_only` | ungrounded | wrong | 0.9656 | 1.0000 | 0.0000 | False | 0.9965 | 92 |
| base_18_f0afeaf2_t457_f0__wrong | colgrep | `fuse_max` | ungrounded | wrong | 0.9656 | 1.0000 | 0.0000 | False | 0.9965 | 92 |
| base_18_f0afeaf2_t457_f0__ambiguous | colgrep | `gte_only` | ambiguous | ambiguous | 0.9752 | 1.0000 | 0.0000 | False | 0.9958 | 92 |
| base_18_f0afeaf2_t457_f0__ambiguous | colgrep | `fuse_max` | ambiguous | ambiguous | 0.9752 | 1.0000 | 0.0000 | False | 0.9958 | 92 |
| base_19_5e875b7e_t1360_f0__correct | colgrep | `gte_only` | grounded | correct | 0.9695 | 1.0000 | 0.0000 | False | 0.9942 | 23 |
| base_19_5e875b7e_t1360_f0__correct | colgrep | `fuse_max` | grounded | correct | 0.9695 | 1.0000 | 0.0000 | False | 0.9942 | 23 |
| base_19_5e875b7e_t1360_f0__wrong | colgrep | `gte_only` | ungrounded | wrong | 0.9659 | 1.0000 | 0.0000 | False | 0.9973 | 23 |
| base_19_5e875b7e_t1360_f0__wrong | colgrep | `fuse_max` | ungrounded | wrong | 0.9659 | 1.0000 | 0.0000 | False | 0.9973 | 23 |
| base_19_5e875b7e_t1360_f0__ambiguous | colgrep | `gte_only` | ambiguous | ambiguous | 0.9693 | 1.0000 | 0.0000 | False | 0.9966 | 23 |
| base_19_5e875b7e_t1360_f0__ambiguous | colgrep | `fuse_max` | ambiguous | ambiguous | 0.9693 | 1.0000 | 0.0000 | False | 0.9966 | 23 |
| base_20_d7fd7127_t1617_f0__correct | colgrep | `gte_only` | grounded | correct | 0.9868 | 1.0000 | 0.0000 | False | 0.9989 | 160 |
| base_20_d7fd7127_t1617_f0__correct | colgrep | `fuse_max` | grounded | correct | 0.9868 | 1.0000 | 0.0000 | False | 0.9989 | 160 |
| base_20_d7fd7127_t1617_f0__wrong | colgrep | `gte_only` | ungrounded | wrong | 0.9862 | 1.0000 | 0.0000 | False | 0.9990 | 160 |
| base_20_d7fd7127_t1617_f0__wrong | colgrep | `fuse_max` | ungrounded | wrong | 0.9862 | 1.0000 | 0.0000 | False | 0.9990 | 160 |
| base_20_d7fd7127_t1617_f0__ambiguous | colgrep | `gte_only` | ambiguous | ambiguous | 0.9865 | 1.0000 | 0.0000 | False | 0.9989 | 160 |
| base_20_d7fd7127_t1617_f0__ambiguous | colgrep | `fuse_max` | ambiguous | ambiguous | 0.9865 | 1.0000 | 0.0000 | False | 0.9989 | 160 |
