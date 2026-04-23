# Coding-Agent Groundedness: Scorer Stack x Chunker

> SKIP — the flagship cell (colgrep x fuse_mean) was not evaluated in this run (code encoder likely unavailable).

- run tag: `v2_full_sentence_packed_gte`
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
| sentence_packed | `gte_only` | 0.7917 | 1.0000 | 0.0000 | 0.0000 | n/a | 0.0000 | 2386.97 | 108.3 |

`*` marks cells that pass `reverse_context AUROC >= 0.90`.

## Score distribution diagnostics (reverse_context)

| chunker | scorer | grounded mean | grounded p10-p90 | ungrounded mean | ungrounded p10-p90 | separation | saturation@0.95 |
|---|---|---:|---|---:|---|---:|---:|
| sentence_packed | `gte_only` | 0.9780 | 0.9723-0.9850 | 0.9712 | 0.9638-0.9811 | +0.0067 | 0.9917 |

`separation` is mean(grounded) − mean(ungrounded) on `reverse_context`. `saturation@0.95` is the fraction of anchor cases (grounded + ungrounded) whose score ≥ 0.95 — a ceiling effect indicator.

## Tier metrics (transcripts_v1: correct / ambiguous / wrong)

`AUROC c-vs-w` = AUROC on `tier=correct` vs `tier=wrong`. `AUROC c-vs-a` = AUROC on `correct` vs `ambiguous` (hard bucket). `monotonicity` = fraction of base scenarios where `correct > ambiguous > wrong` holds on `reverse_context`. `partial mono` = fraction where `correct > wrong` holds (ignores ambiguous).

| chunker | scorer | AUROC c-vs-w | AUROC c-vs-a | AUROC c-vs-wa | monotonicity | partial mono | correct mean | ambiguous mean | wrong mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sentence_packed | `gte_only` | 0.7917 | 0.5503 | 0.6710 | 0.7500 | 0.9667 | 0.9780 | 0.9771 | 0.9712 |

## Stacking lift (fuse_* minus gte_only, per chunker)

| chunker | rule | AUROC lift | unused-delta lift | phantom precision lift |
|---|---|---:|---:|---:|
| sentence_packed | fuse_mean | n/a | n/a | n/a |
| sentence_packed | fuse_max | n/a | n/a | n/a |
| sentence_packed | fuse_min | n/a | n/a | n/a |

## Chunker lift (colgrep minus sentence_packed, per scorer)

| scorer | AUROC lift | unused-delta lift | phantom precision lift |
|---|---:|---:|---:|
| `gte_only` | n/a | n/a | n/a |

## Subcategory snapshot (per cell)

### `sentence_packed` x `gte_only`

| subcategory | count | reverse_context | coverage | unused ratio | max evidence |
|---|---:|---:|---:|---:|---:|
| ambiguous | 60 | 0.9771 | 1.0000 | 0.0000 | 0.9974 |
| correct | 60 | 0.9780 | 1.0000 | 0.0000 | 0.9979 |
| wrong | 60 | 0.9712 | 1.0000 | 0.0000 | 0.9979 |

## Per-case scores

| id | chunker | scorer | label | subcategory | reverse_context | coverage | unused | phantom | max evidence | units |
|---|---|---|---|---|---:|---:|---:|---|---:|---:|
| base_01_f0afeaf2_t952_f1__correct | sentence_packed | `gte_only` | grounded | correct | 0.9843 | 1.0000 | 0.0000 | False | 0.9981 | 145 |
| base_01_f0afeaf2_t952_f1__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9820 | 1.0000 | 0.0000 | False | 0.9999 | 145 |
| base_01_f0afeaf2_t952_f1__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9844 | 1.0000 | 0.0000 | False | 0.9981 | 145 |
| base_02_61e1f5a1_t258_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9725 | 1.0000 | 0.0000 | False | 0.9988 | 88 |
| base_02_61e1f5a1_t258_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9685 | 1.0000 | 0.0000 | False | 0.9981 | 88 |
| base_02_61e1f5a1_t258_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9724 | 1.0000 | 0.0000 | False | 0.9972 | 88 |
| base_03_f0afeaf2_t2082_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9774 | 1.0000 | 0.0000 | False | 0.9979 | 198 |
| base_03_f0afeaf2_t2082_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9705 | 1.0000 | 0.0000 | False | 0.9955 | 198 |
| base_03_f0afeaf2_t2082_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9776 | 1.0000 | 0.0000 | False | 0.9974 | 198 |
| base_04_8a0f6f68_t521_f3__correct | sentence_packed | `gte_only` | grounded | correct | 0.9850 | 1.0000 | 0.0000 | False | 0.9975 | 71 |
| base_04_8a0f6f68_t521_f3__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9811 | 1.0000 | 0.0000 | False | 0.9975 | 71 |
| base_04_8a0f6f68_t521_f3__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9851 | 1.0000 | 0.0000 | False | 0.9975 | 71 |
| base_05_126e38fc_t1159_f4__correct | sentence_packed | `gte_only` | grounded | correct | 0.9748 | 1.0000 | 0.0000 | False | 1.0000 | 130 |
| base_05_126e38fc_t1159_f4__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9726 | 1.0000 | 0.0000 | False | 1.0000 | 130 |
| base_05_126e38fc_t1159_f4__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9745 | 1.0000 | 0.0000 | False | 0.9896 | 130 |
| base_06_f0afeaf2_t1771_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9755 | 1.0000 | 0.0000 | False | 0.9929 | 199 |
| base_06_f0afeaf2_t1771_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9700 | 1.0000 | 0.0000 | False | 0.9705 | 199 |
| base_06_f0afeaf2_t1771_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9709 | 1.0000 | 0.0000 | False | 0.9674 | 199 |
| base_07_7b7fb621_t328_f1__correct | sentence_packed | `gte_only` | grounded | correct | 0.9776 | 1.0000 | 0.0000 | False | 0.9999 | 36 |
| base_07_7b7fb621_t328_f1__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9703 | 1.0000 | 0.0000 | False | 0.9999 | 36 |
| base_07_7b7fb621_t328_f1__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9777 | 1.0000 | 0.0000 | False | 1.0000 | 36 |
| base_08_126e38fc_t1785_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9807 | 1.0000 | 0.0000 | False | 0.9983 | 210 |
| base_08_126e38fc_t1785_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9792 | 1.0000 | 0.0000 | False | 0.9982 | 210 |
| base_08_126e38fc_t1785_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9806 | 1.0000 | 0.0000 | False | 0.9983 | 210 |
| base_09_0b1422bb_t571_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9584 | 1.0000 | 0.0000 | False | 0.9980 | 20 |
| base_09_0b1422bb_t571_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9565 | 1.0000 | 0.0000 | False | 0.9979 | 20 |
| base_09_0b1422bb_t571_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9582 | 1.0000 | 0.0000 | False | 0.9980 | 20 |
| base_10_d7fd7127_t1579_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9830 | 1.0000 | 0.0000 | False | 0.9951 | 218 |
| base_10_d7fd7127_t1579_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9729 | 1.0000 | 0.0000 | False | 0.9948 | 218 |
| base_10_d7fd7127_t1579_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9828 | 1.0000 | 0.0000 | False | 0.9951 | 218 |
| base_11_f0afeaf2_t1418_f1__correct | sentence_packed | `gte_only` | grounded | correct | 0.9850 | 1.0000 | 0.0000 | False | 0.9960 | 195 |
| base_11_f0afeaf2_t1418_f1__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9731 | 1.0000 | 0.0000 | False | 0.9964 | 195 |
| base_11_f0afeaf2_t1418_f1__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9848 | 1.0000 | 0.0000 | False | 0.9961 | 195 |
| base_12_f0afeaf2_t955_f2__correct | sentence_packed | `gte_only` | grounded | correct | 0.9757 | 1.0000 | 0.0000 | False | 0.9946 | 145 |
| base_12_f0afeaf2_t955_f2__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9673 | 1.0000 | 0.0000 | False | 1.0000 | 145 |
| base_12_f0afeaf2_t955_f2__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9781 | 1.0000 | 0.0000 | False | 0.9946 | 145 |
| base_13_d7fd7127_t1553_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9851 | 1.0000 | 0.0000 | False | 1.0000 | 218 |
| base_13_d7fd7127_t1553_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9720 | 1.0000 | 0.0000 | False | 1.0000 | 218 |
| base_13_d7fd7127_t1553_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9851 | 1.0000 | 0.0000 | False | 1.0000 | 218 |
| base_14_f0afeaf2_t2825_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9842 | 1.0000 | 0.0000 | False | 1.0000 | 200 |
| base_14_f0afeaf2_t2825_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9756 | 1.0000 | 0.0000 | False | 1.0000 | 200 |
| base_14_f0afeaf2_t2825_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9839 | 1.0000 | 0.0000 | False | 0.9992 | 200 |
| base_15_91cf0ec9_t419_f2__correct | sentence_packed | `gte_only` | grounded | correct | 0.9825 | 1.0000 | 0.0000 | False | 0.9995 | 110 |
| base_15_91cf0ec9_t419_f2__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9779 | 1.0000 | 0.0000 | False | 0.9998 | 110 |
| base_15_91cf0ec9_t419_f2__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9821 | 1.0000 | 0.0000 | False | 0.9995 | 110 |
| base_16_8a0f6f68_t1002_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9810 | 1.0000 | 0.0000 | False | 0.9956 | 93 |
| base_16_8a0f6f68_t1002_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9738 | 1.0000 | 0.0000 | False | 1.0000 | 93 |
| base_16_8a0f6f68_t1002_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9808 | 1.0000 | 0.0000 | False | 0.9971 | 93 |
| base_17_126e38fc_t712_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9797 | 1.0000 | 0.0000 | False | 0.9959 | 58 |
| base_17_126e38fc_t712_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9762 | 1.0000 | 0.0000 | False | 0.9945 | 58 |
| base_17_126e38fc_t712_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9797 | 1.0000 | 0.0000 | False | 0.9959 | 58 |
| base_18_f0afeaf2_t457_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9789 | 1.0000 | 0.0000 | False | 1.0000 | 89 |
| base_18_f0afeaf2_t457_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9697 | 1.0000 | 0.0000 | False | 0.9963 | 89 |
| base_18_f0afeaf2_t457_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9800 | 1.0000 | 0.0000 | False | 0.9956 | 89 |
| base_19_5e875b7e_t1360_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9783 | 1.0000 | 0.0000 | False | 0.9948 | 60 |
| base_19_5e875b7e_t1360_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9740 | 1.0000 | 0.0000 | False | 0.9971 | 60 |
| base_19_5e875b7e_t1360_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9782 | 1.0000 | 0.0000 | False | 0.9969 | 60 |
| base_20_d7fd7127_t1617_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9876 | 1.0000 | 0.0000 | False | 0.9988 | 226 |
| base_20_d7fd7127_t1617_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9870 | 1.0000 | 0.0000 | False | 0.9988 | 226 |
| base_20_d7fd7127_t1617_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9874 | 1.0000 | 0.0000 | False | 0.9988 | 226 |
| base_21_a4d14a56_t732_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9824 | 1.0000 | 0.0000 | False | 1.0000 | 93 |
| base_21_a4d14a56_t732_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9723 | 1.0000 | 0.0000 | False | 1.0000 | 93 |
| base_21_a4d14a56_t732_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9778 | 1.0000 | 0.0000 | False | 1.0000 | 93 |
| base_22_d7fd7127_t1209_f1__correct | sentence_packed | `gte_only` | grounded | correct | 0.9744 | 1.0000 | 0.0000 | False | 1.0000 | 165 |
| base_22_d7fd7127_t1209_f1__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9748 | 1.0000 | 0.0000 | False | 1.0000 | 165 |
| base_22_d7fd7127_t1209_f1__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9746 | 1.0000 | 0.0000 | False | 1.0000 | 165 |
| base_23_a4d14a56_t579_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9790 | 1.0000 | 0.0000 | False | 0.9975 | 68 |
| base_23_a4d14a56_t579_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9672 | 1.0000 | 0.0000 | False | 0.9981 | 68 |
| base_23_a4d14a56_t579_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9788 | 1.0000 | 0.0000 | False | 0.9976 | 68 |
| base_24_126e38fc_t1774_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9788 | 1.0000 | 0.0000 | False | 0.9999 | 210 |
| base_24_126e38fc_t1774_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9706 | 1.0000 | 0.0000 | False | 1.0000 | 210 |
| base_24_126e38fc_t1774_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9789 | 1.0000 | 0.0000 | False | 1.0000 | 210 |
| base_25_8a0f6f68_t1261_f1__correct | sentence_packed | `gte_only` | grounded | correct | 0.9757 | 1.0000 | 0.0000 | False | 1.0000 | 120 |
| base_25_8a0f6f68_t1261_f1__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9672 | 1.0000 | 0.0000 | False | 1.0000 | 120 |
| base_25_8a0f6f68_t1261_f1__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9740 | 1.0000 | 0.0000 | False | 1.0000 | 120 |
| base_26_f0afeaf2_t1530_f1__correct | sentence_packed | `gte_only` | grounded | correct | 0.9802 | 1.0000 | 0.0000 | False | 0.9989 | 195 |
| base_26_f0afeaf2_t1530_f1__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9804 | 1.0000 | 0.0000 | False | 0.9993 | 195 |
| base_26_f0afeaf2_t1530_f1__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9802 | 1.0000 | 0.0000 | False | 0.9989 | 195 |
| base_27_8a0f6f68_t1163_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9793 | 1.0000 | 0.0000 | False | 1.0000 | 112 |
| base_27_8a0f6f68_t1163_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9710 | 1.0000 | 0.0000 | False | 1.0000 | 112 |
| base_27_8a0f6f68_t1163_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9754 | 1.0000 | 0.0000 | False | 0.9997 | 112 |
| base_28_126e38fc_t1892_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9877 | 1.0000 | 0.0000 | False | 0.9978 | 223 |
| base_28_126e38fc_t1892_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9816 | 1.0000 | 0.0000 | False | 0.9976 | 223 |
| base_28_126e38fc_t1892_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9878 | 1.0000 | 0.0000 | False | 0.9978 | 223 |
| base_29_8a0f6f68_t1278_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9763 | 1.0000 | 0.0000 | False | 0.9999 | 126 |
| base_29_8a0f6f68_t1278_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9673 | 1.0000 | 0.0000 | False | 1.0000 | 126 |
| base_29_8a0f6f68_t1278_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9711 | 1.0000 | 0.0000 | False | 1.0000 | 126 |
| base_30_d7fd7127_t807_f1__correct | sentence_packed | `gte_only` | grounded | correct | 0.9816 | 1.0000 | 0.0000 | False | 0.9988 | 113 |
| base_30_d7fd7127_t807_f1__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9816 | 1.0000 | 0.0000 | False | 0.9988 | 113 |
| base_30_d7fd7127_t807_f1__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9796 | 1.0000 | 0.0000 | False | 1.0000 | 113 |
| base_31_8a0f6f68_t1301_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9738 | 1.0000 | 0.0000 | False | 1.0000 | 125 |
| base_31_8a0f6f68_t1301_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9666 | 1.0000 | 0.0000 | False | 1.0000 | 125 |
| base_31_8a0f6f68_t1301_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9720 | 1.0000 | 0.0000 | False | 1.0000 | 125 |
| base_32_126e38fc_t1007_f1__correct | sentence_packed | `gte_only` | grounded | correct | 0.9856 | 1.0000 | 0.0000 | False | 1.0000 | 102 |
| base_32_126e38fc_t1007_f1__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9739 | 1.0000 | 0.0000 | False | 1.0000 | 102 |
| base_32_126e38fc_t1007_f1__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9847 | 1.0000 | 0.0000 | False | 0.9999 | 102 |
| base_33_d7fd7127_t1397_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9850 | 1.0000 | 0.0000 | False | 0.9972 | 198 |
| base_33_d7fd7127_t1397_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9812 | 1.0000 | 0.0000 | False | 0.9973 | 198 |
| base_33_d7fd7127_t1397_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9822 | 1.0000 | 0.0000 | False | 0.9974 | 198 |
| base_34_5e875b7e_t1637_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9779 | 1.0000 | 0.0000 | False | 1.0000 | 61 |
| base_34_5e875b7e_t1637_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9743 | 1.0000 | 0.0000 | False | 1.0000 | 61 |
| base_34_5e875b7e_t1637_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9772 | 1.0000 | 0.0000 | False | 1.0000 | 61 |
| base_35_a4d14a56_t1005_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9767 | 1.0000 | 0.0000 | False | 0.9987 | 184 |
| base_35_a4d14a56_t1005_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9738 | 1.0000 | 0.0000 | False | 0.9976 | 184 |
| base_35_a4d14a56_t1005_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9763 | 1.0000 | 0.0000 | False | 0.9987 | 184 |
| base_36_a4d14a56_t693_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9826 | 1.0000 | 0.0000 | False | 1.0000 | 93 |
| base_36_a4d14a56_t693_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9697 | 1.0000 | 0.0000 | False | 1.0000 | 93 |
| base_36_a4d14a56_t693_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9785 | 1.0000 | 0.0000 | False | 1.0000 | 93 |
| base_37_126e38fc_t1936_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9819 | 1.0000 | 0.0000 | False | 0.9999 | 216 |
| base_37_126e38fc_t1936_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9698 | 1.0000 | 0.0000 | False | 1.0000 | 216 |
| base_37_126e38fc_t1936_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9798 | 1.0000 | 0.0000 | False | 0.9999 | 216 |
| base_38_a4d14a56_t869_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9727 | 1.0000 | 0.0000 | False | 1.0000 | 184 |
| base_38_a4d14a56_t869_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9642 | 1.0000 | 0.0000 | False | 1.0000 | 184 |
| base_38_a4d14a56_t869_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9703 | 1.0000 | 0.0000 | False | 1.0000 | 184 |
| base_39_5e875b7e_t2278_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9681 | 1.0000 | 0.0000 | False | 0.9951 | 68 |
| base_39_5e875b7e_t2278_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9677 | 1.0000 | 0.0000 | False | 0.9949 | 68 |
| base_39_5e875b7e_t2278_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9682 | 1.0000 | 0.0000 | False | 0.9950 | 68 |
| base_40_ed69b0cd_t395_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9857 | 1.0000 | 0.0000 | False | 1.0000 | 81 |
| base_40_ed69b0cd_t395_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9674 | 1.0000 | 0.0000 | False | 1.0000 | 81 |
| base_40_ed69b0cd_t395_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9852 | 1.0000 | 0.0000 | False | 1.0000 | 81 |
| base_41_a4d14a56_t994_f1__correct | sentence_packed | `gte_only` | grounded | correct | 0.9859 | 1.0000 | 0.0000 | False | 0.9908 | 184 |
| base_41_a4d14a56_t994_f1__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9829 | 1.0000 | 0.0000 | False | 0.9969 | 184 |
| base_41_a4d14a56_t994_f1__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9855 | 1.0000 | 0.0000 | False | 0.9907 | 184 |
| base_42_a4d14a56_t531_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9811 | 1.0000 | 0.0000 | False | 0.9965 | 67 |
| base_42_a4d14a56_t531_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9728 | 1.0000 | 0.0000 | False | 0.9967 | 67 |
| base_42_a4d14a56_t531_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9804 | 1.0000 | 0.0000 | False | 0.9967 | 67 |
| base_43_ed69b0cd_t364_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9832 | 1.0000 | 0.0000 | False | 0.9970 | 81 |
| base_43_ed69b0cd_t364_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9739 | 1.0000 | 0.0000 | False | 0.9982 | 81 |
| base_43_ed69b0cd_t364_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9778 | 1.0000 | 0.0000 | False | 0.9940 | 81 |
| base_44_a4d14a56_t568_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9825 | 1.0000 | 0.0000 | False | 0.9975 | 67 |
| base_44_a4d14a56_t568_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9655 | 1.0000 | 0.0000 | False | 0.9982 | 67 |
| base_44_a4d14a56_t568_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9804 | 1.0000 | 0.0000 | False | 0.9976 | 67 |
| base_45_ed69b0cd_t367_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9804 | 1.0000 | 0.0000 | False | 1.0000 | 81 |
| base_45_ed69b0cd_t367_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9732 | 1.0000 | 0.0000 | False | 1.0000 | 81 |
| base_45_ed69b0cd_t367_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9794 | 1.0000 | 0.0000 | False | 1.0000 | 81 |
| base_46_d7fd7127_t317_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9788 | 1.0000 | 0.0000 | False | 0.9939 | 81 |
| base_46_d7fd7127_t317_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9684 | 1.0000 | 0.0000 | False | 0.9919 | 81 |
| base_46_d7fd7127_t317_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9773 | 1.0000 | 0.0000 | False | 0.9939 | 81 |
| base_47_0b1422bb_t847_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9507 | 1.0000 | 0.0000 | False | 0.9999 | 25 |
| base_47_0b1422bb_t847_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9463 | 1.0000 | 0.0000 | False | 0.9966 | 25 |
| base_47_0b1422bb_t847_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9502 | 1.0000 | 0.0000 | False | 0.9997 | 25 |
| base_48_ed69b0cd_t10_f2__correct | sentence_packed | `gte_only` | grounded | correct | 0.9600 | 1.0000 | 0.0000 | False | 0.9890 | 19 |
| base_48_ed69b0cd_t10_f2__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9524 | 1.0000 | 0.0000 | False | 0.9902 | 19 |
| base_48_ed69b0cd_t10_f2__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9598 | 1.0000 | 0.0000 | False | 0.9892 | 19 |
| base_49_fa052757_t213_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9709 | 1.0000 | 0.0000 | False | 0.9988 | 44 |
| base_49_fa052757_t213_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9597 | 1.0000 | 0.0000 | False | 0.9989 | 44 |
| base_49_fa052757_t213_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9706 | 1.0000 | 0.0000 | False | 0.9988 | 44 |
| base_50_0b1422bb_t839_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9615 | 1.0000 | 0.0000 | False | 0.9999 | 25 |
| base_50_0b1422bb_t839_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9578 | 1.0000 | 0.0000 | False | 0.9999 | 25 |
| base_50_0b1422bb_t839_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9587 | 1.0000 | 0.0000 | False | 0.9999 | 25 |
| base_51_5e875b7e_t1929_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9741 | 1.0000 | 0.0000 | False | 0.9999 | 61 |
| base_51_5e875b7e_t1929_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9608 | 1.0000 | 0.0000 | False | 1.0000 | 61 |
| base_51_5e875b7e_t1929_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9727 | 1.0000 | 0.0000 | False | 1.0000 | 61 |
| base_52_7b7fb621_t310_f3__correct | sentence_packed | `gte_only` | grounded | correct | 0.9835 | 1.0000 | 0.0000 | False | 0.9989 | 37 |
| base_52_7b7fb621_t310_f3__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9754 | 1.0000 | 0.0000 | False | 0.9984 | 37 |
| base_52_7b7fb621_t310_f3__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9850 | 1.0000 | 0.0000 | False | 1.0000 | 37 |
| base_53_7b7fb621_t347_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9785 | 1.0000 | 0.0000 | False | 0.9985 | 36 |
| base_53_7b7fb621_t347_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9738 | 1.0000 | 0.0000 | False | 0.9993 | 36 |
| base_53_7b7fb621_t347_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9788 | 1.0000 | 0.0000 | False | 0.9991 | 36 |
| base_54_5e875b7e_t630_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9725 | 1.0000 | 0.0000 | False | 0.9932 | 59 |
| base_54_5e875b7e_t630_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9710 | 1.0000 | 0.0000 | False | 0.9946 | 59 |
| base_54_5e875b7e_t630_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9713 | 1.0000 | 0.0000 | False | 0.9924 | 59 |
| base_55_0b1422bb_t652_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9731 | 1.0000 | 0.0000 | False | 0.9977 | 23 |
| base_55_0b1422bb_t652_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9645 | 1.0000 | 0.0000 | False | 0.9975 | 23 |
| base_55_0b1422bb_t652_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9731 | 1.0000 | 0.0000 | False | 0.9978 | 23 |
| base_56_7b7fb621_t406_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9745 | 1.0000 | 0.0000 | False | 1.0000 | 47 |
| base_56_7b7fb621_t406_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9720 | 1.0000 | 0.0000 | False | 0.9999 | 47 |
| base_56_7b7fb621_t406_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9742 | 1.0000 | 0.0000 | False | 1.0000 | 47 |
| base_57_91cf0ec9_t84_f10__correct | sentence_packed | `gte_only` | grounded | correct | 0.9789 | 1.0000 | 0.0000 | False | 0.9874 | 40 |
| base_57_91cf0ec9_t84_f10__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9728 | 1.0000 | 0.0000 | False | 0.9983 | 40 |
| base_57_91cf0ec9_t84_f10__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9787 | 1.0000 | 0.0000 | False | 0.9961 | 40 |
| base_58_7b7fb621_t237_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9779 | 1.0000 | 0.0000 | False | 0.9997 | 23 |
| base_58_7b7fb621_t237_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9690 | 1.0000 | 0.0000 | False | 0.9999 | 23 |
| base_58_7b7fb621_t237_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9772 | 1.0000 | 0.0000 | False | 0.9996 | 23 |
| base_59_ee0ad405_t137_f0__correct | sentence_packed | `gte_only` | grounded | correct | 0.9761 | 1.0000 | 0.0000 | False | 0.9999 | 33 |
| base_59_ee0ad405_t137_f0__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9712 | 1.0000 | 0.0000 | False | 0.9999 | 33 |
| base_59_ee0ad405_t137_f0__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9759 | 1.0000 | 0.0000 | False | 1.0000 | 33 |
| base_60_7b7fb621_t178_f1__correct | sentence_packed | `gte_only` | grounded | correct | 0.9808 | 1.0000 | 0.0000 | False | 0.9999 | 21 |
| base_60_7b7fb621_t178_f1__wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9786 | 1.0000 | 0.0000 | False | 0.9999 | 21 |
| base_60_7b7fb621_t178_f1__ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9807 | 1.0000 | 0.0000 | False | 0.9999 | 21 |
