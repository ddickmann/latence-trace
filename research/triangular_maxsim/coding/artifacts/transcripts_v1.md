# Coding-Agent Groundedness: Scorer Stack x Chunker

> NO — the flagship colgrep x fuse_mean cell did not clear the RAG-style gate: reverse_context AUROC=0.625, phantom-API precision@0.35=n/a, ungrounded minus grounded unused-ratio delta=+0.0000. Stacking lift vs gte_only=-0.0938. Best fusion rule on colgrep is `fuse_max` (AUROC=0.719); consider flipping the flagship.

- run tag: `transcripts_v1`
- case bank: `transcripts_v1`
- primary scorer: `lightonai/GTE-ModernColBERT-v1`
- orthogonal scorer: `lightonai/LateOn-Code-edge` (available: `True`)
- shared chunk budget: `256` tokens (GTE tokenizer)
- max SWE-bench instances: `15` (ignored for `transcripts_v1`)
- phantom threshold: `0.35`
- flagship cell: `colgrep` x `fuse_mean` (present: `True`)

## Headline matrix

| chunker | scorer | AUROC | grounded cov | cov delta | unused delta | phantom@thr | grounded held-out rate | p95 ms | support units |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sentence_packed | `gte_only` | 0.7344 | 1.0000 | 0.0000 | 0.0000 | n/a | 0.0000 | 2915.58 | 84.4 |
| sentence_packed | `code_only` | 0.6094 | 1.0000 | 0.0000 | 0.0000 | n/a | 0.0000 | 2321.68 | 84.4 |
| sentence_packed | `fuse_mean` | 0.6562 | 1.0000 | 0.0000 | 0.0000 | n/a | 0.0000 | 2800.93 | 84.4 |
| sentence_packed | `fuse_max` | 0.7344 | 1.0000 | 0.0000 | 0.0000 | n/a | 0.0000 | 2800.93 | 84.4 |
| sentence_packed | `fuse_min` | 0.6094 | 1.0000 | 0.0000 | 0.0000 | n/a | 0.0000 | 2800.93 | 84.4 |
| colgrep | `gte_only` | 0.7188 | 1.0000 | 0.0000 | 0.0000 | n/a | 0.0000 | 2015.97 | 84.5 |
| colgrep | `code_only` | 0.5781 | 1.0000 | 0.0000 | 0.0000 | n/a | 0.0000 | 1798.48 | 84.5 |
| colgrep | `fuse_mean` (flagship) | 0.6250 | 1.0000 | 0.0000 | 0.0000 | n/a | 0.0000 | 1988.58 | 84.5 |
| colgrep | `fuse_max` | 0.7188 | 1.0000 | 0.0000 | 0.0000 | n/a | 0.0000 | 1988.58 | 84.5 |
| colgrep | `fuse_min` | 0.5781 | 1.0000 | 0.0000 | 0.0000 | n/a | 0.0000 | 1988.58 | 84.5 |

`*` marks cells that pass `reverse_context AUROC >= 0.90`.

## Score distribution diagnostics (reverse_context)

| chunker | scorer | grounded mean | grounded p10-p90 | ungrounded mean | ungrounded p10-p90 | separation | saturation@0.95 |
|---|---|---:|---|---:|---|---:|---:|
| sentence_packed | `gte_only` | 0.9773 | 0.9691-0.9834 | 0.9723 | 0.9634-0.9795 | +0.0050 | 1.0000 |
| sentence_packed | `code_only` | 0.9325 | 0.9057-0.9498 | 0.9283 | 0.9089-0.9446 | +0.0042 | 0.1250 |
| sentence_packed | `fuse_mean` | 0.9549 | 0.9374-0.9656 | 0.9503 | 0.9344-0.9620 | +0.0046 | 0.6250 |
| sentence_packed | `fuse_max` | 0.9773 | 0.9691-0.9834 | 0.9723 | 0.9634-0.9795 | +0.0050 | 1.0000 |
| sentence_packed | `fuse_min` | 0.9325 | 0.9057-0.9498 | 0.9283 | 0.9089-0.9446 | +0.0042 | 0.1250 |
| colgrep | `gte_only` | 0.9753 | 0.9604-0.9836 | 0.9707 | 0.9562-0.9790 | +0.0046 | 1.0000 |
| colgrep | `code_only` | 0.9285 | 0.8949-0.9499 | 0.9252 | 0.9041-0.9436 | +0.0033 | 0.1250 |
| colgrep | `fuse_mean` | 0.9519 | 0.9276-0.9654 | 0.9479 | 0.9302-0.9602 | +0.0040 | 0.6250 |
| colgrep | `fuse_max` | 0.9753 | 0.9604-0.9836 | 0.9707 | 0.9562-0.9790 | +0.0046 | 1.0000 |
| colgrep | `fuse_min` | 0.9285 | 0.8949-0.9499 | 0.9252 | 0.9041-0.9436 | +0.0033 | 0.1250 |

`separation` is mean(grounded) − mean(ungrounded) on `reverse_context`. `saturation@0.95` is the fraction of anchor cases (grounded + ungrounded) whose score ≥ 0.95 — a ceiling effect indicator.

## Tier metrics (transcripts_v1: correct / ambiguous / wrong)

`AUROC c-vs-w` = AUROC on `tier=correct` vs `tier=wrong`. `AUROC c-vs-a` = AUROC on `correct` vs `ambiguous` (hard bucket). `monotonicity` = fraction of base scenarios where `correct > ambiguous > wrong` holds on `reverse_context`. `partial mono` = fraction where `correct > wrong` holds (ignores ambiguous).

| chunker | scorer | AUROC c-vs-w | AUROC c-vs-a | AUROC c-vs-wa | monotonicity | partial mono | correct mean | ambiguous mean | wrong mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sentence_packed | `gte_only` | 0.7344 | 0.5625 | 0.6771 | 0.2500 | 1.0000 | 0.9773 | 0.9737 | 0.9723 |
| sentence_packed | `code_only` | 0.6094 | 0.5312 | 0.5833 | 0.5000 | 0.8750 | 0.9325 | 0.9308 | 0.9283 |
| sentence_packed | `fuse_mean` | 0.6562 | 0.5000 | 0.6042 | 0.2500 | 0.8750 | 0.9549 | 0.9523 | 0.9503 |
| sentence_packed | `fuse_max` | 0.7344 | 0.5625 | 0.6771 | 0.2500 | 1.0000 | 0.9773 | 0.9737 | 0.9723 |
| sentence_packed | `fuse_min` | 0.6094 | 0.5312 | 0.5833 | 0.5000 | 0.8750 | 0.9325 | 0.9308 | 0.9283 |
| colgrep | `gte_only` | 0.7188 | 0.5625 | 0.6667 | 0.2500 | 1.0000 | 0.9753 | 0.9729 | 0.9707 |
| colgrep | `code_only` | 0.5781 | 0.5625 | 0.5729 | 0.2500 | 0.7500 | 0.9285 | 0.9269 | 0.9252 |
| colgrep | `fuse_mean` | 0.6250 | 0.5312 | 0.5938 | 0.2500 | 0.8750 | 0.9519 | 0.9499 | 0.9479 |
| colgrep | `fuse_max` | 0.7188 | 0.5625 | 0.6667 | 0.2500 | 1.0000 | 0.9753 | 0.9729 | 0.9707 |
| colgrep | `fuse_min` | 0.5781 | 0.5625 | 0.5729 | 0.2500 | 0.7500 | 0.9285 | 0.9269 | 0.9252 |

## Stacking lift (fuse_* minus gte_only, per chunker)

| chunker | rule | AUROC lift | unused-delta lift | phantom precision lift |
|---|---|---:|---:|---:|
| sentence_packed | fuse_mean | -0.0781 | +0.0000 | n/a |
| sentence_packed | fuse_max | +0.0000 | +0.0000 | n/a |
| sentence_packed | fuse_min | -0.1250 | +0.0000 | n/a |
| colgrep | fuse_mean | -0.0938 | +0.0000 | n/a |
| colgrep | fuse_max | +0.0000 | +0.0000 | n/a |
| colgrep | fuse_min | -0.1406 | +0.0000 | n/a |

## Chunker lift (colgrep minus sentence_packed, per scorer)

| scorer | AUROC lift | unused-delta lift | phantom precision lift |
|---|---:|---:|---:|
| `gte_only` | -0.0156 | +0.0000 | n/a |
| `code_only` | -0.0312 | +0.0000 | n/a |
| `fuse_mean` | -0.0312 | +0.0000 | n/a |
| `fuse_max` | -0.0156 | +0.0000 | n/a |
| `fuse_min` | -0.0312 | +0.0000 | n/a |

## Subcategory snapshot (per cell)

### `sentence_packed` x `gte_only`

| subcategory | count | reverse_context | coverage | unused ratio | max evidence |
|---|---:|---:|---:|---:|---:|
| ambiguous | 4 | 0.9737 | 1.0000 | 0.0000 | 0.9977 |
| correct | 8 | 0.9773 | 1.0000 | 0.0000 | 0.9986 |
| wrong | 8 | 0.9723 | 1.0000 | 0.0000 | 0.9964 |

### `sentence_packed` x `code_only`

| subcategory | count | reverse_context | coverage | unused ratio | max evidence |
|---|---:|---:|---:|---:|---:|
| ambiguous | 4 | 0.9308 | 1.0000 | 0.0000 | 0.9886 |
| correct | 8 | 0.9325 | 1.0000 | 0.0000 | 0.9910 |
| wrong | 8 | 0.9283 | 1.0000 | 0.0000 | 0.9920 |

### `sentence_packed` x `fuse_mean`

| subcategory | count | reverse_context | coverage | unused ratio | max evidence |
|---|---:|---:|---:|---:|---:|
| ambiguous | 4 | 0.9523 | 1.0000 | 0.0000 | 0.9931 |
| correct | 8 | 0.9549 | 1.0000 | 0.0000 | 0.9948 |
| wrong | 8 | 0.9503 | 1.0000 | 0.0000 | 0.9942 |

### `sentence_packed` x `fuse_max`

| subcategory | count | reverse_context | coverage | unused ratio | max evidence |
|---|---:|---:|---:|---:|---:|
| ambiguous | 4 | 0.9737 | 1.0000 | 0.0000 | 0.9977 |
| correct | 8 | 0.9773 | 1.0000 | 0.0000 | 0.9986 |
| wrong | 8 | 0.9723 | 1.0000 | 0.0000 | 0.9968 |

### `sentence_packed` x `fuse_min`

| subcategory | count | reverse_context | coverage | unused ratio | max evidence |
|---|---:|---:|---:|---:|---:|
| ambiguous | 4 | 0.9308 | 1.0000 | 0.0000 | 0.9886 |
| correct | 8 | 0.9325 | 1.0000 | 0.0000 | 0.9910 |
| wrong | 8 | 0.9283 | 1.0000 | 0.0000 | 0.9916 |

### `colgrep` x `gte_only`

| subcategory | count | reverse_context | coverage | unused ratio | max evidence |
|---|---:|---:|---:|---:|---:|
| ambiguous | 4 | 0.9729 | 1.0000 | 0.0000 | 0.9976 |
| correct | 8 | 0.9753 | 1.0000 | 0.0000 | 0.9985 |
| wrong | 8 | 0.9707 | 1.0000 | 0.0000 | 0.9965 |

### `colgrep` x `code_only`

| subcategory | count | reverse_context | coverage | unused ratio | max evidence |
|---|---:|---:|---:|---:|---:|
| ambiguous | 4 | 0.9269 | 1.0000 | 0.0000 | 0.9884 |
| correct | 8 | 0.9285 | 1.0000 | 0.0000 | 0.9895 |
| wrong | 8 | 0.9252 | 1.0000 | 0.0000 | 0.9906 |

### `colgrep` x `fuse_mean`

| subcategory | count | reverse_context | coverage | unused ratio | max evidence |
|---|---:|---:|---:|---:|---:|
| ambiguous | 4 | 0.9499 | 1.0000 | 0.0000 | 0.9930 |
| correct | 8 | 0.9519 | 1.0000 | 0.0000 | 0.9940 |
| wrong | 8 | 0.9479 | 1.0000 | 0.0000 | 0.9935 |

### `colgrep` x `fuse_max`

| subcategory | count | reverse_context | coverage | unused ratio | max evidence |
|---|---:|---:|---:|---:|---:|
| ambiguous | 4 | 0.9729 | 1.0000 | 0.0000 | 0.9976 |
| correct | 8 | 0.9753 | 1.0000 | 0.0000 | 0.9985 |
| wrong | 8 | 0.9707 | 1.0000 | 0.0000 | 0.9967 |

### `colgrep` x `fuse_min`

| subcategory | count | reverse_context | coverage | unused ratio | max evidence |
|---|---:|---:|---:|---:|---:|
| ambiguous | 4 | 0.9269 | 1.0000 | 0.0000 | 0.9884 |
| correct | 8 | 0.9285 | 1.0000 | 0.0000 | 0.9895 |
| wrong | 8 | 0.9252 | 1.0000 | 0.0000 | 0.9904 |

## Held-out support units (flagship cell)

Flagship cell: `colgrep` x `fuse_mean`. Consensus held-out ids are the intersection of GTE's and LateOn-Code-edge's independently computed `usage_state == 'unused'` sets. `gte` / `code` columns show each encoder's raw unused set.

| case | label | subcategory | consensus held-out | gte unused | code unused | consensus held-out ids |
|---|---|---|---:|---:|---:|---|
| base_01_pooler_helper__correct | grounded | correct | 0 | 0 | 0 | - |
| base_01_pooler_helper_wrong | ungrounded | wrong | 0 | 0 | 0 | - |
| base_01_pooler_helper_ambiguous | ambiguous | ambiguous | 0 | 0 | 0 | - |
| base_02_nli_source_init__correct | grounded | correct | 0 | 0 | 0 | - |
| base_02_nli_source_init_wrong | ungrounded | wrong | 0 | 0 | 0 | - |
| base_02_nli_source_init_ambiguous | ambiguous | ambiguous | 0 | 0 | 0 | - |
| base_03_benchmark_measure__correct | grounded | correct | 0 | 0 | 0 | - |
| base_03_benchmark_measure_wrong | ungrounded | wrong | 0 | 0 | 0 | - |
| base_04_structured_evidence_detector__correct | grounded | correct | 0 | 0 | 0 | - |
| base_04_structured_evidence_detector_wrong | ungrounded | wrong | 0 | 0 | 0 | - |
| base_04_structured_evidence_detector_ambiguous | ambiguous | ambiguous | 0 | 0 | 0 | - |
| base_05_maxsim_timing_block__correct | grounded | correct | 0 | 0 | 0 | - |
| base_05_maxsim_timing_block_wrong | ungrounded | wrong | 0 | 0 | 0 | - |
| base_06_benchmark_invocation__correct | grounded | correct | 0 | 0 | 0 | - |
| base_06_benchmark_invocation_wrong | ungrounded | wrong | 0 | 0 | 0 | - |
| base_07_wire_marker_module__correct | grounded | correct | 0 | 0 | 0 | - |
| base_07_wire_marker_module_wrong | ungrounded | wrong | 0 | 0 | 0 | - |
| base_08_cleanup_audit_entry__correct | grounded | correct | 0 | 0 | 0 | - |
| base_08_cleanup_audit_entry_wrong | ungrounded | wrong | 0 | 0 | 0 | - |
| base_08_cleanup_audit_entry_ambiguous | ambiguous | ambiguous | 0 | 0 | 0 | - |

## Per-case scores

| id | chunker | scorer | label | subcategory | reverse_context | coverage | unused | phantom | max evidence | units |
|---|---|---|---|---|---:|---:|---:|---|---:|---:|
| base_01_pooler_helper__correct | sentence_packed | `gte_only` | grounded | correct | 0.9795 | 1.0000 | 0.0000 | False | 1.0000 | 69 |
| base_01_pooler_helper__correct | sentence_packed | `code_only` | grounded | correct | 0.9462 | 1.0000 | 0.0000 | False | 0.9999 | 69 |
| base_01_pooler_helper__correct | sentence_packed | `fuse_mean` | grounded | correct | 0.9629 | 1.0000 | 0.0000 | False | 1.0000 | 69 |
| base_01_pooler_helper__correct | sentence_packed | `fuse_max` | grounded | correct | 0.9795 | 1.0000 | 0.0000 | False | 1.0000 | 69 |
| base_01_pooler_helper__correct | sentence_packed | `fuse_min` | grounded | correct | 0.9462 | 1.0000 | 0.0000 | False | 0.9999 | 69 |
| base_01_pooler_helper_wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9738 | 1.0000 | 0.0000 | False | 1.0000 | 69 |
| base_01_pooler_helper_wrong | sentence_packed | `code_only` | ungrounded | wrong | 0.9379 | 1.0000 | 0.0000 | False | 0.9999 | 69 |
| base_01_pooler_helper_wrong | sentence_packed | `fuse_mean` | ungrounded | wrong | 0.9559 | 1.0000 | 0.0000 | False | 1.0000 | 69 |
| base_01_pooler_helper_wrong | sentence_packed | `fuse_max` | ungrounded | wrong | 0.9738 | 1.0000 | 0.0000 | False | 1.0000 | 69 |
| base_01_pooler_helper_wrong | sentence_packed | `fuse_min` | ungrounded | wrong | 0.9379 | 1.0000 | 0.0000 | False | 0.9999 | 69 |
| base_01_pooler_helper_ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9814 | 1.0000 | 0.0000 | False | 1.0000 | 69 |
| base_01_pooler_helper_ambiguous | sentence_packed | `code_only` | ambiguous | ambiguous | 0.9484 | 1.0000 | 0.0000 | False | 0.9999 | 69 |
| base_01_pooler_helper_ambiguous | sentence_packed | `fuse_mean` | ambiguous | ambiguous | 0.9649 | 1.0000 | 0.0000 | False | 1.0000 | 69 |
| base_01_pooler_helper_ambiguous | sentence_packed | `fuse_max` | ambiguous | ambiguous | 0.9814 | 1.0000 | 0.0000 | False | 1.0000 | 69 |
| base_01_pooler_helper_ambiguous | sentence_packed | `fuse_min` | ambiguous | ambiguous | 0.9484 | 1.0000 | 0.0000 | False | 0.9999 | 69 |
| base_02_nli_source_init__correct | sentence_packed | `gte_only` | grounded | correct | 0.9824 | 1.0000 | 0.0000 | False | 1.0000 | 93 |
| base_02_nli_source_init__correct | sentence_packed | `code_only` | grounded | correct | 0.9425 | 1.0000 | 0.0000 | False | 0.9999 | 93 |
| base_02_nli_source_init__correct | sentence_packed | `fuse_mean` | grounded | correct | 0.9625 | 1.0000 | 0.0000 | False | 0.9999 | 93 |
| base_02_nli_source_init__correct | sentence_packed | `fuse_max` | grounded | correct | 0.9824 | 1.0000 | 0.0000 | False | 1.0000 | 93 |
| base_02_nli_source_init__correct | sentence_packed | `fuse_min` | grounded | correct | 0.9425 | 1.0000 | 0.0000 | False | 0.9999 | 93 |
| base_02_nli_source_init_wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9788 | 1.0000 | 0.0000 | False | 0.9993 | 93 |
| base_02_nli_source_init_wrong | sentence_packed | `code_only` | ungrounded | wrong | 0.9409 | 1.0000 | 0.0000 | False | 0.9885 | 93 |
| base_02_nli_source_init_wrong | sentence_packed | `fuse_mean` | ungrounded | wrong | 0.9599 | 1.0000 | 0.0000 | False | 0.9939 | 93 |
| base_02_nli_source_init_wrong | sentence_packed | `fuse_max` | ungrounded | wrong | 0.9788 | 1.0000 | 0.0000 | False | 0.9993 | 93 |
| base_02_nli_source_init_wrong | sentence_packed | `fuse_min` | ungrounded | wrong | 0.9409 | 1.0000 | 0.0000 | False | 0.9885 | 93 |
| base_02_nli_source_init_ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9833 | 1.0000 | 0.0000 | False | 1.0000 | 93 |
| base_02_nli_source_init_ambiguous | sentence_packed | `code_only` | ambiguous | ambiguous | 0.9416 | 1.0000 | 0.0000 | False | 1.0000 | 93 |
| base_02_nli_source_init_ambiguous | sentence_packed | `fuse_mean` | ambiguous | ambiguous | 0.9625 | 1.0000 | 0.0000 | False | 1.0000 | 93 |
| base_02_nli_source_init_ambiguous | sentence_packed | `fuse_max` | ambiguous | ambiguous | 0.9833 | 1.0000 | 0.0000 | False | 1.0000 | 93 |
| base_02_nli_source_init_ambiguous | sentence_packed | `fuse_min` | ambiguous | ambiguous | 0.9416 | 1.0000 | 0.0000 | False | 1.0000 | 93 |
| base_03_benchmark_measure__correct | sentence_packed | `gte_only` | grounded | correct | 0.9817 | 1.0000 | 0.0000 | False | 1.0000 | 102 |
| base_03_benchmark_measure__correct | sentence_packed | `code_only` | grounded | correct | 0.9382 | 1.0000 | 0.0000 | False | 0.9999 | 102 |
| base_03_benchmark_measure__correct | sentence_packed | `fuse_mean` | grounded | correct | 0.9599 | 1.0000 | 0.0000 | False | 0.9999 | 102 |
| base_03_benchmark_measure__correct | sentence_packed | `fuse_max` | grounded | correct | 0.9817 | 1.0000 | 0.0000 | False | 1.0000 | 102 |
| base_03_benchmark_measure__correct | sentence_packed | `fuse_min` | grounded | correct | 0.9382 | 1.0000 | 0.0000 | False | 0.9999 | 102 |
| base_03_benchmark_measure_wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9742 | 1.0000 | 0.0000 | False | 1.0000 | 102 |
| base_03_benchmark_measure_wrong | sentence_packed | `code_only` | ungrounded | wrong | 0.9326 | 1.0000 | 0.0000 | False | 1.0000 | 102 |
| base_03_benchmark_measure_wrong | sentence_packed | `fuse_mean` | ungrounded | wrong | 0.9534 | 1.0000 | 0.0000 | False | 1.0000 | 102 |
| base_03_benchmark_measure_wrong | sentence_packed | `fuse_max` | ungrounded | wrong | 0.9742 | 1.0000 | 0.0000 | False | 1.0000 | 102 |
| base_03_benchmark_measure_wrong | sentence_packed | `fuse_min` | ungrounded | wrong | 0.9326 | 1.0000 | 0.0000 | False | 1.0000 | 102 |
| base_04_structured_evidence_detector__correct | sentence_packed | `gte_only` | grounded | correct | 0.9755 | 1.0000 | 0.0000 | False | 0.9929 | 199 |
| base_04_structured_evidence_detector__correct | sentence_packed | `code_only` | grounded | correct | 0.9338 | 1.0000 | 0.0000 | False | 0.9863 | 199 |
| base_04_structured_evidence_detector__correct | sentence_packed | `fuse_mean` | grounded | correct | 0.9547 | 1.0000 | 0.0000 | False | 0.9896 | 199 |
| base_04_structured_evidence_detector__correct | sentence_packed | `fuse_max` | grounded | correct | 0.9755 | 1.0000 | 0.0000 | False | 0.9929 | 199 |
| base_04_structured_evidence_detector__correct | sentence_packed | `fuse_min` | grounded | correct | 0.9338 | 1.0000 | 0.0000 | False | 0.9863 | 199 |
| base_04_structured_evidence_detector_wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9707 | 1.0000 | 0.0000 | False | 0.9800 | 199 |
| base_04_structured_evidence_detector_wrong | sentence_packed | `code_only` | ungrounded | wrong | 0.9262 | 1.0000 | 0.0000 | False | 0.9820 | 199 |
| base_04_structured_evidence_detector_wrong | sentence_packed | `fuse_mean` | ungrounded | wrong | 0.9485 | 1.0000 | 0.0000 | False | 0.9810 | 199 |
| base_04_structured_evidence_detector_wrong | sentence_packed | `fuse_max` | ungrounded | wrong | 0.9707 | 1.0000 | 0.0000 | False | 0.9820 | 199 |
| base_04_structured_evidence_detector_wrong | sentence_packed | `fuse_min` | ungrounded | wrong | 0.9262 | 1.0000 | 0.0000 | False | 0.9800 | 199 |
| base_04_structured_evidence_detector_ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9744 | 1.0000 | 0.0000 | False | 0.9928 | 199 |
| base_04_structured_evidence_detector_ambiguous | sentence_packed | `code_only` | ambiguous | ambiguous | 0.9285 | 1.0000 | 0.0000 | False | 0.9671 | 199 |
| base_04_structured_evidence_detector_ambiguous | sentence_packed | `fuse_mean` | ambiguous | ambiguous | 0.9514 | 1.0000 | 0.0000 | False | 0.9799 | 199 |
| base_04_structured_evidence_detector_ambiguous | sentence_packed | `fuse_max` | ambiguous | ambiguous | 0.9744 | 1.0000 | 0.0000 | False | 0.9928 | 199 |
| base_04_structured_evidence_detector_ambiguous | sentence_packed | `fuse_min` | ambiguous | ambiguous | 0.9285 | 1.0000 | 0.0000 | False | 0.9671 | 199 |
| base_05_maxsim_timing_block__correct | sentence_packed | `gte_only` | grounded | correct | 0.9857 | 1.0000 | 0.0000 | False | 1.0000 | 81 |
| base_05_maxsim_timing_block__correct | sentence_packed | `code_only` | grounded | correct | 0.9581 | 1.0000 | 0.0000 | False | 0.9999 | 81 |
| base_05_maxsim_timing_block__correct | sentence_packed | `fuse_mean` | grounded | correct | 0.9719 | 1.0000 | 0.0000 | False | 0.9999 | 81 |
| base_05_maxsim_timing_block__correct | sentence_packed | `fuse_max` | grounded | correct | 0.9857 | 1.0000 | 0.0000 | False | 1.0000 | 81 |
| base_05_maxsim_timing_block__correct | sentence_packed | `fuse_min` | grounded | correct | 0.9581 | 1.0000 | 0.0000 | False | 0.9999 | 81 |
| base_05_maxsim_timing_block_wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9810 | 1.0000 | 0.0000 | False | 0.9988 | 81 |
| base_05_maxsim_timing_block_wrong | sentence_packed | `code_only` | ungrounded | wrong | 0.9533 | 1.0000 | 0.0000 | False | 1.0000 | 81 |
| base_05_maxsim_timing_block_wrong | sentence_packed | `fuse_mean` | ungrounded | wrong | 0.9671 | 1.0000 | 0.0000 | False | 0.9994 | 81 |
| base_05_maxsim_timing_block_wrong | sentence_packed | `fuse_max` | ungrounded | wrong | 0.9810 | 1.0000 | 0.0000 | False | 1.0000 | 81 |
| base_05_maxsim_timing_block_wrong | sentence_packed | `fuse_min` | ungrounded | wrong | 0.9533 | 1.0000 | 0.0000 | False | 0.9988 | 81 |
| base_06_benchmark_invocation__correct | sentence_packed | `gte_only` | grounded | correct | 0.9799 | 1.0000 | 0.0000 | False | 0.9998 | 21 |
| base_06_benchmark_invocation__correct | sentence_packed | `code_only` | grounded | correct | 0.9342 | 1.0000 | 0.0000 | False | 1.0000 | 21 |
| base_06_benchmark_invocation__correct | sentence_packed | `fuse_mean` | grounded | correct | 0.9571 | 1.0000 | 0.0000 | False | 0.9999 | 21 |
| base_06_benchmark_invocation__correct | sentence_packed | `fuse_max` | grounded | correct | 0.9799 | 1.0000 | 0.0000 | False | 1.0000 | 21 |
| base_06_benchmark_invocation__correct | sentence_packed | `fuse_min` | grounded | correct | 0.9342 | 1.0000 | 0.0000 | False | 0.9998 | 21 |
| base_06_benchmark_invocation_wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9766 | 1.0000 | 0.0000 | False | 0.9999 | 21 |
| base_06_benchmark_invocation_wrong | sentence_packed | `code_only` | ungrounded | wrong | 0.9232 | 1.0000 | 0.0000 | False | 1.0000 | 21 |
| base_06_benchmark_invocation_wrong | sentence_packed | `fuse_mean` | ungrounded | wrong | 0.9499 | 1.0000 | 0.0000 | False | 0.9999 | 21 |
| base_06_benchmark_invocation_wrong | sentence_packed | `fuse_max` | ungrounded | wrong | 0.9766 | 1.0000 | 0.0000 | False | 1.0000 | 21 |
| base_06_benchmark_invocation_wrong | sentence_packed | `fuse_min` | ungrounded | wrong | 0.9232 | 1.0000 | 0.0000 | False | 0.9999 | 21 |
| base_07_wire_marker_module__correct | sentence_packed | `gte_only` | grounded | correct | 0.9724 | 1.0000 | 0.0000 | False | 0.9963 | 61 |
| base_07_wire_marker_module__correct | sentence_packed | `code_only` | grounded | correct | 0.9092 | 1.0000 | 0.0000 | False | 0.9602 | 61 |
| base_07_wire_marker_module__correct | sentence_packed | `fuse_mean` | grounded | correct | 0.9408 | 1.0000 | 0.0000 | False | 0.9782 | 61 |
| base_07_wire_marker_module__correct | sentence_packed | `fuse_max` | grounded | correct | 0.9724 | 1.0000 | 0.0000 | False | 0.9963 | 61 |
| base_07_wire_marker_module__correct | sentence_packed | `fuse_min` | grounded | correct | 0.9092 | 1.0000 | 0.0000 | False | 0.9602 | 61 |
| base_07_wire_marker_module_wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9660 | 1.0000 | 0.0000 | False | 0.9957 | 61 |
| base_07_wire_marker_module_wrong | sentence_packed | `code_only` | ungrounded | wrong | 0.8994 | 1.0000 | 0.0000 | False | 0.9751 | 61 |
| base_07_wire_marker_module_wrong | sentence_packed | `fuse_mean` | ungrounded | wrong | 0.9327 | 1.0000 | 0.0000 | False | 0.9854 | 61 |
| base_07_wire_marker_module_wrong | sentence_packed | `fuse_max` | ungrounded | wrong | 0.9660 | 1.0000 | 0.0000 | False | 0.9957 | 61 |
| base_07_wire_marker_module_wrong | sentence_packed | `fuse_min` | ungrounded | wrong | 0.8994 | 1.0000 | 0.0000 | False | 0.9751 | 61 |
| base_08_cleanup_audit_entry__correct | sentence_packed | `gte_only` | grounded | correct | 0.9615 | 1.0000 | 0.0000 | False | 0.9999 | 25 |
| base_08_cleanup_audit_entry__correct | sentence_packed | `code_only` | grounded | correct | 0.8975 | 1.0000 | 0.0000 | False | 0.9818 | 25 |
| base_08_cleanup_audit_entry__correct | sentence_packed | `fuse_mean` | grounded | correct | 0.9295 | 1.0000 | 0.0000 | False | 0.9909 | 25 |
| base_08_cleanup_audit_entry__correct | sentence_packed | `fuse_max` | grounded | correct | 0.9615 | 1.0000 | 0.0000 | False | 0.9999 | 25 |
| base_08_cleanup_audit_entry__correct | sentence_packed | `fuse_min` | grounded | correct | 0.8975 | 1.0000 | 0.0000 | False | 0.9818 | 25 |
| base_08_cleanup_audit_entry_wrong | sentence_packed | `gte_only` | ungrounded | wrong | 0.9572 | 1.0000 | 0.0000 | False | 0.9977 | 25 |
| base_08_cleanup_audit_entry_wrong | sentence_packed | `code_only` | ungrounded | wrong | 0.9130 | 1.0000 | 0.0000 | False | 0.9907 | 25 |
| base_08_cleanup_audit_entry_wrong | sentence_packed | `fuse_mean` | ungrounded | wrong | 0.9351 | 1.0000 | 0.0000 | False | 0.9942 | 25 |
| base_08_cleanup_audit_entry_wrong | sentence_packed | `fuse_max` | ungrounded | wrong | 0.9572 | 1.0000 | 0.0000 | False | 0.9977 | 25 |
| base_08_cleanup_audit_entry_wrong | sentence_packed | `fuse_min` | ungrounded | wrong | 0.9130 | 1.0000 | 0.0000 | False | 0.9907 | 25 |
| base_08_cleanup_audit_entry_ambiguous | sentence_packed | `gte_only` | ambiguous | ambiguous | 0.9557 | 1.0000 | 0.0000 | False | 0.9980 | 25 |
| base_08_cleanup_audit_entry_ambiguous | sentence_packed | `code_only` | ambiguous | ambiguous | 0.9048 | 1.0000 | 0.0000 | False | 0.9872 | 25 |
| base_08_cleanup_audit_entry_ambiguous | sentence_packed | `fuse_mean` | ambiguous | ambiguous | 0.9302 | 1.0000 | 0.0000 | False | 0.9926 | 25 |
| base_08_cleanup_audit_entry_ambiguous | sentence_packed | `fuse_max` | ambiguous | ambiguous | 0.9557 | 1.0000 | 0.0000 | False | 0.9980 | 25 |
| base_08_cleanup_audit_entry_ambiguous | sentence_packed | `fuse_min` | ambiguous | ambiguous | 0.9048 | 1.0000 | 0.0000 | False | 0.9872 | 25 |
| base_01_pooler_helper__correct | colgrep | `gte_only` | grounded | correct | 0.9791 | 1.0000 | 0.0000 | False | 1.0000 | 94 |
| base_01_pooler_helper__correct | colgrep | `code_only` | grounded | correct | 0.9464 | 1.0000 | 0.0000 | False | 0.9999 | 94 |
| base_01_pooler_helper__correct | colgrep | `fuse_mean` | grounded | correct | 0.9628 | 1.0000 | 0.0000 | False | 1.0000 | 94 |
| base_01_pooler_helper__correct | colgrep | `fuse_max` | grounded | correct | 0.9791 | 1.0000 | 0.0000 | False | 1.0000 | 94 |
| base_01_pooler_helper__correct | colgrep | `fuse_min` | grounded | correct | 0.9464 | 1.0000 | 0.0000 | False | 0.9999 | 94 |
| base_01_pooler_helper_wrong | colgrep | `gte_only` | ungrounded | wrong | 0.9742 | 1.0000 | 0.0000 | False | 1.0000 | 94 |
| base_01_pooler_helper_wrong | colgrep | `code_only` | ungrounded | wrong | 0.9393 | 1.0000 | 0.0000 | False | 0.9999 | 94 |
| base_01_pooler_helper_wrong | colgrep | `fuse_mean` | ungrounded | wrong | 0.9567 | 1.0000 | 0.0000 | False | 1.0000 | 94 |
| base_01_pooler_helper_wrong | colgrep | `fuse_max` | ungrounded | wrong | 0.9742 | 1.0000 | 0.0000 | False | 1.0000 | 94 |
| base_01_pooler_helper_wrong | colgrep | `fuse_min` | ungrounded | wrong | 0.9393 | 1.0000 | 0.0000 | False | 0.9999 | 94 |
| base_01_pooler_helper_ambiguous | colgrep | `gte_only` | ambiguous | ambiguous | 0.9809 | 1.0000 | 0.0000 | False | 1.0000 | 94 |
| base_01_pooler_helper_ambiguous | colgrep | `code_only` | ambiguous | ambiguous | 0.9482 | 1.0000 | 0.0000 | False | 0.9999 | 94 |
| base_01_pooler_helper_ambiguous | colgrep | `fuse_mean` | ambiguous | ambiguous | 0.9645 | 1.0000 | 0.0000 | False | 1.0000 | 94 |
| base_01_pooler_helper_ambiguous | colgrep | `fuse_max` | ambiguous | ambiguous | 0.9809 | 1.0000 | 0.0000 | False | 1.0000 | 94 |
| base_01_pooler_helper_ambiguous | colgrep | `fuse_min` | ambiguous | ambiguous | 0.9482 | 1.0000 | 0.0000 | False | 0.9999 | 94 |
| base_02_nli_source_init__correct | colgrep | `gte_only` | grounded | correct | 0.9807 | 1.0000 | 0.0000 | False | 1.0000 | 98 |
| base_02_nli_source_init__correct | colgrep | `code_only` | grounded | correct | 0.9345 | 1.0000 | 0.0000 | False | 0.9999 | 98 |
| base_02_nli_source_init__correct | colgrep | `fuse_mean` | grounded | correct | 0.9576 | 1.0000 | 0.0000 | False | 1.0000 | 98 |
| base_02_nli_source_init__correct | colgrep | `fuse_max` | grounded | correct | 0.9807 | 1.0000 | 0.0000 | False | 1.0000 | 98 |
| base_02_nli_source_init__correct | colgrep | `fuse_min` | grounded | correct | 0.9345 | 1.0000 | 0.0000 | False | 0.9999 | 98 |
| base_02_nli_source_init_wrong | colgrep | `gte_only` | ungrounded | wrong | 0.9786 | 1.0000 | 0.0000 | False | 0.9997 | 98 |
| base_02_nli_source_init_wrong | colgrep | `code_only` | ungrounded | wrong | 0.9363 | 1.0000 | 0.0000 | False | 0.9879 | 98 |
| base_02_nli_source_init_wrong | colgrep | `fuse_mean` | ungrounded | wrong | 0.9574 | 1.0000 | 0.0000 | False | 0.9938 | 98 |
| base_02_nli_source_init_wrong | colgrep | `fuse_max` | ungrounded | wrong | 0.9786 | 1.0000 | 0.0000 | False | 0.9997 | 98 |
| base_02_nli_source_init_wrong | colgrep | `fuse_min` | ungrounded | wrong | 0.9363 | 1.0000 | 0.0000 | False | 0.9879 | 98 |
| base_02_nli_source_init_ambiguous | colgrep | `gte_only` | ambiguous | ambiguous | 0.9820 | 1.0000 | 0.0000 | False | 1.0000 | 98 |
| base_02_nli_source_init_ambiguous | colgrep | `code_only` | ambiguous | ambiguous | 0.9329 | 1.0000 | 0.0000 | False | 1.0000 | 98 |
| base_02_nli_source_init_ambiguous | colgrep | `fuse_mean` | ambiguous | ambiguous | 0.9574 | 1.0000 | 0.0000 | False | 1.0000 | 98 |
| base_02_nli_source_init_ambiguous | colgrep | `fuse_max` | ambiguous | ambiguous | 0.9820 | 1.0000 | 0.0000 | False | 1.0000 | 98 |
| base_02_nli_source_init_ambiguous | colgrep | `fuse_min` | ambiguous | ambiguous | 0.9329 | 1.0000 | 0.0000 | False | 1.0000 | 98 |
| base_03_benchmark_measure__correct | colgrep | `gte_only` | grounded | correct | 0.9831 | 1.0000 | 0.0000 | False | 0.9999 | 126 |
| base_03_benchmark_measure__correct | colgrep | `code_only` | grounded | correct | 0.9380 | 1.0000 | 0.0000 | False | 0.9999 | 126 |
| base_03_benchmark_measure__correct | colgrep | `fuse_mean` | grounded | correct | 0.9605 | 1.0000 | 0.0000 | False | 0.9999 | 126 |
| base_03_benchmark_measure__correct | colgrep | `fuse_max` | grounded | correct | 0.9831 | 1.0000 | 0.0000 | False | 0.9999 | 126 |
| base_03_benchmark_measure__correct | colgrep | `fuse_min` | grounded | correct | 0.9380 | 1.0000 | 0.0000 | False | 0.9999 | 126 |
| base_03_benchmark_measure_wrong | colgrep | `gte_only` | ungrounded | wrong | 0.9759 | 1.0000 | 0.0000 | False | 1.0000 | 126 |
| base_03_benchmark_measure_wrong | colgrep | `code_only` | ungrounded | wrong | 0.9294 | 1.0000 | 0.0000 | False | 1.0000 | 126 |
| base_03_benchmark_measure_wrong | colgrep | `fuse_mean` | ungrounded | wrong | 0.9527 | 1.0000 | 0.0000 | False | 1.0000 | 126 |
| base_03_benchmark_measure_wrong | colgrep | `fuse_max` | ungrounded | wrong | 0.9759 | 1.0000 | 0.0000 | False | 1.0000 | 126 |
| base_03_benchmark_measure_wrong | colgrep | `fuse_min` | ungrounded | wrong | 0.9294 | 1.0000 | 0.0000 | False | 1.0000 | 126 |
| base_04_structured_evidence_detector__correct | colgrep | `gte_only` | grounded | correct | 0.9747 | 1.0000 | 0.0000 | False | 0.9928 | 154 |
| base_04_structured_evidence_detector__correct | colgrep | `code_only` | grounded | correct | 0.9287 | 1.0000 | 0.0000 | False | 0.9824 | 154 |
| base_04_structured_evidence_detector__correct | colgrep | `fuse_mean` | grounded | correct | 0.9517 | 1.0000 | 0.0000 | False | 0.9876 | 154 |
| base_04_structured_evidence_detector__correct | colgrep | `fuse_max` | grounded | correct | 0.9747 | 1.0000 | 0.0000 | False | 0.9928 | 154 |
| base_04_structured_evidence_detector__correct | colgrep | `fuse_min` | grounded | correct | 0.9287 | 1.0000 | 0.0000 | False | 0.9824 | 154 |
| base_04_structured_evidence_detector_wrong | colgrep | `gte_only` | ungrounded | wrong | 0.9689 | 1.0000 | 0.0000 | False | 0.9825 | 154 |
| base_04_structured_evidence_detector_wrong | colgrep | `code_only` | ungrounded | wrong | 0.9214 | 1.0000 | 0.0000 | False | 0.9773 | 154 |
| base_04_structured_evidence_detector_wrong | colgrep | `fuse_mean` | ungrounded | wrong | 0.9451 | 1.0000 | 0.0000 | False | 0.9799 | 154 |
| base_04_structured_evidence_detector_wrong | colgrep | `fuse_max` | ungrounded | wrong | 0.9689 | 1.0000 | 0.0000 | False | 0.9825 | 154 |
| base_04_structured_evidence_detector_wrong | colgrep | `fuse_min` | ungrounded | wrong | 0.9214 | 1.0000 | 0.0000 | False | 0.9773 | 154 |
| base_04_structured_evidence_detector_ambiguous | colgrep | `gte_only` | ambiguous | ambiguous | 0.9735 | 1.0000 | 0.0000 | False | 0.9925 | 154 |
| base_04_structured_evidence_detector_ambiguous | colgrep | `code_only` | ambiguous | ambiguous | 0.9225 | 1.0000 | 0.0000 | False | 0.9654 | 154 |
| base_04_structured_evidence_detector_ambiguous | colgrep | `fuse_mean` | ambiguous | ambiguous | 0.9480 | 1.0000 | 0.0000 | False | 0.9789 | 154 |
| base_04_structured_evidence_detector_ambiguous | colgrep | `fuse_max` | ambiguous | ambiguous | 0.9735 | 1.0000 | 0.0000 | False | 0.9925 | 154 |
| base_04_structured_evidence_detector_ambiguous | colgrep | `fuse_min` | ambiguous | ambiguous | 0.9225 | 1.0000 | 0.0000 | False | 0.9654 | 154 |
| base_05_maxsim_timing_block__correct | colgrep | `gte_only` | grounded | correct | 0.9848 | 1.0000 | 0.0000 | False | 1.0000 | 88 |
| base_05_maxsim_timing_block__correct | colgrep | `code_only` | grounded | correct | 0.9580 | 1.0000 | 0.0000 | False | 0.9999 | 88 |
| base_05_maxsim_timing_block__correct | colgrep | `fuse_mean` | grounded | correct | 0.9714 | 1.0000 | 0.0000 | False | 0.9999 | 88 |
| base_05_maxsim_timing_block__correct | colgrep | `fuse_max` | grounded | correct | 0.9848 | 1.0000 | 0.0000 | False | 1.0000 | 88 |
| base_05_maxsim_timing_block__correct | colgrep | `fuse_min` | grounded | correct | 0.9580 | 1.0000 | 0.0000 | False | 0.9999 | 88 |
| base_05_maxsim_timing_block_wrong | colgrep | `gte_only` | ungrounded | wrong | 0.9801 | 1.0000 | 0.0000 | False | 0.9986 | 88 |
| base_05_maxsim_timing_block_wrong | colgrep | `code_only` | ungrounded | wrong | 0.9536 | 1.0000 | 0.0000 | False | 1.0000 | 88 |
| base_05_maxsim_timing_block_wrong | colgrep | `fuse_mean` | ungrounded | wrong | 0.9669 | 1.0000 | 0.0000 | False | 0.9993 | 88 |
| base_05_maxsim_timing_block_wrong | colgrep | `fuse_max` | ungrounded | wrong | 0.9801 | 1.0000 | 0.0000 | False | 1.0000 | 88 |
| base_05_maxsim_timing_block_wrong | colgrep | `fuse_min` | ungrounded | wrong | 0.9536 | 1.0000 | 0.0000 | False | 0.9986 | 88 |
| base_06_benchmark_invocation__correct | colgrep | `gte_only` | grounded | correct | 0.9792 | 1.0000 | 0.0000 | False | 0.9999 | 24 |
| base_06_benchmark_invocation__correct | colgrep | `code_only` | grounded | correct | 0.9337 | 1.0000 | 0.0000 | False | 1.0000 | 24 |
| base_06_benchmark_invocation__correct | colgrep | `fuse_mean` | grounded | correct | 0.9564 | 1.0000 | 0.0000 | False | 0.9999 | 24 |
| base_06_benchmark_invocation__correct | colgrep | `fuse_max` | grounded | correct | 0.9792 | 1.0000 | 0.0000 | False | 1.0000 | 24 |
| base_06_benchmark_invocation__correct | colgrep | `fuse_min` | grounded | correct | 0.9337 | 1.0000 | 0.0000 | False | 0.9999 | 24 |
| base_06_benchmark_invocation_wrong | colgrep | `gte_only` | ungrounded | wrong | 0.9757 | 1.0000 | 0.0000 | False | 1.0000 | 24 |
| base_06_benchmark_invocation_wrong | colgrep | `code_only` | ungrounded | wrong | 0.9240 | 1.0000 | 0.0000 | False | 1.0000 | 24 |
| base_06_benchmark_invocation_wrong | colgrep | `fuse_mean` | ungrounded | wrong | 0.9499 | 1.0000 | 0.0000 | False | 1.0000 | 24 |
| base_06_benchmark_invocation_wrong | colgrep | `fuse_max` | ungrounded | wrong | 0.9757 | 1.0000 | 0.0000 | False | 1.0000 | 24 |
| base_06_benchmark_invocation_wrong | colgrep | `fuse_min` | ungrounded | wrong | 0.9240 | 1.0000 | 0.0000 | False | 1.0000 | 24 |
| base_07_wire_marker_module__correct | colgrep | `gte_only` | grounded | correct | 0.9605 | 1.0000 | 0.0000 | False | 0.9958 | 26 |
| base_07_wire_marker_module__correct | colgrep | `code_only` | grounded | correct | 0.8933 | 1.0000 | 0.0000 | False | 0.9503 | 26 |
| base_07_wire_marker_module__correct | colgrep | `fuse_mean` | grounded | correct | 0.9269 | 1.0000 | 0.0000 | False | 0.9731 | 26 |
| base_07_wire_marker_module__correct | colgrep | `fuse_max` | grounded | correct | 0.9605 | 1.0000 | 0.0000 | False | 0.9958 | 26 |
| base_07_wire_marker_module__correct | colgrep | `fuse_min` | grounded | correct | 0.8933 | 1.0000 | 0.0000 | False | 0.9503 | 26 |
| base_07_wire_marker_module_wrong | colgrep | `gte_only` | ungrounded | wrong | 0.9559 | 1.0000 | 0.0000 | False | 0.9937 | 26 |
| base_07_wire_marker_module_wrong | colgrep | `code_only` | ungrounded | wrong | 0.8853 | 1.0000 | 0.0000 | False | 0.9678 | 26 |
| base_07_wire_marker_module_wrong | colgrep | `fuse_mean` | ungrounded | wrong | 0.9206 | 1.0000 | 0.0000 | False | 0.9808 | 26 |
| base_07_wire_marker_module_wrong | colgrep | `fuse_max` | ungrounded | wrong | 0.9559 | 1.0000 | 0.0000 | False | 0.9937 | 26 |
| base_07_wire_marker_module_wrong | colgrep | `fuse_min` | ungrounded | wrong | 0.8853 | 1.0000 | 0.0000 | False | 0.9678 | 26 |
| base_08_cleanup_audit_entry__correct | colgrep | `gte_only` | grounded | correct | 0.9603 | 1.0000 | 0.0000 | False | 0.9999 | 41 |
| base_08_cleanup_audit_entry__correct | colgrep | `code_only` | grounded | correct | 0.8956 | 1.0000 | 0.0000 | False | 0.9836 | 41 |
| base_08_cleanup_audit_entry__correct | colgrep | `fuse_mean` | grounded | correct | 0.9279 | 1.0000 | 0.0000 | False | 0.9917 | 41 |
| base_08_cleanup_audit_entry__correct | colgrep | `fuse_max` | grounded | correct | 0.9603 | 1.0000 | 0.0000 | False | 0.9999 | 41 |
| base_08_cleanup_audit_entry__correct | colgrep | `fuse_min` | grounded | correct | 0.8956 | 1.0000 | 0.0000 | False | 0.9836 | 41 |
| base_08_cleanup_audit_entry_wrong | colgrep | `gte_only` | ungrounded | wrong | 0.9563 | 1.0000 | 0.0000 | False | 0.9976 | 41 |
| base_08_cleanup_audit_entry_wrong | colgrep | `code_only` | ungrounded | wrong | 0.9122 | 1.0000 | 0.0000 | False | 0.9917 | 41 |
| base_08_cleanup_audit_entry_wrong | colgrep | `fuse_mean` | ungrounded | wrong | 0.9343 | 1.0000 | 0.0000 | False | 0.9947 | 41 |
| base_08_cleanup_audit_entry_wrong | colgrep | `fuse_max` | ungrounded | wrong | 0.9563 | 1.0000 | 0.0000 | False | 0.9976 | 41 |
| base_08_cleanup_audit_entry_wrong | colgrep | `fuse_min` | ungrounded | wrong | 0.9122 | 1.0000 | 0.0000 | False | 0.9917 | 41 |
| base_08_cleanup_audit_entry_ambiguous | colgrep | `gte_only` | ambiguous | ambiguous | 0.9551 | 1.0000 | 0.0000 | False | 0.9978 | 41 |
| base_08_cleanup_audit_entry_ambiguous | colgrep | `code_only` | ambiguous | ambiguous | 0.9041 | 1.0000 | 0.0000 | False | 0.9885 | 41 |
| base_08_cleanup_audit_entry_ambiguous | colgrep | `fuse_mean` | ambiguous | ambiguous | 0.9296 | 1.0000 | 0.0000 | False | 0.9931 | 41 |
| base_08_cleanup_audit_entry_ambiguous | colgrep | `fuse_max` | ambiguous | ambiguous | 0.9551 | 1.0000 | 0.0000 | False | 0.9978 | 41 |
| base_08_cleanup_audit_entry_ambiguous | colgrep | `fuse_min` | ambiguous | ambiguous | 0.9041 | 1.0000 | 0.0000 | False | 0.9885 | 41 |
