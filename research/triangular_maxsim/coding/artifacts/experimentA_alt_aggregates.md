# Experiment A (isolated): alternative aggregates on transcripts_v1

> Same rows as the production `transcripts_v1` run. Tests whether an aggregate other than `reverse_context` (weighted mean) separates grounded-vs-ungrounded / correct-vs-wrong better. No production code changed.

Gate: `AUROC grounded_vs_ungrounded >= 0.90` for at least one (chunker, scorer, aggregate) triple.

## Round 1 — existing scalar aggregates (`gte_only`)

| chunker | scorer | aggregate | AUROC anchor | separation | AUROC c-vs-w | AUROC c-vs-a | full mono | partial mono |
|---|---|---|---:|---:|---:|---:|---:|---:|
| colgrep | `gte_only` | reverse_context (baseline) | 0.7188 | 0.00461 | 0.7188 | 0.5625 | 0.2500 | 1.0000 |
| colgrep | `gte_only` | reverse_context_calibrated | n/a | n/a | n/a | n/a | n/a | n/a |
| colgrep | `gte_only` | literal_guarded | 0.7031 | 0.12386 | 0.7031 | 0.5469 | 0.0000 | 0.8750 |
| colgrep | `gte_only` | consensus_hardened | 0.6875 | 0.00453 | 0.6875 | 0.5625 | 0.2500 | 1.0000 |
| colgrep | `gte_only` | triangular | 0.7031 | 0.00338 | 0.7031 | 0.4062 | 0.2500 | 0.7500 |
| colgrep | `gte_only` | echo_mean | 0.6875 | 0.00436 | 0.6875 | 0.3438 | 0.2500 | 0.8750 |
| colgrep | `gte_only` | grounded_coverage | 0.5156 | 0.00115 | 0.5156 | 0.5625 | 0.2500 | 0.5000 |
| colgrep | `gte_only` | reverse_query_context | 0.7031 | 0.00482 | 0.7031 | 0.5312 | 0.2500 | 1.0000 |
| colgrep | `gte_only` | groundedness_v2 | 0.7031 | 0.03868 | 0.7031 | 0.5625 | 0.2500 | 0.8750 |
| colgrep | `gte_only` | max_top_evidence_score | 0.6250 | 0.00201 | 0.6250 | 0.6250 | 0.7500 | 0.7500 |
| colgrep | `gte_only` | context_usage_ratio | 0.4219 | -0.00463 | 0.4219 | 0.5000 | 0.0000 | 0.2500 |
| colgrep | `gte_only` | context_attribution_ratio | 0.4062 | -0.04651 | 0.4062 | 0.5938 | 0.0000 | 0.0000 |
| sentence_packed | `gte_only` | reverse_context (baseline) | 0.7344 | 0.00503 | 0.7344 | 0.5625 | 0.2500 | 1.0000 |
| sentence_packed | `gte_only` | reverse_context_calibrated | n/a | n/a | n/a | n/a | n/a | n/a |
| sentence_packed | `gte_only` | literal_guarded | 0.7031 | 0.12415 | 0.7031 | 0.5469 | 0.0000 | 0.8750 |
| sentence_packed | `gte_only` | consensus_hardened | 0.7344 | 0.00497 | 0.7344 | 0.5625 | 0.2500 | 1.0000 |
| sentence_packed | `gte_only` | triangular | 0.7031 | 0.00355 | 0.7031 | 0.4062 | 0.2500 | 0.8750 |
| sentence_packed | `gte_only` | echo_mean | 0.6875 | 0.00436 | 0.6875 | 0.3438 | 0.2500 | 0.8750 |
| sentence_packed | `gte_only` | grounded_coverage | 0.5312 | 0.00114 | 0.5312 | 0.5312 | 0.2500 | 0.5000 |
| sentence_packed | `gte_only` | reverse_query_context | 0.7500 | 0.00479 | 0.7500 | 0.5625 | 0.2500 | 1.0000 |
| sentence_packed | `gte_only` | groundedness_v2 | 0.7031 | 0.03907 | 0.7031 | 0.5625 | 0.2500 | 0.8750 |
| sentence_packed | `gte_only` | max_top_evidence_score | 0.6250 | 0.00219 | 0.6250 | 0.5625 | 0.7500 | 0.7500 |
| sentence_packed | `gte_only` | context_usage_ratio | 0.5625 | 0.01820 | 0.5625 | 0.5625 | 0.0000 | 0.5000 |
| sentence_packed | `gte_only` | context_attribution_ratio | 0.3516 | -0.05498 | 0.3516 | 0.5312 | 0.0000 | 0.0000 |

## Round 2 — per-token worst-k aggregates (`gte_only`)

| chunker | scorer | aggregate | AUROC anchor | separation | AUROC c-vs-w | AUROC c-vs-a | full mono | partial mono |
|---|---|---|---:|---:|---:|---:|---:|---:|
| sentence_packed | `gte_only` | per_token_min | 0.5781 | 0.00168 | 0.5781 | 0.4062 | 0.2500 | 0.6250 |
| sentence_packed | `gte_only` | per_token_p5 | 0.6250 | 0.00808 | 0.6250 | 0.5312 | 0.2500 | 0.8750 |
| sentence_packed | `gte_only` | per_token_p10 | 0.6406 | 0.00944 | 0.6406 | 0.6250 | 0.5000 | 1.0000 |
| sentence_packed | `gte_only` | per_token_mean_weakest_5 | 0.6250 | 0.00759 | 0.6250 | 0.5312 | 0.2500 | 0.7500 |
| sentence_packed | `gte_only` | per_token_mean_minus_min | 0.5156 | 0.00253 | 0.5156 | 0.5938 | 0.2500 | 0.6250 |
| colgrep | `gte_only` | per_token_min | 0.6094 | 0.00586 | 0.6094 | 0.5156 | 0.2500 | 0.7500 |
| colgrep | `gte_only` | per_token_p5 | 0.6094 | 0.00736 | 0.6094 | 0.5312 | 0.5000 | 0.8750 |
| colgrep | `gte_only` | per_token_p10 | 0.6406 | 0.00872 | 0.6406 | 0.6250 | 0.5000 | 1.0000 |
| colgrep | `gte_only` | per_token_mean_weakest_5 | 0.6094 | 0.00785 | 0.6094 | 0.5000 | 0.2500 | 0.6250 |
| colgrep | `gte_only` | per_token_mean_minus_min | 0.4375 | -0.00176 | 0.4375 | 0.5938 | 0.0000 | 0.3750 |

## Top 10 aggregates by AUROC grounded-vs-ungrounded

| rank | chunker | scorer | aggregate | AUROC | separation | AUROC c-vs-w |
|---:|---|---|---|---:|---:|---:|
| 1 | sentence_packed | `gte_only` | reverse_query_context | 0.7500 | 0.00479 | 0.7500 |
| 2 | sentence_packed | `fuse_max` | reverse_context (baseline) | 0.7344 | 0.00503 | 0.7344 |
| 3 | sentence_packed | `fuse_max` | consensus_hardened | 0.7344 | 0.00497 | 0.7344 |
| 4 | sentence_packed | `gte_only` | reverse_context (baseline) | 0.7344 | 0.00503 | 0.7344 |
| 5 | sentence_packed | `gte_only` | consensus_hardened | 0.7344 | 0.00497 | 0.7344 |
| 6 | colgrep | `fuse_max` | reverse_context (baseline) | 0.7188 | 0.00461 | 0.7188 |
| 7 | colgrep | `gte_only` | reverse_context (baseline) | 0.7188 | 0.00461 | 0.7188 |
| 8 | colgrep | `gte_only` | literal_guarded | 0.7031 | 0.12386 | 0.7031 |
| 9 | colgrep | `gte_only` | triangular | 0.7031 | 0.00338 | 0.7031 |
| 10 | colgrep | `gte_only` | reverse_query_context | 0.7031 | 0.00482 | 0.7031 |
