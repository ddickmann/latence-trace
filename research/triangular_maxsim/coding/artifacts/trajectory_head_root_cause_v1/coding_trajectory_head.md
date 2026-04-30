# Coding Agent Trajectory Head

Trained on `transcripts_v1` using `sentence_packed|gte_only` rows.

| split | rows | AUROC | accuracy | false allow | false block |
|---|---:|---:|---:|---:|---:|
| `train` | 16 | 0.7656 | 0.8125 | 0.125 | 0.25 |
| `transcripts_v2` | 120 | 0.6931 | 0.6333 | 0.25 | 0.4833 |
| `both` | 55 | 0.6061 | 0.6364 | 0.1724 | 0.5769 |

Promotion decision: `do_not_promote_keep_code_agentic_trace_repair_only`.

This head is intentionally separated from production promotion. It becomes eligible only if held-out manufactured trajectory banks clear AUROC and false-decision gates.
