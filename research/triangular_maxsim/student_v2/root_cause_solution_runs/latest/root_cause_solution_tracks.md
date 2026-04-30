# Root-Cause-Derived TRACE Solution Tracks

## Verdict

This run builds concrete candidate heads from the diagnosed root causes. A class is production-promoted only if the candidate beats or matches v1 and clears false-allow, false-block, and latency gates.

## Class Results

| class | rows | selected solution | mode | reason |
|---|---:|---|---|---|
| `rag.prose.enterprise` | 288 | `optimized_calibrator` | `allow_block_repair_candidate` | optimized no-regression fusion artifact clears class policy gates |
| `rag.prose.short_factoid` | 401 | `v1_passthrough_repair_only` | `auto_repair_only` | no candidate cleared no-regression, false-decision, and latency gates |
| `rag.prose.multi_claim` | 672 | `v1_passthrough_repair_only` | `auto_repair_only` | no candidate cleared no-regression, false-decision, and latency gates |
| `rag.structured` | 243 | `v1_passthrough_repair_only` | `auto_repair_only` | no candidate cleared no-regression, false-decision, and latency gates |
| `rag.code_in_context` | 1 | `v1_passthrough_repair_only` | `auto_repair_only` | no candidate cleared no-regression, false-decision, and latency gates |
| `code.agentic_trace` | 1 | `trajectory_symbolic_ranker` | `auto_repair_only` | dedicated manufactured trajectory head failed held-out promotion gates |

## Candidate Metrics

| class | candidate | status | acc | ungrounded F1 | false allow | false block | p95 ms |
|---|---|---|---:|---:|---:|---:|---:|
| `rag.prose.enterprise` | `v1_router` | `baseline` | 0.8889 | 0.9116 | 0.1406 | 0.0521 | 0.0 |
| `rag.prose.enterprise` | `optimized_calibrator` | `artifact_proven` | 0.9861 | 0.9896 | 0.0106 | 0.0 | 0.0 |
| `rag.prose.enterprise` | `trace_feature_logreg` | `evaluated` | 0.9802 | 0.9853 | 0.0 | 0.0588 | 0.2223 |
| `rag.prose.short_factoid` | `v1_router` | `baseline` | 0.5611 | 0.6379 | 0.2289 | 0.65 | 0.0 |
| `rag.prose.short_factoid` | `optimized_calibrator` | `artifact_repair_only` | 0.6334 | 0.6142 | 0.0 | 0.0 | 0.0 |
| `rag.prose.short_factoid` | `trace_feature_logreg` | `evaluated` | 0.7447 | 0.7722 | 0.1408 | 0.3714 | 0.2135 |
| `rag.prose.short_factoid` | `compact_claim_evidence_head` | `evaluated` | 0.305 | 0.2687 | 0.7465 | 0.6429 | 0.4929 |
| `rag.prose.short_factoid` | `factoid_atom_verifier` | `evaluated` | 0.7589 | 0.7875 | 0.1127 | 0.3714 | 0.206 |
| `rag.prose.multi_claim` | `v1_router` | `baseline` | 0.6726 | 0.2029 | 0.8654 | 0.0862 | 0.0 |
| `rag.prose.multi_claim` | `optimized_calibrator` | `artifact_repair_only` | 0.436 | 0.4899 | 0.0 | 0.0 | 0.0 |
| `rag.prose.multi_claim` | `trace_feature_logreg` | `evaluated` | 0.5975 | 0.5455 | 0.2192 | 0.4847 | 0.3267 |
| `rag.prose.multi_claim` | `compact_claim_evidence_head` | `evaluated` | 0.5212 | 0.3687 | 0.5479 | 0.4479 | 1.5557 |
| `rag.structured` | `v1_router` | `baseline` | 0.6173 | 0.7546 | 0.1006 | 0.9167 | 0.0 |
| `rag.structured` | `optimized_calibrator` | `artifact_repair_only` | 0.6543 | 0.791 | 0.0 | 0.0 | 0.0 |
| `rag.structured` | `trace_feature_logreg` | `evaluated` | 0.6744 | 0.7971 | 0.0179 | 0.9 | 0.2144 |
| `rag.structured` | `compact_claim_evidence_head` | `evaluated` | 0.593 | 0.7287 | 0.1607 | 0.8667 | 0.8795 |
| `rag.structured` | `structured_cell_verifier` | `evaluated` | 0.6628 | 0.7914 | 0.0179 | 0.9333 | 0.2157 |
| `rag.code_in_context` | `v1_router` | `baseline` | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 |
| `rag.code_in_context` | `optimized_calibrator` | `artifact_repair_only` | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| `rag.code_in_context` | `trace_feature_logreg` | `skipped` | n/a | n/a | n/a | n/a | n/a |
| `rag.code_in_context` | `code_identifier_ranker` | `skipped` | n/a | n/a | n/a | n/a | n/a |
| `code.agentic_trace` | `v1_router` | `baseline` | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 |
| `code.agentic_trace` | `optimized_calibrator` | `artifact_repair_only` | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 |
| `code.agentic_trace` | `trace_feature_logreg` | `skipped` | n/a | n/a | n/a | n/a | n/a |
| `code.agentic_trace` | `code_identifier_ranker` | `skipped` | n/a | n/a | n/a | n/a | n/a |

## Coding Trajectory Head

Promotion decision: `do_not_promote_keep_code_agentic_trace_repair_only`
- `transcripts_v2`: AUROC=0.6931, false_allow=0.25, false_block=0.4833
- `both`: AUROC=0.6061, false_allow=0.1724, false_block=0.5769

## Runtime Registry Proposal

- `rag.prose.enterprise`: head=`optimized_calibrator`, enabled=True, artifact=`latence_trace/data/heads/rag.prose.enterprise.optimized_calibrator.v1.json`, reason=optimized no-regression fusion artifact clears class policy gates
- `rag.prose.short_factoid`: head=`atom_verifier`, enabled=False, artifact=`latence_trace/data/heads/rag.prose.short_factoid.atom_verifier.v1.json`, reason=no candidate cleared no-regression, false-decision, and latency gates
- `rag.prose.multi_claim`: head=`claim_decomposer`, enabled=False, artifact=`latence_trace/data/heads/rag.prose.multi_claim.claim_decomposer.v1.json`, reason=no candidate cleared no-regression, false-decision, and latency gates
- `rag.structured`: head=`cell_schema_verifier`, enabled=False, artifact=`latence_trace/data/heads/rag.structured.cell_schema_verifier.v1.json`, reason=no candidate cleared no-regression, false-decision, and latency gates
- `rag.code_in_context`: head=`identifier_ranker`, enabled=False, artifact=`latence_trace/data/heads/rag.code_in_context.identifier_ranker.v1.json`, reason=no candidate cleared no-regression, false-decision, and latency gates
- `code.agentic_trace`: head=`trajectory_ranker`, enabled=False, artifact=`latence_trace/data/heads/code.agentic_trace.trajectory_ranker.v1.json`, reason=dedicated manufactured trajectory head failed held-out promotion gates

## Next Failure Targets

- `rag.prose.enterprise`: preserve no-regression and reproduce on live RunPod
- `rag.prose.short_factoid`: reduce high-overlap entity/date/number false allows without increasing grounded false blocks
- `rag.prose.multi_claim`: replace whole-answer features with atomic claim decomposition and per-claim unsupported penalties
- `rag.structured`: add row/column provenance and numeric tolerance labels to reduce false blocks
- `rag.code_in_context`: build a real code-in-context eval lane with positive and phantom identifier/API cases
- `code.agentic_trace`: add turn-order, patch/test outcome, file ownership, AST/API drift, and action-result labels
