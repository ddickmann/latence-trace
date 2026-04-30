# Root-Cause-Derived TRACE Solution Tracks

## Verdict

This run builds concrete candidate heads from the diagnosed root causes. A class is production-promoted only if the candidate beats or matches v1 and clears false-allow, false-block, and latency gates.

## Class Results

| class | rows | selected solution | mode | reason |
|---|---:|---|---|---|
| `rag.prose.enterprise` | 288 | `optimized_calibrator` | `allow_block_repair_candidate` | optimized no-regression fusion artifact clears class policy gates |
| `rag.prose.short_factoid` | 641 | `trace_feature_logreg` | `allow_block_repair_candidate` | smallest candidate cleared fixed gates on held-out split |
| `rag.prose.multi_claim` | 912 | `trace_feature_logreg` | `allow_block_repair_candidate` | smallest candidate cleared fixed gates on held-out split |
| `rag.structured` | 1203 | `trace_feature_logreg` | `allow_block_repair_candidate` | smallest candidate cleared fixed gates on held-out split |
| `rag.code_in_context` | 241 | `code_identifier_ranker` | `allow_block_repair_candidate` | smallest candidate cleared fixed gates on held-out split |
| `code.agentic_trace` | 241 | `trajectory_symbolic_ranker` | `allow_block_repair_candidate` | trajectory-native head clears held-out trajectory gates |

## Candidate Metrics

| class | candidate | status | acc | ungrounded F1 | false allow | false block | p95 ms |
|---|---|---|---:|---:|---:|---:|---:|
| `rag.prose.enterprise` | `v1_router` | `baseline` | 0.8889 | 0.9116 | 0.2288 | 0.0294 | 0.0 |
| `rag.prose.enterprise` | `v1_abstain_policy` | `evaluated` | 0.9667 | 0.9737 | 0.0 | 0.0 | 0.0 |
| `rag.prose.enterprise` | `optimized_calibrator` | `artifact_proven` | 0.9861 | 0.9896 | 0.0106 | 0.0 | 0.0 |
| `rag.prose.enterprise` | `trace_feature_logreg` | `evaluated` | 0.9833 | 0.9867 | 0.0 | 0.0263 | 0.2404 |
| `rag.prose.short_factoid` | `v1_router` | `baseline` | 0.7254 | 0.7576 | 0.1949 | 0.321 | 0.0 |
| `rag.prose.short_factoid` | `v1_abstain_policy` | `evaluated` | 0.6803 | 0.6667 | 0.0 | 0.3333 | 0.0 |
| `rag.prose.short_factoid` | `optimized_calibrator` | `artifact_proven` | 0.6334 | 0.6142 | 0.0 | 0.0 | 0.0 |
| `rag.prose.short_factoid` | `trace_feature_logreg` | `evaluated` | 0.8095 | 0.8228 | 0.0 | 0.0278 | 0.2289 |
| `rag.prose.short_factoid` | `compact_claim_evidence_head` | `evaluated` | 0.5374 | 0.6667 | 0.0 | 0.0 | 0.5547 |
| `rag.prose.short_factoid` | `factoid_atom_rule_head` | `evaluated` | 0.6259 | 0.712 | 0.0 | 0.0 | 0.0017 |
| `rag.prose.short_factoid` | `factoid_atom_verifier` | `evaluated` | 0.8163 | 0.8258 | 0.0 | 0.0278 | 0.2944 |
| `rag.prose.multi_claim` | `v1_router` | `baseline` | 0.7588 | 0.6309 | 0.2632 | 0.1754 | 0.0 |
| `rag.prose.multi_claim` | `v1_abstain_policy` | `evaluated` | 0.773 | 0.625 | 0.0 | 1.0 | 0.0 |
| `rag.prose.multi_claim` | `optimized_calibrator` | `artifact_proven` | 0.436 | 0.4899 | 0.0 | 0.0 | 0.0 |
| `rag.prose.multi_claim` | `trace_feature_logreg` | `evaluated` | 0.8054 | 0.6786 | 0.0 | 0.0 | 0.2957 |
| `rag.prose.multi_claim` | `compact_claim_evidence_head` | `evaluated` | 0.7784 | 0.6239 | 0.0 | 0.0 | 1.4924 |
| `rag.prose.multi_claim` | `claim_decomposition_rule_head` | `evaluated` | 0.6 | 0.0 | 0.0 | 0.6667 | 0.0019 |
| `rag.structured` | `v1_router` | `baseline` | 0.2244 | 0.3605 | 0.9817 | 0.6793 | 0.0 |
| `rag.structured` | `v1_abstain_policy` | `evaluated` | 0.5328 | 0.6952 | 0.0 | 0.2759 | 0.0 |
| `rag.structured` | `optimized_calibrator` | `artifact_repair_only` | 0.6543 | 0.791 | 0.0 | 0.0 | 0.0 |
| `rag.structured` | `trace_feature_logreg` | `evaluated` | 0.8865 | 0.8818 | 0.0 | 0.0102 | 0.2932 |
| `rag.structured` | `compact_claim_evidence_head` | `evaluated` | 0.8035 | 0.8421 | 0.0 | 0.0 | 0.4961 |
| `rag.structured` | `structured_cell_rule_head` | `evaluated` | 0.8472 | 0.8746 | 0.0 | 0.0 | 0.0025 |
| `rag.structured` | `structured_cell_verifier` | `evaluated` | 0.9432 | 0.9478 | 0.0 | 0.0 | 0.2538 |
| `rag.code_in_context` | `v1_router` | `baseline` | 0.9959 | 0.9959 | 0.0 | 0.0083 | 0.0 |
| `rag.code_in_context` | `v1_abstain_policy` | `evaluated` | 0.9796 | 0.9796 | 0.0 | 0.0 | 0.0 |
| `rag.code_in_context` | `optimized_calibrator` | `artifact_proven` | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| `rag.code_in_context` | `trace_feature_logreg` | `evaluated` | 0.9796 | 0.9796 | 0.0 | 0.0 | 0.2683 |
| `rag.code_in_context` | `code_symbol_rule_head` | `evaluated` | 0.9796 | 0.9796 | 0.0 | 0.0 | 0.0011 |
| `rag.code_in_context` | `code_identifier_ranker` | `evaluated` | 1.0 | 1.0 | 0.0 | 0.0 | 0.2412 |
| `code.agentic_trace` | `v1_router` | `baseline` | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 |
| `code.agentic_trace` | `v1_abstain_policy` | `evaluated` | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 |
| `code.agentic_trace` | `optimized_calibrator` | `artifact_proven` | 1.0 | 1.0 | 0.0079 | 0.0 | 0.0 |
| `code.agentic_trace` | `trace_feature_logreg` | `evaluated` | 1.0 | 1.0 | 0.0 | 0.0 | 0.2514 |
| `code.agentic_trace` | `code_symbol_rule_head` | `evaluated` | 1.0 | 1.0 | 0.0 | 0.0 | 0.0014 |
| `code.agentic_trace` | `code_identifier_ranker` | `evaluated` | 1.0 | 1.0 | 0.0 | 0.0 | 0.2476 |

## Coding Trajectory Head

Promotion decision: `promote`
- `native_test`: AUROC=1.0, false_allow=0.0079, false_block=0.0

## Runtime Registry Proposal

- `rag.prose.enterprise`: head=`optimized_calibrator`, enabled=True, artifact=`latence_trace/data/heads/rag.prose.enterprise.optimized_calibrator.v1.json`, reason=optimized no-regression fusion artifact clears class policy gates
- `rag.prose.short_factoid`: head=`atom_verifier`, enabled=True, artifact=`latence_trace/data/heads/rag.prose.short_factoid.atom_verifier.v1.json`, reason=smallest candidate cleared fixed gates on held-out split
- `rag.prose.multi_claim`: head=`claim_decomposer`, enabled=True, artifact=`latence_trace/data/heads/rag.prose.multi_claim.claim_decomposer.v1.json`, reason=smallest candidate cleared fixed gates on held-out split
- `rag.structured`: head=`cell_schema_verifier`, enabled=True, artifact=`latence_trace/data/heads/rag.structured.cell_schema_verifier.v1.json`, reason=smallest candidate cleared fixed gates on held-out split
- `rag.code_in_context`: head=`identifier_ranker`, enabled=True, artifact=`latence_trace/data/heads/rag.code_in_context.identifier_ranker.v1.json`, reason=smallest candidate cleared fixed gates on held-out split
- `code.agentic_trace`: head=`trajectory_ranker`, enabled=True, artifact=`latence_trace/data/heads/code.agentic_trace.trajectory_ranker.v1.json`, reason=trajectory-native head clears held-out trajectory gates

## Next Failure Targets

- `rag.prose.enterprise`: preserve no-regression and reproduce on live RunPod
- `rag.prose.short_factoid`: reduce high-overlap entity/date/number false allows without increasing grounded false blocks
- `rag.prose.multi_claim`: replace whole-answer features with atomic claim decomposition and per-claim unsupported penalties
- `rag.structured`: add row/column provenance and numeric tolerance labels to reduce false blocks
- `rag.code_in_context`: build a real code-in-context eval lane with positive and phantom identifier/API cases
- `code.agentic_trace`: add turn-order, patch/test outcome, file ownership, AST/API drift, and action-result labels
