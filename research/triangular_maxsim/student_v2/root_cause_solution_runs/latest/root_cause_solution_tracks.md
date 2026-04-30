# Root-Cause-Derived TRACE Solution Tracks

## Verdict

This run builds concrete candidate heads from the diagnosed root causes. A class is production-promoted only if the candidate beats or matches v1 and clears false-allow, false-block, and latency gates.

## Class Results

| class | rows | selected solution | mode | reason |
|---|---:|---|---|---|
| `rag.prose.enterprise` | 288 | `optimized_calibrator` | `allow_block_repair_candidate` | optimized no-regression fusion artifact clears class policy gates |
| `rag.prose.short_factoid` | 641 | `trace_feature_logreg` | `allow_block_repair_candidate` | smallest candidate cleared fixed gates on held-out split |
| `rag.prose.multi_claim` | 912 | `trace_feature_logreg` | `allow_block_repair_candidate` | smallest candidate cleared fixed gates on held-out split |
| `rag.structured` | 483 | `compact_claim_evidence_head` | `allow_block_repair_candidate` | smallest candidate cleared fixed gates on held-out split |
| `rag.code_in_context` | 241 | `trace_feature_logreg` | `allow_block_repair_candidate` | smallest candidate cleared fixed gates on held-out split |
| `code.agentic_trace` | 241 | `trajectory_symbolic_ranker` | `auto_repair_only` | dedicated manufactured trajectory head failed held-out promotion gates |

## Candidate Metrics

| class | candidate | status | acc | ungrounded F1 | false allow | false block | p95 ms |
|---|---|---|---:|---:|---:|---:|---:|
| `rag.prose.enterprise` | `v1_router` | `baseline` | 0.8889 | 0.9116 | 0.2288 | 0.0294 | 0.0 |
| `rag.prose.enterprise` | `v1_abstain_policy` | `evaluated` | 0.9432 | 0.9593 | 0.0 | 0.0566 | 0.0 |
| `rag.prose.enterprise` | `optimized_calibrator` | `artifact_proven` | 0.9861 | 0.9896 | 0.0106 | 0.0 | 0.0 |
| `rag.prose.enterprise` | `trace_feature_logreg` | `evaluated` | 1.0 | 1.0 | 0.0 | 0.0 | 0.2307 |
| `rag.prose.short_factoid` | `v1_router` | `baseline` | 0.7254 | 0.7576 | 0.1949 | 0.321 | 0.0 |
| `rag.prose.short_factoid` | `v1_abstain_policy` | `evaluated` | 0.7423 | 0.7619 | 0.6667 | 0.1538 | 0.0 |
| `rag.prose.short_factoid` | `optimized_calibrator` | `artifact_repair_only` | 0.6334 | 0.6142 | 0.0 | 0.0 | 0.0 |
| `rag.prose.short_factoid` | `trace_feature_logreg` | `evaluated` | 0.8351 | 0.8476 | 0.0 | 0.0 | 0.2321 |
| `rag.prose.short_factoid` | `compact_claim_evidence_head` | `evaluated` | 0.5722 | 0.6982 | 0.0 | 0.0 | 0.5091 |
| `rag.prose.short_factoid` | `factoid_atom_rule_head` | `evaluated` | 0.6443 | 0.7376 | 0.0 | 0.0 | 0.0014 |
| `rag.prose.short_factoid` | `factoid_atom_verifier` | `evaluated` | 0.8351 | 0.8476 | 0.0 | 0.0 | 0.2399 |
| `rag.prose.multi_claim` | `v1_router` | `baseline` | 0.7588 | 0.6309 | 0.2632 | 0.1754 | 0.0 |
| `rag.prose.multi_claim` | `v1_abstain_policy` | `evaluated` | 0.779 | 0.6474 | 0.0 | 0.0741 | 0.0 |
| `rag.prose.multi_claim` | `optimized_calibrator` | `artifact_repair_only` | 0.436 | 0.4899 | 0.0 | 0.0 | 0.0 |
| `rag.prose.multi_claim` | `trace_feature_logreg` | `evaluated` | 0.7935 | 0.6545 | 0.0 | 0.0 | 0.2314 |
| `rag.prose.multi_claim` | `compact_claim_evidence_head` | `evaluated` | 0.7717 | 0.6038 | 0.0 | 0.0 | 1.5589 |
| `rag.prose.multi_claim` | `claim_decomposition_rule_head` | `evaluated` | 0.5906 | 0.0174 | 0.0 | 0.6 | 0.0018 |
| `rag.structured` | `v1_router` | `baseline` | 0.559 | 0.7118 | 0.6957 | 0.4283 | 0.0 |
| `rag.structured` | `v1_abstain_policy` | `evaluated` | 0.5578 | 0.5752 | 0.0 | 0.3582 | 0.0 |
| `rag.structured` | `optimized_calibrator` | `artifact_repair_only` | 0.6543 | 0.791 | 0.0 | 0.0 | 0.0 |
| `rag.structured` | `trace_feature_logreg` | `evaluated` | 0.6667 | 0.6142 | 0.0 | 0.027 | 0.2354 |
| `rag.structured` | `compact_claim_evidence_head` | `evaluated` | 0.8027 | 0.8513 | 0.0 | 0.0 | 0.7256 |
| `rag.structured` | `structured_cell_rule_head` | `evaluated` | 0.6667 | 0.595 | 0.625 | 0.0 | 0.0012 |
| `rag.structured` | `structured_cell_verifier` | `evaluated` | 0.6667 | 0.6142 | 0.0 | 0.027 | 0.2394 |
| `rag.code_in_context` | `v1_router` | `baseline` | 0.9959 | 0.9959 | 0.0 | 0.0083 | 0.0 |
| `rag.code_in_context` | `v1_abstain_policy` | `evaluated` | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 |
| `rag.code_in_context` | `optimized_calibrator` | `artifact_repair_only` | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| `rag.code_in_context` | `trace_feature_logreg` | `evaluated` | 1.0 | 1.0 | 0.0 | 0.0 | 0.2358 |
| `rag.code_in_context` | `code_symbol_rule_head` | `evaluated` | 1.0 | 1.0 | 0.0 | 0.0 | 0.0013 |
| `rag.code_in_context` | `code_identifier_ranker` | `evaluated` | 1.0 | 1.0 | 0.0 | 0.0 | 0.2311 |
| `code.agentic_trace` | `v1_router` | `baseline` | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 |
| `code.agentic_trace` | `v1_abstain_policy` | `evaluated` | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 |
| `code.agentic_trace` | `optimized_calibrator` | `artifact_repair_only` | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 |
| `code.agentic_trace` | `trace_feature_logreg` | `evaluated` | 1.0 | 1.0 | 0.0 | 0.0 | 0.2364 |
| `code.agentic_trace` | `code_symbol_rule_head` | `evaluated` | 1.0 | 1.0 | 0.0 | 0.0 | 0.0012 |
| `code.agentic_trace` | `code_identifier_ranker` | `evaluated` | 1.0 | 1.0 | 0.0 | 0.0 | 0.2349 |

## Coding Trajectory Head

Promotion decision: `do_not_promote_keep_code_agentic_trace_repair_only`
- `transcripts_v2`: AUROC=0.6931, false_allow=0.25, false_block=0.4833
- `both`: AUROC=0.6061, false_allow=0.1724, false_block=0.5769

## Runtime Registry Proposal

- `rag.prose.enterprise`: head=`optimized_calibrator`, enabled=True, artifact=`latence_trace/data/heads/rag.prose.enterprise.optimized_calibrator.v1.json`, reason=optimized no-regression fusion artifact clears class policy gates
- `rag.prose.short_factoid`: head=`atom_verifier`, enabled=True, artifact=`latence_trace/data/heads/rag.prose.short_factoid.atom_verifier.v1.json`, reason=smallest candidate cleared fixed gates on held-out split
- `rag.prose.multi_claim`: head=`claim_decomposer`, enabled=True, artifact=`latence_trace/data/heads/rag.prose.multi_claim.claim_decomposer.v1.json`, reason=smallest candidate cleared fixed gates on held-out split
- `rag.structured`: head=`cell_schema_verifier`, enabled=True, artifact=`latence_trace/data/heads/rag.structured.cell_schema_verifier.v1.json`, reason=smallest candidate cleared fixed gates on held-out split
- `rag.code_in_context`: head=`identifier_ranker`, enabled=True, artifact=`latence_trace/data/heads/rag.code_in_context.identifier_ranker.v1.json`, reason=smallest candidate cleared fixed gates on held-out split
- `code.agentic_trace`: head=`trajectory_ranker`, enabled=False, artifact=`latence_trace/data/heads/code.agentic_trace.trajectory_ranker.v1.json`, reason=dedicated manufactured trajectory head failed held-out promotion gates

## Next Failure Targets

- `rag.prose.enterprise`: preserve no-regression and reproduce on live RunPod
- `rag.prose.short_factoid`: reduce high-overlap entity/date/number false allows without increasing grounded false blocks
- `rag.prose.multi_claim`: replace whole-answer features with atomic claim decomposition and per-claim unsupported penalties
- `rag.structured`: add row/column provenance and numeric tolerance labels to reduce false blocks
- `rag.code_in_context`: build a real code-in-context eval lane with positive and phantom identifier/API cases
- `code.agentic_trace`: add turn-order, patch/test outcome, file ownership, AST/API drift, and action-result labels
