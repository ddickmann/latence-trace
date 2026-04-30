# Root-Cause-Driven TRACE Model Selection

## Verdict

Architecture must stay class-specific. The evidence promotes only `rag.prose.enterprise` to opt-in automatic allow/block/repair with the optimized feature calibrator. Other classes remain repair-only until their diagnosed root causes clear a bake-off gate.

## Per-Class Root Causes And Selection

| class | root causes | selected head | production mode |
|---|---|---|---|
| `rag.prose.enterprise` | aggregation_error, data_label_pathology, false_allow, false_block, grounded_false_block_bias | `optimized_feature_calibrator` | `allow_block_repair_opt_in` |
| `rag.prose.short_factoid` | aggregation_error, data_label_pathology, false_allow, false_block, feature_calibrator_no_lift, partial_support_blind_spot, short_answer_overlap | `compact_claim_evidence_encoder` | `auto_repair_only_until_bakeoff_passes` |
| `rag.prose.multi_claim` | aggregation_error, data_label_pathology, false_allow, false_block, feature_calibrator_no_lift, low_score_separation, multi_claim_aggregation, partial_support_blind_spot, partial_support_collapse | `compact_claim_evidence_encoder` | `auto_repair_only_until_bakeoff_passes` |
| `rag.structured` | aggregation_error, false_block, feature_calibrator_no_lift, low_score_separation, partial_support_collapse, structured_cell_alignment | `compact_claim_evidence_encoder` | `auto_repair_only_until_bakeoff_passes` |
| `rag.code_in_context` | code_identifier_morphology, feature_calibrator_no_lift | `v1_passthrough` | `auto_repair_only` |
| `code.agentic_trace` | code_identifier_morphology, feature_calibrator_no_lift, trajectory_ranker_failure | `trajectory_symbolic_ranker` | `auto_repair_only` |

## Candidate Bake-Off Summary

| class | candidate | family | status | evidence |
|---|---|---|---|---|
| `rag.prose.enterprise` | `v1_router` | `baseline` | `baseline` | binary_grounded_accuracy=0.9201, ungrounded_f1=0.9426 |
| `rag.prose.enterprise` | `optimized_feature_calibrator` | `mlp_logistic_xgboost_feature_head` | `available` | binary_grounded_accuracy=0.9861, ungrounded_f1=0.9896 |
| `rag.prose.enterprise` | `compact_claim_evidence_encoder` | `t5_small_or_tiny_cross_encoder` | `hypothesis` | hypothesis only |
| `rag.prose.enterprise` | `span_sequence_aggregator` | `xlstm_or_light_sequence_head` | `hypothesis` | hypothesis only |
| `rag.prose.enterprise` | `data_repair_first` | `label_pipeline_fix` | `required` | hypothesis only |
| `rag.prose.enterprise` | `existing_tiny_mlp_probe` | `mlp_probe` | `rejected` | train_accuracy=0.4964 |
| `rag.prose.short_factoid` | `v1_router` | `baseline` | `baseline` | binary_grounded_accuracy=0.6334, ungrounded_f1=0.6142, paired_score_accuracy=0.65 |
| `rag.prose.short_factoid` | `optimized_feature_calibrator` | `mlp_logistic_xgboost_feature_head` | `passthrough_no_lift` | binary_grounded_accuracy=0.6334, ungrounded_f1=0.6142, paired_score_accuracy=0.65 |
| `rag.prose.short_factoid` | `compact_claim_evidence_encoder` | `t5_small_or_tiny_cross_encoder` | `hypothesis` | hypothesis only |
| `rag.prose.short_factoid` | `span_sequence_aggregator` | `xlstm_or_light_sequence_head` | `hypothesis` | hypothesis only |
| `rag.prose.short_factoid` | `data_repair_first` | `label_pipeline_fix` | `required` | hypothesis only |
| `rag.prose.short_factoid` | `existing_tiny_mlp_probe` | `mlp_probe` | `rejected` | train_accuracy=0.4964 |
| `rag.prose.multi_claim` | `v1_router` | `baseline` | `baseline` | binary_grounded_accuracy=0.436, ungrounded_f1=0.4899, paired_score_accuracy=0.5941 |
| `rag.prose.multi_claim` | `optimized_feature_calibrator` | `mlp_logistic_xgboost_feature_head` | `passthrough_no_lift` | binary_grounded_accuracy=0.436, ungrounded_f1=0.4899, paired_score_accuracy=0.5941 |
| `rag.prose.multi_claim` | `compact_claim_evidence_encoder` | `t5_small_or_tiny_cross_encoder` | `hypothesis` | hypothesis only |
| `rag.prose.multi_claim` | `span_sequence_aggregator` | `xlstm_or_light_sequence_head` | `hypothesis` | hypothesis only |
| `rag.prose.multi_claim` | `data_repair_first` | `label_pipeline_fix` | `required` | hypothesis only |
| `rag.prose.multi_claim` | `existing_tiny_mlp_probe` | `mlp_probe` | `rejected` | train_accuracy=0.4964 |
| `rag.structured` | `v1_router` | `baseline` | `baseline` | binary_grounded_accuracy=0.6543, ungrounded_f1=0.791, paired_score_accuracy=0.0 |
| `rag.structured` | `optimized_feature_calibrator` | `mlp_logistic_xgboost_feature_head` | `available` | binary_grounded_accuracy=0.6543, ungrounded_f1=0.791, paired_score_accuracy=1.0 |
| `rag.structured` | `compact_claim_evidence_encoder` | `t5_small_or_tiny_cross_encoder` | `hypothesis` | hypothesis only |
| `rag.structured` | `span_sequence_aggregator` | `xlstm_or_light_sequence_head` | `hypothesis` | hypothesis only |
| `rag.structured` | `data_repair_first` | `label_pipeline_fix` | `not_primary` | hypothesis only |
| `rag.structured` | `existing_tiny_mlp_probe` | `mlp_probe` | `rejected` | train_accuracy=0.4964 |
| `rag.code_in_context` | `v1_router` | `baseline` | `baseline` | binary_grounded_accuracy=1.0, ungrounded_f1=0.0 |
| `rag.code_in_context` | `optimized_feature_calibrator` | `mlp_logistic_xgboost_feature_head` | `passthrough_no_lift` | binary_grounded_accuracy=1.0, ungrounded_f1=0.0 |
| `rag.code_in_context` | `compact_claim_evidence_encoder` | `t5_small_or_tiny_cross_encoder` | `hypothesis` | hypothesis only |
| `rag.code_in_context` | `span_sequence_aggregator` | `xlstm_or_light_sequence_head` | `hypothesis` | hypothesis only |
| `rag.code_in_context` | `data_repair_first` | `label_pipeline_fix` | `not_primary` | hypothesis only |
| `rag.code_in_context` | `existing_tiny_mlp_probe` | `mlp_probe` | `rejected` | train_accuracy=0.4964 |
| `code.agentic_trace` | `v1_router` | `baseline` | `baseline` | binary_grounded_accuracy=1.0, ungrounded_f1=1.0 |
| `code.agentic_trace` | `optimized_feature_calibrator` | `mlp_logistic_xgboost_feature_head` | `passthrough_no_lift` | binary_grounded_accuracy=1.0, ungrounded_f1=1.0 |
| `code.agentic_trace` | `compact_claim_evidence_encoder` | `t5_small_or_tiny_cross_encoder` | `hypothesis` | hypothesis only |
| `code.agentic_trace` | `span_sequence_aggregator` | `xlstm_or_light_sequence_head` | `hypothesis` | hypothesis only |
| `code.agentic_trace` | `data_repair_first` | `label_pipeline_fix` | `not_primary` | hypothesis only |
| `code.agentic_trace` | `trajectory_symbolic_ranker` | `code_specific_symbolic_plus_learned_ranker` | `evaluated_rejected` | best_trajectory_auroc=0.7917, dedicated_head_min_heldout_auroc=0.6061007957559682 |
| `code.agentic_trace` | `existing_tiny_mlp_probe` | `mlp_probe` | `rejected` | train_accuracy=0.4964 |

## Production Gating

Enabled allow/block classes:
- `rag.prose.enterprise` via `optimized_feature_calibrator`

Repair-only classes:
- `rag.prose.short_factoid`: partial-support errors require semantic claim/evidence modelling
- `rag.prose.multi_claim`: partial-support errors require semantic claim/evidence modelling
- `rag.structured`: partial-support errors require semantic claim/evidence modelling
- `rag.code_in_context`: no candidate has class-specific proof beyond v1 passthrough
- `code.agentic_trace`: manufactured trajectory AUROC 0.7917 is below 0.90 gate

## Coding Trajectory Evidence
- `transcripts_v1`: cases=20, best_auroc=0.7344, best_cell=`sentence_packed|gte_only`, flagship_auroc=0.625
- `transcripts_v2`: cases=180, best_auroc=0.7917, best_cell=`sentence_packed|gte_only`, flagship_auroc=0.6969
- `both`: cases=62, best_auroc=0.6247, best_cell=`sentence_packed|gte_only`, flagship_auroc=0.5955
- dedicated trajectory head: train_bank=`transcripts_v1`, promotion_decision=`do_not_promote_keep_code_agentic_trace_repair_only`

## Definition Of Done Status

- Root-cause report: complete.
- Candidate bake-off harness: complete, with hypotheses separated from proven heads.
- Class-head selection: complete, conservative.
- Coding trajectory model: built and evaluated; not production-promoted.
- Production gating: only proven class is enabled; the rest remain repair-only.
