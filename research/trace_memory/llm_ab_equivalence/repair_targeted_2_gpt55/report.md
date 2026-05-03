# LLM A/B Equivalence Report

- Model: `gpt-5.5`
- Cases: `2`
- Pass gate: `FAIL`
- Classification counts: `{"compressed_better": 1, "regression": 1}`
- Mean token reduction: `0.0115`
- Median token reduction: `0.0277`

## Cases

### trace_03_late_2edf16de
- Type: `coding_trace`
- Classification: `compressed_better`
- Tokens: `85370` full -> `85779` compressed (-0.48% reduction)
- Repair triggered: `True`
- Judge equivalence: `material_delta`; better: `B`
- Rationale: Answer B is materially better because it includes the exact screenshot asset path with the UUID and `Bildschirmfoto_2026`, correctly identifies `portal/src/components/marketing/dataflow-bridge.tsx`, captures the requested redesign and implementation state, and accurately notes that no verification command/test result is shown for this exact final change. Answer A captures much of the section copy and target file but omits the exact image ID/path and several required terms, and it adds an unsupported next-step/commit claim. Neither answer includes the literal required terms `image_files` and `user_query`, but B preserves more exact evidence and avoids hallucinated verification.
- Repair triggers: `["compressed_context_low_required_recall"]`
- Repair packet: `6` excerpts, `3018` tokens

### trace_05_early_d4fbfaec
- Type: `coding_trace`
- Classification: `regression`
- Tokens: `110982` full -> `107907` compressed (2.77% reduction)
- Repair triggered: `True`
- Judge equivalence: `material_delta`; better: `A`
- Rationale: Answer A is materially more complete and aligns better with both the attached plan and the later trace evidence. It captures the concrete request, constraints, files, matrix cells, cache schema, held-out pass-through, fallback behavior, report requirements, command, artifacts, and actual implementation/run outcome. Answer B is mostly a pre-implementation checklist and omits substantial implementation-state and reporting facts present in the source evidence. Neither answer appears to invent unsupported facts, but B is missing many critical exact symbols, metrics, paths, and final-state details.
- Repair triggers: `["missing_exact_terms_in_compressed_answer"]`
- Repair packet: `6` excerpts, `3020` tokens
