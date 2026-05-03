# LLM A/B Equivalence Report

- Model: `gpt-5.5`
- Cases: `1`
- Pass gate: `PASS`
- Classification counts: `{"compressed_better": 1}`
- Mean token reduction: `0.0279`
- Median token reduction: `0.0279`

## Cases

### trace_05_early_d4fbfaec
- Type: `coding_trace`
- Classification: `compressed_better`
- Tokens: `110982` full -> `107889` compressed (2.79% reduction)
- Repair triggered: `True`
- Judge equivalence: `material_delta`; better: `B`
- Rationale: Answer B is materially better because it includes the concrete request, exact todo IDs, exact command, files, matrix, cache/schema requirements, held-out passthrough fields, and the later implementation state and measured results from the trace. Answer A accurately summarizes the plan constraints but omits the exact todo IDs and most of the trace-derived implementation/results state, which the task explicitly asks for. Neither answer appears to invent unsupported facts, but B is much more complete and grounded in the source evidence.
- Repair triggers: `["missing_exact_terms_in_compressed_answer"]`
- Repair packet: `8` excerpts, `4424` tokens
