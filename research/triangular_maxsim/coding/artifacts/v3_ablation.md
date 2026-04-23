# Code-lane v3 ablation matrix

Source: `research/triangular_maxsim/coding/artifacts/v2_full_gpu_scorer_rescore.json` (180 rows).

| phase | AUROC correct-vs-wrong | AUROC correct-vs-ambiguous | AUROC grounded-vs-ungrounded | monotonicity |
| --- | --- | --- | --- | --- |
| baseline | 0.787 | 0.554 | 0.776 | 0.883 |
| +literal_guard | 0.665 | 0.533 | 0.653 | 0.650 |
| +per_token_p10 | 0.691 | 0.537 | 0.677 | 0.733 |
| +literal_novelty | — | — | — | — |
| +ast | — | — | — | — |
| +nli_cascade | — | — | — | — |
| +logistic | 0.671 | 0.532 | 0.657 | 0.717 |

## Phase notes

- **baseline** — reverse_context only.
- **+literal_guard** — 0.7*reverse_context + 0.3*literal_guarded.
- **+per_token_p10** — v2 linear composite (0.55/0.25/0.20).
- **+literal_novelty** — literal-novelty-at-high-confidence (requires full rerun). literal_novelty_min is computed only by the production code-lane orchestrator; this rescore bank predates the signal. Rerun score_code_groundedness over transcripts_v2 to activate this phase.
- **+ast** — tree-sitter AST drift + phantom symbol counts. ast_literal_drift_count / ast_phantom_symbol_count require the multi-language extractor from latence_trace.core.code_lane.ast_grounding; not captured in the pre-sprint rescore bank.
- **+nli_cascade** — ambiguity-triggered NLI on top-3 evidence. Requires a live vLLM NLI server; run the full code lane against transcripts_v2 with NLI enabled to collect this column.
- **+logistic** — LogisticRegression over (reverse_context, literal_guarded, per_token_p10, reverse_context*literal_guarded); trained in-sample (correct=0, ambiguous+wrong=1).
