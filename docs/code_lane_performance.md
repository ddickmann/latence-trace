# Code lane — per-signal performance budget

Every signal in the code lane is held to an explicit latency budget.
The budgets are additive — the sum is the worst-case code-lane p95
and MUST remain under the **150 ms lane SLO**. Cascade-only signals
(NLI, semantic entropy) are only counted when the composite lands in
the ambiguity band, so their amortised contribution is the per-turn
cost × the measured fire rate.

| Signal | Budget (p95) | Cascade? | Measured (sprint bench) |
| --- | --- | --- | --- |
| `GPUScorer` (MaxSim + per-token p10 + owner) | 15 ms | always | 6 ms |
| `AstSymbolExtractor` (tree-sitter parse + diff) | 10 ms | always | 4 ms |
| `LiteralNovelty` | 2 ms | always | <1 ms |
| `LogisticComposite` | 1 ms | always | <1 ms |
| `FileAttribution` (reason codes, EMA inputs) | 5 ms | always | 2 ms |
| `NLICascade` (vLLM-factory entail, top-3 premises) | 25 ms | ~30 % fire | 22 ms (stub) |
| `SemanticEntropy` (caller samples) | 8 ms | opt-in | 3 ms |
| Orchestrator glue | 5 ms | always | 2 ms |
| **Total p95 (cascade on)** | **~71 ms** | | **~9 ms** (stub) |
| **Total p95 (cascade off)** | **~38 ms** | | **~5 ms** |

Budgets are re-validated in
`research/triangular_maxsim/coding/experiments/v3_cascade_latency_bench.py`.
Running the benchmark with a real vLLM-factory NLI server should
replace the stub NLI latency with ~15-25 ms of real round-trip,
keeping the end-to-end p95 well under 50 ms on an A100.

## How to refresh the budget

1. Run the cascade benchmark (`python -m research.triangular_maxsim.coding.experiments.v3_cascade_latency_bench --turns 120`).
2. Run the full-lane latency bench
   (`research/triangular_maxsim/coding/experiments/latency_bench.py`).
3. Update the "Measured" column; open a PR if any row exceeds its budget.
4. If the budget is tight, prefer these levers before widening a budget:
   - Tighten the cascade band (fewer turns trigger NLI).
   - Drop the AST extractor's node visit limit (see
     `latence_trace/core/code_lane/ast_grounding.py`).
   - Keep the logistic composite under 10 features (≤4 interaction terms).
