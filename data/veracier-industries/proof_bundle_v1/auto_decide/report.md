# Phase 2 auto-decide bench report

Generated: 2026-04-28. Config tag: `auto_decide_openai_gpt4omini_2026_04_28`.

## Method

- Live dev_app with English-NLI quality profile (same as Phase 1 reconciliation).
- `LATENCE_TRACE_AUTO_DECIDE_ENABLED=1`,
  `LATENCE_TRACE_AUTO_DECIDE_PROVIDER=openai`,
  `LATENCE_TRACE_AUTO_DECIDE_MODEL=gpt-4o-mini`.
- All 480 external-bench rows scored with `auto_decide=True`.
- The judge is called exactly once per amber response with the
  atomic-claim / evidence pair list plus the original query.

## Results

| Bench | Original amber-rate | Final amber rate | Auto-decide accuracy | Target | Verdict |
|---|---|---|---|---|---|
| HaluEval QA | 12.5% | 0.0% | **0.650** | >=0.85 | **fail** |
| HaluEval Summ | 25.8% | 2.5% | **0.632** | >=0.85 | **fail** |
| RAGTruth QA | 31.7% | 0.0% | **0.458** | >=0.90 | **fail** |
| RAGTruth Summ | 41.7% | 0.8% | **0.647** | >=0.90 | **fail** |

## Analysis

On every stratum the gate is missed. Breakdown per bench:

- **HaluEval QA**: Pre-escalation accuracy of the non-amber rows was ~0.64
  (57/89). Escalating the 31 amber rows recovered ~17/31 = 55% correct.
  The judge added only marginal signal on top of the ColBERT/NLI
  headline.
- **RAGTruth QA**: The judge, given the atomic-claim list produced by the
  NLI decomposer, collapsed amber mostly toward "red" even on faithful
  responses that contained paraphrased but supported claims. Net
  accuracy dropped from the TRACE-only baseline (where amber acts as a
  conservative hedge).
- **HaluEval Summ / RAGTruth Summ**: Same pattern. Multi-claim
  summaries with partial support are the hardest class for any single-
  hop judge.

## Cost + latency envelope

- Judge latency: ~2 s median per call (openai gpt-4o-mini, one call per amber).
- Judge cost: ~$0.0001 per call on average. On the full 480-row
  bench the total cost was < $0.02.
- Added end-to-end latency: proportional to the original amber rate. On
  a hosted tenant with 15% amber rate the p95 full-round-trip grows by
  ~300 ms per request on average.

## Verdict

- Do **not** ship auto-decide ON as the hosted default. The judge does
  not, at the current payload size + model, move the needle on the
  external benchmarks.
- Keep the middleware enabled as an opt-in knob for pilot tenants and
  for the code lane (Phase 3), where the judge call has access to
  stronger lexical/structural signals (identifier / literal diffs).
- Add to the Phase 3.5 evidence-report list: the current single-claim-
  list judge payload is insufficient; a richer payload (full atomic
  verification table plus the top-k supporting spans per claim) is
  the next experiment before considering auto-decide as a shipping
  feature.

## Shipping decision

Gate not met. Leave auto-decide OFF by default. Reconsider after the
richer payload experiment in Phase 3.5.
