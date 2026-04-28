# Config audit — external bench vs website/benchmarks.md

**Scope:** reconcile what `bench_external.py`, the RunPod `handler.py`, and the
`quality` runtime preset actually enable vs what the website claims
(`portal/src/components/marketing/trace/benchmark-proof.tsx`,
`docs/benchmarks.md`). All line numbers are at the commit that produced this
audit.

## Published numbers (what we have to reproduce)

| Bench | Metric | Website value | Source artefact |
|---|---|---|---|
| HaluEval QA (English NLI) | paired_accuracy | 0.783 | `research/triangular_maxsim/reports/halueval_diagnose_deberta_en_n60.json` |
| HaluEval Summ (English NLI) | paired_accuracy | 0.75 | same report |
| HaluEval QA (multilingual NLI) | paired_accuracy | 0.667 | `research/triangular_maxsim/reports/halueval_diagnose_mdeberta_n60.json` |
| RAGTruth QA | F1 @ best-threshold, paired | >=0.73 (docs/benchmarks.md) | `scripts/factscore_quality_experiments.py` family |

## Gaps found

1. **Hosted-profile fusion weights differ from the sweep winner the website cites.**
   `latence_trace/api/service.py::PROFILE_ENV_PRESETS["quality"]` sets
   `VOYAGER_GROUNDEDNESS_FUSION_W_CALIBRATED=0.0`,
   `..._W_LITERAL=0.2`, `..._W_NLI=0.8`.
   The reconciled `diagnose_halueval.py` reports (which the website quotes) used
   `calibrated=0.5`, `literal=0.2`, `nli=0.3`.
   Consequence: the hosted `quality` request profile is *different* from the
   config that produced the website numbers. Not a runtime bug, but the
   metric the website advertises is not what the hosted endpoint computes
   under default `profile=quality`.

2. **English NLI vs multilingual NLI not routed automatically.**
   `diagnose_halueval.py` runs English inputs through
   `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`. The hosted worker pulls
   whatever `VOYAGER_GROUNDEDNESS_NLI_MODEL` is at launch and uses it for every
   request regardless of input language. The website's +12pp lift on HaluEval
   QA (0.667 -> 0.783) comes from the English model exclusively. `bench_external.py`
   did not pin the English model, so it measured the multilingual number.

3. **`nli_use_atomic_claims` is a request-scoped override, but my prior
   `bench_external.py` runs relied on the preset default.** Under hosted
   `quality` the default is already `true` (`PROFILE_ENV_PRESETS["quality"]`
   sets `VOYAGER_GROUNDEDNESS_NLI_ATOMIC_CLAIMS=1`). Under hosted `standard`
   the default is `false`. My `standard` run therefore had atomic claims
   OFF; my `quality` run had them ON - consistent with expectations, but
   under a different fusion.

4. **Metric reporting mismatch.** `bench_external.py` reports red/green/amber
   precision at the fixed hosted thresholds only. The website reports
   `F1 @ best-threshold` and `paired_accuracy`. These are not comparable.
   Recomputing F1@best and paired_accuracy from the existing quality-profile
   per-row artefacts is required before concluding anything about regression
   vs reconciliation.

## Decisions

- Treat gap 1 (fusion weights) as a config *drift*, not a bug. The "quality"
  preset was updated after the `diagnose_halueval.py` sweep produced the
  published numbers. Phase 1 bench run must pin `calibrated=0.5, literal=0.2,
  nli=0.3` explicitly via per-request env-override to match the
  published methodology; otherwise we are not measuring the advertised
  configuration.
- Treat gap 2 (English-vs-multilingual NLI) as a config lever, not a bug.
  Fix in bench harness by launching dev_app with
  `VOYAGER_GROUNDEDNESS_NLI_MODEL=MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`
  for English-only benches and passing `profile=quality` so the rest of the
  quality stack (atomic claims, reranker, premise concat) stays on.
- Treat gap 4 (metric mismatch) as the primary reason my prior conclusion
  ("score distributions for hallucinated vs grounded are indistinguishable")
  was wrong. Paired-accuracy and F1@best are the defensible metrics; red/green
  precision at fixed thresholds is a secondary operational metric.

## Action items (feeding Phase 1.2 onward)

1. Extend `scripts/bench_external.py` to:
   - pass `extra={"nli_use_atomic_claims": True, "nli_fusion_weights": {"calibrated": 0.5, "literal": 0.2, "nli": 0.3}}` on every call (requires service-side support; see below).
   - add F1@best-threshold, paired ranking accuracy, and ROC-AUC alongside the existing precision metrics.
   - document the effective model/fusion in the run header.
2. Extend `latence_trace/api/service.py::resolve_request_runtime_profile` so
   callers can pass a per-request fusion override (not just atomic-claims).
   The current path only accepts `nli_use_atomic_claims`; other knobs require
   process-env changes. Add an optional `nli_fusion_weights` field to the
   request DTO and thread it through to `groundedness.compute_groundedness_v2`.
3. Launch dev_app for the reconciliation run with:
   ```
   LATENCE_TRACE_PROFILE=quality \
   VOYAGER_GROUNDEDNESS_NLI_ATOMIC_CLAIMS=1 \
   VOYAGER_GROUNDEDNESS_NLI_MODEL=MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli \
   VOYAGER_GROUNDEDNESS_NLI_PREMISE_RERANKER_MODEL=BAAI/bge-reranker-v2-m3 \
   VOYAGER_GROUNDEDNESS_FUSION_W_CALIBRATED=0.5 \
   VOYAGER_GROUNDEDNESS_FUSION_W_LITERAL=0.2 \
   VOYAGER_GROUNDEDNESS_FUSION_W_NLI=0.3 \
   python runpod/dev_app.py
   ```
   This should reproduce the `diagnose_halueval.py` config exactly.
4. Reconcile: if paired_accuracy on HaluEval QA is within 3pp of 0.78,
   ratify the website. Otherwise, log a regression with the specific source
   (model checksum, fusion weights, prompt bundle).
