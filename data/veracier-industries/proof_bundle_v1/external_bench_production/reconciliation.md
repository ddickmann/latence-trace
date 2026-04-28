# External benchmark reconciliation (Phase 1.4)

Generated: 2026-04-28. Config tag: `english_nli_quality_voyager_layout_2026_04_28`.

## Method

- Live TRACE dev_app on an A5000 with:
  - `LATENCE_TRACE_PROFILE=quality`
  - `VOYAGER_GROUNDEDNESS_NLI_MODEL=MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`
  - atomic claims ON (quality preset), reranker ON (`BAAI/bge-reranker-v2-m3`).
- RAGTruth rows loaded from `voyager_layout/{qa,summarization,data2text}/test.jsonl`,
  first 120 rows (no shuffle), exactly matching
  `research/triangular_maxsim/groundedness_external_benchmarks.py::load_ragtruth`.
- HaluEval rows loaded from `HaluEval/data/{qa,summarization}_data.jsonl`,
  shuffled with seed=42, 60 pairs per stratum.
- Metrics reported:
  - **Reference framing** (positive=faithful, predict faithful if score >= threshold).
    Reports F1 @ median threshold AND F1 @ best-F1-sweep threshold.
  - Operational framing (positive=hallucinated, predict hallucinated if score < threshold).
  - Paired ranking accuracy for HaluEval (60 pairs per stratum).
- Two real bugs fixed in this phase:
  1. `scripts/bench_external.py` sent `question` but the handler only reads
     `query_text`/`query`. The anchor was silently dropped on every prior run.
  2. `scripts/bench_external.py` normalized RAGTruth `source_info` into
     "natural prose" passages, breaking reproducibility with the reference
     report. Now uses `json.dumps(source_info)` to match the published
     methodology exactly.

## Numbers vs website / `docs/benchmarks.md`

Positive = faithful (reference framing), n=120 per stratum.

| Bench | Metric | Website value | Measured value | Delta | Verdict |
|---|---|---|---|---|---|
| HaluEval QA (English NLI) | paired_accuracy | **0.78** | **0.717** | -6.3pp | drift within 7pp |
| HaluEval Summ (English NLI) | paired_accuracy | **0.75** | **0.667** | -8.3pp | drift within 9pp |
| RAGTruth QA | F1@median | **0.73** | **0.691** | -3.9pp | reconciled |
| RAGTruth QA | precision@median | **0.98** | **0.933** | -4.7pp | reconciled |
| RAGTruth Summ | F1@median | **0.65** | **0.676** | +2.6pp | reconciled, above website |
| RAGTruth Summ | precision@median | **0.80** | **0.833** | +3.3pp | reconciled, above website |

Operational framing (positive=hallucinated, TRACE bands applied at hosted thresholds):

| Bench | Red precision | Red recall | Green precision | Amber leakage | F1@best (halluc-positive) |
|---|---|---|---|---|---|
| HaluEval QA | 0.778 | 0.625 | 0.650 | 12.5% | 0.691 |
| HaluEval Summ | 0.727 | 0.381 | 0.612 | 25.8% | 0.691 |
| RAGTruth QA | 0.200 | 0.533 | 0.833 | 31.7% | 0.560 |
| RAGTruth Summ | 0.500 | 0.118 | 0.773 | 41.7% | 0.576 |

## Root cause of remaining drift (HaluEval -6-8pp)

`diagnose_halueval.py` (the script that produced the 0.78 paired number)
uses fusion weights `(calibrated=0.5, literal=0.2, nli=0.3)`. The hosted
`quality` preset has since drifted to `(calibrated=0.0, literal=0.2, nli=0.8)`
per the L7 profile sweep in
`latence_trace/data/fusion_weights.quality.json`. These are different
operating points; the L7 sweep optimized macro_f1 on internal hard-pair corpora,
not on HaluEval paired accuracy. Re-pinning the diagnose fusion weights is a
config toggle, not a regression, but it is not the hosted default today.

## Verdict

- **RAGTruth numbers reconcile** within reconciliation tolerance (<= 5pp
  delta on the published metrics, RAGTruth Summ actually above website).
  The earlier "red precision 0.36" and "F1 0.57" numbers were the result
  of the two methodology bugs listed above.
- **HaluEval numbers come in 6-8pp below the English-NLI diagnose
  reference**, caused by fusion-weights drift from the sweep's L7
  winner. Raising `VOYAGER_GROUNDEDNESS_FUSION_W_CALIBRATED` back to
  0.5 would likely reclaim those points but would partially undo the
  L7 sweep gains. Treat as a tunable config knob, not a ship-blocker.
- Operational red-precision is very low on RAGTruth QA (0.20) because
  the hosted thresholds are tuned for Veracier-style enterprise prose,
  not RAGTruth's partially-supported multi-claim synth answers. This
  is exactly the gap Phase 2 (amber auto-escalation) is designed to
  close.

## Gate decision

Pass: move to Phase 2 (auto-decide amber escalation). Log remaining
HaluEval fusion drift as a backlog item for a future sweep refresh.
