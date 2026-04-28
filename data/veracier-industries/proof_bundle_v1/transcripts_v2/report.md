# Transcripts-v2 agentic-coding bench

Generated: 2026-04-28. Code lane on 60 real agentic-coding scenarios
mined from Cursor agent sessions.

* Corpus: `research/triangular_maxsim/coding/cases_transcripts_v2.yaml`
  — 60 base scenarios × 3 variants (correct / wrong / ambiguous) = 180
  rows. Each row ships with the real context-file bundle from the agent
  turn.
* Context packing: up to 20 files, up to 60 KB per request, delimited
  by `=== <path> ===` headers so tree-sitter and the literal-novelty
  channel can anchor spans to paths.
* Stack: dev_app `quality` profile + English DeBERTa NLI + atomic
  claims ON + BGE reranker, `scoring_mode=code`,
  `response_language_hint=python`, n=60 seed-free (the corpus order
  is deterministic).

## Headline

| Matchup | n_pairs | **paired_accuracy** | ties | F1@best-threshold |
|---|---:|---:|---:|---:|
| correct vs wrong      | 60 | **0.817** (49 / 9) | 2 | 0.702 |
| correct vs ambiguous  | 60 | 0.550 (33 / 23)    | 4 | 0.670 |

**The code lane clears the 0.80 paired-accuracy gate on real
agentic-coding traces.** The earlier bench on HumanEval+ /
CRUXEval-synthetic adversarials (0.22 / 0.50 / 0.11) was measuring the
wrong thing — swaps inside the prompt that was also in `raw_context`
produced artificial ties. On real traces the code lane actually works.

## Score distributions (composite_phantom_score)

| Subcategory | n | mean | stdev | min | max |
|---|---:|---:|---:|---:|---:|
| correct   | 60 | 0.6586 | 0.0092 | 0.6320 | 0.6774 |
| wrong     | 60 | 0.6542 | 0.0089 | 0.6320 | 0.6730 |
| ambiguous | 60 | 0.6581 | 0.0090 | 0.6343 | 0.6771 |

Correct–wrong delta: mean +0.00437, median +0.00283. Small in absolute
terms but directionally consistent (49 of 60 pairs positive, 9
negative, 2 tie). That is not noise — it is a real signal on a hard
corpus where grounded and ungrounded responses share 95 %+ of their
token surface by construction (same imports, same braces, same Python
keywords, swapped identifiers only).

## The classifier layer is not calibrated for this corpus

| Subcategory | `composite_phantom_verdict == True` | `composite_phantom_verdict == False` |
|---|---:|---:|
| correct   | 0 | 60 |
| wrong     | 0 | 60 |
| ambiguous | 0 | 60 |

The composite phantom verdict fires `False` on **every row**, including
the wrong cases. The underlying score discriminates, but the threshold
/ classifier head that converts score → band is tuned for a different
operating distribution. Same pattern observed on external prose
benches: signal is there, deployment defaults are not universal.

This is exactly the lever the use-case classifier is designed to pull:
detect "agentic-coding trace" as a corpus type and load thresholds +
composite-head calibration specific to that corpus.

## What this means for the "production-ready" question

| Dimension | Verdict |
|---|---|
| Score-level discrimination of correct vs wrong (paired_acc) | **0.817 → clears gate** |
| Score-level discrimination of correct vs ambiguous | 0.55 — by design hard (ambiguous ≈ correct) |
| Band / phantom-verdict output today | **Wrong**: fires `False` on 100 % of rows |
| Ready to ship as-is with hosted thresholds | **No** |
| Ready to ship behind a use-case classifier that picks agentic-coding thresholds | **Plausibly yes** (pending threshold calibration pass on this corpus) |

## Why the earlier phase-3 result was misleading

The phase-3 coding bench (`coding_bench/` and
`coding_bench_code_lane/`) used HumanEval+ and CRUXEval as the corpus.
Our adversarial generator swapped identifiers inside the function
signature, but in the row shape we send to TRACE the `raw_context`
also contains the function signature. So the swap was present in both
context and response, and the code lane saw no novelty between them,
which is exactly the correct behaviour for a local-support detector.

That result (0.22 / 0.50 / 0.11 paired accuracy) was measuring the
contamination, not the code lane. transcripts_v2 does not have that
contamination: the context is the full file set from the agent turn,
the response is the assistant's reply, and the swap lives inside
prose/code that is not otherwise present in the context. The 0.817
number is the one to carry forward.

## Next steps

1. Calibrate `composite_phantom_score` → `composite_phantom_verdict`
   thresholds against transcripts_v2. Today the head never fires; it
   should fire on the 9 cases where wrong > correct is the ceiling.
2. Add "agentic-coding-trace" as a corpus label to the planned
   use-case classifier so a tenant running a coding agent lands on the
   right thresholds + fusion defaults automatically.
3. Re-score the same 60 scenarios through the RAG lane for a side-by
   -side (running separately as
   `transcripts_v2_rag_lane/`).

## Artefacts

* Rows: `rows.jsonl` (180 lines)
* Summary: `summary.json`
* Bench script: `scripts/bench_transcripts_v2.py`
* Corpus loader: `research/triangular_maxsim/coding/transcript_cases_v2.py`
