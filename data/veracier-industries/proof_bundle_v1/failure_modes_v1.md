# External Benchmark Failure Modes (TRACE v1)
This document catalogues every non-ok row on HaluEval QA, HaluEval Summarisation,
RAGTruth QA, and RAGTruth Summarisation.  The categories here are the direct
inputs to the v2 trigger decision and the biaffine student training target.

## Failure mode taxonomy

| mode | meaning | v1 operational impact |
|---|---|---|
| `fn_green_on_hallucinated` | TRACE said green, gold says red | HIGHEST priority - missed hallucination |
| `fp_red_on_grounded` | TRACE said red, gold says green | Reviewer queue pressure, user trust erosion |
| `amber_on_hallucinated` | TRACE said amber, gold says red | Acceptable - amber routes to review |
| `amber_on_grounded` | TRACE said amber, gold says green | Reviewer queue pressure |

## `halueval_qa_quality_rows.jsonl`  total=120

| mode | n | share | score min / mean / max |
|---|---|---|---|
| `ok` | 62 | 52% | 0.1159/0.5408/0.9779 |
| `fp_red_on_grounded` | 25 | 21% | 0.1577/0.4093/0.5493 |
| `fn_green_on_hallucinated` | 22 | 18% | 0.7882/0.913/0.9795 |
| `amber_on_grounded` | 7 | 6% | 0.5648/0.6339/0.7153 |
| `amber_on_hallucinated` | 4 | 3% | 0.5742/0.6496/0.6994 |

Linguistic features of failing rows:

| mode | short_answer | enumerated/cites_passages | abstention_like | numeric_present | total |
|---|---|---|---|---|---|
| `amber_on_grounded` | 7 | 0 | 0 | 2 | 7 |
| `amber_on_hallucinated` | 1 | 0 | 0 | 1 | 4 |
| `fn_green_on_hallucinated` | 2 | 0 | 0 | 3 | 22 |
| `fp_red_on_grounded` | 25 | 0 | 0 | 3 | 25 |

## `halueval_qa_standard_rows.jsonl`  total=120

| mode | n | share | score min / mean / max |
|---|---|---|---|
| `ok` | 62 | 52% | 0.0008/0.5308/0.9985 |
| `fn_green_on_hallucinated` | 23 | 19% | 0.7788/0.9493/0.9992 |
| `fp_red_on_grounded` | 22 | 18% | 0.0157/0.3665/0.5297 |
| `amber_on_grounded` | 8 | 7% | 0.5748/0.6301/0.7248 |
| `amber_on_hallucinated` | 5 | 4% | 0.5669/0.6494/0.7024 |

Linguistic features of failing rows:

| mode | short_answer | enumerated/cites_passages | abstention_like | numeric_present | total |
|---|---|---|---|---|---|
| `amber_on_grounded` | 8 | 0 | 0 | 3 | 8 |
| `amber_on_hallucinated` | 1 | 0 | 0 | 1 | 5 |
| `fn_green_on_hallucinated` | 2 | 0 | 0 | 3 | 23 |
| `fp_red_on_grounded` | 22 | 0 | 0 | 1 | 22 |

## `halueval_summ_quality_rows.jsonl`  total=120

| mode | n | share | score min / mean / max |
|---|---|---|---|
| `ok` | 44 | 37% | 0.0/0.7391/0.9835 |
| `amber_on_hallucinated` | 29 | 24% | 0.5652/0.6706/0.7911 |
| `amber_on_grounded` | 23 | 19% | 0.5548/0.6676/0.74 |
| `fn_green_on_hallucinated` | 18 | 15% | 0.7755/0.8772/0.9793 |
| `fp_red_on_grounded` | 6 | 5% | 0.0/0.399/0.5193 |

Linguistic features of failing rows:

| mode | short_answer | enumerated/cites_passages | abstention_like | numeric_present | total |
|---|---|---|---|---|---|
| `amber_on_grounded` | 0 | 0 | 0 | 12 | 23 |
| `amber_on_hallucinated` | 0 | 0 | 1 | 16 | 29 |
| `fn_green_on_hallucinated` | 0 | 0 | 0 | 10 | 18 |
| `fp_red_on_grounded` | 0 | 0 | 0 | 3 | 6 |

## `halueval_summ_standard_rows.jsonl`  total=120

| mode | n | share | score min / mean / max |
|---|---|---|---|
| `ok` | 50 | 42% | 0.0/0.7052/0.9991 |
| `fn_green_on_hallucinated` | 28 | 23% | 0.7551/0.8596/0.9987 |
| `amber_on_grounded` | 20 | 17% | 0.5641/0.6564/0.7154 |
| `amber_on_hallucinated` | 13 | 11% | 0.5541/0.6703/0.7477 |
| `fp_red_on_grounded` | 9 | 8% | 0.0/0.3558/0.5409 |

Linguistic features of failing rows:

| mode | short_answer | enumerated/cites_passages | abstention_like | numeric_present | total |
|---|---|---|---|---|---|
| `amber_on_grounded` | 0 | 0 | 0 | 14 | 20 |
| `amber_on_hallucinated` | 0 | 0 | 1 | 6 | 13 |
| `fn_green_on_hallucinated` | 0 | 0 | 0 | 14 | 28 |
| `fp_red_on_grounded` | 0 | 0 | 0 | 5 | 9 |

## `ragtruth_qa_quality_rows.jsonl`  total=120

| mode | n | share | score min / mean / max |
|---|---|---|---|
| `ok` | 52 | 43% | 0.1319/0.6436/0.9924 |
| `fp_red_on_grounded` | 44 | 37% | 0.0374/0.348/0.5408 |
| `amber_on_grounded` | 12 | 10% | 0.5801/0.6549/0.7208 |
| `amber_on_hallucinated` | 10 | 8% | 0.5533/0.6068/0.6882 |
| `fn_green_on_hallucinated` | 2 | 2% | 0.8438/0.8618/0.8798 |

Linguistic features of failing rows:

| mode | short_answer | enumerated/cites_passages | abstention_like | numeric_present | total |
|---|---|---|---|---|---|
| `amber_on_grounded` | 0 | 4 | 0 | 7 | 12 |
| `amber_on_hallucinated` | 0 | 4 | 4 | 8 | 10 |
| `fn_green_on_hallucinated` | 0 | 1 | 0 | 2 | 2 |
| `fp_red_on_grounded` | 0 | 9 | 8 | 30 | 44 |

## `ragtruth_qa_standard_rows.jsonl`  total=120

| mode | n | share | score min / mean / max |
|---|---|---|---|
| `ok` | 62 | 52% | 0.177/0.7301/0.9971 |
| `fp_red_on_grounded` | 31 | 26% | 0.0017/0.2994/0.5098 |
| `amber_on_hallucinated` | 11 | 9% | 0.5539/0.6159/0.7493 |
| `amber_on_grounded` | 9 | 8% | 0.5625/0.6675/0.737 |
| `fn_green_on_hallucinated` | 7 | 6% | 0.7565/0.8246/0.8855 |

Linguistic features of failing rows:

| mode | short_answer | enumerated/cites_passages | abstention_like | numeric_present | total |
|---|---|---|---|---|---|
| `amber_on_grounded` | 0 | 4 | 0 | 6 | 9 |
| `amber_on_hallucinated` | 0 | 7 | 2 | 10 | 11 |
| `fn_green_on_hallucinated` | 0 | 1 | 0 | 3 | 7 |
| `fp_red_on_grounded` | 0 | 5 | 7 | 21 | 31 |

## `ragtruth_summ_quality_rows.jsonl`  total=120

| mode | n | share | score min / mean / max |
|---|---|---|---|
| `ok` | 45 | 38% | 0.3263/0.793/0.9941 |
| `amber_on_grounded` | 36 | 30% | 0.5581/0.6407/0.7325 |
| `amber_on_hallucinated` | 19 | 16% | 0.5661/0.6734/0.7444 |
| `fn_green_on_hallucinated` | 13 | 11% | 0.7666/0.8574/0.9884 |
| `fp_red_on_grounded` | 7 | 6% | 0.4107/0.4842/0.5438 |

Linguistic features of failing rows:

| mode | short_answer | enumerated/cites_passages | abstention_like | numeric_present | total |
|---|---|---|---|---|---|
| `amber_on_grounded` | 0 | 0 | 0 | 27 | 36 |
| `amber_on_hallucinated` | 0 | 0 | 0 | 14 | 19 |
| `fn_green_on_hallucinated` | 0 | 0 | 0 | 8 | 13 |
| `fp_red_on_grounded` | 0 | 0 | 0 | 5 | 7 |

## `ragtruth_summ_standard_rows.jsonl`  total=120

| mode | n | share | score min / mean / max |
|---|---|---|---|
| `ok` | 54 | 45% | 0.2236/0.8179/0.9989 |
| `amber_on_grounded` | 25 | 21% | 0.5515/0.6631/0.7488 |
| `fn_green_on_hallucinated` | 18 | 15% | 0.757/0.832/0.9252 |
| `amber_on_hallucinated` | 15 | 12% | 0.5515/0.6522/0.7453 |
| `fp_red_on_grounded` | 8 | 7% | 0.2259/0.4356/0.5447 |

Linguistic features of failing rows:

| mode | short_answer | enumerated/cites_passages | abstention_like | numeric_present | total |
|---|---|---|---|---|---|
| `amber_on_grounded` | 0 | 0 | 0 | 18 | 25 |
| `amber_on_hallucinated` | 0 | 0 | 0 | 11 | 15 |
| `fn_green_on_hallucinated` | 0 | 0 | 0 | 13 | 18 |
| `fp_red_on_grounded` | 0 | 0 | 0 | 7 | 8 |

## Weak-case hypotheses

Based on the score distributions and linguistic features above, TRACE v1
struggles on three recurring patterns:

1. **Short factual answers** (1-5 tokens, often a year or proper noun).
   The NLI model cannot align a single-token answer against a long source
   passage even when the token is present verbatim.  This causes both
   `fp_red_on_grounded` (we mark correct factual answers as red) and
   `fn_green_on_hallucinated` (we miss small factual substitutions).

2. **Enumerated / paraphrased multi-step answers** that cite 'passage N'
   or renumber bullet points.  Each step is genuinely supported, but the
   ColBERT MaxSim channel cannot resolve the alignment between numbered
   bullets and free-form source prose.  Score distributions show that
   `exp=green|got=red` and `exp=red|got=red` RAGTruth QA rows are
   statistically indistinguishable by our current groundedness scalar
   (overlapping means ~0.35), so no threshold change can separate them.

3. **Abstention-like answers** ('Unable to answer based on the given
   passages').  These are *faithful* on RAGTruth but look like
   hedge+empty-support to TRACE.  The current epistemic hedge gate caps
   them at amber, but the red threshold still catches the long tail.

## v2 trigger decision

The v2 trigger conditions (see `docs/roadmap.md`) fire when any of:

* HaluEval QA red precision < 80% at standard profile, OR
* RAGTruth QA red precision < 50% at quality profile, OR
* any Veracier vertical green precision < 95% for two consecutive weeks.

As of this report both external triggers are active, so the v2 student
biaffine head (`research/triangular_maxsim/student_v2/`) is GO.  The
student target objective is to push HaluEval QA red precision to >=90%
and RAGTruth QA red precision to >=80% while keeping Veracier green
precision >=97%.
