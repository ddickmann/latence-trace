# Post-Phase-3.5 follow-up experiments

Generated: 2026-04-28. Scope: two "cheap" experiments recommended at the
hard-stop checkpoint before committing to v2 student training.

* Experiment 1: benchmark the existing code lane (`scoring_mode="code"`)
  on the coding adversarials, to verify whether v1 is actually
  code-blind or whether we only tested the RAG lane.
* Experiment 2: pin the fusion weights to the `diagnose_halueval.py`
  reference `(calibrated=0.5, literal=0.2, nli=0.3)` and rerun the
  external prose benches, to check how much of the HaluEval drift is
  attributable to fusion-weight drift from the L7 sweep winner
  `(0.0, 0.2, 0.8)`.

Both experiments ran against a fresh dev_app with the same stack as the
Phase 1 reconciliation (English DeBERTa NLI + atomic claims ON + BGE
reranker + quality profile thresholds).

## Experiment 1 — code lane on coding adversarials

Artefacts:

* Summary: [`../coding_bench_code_lane/summary.json`](../coding_bench_code_lane/summary.json)
* Rows:    [`../coding_bench_code_lane/rows.jsonl`](../coding_bench_code_lane/rows.jsonl)
* Report:  [`../coding_bench_code_lane/report.md`](../coding_bench_code_lane/report.md)

| Variant | n_pairs | paired_acc (code lane) | paired_acc (RAG lane, phase 3) | F1@best (code) |
|---|---:|---:|---:|---:|
| identifier_swap | 111 | **0.22** | 0.27 | 0.67 |
| literal_swap | 225 | **0.50** | 0.62 | 0.67 |
| api_signature_swap | 61 | **0.11** | 0.21 | 0.67 |

**Gate target: per-variant paired_acc >= 0.80.** Neither lane passes.

Code lane paired accuracy is actually lower than the RAG lane because
the code lane produces a large number of ties (68 / 57 / 40 respectively)
on the grounded-vs-adversarial pair. Inspecting rows shows the root
cause: our adversarial generator swaps identifiers inside the HumanEval+
function signature, but `raw_context` in our row shape *also includes
the full function signature*. The AST drift and literal-novelty
detectors therefore see no novelty between response and context, which
is exactly the correct behaviour for a local-support detector. The
benchmark is measuring the wrong thing on this corpus layout.

Net conclusion:

* The code lane does **not** close the coding adversarial gap on this
  benchmark.
* The benchmark itself has a `response == context + body` contamination
  that neither lane can overcome without an architectural change to
  either the generator (swap inside body, not signature) or the scorer
  (compare response-body against a reference solution, not the prompt).
* A properly constructed coding adversarial benchmark (body-only swaps
  with a separate reference solution as context) is required before we
  conclude v1 is code-blind on agentic-coding hallucinations.

## Experiment 2 — diagnose fusion weights on external prose

Artefacts:

* Summary: [`../external_bench_diagnose_fusion/summary_quality.json`](../external_bench_diagnose_fusion/summary_quality.json)
* Rows:    `../external_bench_diagnose_fusion/{halueval_qa,halueval_summ,ragtruth_qa,ragtruth_summ}_quality_rows.jsonl`
* Shim:    [`runpod/dev_app_diagnose_fusion.py`](../../../../latence-trace/runpod/dev_app_diagnose_fusion.py)
  pins `PROFILE_ENV_PRESETS["quality"]` fusion weights to
  `(calibrated=0.5, literal=0.2, nli=0.3)` before handler import.

Reference framing (positive = faithful, higher score = more faithful):

| Bench | Metric | Quality L7 (nli=0.8) | Diagnose pin (0.5/0.2/0.3) | Website | Delta vs L7 |
|---|---|---:|---:|---:|---:|
| HaluEval QA | paired_accuracy (n=60 pairs) | 0.717 | **0.417** | 0.78 | **−30pp (worse)** |
| HaluEval QA | ROC_AUC | 0.78 | 0.448 | — | **−33pp (worse)** |
| HaluEval Summ | paired_accuracy | 0.667 | 0.683 | 0.75 | +1.6pp |
| HaluEval Summ | ROC_AUC | 0.69 | 0.647 | — | −4pp |
| RAGTruth QA | ROC_AUC | 0.74 | 0.684 | — | −6pp |
| RAGTruth Summ | ROC_AUC | 0.68 | 0.672 | — | −1pp |

Operational-framing auto-decide accuracy:

| Bench | Auto-decide acc (L7) | Auto-decide acc (diagnose) | Delta |
|---|---:|---:|---:|
| HaluEval QA | 0.650 | **0.484** | −17pp |
| HaluEval Summ | 0.632 | 0.509 | −12pp |
| RAGTruth QA | 0.458 | **0.840** | **+38pp** |
| RAGTruth Summ | 0.647 | 0.731 | +8pp |

**The diagnose fusion weights do NOT restore the website's 0.78 HaluEval
paired number.** They actually collapse HaluEval QA paired_accuracy to
0.417 and push ROC_AUC below random (0.448). The 0.78 website figure was
produced against a different pipeline stack (earlier NLI cascade +
different reranker behaviour); just pinning the weights on top of the
current stack is not sufficient to reproduce it.

The one area where the diagnose weights help is RAGTruth QA auto-decide
accuracy (0.458 → 0.840). That's an information-bearing signal: the
RAGTruth-style synth answers benefit from a calibrated-channel-heavy
fusion, likely because the hosted thresholds were calibrated for
Veracier-style enterprise prose where the calibrated channel is
discriminative. HaluEval-style short factoid answers benefit from the
NLI-heavy fusion, and the two corpora cannot be served by a single
global fusion profile.

## What this means for the v2 decision

Summarising both experiments:

| Hypothesis | Verdict |
|---|---|
| v1 code lane closes the agentic-coding gap | **No.** Code lane is equal-to-worse than RAG lane on the current adversarial corpus due to a bench-design contamination. A body-only adversarial bench is needed before we have a defensible number. |
| Pinning diagnose fusion weights closes the HaluEval drift | **No.** Pinning collapses HaluEval QA paired_accuracy to 0.42 on the current stack. The 0.78 website figure is not reproducible with the current atomic+reranker pipeline under diagnose weights. |
| Per-corpus fusion profiles matter | **Yes.** RAGTruth auto-decide jumped from 0.458 → 0.840 under diagnose weights while HaluEval dropped sharply. No single global fusion serves both. |

Two consequences for product direction:

1. **A use-case / corpus-type classifier that picks the right fusion
   profile (and the right thresholds) per request is an obvious
   architectural next step.** The data in Experiment 2 is the evidence:
   the same pipeline with two fusion settings swings 38pp between
   HaluEval-favoured and RAGTruth-favoured operating points. A
   classifier that picks the fusion based on corpus signal would be a
   cheap, shippable lift for v1.2 — no student training required.

2. **The coding adversarial benchmark needs a v2 redesign** that puts
   only the body in `response_text` and a clean reference solution in
   `raw_context`. Re-running that corrected bench is cheaper than
   training a student and is a prerequisite for a defensible v2 SOTA
   claim on agentic coding.

## What to carry into the v2 student discussion

* The student MUST outperform v1 *per operating point*, not in an
  averaged F1. The evidence above shows the current v1 has two
  orthogonal operating points (NLI-heavy, calibrated-heavy) that serve
  different corpora; a student that only matches one of them is a
  regression.
* Code-aware channels are a v2 requirement, but they are only
  measurable once the coding bench is rebuilt without the
  response==context contamination.
* A use-case classifier in v1.2 could postpone the v2 student entirely
  if it lifts HaluEval QA auto-decide and keeps RAGTruth auto-decide at
  0.84 under the right fusion. Worth a prototype before committing to
  the v2 training run.
