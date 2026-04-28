# External benchmark report

Generated 2026-04-28.  Reproducer: `scripts/bench_external.py`.

Every row below comes from running the local `dev_app` (wrapping
`runpod/handler.py`) against the public HaluEval and RAGTruth
benchmarks with a fixed seed.  Raw per-row JSONL is in this
directory's `external_bench/` subfolder.

## 1. Headline numbers

| Benchmark | Profile | Rows | Red precision | Red recall | Green precision | Amber leakage |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| HaluEval QA | standard | 40 | 0.650 | 0.650 | 0.533 | 0.125 |
| HaluEval QA | standard | 80 | 0.714 | 0.526 | 0.571 | 0.125 |
| HaluEval QA | quality | 40 | 0.636 | 0.737 | 0.583 | 0.150 |
| HaluEval Summarisation | standard | 80 | 0.600 | 0.462 | 0.576 | 0.338 |
| HaluEval Summarisation | quality | 40 | 0.909 | 0.625 | 0.667 | 0.275 |
| RAGTruth QA | standard | 60 | 0.375 | 0.923 | 0.944 | 0.167 |
| RAGTruth QA | quality | 60 | 0.394 | 0.929 | 0.933 | 0.200 |

Same row shape as the curated Veracier variants (`variants.curated.jsonl`),
enabling the v2 student pipeline to consume both without a separate
schema adapter.

## 2. Veracier vs external, by design

This table is the honest story.  TRACE v1 was calibrated against
structured, paragraph-length enterprise evidence (the Veracier
corpus).  The external benchmarks stress two different corners:

| Benchmark | Typical evidence | Typical answer | TRACE behaviour |
| --- | --- | --- | --- |
| Veracier | 1-4 short legal / finance / HR paragraphs | 1-3 sentences with specific claims | SOTA: 97.5% green precision, 95% red precision, 85% amber agreement |
| RAGTruth QA | CNN/DM-style article + GPT-4 generated answer | 3-5 sentence paraphrased summary | **High red recall (0.92), high green precision (0.94), low red precision (0.38)** — TRACE correctly catches nearly all hallucinations but over-flags valid-but-paraphrased answers as red |
| HaluEval QA | 1-2 sentence fact | 1-sentence factoid answer | Mid precision on both bands; evidence is too short for MaxSim coverage to dominate and NLI has to carry the whole signal |
| HaluEval Summarisation | CNN/DM article | ~100-word summary | Mid red recall but strong red precision on `quality` (0.91) — the quality profile's NLI aggregation dominates |

## 3. What this means for enterprise RAG customers

* **Enterprise RAG pipelines** (legal, finance, HR, compliance,
  engineering, marketing) look structurally like Veracier, not
  HaluEval.  They retrieve paragraph-length passages and generate
  paragraph-length grounded answers.  This is where TRACE v1 is
  SOTA and where customers should trust the bands as reported.
* **Short-form factoid pipelines** (one-sentence questions, one-
  sentence answers) and **long-form paraphrase pipelines** (news
  summarisation) both sit in the corner where TRACE v1 is
  noticeably weaker, especially on red precision.  Customers in
  those shapes should treat amber as a reviewer queue and avoid
  fail-closed policies on red until v2 ships.

## 4. V2 trigger analysis

The external numbers above, combined with the Veracier numbers in
`proof_report.md`, are the empirical basis for the v2 student plan:

* RAGTruth QA: **red precision 0.38** against a red-recall SLO of
  >= 0.90.  The student model described in
  `/root/.cursor/plans/trace_v1_enterprise_sellability_7b897b50.plan.md`
  (tiny encoder + low-rank biaffine + MaxSim) is specifically
  engineered to close this gap: the biaffine head learns a
  "same topic, wrong fact" penalty that a raw dot-product
  plus NLI aggregate cannot express.
* HaluEval QA: **red recall 0.53, green precision 0.57**.  The
  signal is genuinely confusable on short evidence without the
  learned local-support head.
* Veracier: within-SLO on all three axes.  The v2 student must not
  regress these; they are the contract we already sell.

The `v2_trigger_monitor` task owns the weekly re-run of this
harness and the go/no-go on training the student.  Triggers:

- External red precision (RAGTruth QA) stays < 0.55 for two weeks
  with the current thresholds.
- External red recall (HaluEval QA) stays < 0.70 for two weeks.
- Veracier headline metrics regress more than 2 pp.

Any of those three fires, we greenlight the student training run.

## 5. Reading the table

- `Red precision` = faithfulness of TRACE's red flags.  Hosted SLO
  `>= 0.95` on Veracier.
- `Red recall` = fraction of truly hallucinated items TRACE routes
  to red.  Hosted SLO `>= 0.90` on Veracier.
- `Green precision` = faithfulness of TRACE's green flags.
  Hosted SLO `>= 0.97` on Veracier.
- `Amber leakage` = fraction of rows TRACE declined to band
  decisively (the hedge gate fired or channels disagreed).  These
  rows go to the reviewer queue in production and are **not** counted
  as errors by the SLO.

## 6. What is not covered here

- The external benchmarks are English-only.  Multilingual behaviour
  is only measured on Veracier (FR/DE); see
  `docs/guides/multilingual.md` for the language matrix.
- RAGTruth has fine-grained span-level labels that the current
  harness collapses to a binary `faithful` vs `hallucinated`.  A
  follow-up under `v2_trigger_monitor` will re-run with span-level
  labels so we can report partial-faithfulness precision.
- Amber agreement is not reported because the external benchmarks
  only carry binary labels.  Amber agreement remains a Veracier-only
  metric until we collect labelled ambiguous variants for these
  corpora.
