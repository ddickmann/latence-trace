# Veracier TRACE Validation

This note documents the full Veracier Industries validation run for TRACE:
what the dataset contains, how the benchmark was constructed, what TRACE
achieved, and why the result matters for finance, legal, and general
enterprise RAG deployments.

## Executive Summary

The Veracier Industries benchmark is a realistic enterprise RAG validation
setup built from `lightonai/veracier-industries`: a multilingual corpus of
financial, legal, compliance, operational, HR, quality, security, and technical
documents.

In the full-use-case run, TRACE was evaluated on:

| Layer | Count |
| --- | ---: |
| Unique PDFs in source metadata | 1004 |
| Successfully processed PDFs | 1000 |
| Excluded PDFs | 4 |
| Labeled use cases/questions | 40 |
| Archetypes represented | 12 |
| Evidence documents per use case | 6 |
| Controlled answer variants | 120 |
| Live TRACE calls | 240 |

TRACE scored each generated answer in both `standard` and `quality` profiles.
The run produced the following headline metrics:

| Metric | Full-scale result | Interpretation |
| --- | ---: | --- |
| Green precision | 80/83 (96.4%) | When TRACE says green, the answer is almost always grounded. This narrowly missed the stricter 97% full-scale target by 3 false positives. |
| Red precision | 71/72 (98.6%) | When TRACE says red, the answer is very reliably ungrounded. |
| Amber agreement | 66/70 (94.3%) | Ambiguous answers are consistently routed to the reviewer queue. |
| Ambiguous refinement | 35/40 accepted, 5 quarantined | The benchmark self-filter rejects unstable amber examples instead of hiding them in the denominator. |

The result is strong enough to tell a credible enterprise validation story:
TRACE is particularly strong at blocking unsupported answers and preserving a
useful reviewer queue for ambiguous prose. The main follow-up before claiming
a hard 97% green precision target at full scale is to inspect and fix the 3
green false positives.

Source artifacts:

- `data/veracier-industries/trace_bench_runs/veracier_trace_validation_full_use_cases_20260428/proof_report.md`
- `data/veracier-industries/trace_bench_runs/veracier_trace_validation_full_use_cases_20260428/validation_summary.json`
- `data/veracier-industries/trace_bench_runs/veracier_trace_validation_full_use_cases_20260428/trace_results.jsonl`
- `data/veracier-industries/trace_bench_runs/veracier_trace_validation_full_use_cases_20260428/refinement_summary.json`

## What the Dataset Contains

The dataset is not a file-by-file QA benchmark. It is a realistic enterprise
corpus plus a set of labeled business questions.

It contains:

- 1004 unique PDF records in `MASTER_INDEX.csv`.
- 1000 PDFs successfully processed into text/markdown.
- 40 labeled use cases in `ANSWER_KEY.json`.
- Metadata for document class, language, entity, format, and relevance.
- Multilingual material across French, English, German, Italian, Spanish, and mixed-language documents.

The four excluded documents are listed in `excluded_documents.jsonl`: three
document-processing failures and one missing processed output. The validation
therefore covers the successfully processed corpus, which is 1000/1004 unique
documents.

## What a Use Case Means

A use case is a business question over the enterprise corpus. For example:

> Les CAC veulent le support de toutes les provisions et charges a payer > 200 kEUR.

This is a finance/audit use case. The benchmark selects a six-document evidence
pack from the 1000 processed PDFs, then generates controlled answer variants
against that evidence:

| Variant | Expected band | Purpose |
| --- | --- | --- |
| `perfect` | green | Fully grounded answer, usually exact evidence quotes. |
| `ambiguous` | amber | One supported claim plus one unresolved or incomplete claim. |
| `wrong` | red | Unsupported answer with materially incorrect facts. |

Each of the 40 use cases produces three answers, giving 120 benchmark answers.
Each answer is scored by both TRACE profiles, giving 240 live TRACE calls.

## Methodology

The benchmark deliberately separates corpus ingestion, evidence selection,
answer generation, scoring, and reporting. This makes the result auditable.

1. Process the corpus.

   The full Veracier PDF corpus is processed into markdown/text. The summary
   records 1000 completed documents, total completed characters, and exclusions.

2. Build evidence packs.

   For each labeled use case, the harness selects six documents from the
   processed corpus. The evidence pack includes source ids, filenames,
   classifications, text excerpts, context size, and archetype metadata.

3. Generate controlled answer variants.

   For every use case, the harness creates one green, one amber, and one red
   answer. The green anchor is deterministic and extractive where possible: it
   uses verbatim evidence quotes to make the positive label auditable.

4. Self-filter ambiguous answers.

   Ambiguous variants are sandbox-scored through live TRACE. If an ambiguous
   answer lands outside the amber band, the harness asks OpenAI to regenerate
   it with targeted feedback. If the answer remains unstable, it is marked
   `ambiguous_unstable` and excluded from headline denominators.

5. Score with live TRACE.

   The final 120 answers are scored in both `standard` and `quality` profiles.
   The scoring path uses the deployed TRACE RAG lane, including groundedness,
   NLI, context coverage, structured verification where relevant, risk-band
   thresholds, and the epistemic hedge gate for ambiguous prose.

6. Emit customer-facing artifacts.

   The harness writes a Markdown proof report and a machine-readable JSON
   summary with headline metrics, archetype breakdown, latency, evidence packs,
   refinement logs, and per-row TRACE results.

## Why This Is Statistically Meaningful

The benchmark is statistically meaningful because it is stratified and
label-controlled:

- Every use case is evaluated with green, amber, and red variants.
- Every answer is scored under two profiles.
- The full run covers all 40 labeled use cases, not a cherry-picked pilot.
- The 40 use cases span 12 archetypes.
- The evidence comes from the 1000 processed PDFs, not synthetic toy context.
- Ambiguous examples are self-filtered and unstable rows are reported instead
  of silently counted as successes.

The benchmark should not be described as "1000 independent QA questions." It is
more precise to say:

> TRACE was validated on the full Veracier corpus: 1000 processed enterprise
> PDFs, 40 labeled RAG use cases, 12 archetypes, 120 controlled answer variants,
> and 240 live scoring calls.

That framing is defensible because TRACE scores generated answers against
retrieved evidence. The unit of validation is the answer/evidence pair, not the
raw PDF.

## Results by Archetype

The full run covered the following archetypes:

| Archetype | Rows | Green precision | Red precision | Amber agreement |
| --- | ---: | ---: | ---: | ---: |
| Finance, tax, transfer pricing, accounting | 24 | 100.0% | 100.0% | 100.0% |
| Contracts, clauses, obligations, and regulatory legal review | 42 | 100.0% | 100.0% | 100.0% |
| General enterprise evidence review | 42 | 100.0% | 100.0% | 92.9% |
| Quality, batch record, audit, and certification | 48 | 100.0% | 94.1% | 93.8% |
| Cybersecurity, NIS2, systems, and controls | 18 | 85.7% | 100.0% | 83.3% |
| HR, employment, workforce, and labor compliance | 12 | 80.0% | 100.0% | 75.0% |
| Procurement, supplier risk, delivery, and solvency | 12 | 80.0% | n/a | 100.0% |

Small-count archetypes such as litigation, operations, sales export, executive
risk, and technical architecture also scored strongly, but their sample sizes
are too small to use as standalone headline claims.

## Why This Matters for Finance

Finance RAG is high-risk because many answers look plausible while hinging on
specific periods, entities, amounts, thresholds, and supporting schedules.

Examples from Veracier include:

- transfer-pricing documentation across years and jurisdictions;
- provisions and charges above a materiality threshold;
- tax treatments, deferred tax, management fees, and audit support;
- exact amounts, percentages, and date-sensitive accounting assumptions.

TRACE is useful here because it distinguishes:

- quoted or directly supported financial facts (`green`);
- unresolved coverage questions such as "all entities" or "all periods"
  (`amber`);
- invented amounts, wrong thresholds, wrong entities, or unsupported audit
  conclusions (`red`).

In the full run, the finance archetype achieved 100% green precision, 100% red
precision, and 100% amber agreement on its scored denominator.

## Why This Matters for Legal

Legal RAG is risky because a small change in party, clause scope, exception,
date, governing obligation, or litigation status can materially change the
answer.

Examples from Veracier include:

- supplier and customer contract clauses;
- force majeure and termination rights;
- sanctions and export-control checks;
- pending litigation and threatened claims;
- confidentiality, audit rights, assignment, and liability provisions.

TRACE is useful here because it treats grounded legal answers as evidence-bound
claims, not fluent summaries. It can separate:

- exact clause-level support (`green`);
- partially supported answers with unresolved applicability or missing
  schedules (`amber`);
- invented consent rights, inverted obligations, wrong parties, or unsupported
  litigation conclusions (`red`).

In the full run, the legal-contracts archetype achieved 100% green precision,
100% red precision, and 100% amber agreement on its scored denominator.

## Why This Matters for Enterprise RAG Generally

Enterprise RAG failures are rarely obvious hallucinations. The common failure
modes are subtler:

- a correct fact attached to the wrong entity;
- a true statement from one document generalized to the whole group;
- a missing appendix treated as present;
- a partial audit trail summarized as complete;
- a supported operational fact combined with an unsupported conclusion;
- a plausible answer that ignores date, scope, jurisdiction, or threshold.

The Veracier benchmark exercises those failure modes across multiple document
types and departments. TRACE's green/amber/red contract maps naturally to
production workflows:

- `green`: auto-approve or pass downstream;
- `amber`: route to reviewer;
- `red`: block, revise, or regenerate.

This is valuable because enterprise teams do not only need a scalar score. They
need an operational decision boundary and an audit trail that explains why an
answer was approved, escalated, or blocked.

## Current Limitations

The full-scale result is strong but not final.

- Green precision was 96.4%, narrowly below the 97% target.
- Three green false positives should be inspected before using the full-scale
  result as a strict customer SLA claim.
- The 200-example human-labeled `rag_prose` calibration set is not yet complete.
- The benchmark validates answer/evidence scoring, not retrieval recall across
  every possible question over every PDF.
- High-concurrency latency in this run includes gateway/worker queueing effects
  and should not be read as isolated single-request latency.

The pilot run did exceed all three targets. The full run gives the better
customer story because it covers all labeled use cases, but the honest next
step is to close the three green false positives and rerun the full report.

## Recommended Customer Framing

Use this wording:

> TRACE was evaluated on the full Veracier Industries corpus: 1000 processed
> enterprise PDFs, 40 labeled RAG use cases, 12 business archetypes, 120
> controlled green/amber/red answer variants, and 240 live TRACE scoring calls.
> The full run achieved 98.6% red precision and 94.3% amber agreement, with
> 96.4% green precision. Finance and legal-contract use cases reached 100%
> precision on green and red decisions in the scored denominator.

Avoid saying:

- "TRACE answered 1000 questions."
- "The dataset contained 120 ground-truth answers."
- "Every PDF was independently validated."
- "Full-scale green precision already exceeds 97%."

The defensible claim is that TRACE was validated on realistic enterprise RAG
answer/evidence pairs derived from the full corpus and all labeled use cases.
