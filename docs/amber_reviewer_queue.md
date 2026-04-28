# Amber is a reviewer queue, not an auto-decision

> **Product contract**: in the Latence TRACE RAG validation surface, the
> `band = amber` output is **a signal to route the answer to a human
> reviewer**, not a signal to retry, rewrite, or auto-approve. Any
> customer-facing pricing, accuracy, or SLA claim about TRACE is stated
> against this contract.

This document is the reference paying users are pointed to from the
Veracier validation report (`proof_report.md`, section 6) and from the SDK
docs whenever the band taxonomy is described.

## The bands

| Band   | Meaning                                                                 | Customer action                                                                 |
| ------ | ----------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| green  | The answer is grounded in the retrieved evidence.                       | Auto-approve to the downstream workflow; log the score for audit.              |
| amber  | The answer mixes grounded claims with at least one unresolved claim.    | Route to the human reviewer queue with the hedge-gate diagnostic attached.     |
| red    | At least one claim is unsupported or contradicts the retrieved context. | Auto-block / hold / reject; surface the failing claim to the author workflow.  |

The precision targets TRACE is fitted and reported against are:

- Green precision on the Veracier labeled set: **≥ 97 %**. A green
  verdict implies the answer is truly grounded in at least 97 out of 100
  cases.
- Red precision on the Veracier labeled set: **≥ 95 %**. A red verdict
  implies the answer is truly not grounded in at least 95 out of 100
  cases.
- Amber agreement on the Veracier pilot (34 ambiguous variants): **≥
  28/34**. Amber agreement on the 200-example rag_prose calibration set:
  **≥ 85 %**.

Note that amber is **not** pitched as a precision number. It is measured
as agreement between the benchmark label and TRACE's output on the
labeled calibration set, because amber is by construction a
"needs-a-human" band.

## Why amber is valuable

Amber is where TRACE creates differentiated value over naive RAG
scorers. Two common enterprise cases produce naturally ambiguous
answers:

1. **Partial coverage**. The evidence answers half of the question
   confidently and half of it only indirectly. Example: "The contract
   defines the 30-day notice period (supported) but does not establish
   whether clause 7.2 applies to pre-signature amendments
   (unresolved)."
2. **Epistemic hedging**. The answer carries explicit uncertainty cues
   ("unclear", "not established", "ne permet pas d'établir", "nicht
   belegt") because the author wants to be honest about what the
   retrieved evidence actually says.

If the scorer collapses these into either green or red, the customer
loses information that a reviewer would have used. TRACE keeps them in
their own band so the reviewer queue is the right size (neither
zero-volume nor dominated by easy green cases).

## Reviewer workflow contract

The contract we make to paying customers about amber-band rows:

1. Each amber output arrives with:
   - the TRACE band and fused score,
   - per-claim NLI entailment/contradiction scores,
   - the evidence snippets TRACE used,
   - the `epistemic_hedge_gate` diagnostic payload (claim indices,
     cap/floor values, hedge cues detected).
2. The reviewer inspects the answer against the flagged evidence
   snippets and one of:
   - confirms the answer (promote to green in the queue),
   - edits the answer (promote to green after the edit is re-scored),
   - rejects the answer (promote to red with a rationale).
3. The target review time per amber row is under 60 seconds on prose
   domains (legal, cyber, HR, quality) and under 120 seconds on
   finance/technical domains that require cross-referencing.

Retrying the same answer against the same evidence is **not** part of
the amber contract. Retry logic belongs to the upstream generator, not
to TRACE.

## Where this contract is enforced

- In the scorer itself, via the `_epistemic_hedge_gate` cap/floor in
  `latence_trace/core/groundedness.py`, which keeps hedged
  `rag_prose` answers inside the amber band instead of letting a single
  supported first sentence lift them into green.
- In the benchmark, via the `refine-ambiguous` stage in
  `scripts/bench_veracier_rag_validation.py`, which live-filters
  ambiguous variants through TRACE before they reach the scoring run
  and marks unreachable ones as `ambiguous_unstable`.
- In the Veracier report, via the dedicated "Amber is a reviewer
  queue, not an auto-decision" section (`proof_report.md`, section 6)
  and the per-archetype amber-agreement breakdown.
- In the threshold fit, via the
  `latence-trace-calibrate-rag-prose` entry point, which tunes
  `green_min` and `amber_min` on the 200-example labeled `rag_prose`
  calibration set to hit the precision targets above.

## What customers should **not** infer

- "Amber means the answer is wrong." It does not; it means the answer
  is partially supported.
- "Amber means TRACE is unsure." TRACE is confident the answer carries
  unresolved claims; that confidence is why the row is routed to a
  reviewer.
- "Amber is noise that should disappear after more training." Amber is
  the product surface for ambiguity. A TRACE deployment with zero
  amber volume on prose domains is a sign of mis-calibrated
  thresholds, not of a better model.

## Per-vertical risk-band semantics (v1)

The meaning of amber is not uniform across verticals.  The table
below is the v1 policy; it is encoded as the default per-tenant
configuration under `config/amber-policy/` and can be overridden
per tenant through the per-tenant threshold endpoint (see B2).

| Vertical | Green semantics | Amber semantics | Red semantics | Recommended amber SLA |
| --- | --- | --- | --- | --- |
| Finance | Release to FP&A / tax workflow | Route to Controller or Head of Tax with evidence | Block and log | 2 business hours |
| Legal | Release to drafting tool / matter | Route to Counsel with hedge-gate rationale | Block; tag matter | 4 business hours |
| HR | Release to HRBP workflow | Route to HRBP or Legal-HR with evidence | Block; open HR-ops ticket | 1 business day |
| Compliance | Release to downstream control | Route to 2LoD reviewer | Block and route to CCO | 1 business hour |
| Engineering | Release to engineer / PLM | Route to principal engineer or quality | Block and flag spec revision | 1 business day |
| Marketing | Release to content workflow | Route to Legal-Marketing | Block and do not publish | 4 business hours |

### How to read the table

- Green is **permission**, never obligation.  The pipeline owner
  can always require a reviewer override for high-value decisions.
- Amber volume on prose verticals is expected.  Under the Veracier
  run, ~29% of variants are amber; that is the product surface for
  ambiguity.
- Red should be rare in production because it maps to a detected
  contradiction; alert on red-rate spikes (e.g. >5% over a rolling
  hour) as a content-quality signal.

The portal amber reviewer queue (see C4) records accept / edit /
reject decisions per row; those decisions are the labeled data that
powers any future v2 student training (`v2_prep_c4_reviewer_labels`).

## Links

- `proof_report.md` section 6 (customer-facing summary, generated per
  Veracier run).
- `latence_trace/data/calibration/README.md` (labeled calibration
  set schema and labeling instructions).
- `scripts/calibrate_rag_prose.py` (fit harness).
- `tests/core/test_epistemic_hedge_gate.py` (decision-matrix tests
  for the scorer-side gate).
- `commercial/verticals/*.md` — per-vertical one-pagers that cite the
  semantics above.
