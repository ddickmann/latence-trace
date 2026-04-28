# What TRACE gets wrong - and what to pair it with

This is the deliberate public failure appendix for TRACE v1.  Everything
here is observed in the Veracier v1 proof bundle or confirmed by internal
stress tests.  We publish it so prospects can make informed decisions
rather than discovering the limits in production.

## 1. Red recall on procurement-archetype wrong answers

Observed on the uncurated Veracier run: two `PROC-01:wrong` and two
`PROC-02:wrong` rows scored amber instead of red under the `quality`
profile, and one `PROC-02:wrong` row scored green.  Root cause: wrong
variants leaned on plausible-sounding "all suppliers certified /
automatic compensation mechanism" phrasing that overlapped the surface
form of the evidence more than it deviated in fact.

Mitigation in v1:

- Curate procurement `wrong` variants for numeric / identifier deltas,
  not tonal exaggeration (implemented in
  `scripts/curate_veracier_variants.py`).
- Route procurement answers with coverage ratio <0.55 to amber reviewer
  queue regardless of band (per-vertical override; see
  `docs/amber_reviewer_queue.md`).
- Pair TRACE with a lightweight supplier-registry lookup before
  releasing green for any procurement claim that cites a counterparty.

Target: close red recall from 89.9% to >=95% on the curated rerun.

## 2. Epistemic hedging in English still under-indexed

`EPISTEMIC_HEDGE_CUES` is richer in French and German than in English.
On English-only inputs the hedge gate is less likely to fire, so
ambiguous English answers sometimes promote to green on the `quality`
profile.

Mitigation in v1:

- Hedge-cue dictionary is being extended alongside the FR/ES/IT work
  (`c1_multilingual_fr_es_it`).
- Customers can extend hedge cues per tenant via the
  `/v1/thresholds/{tenant}/hedge-cues` endpoint (rolling out with B2).

## 3. Latency posture

P95 on the quality profile was 83 s on the uncurated run with 32-way
concurrency against a single RunPod worker.  This is **not** a live
latency figure; it is throughput under contention.

- Isolated p50/p95/p99 per profile land under `latency_contract.md` in
  the proof bundle after `f1_isolated_latency` runs.
- The live SLA target is the isolated measurement, not the sustained
  32-concurrency measurement.  See `commercial/hosted-sla.md`.

## 4. Ambiguity in the "classified" CISO archetype

The `CLASSIFIED_SYSTEM` documents in the Veracier corpus are
deliberately noisy (mixed fragments).  TRACE green precision on the
`cyber_security` archetype is 85.7% (6/7), below the 97% headline.  This
is an evidence-pack problem, not a TRACE scoring problem: the curated
`CISO-01:perfect` anchor was rewritten to cite the clean habilitation
registry instead.

## 5. Things TRACE is NOT today

- **TRACE does not generate answers.**  It scores answers you already
  have.  Pair with your LLM or RAG pipeline; don't try to use TRACE
  alone as a reasoning engine.
- **TRACE does not retrieve.**  Evidence must be provided to the score
  call.  Pair with a retriever you trust.
- **TRACE is not a policy engine.**  It returns `band` +
  `groundedness_score`.  Your pipeline decides what to do with amber
  (we strongly recommend a human reviewer queue; see
  `docs/amber_reviewer_queue.md`).
- **TRACE does not guarantee zero false positives.**  It reduces them.
  See `commercial/bold-claims-v1.md` for the exact claim set and the
  evidence each claim rests on.
