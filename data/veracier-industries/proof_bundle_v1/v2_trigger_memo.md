# V2 Trigger Decision Memo

**Run date:** 2026-04-28 20:30 UTC (refreshed from production-config reconciliation).

## Triggers evaluated

| trigger | threshold | measured | status |
|---|---|---|---|
| HaluEval QA paired_accuracy (English NLI, quality) | >= 0.80 | 0.717 < 0.80 | **FIRED** |
| HaluEval Summ paired_accuracy (English NLI, quality) | >= 0.75 | 0.667 < 0.75 | **FIRED** |
| RAGTruth QA F1@median (quality, reference framing) | >= 0.70 | 0.691 < 0.70 | borderline |
| RAGTruth QA precision@median (quality) | >= 0.95 | 0.933 < 0.95 | borderline |
| RAGTruth Summ F1@median (quality) | >= 0.65 | 0.676 > 0.65 | ok |
| Coding paired_accuracy per variant (quality) | >= 0.80 | 0.27 / 0.62 / 0.21 all < 0.80 | **FIRED** |
| Auto-decide accuracy per bench | varies | all fail ship gate | **FIRED** |
| Veracier worst-vertical green precision | >= 0.95 | 1.0 this week, 0.964 overall | ok |

## Decision: GO (conditional on user confirmation)

**Rationale**: five triggers fired — two on HaluEval paired accuracy,
three on coding adversarials, plus all four auto-decide ship gates.

The student is **not automatically kicked off**. Per the plan's Phase
3.5 hard stop, the evidence report at
[`data/veracier-industries/proof_bundle_v2/EVIDENCE_REPORT.md`](../proof_bundle_v2/EVIDENCE_REPORT.md)
is the decision document and user confirmation is required on corpus
size, source mix, and public-SOTA target before Phase 4 (training) may
begin.

## What v1.1 still ships defensibly today

- Veracier enterprise RAG headline: red_p 0.986, green_p 0.964, amber 0.957.
- RAGTruth reconciled within 5pp of website on F1@median / precision@median.
- 15/15 adapter integration proof.
- Auto-decide middleware shipped as opt-in (design-partner only).

## What needs v2 to close

- Agentic coding hallucination detection (current paired_acc 0.21-0.62).
- HaluEval paired accuracy back to the diagnose reference (0.75-0.78 vs
  current 0.67-0.72).
- Zero-human-in-the-loop auto-decide accuracy (current 0.46-0.65).

## Next review

After the Phase 3.5 evidence checkpoint is reviewed by the user and
either (a) a GO-v2 corpus spec is confirmed or (b) v1.1 is shipped as-is
and v2 is deferred.

---
Superseded auto-generated entry below kept for reference:

> 2026-04-28 18:56 UTC: halueval_qa_red_prec FIRED, ragtruth_qa_red_prec
> FIRED, veracier_worst ok. Decision: GO train v2 student.

The 18:56 entry used the pre-reconciliation (dropped-question) bench run
and is superseded by this refresh. Both entries arrive at the same
GO-subject-to-confirmation decision; the reasoning here is based on the
correct, reconciled numbers.
