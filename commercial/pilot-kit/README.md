# Pilot Kit

> Everything a Provider AE / SE needs to land a 60-day paid pilot
> with an enterprise design partner, plus the artefacts used to
> close out the pilot into a renewable subscription.

## Contents

| File | Purpose |
|---|---|
| [`pilot-agreement-template.md`](pilot-agreement-template.md) | Short-form letter agreement for the pilot (separate from the MSA). |
| [`pilot-success-criteria-template.md`](pilot-success-criteria-template.md) | The success criteria the pilot is graded against. |
| [`roi-calculator.md`](roi-calculator.md) | The ROI worksheet and worked example. |
| [`roi-calculator.py`](roi-calculator.py) | Pure-Python CLI calculator -- no dependencies. |
| [`case-study-template.md`](case-study-template.md) | The case-study skeleton produced after a successful pilot. |
| [`pilot-tracker.md`](pilot-tracker.md) | The pilot pipeline tracker (markdown table format). |
| [`onboarding-checklist.md`](onboarding-checklist.md) | Day-1 to day-60 onboarding cadence. |

## Recommended workflow

1. **Qualify** -- prospect must have (a) a live RAG or agent
   workflow in production, (b) a measurable hallucination /
   compliance pain (false positives logged, escalations counted, or
   regulator-driven), (c) >= 1 named exec sponsor.
2. **Sign the pilot agreement.** Use `pilot-agreement-template.md`.
3. **Lock the success criteria** with the customer using
   `pilot-success-criteria-template.md` and add a row to the tracker.
4. **Onboard** following `onboarding-checklist.md`. Calibrate
   thresholds per `docs/operations/calibration-runbook.md`.
5. **Mid-pilot review at day 30** -- redo ROI calculation with real
   numbers, share with sponsor.
6. **End-of-pilot review at day 60** -- write up using
   `case-study-template.md`. Convert to MSA + Order Form.
