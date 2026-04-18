# Pilot Tracker

> Single source of truth for the pilot pipeline. Update at every
> stage transition. Numbers feed the quarterly business review.

## Stages

`prospect` -> `qualified` -> `agreement-sent` -> `signed` ->
`onboarding` -> `mid-pilot` -> `end-of-pilot` -> `won` / `lost` /
`paused`.

## Pipeline

| Customer | Industry | Region | AE | SE | Stage | Pilot Fee | ARR potential | Day-30 review | Day-60 review | Notes |
|---|---|---|---|---|---|---|---|---|---|---|
| <Customer A> | financial services | EU | <ae> | <se> | signed | EUR 25k | EUR 250k | <YYYY-MM-DD> | <YYYY-MM-DD> | KYC summarisation, en+de |
| <Customer B> | health insurance | EU | <ae> | <se> | onboarding | EUR 25k | EUR 180k | <YYYY-MM-DD> | <YYYY-MM-DD> | Claims-doc QA |
| <Customer C> | SaaS | US | <ae> | <se> | mid-pilot | EUR 0 (waived for ref) | EUR 120k | <YYYY-MM-DD> | <YYYY-MM-DD> | Support copilot |

(Replace the rows with real entries. Keep the table sorted by
expected close date.)

## Win-rate

| Quarter | Pilots started | Pilots won | Win % | ARR closed |
|---|---|---|---|---|
| <YYYY-Qx> | <n> | <n> | <%> | EUR <amount> |

## Failure modes (track these too)

| Failure mode | Count this quarter | Next action |
|---|---|---|
| Pilot signed but never deployed | <n> | Follow-up after 14 days; otherwise close `lost-no-deploy` |
| AUROC under target on Customer's data | <n> | Add to calibration backlog |
| P95 latency over target | <n> | Profile, escalate to Engineering |
| Procurement timeout (>90d) | <n> | Escalate to AE manager |

## Lessons learned (rolling)

- <YYYY-MM-DD> -- <Customer> -- <one-line lesson, e.g. "calibration
  with 200 examples is too few for high-cost regulated workflows --
  default to 400 in this segment">
