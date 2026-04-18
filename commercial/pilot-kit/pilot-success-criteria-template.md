# Pilot Success Criteria

> Filled out jointly by Provider and Customer at the start of the
> pilot. Attached as Schedule A to the Pilot Agreement. Re-evaluated
> at day 30 and day 60.

## Customer profile

| Field | Value |
|---|---|
| Customer | <Customer Legal Name> |
| Industry | <industry, e.g. financial services / health / SaaS / public sector> |
| Use case | <e.g. internal RAG over policy docs / agent that emails customers / claims-summarization assistant> |
| Languages | <e.g. en, de> |
| Volume (today) | <queries / day> |
| Volume (target end-of-year) | <queries / day> |
| Sponsor (exec) | <name, title> |
| Day-to-day owner | <name, title> |

## Baseline (before latence-trace)

| Metric | Value | How measured |
|---|---|---|
| % responses flagged for human review (current process) | <%> | <e.g. manual sample of 200 / week> |
| Hallucination rate (estimated) | <%> | <e.g. spot check of 100 / week> |
| Mean time to detect a bad answer | <hours> | <e.g. customer complaints> |
| Cost of one bad answer in production | EUR <X> | <e.g. avg refund + handling> |

## Success criteria

A pilot is **successful** if **all** of the following are met by day 60:

1. **Quality**: groundedness score AUROC >= **0.85** on a 200-item
   labelled sample drawn from Customer's own traffic.
2. **Operations**: P95 end-to-end scoring latency <= **300 ms** on
   the Customer's representative hardware.
3. **Business**: at least one of
   - amber+red review queue reduced by >= **40%** vs baseline, or
   - hallucination rate (post-review) reduced by >= **30%** vs
     baseline, or
   - mean time to detect reduced by >= **70%** vs baseline.
4. **Integration**: end-to-end working integration in Customer's
   pipeline using either the Python SDK or the HTTP API, with logs
   shipped to Customer's centralised log store.
5. **Sponsor sign-off**: written confirmation from the exec sponsor
   that the success criteria have been met.

## Calibration

| Item | Value |
|---|---|
| Profile (`fast` / `balanced` / `quality`) | <e.g. balanced> |
| Calibration sample size | <e.g. 200 labelled examples> |
| Re-calibrated thresholds (green/amber/red) | <e.g. 0.78 / 0.62> |

## Day-30 mid-pilot review

| Metric | Target | Actual at day 30 |
|---|---|---|
| AUROC | 0.85 | <value> |
| P95 latency | 300 ms | <value> |
| Review queue reduction | 40% | <value> |
| Open issues | 0 P1 | <count> |

## Day-60 end-of-pilot review

| Metric | Target | Actual at day 60 | Pass? |
|---|---|---|---|
| AUROC | 0.85 | <value> | yes / no |
| P95 latency | 300 ms | <value> | yes / no |
| Review queue reduction | 40% | <value> | yes / no |
| Hallucination rate reduction | 30% | <value> | yes / no |
| Mean time to detect reduction | 70% | <value> | yes / no |
| Sponsor sign-off | required | yes / no | yes / no |

## Conversion plan

| Step | Owner | Target date |
|---|---|---|
| MSA red-line cycle | Customer Legal | <date> |
| Order Form signed | Provider AE | <date> |
| Production License JWT issued | Provider Ops | <date> |
| Production rollout (canary -> 100%) | Customer Eng | <date> |
| Case study published (if approved) | Provider Marketing | <date> |
