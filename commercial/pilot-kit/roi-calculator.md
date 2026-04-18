# ROI Calculator

> The model used to size the business case for a latence-trace
> deployment. Use the worksheet below for hand-calculations and
> [`roi-calculator.py`](roi-calculator.py) for repeatable runs.

## Inputs

| Symbol | Name | Typical value | How to source it |
|---|---|---|---|
| `Q` | Annual queries (RAG / agent calls) | 1,000,000 - 50,000,000 | Customer's BI / observability |
| `H` | Baseline hallucination rate | 1% - 8% | Customer's audit sample |
| `C_bad` | Cost of one bad answer reaching the user | EUR 5 - 5,000 | Customer's ops + comp + risk |
| `R` | Reduction in undetected bad answers (Provider commitment) | 60% - 90% | Pilot result; default 70% |
| `F` | False-positive rate at Customer's risk band threshold | 1% - 5% | Pilot result; default 2% |
| `C_review` | Cost of one human review of an amber/red flagged answer | EUR 0.50 - 5 | Customer's ops |
| `S` | latence-trace subscription fee (annual) | EUR 60k - 500k | Provider Order Form |
| `O` | Customer integration + ops cost (one-time + annual) | EUR 20k - 80k | Customer estimate |

## Formulas

```
B (avoided losses)        = Q * H * R * C_bad
F_cost (false positives)  = Q * F * C_review
Net annual benefit        = B - F_cost - O
ROI %                     = (Net annual benefit - S) / S * 100
Payback (months)          = S / ((B - F_cost - O) / 12)
```

## Worked example -- mid-market RAG over support docs

| Input | Value |
|---|---|
| Q | 12,000,000 |
| H | 4% |
| C_bad | EUR 25 (refund + handling) |
| R | 70% |
| F | 2% |
| C_review | EUR 1.50 |
| S | EUR 120,000 |
| O | EUR 40,000 |

```
B          = 12,000,000 * 0.04 * 0.70 * 25  =  EUR  8,400,000
F_cost     = 12,000,000 * 0.02 * 1.50       =  EUR    360,000
Net annual = 8,400,000 - 360,000 - 40,000   =  EUR  8,000,000
ROI %      = (8,000,000 - 120,000) / 120,000 * 100  ~  6,567%
Payback    = 120,000 / (8,000,000 / 12)     ~  0.18 months  (~5.4 days)
```

## Worked example -- regulated workflow (claims / KYC / clinical)

| Input | Value |
|---|---|
| Q | 800,000 |
| H | 2% |
| C_bad | EUR 1,500 (rework + regulator exposure) |
| R | 70% |
| F | 3% |
| C_review | EUR 4 |
| S | EUR 250,000 |
| O | EUR 80,000 |

```
B          = 800,000 * 0.02 * 0.70 * 1,500  =  EUR 16,800,000
F_cost     = 800,000 * 0.03 * 4             =  EUR     96,000
Net annual = 16,800,000 - 96,000 - 80,000   =  EUR 16,624,000
ROI %      = (16,624,000 - 250,000) / 250,000 * 100  ~  6,549%
Payback    = 250,000 / (16,624,000 / 12)    ~  0.18 months
```

## Sanity checks

- If `R * H * C_bad < S / Q` then the deployment is **not** ROI-
  positive at this volume. Consider the *team* tier, a smaller
  deployment, or focusing on the highest-stakes subset of traffic.
- `F_cost / B` should be **< 10%**; if it is higher, recalibrate
  thresholds (see `docs/operations/calibration-runbook.md`).
- `Payback` > 12 months in a Fortune-500-scale deployment is a
  yellow flag for the deal -- revisit `H` (often understated) or
  `C_bad` (often understated).

## Things to NOT count

- Avoided regulator fines (too speculative; mention as upside, do
  not include in net benefit).
- Brand-equity gains (mention as upside).
- Productivity gains from faster review (mention as upside).

## Reporting back

Use the day-30 and day-60 templates in
`pilot-success-criteria-template.md` to redo this calculation with
real numbers. The case-study template will pick those up
automatically.
