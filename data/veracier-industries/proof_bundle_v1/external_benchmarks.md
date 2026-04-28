# External benchmark report

Generated 2026-04-28T20:54:13Z.  Reproducer: `scripts/bench_external.py`.

| Benchmark | Profile | Rows | Red precision | Red recall | Green precision | Amber leakage | Errors |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| halueval_qa | quality | 120 | 0.435 | 0.345 | 0.513 | 0.483 | 0 |
| halueval_summ | quality | 120 | 0.500 | 0.018 | 0.509 | 0.050 | 0 |
| ragtruth_qa | quality | 120 | 0.000 | 0.000 | 0.875 | 0.167 | 0 |
| ragtruth_summ | quality | 120 | - | 0.000 | 0.731 | 0.008 | 0 |

## Reading the table

- `Red precision` = faithfulness of TRACE's red flags.  The
  hosted SLO is >= 0.95.
- `Red recall` = fraction of truly hallucinated items that
  TRACE correctly routes to red / amber.
- `Green precision` = faithfulness of TRACE's green flags.
  The hosted SLO is >= 0.97.
- `Amber leakage` = fraction of rows TRACE declined to band
  decisively.  Because the external benchmarks only have
  binary labels, amber rows are conservative hedge-gate
  outputs rather than errors; customers route them through
  the reviewer queue.

## What is not covered

- The external benchmarks are English-only and do not cover
  the legal / finance / procurement strata where the Veracier
  proof is strongest.  See `proof_report.md` for the
  headline defence against enterprise-buyer scrutiny.
- RAGTruth "faithful" examples are sometimes partial
  paraphrases of the source; TRACE may band these amber for
  borderline support, which the aggregate correctly counts
  as amber leakage rather than an error.
