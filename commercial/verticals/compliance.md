# TRACE for Compliance

**Who this is for:** Chief Compliance Officers, CISOs, AML / sanctions
teams, regulatory reporting teams running a RAG or agent pipeline over
compliance evidence (sanctions screening, NIS2 controls, GDPR impact
assessments, classified-system registers, CSRD evidence).

## Why TRACE matters for compliance RAG

Compliance answers expose the firm to regulator-visible consequences.
Green is not an invitation to skip evidence review; it is permission to
short-circuit routine human review.  Amber routes the answer to a
reviewer queue with the evidence attached.

Compliance-adjacent archetypes in the Veracier benchmark:

| Archetype | Rows | Green precision | Red precision | Amber agreement |
| --- | ---: | ---: | ---: | ---: |
| Cybersecurity, NIS2, systems, and controls | 18 | 85.7% (7) | 100% (6) | 83.3% (6) |
| Executive contract, sanctions, and compliance risk map | 6 | 100% (2) | 100% (2) | 100% (2) |
| Sales, export control, sanctions, customer clearance | 6 | 100% (2) | 100% (1) | 100% (2) |

See `data/veracier-industries/proof_bundle_v1/proof_report.md`.  The
`CISO-01:perfect` and `CISO-01:ambiguous` rows were both curated in v1
(see `curation_log.md`).

## Anchor use cases

| Use case | Question |
| --- | --- |
| `CEO-01` | Inherited-contract risk map after M&A (sanctions + Sapin II) |
| `CISO-01` | Which systems process Confidential Defense / NATO information |
| `DEF-01` | Defense-activity divestiture and DGAM clearance chain |
| `SALES-01` | Export-control clearance for sanctioned counterparties |

## Integration shapes

Python SDK, TypeScript SDK, LangChain / LangGraph / CrewAI / Haystack,
n8n community node, remote MCP endpoint.

## What TRACE does NOT do

- Does not substitute for sanctions-screening systems; pair with
  OFAC/EU-consolidated lists.
- Does not provide legal opinions on export-license eligibility.
- Does not retain customer data past the score response (see
  `commercial/hosted-data-processing-addendum.md`).

## Latency contract

- **Isolated single-request**: p95 166 ms (standard), 178 ms (quality).  This is the number to quote for "how fast is one call?".
- **Sustained (concurrency = 8)**: p95 509 ms (standard), 862 ms (quality).  This is the production-pipeline number.
- Hosted SLA ceiling: 900 ms p95 standard, 2 200 ms p95 quality; see `commercial/hosted-sla.md`.
- Reproducer: `scripts/bench_latency.py --lane isolated --profile standard`.
- Raw numbers: `data/veracier-industries/proof_bundle_v1/latency_bench/` and `docs/operations/observability.md`.
