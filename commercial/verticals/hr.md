# TRACE for HR

**Who this is for:** Chief People Officers, HR Directors, HR
Operations running a RAG pipeline over HR policies, employment
contracts, labor-compliance filings, and due-diligence data rooms.

## Why TRACE matters for HR RAG

HR answers are sensitive along multiple axes simultaneously: personal
data, collective-bargaining obligations, non-compete coverage, pay
equity, and M&A due diligence.  TRACE scores every generated answer
against the retrieved HR evidence and routes uncertain answers to a
reviewer queue.

HR archetype in the Veracier benchmark (uncurated baseline):

- Green precision **80% (5)** — one `HR-01:ambiguous` row promoted to
  green under the quality profile; the curation in
  `variants.curated.jsonl` rewrote the hedge cues so the amber
  classification is defensible (target: >=97% on the curated rerun).
- Red precision **100% (3)**.
- Amber agreement **75% (4)** — upgraded via hedge-gate improvements
  shipped in `latence_trace/core/groundedness.py::_epistemic_hedge_gate`.

Full numbers: `data/veracier-industries/proof_bundle_v1/proof_report.md`.

## Anchor use cases

| Use case | Question |
| --- | --- |
| `HR-01` | Non-compete clause inventory for M&A due diligence |
| `HR-02` | Gender-pay-equity index publication obligation |

## Integration shapes

Same set as the other verticals (Python SDK, TypeScript SDK, LangChain,
LangGraph, CrewAI, Haystack, n8n, MCP).

## What TRACE does NOT do

- Does not render legal advice on employment questions.
- Does not automate HR decisions.
- Does not process special-category personal data without a signed DPA.

## Latency contract

- **Isolated single-request**: p95 166 ms (standard), 178 ms (quality).  This is the number to quote for "how fast is one call?".
- **Sustained (concurrency = 8)**: p95 509 ms (standard), 862 ms (quality).  This is the production-pipeline number.
- Hosted SLA ceiling: 900 ms p95 standard, 2 200 ms p95 quality; see `commercial/hosted-sla.md`.
- Reproducer: `scripts/bench_latency.py --lane isolated --profile standard`.
- Raw numbers: `data/veracier-industries/proof_bundle_v1/latency_bench/` and `docs/operations/observability.md`.
