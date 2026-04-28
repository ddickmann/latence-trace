# TRACE for Finance

**Who this is for:** CFOs, Controllers, FP&A, Head of Tax, Head of Audit
interested in deploying a RAG or agent pipeline over financial documents
(transfer pricing, CbCR, statutory filings, management accounts, audit
evidence) without giving up evidence traceability.

## What TRACE does for finance RAG

TRACE scores every generated answer against the evidence actually
retrieved for it and returns `green` / `amber` / `red`:

- **Green:** auto-release to the finance user / downstream workflow.
- **Amber:** route to a human reviewer queue with evidence links.
- **Red:** block the answer and capture it for audit.

Green precision on the Veracier benchmark finance archetype is **100%
(8/8)** and red precision is **100% (8/8)**.  See
`data/veracier-industries/proof_bundle_v1/proof_report.md` §3.

## Anchor use cases (curated, reproducible)

| Use case | Question | Why it matters |
| --- | --- | --- |
| `FIN-01` | Transfer pricing documentation over five years for an FR tax audit | Loss-materiality risk if the wrong LF/Master/CbCR pairing is released |
| `FIN-04` | Impairment testing assumptions vs. CAPEX committee approvals | Audit-committee exposure on asset carrying values |
| `MAROC-02` | Moroccan cost-plus margin benchmarking vs DGFIP first-quartile | Direct contention with the tax administration |
| `UK-02` | UK HMRC provision tax-deductibility review | Litigation-provision reporting obligation |

All four use cases ship with evidence packs under
`proof_bundle_v1/evidence_packs/` and three controlled response variants
each (`perfect`, `ambiguous`, `wrong`).

## Latency contract

Isolated (single request) p50 / p95 per profile — see
`proof_report.md` §4 and `docs/operations/observability.md`.

## Integration shapes

- Python SDK (`latence-python`)
- Hosted `api.latence.ai` with tenant API key
- LangChain / LlamaIndex / LangGraph / CrewAI adapters (`clients/python/`)
- n8n community node + MCP endpoint for agent frameworks
- TypeScript SDK (see `clients/typescript/`)

## What TRACE does NOT do in finance

- Does not do retrieval.  Bring your retriever.
- Does not compute IFRS adjustments.  It does not rewrite numbers; it
  scores whether an answer is grounded in the evidence supplied.
- Does not make the final release decision.  Green is a permission, not
  an obligation; your pipeline owner always retains veto.

## Proof and evidence

- `data/veracier-industries/proof_bundle_v1/proof_report.md`
- `data/veracier-industries/proof_bundle_v1/failure_appendix.md`
- External benchmarks: `external_benchmarks.md` (HaluEval QA,
  RAGTruth QA, summarisation).
- Security attestation pack (SOC 2 Type I, ISO 27001 readiness, pentest
  summary).

## Latency contract

- **Isolated single-request**: p95 166 ms (standard), 178 ms (quality).  This is the number to quote for "how fast is one call?".
- **Sustained (concurrency = 8)**: p95 509 ms (standard), 862 ms (quality).  This is the production-pipeline number.
- Hosted SLA ceiling: 900 ms p95 standard, 2 200 ms p95 quality; see `commercial/hosted-sla.md`.
- Reproducer: `scripts/bench_latency.py --lane isolated --profile standard`.
- Raw numbers: `data/veracier-industries/proof_bundle_v1/latency_bench/` and `docs/operations/observability.md`.
