# TRACE for Engineering

**Who this is for:** CTOs, Heads of Engineering, Principal Engineers,
Quality Engineering leads running a RAG or agent pipeline over
engineering artefacts: specifications, build dossiers, test
protocols, NDT reports, CAD traceability.

## Why TRACE matters for engineering RAG

Engineering answers refer to specific revisions, tolerance bands,
batch numbers, and test-protocol identifiers.  TRACE's grounding
signal catches the most common engineering-RAG failure mode:
"semantically close but operationally wrong" — citing the right part
number against the wrong revision.

Engineering archetypes in the Veracier benchmark:

| Archetype | Rows | Green precision | Red precision | Amber agreement |
| --- | ---: | ---: | ---: | ---: |
| Technology architecture, migration, and dependencies | 18 | 100% (6) | 100% (6) | 100% (4) |
| Quality, batch record, audit, and certification | 48 | 100% (16) | 94.1% (17) | 93.8% (16) |
| Operations, production, maintenance, capacity, and planning | 6 | 100% (2) | 100% (2) | 100% (2) |

See `data/veracier-industries/proof_bundle_v1/proof_report.md`.

## Anchor use cases

| Use case | Question |
| --- | --- |
| `CTO-01` | Specification revision used for AV-3000 serial 20-0847 in 2020 |
| `AERO-01` | Batch record completeness for an aerospace EASA Form 1 release |
| `OPS-01` | Capacity allocation consistency between MRP and shop-floor records |

## Integration shapes

Python and TypeScript SDKs, LangChain / LangGraph / CrewAI / Haystack
adapters, n8n node, remote MCP endpoint.

## What TRACE does NOT do

- Does not parse CAD geometry.
- Does not replace a PLM change-control workflow.
- Does not read scanned PDFs without OCR pre-processing.

## Latency contract

- **Isolated single-request**: p95 166 ms (standard), 178 ms (quality).  This is the number to quote for "how fast is one call?".
- **Sustained (concurrency = 8)**: p95 509 ms (standard), 862 ms (quality).  This is the production-pipeline number.
- Hosted SLA ceiling: 900 ms p95 standard, 2 200 ms p95 quality; see `commercial/hosted-sla.md`.
- Reproducer: `scripts/bench_latency.py --lane isolated --profile standard`.
- Raw numbers: `data/veracier-industries/proof_bundle_v1/latency_bench/` and `docs/operations/observability.md`.
