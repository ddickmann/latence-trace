# TRACE for Legal

**Who this is for:** General Counsel, Legal Operations, Compliance
Counsel, Privacy Officers running a RAG pipeline over contracts,
litigation files, privacy-impact assessments, or regulator
correspondence.

## Why TRACE matters for legal RAG

Legal retrieval-augmented answers are high-consequence: misstating a
non-compete clause, an indemnity cap, or a litigation provision can
turn into liability.  TRACE scores every answer against the evidence
actually retrieved and routes ambiguous-prose answers to a human
reviewer queue instead of auto-releasing them.

Legal archetypes in the Veracier benchmark:

| Archetype | Rows | Green precision | Red precision | Amber agreement |
| --- | ---: | ---: | ---: | ---: |
| Contracts, clauses, obligations, regulatory legal review | 42 | 100% (14) | 100% (13) | 100% (12) |
| Litigation and legal proceedings | 6 | 100% (2) | 100% (2) | 100% (2) |

See `data/veracier-industries/proof_bundle_v1/proof_report.md` §3.

## Anchor use cases

| Use case | Question | Why it matters |
| --- | --- | --- |
| `LEGAL-01` | Litigation provision in the German subsidiary management accounts | Statutory disclosure exposure |
| `CEO-02` | Contract novation obligations after a change-of-control event | Assignment-clause trigger |
| `COMP-01` | Sapin II anti-corruption due diligence on commercial agents | Regulatory enforcement risk |
| `US-01` | Cross-indemnity review between US and EU entities | Financial-burden allocation |

## Integration shapes

- LangChain, LlamaIndex, LangGraph, CrewAI, Haystack, Pydantic AI
  adapters.
- n8n community node: two actions — `Score Groundedness` and
  `Route by Band` — with an example workflow committed under
  `integrations/n8n/examples/`.
- Hosted MCP endpoint at `api.latence.ai/mcp` for Claude Desktop,
  Cursor, and custom agent frameworks.

## Amber = reviewer queue, not auto-decision

Legal is the archetype where amber matters most.  Amber routes the
answer to a named reviewer with evidence highlights, scoring
diagnostics, and the hedge-gate rationale.  The reviewer accepts,
edits, or rejects in under a minute.  See
`docs/amber_reviewer_queue.md` for the per-vertical risk-band
semantics (E3).

## What TRACE does NOT do

- Does not draft contracts.  Bring your LLM.
- Does not provide legal advice.  It provides grounding signals.
- Does not substitute for outside counsel on novel questions.

## Proof and evidence

- `data/veracier-industries/proof_bundle_v1/proof_report.md`
- `commercial/bold-claims-v1.md` — every claim on this page is backed
  by a benchmark row or a signed artefact.
- Master Subscription Agreement, DPA, Trust Center, SOC 2 Type I
  letter.

## Latency contract

- **Isolated single-request**: p95 166 ms (standard), 178 ms (quality).  This is the number to quote for "how fast is one call?".
- **Sustained (concurrency = 8)**: p95 509 ms (standard), 862 ms (quality).  This is the production-pipeline number.
- Hosted SLA ceiling: 900 ms p95 standard, 2 200 ms p95 quality; see `commercial/hosted-sla.md`.
- Reproducer: `scripts/bench_latency.py --lane isolated --profile standard`.
- Raw numbers: `data/veracier-industries/proof_bundle_v1/latency_bench/` and `docs/operations/observability.md`.
