# TRACE for Marketing

**Who this is for:** CMOs, VPs of Marketing, Content Leads, Legal
Marketing Reviewers running a RAG or agent pipeline over marketing
claims, case-study evidence, analyst studies, and brand-usage
policies.

## Why TRACE matters for marketing RAG

Most marketing-RAG failure modes are about claim substantiation:

1. Publishing a performance claim unsupported by the current
   measurement window.
2. Using restricted third-party brand language ("verified by X").
3. Naming a customer before the release form is signed.
4. Over-claiming certifications that are still in progress.

TRACE scores each generated answer against the actual evidence
retrieved and routes uncertain claims to a human reviewer queue.

## Anchor use cases (ship with the v1 proof bundle)

| Use case | Question |
| --- | --- |
| `MKT-01` | Can we publish the claim "99% uptime" in the October campaign? |
| `MKT-02` | Can the pricing page cite "verified by Forrester" for the ROI claim? |
| `MKT-03` | Are the HIPAA-compliance claims safe to publish on the healthcare landing page? |
| `MKT-04` | Draft the Acme Bank case-study blurb — what can we claim about cost savings? |

All four ship in `data/veracier-industries/proof_bundle_v1/marketing_use_cases.json`
with three controlled variants each.

## Integration shapes

Python SDK, TypeScript SDK, LangChain / LangGraph / CrewAI / Haystack,
n8n community node, remote MCP endpoint.

## What TRACE does NOT do

- Does not generate marketing copy; pair with your LLM.
- Does not track prior approvals; pair with your CMS workflow.
- Does not replace legal-marketing review on novel claims.

## Latency contract

- **Isolated single-request**: p95 166 ms (standard), 178 ms (quality).  This is the number to quote for "how fast is one call?".
- **Sustained (concurrency = 8)**: p95 509 ms (standard), 862 ms (quality).  This is the production-pipeline number.
- Hosted SLA ceiling: 900 ms p95 standard, 2 200 ms p95 quality; see `commercial/hosted-sla.md`.
- Reproducer: `scripts/bench_latency.py --lane isolated --profile standard`.
- Raw numbers: `data/veracier-industries/proof_bundle_v1/latency_bench/` and `docs/operations/observability.md`.
