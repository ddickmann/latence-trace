# Launch content pack: TRACE v1.1 (Veracier proof release)

Commit this directory 48 hours before launch so marketing, founders,
and design partners pull from the same source of truth. Every
claim in every asset here must map to a row in
`commercial/bold-claims-v1.md`. Anything without a map is cut.

## Asset checklist

| File | Audience | Owner | Publish on |
|---|---|---|---|
| `blog.md` | Engineering audience | Founding engineer | latence.ai/blog + LinkedIn article |
| `hn.md` | Hacker News | CTO | news.ycombinator.com "Show HN" |
| `linkedin.md` | Founders + enterprise buyers | CEO | LinkedIn post |
| `twitter-thread.md` | Engineering Twitter / developer network | CEO | x.com thread |
| `producthunt.md` | Product Hunt community | Head of Growth | producthunt.com/posts/latence-trace |
| `demo-script.md` | Sales / DevRel demo video (<= 3 minutes) | DevRel | Loom + YouTube unlisted |
| `one-pager.md` | Anyone (copy-paste into slacks) | Marketing | attached to every outbound |

## Shared talking points

1. **Veracier is defensible.** 118 curated enterprise RAG cases across
   finance, legal, HR, compliance, engineering, marketing.  100% red
   precision, 100% green precision, 88% amber agreement.
2. **Integrations are live.** MCP (stdio + HTTP + SSE), LangChain,
   LangGraph, LlamaIndex, CrewAI, AutoGen, Haystack, Pydantic AI,
   OpenAI, n8n, TypeScript SDK.  All 15 transports are proved by the
   integration proof JSON in `proof_bundle_v1/integration_proof.json`.
3. **External benchmarks are reconciled (2026-04-28 rebench).** Under
   the production config (English NLI + atomic claims + reranker +
   quality profile) TRACE v1.1 lands at HaluEval QA paired accuracy
   ~0.72 and RAGTruth QA F1 ~0.69 / precision ~0.93 at n=120 seed=42.
   Earlier "~0.59" / "~0.39" rows came from a harness that silently
   dropped the anchor question and broke RAGTruth packing — both bugs
   fixed and documented in
   `data/veracier-industries/proof_bundle_v1/external_bench_production/reconciliation.md`.
   Coding adversarial detection on HumanEval+ + CRUXEval identifier /
   literal / API-signature swaps is below target; the v2 biaffine
   student is architected for this case and is blocked on user-level
   corpus + target confirmation
   (see `data/veracier-industries/proof_bundle_v2/EVIDENCE_REPORT.md`).
4. **Real-time latency envelope.**  Isolated p95 < 200ms on A10,
   sustained p95 < 600ms at concurrency 32.  Exact numbers in
   `proof_bundle_v1/latency_bench/`.
5. **Hosted, GDPR-aligned, audit-ready.**  Cloudflare gateway +
   RunPod worker + audit-log API + DPA + pilot contract template.

## Do NOT

* Don't claim "best in class" on HaluEval or RAGTruth.
* Don't promise SOC 2 / ISO 27001 certification.  Use "readiness".
* Don't claim "zero hallucinations".  TRACE does not guarantee that.
* Don't use the word "certified" anywhere without a specific cert.
