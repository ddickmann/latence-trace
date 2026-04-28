# Product Hunt launch page

**Name:** Latence TRACE
**Tagline (60 chars):** Real-time groundedness for RAG, with public proof
**Category:** Artificial Intelligence > Developer Tools

## Description (260 chars)

TRACE is a real-time groundedness verifier for RAG pipelines and AI agents. Calibrated green/amber/red bands, token heatmaps, <200ms p95 on A10. 100% precision on our 118-case enterprise benchmark. Drops into MCP, LangChain, CrewAI, n8n.

## Top features

1. **100% red + green precision on Veracier Industries v1** - 118 hand-reviewed RAG cases across six verticals.
2. **Real-time** - <200ms isolated p95 on A10, <600ms at concurrency 32.
3. **15 integrations, all proved live** - MCP (Cursor, Claude Desktop, OpenAI Agents), LangChain, LangGraph, LlamaIndex, CrewAI, AutoGen, Haystack, Pydantic AI, n8n, TypeScript + Python SDKs.
4. **Honest about external benchmarks** - reconciled production config n=120 seed=42: HaluEval QA paired accuracy 0.72, RAGTruth QA F1 0.69 / precision 0.93, RAGTruth Summ F1 0.68. Competitive, not SOTA on adversarial coding; failure modes documented, v2 biaffine student architected for that gap.
5. **Audit-ready** - append-only per-score log, per-tenant thresholds, DPA, pilot contract template.

## First comment (from maker)

Hi Product Hunt!

We are shipping TRACE v1.1 today. It is the real-time groundedness verifier we wish we'd had while building our last three RAG pipelines.

Two asks for this community:

1. Try the **replayable sandbox at latence.ai/try** - it replays the Veracier benchmark in your browser so you can see the bands + token heatmaps without a signup.
2. If you run a RAG pipeline in a regulated vertical, **book a 20-min pilot call at latence.ai/pilot**. We have three design partners live and capacity for three more before we close v1 pilots.

Everything we claim maps to an artifact in our public repo:
- `commercial/bold-claims-v1.md` - every claim, every proof link
- `data/veracier-industries/proof_bundle_v1/` - the benchmark bundle
- `research/triangular_maxsim/student_v2/` - the v2 biaffine student

External benchmarks are the hard part. Under the production config we land at HaluEval QA paired accuracy 0.72 and RAGTruth QA F1 0.69 at precision 0.93 (n=120 seed=42, reconciled after fixing two real bench-harness bugs - anchor-question drop + RAGTruth packing). Competitive, not best-in-class on adversarial code hallucinations. The v2 biaffine student is architected to close that gap. Happy to discuss the architecture.

Thanks for reading!

## Gallery

| Asset | Filename | Notes |
|---|---|---|
| Hero image | `../brand/ph-hero.png` | 1270x760, light background, token heatmap visual |
| Screenshot 1 | `../brand/ph-band-detail.png` | Close-up of the band + score + NLI verdict |
| Screenshot 2 | `../brand/ph-integrations.png` | Logo strip of supported frameworks |
| Screenshot 3 | `../brand/ph-audit-log.png` | Audit log viewer in the portal |
| Demo video | `demo-script.md` -> Loom link | <= 3 minutes |

## Tags

`api` `artificial-intelligence` `developer-tools` `rag` `agents`
