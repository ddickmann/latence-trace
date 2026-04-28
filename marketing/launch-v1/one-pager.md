# Latence TRACE - one-pager

Paste into Slack DMs, sales emails, investor updates.  Every claim
links to an artifact so the reader can self-verify.

---

**What it is.**  Latence TRACE is a real-time groundedness verifier
for RAG pipelines and AI agents.  Token-level scoring, calibrated
green / amber / red bands, sub-second latency.

**Why it matters.**  RAG pipelines in regulated verticals still run
on "hope plus spot-check".  TRACE is the check as a service - agents
can call it inline, reviewers only see what actually needs review.

**Proof.**  On [Veracier Industries v1]
(https://github.com/latence-ai/latence-trace/tree/main/data/veracier-industries/proof_bundle_v1)
\- 118 curated enterprise RAG cases across six verticals - TRACE
scored:

* 100% red precision
* 100% green precision
* 88% amber agreement

**Integrations.**  15/15 transports proved live:

* MCP (stdio + HTTP + SSE) - Cursor, Claude Desktop, OpenAI Agents SDK
* LangChain, LangGraph, LlamaIndex
* CrewAI, AutoGen, Haystack 2, Pydantic AI
* n8n community node
* TypeScript + Python SDKs

Proof JSON: `proof_bundle_v1/integration_proof.json`

**Latency.**  Isolated p95 < 200 ms on A10.  Sustained p95 < 600 ms
at concurrency 32.

**Honest about external benchmarks.**

| benchmark | red precision | target |
|---|---|---|
| HaluEval QA standard | 0.59 | 0.80 |
| RAGTruth QA quality | 0.36 | 0.80 |

We publish the gap, the failure modes, and the v2 biaffine student
architecture that targets exactly those weak cases.

**Pricing.**  Free tier with $50 of usage.  Self-serve pro tier.
Enterprise pilot contract with Veracier PDF annex.

**Links.**

* Free tier: latence.ai/signup
* Replayable sandbox: latence.ai/try
* Docs: latence.ai/docs
* Repo: github.com/latence-ai/latence-trace
* Pilot contract: `commercial/pilot-contract-template.md`
* Trust center: latence.ai/trust
* Public roadmap: `docs/roadmap.md`
