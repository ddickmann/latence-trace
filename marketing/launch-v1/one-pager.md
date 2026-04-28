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

**Honest about external benchmarks (production config, n=120, seed=42, reconciled 2026-04-28).**

| benchmark | metric | measured | website value | delta |
|---|---|---:|---:|---:|
| HaluEval QA | paired accuracy | 0.72 | 0.78 | -0.06 |
| HaluEval Summ | paired accuracy | 0.67 | 0.75 | -0.08 |
| RAGTruth QA | F1 @ median | 0.69 | 0.73 | -0.04 |
| RAGTruth QA | precision @ median | 0.93 | 0.98 | -0.05 |
| RAGTruth Summ | F1 @ median | 0.68 | 0.65 | +0.03 |

Competitive, not best-in-class. On adversarial code hallucinations
(HumanEval+ + CRUXEval identifier / literal / API-signature swaps)
the v1 RAG lane is not code-aware and misses the 0.80 gate; the v2
biaffine student with explicit code channels is architected for
that gap.

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
