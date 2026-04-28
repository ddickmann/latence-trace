# Twitter / X thread

Post as a thread of 8 tweets.  Each tweet is <= 280 chars.  Strictly
one claim per tweet; every claim maps to a proof link.

---

**1/** TRACE v1.1 is live today.

Real-time groundedness verifier for RAG pipelines + agent frameworks.

100% red + green precision on our 118-case enterprise benchmark (Veracier Industries v1).

Proof bundle: github.com/latence-ai/latence-trace/tree/main/data/veracier-industries/proof_bundle_v1 [link]

**2/** Why a dedicated model instead of "LLM-as-a-judge"?

Because you need:
- token-level attribution, not vibes
- calibrated bands, not free text
- <200ms isolated p95 on A10
- an auditable failure appendix

LLM-as-judge can't give you any of that.

**3/** How we validated it:

118 enterprise RAG cases across finance, legal, HR, compliance, engineering, marketing.

3 answer variants per case: perfect / ambiguous / wrong.

Every row hand-reviewed by a domain specialist.

Dataset + curation log + failure appendix = all public.

**4/** External benchmarks are NOT SOTA at this snapshot.

HaluEval QA standard red precision: 0.59
RAGTruth QA quality red precision: 0.36

We publish the numbers, the failure modes, and the v2 biaffine student architecture that targets exactly those weak cases.

No hiding.

**5/** The v2 student is a tiny learned local-support head on top of the existing late-interaction encoder.

Distilled from the v1 API's audit log + gold labels on HaluEval + RAGTruth + Veracier.

Architecture + training loop in `research/triangular_maxsim/student_v2/`.

**6/** Integrations proved live, 15/15:

MCP stdio + HTTP + SSE
LangChain, LangGraph, LlamaIndex
CrewAI, AutoGen, Haystack 2, Pydantic AI
OpenAI, n8n, TypeScript SDK, Python SDK

Proof JSON: `proof_bundle_v1/integration_proof.json`

**7/** Agents can call TRACE inline.

Cursor MCP manifest: `clients/mcp-manifests/cursor.json`

Claude Desktop SSE manifest: `clients/mcp-manifests/claude-desktop.json`

OpenAI Agents SDK sample: `clients/mcp-manifests/openai-agents.py`

**8/** Try it:

Free tier + $50 of usage: latence.ai/signup

Replayable sandbox (no signup): latence.ai/try

Pilot contract template with the Veracier proof PDF attached: commercial/pilot-contract-template.md

Questions? DMs open.
