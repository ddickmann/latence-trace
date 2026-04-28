# Show HN: TRACE - a real-time groundedness verifier for RAG and agents

**Title draft** (<= 80 chars):

> Show HN: TRACE - real-time groundedness scoring for RAG, with proof

**Body** (plain text, no markdown on HN):

---

Hi HN - we built TRACE, an API that tells your RAG pipeline or your agent framework whether a generated response is actually grounded in the retrieved evidence.

What's different:

- token-level scoring, not just a pass/fail label
- three calibrated bands: green / amber / red, plus an NLI verdict and a token heatmap
- real-time latency: isolated p95 < 200 ms on an A10, sustained p95 < 600 ms at concurrency 32
- zero-shot - no tenant fine-tuning required for the six verticals we ship (finance, legal, HR, compliance, engineering, marketing)

Proof we didn't just cherry-pick:

- Veracier Industries v1, 118 curated enterprise RAG cases with three answer variants each. TRACE hit 100% red precision, 100% green precision, 88% amber agreement. Full curated dataset, curation log, failure appendix and reproducer script are in the repo.
- External benchmarks under the production config (English NLI + atomic claims + reranker, n=120 seed=42, reconciled 2026-04-28): HaluEval QA paired accuracy ~0.72, RAGTruth QA F1 ~0.69 / precision ~0.93, RAGTruth Summ F1 ~0.68. Earlier public numbers around 0.59 / 0.36 came from a bench harness that silently dropped the anchor question and broke RAGTruth packing - both fixed, both documented under `data/veracier-industries/proof_bundle_v1/external_bench_production/reconciliation.md`.
- On adversarial coding (HumanEval+ + CRUXEval identifier / literal / API-signature swaps) the v1 RAG lane is not code-aware and misses the 0.80 paired-accuracy gate. We publish that result too. The v2 biaffine student with explicit code channels (identifier-match bit, numeric-match bit, AST role, source-type) targets exactly that gap; architecture + training loop live in `research/triangular_maxsim/student_v2/`.

Integrations: MCP (stdio + HTTP + SSE), LangChain, LangGraph, LlamaIndex, CrewAI, AutoGen, Haystack 2, Pydantic AI, OpenAI, n8n, TypeScript SDK, Python SDK. Every one is proved live in `proof_bundle_v1/integration_proof.json`.

Free tier: latence.ai/signup. Sandbox replay without a signup: latence.ai/try.

Happy to answer questions about: the scoring algorithm, why external benchmarks lag Veracier, the v2 student architecture, latency numbers, or how the Cloudflare gateway isolates tenants.

---

**Commenting strategy**

* First hour: be online, answer every question in under 15 minutes.
* If someone accuses us of cherry-picking Veracier, point at `failure_modes_v1.md` immediately.  Do not get defensive.
* If someone asks "why not just use OpenAI's moderation?", explain the local-support / groundedness distinction with the Eiffel Tower example from the blog.
* If someone asks for the architecture diagram, link to `research/triangular_maxsim/student_v2/README.md` for v2 and to `docs/scoring.md` for v1.
* If someone asks about prices, link to `commercial/price-card.md` - do not quote in-thread.

**Do not say**

* "We are the best."
* "We have zero false positives." (Only true on Veracier, and only after curation.)
* "Our model has 100% accuracy." (Unscoped claim.)
* "We will detect all hallucinations." (We won't.)
