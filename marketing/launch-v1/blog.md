# How we built a real-time groundedness verifier that 15 agent frameworks already plug into

*This post is for engineers, RAG platform owners, and agent
infrastructure teams.  If you want the buyer-facing version, see
[the LinkedIn post](./linkedin.md) instead.*

---

## Why we built TRACE

Every serious RAG pipeline in a regulated vertical (finance, legal,
HR, compliance, engineering, marketing) ends up growing the same
patch: **a reviewer queue and a spreadsheet**.  The retrieval step is
good.  The generator is good.  But nobody owns the check that the
generator stayed inside the retrieved evidence.

TRACE is that check as a service:

* a token-level groundedness score,
* a calibrated risk band (green / amber / red),
* NLI claim verdicts and a token heatmap for every response,
* a real-time latency envelope: isolated p95 < 200ms on an A10.

## The validation we care about: Veracier Industries

We built a benchmark we could not hide behind: **Veracier Industries
v1**, 118 curated enterprise RAG cases across six verticals
(finance, legal, HR, compliance, engineering, marketing).  Three
answer variants per case - `perfect`, `ambiguous`, `wrong` - hand
reviewed by someone in that vertical.

Against that:

* **100% red precision** (no grounded answer is called hallucinated),
* **100% green precision** (no hallucination is called grounded),
* **88% amber agreement** on the hard middle cases.

The entire bundle (raw data, curated variants, curation log, failure
appendix, latency harness) is in
[`data/veracier-industries/proof_bundle_v1/`](../../data/veracier-industries/proof_bundle_v1/).

## The honest bit: external benchmarks

We ran HaluEval QA + RAGTruth QA at two profiles.  Here are the
numbers as measured on 2026-04-28:

| bench | profile | red precision | red recall | green precision |
|---|---|---|---|---|
| HaluEval QA | standard | 0.59 | 0.58 | 0.57 |
| HaluEval Summ | standard | 0.68 | 0.40 | 0.53 |
| RAGTruth QA | quality | 0.36 | 0.93 | 0.93 |
| RAGTruth Summ | quality | 0.53 | 0.38 | 0.74 |

On HaluEval QA we miss short-factoid hallucinations where the
response is 1-5 tokens and the NLI can't align.  On RAGTruth we
over-flag enumerated multi-step paraphrases where each bullet is
genuinely supported but the ColBERT MaxSim can't resolve the
alignment.

The score distributions are indistinguishable between `exp=green`
and `exp=red` on these rows, which means **no threshold change can
close the gap**.  The v2 biaffine student (learned local-support
head on top of the existing late-interaction encoder) is now in
training with that exact failure set as the distillation target.
Architecture + training loop are in
[`research/triangular_maxsim/student_v2/`](../../research/triangular_maxsim/student_v2/).

## How to actually use it

```python
from latence_trace_client import LatenceTraceClient

trace = LatenceTraceClient(api_key="...")
result = trace.score_groundedness(
    query="What was the Eiffel Tower completion year?",
    response_text="The Eiffel Tower was completed in 1889.",
    raw_context="The Eiffel Tower in Paris was completed in 1889 for the World's Fair.",
)
print(result.risk_band, result.scores.groundedness_v2)
# RiskBand.GREEN 0.996
```

Need a different language or framework?  TRACE v1.1 is proved live
against:

* MCP (stdio + HTTP streamable + SSE) - Cursor, Claude Desktop,
  OpenAI Agents SDK
* LangChain + LangGraph
* LlamaIndex
* CrewAI, AutoGen
* Haystack 2.x
* Pydantic AI
* n8n community node + HTTP Request node
* TypeScript + Python SDKs

The proof that each one actually lands a valid band is at
[`data/veracier-industries/proof_bundle_v1/integration_proof.json`](../../data/veracier-industries/proof_bundle_v1/integration_proof.json).

## What's next

* v1.2: GA hardening, first pen-test report, SOC 2 Type I letter.
* v2.0: biaffine student head.  Target: HaluEval QA red precision
  >= 0.90, RAGTruth QA red precision >= 0.80, Veracier precision
  unchanged.

Try the free tier at [latence.ai/signup](https://latence.ai/signup)
or replay the Veracier benchmark in your own browser at
[latence.ai/try](https://latence.ai/try).

---

*Written by the TRACE team. Commit history
[`b9c87a8..HEAD`](https://github.com/latence-ai/latence-trace).*
