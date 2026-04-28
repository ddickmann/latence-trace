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

We ran HaluEval QA, HaluEval Summ, RAGTruth QA, and RAGTruth Summ
under the **production config** (English NLI + atomic claims +
reranker + quality profile) on 2026-04-28, at n=120 seed=42. Reference
framing (positive = faithful):

| bench | F1@best | paired accuracy | precision@median | recall@median |
|---|---:|---:|---:|---:|
| HaluEval QA | 0.68 | 0.72 | 0.57 | 0.90 |
| HaluEval Summ | 0.65 | 0.67 | 0.53 | 0.85 |
| RAGTruth QA | 0.69 | -- | 0.93 | 0.55 |
| RAGTruth Summ | 0.68 | -- | 0.83 | 0.57 |

These are the numbers you can reproduce; the raw rows and summaries
live under
[`proof_bundle_v1/external_bench_production/`](../../data/veracier-industries/proof_bundle_v1/external_bench_production/).

**Two bug fixes landed on the way here.** Earlier public numbers
around "0.59 red precision" on HaluEval QA and "0.36" on RAGTruth QA
came from a bench harness that (a) sent `{"question": ...}` on each
call while the RunPod handler only reads `query_text` - the anchor
question was silently dropped on every row, and (b) flattened
RAGTruth's `source_info` dict into "natural prose" instead of the
reference methodology's `json.dumps(source_info)`. Both fixes are in
[`reconciliation.md`](../../data/veracier-industries/proof_bundle_v1/external_bench_production/reconciliation.md).
The delta is real, the root cause is documented, the numbers are
lower-bounded.

### Where v1 still misses

On adversarial coding (HumanEval+ + CRUXEval, three rule-based
variants per grounded row: identifier swap, literal swap, API-
signature swap) the v1 RAG lane is **not** code-aware. Paired
accuracy lands between 0.21 and 0.62 per variant - below the 0.80
ship gate. This is architectural, not a threshold issue: the
late-interaction encoder scores `authorize` and `authenticate` as
near-neighbours, which is exactly wrong for a symbol-swap
hallucination. The v2 biaffine student adds explicit code channels
(identifier-match bit, numeric-match bit, AST role, source-type) on
top of the existing encoder precisely to close this.  Artefact:
[`proof_bundle_v1/coding_bench/report.md`](../../data/veracier-industries/proof_bundle_v1/coding_bench/report.md).

Architecture + training loop for the v2 student are in
[`research/triangular_maxsim/student_v2/`](../../research/triangular_maxsim/student_v2/).
Student training is **blocked on user-level confirmation** of the
corpus mix and the public-SOTA target. Evidence report driving that
decision:
[`proof_bundle_v2/EVIDENCE_REPORT.md`](../../data/veracier-industries/proof_bundle_v2/EVIDENCE_REPORT.md).

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

* v1.2: GA hardening, first pen-test report, SOC 2 Type I letter,
  auto-decide middleware as opt-in (per-tenant flag
  `auto_decide=true` on the request). The middleware is already
  wired end-to-end; the ship gate on external corpora is not yet
  met with the compact judge payload, see
  [`proof_bundle_v1/auto_decide/report.md`](../../data/veracier-industries/proof_bundle_v1/auto_decide/report.md).
* v2.0: biaffine student head.  Target: HaluEval QA paired accuracy
  >= 0.85, RAGTruth QA F1 >= 0.80, coding adversarial paired
  accuracy >= 0.80 on identifier / literal / API-signature swaps,
  Veracier precision unchanged.

Try the free tier at [latence.ai/signup](https://latence.ai/signup)
or replay the Veracier benchmark in your own browser at
[latence.ai/try](https://latence.ai/try).

---

*Written by the TRACE team. Commit history
[`b9c87a8..HEAD`](https://github.com/latence-ai/latence-trace).*
