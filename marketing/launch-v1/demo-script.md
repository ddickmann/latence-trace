# Demo script (<= 3 minutes)

Target: a sales-ready video explaining TRACE to a technical buyer.
Record once, reuse in pitch decks, on Product Hunt, and as the
latence.ai/demo hero asset.

## Timing budget

| Segment | Duration | Notes |
|---|---|---|
| 0:00 - 0:15 | Hook | What TRACE is, in one sentence |
| 0:15 - 0:45 | The problem | RAG pipelines in regulated verticals |
| 0:45 - 1:30 | The product | Live call + token heatmap |
| 1:30 - 2:10 | The proof | Veracier bundle + integration proof |
| 2:10 - 2:45 | The honest bit | External benchmarks, v2 roadmap |
| 2:45 - 3:00 | CTA | free tier / sandbox / pilot |

Total: 3 minutes. Hard-cap at 3:00.

## Screen-recording shot list

1. Portal dashboard @ latence.ai/app showing a live score stream.
2. Terminal with `curl` call to `/v1/score/groundedness` returning
   green band for the grounded fixture and red band for the
   contradiction fixture.  Use the same fixtures from
   `scripts/prove_integrations.py`.
3. Quick scroll through `proof_bundle_v1/integration_proof.json` to
   show 15/15 passing.
4. Quick scroll through `proof_bundle_v1/proof_report.md` headline.
5. Open `proof_bundle_v1/failure_modes_v1.md` and say "here's where
   we're still weak, here's the v2 student".
6. Zoom to `commercial/pilot-contract-template.md` to close on
   "pilot-ready today."

## Script

> [00:00]  "This is TRACE. It's a real-time groundedness verifier
> for RAG pipelines and agent frameworks. It tells you, right now,
> in the same request, whether the thing your LLM just said is
> actually supported by the evidence it retrieved."
>
> [00:15]  "If you've run a RAG pipeline in finance or legal or HR
> you already know the pain. The retrieval is good, the generator is
> good, but somebody in your team is still spot-checking a
> spreadsheet. TRACE is the check as a service, and the agent can
> call it inline."
>
> [00:45]  "Here's a live call. I'm sending a question, a response,
> and the retrieved context. TRACE gives me back a score, a
> calibrated band, a list of NLI verdicts, and a token heatmap. Under
> 200 milliseconds on a single A10."
>
> [01:15]  "Now I flip the answer to something that contradicts the
> evidence. Same call, same latency, but the band flips to red and
> the score drops to 0.04. That's what the agent loop catches before
> it ever shows up to a user."
>
> [01:30]  "We built a benchmark we couldn't hide behind. 118
> enterprise RAG cases, six verticals, three answer variants per
> case, every row hand-reviewed. TRACE scored 100% red precision,
> 100% green precision, 88% amber agreement on that bundle. And
> every integration - MCP, LangChain, LlamaIndex, CrewAI, AutoGen,
> Haystack, Pydantic AI, n8n, TypeScript SDK - is proved live in one
> JSON file."
>
> [02:10]  "The honest bit. On external corpora like HaluEval and
> RAGTruth we're below SOTA right now. Short-factoid hallucinations
> and multi-step paraphrased answers are the two weak modes. We
> publish those numbers, we publish the failure analysis, and the
> v2 biaffine student is in training against that exact gap."
>
> [02:45]  "Free tier at latence.ai/signup with fifty dollars of
> usage. No-signup sandbox at latence.ai/try. Pilot contract
> template in the repo. If you run RAG in a regulated vertical,
> book a pilot call."

## Do not say

* "Always", "never", "zero hallucinations", "100% accuracy".
* "Certified" for anything we haven't been audited on.
* Anything about the v2 student that implies it is shipping this
  quarter.

## Do say

* "Under conditions X, TRACE does Y."
* "Here is the failure mode. Here is the mitigation."
* "Proof is in the public repo at this path."
