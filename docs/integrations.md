# Supported integrations

Latence TRACE plugs into your existing RAG / agent stack without a
rewrite. The Python adapters live in the sibling `latence-trace-python`
repository under `latence.integrations` and share the same client
(`Latence`).

## Matrix

| Framework | Adapter | Hook point | Extras install | Tested against |
| --- | --- | --- | --- | --- |
| OpenAI Python SDK | `latence.integrations.openai.wrap_openai_chat` | `client.chat.completions.create` wrapper + post-hoc `score_openai_response` | `latence[openai]` | `openai>=1.35` |
| LangChain | `latence.integrations.langchain.LatenceTraceCallback` | BaseCallbackHandler - `on_chain_end` | `latence[langchain]` | `langchain>=0.1.0` |
| LlamaIndex | `latence.integrations.llama_index.LatenceTracePostProcessor` | NodePostprocessor - runs after retrieval before synthesis | `latence[llama_index]` | `llama-index>=0.10` |
| LangGraph | `integrations.langgraph.score_groundedness_node` | Drop-in graph node + `trace_band` conditional edge | included | `langgraph>=0.0.50` |
| CrewAI | `integrations.crewai.LatenceTraceCallback` | `Task.callback` | included | `crewai>=0.30` |
| Microsoft AutoGen | `integrations.autogen.register_trace_hook` | `ConversableAgent.register_reply` | included | `pyautogen>=0.2.30` |
| Haystack 2.x | `latence.integrations.haystack.LatenceTraceScorer` | `@component` decorator -> Pipeline node | `latence[haystack]` | `haystack-ai>=2.0` |
| Pydantic AI | `integrations.pydantic_ai.trace_result_validator` | `Agent.output_validator` | included | `pydantic-ai>=0.1` |
| n8n (low-code) | `clients/n8n-nodes-latence-trace` | Community node + HTTP Request blueprint | npm `n8n-nodes-latence-trace` | `n8n>=1.70` |
| MCP (generic agents) | `clients/mcp-manifests/` + `latence_trace.mcp` | stdio + SSE + streamable-HTTP | included | Cursor, Claude Desktop, OpenAI Agents SDK |
| TypeScript / Node | `clients/typescript` (`@latence/trace`) | `scoreGroundedness()` | npm `@latence/trace` | Node 18+, Deno, Bun, Workers |

## Minimum contract

Every adapter does the same four things:

1. Extract the **question** the user asked (or delegate via a
   caller-supplied getter).
2. Extract the **response text** (the model's candidate answer).
3. Extract the **raw context** (concatenated retrieved evidence).
4. Call `client.grounding.rag(...)` or `client.grounding.code(...)` and expose `band` +
   `groundedness` on the downstream object.

This contract lets you swap adapters freely: the scoring, banding,
and audit behaviour stays identical.

## Error posture

All adapters **fail open** by default: if TRACE returns an error,
the adapter logs a structured warning and lets the original reply
through.  Security-sensitive callers can opt in to fail-closed by:

- Setting `block_on_red=True` on AutoGen / result validators.
- Switching the LangGraph conditional edge's default branch to
  `retry` instead of `END`.
- Adding a Haystack branching component on `scored[*].band`.

## What is tested in CI

Unit tests for each adapter live under `tests/integrations/`.  The
end-to-end tests use the in-process `GroundednessService` factory so
the CI run doesn't depend on the hosted API.

The Veracier proof bundle (`data/veracier-industries/proof_bundle_v1/`)
is re-run against every release candidate; the same proof numbers
therefore apply to every adapter on the matrix.
