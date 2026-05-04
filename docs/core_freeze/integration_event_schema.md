# TRACE Integration Event Schema

All integrations should convert native tool events into one TRACE event/check schema. Integrations must not implement their own scoring logic.

## Standard Event

```json
{
  "event_id": "evt_...",
  "session_id": "sess_...",
  "event_type": "tool_result",
  "integration": "langgraph",
  "agent_type": "rag",
  "content": "retrieved or generated text",
  "metadata": {
    "tool_name": "retrieve_policy",
    "source": "customer_app"
  }
}
```

## Standard Check

```json
{
  "session_id": "sess_...",
  "mode": "rag",
  "query_text": "Can this invoice be refunded?",
  "response_text": "The invoice will be refunded within 48 hours.",
  "support_units": [
    {
      "text": "Refunds require manual approval.",
      "source_id": "policy_2026"
    }
  ],
  "options": {
    "privacy_redact": true,
    "verify_grounding": true,
    "memory_step": true,
    "decision": "review_on_amber"
  }
}
```

## Adapter Rules

- LangChain and LangGraph adapters wrap `TraceSession` and support async flows.
- LlamaIndex adapters should support both retrieval-postprocessing and final-answer verification.
- n8n starts with OpenAPI/HTTP recipe nodes; a custom node can follow after the contract freezes.
- Cursor, Claude Code, and Codex should prefer MCP/tool schemas and HTTP recipes first.
- MCP tools should expose product paths, not just `score_groundedness`.

## Product Path Mapping

| Intent | HTTP path | RunPod action | SDK path |
|---|---|---|---|
| Redact private data | `POST /v1/compliance/redact` | `redact` | `client.privacy.redact(...)` |
| Verify RAG output | `POST /groundedness` with `scoring_mode=rag` | `score` | `client.grounding.rag(...)` |
| Verify code-agent output | `POST /groundedness` with `scoring_mode=code` | `score` | `client.grounding.code(...)` |
| Compress text/messages | `POST /v1/compression` | `compress` | `client.compression.text(...)` / `messages(...)` |
| Update caller-carried memory | `POST /v1/memory/update` | `memory.update` | `client.memory.step(...)` |
| Roll up turns | `POST /groundedness/rollup` | `rollup` | `client.rollup(...)` |

## State Rules

- SDK/plugin owns local session ids, idempotency keys, and optional buffering.
- SDK/plugin may own memory state for stateless deployments.
- Tenant backend owns durable policy, logs, traces, server sessions, and dashboard state after the compute freeze.
- Compute runtime owns no durable customer state by default.
