# Stateful TRACE Sessions

Stateful TRACE sessions compose the existing TRACE APIs into a continuous
runtime object for long-running coding, RAG, and general-purpose agents.

The basic APIs remain permanent:

- `POST /groundedness`
- `POST /groundedness/rollup`
- `POST /v1/memory/update`
- `POST /v1/compression`
- `POST /v1/compliance/redact`

Sessions add managed state on top. Each event updates InfiniMem immediately,
each score call loads and saves code-lane state plus memory state, and callers
receive bounded hot context without waiting for a dedicated summarization step.

## Endpoints

| Endpoint | Purpose |
|----------|---------|
| `POST /v1/trace/sessions` | Create a session with `kind=code`, `rag`, or `general`. |
| `GET /v1/trace/sessions/{session_id}` | Inspect session state and counters. |
| `POST /v1/trace/sessions/{session_id}/events` | Append user/tool/file/retrieval/error/decision events. |
| `POST /v1/trace/sessions/{session_id}/score` | Score a turn and update TRACE + InfiniMem state. |
| `POST /v1/trace/sessions/{session_id}/memory/update` | Memory-only stateful update. |
| `GET /v1/trace/sessions/{session_id}/context` | Fetch bounded hot context. |
| `POST /v1/trace/sessions/{session_id}/rollup` | Roll up stored scored turns. |
| `DELETE /v1/trace/sessions/{session_id}` | Close a session. |

Gateway paths are exposed under `/api/v1/trace/sessions`.

## Example

```bash
curl -X POST https://api.latence.ai/api/v1/trace/sessions \
  -H "Authorization: Bearer $LATENCE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"kind":"code","metadata":{"runtime":"cursor"}}'
```

```bash
curl -X POST https://api.latence.ai/api/v1/trace/sessions/trcsess_123/score \
  -H "Authorization: Bearer $LATENCE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "lane": "code",
    "trace_request": {
      "response_text": "I will update CacheClient.get_many.",
      "raw_context": "class CacheClient: def get_many(self, keys): ...",
      "response_language_hint": "python"
    }
  }'
```

## Safety

Gateway logging stores compact counters and diagnostics by default. Raw session
content should only be exported under an explicit debug path with redaction.
