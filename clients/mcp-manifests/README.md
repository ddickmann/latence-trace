# Latence TRACE MCP client manifests

Drop-in configuration files that let AI coding agents discover
Latence TRACE as a tool without any glue code.

## Files

| File | Transport | Client |
| --- | --- | --- |
| `claude-desktop.json` | SSE | Anthropic Claude Desktop |
| `cursor.json` | streamable-HTTP | Cursor IDE |
| `openai-agents.py` | streamable-HTTP | OpenAI Agents SDK (Python) |

All three talk to the same remote endpoint at
`https://api.latence.ai/mcp`.  The Cloudflare gateway verifies the
bearer token from `LATENCE_API_KEY`, enforces per-tenant rate limits
and quotas, then proxies the JSON-RPC payload to the RunPod worker.

## Getting a key

Free tier keys are issued self-serve at `https://latence.ai/signup`
(1 000 scores / month on the `standard` profile).  Paid tiers and
higher-volume keys are available from `https://latence.ai/pricing`.

## Claude Desktop

```json
// ~/Library/Application Support/Claude/claude_desktop_config.json
{
  "mcpServers": {
    "latence-trace": {
      "command": "curl",
      "args": ["-N", "-H", "Authorization: Bearer $LATENCE_API_KEY",
               "https://api.latence.ai/mcp/sse"]
    }
  }
}
```

Or import `claude-desktop.json` directly if your Claude build
supports the `import` action in Settings → Developer.

## Cursor

1. Open Cursor → Settings → MCP.
2. Click `+ Add new MCP server`.
3. Paste the contents of `cursor.json` (adjust the env reference).

## OpenAI Agents SDK

```python
# See openai-agents.py for the full example.
from agents.mcp import MCPServerSse

server = MCPServerSse(
    params={
        "url": "https://api.latence.ai/mcp/sse",
        "headers": {"Authorization": f"Bearer {os.environ['LATENCE_API_KEY']}"},
    },
)
```

## Available tools

* `score_groundedness` - single tool that accepts `{question,
  response_text, raw_context, profile?}` and returns the TRACE
  scoring envelope with `groundedness`, `band`, `teacher_channels`,
  and `top_k_attributions`.

See the stdio reference client (`python -m latence_trace.mcp`) for
the schema.
