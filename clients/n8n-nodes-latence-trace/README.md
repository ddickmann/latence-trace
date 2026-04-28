# n8n-nodes-latence-trace

Community [n8n](https://n8n.io) node for [Latence TRACE](https://latence.ai).

Score the groundedness of any RAG / agent response and route it to
the right downstream branch (`green`, `amber`, `red`) without writing
a single line of glue code.

## Operations

| Operation | What it does | Outputs |
| --- | --- | --- |
| **Score Groundedness** | POSTs `{question, response_text, raw_context}` to `POST /v1/score/groundedness` and adds `latence_trace` + `latence_band` fields to the item. | 1 (green) |
| **Route by Band** | Same scoring call, but sends the item to one of three outputs depending on `band`. | 3 (green / amber / red) |

## Installation

In n8n **Settings → Community Nodes**, install `n8n-nodes-latence-trace`
and restart.  Self-hosted operators can also `npm install` the
package into their `~/.n8n/nodes` directory.

## Credentials

- **API Key** — issue from `https://latence.ai/portal/keys`.
- **Base URL** — defaults to `https://api.latence.ai`.  Override when
  self-hosting (`http://trace.internal:8000`).

The node uses the key test endpoint `GET /v1/health` to verify the
credential before the first run.

## Example workflow

See `workflows/rag-review.json` for a ready-to-import flow:

1. **HTTP Request** node fetches retrieved chunks.
2. **OpenAI** node generates the candidate answer.
3. **Latence TRACE** node (Route by Band) splits the traffic:
   - **green** → deliver to user
   - **amber** → reviewer queue (Slack / Airtable)
   - **red** → fail closed and ask the model to retry with more
     evidence.

## HTTP Request blueprint (same-day fallback)

If you can't install community nodes yet (corporate restrictions),
the `workflows/rag-review-http-request.json` workflow implements the
same behaviour using only the built-in HTTP Request node.  Drop it
in, set the `LATENCE_API_KEY` credential, and you have routing in
ten minutes.

## Development

```bash
pnpm install
pnpm build
N8N_CUSTOM_EXTENSIONS=./dist n8n start
```

## License

Apache-2.0 © Latence AI.
