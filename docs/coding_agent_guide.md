# Coding-agent integration guide

This is the copy-paste integration recipe for teams shipping coding
agents — IDE plugins (Cursor, Claude Code, Codex, OpenCode, Continue,
Aider, Roo Code, …) or autonomous agents that generate code against
an evolving file context.

## When to use the code lane

- The LLM context is **opened files / read-in source / markdown** and
  keeps accumulating as the agent runs.
- The LLM output contains **code fences** (Python, TypeScript,
  JavaScript, Go, Rust today — more on the way).
- You want a real-time signal for:
  - *drift* — is the generated code subtly wrong relative to the file
    the user is editing? (e.g. renamed a method and the agent reverts
    it silently),
  - *phantom* — did the agent invent an import, class, or function
    that exists in no opened file?
  - *dead weight* — which files in the 200 k-token context have
    contributed zero evidence for the last N turns and are just
    costing money?

If you are running a classic RAG pipeline over retrieved passages,
see [enterprise_rag_guide.md](enterprise_rag_guide.md) instead.

## Minimum viable request

```bash
curl -X POST http://127.0.0.1:8090/groundedness \
  -H "Content-Type: application/json" \
  -d '{
    "scoring_mode": "code",
    "session_id": "ide-session-abc123",
    "response_language_hint": "python",
    "query_text": "Rename calibrate to calibrate_threshold",
    "raw_context": "# file: tracer.py\nclass Tracer:\n    def calibrate_threshold(self, x): ...\n",
    "response_text": "```python\ntracer.calibrate(x)\n```"
  }'
```

Two knobs are worth setting from day one:

- **`scoring_mode: "code"`** — the discriminator that routes to the
  code-lane orchestrator. Without it the request goes through the
  RAG lane.
- **`session_id`** — stable identifier per IDE session. Enables
  multi-turn ownership accumulation (dead-weight detection) and
  per-session `per_token_p10` z-scoring. The server stays stateless —
  see [session_semantics.md](session_semantics.md) for the
  caller-portable `session_state` round-trip that unlocks drift,
  EMA groundedness, and eviction recommendations without any
  server-side conversation storage.

Optional but recommended:

- **`response_language_hint`** — `"python" | "typescript" | "javascript" | "go" | "rust"`.
  Speeds up AST symbol extraction by skipping language detection.
- **`support_units`** — pass pre-chunked opened files with
  `source_id` so ownership rolls up per file.

## What you get back

The body is the usual `GroundednessResponse` plus a dedicated
`code_lane_diagnostics` block:

```json
{
  "scoring_mode": "code",
  "session_id": "ide-session-abc123",
  "risk_band": "red",
  "scores": {
    "groundedness_v2": 0.52,
    "composite_logistic_v3": 0.41,
    "per_token_p10": 0.28,
    "literal_novelty_min": 0.17,
    "nli_contradiction_prob_max": 0.74
  },
  "code_lane_diagnostics": {
    "ast_literal_drift_count": 1,
    "ast_phantom_symbol_count": 0,
    "ast_phantom_verdict": false,
    "nli_cascade": {
      "triggered": true,
      "latency_ms": 22,
      "top_premises": ["tracer.calibrate_threshold ..."],
      "entailment": 0.11,
      "contradiction": 0.74
    },
    "file_attribution": [
      {
        "source_id": "tracer.py",
        "owner_share": 0.93,
        "query_owner_share": 0.88,
        "reason_codes": [],
        "unit_owners": [
          { "unit_id": 0, "max_cos": 0.71, "start": 0, "end": 154 }
        ]
      }
    ]
  }
}
```

Key fields:

- **`ast_literal_drift_count`** — deterministic count of identifiers
  used in the response but not present in any context AST. Catches
  the `calibrate` ↔ `calibrate_threshold` swap above.
- **`ast_phantom_verdict`** — `true` when the response imports /
  instantiates a symbol that exists in no context file.
- **`nli_cascade.triggered`** — fires only when the composite lands in
  the ambiguity band (`[0.65, 0.90]`). `triggered == false` means the
  fast path already had enough signal.
- **`file_attribution[*].owner_share`** — fraction of response tokens
  for which this file owned the argmax. EMA this on the client to
  recommend evictions (see `plugin_client/latence_trace_client/ema_owner_share.ts`).
- **`file_attribution[*].reason_codes`** — `never_won_argmax`,
  `all_tokens_below_0_40`, `dominated_by_single_file` — actionable
  reasons for flagging a file.
- **`file_attribution[*].unit_owners`** — per-chunk ownership with
  byte offsets. Enables *"drop lines 40-180 of this file"*
  sub-file routing when context gets tight.

## Per-plugin recipes

### Claude Code (Anthropic)

Wrap the agent's post-message hook:

```ts
import { SessionState } from "./plugin_client/latence_trace_client/session_state";
import { emaOwnerShare, evictionCandidates } from "./plugin_client/latence_trace_client/ema_owner_share";

const state = new SessionState(sessionId);

async function onAssistantMessage(msg: AssistantMessage) {
  const body = {
    scoring_mode: "code",
    session_id: sessionId,
    response_language_hint: detectLanguage(msg),
    query_text: lastUserMessage(),
    raw_context: concatenateOpenedFiles(),
    response_text: msg.text,
    support_units: openedFiles.map(f => ({ text: f.content, source_id: f.path })),
  };
  const result = await fetch("http://localhost:8090/groundedness", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }).then(r => r.json());

  state.ingest(result);
  renderStatusBar(state.snapshot());

  if (state.snapshot().phantomRate > 0.2) {
    notify("Claude has referenced symbols not in your opened files 3+ times.");
  }
}
```

### Cursor

Cursor plugin authors get the same flow via the
[`cursor-ide-browser` MCP](../plugin_client/example_plugin.ts).
Attach the score to each chat turn and surface the per-file owner
share in the side panel. The `session_id` should be the Cursor chat
UUID so a new chat starts fresh EMA windows.

### OpenAI Codex

For autonomous Codex runs (no IDE), keep `session_id` stable for the
whole run and evict files when
`evictionCandidates(ema, { minTurns: 5, maxOwnerShare: 0.02 })`
returns them. Cuts input tokens 20-40 % on long agentic runs
without hurting success rate.

### OpenCode

OpenCode plugin authors can subscribe to the structured
`groundedness_turn` log line (see
[`docs/code_lane_turn_event.md`](code_lane_turn_event.md)) instead
of polling the HTTP response. Same fields, lower latency.

### Continue / Aider / Roo Code

Any agent harness with a post-response callback can integrate with
the same shape. The
[`plugin_client/example_plugin.ts`](../plugin_client/example_plugin.ts)
reference implementation is framework-agnostic TypeScript.

## Real-time dashboard shape

`plugin_client/latence_trace_client/dashboard_shape.ts` exposes
`toDashboardShape(snapshot)` — a zero-dependency helper that turns
the session snapshot into the fields a status-bar / mini-dashboard
UI consumes:

```ts
const { driftScore, phantomRate, deadFiles, cascadeDensity } = toDashboardShape(state.snapshot());
statusBar.set({
  drift: driftScore,       // 0..1, z-scored per_token_p10
  phantom: phantomRate,    // 0..1, rolling rate of ast_phantom_verdict
  dead: deadFiles.length,  // number of files with EMA owner_share < 0.02
  cascade: cascadeDensity, // 0..1, fraction of turns that fired the NLI cascade
});
```

## Latency and budget

| Profile | p95 | Notes |
| --- | --- | --- |
| Code lane, cascade off | ~40 ms | Typical turn. |
| Code lane, cascade on | ~70 ms (target ≤ 150 ms) | Ambiguous responses only (~30 % of traffic in benchmarks). |

See [`docs/code_lane_performance.md`](code_lane_performance.md) for
the per-signal budget table and how to re-validate it.

## Also see

- [`docs/code_lane_v3.md`](code_lane_v3.md) — design doc for every
  signal.
- [`docs/session_semantics.md`](session_semantics.md) — the
  caller-portable `session_state` protocol that unlocks drift, EMA
  groundedness, and eviction recommendations without server-side
  state.
- [`docs/code_lane_turn_event.md`](code_lane_turn_event.md) — JSONL
  event spec for IDE dashboards.
- [`plugin_client/`](../plugin_client) — reference TypeScript
  helpers for IDE integration.
- [`research/triangular_maxsim/coding/`](../research/triangular_maxsim/coding/)
  — ablation matrix, transcripts, and latency benchmarks.
