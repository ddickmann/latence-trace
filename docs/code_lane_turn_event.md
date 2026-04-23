# Code-lane turn-event format (v1)

Every `/groundedness` call with `scoring_mode == "code"` produces a
single structured log line that IDE plugins consume to drive the live
"agent quality" dashboard. The runtime emits the event at logger
`latence_trace.runpod.handler` as `groundedness_turn` at INFO level.
It is also available over Prometheus scrape via the counters
`latence_trace_lane_requests_total{lane}`,
`latence_trace_code_lane_cascade_fires_total{lane}`,
`latence_trace_code_lane_phantom_verdicts_total{verdict}`, and
`latence_trace_lane_budget_exceeded_total{lane}`.

## Why
The code lane is stateless on the server; session quality is reconstructed
on the client from a stream of per-turn events. Plugins running inside
Claude Code, Cursor, OpenAI Codex, or OpenCode subscribe to this log
line (or scrape the JSON response of `/groundedness`) and feed it into a
session store that powers:

- Rolling `per_token_p10` z-score for drift detection.
- EMA of `owner_share` per file for multi-turn dead-weight evict hints.
- Cascade-fire density to visualise when the model is skating on thin ice.
- Phantom verdict histogram to trigger "trust mode" UI affordances.

## Wire schema

```jsonc
{
  "ts": "2026-04-23T08:01:22.518Z",
  "lane": "code",
  "session_id": "sess-a1b2c3",
  "cascade_fired": true,
  "phantom_verdict": false,
  "phantom_probability": 0.732,
  "dead_weight_ratio": 0.18,
  "nli_ms": 22.4,
  "ast_ms": 4.1,
  "scorer_ms": 6.8,
  "composite_ms": 3.2,
  "file_attribution_ms": 1.8,
  "literal_novelty_ms": 0.4,
  "total_ms": 142.0,
  "risk_band": "amber",

  // Caller-portable session signals (emitted when the request carries
  // a ``session_id`` or ``session_state`` — otherwise ``null``). See
  // docs/session_semantics.md for the full protocol.
  "session_total_turns": 8,
  "session_drift_z": 1.42,
  "session_ema_groundedness": 0.71,
  "session_cascade_density": 0.34,
  "session_phantom_rate": 0.08,
  "session_red_streak": 0,
  "session_recommendation": "continue"
}
```

All fields are PII-safe: `session_id` is echoed from the caller; it
MUST be a hash or opaque token. The server never logs `query_text`,
`response_text`, or raw file contents.

The component latency fields map 1:1 to the orchestrator phases in
[`latence_trace/core/code_lane/orchestrator.py`](../latence_trace/core/code_lane/orchestrator.py).
`composite_ms` is the sum of the pre-NLI and final composite calls;
`nli_ms` is zero whenever the cascade does not fire.

## Client-side state (reference)

The reference TypeScript helpers live in
`plugin_client/latence_trace_client/` and are designed to be
copy-pasted into any IDE plugin. They expose three primitives:

- `SessionState`: rolling per-session statistics (mean/std of
  `per_token_p10`, EMA of `owner_share` per file, cascade fire count,
  recent phantom verdicts).
- `emaOwnerShare`: EMA helper with a configurable half-life in turns.
- `dashboardShape`: derives the exact shape the UI widget consumes
  (drift z-score, dead-weight files, cascade heatmap, verdict trail).

See the source for the full contract and a minimal integration
example.
