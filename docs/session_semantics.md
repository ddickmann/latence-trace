# Session semantics — temporal signals without server-side state

## TL;DR

The `/groundedness` API is **stateless for per-turn measurement**.
Temporal signals (drift, EMA groundedness, dead-weight streaks, session
recommendations) need memory *somewhere* — we make that memory a
**first-class, caller-carried** payload. The server never persists any
conversation state.

```text
turn 1:  client → {request}                          → server
         client ← {response, next_session_state=S1}  ← server
turn 2:  client → {request, session_state=S1}        → server
         client ← {response, next_session_state=S2}  ← server
...
```

Opting in is one line on the request. The blob is opaque, bounded,
and round-trips verbatim.

## Why this shape

- **Stateless transport / service layer** — easier privacy posture,
  regional compliance, no conversation storage, no sticky routing.
- **Honest contract** — we do not pretend drift can be inferred
  reliably from one turn; it is a session-level question and we
  say so.
- **Plays well with MCP / plugins** — host runtimes already carry
  session memory.
- **Sensor / control split** — the server is the sensor (stateless
  per-turn measurement); the caller is the control layer (history,
  thresholds, actions).

## Opting in

A request opts into session mode if **either** field is present:

| Field on request | Meaning |
| --- | --- |
| `session_id: "<opaque-token>"` | First turn. Server returns a fresh `next_session_state` (and `session_signals`). |
| `session_state: {...}` | Subsequent turn. Server runs the pure transform and returns the updated blob. |

Requests that pass neither get the classic per-turn response with no
session fields — zero overhead.

## The wire shape

```jsonc
// Request (turn 2+)
{
  "scoring_mode": "code",
  "session_id": "ide-sess-abc123",
  "session_state": {
    "schema_version": 1,
    "total_turns": 5,
    "cascade_fires": 1,
    "per_token_rolling": { "n": 5, "mean": 0.52, "m2": 0.01 },
    "composite_rolling": { "n": 5, "mean": 0.71, "m2": 0.03 },
    "groundedness_rolling": { "n": 5, "mean": 0.78, "m2": 0.02 },
    "groundedness_baseline": { "n": 5, "mean": 0.78, "m2": 0.02 },
    "ema_groundedness": 0.76,
    "file_stats": {
      "src/tracer.py": {
        "ema_owner_share": 0.63,
        "ema_query_owner_share": 0.58,
        "dead_turns": 0,
        "total_turns": 5
      }
    },
    "phantom_trail": [false, false, true, false, false],
    "risk_band_trail": ["green", "green", "amber", "green", "green"],
    "ema_half_life_turns": 5,
    "groundedness_ema_half_life": 3,
    "verdict_window": 20,
    "risk_band_window": 20
  },
  "query_text": "...",
  "raw_context": "...",
  "response_text": "..."
}
```

```jsonc
// Response
{
  "scoring_mode": "code",
  "session_id": "ide-sess-abc123",
  "scores": { ... },
  "code_lane_diagnostics": { ... },
  "next_session_state": { ... same shape as session_state ... },
  "session_signals": {
    "total_turns": 6,
    "drift_z_score": 0.42,
    "ema_groundedness": 0.74,
    "groundedness_drift": -0.04,
    "dead_file_candidates": [],
    "dead_weight_streak": 0,
    "cascade_density": 0.17,
    "phantom_rate": 0.2,
    "red_streak": 0,
    "recommendation": "continue"
  }
}
```

## Guarantees

- **Determinism** —
  `update_session_state(prior, turn)` is a pure function: same inputs
  produce bit-identical outputs. Persistent across server restarts,
  zone migrations, canary rollouts, or horizontal scaling.
- **Bounded size** —
  `file_stats` is capped at 64 entries (lowest-EMA evicted first).
  Trail windows default to 20. The blob typically weighs
  **\< 2 KB** per session.
- **Forward compatibility** —
  The blob carries a `schema_version`. Servers that see an unknown
  version treat it as a fresh session (no crash). Bump-and-forget
  upgrades are safe.
- **No PII** —
  Only hashed session IDs and file-path keys. Put **never** raw user
  text in `session_id`.

## Derived signals

`session_signals` are cheap rollups the caller can render directly
without cracking the blob open. They are **advisory** — every per-turn
score in `scores` / `code_lane_diagnostics` is still available
independently.

| Field | What it means |
| --- | --- |
| `drift_z_score` | Absolute z-score of current `per_token_p10` vs the session's rolling mean. ≥ 2.0 is a strong drift signal. |
| `ema_groundedness` | EMA of the turn-level groundedness headline over the session (half-life: 3 turns). |
| `groundedness_drift` | `ema_groundedness − baseline_mean` (baseline = first ~10 turns). Negative ⇒ session is degrading vs its own early behaviour. |
| `dead_file_candidates` | Files whose EMA owner_share < 0.02 for ≥ 5 consecutive turns. Safe to evict from the agent's context. |
| `dead_weight_streak` | Longest active dead-turn run across tracked files. |
| `cascade_density` | Fraction of turns that triggered the NLI cascade. Sustained > 0.5 ⇒ session in an ambiguous regime. |
| `phantom_rate` | Rolling fraction of turns flagged `ast_phantom_verdict=True`. |
| `red_streak` | Trailing count of consecutive `red` risk-band turns. |
| `recommendation` | One of `continue` / `re-anchor` / `fresh-chat`. See policy below. |

## Recommendation policy

```python
if total_turns < 3:                        recommendation = "continue"
elif red_streak >= 3 or phantom_rate >= 0.3: recommendation = "fresh-chat"
elif (drift_z >= 2.0
      or groundedness_drift <= -0.15
      or red_streak >= 2):                   recommendation = "re-anchor"
else:                                        recommendation = "continue"
```

Thresholds are conservative by design — we'd rather `continue` a
genuine edge case than nudge users to reset working sessions.

## Reset events

The caller should throw the blob away and start over when:

- A new user conversation begins (new chat pane, new task root).
- Model / system prompt changes materially.
- The caller explicitly rejects the recommendation (`fresh-chat`)
  and starts over.
- The server reports a schema mismatch (`schema_version` in the
  response differs from the version the client last saw).

## Language bindings

- **Python** — wire models in `latence_trace.api.models`
  (`SessionStatePayload`, `SessionSignals`).
- **TypeScript** — reference client in
  [`plugin_client/latence_trace_client/session_state.ts`](../plugin_client/latence_trace_client/session_state.ts).
  The TS helper implements the same algorithm locally for plugins
  that want fully-offline computation; it reads the same
  `next_session_state` shape emitted by the server.

## See also

- [`docs/code_lane_v3.md`](code_lane_v3.md) — full code-lane design.
- [`docs/code_lane_turn_event.md`](code_lane_turn_event.md) — the
  structured log event that pairs with each turn.
- [`latence_trace/core/code_lane/session.py`](../latence_trace/core/code_lane/session.py)
  — the deterministic transform.
