# latence-trace plugin client

Reference TypeScript helpers for IDE plugins that display a live
"agent quality" dashboard on top of the `/groundedness?scoring_mode=code`
response stream. Zero external dependencies; drop the file into a
Claude Code, Cursor, OpenAI Codex, or OpenCode plugin and import.

## Files

- `latence_trace_client/session_state.ts` — per-session rolling stats
  (drift z-score, cascade density, phantom verdict trail).
- `latence_trace_client/ema_owner_share.ts` — EMA helper with a
  configurable half-life in turns (used to decide when a file has
  been "dead" for enough turns to recommend eviction).
- `latence_trace_client/dashboard_shape.ts` — projects the session
  state into the exact shape a UI widget consumes.
- `example_plugin.ts` — minimal usage sketch.

## Integration recipe

```ts
import { SessionState } from "./latence_trace_client/session_state";
import { toDashboardShape } from "./latence_trace_client/dashboard_shape";

const session = new SessionState({ emaHalfLifeTurns: 5 });

// On every turn, after calling /groundedness:
session.record(turnEvent);
const dashboard = toDashboardShape(session);
ui.render(dashboard);
```

See `docs/code_lane_turn_event.md` for the wire-format spec.
