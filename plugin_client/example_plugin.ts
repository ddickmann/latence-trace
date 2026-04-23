/**
 * Minimal usage sketch for an IDE plugin. Drop this next to your
 * ``/groundedness`` call site — works in VS Code, Cursor, and
 * JetBrains (via Node bridge) without changes.
 */

import { SessionState, type TurnEvent } from "./latence_trace_client/session_state";
import { toDashboardShape } from "./latence_trace_client/dashboard_shape";

const session = new SessionState({ emaHalfLifeTurns: 5 });

/**
 * Wrap the user's turn. ``scoreResponse`` is the function your plugin
 * already uses to POST to ``/groundedness``; we only add the
 * dashboard glue.
 */
export async function runTurn(
  scoreResponse: (payload: unknown) => Promise<{
    scoring_mode: "code";
    session_id: string;
    scores: Record<string, unknown>;
    code_lane_diagnostics: Record<string, unknown>;
  }>,
  payload: unknown,
): Promise<ReturnType<typeof toDashboardShape>> {
  const started = Date.now();
  const response = await scoreResponse(payload);
  const elapsed = Date.now() - started;

  const scores = response.scores as Record<string, any>;
  const diag = response.code_lane_diagnostics as Record<string, any>;
  const component = (diag?.component_latency_ms ?? {}) as Record<string, number>;
  const event: TurnEvent = {
    ts: new Date().toISOString(),
    lane: "code",
    session_id: response.session_id,
    cascade_fired: Boolean(scores.nli_cascade_triggered),
    phantom_verdict:
      typeof scores.composite_phantom_verdict === "boolean"
        ? scores.composite_phantom_verdict
        : null,
    phantom_probability: scores.composite_phantom_probability ?? null,
    dead_weight_ratio: scores.dead_weight_ratio ?? null,
    nli_ms: component.nli_ms ?? null,
    ast_ms: component.ast_ms ?? null,
    scorer_ms: component.scorer_ms ?? null,
    composite_ms: component.file_attribution_ms ?? null,
    total_ms: elapsed,
    risk_band: (scores.risk_band as TurnEvent["risk_band"]) ?? null,
    file_attribution: diag?.file_attribution?.entries ?? [],
  };
  return toDashboardShape(session.record(event));
}
