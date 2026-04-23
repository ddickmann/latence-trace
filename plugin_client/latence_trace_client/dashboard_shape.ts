/**
 * Projection of :class:`SessionState` into the exact shape the UI
 * widget consumes. Keeping this separate from the state class means
 * UI developers can build the widget against a stable contract
 * without coupling to the rolling-stats internals.
 */

import { evictionCandidates } from "./ema_owner_share";
import type { SessionSnapshot, TurnEvent } from "./session_state";

export interface DashboardShape {
  /** Total number of turns observed on this session. */
  turns: number;
  /** Fraction of turns that triggered the NLI cascade. */
  cascadeFireRate: number;
  /** Most recent overall risk band for a giant stoplight in the UI. */
  lastRiskBand: TurnEvent["risk_band"];
  /** Rolling phantom-verdict trail — displayed as dots under the stoplight. */
  verdictTrail: Array<boolean | null>;
  /** Files sorted by EMA of ``owner_share`` (low first — eviction candidates). */
  filesByUsage: Array<{ path: string; ema: number; deadTurns: number }>;
  /** Files we'd recommend evicting (dead for N turns and EMA below threshold). */
  evictionCandidates: string[];
  /** Last per-turn component latencies — displayed as a stacked bar. */
  lastLatencyStack: {
    scorer_ms: number | null;
    ast_ms: number | null;
    nli_ms: number | null;
    composite_ms: number | null;
    total_ms: number | null;
  };
}

export function toDashboardShape(
  snapshot: SessionSnapshot,
  opts: { deadThreshold?: number; consecutiveTurns?: number } = {},
): DashboardShape {
  const files: Array<{ path: string; ema: number; deadTurns: number }> = [];
  for (const [path, ema] of snapshot.fileEma) {
    files.push({
      path,
      ema,
      deadTurns: snapshot.fileDeadTurns.get(path) ?? 0,
    });
  }
  files.sort((a, b) => a.ema - b.ema);

  const lastEvent = snapshot.lastEvent;

  return {
    turns: snapshot.totalTurns,
    cascadeFireRate: snapshot.cascadeFireRate,
    lastRiskBand: snapshot.lastRiskBand,
    verdictTrail: snapshot.verdictTrail,
    filesByUsage: files,
    evictionCandidates: evictionCandidates(
      snapshot.fileDeadTurns,
      snapshot.fileEma,
      opts,
    ),
    lastLatencyStack: {
      scorer_ms: lastEvent?.scorer_ms ?? null,
      ast_ms: lastEvent?.ast_ms ?? null,
      nli_ms: lastEvent?.nli_ms ?? null,
      composite_ms: lastEvent?.composite_ms ?? null,
      total_ms: lastEvent?.total_ms ?? null,
    },
  };
}
