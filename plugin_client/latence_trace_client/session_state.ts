/**
 * Per-session rolling state for the code-lane quality dashboard.
 *
 * Consumes ``groundedness_turn`` events produced by
 * ``/groundedness?scoring_mode=code`` and exposes aggregates that
 * power the live UI widget in Claude Code, Cursor, Codex, and
 * OpenCode plugins.
 *
 * Design notes
 * ------------
 * - Zero external deps; pure TypeScript so plugins can ship it
 *   verbatim.
 * - Stateless w.r.t. the server: every event is append-only, and the
 *   client decides the retention window.
 * - All statistics are computed incrementally so the helper never
 *   allocates beyond O(unique files) + O(retention window).
 */

export interface TurnEvent {
  /** ISO-8601 timestamp of the server-side turn emission. */
  ts: string;
  /** ``"rag"`` or ``"code"``. Only ``"code"`` events land here. */
  lane: "rag" | "code";
  /** Opaque/hashed session id echoed from the request. */
  session_id: string;
  cascade_fired: boolean;
  phantom_verdict: boolean | null;
  phantom_probability: number | null;
  dead_weight_ratio: number | null;
  nli_ms: number | null;
  ast_ms: number | null;
  scorer_ms: number | null;
  composite_ms: number | null;
  total_ms: number;
  risk_band: "green" | "amber" | "red" | null;
  /**
   * Optional payload the plugin can stuff from the full
   * ``GroundednessResponse`` — e.g. the ``code_lane_diagnostics.file_attribution``
   * entries. Kept as ``unknown`` so this helper stays compatible
   * with schema changes.
   */
  file_attribution?: FileAttributionEntry[];
}

export interface FileAttributionEntry {
  path: string;
  owner_share: number;
  query_owner_share?: number;
  dead_weight: boolean;
  reason_codes: string[];
}

export interface SessionStateOptions {
  /** Number of turns after which the EMA effectively decays by 1/e. */
  emaHalfLifeTurns?: number;
  /** Retention window for rolling per_token_p10 z-scores. */
  zscoreWindow?: number;
  /** Retention window for the phantom verdict trail. */
  verdictWindow?: number;
}

interface RollingMeanStd {
  n: number;
  mean: number;
  m2: number;
}

function updateRolling(rolling: RollingMeanStd, value: number): RollingMeanStd {
  const n = rolling.n + 1;
  const delta = value - rolling.mean;
  const mean = rolling.mean + delta / n;
  const m2 = rolling.m2 + delta * (value - mean);
  return { n, mean, m2 };
}

function rollingStd(rolling: RollingMeanStd): number {
  return rolling.n > 1 ? Math.sqrt(rolling.m2 / (rolling.n - 1)) : 0;
}

export class SessionState {
  private readonly opts: Required<SessionStateOptions>;
  private readonly perTokenRolling: RollingMeanStd = {
    n: 0,
    mean: 0,
    m2: 0,
  };
  private readonly deadWeightRolling: RollingMeanStd = {
    n: 0,
    mean: 0,
    m2: 0,
  };
  private readonly fileEma: Map<string, number> = new Map();
  private readonly fileDeadTurns: Map<string, number> = new Map();
  private cascadeFires = 0;
  private totalTurns = 0;
  private readonly verdictTrail: Array<boolean | null> = [];
  private lastRiskBand: TurnEvent["risk_band"] = null;
  private lastEvent: TurnEvent | null = null;

  constructor(opts: SessionStateOptions = {}) {
    this.opts = {
      emaHalfLifeTurns: opts.emaHalfLifeTurns ?? 5,
      zscoreWindow: opts.zscoreWindow ?? 50,
      verdictWindow: opts.verdictWindow ?? 20,
    };
  }

  /**
   * Feed one turn event in. Returns the current state snapshot so
   * React-style plugins can render without calling extra accessors.
   */
  record(event: TurnEvent): SessionSnapshot {
    this.totalTurns += 1;
    this.lastEvent = event;
    this.lastRiskBand = event.risk_band;
    if (event.cascade_fired) this.cascadeFires += 1;
    this.verdictTrail.push(event.phantom_verdict);
    while (this.verdictTrail.length > this.opts.verdictWindow) {
      this.verdictTrail.shift();
    }

    if (event.dead_weight_ratio !== null) {
      updateInPlace(this.deadWeightRolling, event.dead_weight_ratio);
    }

    const alpha = halfLifeToAlpha(this.opts.emaHalfLifeTurns);
    for (const entry of event.file_attribution ?? []) {
      const prev = this.fileEma.get(entry.path) ?? entry.owner_share;
      const next = alpha * entry.owner_share + (1 - alpha) * prev;
      this.fileEma.set(entry.path, next);
      if (entry.dead_weight) {
        this.fileDeadTurns.set(
          entry.path,
          (this.fileDeadTurns.get(entry.path) ?? 0) + 1,
        );
      } else {
        this.fileDeadTurns.set(entry.path, 0);
      }
    }

    return this.snapshot();
  }

  /** Incremental z-score of the most recent ``per_token_p10``. */
  perTokenZScore(latest: number): number {
    updateInPlace(this.perTokenRolling, latest);
    const std = rollingStd(this.perTokenRolling);
    if (std <= 1e-6) return 0;
    return (latest - this.perTokenRolling.mean) / std;
  }

  snapshot(): SessionSnapshot {
    return {
      totalTurns: this.totalTurns,
      cascadeFireRate:
        this.totalTurns === 0 ? 0 : this.cascadeFires / this.totalTurns,
      deadWeightRatioMean: this.deadWeightRolling.mean,
      fileEma: new Map(this.fileEma),
      fileDeadTurns: new Map(this.fileDeadTurns),
      verdictTrail: this.verdictTrail.slice(),
      lastRiskBand: this.lastRiskBand,
      lastEvent: this.lastEvent,
    };
  }
}

function updateInPlace(rolling: RollingMeanStd, value: number): void {
  const next = updateRolling(rolling, value);
  rolling.n = next.n;
  rolling.mean = next.mean;
  rolling.m2 = next.m2;
}

function halfLifeToAlpha(halfLifeTurns: number): number {
  return 1 - Math.pow(0.5, 1 / Math.max(1, halfLifeTurns));
}

export interface SessionSnapshot {
  totalTurns: number;
  cascadeFireRate: number;
  deadWeightRatioMean: number;
  fileEma: Map<string, number>;
  fileDeadTurns: Map<string, number>;
  verdictTrail: Array<boolean | null>;
  lastRiskBand: TurnEvent["risk_band"];
  lastEvent: TurnEvent | null;
}
