/**
 * EMA helper for per-file owner_share tracking.
 *
 * Split out from SessionState so plugins that drive a custom state
 * machine (e.g. a redux store) can still reuse the same half-life
 * math.
 */

export interface OwnerShareSample {
  path: string;
  owner_share: number;
}

/**
 * Apply an EMA with the given half-life (measured in turns) to a
 * rolling map. Returns the updated map (mutates the input for speed
 * — clone beforehand if you need immutability).
 */
export function emaOwnerShare(
  previous: Map<string, number>,
  samples: Iterable<OwnerShareSample>,
  halfLifeTurns = 5,
): Map<string, number> {
  const alpha = 1 - Math.pow(0.5, 1 / Math.max(1, halfLifeTurns));
  for (const sample of samples) {
    const prior = previous.get(sample.path) ?? sample.owner_share;
    const next = alpha * sample.owner_share + (1 - alpha) * prior;
    previous.set(sample.path, next);
  }
  return previous;
}

/**
 * Utility: return all files whose EMA has dropped below
 * ``deadThreshold`` for at least ``consecutiveTurns`` turns. Use as
 * the "evict this file from context" recommendation.
 */
export function evictionCandidates(
  fileDeadTurns: Map<string, number>,
  fileEma: Map<string, number>,
  opts: { deadThreshold?: number; consecutiveTurns?: number } = {},
): string[] {
  const deadThreshold = opts.deadThreshold ?? 0.01;
  const consecutiveTurns = opts.consecutiveTurns ?? 3;
  const out: string[] = [];
  for (const [path, turns] of fileDeadTurns) {
    const ema = fileEma.get(path) ?? 0;
    if (turns >= consecutiveTurns && ema < deadThreshold) {
      out.push(path);
    }
  }
  return out;
}
