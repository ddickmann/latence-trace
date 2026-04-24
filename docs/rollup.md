# Rollup — stateless session aggregation

`action="rollup"` turns the list of per-turn outputs a plugin already
has into one business-grade scoreboard: noise / drift / retrieval
waste, risk-band trail, top dead files, drift trend, and an optional
session-level heatmap card.

The rollup is **pure**:

- No model calls.
- No I/O.
- No state — the caller owns the list of turns.
- O(len(turns)) CPU, typically sub-millisecond for 10–100 turns.

That makes it safe to call on every keystroke if a plugin wants a
live session scoreboard.

## When to use it

- IDE plugin wants a "session quality" pill in the status bar that
  updates as the conversation grows.
- Enterprise dashboard wants a single row per conversation summarising
  how much of the retrieved context went to waste.
- Support flow wants to decide whether a conversation should be
  re-anchored or replaced with a fresh chat.

## Request

### HTTP

```http
POST /groundedness/rollup
Content-Type: application/json
Authorization: Bearer $API_KEY
```

### RunPod serverless

```jsonc
POST https://api.runpod.ai/v2/{endpoint_id}/runsync
{
  "input": {
    "action": "rollup",
    "turns": [ … ],
    "heatmap_format": "data"   // "none" | "data" | "html"
  }
}
```

### Per-turn record (`RollupTurnInput`)

Every field is optional. Include what the plugin already has from the
per-turn scoring response; the transform skips what it cannot read.

```jsonc
{
  "scores":            { "groundedness_v2": 0.71,
                         "composite_phantom_score": 0.18,
                         "dead_weight_ratio": 0.34,
                         "dead_weight_file_count": 2 },

  "session_signals":   { "drift_z_score": 0.4,
                         "ema_groundedness": 0.69,
                         "dead_weight_streak": 2,
                         "dead_file_candidates": ["docs/bamboo.md"],
                         "recommendation": "continue" },

  "file_attribution":  { "per_file": [ … ],
                         "reason_code_histogram": { "never_won_argmax": 2 },
                         "dead_weight_ratio": 0.34 },

  "risk_band":         "yellow",
  "recommendation":    "continue",
  "timestamp":         "2026-04-23T14:05:02Z"
}
```

## Response (`RollupResponse`)

```jsonc
{
  "turns": 5,

  "noise_pct":            0.34,     // mean dead_weight_ratio across turns
  "model_drift_pct":      0.21,     // drift_z_score + ema groundedness gap
  "retrieval_waste_pct":  0.12,     // dead weight weighted by grounded turns

  "reason_code_histogram": {
    "never_won_argmax": 7,
    "dominated_by_single_file": 2
  },

  "recommendations":   ["continue", "continue", "re_anchor",
                        "continue", "fresh_chat"],
  "risk_band_trail":   ["green", "green", "yellow", "yellow", "red"],

  "drift_trend": {
    "min": -0.2,  "max": 1.4,
    "mean": 0.6,  "last": 1.4
  },

  "top_dead_files": [
    { "path": "docs/bamboo.md",
      "dead_turns": 5,
      "ema_owner_share": 0.00 },
    { "path": "docs/jupiter_rings.md",
      "dead_turns": 3,
      "ema_owner_share": 0.04 }
  ],

  // Only populated when heatmap_format != "none".
  "heatmap":      { "summary": { … }, "files": [ … ], "thresholds": { … } },

  // Only populated when heatmap_format == "html".
  "heatmap_html": "<div class=\"lt-heatmap\">…</div>"
}
```

All three percentages are bounded to `[0, 1]`. The trails are
length-preserving: `risk_band_trail[i]` and `recommendations[i]` both
correspond to `turns[i]`.

## Semantics

| Metric                  | Formula                                                                                   |
| ----------------------- | ----------------------------------------------------------------------------------------- |
| `noise_pct`             | mean of `dead_weight_ratio` across all turns (pulled from `scores` or `file_attribution`) |
| `model_drift_pct`       | blend of `drift_z_score / 3` and `(1 − ema_groundedness)`, clipped to [0, 1]              |
| `retrieval_waste_pct`   | mean `dead_weight_ratio` across turns whose `groundedness_v2 ≥ 0.6`                       |
| `reason_code_histogram` | element-wise sum of per-turn `reason_code_histogram`                                      |
| `top_dead_files`        | derived from the latest non-empty session signal; falls back to per-turn attribution      |

The exact formulas live in `latence_trace/api/rollup.py`. They are
small and documented so a plugin can reproduce them offline if it
prefers local aggregation.

## IDE plugin integration snippet

```ts
// TypeScript — Cursor / Claude Code / Codex-style plugin
type TurnOutput = {
  scores: Record<string, number | null>;
  session_signals: Record<string, unknown> | null;
  file_attribution: Record<string, unknown> | null;
  risk_band: string | null;
  recommendation: string | null;
};

class SessionRollup {
  private turns: TurnOutput[] = [];

  record(output: TurnOutput) {
    this.turns.push({
      scores: output.scores,
      session_signals: output.session_signals,
      file_attribution: output.file_attribution,
      risk_band: output.risk_band,
      recommendation: output.recommendation,
    });
  }

  async refresh(apiKey: string, endpointId: string) {
    const body = {
      input: {
        action: "rollup",
        turns: this.turns,
        heatmap_format: "data",
      },
    };
    const r = await fetch(
      `https://api.runpod.ai/v2/${endpointId}/runsync`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${apiKey}`,
        },
        body: JSON.stringify(body),
      },
    );
    const j = await r.json();
    return j.output?.rollup ?? j.output; // RunPod vs. HTTP route shape
  }
}
```

## Back-pressure and limits

- The rollup runs on the asyncio loop (not the lane semaphore). It
  shares no budget with per-turn scoring.
- A dedicated timeout (`LATENCE_TRACE_ROLLUP_REQUEST_TIMEOUT_S`,
  default 0.25 s) bounds pathological inputs. The transform is O(n);
  10 000 turns complete in a few ms on a single CPU.
- Wire cost scales with the number of turns and any embedded
  `file_attribution` payloads. For a typical 50-turn session the
  request is ≤ 50 KB.

## Also see

- [`docs/architecture.md`](architecture.md) — overall service design.
- [`docs/heatmap.md`](heatmap.md) — structured heatmap + HTML fragment.
- [`docs/session_semantics.md`](session_semantics.md) — the caller-
  portable session-state protocol that produces the per-turn signals.
