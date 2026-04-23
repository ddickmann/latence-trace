# Code lane v3 — design & integration

The **code lane** is the second scoring lane shipped by
`latence-trace`. It sits next to the original RAG lane (retrieval-
augmented-generation groundedness + context coverage for enterprises)
and targets **coding-agent harnesses**: Claude Code, Cursor, OpenAI
Codex, OpenCode, and any CLI/IDE plugin that asks a model to
generate code against a growing context window of files.

This document describes the v3 architecture landed in the
"code-lane quality boost" sprint.

## Why a dedicated lane?

Coding agents fail in patterns that differ from enterprise Q&A:

- They invent APIs (`photon_labs.Tracer.log_span()`) that do not
  exist in any file in context (phantom imports / classes).
- They drift on identifier kwargs — `service.calibrate(thresh=…)`
  when the context only has `service.calibrate_threshold(value=…)`.
- They accumulate context over many turns without ever using
  portions of it again, blowing the budget on dead files.
- They sometimes copy structure from context but change semantics —
  a soft hallucination that embedding-only cosine cannot catch.

The RAG lane's ColBERT MaxSim + literal guard picks up the first
class of issue but is blind to the other three. The code lane adds
deterministic AST checks, a calibrated logistic composite, an
ambiguity-gated NLI cascade, and multi-turn session signals so IDE
plugins can render a live dashboard of agent quality.

## Architecture

```
POST /groundedness (scoring_mode=code)
  │
  ├─► GroundednessService._score_code
  │     │
  │     ├─► GPUScorer (MaxSim + per_token p10 + owner_count)
  │     │
  │     ├─► AstSymbolExtractor (tree-sitter: py/ts/js/go/rust)
  │     │     └─► ast_literal_drift_count, ast_phantom_symbol_count
  │     │
  │     ├─► LiteralNovelty (per-identifier first-match unit score)
  │     │
  │     ├─► LogisticComposite (calibrated v3 artefact)
  │     │     └─► composite_phantom_probability + verdict
  │     │
  │     ├─► NLICascade (if composite in [0.65, 0.90])
  │     │     └─► nli_contradiction_prob_max
  │     │
  │     ├─► SemanticEntropy (if verification_samples >= 3)
  │     │
  │     └─► FileAttribution (owner_share + reason_codes)
  │
  └─► response + code_lane_diagnostics + per-turn log event
```

## Signals added in v3

| Signal | What it catches | Cost |
| --- | --- | --- |
| `literal_novelty_min` | Identifier grounded only in low-score unit | free |
| `ast_phantom_symbol_count` | Invented import / class / function | 3-5 ms tree-sitter |
| `ast_literal_drift_count` | Wrong kwarg / method name | 3-5 ms tree-sitter |
| `nli_contradiction_prob_max` | Subtle semantic drift | 15-25 ms (only on ambiguous turns) |
| `composite_phantom_probability` | Calibrated fused score | <1 ms |
| File-level `reason_codes` | Why a file is flagged | free |
| `query_owner_share` | File ignored by response but relevant to query | <1 ms |

## Cascade behaviour

The NLI cascade fires **only** when the pre-NLI composite phantom
probability lands inside the ambiguity band. Defaults live in
`CodeLaneConfig`:

```python
CodeLaneConfig(
    nli_lower_band = 0.65,  # below = clear phantom, no cascade
    nli_upper_band = 0.90,  # above = clear grounded, no cascade
)
```

This means the expensive NLI round-trip only pays off on turns where
the calibrated composite is genuinely uncertain — typically 20-35 %
of traffic. Operators can observe the cascade fire rate at
`latence_trace_code_lane_cascade_fires_total{lane="code"}` and widen
or tighten the band from the config.

See `research/triangular_maxsim/coding/artifacts/v3_cascade_latency_bench.md`
for the measured p95 — <10 ms on the deterministic benchmark,
well inside the 150 ms SLO.

## Client-side session state

The server is stateless: every turn's score is independent. The
client reconstructs session-scoped quality from the stream of
`groundedness_turn` log events (see `docs/code_lane_turn_event.md`
for the wire schema).

Key client-side computations:

- **EMA of `owner_share` per file** — drives the "dead for N turns"
  eviction recommendation. Default half-life: 5 turns.
- **Rolling z-score of `per_token_p10`** — local drift detector.
  Flag deviations > 2σ from the session baseline.
- **Cascade-fire density** — visualise when the agent is skating on
  thin ice (e.g. >50% of recent turns triggering NLI means the
  task is genuinely ambiguous).

The reference TypeScript helpers in `plugin_client/` are zero-dep
and copy-pasteable into any IDE plugin. See
`plugin_client/README.md` for the integration recipe.

## Per-plugin integration

### Claude Code

Claude Code already streams tool-call invocations. Hook the
`/groundedness` call after each assistant turn that produced a code
fence. Wire the returned `code_lane_diagnostics.file_attribution`
into the left-hand file tree to grey out files that are dead for
N turns.

### Cursor

Cursor's composer surface exposes the response text before it is
applied. Fire `/groundedness` asynchronously with the active
conversation as `support_units[]`. If
`composite_phantom_verdict == true`, surface a banner and require
explicit confirmation before applying the edit.

### OpenAI Codex / OpenCode

Both harnesses emit intermediate drafts. Run a
`verification_samples` array of 3-5 sampled drafts through the
code lane to activate semantic entropy as a second hallucination
signal (orthogonal to MaxSim + NLI).

## See also

- `docs/code_lane_turn_event.md` — JSONL wire format spec.
- `docs/code_lane_performance.md` — per-signal latency budget.
- `docs/enterprise_rag_guide.md` — RAG lane documentation (unchanged
  by this sprint).
- `docs/coding_agent_guide.md` — copy-paste recipe for each plugin.
