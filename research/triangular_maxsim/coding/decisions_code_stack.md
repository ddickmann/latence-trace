# Decisions: code-specific stack for the phantom-guard lane

Owner: `latence-trace` research. Sprint: *phantom-guard + dead-weight
production-readiness*.

## TL;DR

| component | status | reason |
|---|---|---|
| `lightonai/GTE-ModernColBERT-v1` (GTE encoder) | **keep** | sole primary encoder, already in production; drives every signal used by the composite |
| `lightonai/LateOn-Code-edge` (code encoder) | **drop from request path** | `fuse_max(gte, lateon)` gives **+0.0000 AUROC** over `gte_only` on the v2 pilot (sentence_packed and colgrep); every signal saturates above the GTE reverse_context ceiling |
| `colgrep` code chunker | **drop from request path** | vs. `sentence_packed`: **−0.0175 AUROC** on v2 reverse_context, **−0.015 AUROC** on the v1 phantom-bank composite; the modest latency win is irrelevant once the production path moves to the `GPUScorer` |
| `sentence_packed` chunker | **keep** | higher AUROC on both v1 and v2, handles every language (no tree-sitter dep), simpler ops |
| `GPUScorer` (slim, this sprint) | **new — ship as request-path scorer** | ≤3 ms p50 / ≤6 ms p95 total on CUDA, pairs with `SessionContext` for 86 % warm-hit-rate |
| `SessionContext` (per-session LRU) | **new — ship alongside GPUScorer** | keeps warm encode cost at ~0.04 ms p95 vs. full re-encode every turn |
| `composite_phantom_score` | **new — ship as request-path scoring rule** | weights + threshold locked on v1 (AUROC 0.984), re-locked per operating point on v2 |
| `file_attribution` | **new — ship as dead-weight tracer** | F1 = 1.000 on synthetic noise-injection cells with genuine-noise ground truth |

## Evidence

### E.1 — encoder A/B (gte_only vs fuse_max)

Source: `artifacts/v2_pilot_encoder_chunker_ab.json`, 20 base scenarios ×
3 tiers = 60 cases.

| chunker | scorer | AUROC | AUROC c-vs-w | separation | sat@0.95 |
|---|---|---:|---:|---:|---:|
| sentence_packed | `gte_only` | 0.7875 | 0.7875 | +0.0059 | 1.0000 |
| sentence_packed | `fuse_max` | 0.7875 | 0.7875 | +0.0059 | 1.0000 |
| colgrep | `gte_only` | 0.7700 | 0.7700 | +0.0062 | 1.0000 |
| colgrep | `fuse_max` | 0.7700 | 0.7700 | +0.0062 | 1.0000 |

Stacking lift `fuse_max − gte_only` on v2:

| chunker | AUROC lift | unused-delta lift |
|---|---:|---:|
| sentence_packed | **+0.0000** | +0.0000 |
| colgrep         | **+0.0000** | +0.0000 |

On the v1 phantom bank (see `composite_phantom_score.md`) the GTE-only
`per_token_p10` already hits **AUROC 0.984** (sentence_packed) and
**0.969** (colgrep). `fuse_max` adds nothing; `fuse_min` and `fuse_mean`
were similarly neutral in prior experiments (see
`experimentB_phantom_bank.md`).

**Decision:** ship `gte_only`. Skip any vLLM-online path for
`LateOn-Code-edge` (Workstream D.5 cancelled).

### E.2 — chunker A/B (sentence_packed vs colgrep)

Source: same pilot as E.1.

| metric | sentence_packed | colgrep | delta |
|---|---:|---:|---:|
| AUROC grounded vs ungrounded | 0.7875 | 0.7700 | **−0.0175** |
| AUROC correct vs wrong       | 0.7875 | 0.7700 | **−0.0175** |
| Mean support units / case    | 135.4 | 113.0 | colgrep uses 17 % fewer units |
| Pipeline p95 ms (research path, not request path) | 2801 | 1897 | colgrep ~32 % faster |

On v1 phantom bank the composite AUROC drops from **0.984**
(sentence_packed) to **0.969** (colgrep). colgrep is a net *loss* on
accuracy for the phantom-guard decision.

The research-pipeline latency delta is irrelevant to the production
request path, which uses `GPUScorer` directly on pre-encoded units
(total p95 ≤ 6 ms, see `latency_bench.md`). Chunker cost is paid at
encode time, cached in the session, and amortised across every turn.

**Decision:** ship `sentence_packed`. Skip the `colgrep` singleton
prototype (Workstream D.4 cancelled).

### Why the original "keep colgrep/lateon as orthogonal signals"
### hypothesis didn't survive contact with data

- **LateOn-Code-edge is a code-specialised ColBERT.** It does improve
  per-token MaxSim on raw code, but the request lane already scores the
  response against *both* raw-code chunks and surrounding prose (from
  `sentence_packed`), so the GTE signal captures the same alignment.
- **`reverse_context` saturates.** With chunking at 256 tokens, every
  tier (correct / ambiguous / wrong) sits in [0.97, 0.99] — there is
  almost no room for a second encoder to lift the mean.
- **`per_token_p10` is the actual phantom signal.** It measures the
  worst-supported 10 % of response tokens — that's where phantom API
  names show up. GTE-only on `per_token_p10` already hits 0.984.
- **Drift = semantic continuity.** v2 "wrong" responses are genuinely
  close to the context (hand-authored from real transcripts). No encoder
  choice rescues a composite that saturates at the top end.

## Stack that ships

```
┌──────────── Client (agent harness) ─────────────┐
│ POST /groundedness  lane=code               │
└────────┬────────────────────────────────────────┘
         │
    ┌────▼─────────────────────────────────────────────────┐
    │ Handler (runpod/handler.py, singleton ManagedVllm)   │
    │  - GTE-ModernColBERT-v1 via vllm-factory (existing) │
    │  - sentence_packed chunker (existing)               │
    │  - SessionContext LRU (NEW, per session_id)          │
    │  - GPUScorer (NEW, CUDA-resident)                    │
    │  - composite_phantom_score (NEW)                     │
    │  - file_attribution (NEW)                            │
    └──────────────────────────────────────────────────────┘
```

No second vLLM instance, no tree-sitter process singleton. The request
body gains a `lane` field that selects:

- `lane: "chat"` (default) → existing heavy path (NLI + semantic entropy)
- `lane: "code"` → `GPUScorer` + `composite_phantom_score` +
  `file_attribution`, returns `{phantom_flagged, dead_weight_files,
  latency_ms}` within the 150 ms p95 budget.

The two lanes share: the vLLM GTE server, the session-level embedding
cache, the literal-token extractor, and the dead-weight threshold.

## Open items (deliberately out of scope for this sprint)

- Shipping the lane switch in `handler.py`. This doc is the decision
  record; the implementation is a follow-up PR scoped to the handler
  and the service schema.
- Code-language-specific literal extraction (current `extract_literals`
  is language-agnostic and works well enough for v1/v2).
- A code-ColBERT-based re-ranker for the response-generation side of
  the agent — orthogonal to the guard.
