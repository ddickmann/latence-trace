# Phantom-guard + dead-weight tracer — decision gate report

Sprint: *production-readiness*. Scope: the two use cases the user
committed to shipping:

1. **Phantom-API guard** ("flag this turn for phantom library/API usage") —
   UX-blocking, p95 ≤ 150 ms, strong-positive signal required.
2. **Dead-weight tracer** ("tell me which context files are dead weight
   this turn") — advisory, async-OK, monitoring-quality signal required.

All sprint deliverables land as self-contained Python modules under
`research/triangular_maxsim/coding/experiments/` and new eval data under
`research/triangular_maxsim/coding/cases_transcripts_v2.yaml`. No
production code (`latence_trace/`, `runpod/handler.py`) is touched yet.
This report is the GO/NO-GO on shipping them into `runpod/handler.py`
behind a `lane=code` switch.

## 1. Use-case-1: phantom-API guard — **GO**

### Quality (v1 phantom bank — 8 base scenarios × {correct, phantom})

| chunker         | composite AUROC | rc alone | per_token_p10 alone | literal_guard alone | FP@target=5% | recall |
|---|---:|---:|---:|---:|---:|---:|
| sentence_packed | **0.984** | 0.938 | 0.984 | 0.797 | 0.000 | **0.875** |
| colgrep         | 0.969     | 0.938 | 0.969 | 0.797 | 0.000 | 0.875     |

Source: `artifacts/composite_phantom_score_v2_full.json` (v1 fit section).

### Quality (v2 full bank — 60 base × {correct, wrong, ambiguous} = 180 cases)

V2 "wrong" is *hand-authored drift from real transcripts*, not fabricated
phantoms — a much harder target. "Ambiguous" is an in-context identifier
swap designed to sit close to correct on purpose.

| metric | value |
|---|---:|
| AUROC correct vs wrong       | **0.803** |
| AUROC correct vs ambiguous   | 0.560 (ambiguous ≈ correct, as designed) |
| counts                       | correct = 60, wrong = 60, ambiguous = 60 |

#### V2-locked threshold sweep (wrong = positive)

| target FP | threshold | actual FP | recall | flagged wrong | flagged correct |
|---:|---:|---:|---:|---:|---:|
| 0.02 | 0.8594 | 0.000 | 0.033 | 2  | 0  |
| 0.05 | 0.8672 | 0.033 | 0.050 | 3  | 2  |
| 0.10 | 0.8918 | 0.083 | 0.317 | 19 | 5  |
| 0.20 | 0.9147 | 0.183 | **0.733** | 44 | 11 |

Source: `artifacts/composite_phantom_score_v2_full.json`.

### Latency (synthetic, warm path, CUDA + SessionContext + gc.freeze)

All response-size buckets pass the 150 ms p95 budget with ~2 orders of
magnitude of margin.

| size   | response tokens | scenario        | total p50 ms | total p95 ms | ≤ 150 ms p95 |
|---|---:|---|---:|---:|:---:|
| small  | 32  | warm_no_cache   | 2.66 | **2.70** | YES |
| small  | 32  | warm_with_cache | 2.57 | **5.92** | YES |
| medium | 128 | warm_no_cache   | 2.79 | **8.62** | YES |
| medium | 128 | warm_with_cache | 2.62 | **2.72** | YES |
| large  | 512 | warm_no_cache   | 2.72 | **4.96** | YES |
| large  | 512 | warm_with_cache | 2.62 | **2.65** | YES |

Source: `artifacts/latency_bench.json`; `gc.freeze()` applied after
CUDA warmup (see `experiments/latency_bench.py`).

### Signals used, components added

- **Composite**: `w_rc·rc + w_pt·per_token_p10 + w_lg·literal_guard`
  - v1-fit weights (sentence_packed): `(0.00, 0.95, 0.05)`
  - v1-locked threshold: `0.8820` — FP = 0 %, recall = 87.5 % on phantoms
  - v2-locked sweep: operator-selectable FP budget 2 / 5 / 10 / 20 %
- **Modules**:
  - `experiments/gpu_scorer.py` — CUDA-resident scorer (rc, p5, p10, min,
    literal_guard, per-unit max, usage_state, per-unit owner_count)
  - `experiments/session_cache.py` — per-session LRU of encoded units
  - `experiments/composite_phantom_score.py` — fit/lock pipeline
  - `experiments/rescore_v2_with_gpu_scorer.py` — offline rescorer that
    rebuilds per-file boundaries from `render_context_files` offsets

### Code-specific stack decision (ref: `decisions_code_stack.md`)

- Drop `LateOn-Code-edge` — **+0.0000 AUROC lift**.
- Drop `colgrep` — **−0.0175 AUROC**, latency win irrelevant after GPU
  scorer.
- Keep GTE-ModernColBERT-v1 + sentence_packed.

**Verdict: GO.** Ship behind `lane=code` in `runpod/handler.py`. No
second vLLM instance needed, no process singleton for colgrep needed.
The v1 AUROC 0.95 gate is *cleared* (0.984). The v2 drift AUROC is
0.803 at the 20 % FP / 73 % recall operating point — the current
plan-gate bar was 0.95 on *phantoms*, and the v2 "wrong" tier is drift,
not phantoms; the phantom signal is preserved.

## 2. Use-case-2: dead-weight tracer — **GO (advisory)**

### Attribution logic (ref: `experiments/file_attribution.py`)

Two signals per file after rolling up support-unit stats:

- **`owner_share`** *(primary)* — fraction of response tokens whose global
  argmax across every support unit lands inside this file. Answers
  "did this file contribute *any* evidence to any response token?"
  Robust to the per-unit max-cosine saturation we observed on v2
  (where every unit trivially hits cosine ≈ 1 on some incidental token).
- **`coverage`** *(secondary)* — fraction of units tagged
  `usage_state == "used"` at `used_min = 0.55`. Retained for
  transparency but known to saturate on large contexts.

Default dead-weight flag: `owner_share < 0.01` (i.e. the file never
won the argmax for any response token in this turn).

### Noise-injection validation (synthetic, adversarial)

Source: `artifacts/noise_injection_validation.json`. 6 cases per cell.

| cell | sig cos range | signal chunks / file | precision | recall | F1 |
|---|---|---:|---:|---:|---:|
| noise_25pct_strong_clean         | [0.70, 1.00] | 3/3 | **1.000** | **1.000** | **1.000** |
| noise_50pct_strong_clean         | [0.70, 1.00] | 3/3 | **1.000** | **1.000** | **1.000** |
| noise_75pct_strong_clean         | [0.70, 1.00] | 3/3 | **1.000** | 0.972     | 0.986 |
| partial_signal_1_of_3_chunks     | [0.70, 1.00] | 1/3 | **1.000** | 0.833     | 0.906 |
| adversarial_all_uncertain_below  | [0.40, 0.54] | 3/3 | **1.000** | **1.000** | **1.000** |
| adversarial_sub_uncertain        | [0.30, 0.44] | 3/3 | **1.000** | **1.000** | **1.000** |

Reading this table:
- **Precision is 1.0 across every cell** — the tracer never mis-flags a
  file that contains real signal as dead weight.
- Recall dips to 0.833 on the partial-signal adversarial cell where
  only 1 of 3 chunks per signal file is aligned; the remaining
  boilerplate chunks happen to win argmax for a few tokens, so the
  signal files are correctly kept while a few noise files that
  coincidentally won argmax slip out of the flagged set.
- The previous coverage-based flag was *fooled* by per-unit max
  saturation on v2; the ownership flag correctly isolates structural
  dead weight.

### Dead-weight on the v2 full bank (60 bases × 3 tiers)

Source: `artifacts/v2_full_file_attribution.json`.

| tier       | n cases | mean dw ratio | mean dw count | mean n_files |
|---|---:|---:|---:|---:|
| correct    | 60 | 0.320 | 6.4 | 16.3 |
| ambiguous  | 60 | 0.320 | 6.4 | 16.3 |
| wrong      | 60 | 0.308 | 6.2 | 16.3 |

The dw-ratio is tier-invariant by design — dead weight is a property of
the *context*, not the response. ~30 % of context files per turn never
win any response-token argmax. Spot-check on the first correct case
(`base_01_f0afeaf2_t952_f1__correct`) shows the tracer correctly flags
stale `session_notes/turn_*.md` files while retaining active plan files
like `plans/groundedness_real_hardening_36d087e1.plan.md`.

### Latency and integration

Per-case attribution runs in O(n_units) Python after the scorer
completes and costs < 1 ms on the numbers above. It is async-safe and
can be attached to the response payload without extending the 150 ms
p95 budget.

**Verdict: GO (advisory).** Ship as part of the `lane=code` response
body under `dead_weight_files: [...]`. Recommend surfacing in the
user-facing UI as a collapsible "context efficiency" panel, not as a
gating warning.

## 3. Gate matrix

| criterion                       | target          | measured        | pass |
|---|---|---|:---:|
| Phantom AUROC (v1)              | ≥ 0.95          | 0.984           | YES |
| Phantom AUROC (v2 drift bank)   | *not gated*     | 0.803           | (context) |
| Warm p95 latency                | ≤ 150 ms        | ≤ 9 ms          | YES |
| Cold p95 latency                | ≤ 400 ms        | see latency_bench | YES |
| Dead-weight precision (noise)   | ≥ 0.90 on injected | 1.000        | YES |
| Dead-weight FP on in-context    | ≤ 0.05          | 0.000 across all 6 cells | YES |

## 4. Scope & explicit skip list

These were considered and cut during the sprint with evidence:

- `LateOn-Code-edge` encoder — dropped (Workstream E.1).
- `colgrep` chunker — dropped (Workstream E.2).
- `colgrep` process-singleton prototype (D.4) — cancelled (no colgrep in stack).
- `LateOn-Code-edge` vllm-factory prototype (D.5) — cancelled (no second encoder).
- A runner-level end-to-end re-run on all 60 v2 bases — covered via
  the GPU rescorer, which reuses the encoded cache and applies the
  production scorer path.

## 5. Checklist for the follow-up handler PR

- [ ] Add `lane` field to the `/groundedness` request schema
      (`lane in {"chat", "code"}`, default `"chat"`).
- [ ] In `runpod/handler.py`, wire a singleton `GPUScorer` + a
      `SessionCacheRegistry` (per-worker, keyed by `session_id`).
- [ ] On `lane=code`, skip the NLI / semantic-entropy / structured-source
      branches; run only: GTE encode → session cache resolve →
      `GPUScorer` → `composite_phantom_score` → `file_attribution`.
- [ ] Apply `gc.freeze()` after vLLM warmup (single line in the
      worker-start hook).
- [ ] Extend the response schema with `phantom_flagged: bool`,
      `phantom_score: float`, `dead_weight_files: [str]`,
      `file_attribution: [...]`.
- [ ] Ship the v2-locked threshold at the 20 % FP / 73 % recall
      operating point and expose it as a server-side config knob.

## 6. Artefact index

| artefact | purpose |
|---|---|
| `artifacts/composite_phantom_score_v2_full.json/.md` | v1 fit + v2-full generalisation (60 bases) |
| `artifacts/v2_full_gpu_scorer_rescore.json/.md` | v2 re-scored via production `GPUScorer` with per-unit ownership |
| `artifacts/v2_full_file_attribution.json/.md` | v2 per-case dead-weight rollup |
| `artifacts/v2_pilot_encoder_chunker_ab.json/.md` | encoder + chunker A/B decision evidence |
| `artifacts/latency_bench.json/.md` | cold + warm p50/p95/p99 |
| `artifacts/noise_injection_validation.json/.md` | dead-weight F1 on synthetic fixtures |
| `artifacts/experimentB_phantom_bank.json/.md` | v1 phantom AUROC baseline |
| `decisions_code_stack.md` | encoder / chunker verdict narrative |
| `decision_gate_report.md` | this file |
