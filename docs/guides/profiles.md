# Pareto-optimal default profiles

`latence-trace` ships three default configurations: `fast`, `balanced`
(default) and `quality`. Each profile is a **single environment
variable**, picks a Pareto-optimal point on the
quality-vs-latency-vs-VRAM frontier, and reuses the standalone
service's standard env-driven configuration surface so any explicit
override the operator exports always wins.

If you don't care about the trade-off matrix: leave it at `balanced`.

## TL;DR

```bash
# Default
latence-trace-server

# Lowest p95 SLO
latence-trace-server --profile fast       # or LATENCE_TRACE_PROFILE=fast

# Maximum coverage
latence-trace-server --profile quality

# Opt out: no profile, your env vars only
latence-trace-server --profile none
```

## Profile matrix

Numbers are from the offline 8-lane sweep documented in
[`research/triangular_maxsim/reports/profile_pareto.md`](../../research/triangular_maxsim/reports/profile_pareto.md)
on a single A5000 (24 GB), batch size 1, with the multilingual
ModernColBERT encoder loaded in `bf16`.

| | `fast` | `balanced` (default) | `quality` |
|---|---|---|---|
| NLI peer | off | mDeBERTa multilingual | mDeBERTa multilingual |
| Cross-encoder reranker | off | off | `BAAI/bge-reranker-v2-m3` |
| Atomic-claim decomposition | off | off | on |
| Multi-premise concatenation | off | off | on |
| Semantic entropy peer | off | off | on (caller supplies samples) |
| **Internal min-pair F1** (10 EN + 1 DE strata) | 0.66 | **0.89** | 0.87 |
| **DE min-pair accuracy** | 100 % (literal-only) | **92 %** | 87 % |
| **External F1** (RAGTruth + HaluEval + FActScore macro) | 0.39 | **0.49** | 0.44 |
| Encode p95 | ~130 ms | ~50 ms | ~50 ms |
| Score p95 | ~62 ms | ~146 ms | ~145 ms |
| **Total p95** | **~157 ms** | ~192 ms | ~190 ms |
| Peak VRAM | ~330 MB | ~2.3 GB | ~4.5 GB |
| Headline | `groundedness_v2_no_nli` | `groundedness_v2` | `groundedness_v2` |
| Threshold artefact | `latence_trace/data/thresholds.fast.json` | `latence_trace/data/thresholds.balanced.json` | `latence_trace/data/thresholds.quality.json` |
| Fusion-weight artefact | `latence_trace/data/fusion_weights.fast.json` | `latence_trace/data/fusion_weights.balanced.json` | `latence_trace/data/fusion_weights.quality.json` |

The encode-p95 difference between `fast` and the NLI lanes is a JIT
warm-up artefact in the sequential sweep: `fast` is the first lane to
touch CUDA in each run and absorbs the kernel-build cost. Apples-to-apples
end-to-end, `fast` is still the lowest-total-p95 lane.

## What each preset overlays

`apply_profile()` materialises an environment overlay; every key it
sets is documented below. The loader **never overwrites** a value that
is already present in `os.environ`, so production tuning decisions
remain authoritative.

### `fast`

```bash
VOYAGER_GROUNDEDNESS_NLI_ENABLED=0
VOYAGER_GROUNDEDNESS_NLI_ATOMIC_CLAIMS=0
VOYAGER_GROUNDEDNESS_NLI_PREMISE_CONCAT=0
VOYAGER_GROUNDEDNESS_NLI_PREMISE_RERANKER_MODEL=         # (unset)
VOYAGER_GROUNDEDNESS_FUSION_W_CALIBRATED=0.0
VOYAGER_GROUNDEDNESS_FUSION_W_LITERAL=1.0
VOYAGER_GROUNDEDNESS_FUSION_W_NLI=0.0
VOYAGER_GROUNDEDNESS_FUSION_W_SEMANTIC_ENTROPY=0.0
VOYAGER_GROUNDEDNESS_FUSION_W_STRUCTURED=0.0
VOYAGER_GROUNDEDNESS_THRESHOLDS_PATH=…/thresholds.fast.json
```

Use when:
- Sub-200 ms p95 SLO
- Cheap "is it grounded at all?" filter ahead of a more expensive
  verifier
- High-traffic ingestion path that cannot afford to load NLI weights

When the response carries no extracted literals (dates, numbers,
units, IDs, URLs), the fuse helper returns `None` and the headline
classifier falls back to the calibrated reverse-context MaxSim score.
The thresholds artefact is calibrated for that exact behaviour.

### `balanced` (default)

```bash
VOYAGER_GROUNDEDNESS_NLI_ENABLED=1
VOYAGER_GROUNDEDNESS_NLI_ATOMIC_CLAIMS=0
VOYAGER_GROUNDEDNESS_NLI_PREMISE_CONCAT=0
VOYAGER_GROUNDEDNESS_NLI_PREMISE_RERANKER_MODEL=         # (unset)
VOYAGER_GROUNDEDNESS_FUSION_W_CALIBRATED=0.0
VOYAGER_GROUNDEDNESS_FUSION_W_LITERAL=0.0
VOYAGER_GROUNDEDNESS_FUSION_W_NLI=1.0
VOYAGER_GROUNDEDNESS_FUSION_W_SEMANTIC_ENTROPY=0.0
VOYAGER_GROUNDEDNESS_FUSION_W_STRUCTURED=0.0
VOYAGER_GROUNDEDNESS_THRESHOLDS_PATH=…/thresholds.balanced.json
```

Use when:
- Typical RAG QA serving
- You want the best general-purpose groundedness verifier the suite
  offers
- You can absorb ~190 ms p95 per request

The headline is the NLI aggregate; the literal channel is computed
for diagnostics but does not contribute to the fused score. This
matches the sweep winner on the 11-stratum minimal-pair fixture
(macro F1 0.89, with negation / role / number_swap pegged at 1.00).

### `quality`

```bash
VOYAGER_GROUNDEDNESS_NLI_ENABLED=1
VOYAGER_GROUNDEDNESS_NLI_ATOMIC_CLAIMS=1
VOYAGER_GROUNDEDNESS_NLI_PREMISE_CONCAT=1
VOYAGER_GROUNDEDNESS_NLI_PREMISE_RERANKER_MODEL=BAAI/bge-reranker-v2-m3
VOYAGER_GROUNDEDNESS_FUSION_W_CALIBRATED=0.0
VOYAGER_GROUNDEDNESS_FUSION_W_LITERAL=0.2
VOYAGER_GROUNDEDNESS_FUSION_W_NLI=0.8
VOYAGER_GROUNDEDNESS_FUSION_W_SEMANTIC_ENTROPY=0.0
VOYAGER_GROUNDEDNESS_FUSION_W_STRUCTURED=0.0
VOYAGER_GROUNDEDNESS_THRESHOLDS_PATH=…/thresholds.quality.json
```

Use when:
- Compliance / high-stakes outputs
- Long multi-premise contexts where atomic decomposition + concat
  earn their keep
- You can run an LLM ensemble (typically via vLLM-factory) and want
  to opt into the SE channel by overriding the env weight after
  re-running calibration with your ensemble provider attached

The fusion weights match the L7 sweep winner exactly
(`literal=0.2 / nli=0.8`), so the per-stratum F1 numbers in
[`fusion_weights.quality.json`](../../latence_trace/data/fusion_weights.quality.json)
apply to what the runtime actually ships. Semantic entropy ships at
`0.0` because the L7 sweep ran with `include_semantic_entropy=False`
on a synthetic ensemble that underrates SE; opting in is a deliberate
operator decision that requires re-calibration with your real LLM
ensemble (see "Reproducing the picks" below). The fuse helper
renormalises cleanly either way (`fuse_groundedness_v2`).

A regression guard
(`tests/test_profiles.py::test_preset_weights_match_calibration_artefacts`)
locks the runtime preset, the threshold artefact, and the fusion
sweep artefact in sync so any future drift fails CI immediately.

## Methodology

The 8-lane sweep evaluated:

- 11-stratum minimal-pair fixture (10 EN + 1 DE = 226 + 26 pairs at
  20 pairs/stratum), measuring paired-ranking accuracy
- 3 external benchmarks (RAGTruth QA / summarisation / data2text,
  HaluEval QA / dialogue / summarisation, FActScore biography), 20
  samples per sub-stratum, measuring F1 against published labels

The combined score `0.4 · macro_internal + 0.6 · macro_external_f1`
biases toward the harder real-world benchmarks. The selected three
profiles are the Pareto-undominated lanes (`L0` and `L1`) plus the
full-stack `L7` lane that callers will need once they wire a real
LLM ensemble for semantic entropy.

VRAM was capped at 24 GB (single A5000); the entire 8-lane sweep peaks
at 4.5 GB, leaving full headroom for batch > 1 and a future
LLM-as-judge claim verifier inside the same pod.

## Reproducing the picks

```bash
export VOYAGER_GROUNDEDNESS_RAGTRUTH_DIR=/workspace/datasets/ragtruth
export VOYAGER_GROUNDEDNESS_HALUEVAL_DIR=/workspace/datasets/halueval
export VOYAGER_GROUNDEDNESS_FACTSCORE_DIR=/workspace/datasets/factscore

# Re-run the 8-lane harness sweep
python -m research.triangular_maxsim.run_profile_sweep \
    --pairs-per-stratum 20 \
    --max-external-per-stratum 20 \
    --out-dir research/triangular_maxsim/reports

# Re-calibrate per-profile thresholds with the shipped fusion weights
python -m research.triangular_maxsim.calibrate_profile_thresholds \
    --pairs-per-stratum 30 \
    --out-dir latence_trace/data

# Re-run the per-profile fusion-weight grid sweep
python -m research.triangular_maxsim.sweep_profile_fusion_weights \
    --pairs-per-stratum 30 --steps 5 \
    --out-dir latence_trace/data
```

Each command writes its own JSON artefacts under
`research/triangular_maxsim/reports/` and `latence_trace/data/`; the
runtime loader picks them up via the env paths set by `apply_profile`.

## Embedding the loader

If you embed `latence-trace` inside another FastAPI process instead of
running the standalone server, call `apply_profile` once at startup -
ideally before constructing your `GroundednessService`:

```python
from latence_trace.api.service import (
    GroundednessService,
    apply_profile,
)

apply_profile("balanced")           # or "fast" / "quality" / None
service = GroundednessService(device="cuda")
```

`apply_profile` is non-destructive: any environment variable the host
process already set is left untouched and surfaced via the returned
`ProfileApplication.skipped` map for diagnostics.

## See also

- [`profile_pareto.md`](../../research/triangular_maxsim/reports/profile_pareto.md)
  — the underlying 8-lane sweep and Pareto picks
- [`multilingual.md`](multilingual.md) — bilingual EN/DE configuration
- [`beta-overview.md`](beta-overview.md) — Beta scope and limitations
- [`../benchmarks.md`](../benchmarks.md) — head-line benchmark numbers
- [`../algorithm-audit.md`](../algorithm-audit.md) — per-channel
  ablations and verdict
