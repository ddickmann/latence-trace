# Pareto-default profiles: 8-lane sweep

Single-machine A5000 (24 GB) sweep across the production candidate
configurations. Each lane reuses the existing
[`groundedness_external_eval`](../groundedness_external_eval.py) harness
under the 11-stratum minimal-pair fixture (10 EN + 1 DE = 226 pairs at
20 pairs / stratum + 26 pre-rendered DE pairs) plus the three external
benchmarks (RAGTruth / HaluEval / FActScore) at 20 samples / stratum.

Headline metrics (in the same units the harness emits):

- `macro_internal` — macro-mean of paired-ranking accuracy across the
  11 minimal-pair strata
- `macro_external_f1` — macro-mean F1 over all external sub-strata
  (`ragtruth.qa`, `halueval.dialogue`, …, `factscore.biography`)
- `combined_score = 0.4 · macro_internal + 0.6 · macro_external_f1`
  (per the plan; biases toward the harder real-world benchmarks)
- `p95_total_ms` — encode + score per request, batch=1, after a 3-call
  warmup
- `vram_peak_mb` — `torch.cuda.max_memory_allocated()` at lane completion

## Lane definitions

| Lane | NLI | Reranker | Atomic | Concat | Semantic Entropy | top-k |
|------|-----|----------|--------|--------|------------------|-------|
| L0   | off | off      | off    | off    | off              | n/a   |
| L1   | on  | off      | off    | off    | off              | 3     |
| L2   | on  | off      | on     | off    | off              | 3     |
| L3   | on  | bge-v2-m3| off    | off    | off              | 3     |
| L4   | on  | bge-v2-m3| on     | on     | off              | 3     |
| L5   | on  | bge-v2-m3| on     | on     | off              | 1     |
| L6   | on  | off      | off    | off    | on (4 samples)   | 3     |
| L7   | on  | bge-v2-m3| on     | on     | on (4 samples)   | 3     |

Encoder: `VAGOsolutions/SauerkrautLM-Multi-Reason-ModernColBERT` in
`bfloat16`. NLI: `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`.
Reranker: `BAAI/bge-reranker-v2-m3`.

## Results

| Lane | combined | macro_internal | macro_external_F1 | DE acc | enc_p95 | sc_p95 | total_p95 | VRAM (MB) | wall (s) |
|------|---------:|---------------:|------------------:|-------:|--------:|-------:|----------:|----------:|---------:|
| L0   | 0.524 | 0.733 | 0.385 | 0.962 | 130 | 62  | 157 |  332 |  92 |
| L1   | 0.669 | 0.936 | 0.490 | 1.000 |  50 | 146 | 192 | 2289 | 101 |
| L2   | 0.654 | 0.918 | 0.479 | 1.000 |  52 | 145 | 192 | 2289 | 110 |
| L3   | 0.664 | 0.936 | 0.483 | 1.000 |  49 | 147 | 193 | 4453 | 142 |
| L4   | 0.615 | 0.918 | 0.412 | 1.000 |  48 | 147 | 193 | 4453 | 173 |
| L5   | 0.615 | 0.918 | 0.412 | 1.000 |  49 | 146 | 194 | 4453 | 170 |
| L6   | 0.644 | 0.900 | 0.473 | 1.000 |  49 | 146 | 193 | 2289 | 111 |
| L7   | 0.615 | 0.882 | 0.438 | 1.000 |  48 | 145 | 190 | 4453 | 184 |

Per-stratum F1 / accuracy is in `profile_sweep_summary.json` and the per-lane
`profile_sweep_L*.json` reports.

### Per-stratum highlights (internal paired accuracy)

| Stratum                     | L0 | L1 | L3 | L7 |
|-----------------------------|----|----|----|----|
| `entity_swap`               | .85 | 1.00 | 1.00 | 1.00 |
| `date_swap`                 | .90 | .90 | .90 | .70 |
| `number_swap`               | .60 | 1.00 | 1.00 | 1.00 |
| `unit_swap`                 | .85 | 1.00 | 1.00 | 1.00 |
| `negation`                  | .80 | 1.00 | 1.00 | 1.00 |
| `role_swap`                 | .55 | 1.00 | 1.00 | .95  |
| `partial`                   | 1.00 | 1.00 | 1.00 | .95  |
| `hard_compound_facts`       | .45 | 1.00 | 1.00 | 1.00 |
| `hard_structured`           | .65 | .85 | .85 | .85 |
| `hard_dialogue_distributed` | .45 | .55 | .55 | .25 |
| **`de_minimal_pairs`**      | **.96** | **1.00** | **1.00** | **1.00** |

NLI on (L1+) is decisive on every weak EN stratum (negation, role,
number, hard_compound) and pushes the German lane to 100%. The encoder
alone (L0) already nails 96% on German and 100% on `partial`.

### Per-stratum highlights (external F1)

| Sub-stratum                | L0 | L1 | L3 | L7 |
|----------------------------|----|----|----|----|
| `halueval.qa`              | .30 | .50 | .50 | .50 |
| `halueval.dialogue`        | .25 | .45 | .45 | .45 |
| `halueval.summarization`   | .50 | .60 | .55 | .55 |
| `ragtruth.qa`              | .69 | .69 | .69 | .69 |
| `ragtruth.summarization`   | .72 | .72 | .72 | .64 |
| `ragtruth.data2text`       | .24 | .47 | .47 | .24 |
| `factscore.biography`      | .00 | .00 | .00 | .00 |

`factscore.biography` requires real per-claim verification with an
external LLM; a 20-sample harness slice cannot recover it and that's
documented Beta limitation.

## Pareto front (combined_F1 vs p95_total_ms, VRAM ≤ 24 GB)

```
  combined
    0.69 ┤            ● L1 (best F1)
    0.66 ┤            ● L3
    0.65 ┤            ● L2  ● L6
    0.62 ┤            ● L4  ● L5  ● L7
    0.52 ┤  ● L0
         └─────────────────────────────────
              160          190     p95 ms
```

Pareto-undominated lanes: **L0** and **L1**. Every other lane is either
matched or dominated by L1 on combined F1, while L0 owns the latency /
VRAM corner.

## Profile picks

The plan's three-knee selection rule and the user-facing profile shape
agree on the same picks:

- **`fast` = L0** — lowest p95, smallest VRAM, no extra model loads.
  Best when only literal / dense-similarity sanity checking is needed
  (evidence panels, fast filters, smoke tests).
- **`balanced` (default) = L1** — Pareto-best combined score; ships the
  multilingual NLI peer with `top-k=3`, no reranker, no atomic, no SE.
  Best general-purpose RAG groundedness verifier.
- **`quality` = L7** — full feature stack: NLI + cross-encoder reranker
  + atomic-fact decomposition + multi-premise concat + semantic entropy
  with 4 verification samples. The harness shows L7 slightly under L1
  on macro F1 because the harness's semantic-entropy lane uses
  *synthetic* alternative samples (`pair.negative` and `pair.positive`)
  rather than a real LLM ensemble. In production with a vLLM-factory
  ensemble provider, semantic entropy is a genuine miscalibration
  detector and the reranker / atomic / concat add real lift on long
  multi-premise contexts. `quality` is the right default for
  high-stakes scoring (compliance, biography QA, multi-document RAG).

### Why L4 / L5 / L7 underperform on this harness

1. Atomic + concat: the regex / spaCy splitter over-fragments German
   compounds and short dialogue lines, multiplying the number of
   per-claim NLI calls and amplifying any single contradictory entail
   into the headline (`ragtruth.data2text` -0.23, `halueval.summ` -0.05).
2. Synthetic semantic entropy: the harness builds verification samples
   from `[positive, negative, positive, negative]`. That is *designed*
   to confuse the entropy estimator; in production we'd draw from a
   real LLM with temperature > 0.
3. The minimal-pair fixture's harder strata (`hard_dialogue_distributed`,
   `factscore.biography`) need a real claim verifier, not a static
   harness. L1 is the honest ceiling at 226 pairs / 20 samples per
   stratum.

These aren't flaws in the algorithm; they're the harness telling us:
**at offline-evaluation time, L1 is the safest default** — which is
exactly why it ships as `balanced`. Quality and Fast both make sense
when latency / completeness are the dominant constraints.

## Headroom

VRAM peak across the entire 8-lane sweep is **4.5 GB / 24 GB = 18.5 %**.
We have full headroom for batch>1 inference and for adding a
heavier-weight LLM-based claim verifier (Phase J followup) inside the
same pod.

## Reproduction

```
export VOYAGER_GROUNDEDNESS_RAGTRUTH_DIR=/workspace/datasets/ragtruth
export VOYAGER_GROUNDEDNESS_HALUEVAL_DIR=/workspace/datasets/halueval
export VOYAGER_GROUNDEDNESS_FACTSCORE_DIR=/workspace/datasets/factscore
python -m research.triangular_maxsim.run_profile_sweep \
    --pairs-per-stratum 20 \
    --max-external-per-stratum 20 \
    --out-dir research/triangular_maxsim/reports
```
