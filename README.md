# latence-trace

> Calibrated, auditable groundedness scoring for RAG and evidence-bearing LLM
> outputs. **Multilingual: English + German out of the box.** Part of the
> latence.ai product family.

`latence-trace` is the standalone Groundedness Tracker (Beta) extracted from
the `voyager-index` retrieval engine. It scores how well an LLM response is
grounded in its supporting context, returns auditable per-claim evidence, and
classifies every output into a calibrated `green` / `amber` / `red` risk band.

> **New here?** Start with the end-to-end tutorial:
> [`docs/guides/tutorial.md`](docs/guides/tutorial.md) — covers boot,
> first request, the three premise lanes, the per-token heatmap, the
> retrieval-coverage observability metric, profiles, and integrations
> (HTTP / Python SDK / MCP / OpenAI tools).

## Why

Open-source LLM observability tools either
- emit a single opaque score from a 7-70B "judge" model, or
- run a proprietary hosted service against your private context.

`latence-trace` runs locally or as a sidecar, in ~100 ms p95, with five
explainable channels and per-stratum calibrated thresholds. It is licensed for
use inside the latence.ai product family; it is not OSS.

## Headline numbers

Run on **RAGTruth** + **HaluEval**, A5000 batch=1. Headline = `groundedness_v2`.

| Lane | Internal lex / sem / partial | RAGTruth macro F1 | HaluEval QA F1 | Latency p95 |
|------|-----------------------------:|------------------:|---------------:|------------:|
| Dense + literal only | 0.80 / 0.93 / 0.95 | 0.48 | 0.75 | 92 ms |
| + NLI peer (reranker + atomic) | 0.99 / 1.00 / 1.00 | 0.49 | **0.90** | 102 ms |
| + Semantic entropy (synthetic peers) | 0.98 / 1.00 / 1.00 | **0.60** | 0.80 | 125 ms |

See [docs/benchmarks.md](docs/benchmarks.md) and
[docs/algorithm-audit.md](docs/algorithm-audit.md) for the full per-stratum
breakdown and reproduction instructions.

## Pick your profile

Three Pareto-optimal default profiles ship out of the box. Each is a
single environment variable away; the runtime never overwrites a value
the operator already exported, so you keep full override control.

| Profile | Use when | NLI | Reranker | Atomic + concat | Semantic entropy | Internal min-pair F1 | DE min-pair acc | p95 (A5000, batch=1) | Peak VRAM |
|---|---|---|---|---|---|---|---|---|---|
| `fast` | sub-200 ms p95 SLO; cheap "is it grounded at all?" check | off | off | off | off | 0.66 | 100% (literal-only) | ~160 ms | ~3.7 GB |
| `balanced` (default) | typical RAG QA serving | on (mDeBERTa) | off | off | off | **0.89** | **92%** | ~190 ms | ~4.2 GB |
| `quality` | high-stakes outputs; long multi-premise contexts | on (mDeBERTa) | on (bge v2-m3) | on | off (opt-in via env, requires recalibration) | 0.87 | 87% | ~195 ms | ~4.5 GB |

Numbers are from the per-profile sweep in
[`research/triangular_maxsim/reports/profile_pareto.md`](research/triangular_maxsim/reports/profile_pareto.md);
each profile bundles a calibrated thresholds artefact under
[`latence_trace/data/thresholds.<profile>.json`](latence_trace/data/) and
a fusion-weight artefact under
[`latence_trace/data/fusion_weights.<profile>.json`](latence_trace/data/).

```bash
# Default (balanced)
latence-trace-server

# Lowest p95
LATENCE_TRACE_PROFILE=fast latence-trace-server
# or
latence-trace-server --profile fast

# Maximum coverage
latence-trace-server --profile quality

# Opt out of all presets and rely on your own env vars
latence-trace-server --profile none
```

See [`docs/guides/profiles.md`](docs/guides/profiles.md) for the full
per-profile config dump, the evaluation methodology, and reproduction
commands.

## How

Five fused channels:

1. **Calibrated reverse MaxSim** - dense token-level support score against the
   context, calibrated against an internal null bank.
2. **Literal guardrails** - rule-based extraction and matching of dates,
   numbers, units, currencies, URLs, identifiers.
3. **NLI peer** - DeBERTa-MNLI verification with cross-encoder premise
   reranking (BAAI/bge-reranker-v2-m3) and atomic-fact decomposition.
4. **Semantic entropy** - bidirectional NLI clustering over multiple
   verification samples, Shannon entropy over clusters.
5. **Structured-source verification** - JSON / markdown-table triple extraction
   with numeric tolerance and alias resolution.

All channels are renormalized into a single `groundedness_v2` headline; the
runtime classifies that headline into a per-stratum risk band using calibrated
thresholds (`thresholds.json`).

### Retrieval-efficiency observability (context coverage)

Every response carries two scalars and a per-unit flag that turn the
scoring engine into a feedback channel for your retriever:

| Field | Where | What it tells you |
|---|---|---|
| `scores.context_coverage_ratio` | global | **Absolute-strength signal.** Fraction of fetched support units whose `coverage_score` crossed `coverage_threshold` (default 0.5). Range `[0, 1]`. **`0.4` means 60% of the chunks the retriever pulled were dead weight.** Tune the threshold via `coverage_threshold`. |
| `scores.context_attribution_ratio` | global | **Competitive-placement signal.** Fraction of units that won the argmax for at least one response token. Independent of `coverage_threshold`. Useful for dedup-style questions ("which chunks dominated for some span?"). |
| `support_units[i].coverage_score` | per-unit | Max reverse-context similarity any response token had to this unit. Independent of argmax and of threshold. Range `[0, 1]`. |
| `support_units[i].used` | per-unit | `True` when `coverage_score >= coverage_threshold`. Filter `used == False` to surface dead-weight chunks. |

Coverage and attribution are **independent observability axes** — neither
strictly dominates the other. A unit can lose every argmax (low
attribution) yet still have one strong match (high coverage), and
under tight thresholds the reverse holds. Pick the one that maps to
your retrieval-tuning question:

- *"Is the retriever pulling dead weight?"* → `context_coverage_ratio`.
- *"Are sibling chunks crowding each other out for the same spans?"* →
  watch units with `matched_response_tokens > 0` but `used == False`.

Cost: one `max(dim=0)` reduction over the already-computed `(R, U)`
per-unit similarity matrix. Microbench (CPU) shows median ≤ 0.3 ms even
at 4096 response tokens × 256 support units — well below the per-request
encoder/scorer baseline. No extra encoder calls, no extra Triton kernels.

Empirical signal on real HaluEval-QA data
([`scripts/bench_context_coverage.py`](scripts/bench_context_coverage.py)):

| Scenario                                       | `coverage_ratio` (median) | `support_units_used` (median) |
|------------------------------------------------|---------------------------|-------------------------------|
| Grounded response, 1 retrieved chunk           | 1.000                     | 1 / 1                         |
| Grounded response, 1 relevant + 4 distractors  | **0.117**                 | **1.5 / 12**                  |

The signal collapses from 1.0 to ~0.12 when the retriever over-fetches
— exactly the diagnostic we want. Range invariant `0 ≤ ratio ≤ 1` held
for **400 / 400** real samples; reproduction in
[`docs/guides/tutorial.md`](docs/guides/tutorial.md) §5b.

### Long contexts and long responses

Both `raw_context` and `response_text` are **sentence-packed into windows
that fit the encoder's max sequence length and scored chunk-by-chunk**, so
neither side is silently truncated when input exceeds the encoder limit:

| Field | API knob | Default | What it does |
|---|---|---|---|
| `raw_context` | `raw_context_chunk_tokens` | 256 | Splits the context on sentence boundaries into windows of ~N tokens; each window becomes a `support_unit` and is encoded + scored independently. |
| `response_text` | `response_chunk_tokens` | 256 | Splits the response on sentence boundaries into windows of ~N tokens; each window is encoded, scored against the **full** support set, and per-token scores stitched back to global response positions. Single-window responses skip the chunker entirely (parity-preserving fast path). |

The math is exact: `g_t = max_u m_{t,u}` is row-independent, so splitting
the response along the token axis and concatenating per-window results
produces bitwise-identical headline scores to a hypothetical "encode the
full response in one shot" path that would otherwise OOM beyond the
encoder limit. See
[`tests/test_response_chunking_parity.py`](tests/test_response_chunking_parity.py)
for the parity proof and
[`docs/perf/response_chunking_bench.md`](docs/perf/response_chunking_bench.md)
for the linear-scaling microbench.

## Quickstart

```bash
pip install -e ".[dev]"
latence-trace-server --host 0.0.0.0 --port 8090
```

The server boots with the `balanced` profile by default (NLI peer on,
no reranker, ~190 ms p95). Pick `fast` for tighter SLOs or `quality`
for the full stack - see [Pick your profile](#pick-your-profile).

The default encoder is the multilingual
[`VAGOsolutions/SauerkrautLM-Multi-Reason-ModernColBERT`](https://huggingface.co/VAGOsolutions/SauerkrautLM-Multi-Reason-ModernColBERT)
ModernColBERT checkpoint loaded in `bf16`. Override either with
`VOYAGER_GROUNDEDNESS_MODEL` and `VOYAGER_GROUNDEDNESS_TORCH_DTYPE`.

```bash
# English
curl -X POST http://127.0.0.1:8090/groundedness \
  -H "Content-Type: application/json" \
  -d '{
    "raw_context": "Paris is the capital of France.",
    "query_text": "What is the capital of France?",
    "response_text": "The capital of France is Paris."
  }'

# German
curl -X POST http://127.0.0.1:8090/groundedness \
  -H "Content-Type: application/json" \
  -d '{
    "raw_context": "Berlin ist die Hauptstadt Deutschlands seit 1990.",
    "query_text": "Was ist die Hauptstadt Deutschlands?",
    "response_text": "Die Hauptstadt Deutschlands ist Berlin."
  }'
```

The service returns `scores`, `risk_band`, per-token heatmaps,
`literal_diagnostics`, `structured_diagnostics`, and per-claim NLI evidence
for both languages with the same response schema.

Agents and humans can self-discover the request shape, active profile, and
all sibling endpoints in a single GET:

```bash
curl http://127.0.0.1:8090/agent-help            # canonical request shape, profiles block, endpoint map
curl http://127.0.0.1:8090/.well-known/ai-plugin.json  # ChatGPT-style plugin descriptor
curl http://127.0.0.1:8090/openapi.json          # OpenAPI 3.1 schema
```

Validation errors and service errors share a single structured envelope
(`code`, `message`, `hint`, `docs_url`), so retry / repair loops do not
need to special-case 4xx vs 5xx parsing.

## Multilingual support (English + German)

The default models, regex guardrails, stopwords, conjunction splits, and
calibration null bank are all bilingual EN+DE out of the box, so a German
request like the example above works without any per-request flag:

| Component | Default | Coverage |
|---|---|---|
| ColBERT encoder | `VAGOsolutions/SauerkrautLM-Multi-Reason-ModernColBERT` (bf16) | EN + DE multilingual |
| NLI peer | `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` | EN + DE + 100 langs |
| Cross-encoder reranker | `BAAI/bge-reranker-v2-m3` (opt-in) | Multilingual |
| Atomic-claim splitter | spaCy auto-routes `en_core_web_sm` / `de_core_news_sm` | EN + DE |
| Literal guardrails | Date / number / currency / percent / measurement regex | EN + DE formats |
| Calibration null bank | 16 EN + 16 DE diverse sentences | EN + DE |

See [`docs/guides/multilingual.md`](docs/guides/multilingual.md) for the
full configuration matrix, environment overrides, and notes on adding more
languages.

## Layout

- `latence_trace/core/` - scoring core (groundedness, NLI, claims, semantic
  entropy, structured, thresholds).
- `latence_trace/kernels/` - Triton triangular MaxSim kernel.
- `latence_trace/api/` - FastAPI router, Pydantic models, service layer.
- `latence_trace/providers/` - encoder providers (pylate local, vLLM-factory
  remote ModernColBERT pooling).
- `server/` - uvicorn entry point.
- `research/triangular_maxsim/` - evaluation harness, minimal pairs,
  pre-registered targets, committed reports.
- `scripts/` - threshold calibration and fusion-weight sweeps.

## Reproducing benchmarks

```bash
git clone --depth 1 https://github.com/ParticleMedia/RAGTruth.git \
  research/triangular_maxsim/external_data/RAGTruth
git clone --depth 1 https://github.com/RUCAIBox/HaluEval.git \
  research/triangular_maxsim/external_data/HaluEval
export VOYAGER_GROUNDEDNESS_RAGTRUTH_DIR=$PWD/research/triangular_maxsim/external_data/RAGTruth/voyager_layout
export VOYAGER_GROUNDEDNESS_HALUEVAL_DIR=$PWD/research/triangular_maxsim/external_data/HaluEval/data

python -m research.triangular_maxsim.groundedness_external_eval \
  --pairs-per-stratum 20 --max-external-per-stratum 20 \
  --enable-nli --reranker-model BAAI/bge-reranker-v2-m3 \
  --concat-premises --atomic-claims \
  --out research/triangular_maxsim/reports/phase_j_nli.json
```

## License

Proprietary. See [LICENSE](LICENSE).
