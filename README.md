# latence-trace

> Calibrated, auditable groundedness scoring for RAG and evidence-bearing LLM
> outputs. Part of the latence.ai product family.

`latence-trace` is the standalone Groundedness Tracker (Beta) extracted from
the `voyager-index` retrieval engine. It scores how well an LLM response is
grounded in its supporting context, returns auditable per-claim evidence, and
classifies every output into a calibrated `green` / `amber` / `red` risk band.

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

## Quickstart

```bash
pip install -e ".[dev]"
latence-trace-server --host 0.0.0.0 --port 8090
```

```bash
curl -X POST http://127.0.0.1:8090/groundedness \
  -H "Content-Type: application/json" \
  -d '{
    "raw_context": "Paris is the capital of France.",
    "query_text": "What is the capital of France?",
    "response_text": "The capital of France is Paris."
  }'
```

The service returns `scores`, `risk_band`, per-token heatmaps,
`literal_diagnostics`, `structured_diagnostics`, and per-claim NLI evidence.

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
