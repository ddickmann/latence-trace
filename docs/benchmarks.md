# latence-trace Benchmarks

> Honest numbers, A5000 batch=1, n=120 per stratum (RAGTruth + internal
> pairs) or n=60 per label per stratum (HaluEval paired diagnostics).
> Headline = `groundedness_v2` (calibrated reverse MaxSim + literal
> guardrails + NLI peer with cross-encoder premise reranking +
> atomic-claim decomposition + structured-source verification +
> response-chunked path).

All numbers below are sourced from saved JSON reports under
`research/triangular_maxsim/reports/`. Each row of every table cites
its source artefact.

## Headline (production config)

Source: [`reports/truth_bench_n120.json`](../research/triangular_maxsim/reports/truth_bench_n120.json),
[`reports/halueval_diagnose_mdeberta_n60.json`](../research/triangular_maxsim/reports/halueval_diagnose_mdeberta_n60.json),
[`reports/halueval_diagnose_deberta_en_n60.json`](../research/triangular_maxsim/reports/halueval_diagnose_deberta_en_n60.json).

| Lane | Stratum | n | Metric | Value | 95% CI |
|---|---|--:|:------:|------:|:------:|
| Internal min-pairs | lexical (entity / date / number / unit swap) | 120 | paired_acc | **0.95** | [0.63, 1.00] |
| Internal min-pairs | semantic (negation / role swap) | 60 | paired_acc | **0.98** | [0.90, 1.00] |
| Internal min-pairs | partial support | 30 | paired_acc | **1.00** | [1.00, 1.00] |
| Internal min-pairs | hard compound facts | 30 | paired_acc | **1.00** | [1.00, 1.00] |
| Internal min-pairs | hard structured (JSON / md table) | 30 | paired_acc | **0.93** | [0.83, 1.00] |
| Internal min-pairs | hard distributed dialogue | 30 | paired_acc | 0.57 | [0.40, 0.73] |
| Internal min-pairs | German | 26 | paired_acc | **1.00** | [1.00, 1.00] |
| RAGTruth | macro (qa / summ / data2text) | 360 | F1@median | **0.61** | — |
| RAGTruth | qa | 120 | F1@median | **0.73** (precision **0.98**) | — |
| RAGTruth | summarization | 120 | F1@median | **0.65** (precision 0.80) | — |
| RAGTruth | data2text | 120 | F1@median | 0.45 | — |
| HaluEval QA — multilingual NLI | paired ranking (right > halu) | 60 | paired_acc | 0.67 | — |
| HaluEval QA — **English NLI** † | paired ranking | 60 | paired_acc | **0.78** | — |
| HaluEval Summ — multilingual NLI | paired ranking | 60 | paired_acc | 0.65 | — |
| HaluEval Summ — **English NLI** † | paired ranking | 60 | paired_acc | **0.75** | — |
| HaluEval Dialogue — either NLI | paired ranking | 60 | paired_acc | 0.57–0.58 ‡ | — |
| Latency | end-to-end (NLI on, reranker on, atomic on) | — | p95 | **118 ms** | — |

† English NLI peer = `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`. Set
`VOYAGER_GROUNDEDNESS_NLI_MODEL` and restart. Recommended for
English-only deployments — gains ≈ +10 percentage points paired
ranking accuracy on HaluEval QA / Summarization vs. the multilingual
default.

‡ HaluEval Dialogue is **a known mismatch lane** for context-grounded
scoring: hallucinations introduce real-world facts that are not in the
dialogue context, so both right and hallucinated continuations score
"ungrounded" relative to the supplied context. Same caveat applies to
FActScore biographies. See
[`algorithm-audit.md`](algorithm-audit.md) §"Scope and Known
Mismatches" for the formal explanation.

## What changed since prior reports

The earlier `phase_j_*` lane (n=20 per stratum) overstated HaluEval
QA at F1 0.90 and undershot RAGTruth macro at 0.49. Both numbers were
within the n=20 95% CI of the n=120 numbers above (CI ≈ ±0.20 at
n=20, vs ±0.09 at n=120). Lesson: do not publish n=20 single-run
deltas as headline; the n=120 / n=60 numbers above are the only
ones we stand behind.

The headline now uses **paired ranking accuracy** for HaluEval rather
than F1@median. Paired ranking is the metric the dataset's pair
structure is designed for (each `right_answer` has one
`hallucinated_answer` from the same source); F1@median is overly
sensitive to the threshold-picking step at small n. Reports emit
both so callers can audit either.

## Reproduction

```bash
git clone --depth 1 https://github.com/ParticleMedia/RAGTruth.git \
  research/triangular_maxsim/external_data/RAGTruth
git clone --depth 1 https://github.com/RUCAIBox/HaluEval.git \
  research/triangular_maxsim/external_data/HaluEval

# FActScore biographies ship without source context (open-domain
# factuality); enrich with Wikipedia text first if you want the
# loader to emit non-trivial samples instead of skipping them:
python scripts/enrich_factscore_with_wiki.py \
  --in  research/triangular_maxsim/external_data/factscore/biographies.jsonl \
  --out research/triangular_maxsim/external_data/factscore/biographies_wiki.jsonl

export VOYAGER_GROUNDEDNESS_RAGTRUTH_DIR=$PWD/research/triangular_maxsim/external_data/RAGTruth/voyager_layout
export VOYAGER_GROUNDEDNESS_HALUEVAL_DIR=$PWD/research/triangular_maxsim/external_data/HaluEval/data
export VOYAGER_GROUNDEDNESS_FACTSCORE_DIR=$PWD/research/triangular_maxsim/external_data/factscore
export VOYAGER_GROUNDEDNESS_TORCH_DTYPE=bfloat16
export VOYAGER_GROUNDEDNESS_MODEL=lightonai/GTE-ModernColBERT-v1
export VOYAGER_GROUNDEDNESS_NLI_PREMISE_RERANKER_MODEL=BAAI/bge-reranker-v2-m3

# Truth bench (RAGTruth + internal pairs + FActScore at n=120)
export VOYAGER_GROUNDEDNESS_NLI_MODEL=MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7
python -m research.triangular_maxsim.groundedness_external_eval \
  --pairs-per-stratum 30 --max-external-per-stratum 120 \
  --enable-nli --concat-premises --atomic-claims \
  --out research/triangular_maxsim/reports/truth_bench_n120.json

# HaluEval paired ranking diagnostic — multilingual NLI
HALUEVAL_DIAGNOSE_OUT=research/triangular_maxsim/reports/halueval_diagnose_mdeberta_n60.json \
  python scripts/diagnose_halueval.py --limit 60 \
  --nli-model MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7

# HaluEval paired ranking diagnostic — English-only NLI (recommended for EN deployments)
HALUEVAL_DIAGNOSE_OUT=research/triangular_maxsim/reports/halueval_diagnose_deberta_en_n60.json \
  python scripts/diagnose_halueval.py --limit 60 \
  --nli-model MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli
```

The harness uses `score_groundedness_response_chunked` (the same
production code path the FastAPI server hits), so the reproduced
numbers match the deployed engine bit-for-bit modulo encoder
non-determinism.

## Where to dig deeper

- [`algorithm-audit.md`](algorithm-audit.md) — exact formulas for
  every channel, partition-invariance proofs for response chunking,
  per-channel ablations, and the "Scope and Known Mismatches" section
  that scopes out HaluEval Dialogue and FActScore biographies.
- [`api-reference.md`](api-reference.md) — request / response shape
  of every field cited above (`coverage_score`, `claims[i].verdict`,
  `groundedness_v2`, etc.) and the workload-boundary table.
- [`guides/profiles.md`](guides/profiles.md) — how each profile
  (`fast` / `balanced` / `quality`) maps to the channel toggles
  reported here.
- [`guides/multilingual.md`](guides/multilingual.md) — how to swap
  the NLI peer for English-only deployments and pick up the +10pp
  HaluEval QA / Summarization improvement.
