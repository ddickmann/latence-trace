# latence-trace Benchmarks

> Benchmark numbers for the standalone Groundedness Tracker (Beta).

## Real-world headline

Run on **RAGTruth** + **HaluEval**, per-stratum samples, A5000 batch 1.
Headline = `groundedness_v2` (calibrated reverse MaxSim + literal guardrails +
optional NLI peer with cross-encoder premise reranking, atomic-claim
decomposition, semantic-entropy consistency, and structured-source
verification).

| Lane                                     | Internal lex / sem / partial | RAGTruth macro F1 | HaluEval QA F1 | Latency p95 |
|------------------------------------------|-----------------------------:|------------------:|---------------:|------------:|
| Dense + literal only                     |           0.80 / 0.93 / 0.95 |              0.48 |           0.75 |      92 ms  |
| **+ NLI peer (reranker + atomic)**       |       **0.99 / 1.00 / 1.00** |              0.49 |       **0.90** |     102 ms  |
| **+ Semantic entropy (synthetic peers)** |       **0.98 / 1.00 / 1.00** |          **0.60** |           0.80 |     125 ms  |
| Pre-registered exit                      |    ≥ 0.80 / ≥ 0.70 / ≥ 0.65  |            ≥ 0.55 |         ≥ 0.75 |  ≤ 400 ms   |

Semantic-entropy + NLI lane hits **RAGTruth macro ≥ 0.55** and HaluEval QA
≥ 0.75 simultaneously; the NLI-only lane sets the HaluEval QA headline at
**0.90** F1.

Reports live under `research/triangular_maxsim/reports/phase_j_no_nli.json`,
`phase_j_nli.json`, and `phase_j_nli_sem.json`. The full per-stratum tables,
harness, env vars, and reproduction guide live in
`research/triangular_maxsim/README.md`; the algorithm verdict and per-channel
ablations are in [`algorithm-audit.md`](algorithm-audit.md). The product
framing, risk-band policy, and Phase I structured-source verification are in
[`guides/beta-overview.md`](guides/beta-overview.md).

## Reproduction

```bash
git clone --depth 1 https://github.com/ParticleMedia/RAGTruth.git \
  research/triangular_maxsim/external_data/RAGTruth
git clone --depth 1 https://github.com/RUCAIBox/HaluEval.git \
  research/triangular_maxsim/external_data/HaluEval

export VOYAGER_GROUNDEDNESS_RAGTRUTH_DIR=$PWD/research/triangular_maxsim/external_data/RAGTruth/voyager_layout
export VOYAGER_GROUNDEDNESS_HALUEVAL_DIR=$PWD/research/triangular_maxsim/external_data/HaluEval/data

# No-NLI lane
python -m research.triangular_maxsim.groundedness_external_eval \
  --pairs-per-stratum 20 --max-external-per-stratum 20 \
  --out research/triangular_maxsim/reports/phase_j_no_nli.json

# NLI peer (reranker + atomic claims)
python -m research.triangular_maxsim.groundedness_external_eval \
  --pairs-per-stratum 20 --max-external-per-stratum 20 \
  --enable-nli --reranker-model BAAI/bge-reranker-v2-m3 \
  --concat-premises --atomic-claims \
  --out research/triangular_maxsim/reports/phase_j_nli.json

# Semantic-entropy lane (synthetic peers)
python -m research.triangular_maxsim.groundedness_external_eval \
  --pairs-per-stratum 20 --max-external-per-stratum 20 \
  --enable-nli --semantic-entropy-synthetic \
  --out research/triangular_maxsim/reports/phase_j_nli_sem.json
```
