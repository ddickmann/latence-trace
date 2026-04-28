# TRACE v2 learned local-support student

This directory contains the architecture, data prep, and training loop
for the v2 learned-student that replaces the heuristic NLI+MaxSim
aggregator with a distilled biaffine token-to-token scorer on top of a
ColBERT-style late-interaction encoder.

## Design

```
response tokens -> encoder -> h_i  (64d)
evidence tokens -> encoder -> h_j  (64d)

local similarity:   s_ij = h_i^T W h_j + u^T h_i + v^T h_j + b + phi(i,j)
training pooling:   g_i^train = (1/alpha) * log sum_{j in Top-k} exp(alpha * s_ij)
inference pooling:  g_i^infer = max_j s_ij

turn-level head:    p_green, p_amber, p_red from mean/max pooled g_i
                    plus support_units_used and coverage ratio channels.
```

`phi(i,j)` is a small feature MLP over cheap signals:

* exact token match bit
* lemmatised match bit
* identifier / numeric match bit
* source-type pair (code, docstring, passage, test, issue, README, log).

This directly targets the weak cases the v1 stack fails on:

1. short factual answers that the NLI cannot align;
2. enumerated / multi-step paraphrases that the ColBERT MaxSim blurs;
3. abstentions that look like ungrounded output to the NLI channel.

## Files

* `architecture.py` - encoder + biaffine scorer + soft-topk pooling.
* `distill_dataset.py` - reads the audit log, Veracier + HaluEval +
  RAGTruth scores from v1, and produces a multi-task training corpus
  with (response_tokens, evidence_tokens, token_support_labels,
  turn_band, turn_score, pairwise_ranking) examples.
* `train.py` - multi-task training loop:
  - token support BCE
  - turn band cross-entropy + score MSE
  - pairwise ranking loss on (grounded, hallucinated) pairs from the
    same source_id.
* `eval.py` - evaluates the trained student against HaluEval QA /
  RAGTruth QA / Veracier with the same harness as v1 and writes a
  side-by-side comparison table.

## Go/no-go trigger

The student is trained only if v1 fails any of the following for two
consecutive weeks (see `docs/roadmap.md` and
`data/veracier-industries/proof_bundle_v1/failure_modes_v1.md`):

* HaluEval QA red precision < 80% at standard profile
* RAGTruth QA red precision < 50% at quality profile
* any Veracier vertical green precision < 95%

## Success criteria

| Benchmark | v1 red precision | v2 target | Notes |
|---|---|---|---|
| HaluEval QA (standard) | ~71% | >=90% | Hard because of short factual answers |
| HaluEval Summ (standard) | ~60% | >=85% | Easier, more context |
| RAGTruth QA (quality) | ~27% | >=80% | Multi-step paraphrase alignment |
| RAGTruth Summ (quality) | TBD | >=80% | TBD |
| Veracier green precision | 100% | >=97% | No regression allowed |
| Veracier red precision | 100% | >=95% | No regression allowed |

If the student fails any of these, we keep v1 live and fall back to
larger fine-tuned backbone instead.
