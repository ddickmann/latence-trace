# Transcripts-v2 agentic-coding bench

n_scenarios: 60  |  scoring_mode: rag  |  profile: quality

## Paired ranking

| Matchup | n_pairs | paired_acc | ties | F1@best |
| --- | ---: | ---: | ---: | ---: |
| correct vs wrong      | 60 | 0.400 | 4 | 0.674 |
| correct vs ambiguous  | 60 | 0.650 | 3 | 0.674 |

## Band distribution

| Subcategory | green | amber | red | other |
| --- | ---: | ---: | ---: | ---: |
| correct | 0 | 18 | 42 | 0 |
| wrong | 0 | 19 | 41 | 0 |
| ambiguous | 0 | 18 | 42 | 0 |

## Zero-human-in-the-loop precision

Amber counted as incorrect for the zero-human framing (auto-decide would need to collapse it).

* green_precision on correct cases (amber excluded): **0.000**
* red_precision on wrong cases (amber excluded):     **1.000**

Gate target: paired_acc (correct vs wrong) >= 0.80 and red_precision (excluding amber) >= 0.90.
