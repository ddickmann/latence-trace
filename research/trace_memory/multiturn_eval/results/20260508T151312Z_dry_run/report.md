# Multi-Turn InfiniMem A/B Eval Report

- Mode: `dry-run`
- Model: `gpt-5.5`
- Cases: `5` (5 code, 0 rag)


## Token Reduction Stats
- Mean: `-0.67%`
- Median: `-1.48%`
- Min: `-7.27%`
- Max: `10.82%`

## Per-Case Results

### code_en_b731fb27_01
- Type: `code` | Lang: `en` | Turns: `8`
- Overall: `(dry-run)`
- Mean reduction: `-0.07%`
- Degradation slope: `0.0030`

| Turn | Pattern | Full Tok | Compressed Tok | Reduction | Spans | Hot | Classification |
|------|---------|----------|----------------|-----------|-------|-----|----------------|
| 0 | initial_context | 567 | 575 | -1.4% | 51 | 51 | (dry-run) |
| 1 | follow_up | 630 | 628 | 0.3% | 56 | 56 | (dry-run) |
| 2 | new_context | 5,716 | 5,821 | -1.8% | 398 | 397 | (dry-run) |
| 3 | no_context | 9,635 | 9,699 | -0.7% | 659 | 654 | (dry-run) |
| 4 | follow_up | 13,237 | 13,106 | 1.0% | 903 | 893 | (dry-run) |
| 5 | follow_up | 14,125 | 14,036 | 0.6% | 964 | 955 | (dry-run) |
| 6 | new_context | 19,201 | 19,017 | 1.0% | 1297 | 1286 | (dry-run) |
| 7 | follow_up | 21,686 | 21,585 | 0.5% | 1480 | 1470 | (dry-run) |

### code_en_d7cfd257_00
- Type: `code` | Lang: `en` | Turns: `8`
- Overall: `(dry-run)`
- Mean reduction: `-2.23%`
- Degradation slope: `0.0041`

| Turn | Pattern | Full Tok | Compressed Tok | Reduction | Spans | Hot | Classification |
|------|---------|----------|----------------|-----------|-------|-----|----------------|
| 0 | no_context | 4,013 | 4,181 | -4.2% | 171 | 171 | (dry-run) |
| 1 | follow_up | 8,055 | 8,416 | -4.5% | 320 | 317 | (dry-run) |
| 2 | follow_up | 11,035 | 11,273 | -2.2% | 533 | 506 | (dry-run) |
| 3 | no_context | 11,901 | 12,088 | -1.6% | 571 | 542 | (dry-run) |
| 4 | follow_up | 13,204 | 13,345 | -1.1% | 668 | 637 | (dry-run) |
| 5 | follow_up | 14,901 | 15,080 | -1.2% | 753 | 723 | (dry-run) |
| 6 | no_context | 15,611 | 15,717 | -0.7% | 805 | 773 | (dry-run) |
| 7 | new_context | 19,847 | 20,346 | -2.5% | 1066 | 1037 | (dry-run) |

### code_en_9bb922a8_01
- Type: `code` | Lang: `en` | Turns: `8`
- Overall: `(dry-run)`
- Mean reduction: `3.42%`
- Degradation slope: `0.0191`

| Turn | Pattern | Full Tok | Compressed Tok | Reduction | Spans | Hot | Classification |
|------|---------|----------|----------------|-----------|-------|-----|----------------|
| 0 | no_context | 1,624 | 1,661 | -2.3% | 115 | 115 | (dry-run) |
| 1 | new_context | 4,147 | 4,227 | -1.9% | 343 | 341 | (dry-run) |
| 2 | follow_up | 4,904 | 4,989 | -1.7% | 399 | 397 | (dry-run) |
| 3 | new_context | 10,210 | 10,275 | -0.6% | 742 | 738 | (dry-run) |
| 4 | follow_up | 11,355 | 10,394 | 8.5% | 821 | 779 | (dry-run) |
| 5 | follow_up | 12,434 | 11,323 | 8.9% | 889 | 839 | (dry-run) |
| 6 | new_context | 17,584 | 15,682 | 10.8% | 1235 | 1156 | (dry-run) |
| 7 | follow_up | 19,756 | 18,629 | 5.7% | 1396 | 1345 | (dry-run) |

### code_en_63afbe56_01
- Type: `code` | Lang: `en` | Turns: `8`
- Overall: `(dry-run)`
- Mean reduction: `-3.11%`
- Degradation slope: `0.0060`

| Turn | Pattern | Full Tok | Compressed Tok | Reduction | Spans | Hot | Classification |
|------|---------|----------|----------------|-----------|-------|-----|----------------|
| 0 | no_context | 275 | 295 | -7.3% | 29 | 29 | (dry-run) |
| 1 | follow_up | 913 | 943 | -3.3% | 87 | 84 | (dry-run) |
| 2 | follow_up | 1,223 | 1,274 | -4.2% | 112 | 109 | (dry-run) |
| 3 | new_context | 3,834 | 3,892 | -1.5% | 282 | 277 | (dry-run) |
| 4 | follow_up | 4,027 | 4,114 | -2.2% | 299 | 294 | (dry-run) |
| 5 | follow_up | 4,658 | 4,815 | -3.4% | 351 | 346 | (dry-run) |
| 6 | no_context | 6,457 | 6,556 | -1.5% | 519 | 502 | (dry-run) |
| 7 | follow_up | 6,847 | 6,953 | -1.6% | 548 | 531 | (dry-run) |

### code_en_ffc1cd2e_00
- Type: `code` | Lang: `en` | Turns: `8`
- Overall: `(dry-run)`
- Mean reduction: `-1.38%`
- Degradation slope: `0.0001`

| Turn | Pattern | Full Tok | Compressed Tok | Reduction | Spans | Hot | Classification |
|------|---------|----------|----------------|-----------|-------|-----|----------------|
| 0 | no_context | 3,640 | 3,719 | -2.2% | 174 | 174 | (dry-run) |
| 1 | follow_up | 6,532 | 6,572 | -0.6% | 310 | 306 | (dry-run) |
| 2 | follow_up | 7,856 | 7,933 | -1.0% | 383 | 378 | (dry-run) |
| 3 | no_context | 10,134 | 10,291 | -1.6% | 508 | 503 | (dry-run) |
| 4 | follow_up | 11,298 | 11,484 | -1.7% | 562 | 557 | (dry-run) |
| 5 | follow_up | 12,813 | 12,985 | -1.3% | 622 | 617 | (dry-run) |
| 6 | no_context | 13,740 | 13,943 | -1.5% | 672 | 667 | (dry-run) |
| 7 | new_context | 16,890 | 17,105 | -1.3% | 844 | 836 | (dry-run) |
