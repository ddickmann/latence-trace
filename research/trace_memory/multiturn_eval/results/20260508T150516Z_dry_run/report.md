# Multi-Turn InfiniMem A/B Eval Report

- Mode: `dry-run`
- Model: `gpt-5.5`
- Cases: `5` (5 code, 0 rag)


## Token Reduction Stats
- Mean: `26.14%`
- Median: `76.98%`
- Min: `-105.82%`
- Max: `99.65%`

## Per-Case Results

### code_en_b731fb27_01
- Type: `code` | Lang: `en` | Turns: `8`
- Overall: `(dry-run)`
- Mean reduction: `85.30%`
- Degradation slope: `0.0224`

| Turn | Pattern | Full Tok | Compressed Tok | Reduction | Spans | Hot | Classification |
|------|---------|----------|----------------|-----------|-------|-----|----------------|
| 0 | initial_context | 567 | 136 | 76.0% | 75 | 75 | (dry-run) |
| 1 | follow_up | 630 | 136 | 78.4% | 80 | 80 | (dry-run) |
| 2 | new_context | 5,716 | 1,188 | 79.2% | 423 | 422 | (dry-run) |
| 3 | no_context | 9,635 | 1,188 | 87.7% | 684 | 672 | (dry-run) |
| 4 | follow_up | 13,237 | 1,188 | 91.0% | 928 | 909 | (dry-run) |
| 5 | follow_up | 14,125 | 1,188 | 91.6% | 989 | 973 | (dry-run) |
| 6 | new_context | 19,201 | 2,194 | 88.6% | 1322 | 1295 | (dry-run) |
| 7 | follow_up | 21,686 | 2,194 | 89.9% | 1505 | 1487 | (dry-run) |

### code_en_d7cfd257_00
- Type: `code` | Lang: `en` | Turns: `8`
- Overall: `(dry-run)`
- Mean reduction: `-23.04%`
- Degradation slope: `0.1728`

| Turn | Pattern | Full Tok | Compressed Tok | Reduction | Spans | Hot | Classification |
|------|---------|----------|----------------|-----------|-------|-----|----------------|
| 0 | no_context | 4,013 | 7,966 | -98.5% | 333 | 329 | (dry-run) |
| 1 | follow_up | 8,055 | 11,841 | -47.0% | 482 | 463 | (dry-run) |
| 2 | follow_up | 11,035 | 14,238 | -29.0% | 695 | 638 | (dry-run) |
| 3 | no_context | 11,901 | 15,260 | -28.2% | 733 | 680 | (dry-run) |
| 4 | follow_up | 13,204 | 16,769 | -27.0% | 830 | 782 | (dry-run) |
| 5 | follow_up | 14,901 | 18,064 | -21.2% | 915 | 856 | (dry-run) |
| 6 | no_context | 15,611 | 18,409 | -17.9% | 967 | 896 | (dry-run) |
| 7 | new_context | 19,847 | 3,060 | 84.6% | 1228 | 1170 | (dry-run) |

### code_en_9bb922a8_01
- Type: `code` | Lang: `en` | Turns: `8`
- Overall: `(dry-run)`
- Mean reduction: `56.10%`
- Degradation slope: `0.1597`

| Turn | Pattern | Full Tok | Compressed Tok | Reduction | Spans | Hot | Classification |
|------|---------|----------|----------------|-----------|-------|-----|----------------|
| 0 | no_context | 1,624 | 3,281 | -102.0% | 228 | 228 | (dry-run) |
| 1 | new_context | 4,147 | 1,112 | 73.2% | 456 | 454 | (dry-run) |
| 2 | follow_up | 4,904 | 1,112 | 77.3% | 512 | 510 | (dry-run) |
| 3 | new_context | 10,210 | 2,350 | 77.0% | 855 | 816 | (dry-run) |
| 4 | follow_up | 11,355 | 2,350 | 79.3% | 934 | 897 | (dry-run) |
| 5 | follow_up | 12,434 | 2,350 | 81.1% | 1002 | 968 | (dry-run) |
| 6 | new_context | 17,584 | 3,449 | 80.4% | 1348 | 1292 | (dry-run) |
| 7 | follow_up | 19,756 | 3,449 | 82.5% | 1509 | 1459 | (dry-run) |

### code_en_63afbe56_01
- Type: `code` | Lang: `en` | Turns: `8`
- Overall: `(dry-run)`
- Mean reduction: `41.55%`
- Degradation slope: `0.2951`

| Turn | Pattern | Full Tok | Compressed Tok | Reduction | Spans | Hot | Classification |
|------|---------|----------|----------------|-----------|-------|-----|----------------|
| 0 | no_context | 275 | 566 | -105.8% | 55 | 55 | (dry-run) |
| 1 | follow_up | 913 | 1,214 | -33.0% | 113 | 110 | (dry-run) |
| 2 | follow_up | 1,223 | 1,545 | -26.3% | 138 | 135 | (dry-run) |
| 3 | new_context | 3,834 | 24 | 99.4% | 309 | 304 | (dry-run) |
| 4 | follow_up | 4,027 | 24 | 99.4% | 326 | 321 | (dry-run) |
| 5 | follow_up | 4,658 | 24 | 99.5% | 378 | 373 | (dry-run) |
| 6 | no_context | 6,457 | 24 | 99.6% | 547 | 530 | (dry-run) |
| 7 | follow_up | 6,847 | 24 | 99.7% | 577 | 560 | (dry-run) |

### code_en_ffc1cd2e_00
- Type: `code` | Lang: `en` | Turns: `8`
- Overall: `(dry-run)`
- Mean reduction: `-29.19%`
- Degradation slope: `0.1912`

| Turn | Pattern | Full Tok | Compressed Tok | Reduction | Spans | Hot | Classification |
|------|---------|----------|----------------|-----------|-------|-----|----------------|
| 0 | no_context | 3,640 | 7,340 | -101.6% | 334 | 334 | (dry-run) |
| 1 | follow_up | 6,532 | 10,193 | -56.0% | 470 | 466 | (dry-run) |
| 2 | follow_up | 7,856 | 11,554 | -47.1% | 543 | 538 | (dry-run) |
| 3 | no_context | 10,134 | 13,912 | -37.3% | 668 | 663 | (dry-run) |
| 4 | follow_up | 11,298 | 15,105 | -33.7% | 722 | 717 | (dry-run) |
| 5 | follow_up | 12,813 | 16,606 | -29.6% | 782 | 777 | (dry-run) |
| 6 | no_context | 13,740 | 17,564 | -27.8% | 832 | 827 | (dry-run) |
| 7 | new_context | 16,890 | 61 | 99.6% | 1004 | 996 | (dry-run) |
