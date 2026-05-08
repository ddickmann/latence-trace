# Multi-Turn InfiniMem A/B Eval Report

- Mode: `dry-run`
- Model: `gpt-5.5`
- Cases: `40` (20 code, 20 rag)


## Token Reduction Stats
- Mean: `-30.96%`
- Median: `-20.00%`
- Min: `-122.20%`
- Max: `17.41%`

## Per-Case Results

### code_en_b731fb27_01
- Type: `code` | Lang: `en` | Turns: `8`
- Overall: `(dry-run)`
- Mean reduction: `-19.60%`
- Degradation slope: `0.1078`

| Turn | Pattern | Full Tok | Compressed Tok | Reduction | Spans | Hot | Classification |
|------|---------|----------|----------------|-----------|-------|-----|----------------|
| 0 | initial_context | 567 | 1,002 | -76.7% | 75 | 75 | (dry-run) |
| 1 | follow_up | 630 | 1,055 | -67.5% | 80 | 80 | (dry-run) |
| 2 | new_context | 5,716 | 6,255 | -9.4% | 423 | 422 | (dry-run) |
| 3 | no_context | 9,635 | 9,915 | -2.9% | 684 | 672 | (dry-run) |
| 4 | follow_up | 13,237 | 13,262 | -0.2% | 928 | 909 | (dry-run) |
| 5 | follow_up | 14,125 | 14,256 | -0.9% | 989 | 973 | (dry-run) |
| 6 | new_context | 19,201 | 18,969 | 1.2% | 1322 | 1295 | (dry-run) |
| 7 | follow_up | 21,686 | 21,772 | -0.4% | 1505 | 1487 | (dry-run) |

### code_en_d7cfd257_00
- Type: `code` | Lang: `en` | Turns: `8`
- Overall: `(dry-run)`
- Mean reduction: `-35.78%`
- Degradation slope: `0.0878`

| Turn | Pattern | Full Tok | Compressed Tok | Reduction | Spans | Hot | Classification |
|------|---------|----------|----------------|-----------|-------|-----|----------------|
| 0 | no_context | 4,013 | 7,966 | -98.5% | 333 | 329 | (dry-run) |
| 1 | follow_up | 8,055 | 11,841 | -47.0% | 482 | 463 | (dry-run) |
| 2 | follow_up | 11,035 | 14,238 | -29.0% | 695 | 638 | (dry-run) |
| 3 | no_context | 11,901 | 15,260 | -28.2% | 733 | 680 | (dry-run) |
| 4 | follow_up | 13,204 | 16,769 | -27.0% | 830 | 782 | (dry-run) |
| 5 | follow_up | 14,901 | 18,064 | -21.2% | 915 | 856 | (dry-run) |
| 6 | no_context | 15,611 | 18,409 | -17.9% | 967 | 896 | (dry-run) |
| 7 | new_context | 19,847 | 23,294 | -17.4% | 1228 | 1170 | (dry-run) |

### code_en_9bb922a8_01
- Type: `code` | Lang: `en` | Turns: `8`
- Overall: `(dry-run)`
- Mean reduction: `-25.75%`
- Degradation slope: `0.1152`

| Turn | Pattern | Full Tok | Compressed Tok | Reduction | Spans | Hot | Classification |
|------|---------|----------|----------------|-----------|-------|-----|----------------|
| 0 | no_context | 1,624 | 3,281 | -102.0% | 228 | 228 | (dry-run) |
| 1 | new_context | 4,147 | 5,847 | -41.0% | 456 | 454 | (dry-run) |
| 2 | follow_up | 4,904 | 6,609 | -34.8% | 512 | 510 | (dry-run) |
| 3 | new_context | 10,210 | 11,020 | -7.9% | 855 | 816 | (dry-run) |
| 4 | follow_up | 11,355 | 12,195 | -7.4% | 934 | 897 | (dry-run) |
| 5 | follow_up | 12,434 | 13,357 | -7.4% | 1002 | 968 | (dry-run) |
| 6 | new_context | 17,584 | 17,990 | -2.3% | 1348 | 1292 | (dry-run) |
| 7 | follow_up | 19,756 | 20,386 | -3.2% | 1509 | 1459 | (dry-run) |

### code_en_63afbe56_01
- Type: `code` | Lang: `en` | Turns: `8`
- Overall: `(dry-run)`
- Mean reduction: `-25.64%`
- Degradation slope: `0.1050`

| Turn | Pattern | Full Tok | Compressed Tok | Reduction | Spans | Hot | Classification |
|------|---------|----------|----------------|-----------|-------|-----|----------------|
| 0 | no_context | 275 | 566 | -105.8% | 55 | 55 | (dry-run) |
| 1 | follow_up | 913 | 1,214 | -33.0% | 113 | 110 | (dry-run) |
| 2 | follow_up | 1,223 | 1,545 | -26.3% | 138 | 135 | (dry-run) |
| 3 | new_context | 3,834 | 4,177 | -8.9% | 309 | 304 | (dry-run) |
| 4 | follow_up | 4,027 | 4,399 | -9.2% | 326 | 321 | (dry-run) |
| 5 | follow_up | 4,658 | 5,100 | -9.5% | 378 | 373 | (dry-run) |
| 6 | no_context | 6,457 | 6,856 | -6.2% | 547 | 530 | (dry-run) |
| 7 | follow_up | 6,847 | 7,266 | -6.1% | 577 | 560 | (dry-run) |

### code_en_ffc1cd2e_00
- Type: `code` | Lang: `en` | Turns: `8`
- Overall: `(dry-run)`
- Mean reduction: `-44.49%`
- Degradation slope: `0.0892`

| Turn | Pattern | Full Tok | Compressed Tok | Reduction | Spans | Hot | Classification |
|------|---------|----------|----------------|-----------|-------|-----|----------------|
| 0 | no_context | 3,640 | 7,340 | -101.6% | 334 | 334 | (dry-run) |
| 1 | follow_up | 6,532 | 10,193 | -56.0% | 470 | 466 | (dry-run) |
| 2 | follow_up | 7,856 | 11,554 | -47.1% | 543 | 538 | (dry-run) |
| 3 | no_context | 10,134 | 13,912 | -37.3% | 668 | 663 | (dry-run) |
| 4 | follow_up | 11,298 | 15,105 | -33.7% | 722 | 717 | (dry-run) |
| 5 | follow_up | 12,813 | 16,606 | -29.6% | 782 | 777 | (dry-run) |
| 6 | no_context | 13,740 | 17,564 | -27.8% | 832 | 827 | (dry-run) |
| 7 | new_context | 16,890 | 20,726 | -22.7% | 1004 | 996 | (dry-run) |

### code_en_f45173b3_01
- Type: `code` | Lang: `en` | Turns: `8`
- Overall: `(dry-run)`
- Mean reduction: `-35.23%`
- Degradation slope: `0.1280`

| Turn | Pattern | Full Tok | Compressed Tok | Reduction | Spans | Hot | Classification |
|------|---------|----------|----------------|-----------|-------|-----|----------------|
| 0 | no_context | 2,467 | 4,942 | -100.3% | 376 | 376 | (dry-run) |
| 1 | follow_up | 2,982 | 5,467 | -83.3% | 399 | 399 | (dry-run) |
| 2 | new_context | 9,146 | 11,703 | -28.0% | 741 | 734 | (dry-run) |
| 3 | no_context | 10,771 | 13,320 | -23.7% | 839 | 828 | (dry-run) |
| 4 | follow_up | 13,490 | 16,061 | -19.1% | 989 | 975 | (dry-run) |
| 5 | follow_up | 14,090 | 15,568 | -10.5% | 1023 | 978 | (dry-run) |
| 6 | no_context | 15,535 | 16,936 | -9.0% | 1113 | 1064 | (dry-run) |
| 7 | follow_up | 16,185 | 17,476 | -8.0% | 1154 | 1103 | (dry-run) |

### code_en_1b6a0759_00
- Type: `code` | Lang: `en` | Turns: `5`
- Overall: `(dry-run)`
- Mean reduction: `-35.30%`
- Degradation slope: `0.1890`

| Turn | Pattern | Full Tok | Compressed Tok | Reduction | Spans | Hot | Classification |
|------|---------|----------|----------------|-----------|-------|-----|----------------|
| 0 | no_context | 2,018 | 4,065 | -101.4% | 275 | 275 | (dry-run) |
| 1 | new_context | 7,429 | 9,491 | -27.8% | 651 | 641 | (dry-run) |
| 2 | follow_up | 11,112 | 13,282 | -19.5% | 916 | 907 | (dry-run) |
| 3 | no_context | 15,139 | 17,253 | -14.0% | 1158 | 1144 | (dry-run) |
| 4 | follow_up | 15,428 | 17,560 | -13.8% | 1177 | 1163 | (dry-run) |

### code_en_1b6a0759_01
- Type: `code` | Lang: `en` | Turns: `5`
- Overall: `(dry-run)`
- Mean reduction: `-35.30%`
- Degradation slope: `0.1890`

| Turn | Pattern | Full Tok | Compressed Tok | Reduction | Spans | Hot | Classification |
|------|---------|----------|----------------|-----------|-------|-----|----------------|
| 0 | no_context | 2,018 | 4,065 | -101.4% | 275 | 275 | (dry-run) |
| 1 | new_context | 7,429 | 9,491 | -27.8% | 651 | 641 | (dry-run) |
| 2 | follow_up | 11,112 | 13,282 | -19.5% | 916 | 907 | (dry-run) |
| 3 | no_context | 15,139 | 17,253 | -14.0% | 1158 | 1144 | (dry-run) |
| 4 | follow_up | 15,428 | 17,560 | -13.8% | 1177 | 1163 | (dry-run) |

### code_en_1b6a0759_02
- Type: `code` | Lang: `en` | Turns: `5`
- Overall: `(dry-run)`
- Mean reduction: `-35.30%`
- Degradation slope: `0.1890`

| Turn | Pattern | Full Tok | Compressed Tok | Reduction | Spans | Hot | Classification |
|------|---------|----------|----------------|-----------|-------|-----|----------------|
| 0 | no_context | 2,018 | 4,065 | -101.4% | 275 | 275 | (dry-run) |
| 1 | new_context | 7,429 | 9,491 | -27.8% | 651 | 641 | (dry-run) |
| 2 | follow_up | 11,112 | 13,282 | -19.5% | 916 | 907 | (dry-run) |
| 3 | no_context | 15,139 | 17,253 | -14.0% | 1158 | 1144 | (dry-run) |
| 4 | follow_up | 15,428 | 17,560 | -13.8% | 1177 | 1163 | (dry-run) |

### code_en_f45173b3_00
- Type: `code` | Lang: `en` | Turns: `8`
- Overall: `(dry-run)`
- Mean reduction: `-32.14%`
- Degradation slope: `0.1121`

| Turn | Pattern | Full Tok | Compressed Tok | Reduction | Spans | Hot | Classification |
|------|---------|----------|----------------|-----------|-------|-----|----------------|
| 0 | no_context | 1,452 | 2,937 | -102.3% | 149 | 149 | (dry-run) |
| 1 | follow_up | 2,371 | 3,868 | -63.1% | 203 | 203 | (dry-run) |
| 2 | follow_up | 5,224 | 6,688 | -28.0% | 348 | 346 | (dry-run) |
| 3 | new_context | 11,458 | 13,094 | -14.3% | 728 | 722 | (dry-run) |
| 4 | follow_up | 11,854 | 13,498 | -13.9% | 756 | 749 | (dry-run) |
| 5 | follow_up | 13,073 | 14,735 | -12.7% | 842 | 834 | (dry-run) |
| 6 | no_context | 14,167 | 15,838 | -11.8% | 916 | 908 | (dry-run) |
| 7 | follow_up | 15,145 | 16,812 | -11.0% | 960 | 952 | (dry-run) |

### code_en_2ae1ac4f_01
- Type: `code` | Lang: `en` | Turns: `8`
- Overall: `(dry-run)`
- Mean reduction: `-25.91%`
- Degradation slope: `0.0935`

| Turn | Pattern | Full Tok | Compressed Tok | Reduction | Spans | Hot | Classification |
|------|---------|----------|----------------|-----------|-------|-----|----------------|
| 0 | no_context | 1,154 | 2,334 | -102.2% | 137 | 137 | (dry-run) |
| 1 | follow_up | 3,635 | 4,839 | -33.1% | 243 | 242 | (dry-run) |
| 2 | new_context | 8,647 | 9,973 | -15.3% | 561 | 553 | (dry-run) |
| 3 | no_context | 9,083 | 10,413 | -14.6% | 589 | 581 | (dry-run) |
| 4 | follow_up | 10,242 | 11,564 | -12.9% | 656 | 648 | (dry-run) |
| 5 | follow_up | 12,856 | 14,173 | -10.2% | 799 | 788 | (dry-run) |
| 6 | no_context | 13,941 | 15,248 | -9.4% | 859 | 848 | (dry-run) |
| 7 | follow_up | 14,088 | 15,413 | -9.4% | 874 | 863 | (dry-run) |

### code_en_34589790_02
- Type: `code` | Lang: `en` | Turns: `8`
- Overall: `(dry-run)`
- Mean reduction: `-18.02%`
- Degradation slope: `0.0669`

| Turn | Pattern | Full Tok | Compressed Tok | Reduction | Spans | Hot | Classification |
|------|---------|----------|----------------|-----------|-------|-----|----------------|
| 0 | initial_context | 2,274 | 3,351 | -47.4% | 299 | 299 | (dry-run) |
| 1 | follow_up | 2,364 | 3,429 | -45.1% | 307 | 306 | (dry-run) |
| 2 | new_context | 5,589 | 6,707 | -20.0% | 614 | 603 | (dry-run) |
| 3 | no_context | 5,863 | 6,928 | -18.2% | 632 | 618 | (dry-run) |
| 4 | new_context | 9,404 | 9,724 | -3.4% | 966 | 870 | (dry-run) |
| 5 | follow_up | 9,917 | 10,059 | -1.4% | 1008 | 896 | (dry-run) |
| 6 | new_context | 12,144 | 11,929 | 1.8% | 1221 | 1071 | (dry-run) |
| 7 | follow_up | 13,578 | 15,012 | -10.6% | 1339 | 1302 | (dry-run) |

### code_en_b731fb27_02
- Type: `code` | Lang: `en` | Turns: `8`
- Overall: `(dry-run)`
- Mean reduction: `-31.10%`
- Degradation slope: `0.1258`

| Turn | Pattern | Full Tok | Compressed Tok | Reduction | Spans | Hot | Classification |
|------|---------|----------|----------------|-----------|-------|-----|----------------|
| 0 | no_context | 2,484 | 5,023 | -102.2% | 372 | 372 | (dry-run) |
| 1 | follow_up | 4,640 | 7,189 | -54.9% | 540 | 538 | (dry-run) |
| 2 | follow_up | 5,551 | 7,981 | -43.8% | 607 | 600 | (dry-run) |
| 3 | no_context | 8,740 | 10,411 | -19.1% | 827 | 790 | (dry-run) |
| 4 | follow_up | 9,473 | 10,656 | -12.5% | 866 | 815 | (dry-run) |
| 5 | follow_up | 10,426 | 11,400 | -9.3% | 926 | 871 | (dry-run) |
| 6 | new_context | 11,960 | 12,249 | -2.4% | 1053 | 969 | (dry-run) |
| 7 | follow_up | 13,544 | 14,151 | -4.5% | 1145 | 1077 | (dry-run) |

### code_en_63afbe56_00
- Type: `code` | Lang: `en` | Turns: `8`
- Overall: `(dry-run)`
- Mean reduction: `-16.73%`
- Degradation slope: `0.0915`

| Turn | Pattern | Full Tok | Compressed Tok | Reduction | Spans | Hot | Classification |
|------|---------|----------|----------------|-----------|-------|-----|----------------|
| 0 | no_context | 674 | 1,365 | -102.5% | 102 | 102 | (dry-run) |
| 1 | follow_up | 4,715 | 5,303 | -12.5% | 304 | 302 | (dry-run) |
| 2 | follow_up | 7,519 | 7,952 | -5.8% | 449 | 445 | (dry-run) |
| 3 | no_context | 7,743 | 7,983 | -3.1% | 465 | 457 | (dry-run) |
| 4 | follow_up | 10,015 | 10,391 | -3.8% | 592 | 586 | (dry-run) |
| 5 | follow_up | 10,826 | 11,152 | -3.0% | 643 | 637 | (dry-run) |
| 6 | new_context | 12,387 | 12,637 | -2.0% | 778 | 767 | (dry-run) |
| 7 | follow_up | 13,315 | 13,481 | -1.2% | 877 | 866 | (dry-run) |

### code_en_2ae1ac4f_00
- Type: `code` | Lang: `en` | Turns: `8`
- Overall: `(dry-run)`
- Mean reduction: `-42.49%`
- Degradation slope: `0.1325`

| Turn | Pattern | Full Tok | Compressed Tok | Reduction | Spans | Hot | Classification |
|------|---------|----------|----------------|-----------|-------|-----|----------------|
| 0 | no_context | 1,544 | 3,096 | -100.5% | 149 | 149 | (dry-run) |
| 1 | follow_up | 1,769 | 3,327 | -88.1% | 162 | 162 | (dry-run) |
| 2 | follow_up | 3,049 | 4,630 | -51.8% | 241 | 241 | (dry-run) |
| 3 | no_context | 4,204 | 5,816 | -38.3% | 311 | 311 | (dry-run) |
| 4 | follow_up | 6,685 | 7,976 | -19.3% | 416 | 407 | (dry-run) |
| 5 | new_context | 11,697 | 13,419 | -14.7% | 732 | 722 | (dry-run) |
| 6 | no_context | 12,133 | 13,858 | -14.2% | 759 | 749 | (dry-run) |
| 7 | follow_up | 13,292 | 15,009 | -12.9% | 826 | 816 | (dry-run) |

### code_en_34589790_01
- Type: `code` | Lang: `en` | Turns: `8`
- Overall: `(dry-run)`
- Mean reduction: `-21.26%`
- Degradation slope: `0.0889`

| Turn | Pattern | Full Tok | Compressed Tok | Reduction | Spans | Hot | Classification |
|------|---------|----------|----------------|-----------|-------|-----|----------------|
| 0 | no_context | 472 | 944 | -100.0% | 73 | 73 | (dry-run) |
| 1 | new_context | 2,534 | 3,056 | -20.6% | 271 | 271 | (dry-run) |
| 2 | follow_up | 4,543 | 5,106 | -12.4% | 382 | 379 | (dry-run) |
| 3 | no_context | 6,614 | 7,210 | -9.0% | 502 | 497 | (dry-run) |
| 4 | follow_up | 6,679 | 7,277 | -8.9% | 510 | 505 | (dry-run) |
| 5 | follow_up | 9,344 | 10,004 | -7.1% | 666 | 659 | (dry-run) |
| 6 | no_context | 11,274 | 11,984 | -6.3% | 782 | 775 | (dry-run) |
| 7 | follow_up | 12,972 | 13,724 | -5.8% | 904 | 896 | (dry-run) |

### code_en_34589790_00
- Type: `code` | Lang: `en` | Turns: `8`
- Overall: `(dry-run)`
- Mean reduction: `-34.21%`
- Degradation slope: `0.1104`

| Turn | Pattern | Full Tok | Compressed Tok | Reduction | Spans | Hot | Classification |
|------|---------|----------|----------------|-----------|-------|-----|----------------|
| 0 | no_context | 2,355 | 4,711 | -100.0% | 266 | 266 | (dry-run) |
| 1 | new_context | 4,362 | 6,788 | -55.6% | 463 | 459 | (dry-run) |
| 2 | follow_up | 6,204 | 7,990 | -28.8% | 586 | 560 | (dry-run) |
| 3 | no_context | 7,291 | 9,675 | -32.7% | 668 | 655 | (dry-run) |
| 4 | follow_up | 9,222 | 11,616 | -26.0% | 791 | 774 | (dry-run) |
| 5 | follow_up | 10,148 | 10,754 | -6.0% | 868 | 793 | (dry-run) |
| 6 | no_context | 10,252 | 12,623 | -23.1% | 881 | 862 | (dry-run) |
| 7 | new_context | 12,460 | 12,644 | -1.5% | 1062 | 964 | (dry-run) |

### code_en_d7cfd257_02
- Type: `code` | Lang: `en` | Turns: `8`
- Overall: `(dry-run)`
- Mean reduction: `-42.26%`
- Degradation slope: `0.1200`

| Turn | Pattern | Full Tok | Compressed Tok | Reduction | Spans | Hot | Classification |
|------|---------|----------|----------------|-----------|-------|-----|----------------|
| 0 | no_context | 1,413 | 2,864 | -102.7% | 185 | 185 | (dry-run) |
| 1 | follow_up | 2,233 | 3,684 | -65.0% | 225 | 225 | (dry-run) |
| 2 | follow_up | 2,894 | 4,340 | -50.0% | 248 | 248 | (dry-run) |
| 3 | no_context | 3,793 | 5,254 | -38.5% | 307 | 307 | (dry-run) |
| 4 | follow_up | 4,646 | 6,099 | -31.3% | 342 | 342 | (dry-run) |
| 5 | follow_up | 5,348 | 6,809 | -27.3% | 382 | 382 | (dry-run) |
| 6 | no_context | 5,603 | 7,066 | -26.1% | 401 | 401 | (dry-run) |
| 7 | new_context | 11,346 | 11,030 | 2.8% | 706 | 643 | (dry-run) |

### code_en_ffc1cd2e_01
- Type: `code` | Lang: `en` | Turns: `8`
- Overall: `(dry-run)`
- Mean reduction: `-16.67%`
- Degradation slope: `0.1158`

| Turn | Pattern | Full Tok | Compressed Tok | Reduction | Spans | Hot | Classification |
|------|---------|----------|----------------|-----------|-------|-----|----------------|
| 0 | initial_context | 793 | 1,556 | -96.2% | 91 | 91 | (dry-run) |
| 1 | new_context | 1,724 | 2,314 | -34.2% | 187 | 177 | (dry-run) |
| 2 | new_context | 3,438 | 3,848 | -11.9% | 281 | 264 | (dry-run) |
| 3 | new_context | 4,201 | 4,487 | -6.8% | 320 | 299 | (dry-run) |
| 4 | new_context | 6,506 | 6,620 | -1.8% | 528 | 498 | (dry-run) |
| 5 | follow_up | 8,462 | 7,984 | 5.7% | 610 | 564 | (dry-run) |
| 6 | no_context | 10,204 | 9,528 | 6.6% | 702 | 646 | (dry-run) |
| 7 | follow_up | 10,899 | 10,318 | 5.3% | 741 | 685 | (dry-run) |

### code_en_ffc1cd2e_02
- Type: `code` | Lang: `en` | Turns: `8`
- Overall: `(dry-run)`
- Mean reduction: `-20.64%`
- Degradation slope: `0.1672`

| Turn | Pattern | Full Tok | Compressed Tok | Reduction | Spans | Hot | Classification |
|------|---------|----------|----------------|-----------|-------|-----|----------------|
| 0 | no_context | 1,045 | 2,139 | -104.7% | 122 | 122 | (dry-run) |
| 1 | follow_up | 1,781 | 2,875 | -61.4% | 163 | 162 | (dry-run) |
| 2 | follow_up | 3,437 | 4,562 | -32.7% | 231 | 229 | (dry-run) |
| 3 | new_context | 6,106 | 7,267 | -19.0% | 446 | 429 | (dry-run) |
| 4 | new_context | 8,498 | 7,996 | 5.9% | 522 | 494 | (dry-run) |
| 5 | follow_up | 9,253 | 7,990 | 13.7% | 588 | 536 | (dry-run) |
| 6 | no_context | 9,476 | 7,985 | 15.7% | 608 | 548 | (dry-run) |
| 7 | follow_up | 9,670 | 7,986 | 17.4% | 628 | 562 | (dry-run) |

### rag_en_bc7c7d79_00
- Type: `rag` | Lang: `en` | Turns: `8`
- Overall: `(dry-run)`
- Mean reduction: `-23.35%`
- Degradation slope: `0.0700`

| Turn | Pattern | Full Tok | Compressed Tok | Reduction | Spans | Hot | Classification |
|------|---------|----------|----------------|-----------|-------|-----|----------------|
| 0 | initial_context | 2,594 | 4,094 | -57.8% | 259 | 259 | (dry-run) |
| 1 | follow_up | 4,867 | 6,481 | -33.2% | 440 | 437 | (dry-run) |
| 2 | follow_up | 5,339 | 6,994 | -31.0% | 478 | 474 | (dry-run) |
| 3 | no_context | 6,067 | 7,804 | -28.6% | 527 | 523 | (dry-run) |
| 4 | new_context | 10,901 | 12,122 | -11.2% | 831 | 765 | (dry-run) |
| 5 | follow_up | 11,210 | 12,913 | -15.2% | 855 | 803 | (dry-run) |
| 6 | no_context | 13,279 | 14,689 | -10.6% | 995 | 929 | (dry-run) |
| 7 | new_context | 18,599 | 18,448 | 0.8% | 1285 | 1161 | (dry-run) |

### rag_en_8311fbae_02
- Type: `rag` | Lang: `en` | Turns: `8`
- Overall: `(dry-run)`
- Mean reduction: `-40.69%`
- Degradation slope: `0.0625`

| Turn | Pattern | Full Tok | Compressed Tok | Reduction | Spans | Hot | Classification |
|------|---------|----------|----------------|-----------|-------|-----|----------------|
| 0 | initial_context | 5,128 | 9,042 | -76.3% | 587 | 577 | (dry-run) |
| 1 | follow_up | 6,331 | 10,052 | -58.8% | 684 | 664 | (dry-run) |
| 2 | follow_up | 8,571 | 11,843 | -38.2% | 857 | 800 | (dry-run) |
| 3 | no_context | 9,780 | 13,119 | -34.1% | 947 | 865 | (dry-run) |
| 4 | follow_up | 12,464 | 16,761 | -34.5% | 1138 | 1077 | (dry-run) |
| 5 | follow_up | 16,484 | 21,162 | -28.4% | 1372 | 1292 | (dry-run) |
| 6 | no_context | 17,084 | 21,712 | -27.1% | 1417 | 1317 | (dry-run) |
| 7 | follow_up | 18,057 | 23,135 | -28.1% | 1488 | 1383 | (dry-run) |

### rag_en_8311fbae_00
- Type: `rag` | Lang: `en` | Turns: `8`
- Overall: `(dry-run)`
- Mean reduction: `-23.87%`
- Degradation slope: `0.0563`

| Turn | Pattern | Full Tok | Compressed Tok | Reduction | Spans | Hot | Classification |
|------|---------|----------|----------------|-----------|-------|-----|----------------|
| 0 | initial_context | 3,648 | 6,410 | -75.7% | 740 | 740 | (dry-run) |
| 1 | new_context | 9,330 | 11,207 | -20.1% | 1111 | 1070 | (dry-run) |
| 2 | follow_up | 10,852 | 12,957 | -19.4% | 1218 | 1181 | (dry-run) |
| 3 | new_context | 14,521 | 16,825 | -15.9% | 1475 | 1441 | (dry-run) |
| 4 | follow_up | 15,810 | 18,343 | -16.0% | 1565 | 1526 | (dry-run) |
| 5 | follow_up | 16,148 | 18,595 | -15.2% | 1595 | 1530 | (dry-run) |
| 6 | no_context | 16,242 | 18,670 | -14.9% | 1605 | 1529 | (dry-run) |
| 7 | follow_up | 17,287 | 19,656 | -13.7% | 1667 | 1586 | (dry-run) |

### rag_en_8311fbae_01
- Type: `rag` | Lang: `en` | Turns: `8`
- Overall: `(dry-run)`
- Mean reduction: `-25.44%`
- Degradation slope: `0.1427`

| Turn | Pattern | Full Tok | Compressed Tok | Reduction | Spans | Hot | Classification |
|------|---------|----------|----------------|-----------|-------|-----|----------------|
| 0 | no_context | 337 | 679 | -101.5% | 58 | 58 | (dry-run) |
| 1 | follow_up | 431 | 770 | -78.6% | 68 | 68 | (dry-run) |
| 2 | follow_up | 1,476 | 1,817 | -23.1% | 131 | 131 | (dry-run) |
| 3 | new_context | 5,727 | 5,935 | -3.6% | 448 | 441 | (dry-run) |
| 4 | follow_up | 8,124 | 8,155 | -0.4% | 596 | 570 | (dry-run) |
| 5 | new_context | 13,253 | 13,262 | -0.1% | 912 | 856 | (dry-run) |
| 6 | no_context | 14,456 | 14,199 | 1.8% | 1008 | 928 | (dry-run) |
| 7 | follow_up | 16,696 | 16,364 | 2.0% | 1179 | 1089 | (dry-run) |

### rag_en_bc7c7d79_01
- Type: `rag` | Lang: `en` | Turns: `8`
- Overall: `(dry-run)`
- Mean reduction: `-21.47%`
- Degradation slope: `0.1079`

| Turn | Pattern | Full Tok | Compressed Tok | Reduction | Spans | Hot | Classification |
|------|---------|----------|----------------|-----------|-------|-----|----------------|
| 0 | no_context | 727 | 1,535 | -111.1% | 97 | 97 | (dry-run) |
| 1 | new_context | 5,561 | 6,715 | -20.8% | 407 | 402 | (dry-run) |
| 2 | follow_up | 5,870 | 7,056 | -20.2% | 431 | 426 | (dry-run) |
| 3 | no_context | 7,939 | 8,874 | -11.8% | 572 | 553 | (dry-run) |
| 4 | new_context | 13,259 | 13,029 | 1.7% | 861 | 802 | (dry-run) |
| 5 | follow_up | 14,133 | 14,530 | -2.8% | 928 | 879 | (dry-run) |
| 6 | no_context | 15,250 | 15,726 | -3.1% | 1030 | 975 | (dry-run) |
| 7 | follow_up | 15,911 | 16,495 | -3.7% | 1079 | 1021 | (dry-run) |

### rag_en_d3cdf5ad_01
- Type: `rag` | Lang: `en` | Turns: `8`
- Overall: `(dry-run)`
- Mean reduction: `-23.86%`
- Degradation slope: `0.0484`

| Turn | Pattern | Full Tok | Compressed Tok | Reduction | Spans | Hot | Classification |
|------|---------|----------|----------------|-----------|-------|-----|----------------|
| 0 | initial_context | 3,246 | 5,537 | -70.6% | 389 | 389 | (dry-run) |
| 1 | follow_up | 7,299 | 8,844 | -21.2% | 585 | 562 | (dry-run) |
| 2 | follow_up | 10,337 | 12,079 | -16.9% | 716 | 697 | (dry-run) |
| 3 | new_context | 14,054 | 16,525 | -17.6% | 1028 | 1000 | (dry-run) |
| 4 | follow_up | 14,424 | 16,724 | -16.0% | 1048 | 1016 | (dry-run) |
| 5 | follow_up | 15,250 | 17,763 | -16.5% | 1091 | 1058 | (dry-run) |
| 6 | no_context | 15,332 | 17,639 | -15.0% | 1098 | 1059 | (dry-run) |
| 7 | follow_up | 15,450 | 18,112 | -17.2% | 1110 | 1074 | (dry-run) |

### rag_en_cb9ea221_01
- Type: `rag` | Lang: `en` | Turns: `8`
- Overall: `(dry-run)`
- Mean reduction: `-29.23%`
- Degradation slope: `0.1106`

| Turn | Pattern | Full Tok | Compressed Tok | Reduction | Spans | Hot | Classification |
|------|---------|----------|----------------|-----------|-------|-----|----------------|
| 0 | no_context | 770 | 1,602 | -108.0% | 86 | 86 | (dry-run) |
| 1 | follow_up | 1,923 | 2,768 | -43.9% | 154 | 153 | (dry-run) |
| 2 | follow_up | 3,017 | 3,805 | -26.1% | 225 | 216 | (dry-run) |
| 3 | no_context | 6,028 | 7,064 | -17.2% | 356 | 344 | (dry-run) |
| 4 | follow_up | 7,872 | 8,827 | -12.1% | 465 | 438 | (dry-run) |
| 5 | follow_up | 10,799 | 11,717 | -8.5% | 627 | 582 | (dry-run) |
| 6 | no_context | 11,596 | 12,803 | -10.4% | 672 | 627 | (dry-run) |
| 7 | new_context | 14,356 | 15,432 | -7.5% | 844 | 774 | (dry-run) |

### rag_en_4e9f48aa_01
- Type: `rag` | Lang: `en` | Turns: `8`
- Overall: `(dry-run)`
- Mean reduction: `-10.45%`
- Degradation slope: `0.0814`

| Turn | Pattern | Full Tok | Compressed Tok | Reduction | Spans | Hot | Classification |
|------|---------|----------|----------------|-----------|-------|-----|----------------|
| 0 | no_context | 61 | 117 | -91.8% | 7 | 7 | (dry-run) |
| 1 | follow_up | 1,669 | 1,599 | 4.2% | 145 | 143 | (dry-run) |
| 2 | follow_up | 3,083 | 3,067 | 0.5% | 224 | 219 | (dry-run) |
| 3 | new_context | 7,045 | 8,000 | -13.6% | 568 | 549 | (dry-run) |
| 4 | follow_up | 8,577 | 8,000 | 6.7% | 641 | 547 | (dry-run) |
| 5 | new_context | 12,007 | 11,558 | 3.7% | 889 | 763 | (dry-run) |
| 6 | no_context | 13,168 | 12,243 | 7.0% | 967 | 808 | (dry-run) |
| 7 | follow_up | 13,707 | 13,771 | -0.5% | 1000 | 882 | (dry-run) |

### rag_en_4e9f48aa_00
- Type: `rag` | Lang: `en` | Turns: `8`
- Overall: `(dry-run)`
- Mean reduction: `-18.99%`
- Degradation slope: `0.1232`

| Turn | Pattern | Full Tok | Compressed Tok | Reduction | Spans | Hot | Classification |
|------|---------|----------|----------------|-----------|-------|-----|----------------|
| 0 | no_context | 110 | 218 | -98.2% | 16 | 16 | (dry-run) |
| 1 | follow_up | 596 | 754 | -26.5% | 52 | 51 | (dry-run) |
| 2 | follow_up | 732 | 914 | -24.9% | 67 | 66 | (dry-run) |
| 3 | no_context | 840 | 1,019 | -21.3% | 73 | 72 | (dry-run) |
| 4 | follow_up | 3,134 | 3,234 | -3.2% | 338 | 334 | (dry-run) |
| 5 | follow_up | 6,783 | 6,541 | 3.6% | 669 | 661 | (dry-run) |
| 6 | new_context | 8,819 | 7,998 | 9.3% | 881 | 748 | (dry-run) |
| 7 | follow_up | 12,710 | 11,533 | 9.3% | 1204 | 1061 | (dry-run) |

### rag_en_bc7c7d79_02
- Type: `rag` | Lang: `en` | Turns: `7`
- Overall: `(dry-run)`
- Mean reduction: `-27.50%`
- Degradation slope: `0.0949`

| Turn | Pattern | Full Tok | Compressed Tok | Reduction | Spans | Hot | Classification |
|------|---------|----------|----------------|-----------|-------|-----|----------------|
| 0 | no_context | 2,068 | 4,157 | -101.0% | 287 | 287 | (dry-run) |
| 1 | new_context | 7,388 | 8,570 | -16.0% | 587 | 561 | (dry-run) |
| 2 | follow_up | 8,262 | 9,619 | -16.4% | 654 | 627 | (dry-run) |
| 3 | no_context | 9,379 | 10,908 | -16.3% | 756 | 727 | (dry-run) |
| 4 | follow_up | 10,040 | 11,518 | -14.7% | 805 | 764 | (dry-run) |
| 5 | follow_up | 10,800 | 12,191 | -12.9% | 864 | 801 | (dry-run) |
| 6 | no_context | 11,805 | 13,591 | -15.1% | 925 | 880 | (dry-run) |

### rag_en_cb9ea221_02
- Type: `rag` | Lang: `en` | Turns: `8`
- Overall: `(dry-run)`
- Mean reduction: `-32.08%`
- Degradation slope: `0.1118`

| Turn | Pattern | Full Tok | Compressed Tok | Reduction | Spans | Hot | Classification |
|------|---------|----------|----------------|-----------|-------|-----|----------------|
| 0 | no_context | 796 | 1,712 | -115.1% | 89 | 89 | (dry-run) |
| 1 | new_context | 3,556 | 4,854 | -36.5% | 263 | 256 | (dry-run) |
| 2 | follow_up | 4,580 | 6,100 | -33.2% | 347 | 340 | (dry-run) |
| 3 | no_context | 6,240 | 7,958 | -27.5% | 442 | 434 | (dry-run) |
| 4 | follow_up | 7,179 | 8,282 | -15.4% | 504 | 452 | (dry-run) |
| 5 | follow_up | 9,020 | 9,653 | -7.0% | 617 | 530 | (dry-run) |
| 6 | no_context | 10,497 | 11,221 | -6.9% | 697 | 599 | (dry-run) |
| 7 | follow_up | 11,166 | 12,845 | -15.0% | 735 | 669 | (dry-run) |

### rag_en_4e9f48aa_03
- Type: `rag` | Lang: `en` | Turns: `8`
- Overall: `(dry-run)`
- Mean reduction: `-42.59%`
- Degradation slope: `0.1053`

| Turn | Pattern | Full Tok | Compressed Tok | Reduction | Spans | Hot | Classification |
|------|---------|----------|----------------|-----------|-------|-----|----------------|
| 0 | no_context | 827 | 1,701 | -105.7% | 79 | 79 | (dry-run) |
| 1 | follow_up | 1,517 | 2,464 | -62.4% | 111 | 111 | (dry-run) |
| 2 | new_context | 3,716 | 5,337 | -43.6% | 268 | 265 | (dry-run) |
| 3 | no_context | 3,925 | 5,519 | -40.6% | 276 | 273 | (dry-run) |
| 4 | follow_up | 4,815 | 6,446 | -33.9% | 324 | 317 | (dry-run) |
| 5 | follow_up | 7,414 | 8,000 | -7.9% | 465 | 402 | (dry-run) |
| 6 | no_context | 8,735 | 10,708 | -22.6% | 533 | 522 | (dry-run) |
| 7 | new_context | 11,137 | 13,814 | -24.0% | 716 | 704 | (dry-run) |

### rag_en_bc7c7d79_03
- Type: `rag` | Lang: `en` | Turns: `6`
- Overall: `(dry-run)`
- Mean reduction: `-57.13%`
- Degradation slope: `0.0749`

| Turn | Pattern | Full Tok | Compressed Tok | Reduction | Spans | Hot | Classification |
|------|---------|----------|----------------|-----------|-------|-----|----------------|
| 0 | initial_context | 5,319 | 9,554 | -79.6% | 572 | 572 | (dry-run) |
| 1 | follow_up | 6,193 | 10,544 | -70.3% | 639 | 638 | (dry-run) |
| 2 | follow_up | 7,310 | 11,206 | -53.3% | 742 | 714 | (dry-run) |
| 3 | no_context | 7,971 | 12,255 | -53.7% | 791 | 767 | (dry-run) |
| 4 | follow_up | 8,731 | 12,356 | -41.5% | 850 | 763 | (dry-run) |
| 5 | follow_up | 9,736 | 14,054 | -44.4% | 911 | 874 | (dry-run) |

### rag_en_aa6a1f55_01
- Type: `rag` | Lang: `en` | Turns: `8`
- Overall: `(dry-run)`
- Mean reduction: `-46.44%`
- Degradation slope: `0.0804`

| Turn | Pattern | Full Tok | Compressed Tok | Reduction | Spans | Hot | Classification |
|------|---------|----------|----------------|-----------|-------|-----|----------------|
| 0 | no_context | 1,881 | 3,834 | -103.8% | 198 | 198 | (dry-run) |
| 1 | follow_up | 3,944 | 5,960 | -51.1% | 309 | 304 | (dry-run) |
| 2 | follow_up | 4,254 | 6,289 | -47.8% | 333 | 328 | (dry-run) |
| 3 | no_context | 4,905 | 6,990 | -42.5% | 378 | 373 | (dry-run) |
| 4 | follow_up | 6,100 | 8,150 | -33.6% | 439 | 424 | (dry-run) |
| 5 | follow_up | 6,883 | 9,105 | -32.3% | 492 | 473 | (dry-run) |
| 6 | new_context | 9,158 | 11,895 | -29.9% | 661 | 635 | (dry-run) |
| 7 | follow_up | 9,444 | 12,315 | -30.4% | 681 | 653 | (dry-run) |

### rag_en_aa6a1f55_00
- Type: `rag` | Lang: `en` | Turns: `8`
- Overall: `(dry-run)`
- Mean reduction: `-26.59%`
- Degradation slope: `0.0881`

| Turn | Pattern | Full Tok | Compressed Tok | Reduction | Spans | Hot | Classification |
|------|---------|----------|----------------|-----------|-------|-----|----------------|
| 0 | no_context | 295 | 589 | -99.7% | 25 | 25 | (dry-run) |
| 1 | follow_up | 1,269 | 1,591 | -25.4% | 66 | 66 | (dry-run) |
| 2 | follow_up | 1,463 | 1,796 | -22.8% | 84 | 84 | (dry-run) |
| 3 | no_context | 2,241 | 2,528 | -12.8% | 122 | 120 | (dry-run) |
| 4 | new_context | 4,308 | 5,237 | -21.6% | 300 | 291 | (dry-run) |
| 5 | follow_up | 6,190 | 7,140 | -15.3% | 401 | 389 | (dry-run) |
| 6 | no_context | 8,253 | 8,594 | -4.1% | 512 | 468 | (dry-run) |
| 7 | follow_up | 8,563 | 9,513 | -11.1% | 536 | 508 | (dry-run) |

### rag_en_aa6a1f55_02
- Type: `rag` | Lang: `en` | Turns: `8`
- Overall: `(dry-run)`
- Mean reduction: `-49.50%`
- Degradation slope: `0.0965`

| Turn | Pattern | Full Tok | Compressed Tok | Reduction | Spans | Hot | Classification |
|------|---------|----------|----------------|-----------|-------|-----|----------------|
| 0 | no_context | 782 | 1,710 | -118.7% | 103 | 103 | (dry-run) |
| 1 | new_context | 3,057 | 4,609 | -50.8% | 273 | 270 | (dry-run) |
| 2 | follow_up | 3,343 | 4,988 | -49.2% | 293 | 290 | (dry-run) |
| 3 | no_context | 3,748 | 5,351 | -42.8% | 318 | 314 | (dry-run) |
| 4 | follow_up | 4,048 | 5,737 | -41.7% | 337 | 333 | (dry-run) |
| 5 | follow_up | 4,184 | 5,881 | -40.6% | 348 | 344 | (dry-run) |
| 6 | no_context | 5,267 | 6,974 | -32.4% | 400 | 396 | (dry-run) |
| 7 | new_context | 7,387 | 8,853 | -19.9% | 551 | 521 | (dry-run) |

### rag_en_3257a434_03
- Type: `rag` | Lang: `en` | Turns: `8`
- Overall: `(dry-run)`
- Mean reduction: `-18.02%`
- Degradation slope: `0.0883`

| Turn | Pattern | Full Tok | Compressed Tok | Reduction | Spans | Hot | Classification |
|------|---------|----------|----------------|-----------|-------|-----|----------------|
| 0 | no_context | 294 | 587 | -99.7% | 39 | 39 | (dry-run) |
| 1 | new_context | 2,525 | 2,835 | -12.3% | 253 | 252 | (dry-run) |
| 2 | follow_up | 2,664 | 2,955 | -10.9% | 260 | 259 | (dry-run) |
| 3 | new_context | 4,273 | 4,534 | -6.1% | 373 | 372 | (dry-run) |
| 4 | follow_up | 5,050 | 5,285 | -4.7% | 428 | 426 | (dry-run) |
| 5 | follow_up | 5,762 | 5,978 | -3.8% | 480 | 478 | (dry-run) |
| 6 | no_context | 6,387 | 6,607 | -3.4% | 527 | 525 | (dry-run) |
| 7 | follow_up | 6,570 | 6,790 | -3.4% | 546 | 544 | (dry-run) |

### rag_en_3257a434_01
- Type: `rag` | Lang: `en` | Turns: `8`
- Overall: `(dry-run)`
- Mean reduction: `-63.85%`
- Degradation slope: `0.1056`

| Turn | Pattern | Full Tok | Compressed Tok | Reduction | Spans | Hot | Classification |
|------|---------|----------|----------------|-----------|-------|-----|----------------|
| 0 | no_context | 1,374 | 3,053 | -122.2% | 235 | 235 | (dry-run) |
| 1 | follow_up | 2,344 | 4,163 | -77.6% | 307 | 307 | (dry-run) |
| 2 | follow_up | 2,436 | 4,238 | -74.0% | 317 | 317 | (dry-run) |
| 3 | no_context | 2,663 | 4,508 | -69.3% | 334 | 333 | (dry-run) |
| 4 | new_context | 4,476 | 6,567 | -46.7% | 495 | 489 | (dry-run) |
| 5 | follow_up | 5,380 | 7,584 | -41.0% | 546 | 539 | (dry-run) |
| 6 | no_context | 5,414 | 7,643 | -41.2% | 550 | 543 | (dry-run) |
| 7 | follow_up | 5,726 | 7,953 | -38.9% | 571 | 562 | (dry-run) |

### rag_en_3257a434_02
- Type: `rag` | Lang: `en` | Turns: `8`
- Overall: `(dry-run)`
- Mean reduction: `-34.58%`
- Degradation slope: `0.1313`

| Turn | Pattern | Full Tok | Compressed Tok | Reduction | Spans | Hot | Classification |
|------|---------|----------|----------------|-----------|-------|-----|----------------|
| 0 | no_context | 69 | 138 | -100.0% | 20 | 20 | (dry-run) |
| 1 | follow_up | 230 | 405 | -76.1% | 36 | 36 | (dry-run) |
| 2 | follow_up | 614 | 878 | -43.0% | 65 | 65 | (dry-run) |
| 3 | no_context | 957 | 1,204 | -25.8% | 101 | 99 | (dry-run) |
| 4 | follow_up | 2,835 | 3,030 | -6.9% | 266 | 259 | (dry-run) |
| 5 | follow_up | 4,152 | 4,491 | -8.2% | 335 | 325 | (dry-run) |
| 6 | no_context | 4,620 | 4,994 | -8.1% | 375 | 365 | (dry-run) |
| 7 | new_context | 5,590 | 6,070 | -8.6% | 485 | 471 | (dry-run) |

### rag_en_aa6a1f55_03
- Type: `rag` | Lang: `en` | Turns: `5`
- Overall: `(dry-run)`
- Mean reduction: `-45.10%`
- Degradation slope: `0.0565`

| Turn | Pattern | Full Tok | Compressed Tok | Reduction | Spans | Hot | Classification |
|------|---------|----------|----------------|-----------|-------|-----|----------------|
| 0 | initial_context | 2,119 | 3,323 | -56.8% | 217 | 217 | (dry-run) |
| 1 | follow_up | 2,310 | 3,486 | -50.9% | 233 | 232 | (dry-run) |
| 2 | follow_up | 2,720 | 3,897 | -43.3% | 255 | 254 | (dry-run) |
| 3 | no_context | 2,988 | 4,210 | -40.9% | 280 | 278 | (dry-run) |
| 4 | follow_up | 3,800 | 5,076 | -33.6% | 325 | 322 | (dry-run) |
