# File-level attribution — `v2_full_file_attribution`

- input: `v2_full_gpu_scorer_rescore.json`
- dead-weight threshold: `0.20`
- mean dead-weight ratio: **0.316**

## Per-case (first 30)

| id | tier | chunker | scorer | #files | dw ratio | dead-weight paths (first 3) |
|---|---|---|---|---:|---:|---|
| base_01_f0afeaf2_t952_f1__correct | correct | sentence_packed | gte_only_gpu_slim | 24 | 0.625 | session_notes/turn_0029.md, session_notes/turn_0048.md, session_notes/turn_0059.md |
| base_01_f0afeaf2_t952_f1__wrong | wrong | sentence_packed | gte_only_gpu_slim | 24 | 0.542 | session_notes/turn_0029.md, session_notes/turn_0048.md, session_notes/turn_0107.md |
| base_01_f0afeaf2_t952_f1__ambiguous | ambiguous | sentence_packed | gte_only_gpu_slim | 24 | 0.625 | session_notes/turn_0029.md, session_notes/turn_0048.md, session_notes/turn_0059.md |
| base_02_61e1f5a1_t258_f0__correct | correct | sentence_packed | gte_only_gpu_slim | 21 | 0.429 | session_notes/turn_0039.md, session_notes/turn_0090.md, session_notes/turn_0103.md |
| base_02_61e1f5a1_t258_f0__wrong | wrong | sentence_packed | gte_only_gpu_slim | 21 | 0.429 | session_notes/turn_0039.md, session_notes/turn_0083.md, session_notes/turn_0090.md |
| base_02_61e1f5a1_t258_f0__ambiguous | ambiguous | sentence_packed | gte_only_gpu_slim | 21 | 0.476 | session_notes/turn_0039.md, session_notes/turn_0083.md, session_notes/turn_0090.md |
| base_03_f0afeaf2_t2082_f0__correct | correct | sentence_packed | gte_only_gpu_slim | 10 | 0.000 | - |
| base_03_f0afeaf2_t2082_f0__wrong | wrong | sentence_packed | gte_only_gpu_slim | 10 | 0.000 | - |
| base_03_f0afeaf2_t2082_f0__ambiguous | ambiguous | sentence_packed | gte_only_gpu_slim | 10 | 0.000 | - |
| base_04_8a0f6f68_t521_f3__correct | correct | sentence_packed | gte_only_gpu_slim | 19 | 0.579 | session_notes/turn_0020.md, session_notes/turn_0239.md, session_notes/turn_0248.md |
| base_04_8a0f6f68_t521_f3__wrong | wrong | sentence_packed | gte_only_gpu_slim | 19 | 0.579 | session_notes/turn_0020.md, session_notes/turn_0239.md, session_notes/turn_0248.md |
| base_04_8a0f6f68_t521_f3__ambiguous | ambiguous | sentence_packed | gte_only_gpu_slim | 19 | 0.579 | session_notes/turn_0020.md, session_notes/turn_0239.md, session_notes/turn_0248.md |
| base_05_126e38fc_t1159_f4__correct | correct | sentence_packed | gte_only_gpu_slim | 25 | 0.240 | session_notes/turn_0155.md, session_notes/turn_0230.md, session_notes/turn_0511.md |
| base_05_126e38fc_t1159_f4__wrong | wrong | sentence_packed | gte_only_gpu_slim | 25 | 0.280 | session_notes/turn_0230.md, session_notes/turn_0511.md, session_notes/turn_0720.md |
| base_05_126e38fc_t1159_f4__ambiguous | ambiguous | sentence_packed | gte_only_gpu_slim | 25 | 0.320 | plans/recover_quota_and_oom_dc7a2663.plan.md, session_notes/turn_0155.md, session_notes/turn_0230.md |
| base_06_f0afeaf2_t1771_f0__correct | correct | sentence_packed | gte_only_gpu_slim | 26 | 0.462 | session_notes/turn_0059.md, session_notes/turn_0064.md, session_notes/turn_0146.md |
| base_06_f0afeaf2_t1771_f0__wrong | wrong | sentence_packed | gte_only_gpu_slim | 26 | 0.538 | plans/groundedness_beta_docs_9b4d1731.plan.md, plans/chunked_rawcontext_groundedness_358d3172.plan.md, session_notes/turn_0059.md |
| base_06_f0afeaf2_t1771_f0__ambiguous | ambiguous | sentence_packed | gte_only_gpu_slim | 26 | 0.462 | session_notes/turn_0059.md, session_notes/turn_0064.md, session_notes/turn_0146.md |
| base_07_7b7fb621_t328_f1__correct | correct | sentence_packed | gte_only_gpu_slim | 11 | 0.182 | session_notes/turn_0107.md, session_notes/turn_0311.md |
| base_07_7b7fb621_t328_f1__wrong | wrong | sentence_packed | gte_only_gpu_slim | 11 | 0.091 | session_notes/turn_0311.md |
| base_07_7b7fb621_t328_f1__ambiguous | ambiguous | sentence_packed | gte_only_gpu_slim | 11 | 0.182 | session_notes/turn_0107.md, session_notes/turn_0311.md |
| base_08_126e38fc_t1785_f0__correct | correct | sentence_packed | gte_only_gpu_slim | 30 | 0.467 | plans/recover_quota_and_oom_dc7a2663.plan.md, plans/production_surface_unlock_3629d92a.plan.md, session_notes/turn_0155.md |
| base_08_126e38fc_t1785_f0__wrong | wrong | sentence_packed | gte_only_gpu_slim | 30 | 0.500 | plans/recover_quota_and_oom_dc7a2663.plan.md, plans/production_surface_unlock_3629d92a.plan.md, plans/shard_surface_refresh_b1324dcb.plan.md |
| base_08_126e38fc_t1785_f0__ambiguous | ambiguous | sentence_packed | gte_only_gpu_slim | 30 | 0.433 | plans/production_surface_unlock_3629d92a.plan.md, session_notes/turn_0155.md, session_notes/turn_0261.md |
| base_09_0b1422bb_t571_f0__correct | correct | sentence_packed | gte_only_gpu_slim | 11 | 0.091 | session_notes/turn_0390.md |
| base_09_0b1422bb_t571_f0__wrong | wrong | sentence_packed | gte_only_gpu_slim | 11 | 0.091 | session_notes/turn_0390.md |
| base_09_0b1422bb_t571_f0__ambiguous | ambiguous | sentence_packed | gte_only_gpu_slim | 11 | 0.091 | session_notes/turn_0390.md |
| base_10_d7fd7127_t1579_f0__correct | correct | sentence_packed | gte_only_gpu_slim | 25 | 0.480 | session_notes/turn_0021.md, session_notes/turn_0177.md, session_notes/turn_0236.md |
| base_10_d7fd7127_t1579_f0__wrong | wrong | sentence_packed | gte_only_gpu_slim | 25 | 0.520 | session_notes/turn_0000.md, session_notes/turn_0021.md, session_notes/turn_0177.md |
| base_10_d7fd7127_t1579_f0__ambiguous | ambiguous | sentence_packed | gte_only_gpu_slim | 25 | 0.480 | session_notes/turn_0021.md, session_notes/turn_0177.md, session_notes/turn_0236.md |
