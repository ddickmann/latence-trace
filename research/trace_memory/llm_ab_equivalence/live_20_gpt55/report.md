# LLM A/B Equivalence Report

- Model: `gpt-5.5`
- Cases: `20`
- Pass gate: `FAIL`
- Classification counts: `{"compressed_better": 5, "equivalent": 3, "minor_delta": 10, "regression": 2}`
- Mean token reduction: `0.1458`
- Median token reduction: `0.0030`

## Cases

### rag_00_halueval_qa_halueval_qa_78_halluc
- Type: `rag_verification`
- Classification: `minor_delta`
- Tokens: `312` full -> `357` compressed (-14.42% reduction)
- Judge equivalence: `minor_delta`; better: `A`
- Rationale: Both answers correctly judge the candidate answer unsupported because the source says Bizarre was published by Dennis Publishing and Fortean Times was previously published by John Brown Publishing but is now published by Dennis Publishing Ltd. Answer A is slightly better because it states the direct contradiction more explicitly and gives the supported publisher as Dennis Publishing / Dennis Publishing Ltd.

### trace_04_middle_818e476e
- Type: `coding_trace`
- Classification: `compressed_better`
- Tokens: `209591` full -> `112096` compressed (46.52% reduction)
- Judge equivalence: `material_delta`; better: `B`
- Rationale: Answer A is materially off-target: it discusses the mathematical story of rroq158 versus RaBitQ/TurboQuant and does not address the user's latency question about rroq4/rroq4_riem versus rroq158. Answer B directly addresses the target request, gives the right high-level distinction that the shared Riemannian decomposition is not the same as the residual scoring implementation, and proposes the correct kind of next-step breakdown. However, B includes many unsupported exact numbers, report paths, diagnostic plan IDs, and implementation details not present in the supplied evidence, so it should be penalized for hallucination. Despite those issues, B is clearly better because it answers the requested user_query while A does not.

### trace_03_late_2edf16de
- Type: `coding_trace`
- Classification: `regression`
- Tokens: `85370` full -> `85779` compressed (-0.48% reduction)
- Judge equivalence: `material_delta`; better: `A`
- Rationale: Answer A is substantially closer to the source evidence because it preserves the exact file `portal/src/components/marketing/dataflow-bridge.tsx`, the exact intended section structure, and the exact copy/capability/deployment labels from the assistant's response. Answer B captures the general product intent but replaces the actual implemented copy with new copy and adds unsupported structural details like a separate Retrieval Package layer and Bring-your-own-LLM deployment strip. Both answers fail the required exact image/path/ID terms, but A is materially more faithful to the trace.

### trace_06_early_2edf16de
- Type: `coding_trace`
- Classification: `minor_delta`
- Tokens: `85370` full -> `85779` compressed (-0.48% reduction)
- Judge equivalence: `minor_delta`; better: `B`
- Rationale: Both answers accurately identify the concrete request: localhost on RunPod is not reachable from the user's local browser, and the solution was to expose the already-running Next.js portal dev server on port 3000 through a Cloudflare/cloudflared tunnel at the exact URL https://visible-performances-definition-robert.trycloudflare.com without pushing to GitHub/Vercel. Answer B is slightly better because it more explicitly captures the user's no-GitHub/no-Vercel constraint and notes both the dev server and tunnel must keep running. Answer A adds the useful Next.js on-demand compile note, but is slightly less complete on the user's stated deployment constraint. Both miss the required literal term "user_query".

### trace_00_late_ae824cdc
- Type: `coding_trace`
- Classification: `minor_delta`
- Tokens: `85620` full -> `85363` compressed (0.30% reduction)
- Judge equivalence: `minor_delta`; better: `B`
- Rationale: Both answers correctly state that `voyager-index[full]` exists, that it is defined in `pyproject.toml`, that the README already mentions it in lower sections, and that the real issue is the top hero snippet still using `voyager-index[server,shard,gpu]`. Answer B is slightly better because it names the exact optional-dependency symbol, identifies the README locations more concretely, and avoids adding unsupported validation-test claims. Answer A includes useful exact version constraints but adds a speculative test recommendation and is slightly less precise about the schema symbol.

### trace_08_middle_a2982a04
- Type: `coding_trace`
- Classification: `compressed_better`
- Tokens: `244019` full -> `112097` compressed (54.06% reduction)
- Judge equivalence: `material_delta`; better: `A`
- Rationale: Answer A is much closer because it identifies the correct concrete request, most plan todo IDs, major constraints, and several relevant files. However, it is materially flawed because it invents implementation state and numeric results not present in the attached plan, and it misses many exact required symbols and the full vscode-remote plan URI. Answer B answers a completely different later question about BEIR benchmark table boilerplate and omits virtually every critical fact from the target request. Therefore the answers are materially different, and A is better despite hallucinations.

### rag_01_halueval_summ_halueval_summ_1
- Type: `rag_verification`
- Classification: `equivalent`
- Tokens: `833` full -> `903` compressed (-8.40% reduction)
- Judge equivalence: `equivalent`; better: `tie`
- Rationale: Both answers correctly judge the candidate answer as supported and cite the key supporting facts from the source: Palestinian Authority became the 123rd ICC member, ICC jurisdiction over alleged crimes in Palestinian territories including since June 13, 2014, Israel and U.S. opposition, and the preliminary examination paving the way for possible war crimes investigations against Israelis. Neither introduces unsupported claims or omits a critical fact needed to verify the candidate answer.

### rag_05_halueval_summ_halueval_summ_56_halluc
- Type: `rag_verification`
- Classification: `minor_delta`
- Tokens: `1509` full -> `1610` compressed (-6.69% reduction)
- Judge equivalence: `minor_delta`; better: `B`
- Rationale: Both answers correctly judge the candidate answer unsupported and identify the two main unsupported claims: Lynch leaving due to "creative differences" rather than money/deal-point issues, and Showtime moving forward under a new director despite no such director being mentioned. Answer B is slightly better because it also flags "remaining episodes" as unsupported by the source.

### rag_02_ragtruth_qa_ragtruth_qa_7
- Type: `rag_verification`
- Classification: `minor_delta`
- Tokens: `1533` full -> `1804` compressed (-17.68% reduction)
- Judge equivalence: `minor_delta`; better: `A`
- Rationale: Both answers correctly judge the candidate answer as supported and cite the key supported facts, including the required '24 hours' claim, derivation from succinic/tartaric acid, long-acting vs shorter-acting metoprolol forms, and once-daily vs at-least-twice-daily dosing. Answer A is slightly better because it is more concise and includes passage-level citations, while Answer B is somewhat more verbose and lacks explicit passage identifiers, though its claims are still supported.

### trace_01_middle_b19327a6
- Type: `coding_trace`
- Classification: `compressed_better`
- Tokens: `245775` full -> `112097` compressed (54.39% reduction)
- Judge equivalence: `material_delta`; better: `B`
- Rationale: Both answers correctly identify that more than the train script was gone and list the main damaged NLI/data/test/doc files. Answer B is better because it more accurately captures the later recovery facts: `pyproject.toml`, the `.gitignore` root cause, the distinction between Git-restorable files and uncommitted lost files, and the correct missing `src/voyager_zero/compass/t5_nli.py`. Answer A includes more unsupported or wrong exact file names, especially `scripts/train_t5_nli.py`, `scripts/validate_bidir_nli.py`, and `scripts/audit_composite_dataset.py`. Both fail to include the required literal term `user_query`.

### rag_03_halueval_qa_halueval_qa_29_halluc
- Type: `rag_verification`
- Classification: `minor_delta`
- Tokens: `1273` full -> `1297` compressed (-1.89% reduction)
- Judge equivalence: `equivalent`; better: `tie`
- Rationale: Both answers correctly mark the candidate answer as unsupported and identify the contradiction: the source evidence says 750 Seventh Avenue and 101 Park Avenue are in New York City, not Albany, New York. Both cite the key supporting facts sufficiently.

### trace_05_early_d4fbfaec
- Type: `coding_trace`
- Classification: `regression`
- Tokens: `110982` full -> `107889` compressed (2.79% reduction)
- Judge equivalence: `material_delta`; better: `A`
- Rationale: Answer A is materially more complete. It captures the concrete request, constraints, implementation files, matrix, cache/schema change, held-out pass-through, command, artifacts, actual four-cell results, gate failure, and completion state. Answer B captures several plan constraints accurately, including more of the held-out field list, but stops before the actual run outcome and even truncates the command, omitting the decisive implementation state and benchmark results. Neither answer fully preserves every exact path/symbol from the plan, but A is clearly better supported by the source evidence.

### rag_06_ragtruth_qa_ragtruth_qa_41
- Type: `rag_verification`
- Classification: `minor_delta`
- Tokens: `2667` full -> `2962` compressed (-11.06% reduction)
- Judge equivalence: `minor_delta`; better: `A`
- Rationale: Both answers correctly mark the candidate answer as unsupported because parchment paper or aluminum foil is not in the source. Answer A is better because it also flags the unsupported 'If desired' optional cheese wording. Answer B provides useful supported facts, but it misses that unsupported/uncited optionality claim.

### trace_09_late_d4fbfaec
- Type: `coding_trace`
- Classification: `minor_delta`
- Tokens: `110982` full -> `107882` compressed (2.79% reduction)
- Judge equivalence: `minor_delta`; better: `A`
- Rationale: Both answers correctly capture the core state: `latence-trace` pushed at `d0fa073`, `gateway` pushed at `148f7fd`, both clean and matching remotes, and no Wrangler redeploy needed because there were no new gateway changes in that turn. Answer A is slightly better because it preserves more exact operational details from the trace, including `/workspace/gateway`, `campegd1dctnx2`, the three trace endpoints, the `8/8 PASS` bench result, and the exact redeploy command. Answer B is more concise and includes the live `/rollup` HTTP 200 check, but it drops more exact IDs and test context. Neither answer includes the required literal term `user_query`.

### rag_07_halueval_qa_halueval_qa_56
- Type: `rag_verification`
- Classification: `equivalent`
- Tokens: `839` full -> `929` compressed (-10.73% reduction)
- Judge equivalence: `equivalent`; better: `tie`
- Rationale: Both answers correctly judge the candidate answer as supported and cite the key evidence: the tour's first show was at Estadio Ciudad de La Plata, also known as Estadio Único, and the stadium is owned by the Province of Buenos Aires. Answer A includes the supported location detail; Answer B is slightly more concise. No material difference.

### trace_11_early_4c3b24b5
- Type: `coding_trace`
- Classification: `compressed_better`
- Tokens: `372174` full -> `112096` compressed (69.88% reduction)
- Judge equivalence: `material_delta`; better: `B`
- Rationale: Answer B is materially closer because it identifies the actual request: default configurable context chunking and token-level max/argmax merging rather than scalar averaging. Answer A answers a different feature, response chunking, and therefore misses the core request. However, B still invents or imports unsupported implementation state, file paths, tests, and artifacts. Both omit the required term user_query, and neither cleanly states that at the target turn this was a proposal/next-step rather than an already implemented feature.

### trace_07_middle_1551b31d
- Type: `coding_trace`
- Classification: `minor_delta`
- Tokens: `194865` full -> `112094` compressed (42.48% reduction)
- Judge equivalence: `minor_delta`; better: `B`
- Rationale: Both answers are status summaries rather than code. Both omit the required literal `user_query` and both introduce unsupported commit/test/status facts. Answer B is still better because it states the key operational constraint more clearly: upstream engine repos are not integrated yet, wrappers stay on the NoUpstream path, and real binding should not start. It also gives a clearer continuation path and more exact surfaces, although some of those exact paths are unsupported or conflict with the visible source. Answer A is more concise and has fewer invented implementation details, but it misses the critical NoUpstream/upgraded-engine constraint and gives a thinner next-step explanation.

### trace_10_late_a2982a04
- Type: `coding_trace`
- Classification: `minor_delta`
- Tokens: `244019` full -> `112097` compressed (54.06% reduction)
- Judge equivalence: `minor_delta`; better: `B`
- Rationale: Both answers correctly identify `benchmarks/beir_benchmark.py` as the script that produced the table outputs and provide the main command to run it. B is better because it stays closer to the evidence: it includes supported constants/functions like `DATASETS`, `TOP_K`, `format_results_table`, `format_markdown_table`, `format_comparison_table`, and the supported output file `benchmarks/beir_results.jsonl`. A is useful but adds unsupported claims about `benchmarks/beir_cache/` and exact internal symbols not established in the source evidence. Neither answer includes the required literal term `user_query`.

### trace_02_middle_0d3f01d0
- Type: `coding_trace`
- Classification: `compressed_better`
- Tokens: `196730` full -> `112096` compressed (43.02% reduction)
- Judge equivalence: `material_delta`; better: `B`
- Rationale: Both answers capture the core request: do not stop at the `mt5_gliner` parity-script result of `F1=1.0000` and `1.46x`; continue to `mmbert_gliner` and recover prior chart-level performance around `11x`. Answer B is better because it is more focused and has fewer unsupported later-trace claims. Answer A adds substantial unsupported future-state details, exact artifacts, a later user acceptance statement, and rejected workaround claims that are not in the provided source evidence. Both answers miss the required exact strings `validated=the` and `user_query`.

### rag_04_halueval_qa_halueval_qa_29
- Type: `rag_verification`
- Classification: `equivalent`
- Tokens: `1104` full -> `1179` compressed (-6.79% reduction)
- Judge equivalence: `equivalent`; better: `tie`
- Rationale: Answer A and Answer B are identical. Both correctly judge the candidate answer as supported and cite the key primary-source facts that 750 Seventh Avenue is in New York City and 101 Park Avenue is in New York City, New York. No unsupported or contradicted claims are present.
