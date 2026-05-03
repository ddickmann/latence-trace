# LLM A/B Equivalence Report

- Model: `gpt-5.5`
- Cases: `1`
- Pass gate: `FAIL`
- Classification counts: `{"regression": 1}`
- Mean token reduction: `0.0277`
- Median token reduction: `0.0277`

## Cases

### trace_05_early_d4fbfaec
- Type: `coding_trace`
- Classification: `regression`
- Tokens: `110982` full -> `107907` compressed (2.77% reduction)
- Repair triggered: `True`
- Judge equivalence: `minor_delta`; better: `A`
- Rationale: Both answers accurately identify the user request, main constraints, files, matrix, cache schema bump, LateOn fallback, held-out pass-through, run command, artifacts, and reported results. B is slightly more concise and explicitly names `score_groundedness_response_chunked` and `phantom_api_precision@0.35`, but A includes the exact todo IDs, which matter because the target user specifically referenced existing todos and status handling. Neither answer invents material facts. Both miss some exact chunker/source-file details from the attached plan, especially `code_segmenter.py`, `pack_text_by_lines`, `segment_code_files`, `colgrep_parser.parse_code`, `CodeUnit.description()`, and `provider_token_limit(provider)`.
- Repair triggers: `["missing_exact_terms_in_compressed_answer"]`
- Repair packet: `8` excerpts, `4428` tokens
