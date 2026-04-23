# Coding-Agent Groundedness

This package runs a scorer-stacking experiment that tests whether the
`latence_trace.core.groundedness` scorer transfers to coding agents, with the
context for each case represented as code files and the response as a patch or
diff.

## What this runner evaluates

A **scorer-stack x chunker matrix** with two chunkers and five scorer
configurations per chunker (10 cells).

| dimension | options |
|---|---|
| chunker | `sentence_packed` (line-packed text baseline), `colgrep` (code-aware, `colgrep_parser`) |
| scorer | `gte_only`, `code_only`, `fuse_mean`, `fuse_max`, `fuse_min` |

Scorers:

- **Primary**: `lightonai/GTE-ModernColBERT-v1`. This is the existing
  RAG-style groundedness scorer; the coding benchmark does not change it.
- **Orthogonal**: `lightonai/LateOn-Code-edge`. Runs on the **same** chunks
  and response and produces a second per-unit score that is fused with the
  primary signal.

Chunks are shared across encoders: chunking happens once per `(case,
chunker)` at the GTE-ModernColBERT-v1 effective document budget (`256`
tokens). Both encoders then embed the same text windows, which is what makes
per-unit fusion index-aligned. LateOn-Code-edge's 1024-token headroom is
intentionally not exploited in v1 so fusion stays fair; exploiting it is a
v2 item.

## Scorer configurations

All five configs share the same support units, the same response chunks, and
the same coverage threshold (from the primary scorer). They differ only in
how the two encoders' per-unit signals are combined:

- `gte_only` — primary scorer alone (baseline).
- `code_only` — LateOn-Code-edge alone (reference).
- `fuse_mean` — per-unit average of GTE and code-edge `coverage_score`,
  `score` (attribution), and matched response tokens; per-case average for
  `reverse_context`, `consensus_hardened`, and max top-evidence.
- `fuse_max` — per-unit max of the two signals (optimistic).
- `fuse_min` — per-unit min of the two signals (pessimistic, precision-first).

`context_coverage_ratio` in every fused config is recomputed from the fused
`used` flags: a unit is used when its fused `coverage_score` clears the
coverage threshold.

## Held-out under fusion (precision-first intersection)

Every fused row carries three sets of held-out ids:

- `held_out_ids_gte` — units the primary scorer classified as
  `usage_state == "unused"`.
- `held_out_ids_code` — units the code-edge scorer independently classified
  as `unused`.
- `held_out_ids` (consensus) — the **intersection** of the above; a unit is
  fused-held-out only if both encoders agreed it was unused.
- `uncertain_ids_diff` — the symmetric difference (one encoder said
  unused, the other did not).

We do not re-run the upstream usage classifier on fused per-unit scores in
v1; intersection is a precision-first stand-in that matches the classifier's
spirit and is trivial to replace with a proper call later.

## Flagship cell and gate

The flagship go/no-go cell is `colgrep` x `fuse_mean` against the RAG-style
`reverse_context AUROC >= 0.90` gate. The headline paragraph calls out
whether `fuse_max` or `fuse_min` beats `fuse_mean` on the same chunker, so
the flagship rule can be swapped if a different rule clears the gate first.

The report also reports:

- `reverse_context AUROC` for each cell.
- `grounded context coverage mean` and `coverage delta` (grounded minus
  ungrounded).
- `unused delta` (ungrounded minus grounded `context_unused_ratio`).
- `phantom-API precision@threshold` on the hand-crafted `phantom_api`
  subcategory.
- Stacking lift: per chunker, `fuse_*` minus `gte_only` AUROC / unused
  delta / phantom precision — does the orthogonal encoder help?
- Chunker lift: per scorer, `colgrep` minus `sentence_packed` — does
  code-aware chunking help?
- Per-case held-out listing for the flagship cell with the three sets
  described above.

## Case bank

- Hand-crafted Python and TypeScript fixtures in
  [code_cases.py](code_cases.py): grounded edits, phantom-API
  hallucinations, entity swaps, parametric knowledge, partial groundedness,
  and deprecated-API negation flips.
- SWE-bench Lite pairs in [swebench_cases.py](swebench_cases.py): gold
  patches trimmed to their hunk windows, paired with deterministic
  corruptions from [corruptions.py](corruptions.py).

## Run

One command:

```bash
python -m research.triangular_maxsim.coding.coding_agent_validation --tag code_stack_v1
```

Outputs land under `research/triangular_maxsim/coding/artifacts/`:

- `<tag>.json` — full payload (cells, rows, summary, warnings, fusion
  sub-tables).
- `<tag>.md` — human-readable report with the 10-cell headline table,
  stacking-lift and chunker-lift delta tables, per-cell subcategory
  snapshots, and the flagship cell's three-way held-out listing.

Embedding caches live under `research/triangular_maxsim/coding/cache/` and
are keyed on chunker, token budget, `max_swebench_instances`, and whether
the code encoder was loaded. Bump `_CACHE_SCHEMA_VERSION` in
[coding_agent_validation.py](coding_agent_validation.py) to invalidate them.

### Fallbacks

- If `lightonai/LateOn-Code-edge` fails to load (no weights, no GPU,
  pylate import error, etc.), the runner logs a warning, drops the four
  non-`gte_only` scorer configs for the whole run, and ships a report with
  only the two `gte_only` rows. The GTE baseline always ships.
- If the code encoder is available but scoring fails on a specific case
  (e.g. OOM on a giant SWE-bench diff), that case keeps its `gte_only` row
  and its `code_only` / `fuse_*` rows are skipped.
- If `colgrep_parser` is not installed, the `colgrep` chunker is skipped
  with a warning and the `sentence_packed` cells still run.

### Restricting the matrix

To run a subset, for example to smoke-test the flagship cell:

```bash
python -m research.triangular_maxsim.coding.coding_agent_validation \
  --chunker colgrep \
  --scorer fuse_mean \
  --tag code_stack_v1_smoke
```

`--chunker` and `--scorer` can be repeated.

### Overriding the chunk budget and chunking tokenizer

Single-run mode accepts `--shared-chunk-tokens` and `--chunking-provider`. The
shared budget drives both support chunks and response chunks. The chunking
provider chooses whose tokenizer/limit the chunker respects:

- `--chunking-provider gte` (default): chunks are clamped at GTE's ~283-token
  effective budget. Safe because both encoders can ingest them in full.
- `--chunking-provider code`: chunks are clamped at LateOn-Code-edge's
  ~2047-token budget. Larger chunks are possible but GTE will silently
  truncate at encode time — useful as a diagnostic of "does more code context
  help LateOn even if GTE sees only the prefix?"

```bash
python -m research.triangular_maxsim.coding.coding_agent_validation \
  --shared-chunk-tokens 1024 \
  --chunking-provider code \
  --tag code_stack_v1_big_chunks
```

### Chunk-size sweep

To test multiple budgets in one pass and get side-by-side AUROC + score
distribution diagnostics:

```bash
python -m research.triangular_maxsim.coding.coding_agent_validation \
  --tag chunk_size_sweep_v1 \
  --sweep 64 --sweep 128 --sweep 256 \
  --sweep 512:code --sweep 1024:code
```

Each `--sweep` argument is `budget` or `budget:chunking_provider`. The runner
loads both encoders once, then replays the full 10-cell matrix per budget and
writes:

- `<tag>.md` / `<tag>.json` — the top-level sweep report with one headline
  row per `(budget, chunker, scorer)` and a flagship-cell row per budget.
- `<tag>__b<budget>_<provider>.md` / `.json` — full per-budget reports
  including score-distribution diagnostics and held-out tables.

### How to read the score-distribution diagnostics

Every single-run report now includes a "Score distribution diagnostics"
section. The fields that matter when AUROC is low:

- `saturation@0.95` — fraction of anchor cases with `reverse_context >= 0.95`.
  At 1.00 the scorer has pinned everything to the ceiling, so AUROC ranking
  is being decided by ties rather than real signal.
- `separation` — `mean(grounded) − mean(ungrounded)`. For text RAG this is
  typically `> 0.05`. If it's under ~0.01 the encoders cannot discriminate
  grounded from ungrounded regardless of how we chunk.
- `reverse_context_overlap` — `max(0, ungrounded_p90 − grounded_p10)`. A
  positive value means the two distributions overlap; a large value means
  AUROC cannot move without changing the signal itself.

The v1 sweep (`chunk_size_sweep_v1`) landed with `saturation@0.95 ≈ 1.0` and
`separation ≤ 0.008` across every budget from 64 to 1024 tokens, meaning
**chunk size is not the lever**. The signal itself saturates on code because
grounded and ungrounded responses share almost identical token surfaces
(same imports, same braces, same Python/JS keywords); ColBERT's token-level
MaxSim can't distinguish a correct variable name from a swapped one. The
next lever is the scoring signal (query-side weighting, response-token
entropy, response-only residual score), not the chunker.
