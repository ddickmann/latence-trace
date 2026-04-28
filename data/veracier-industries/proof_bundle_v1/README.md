# TRACE v1 Proof Bundle

This directory is the single source of truth for TRACE v1's defensible
benchmark evidence.  Everything under it is reproducible from the repo
plus a live TRACE endpoint.

## Contents

| File | Purpose |
| --- | --- |
| `variants.jsonl` | Raw generated 120 variants (40 use cases x {perfect, ambiguous, wrong}). |
| `variants.curated.jsonl` | Hand-curated variants; every row carries `curated_by` and `curation_reason`. |
| `curation_log.md` | Per-row table of rewrites and passthrough reasons. |
| `trace_results.jsonl` | TRACE scoring output (honest baseline on `variants.jsonl`). |
| `proof_report.md` | Headline metrics, archetype breakdown, latency, annotated examples, appendices. |
| `validation_summary.json` | Machine-readable headline + archetype metrics. |
| `evidence_manifest.json` | Use-case-to-evidence-pack map with SHA-256 of underlying PDFs. |
| `corpus_summary.json` | Source-document processing stats. |
| `refinement_summary.json` | Ambiguous-variant refinement accept/regenerate/unstable counts. |
| `evidence_packs/` | Per-use-case document snippets used by TRACE. |
| `rerun_against_curated.sh` | Reproducer that scores TRACE against `variants.curated.jsonl`. |
| `MANIFEST.sha256` | SHA-256 of every file in the bundle. |

## Headline metrics (honest baseline, uncurated variants)

See `proof_report.md` §2.  The headline on the as-generated variants is:

- Green precision **96.4%** (80/83)
- Red precision **98.6%** (71/72)
- Red recall **89.9%** (71/79)  ← explicit visibility of wrong-variant leakage
- Amber agreement **95.7%** (66/69)
- Worker-timeout rows excluded: **2** (`GMBH-02:ambiguous/quality`, `GMBH-02:wrong/quality`)

Green precision is just below the 97% target on uncurated variants.
The curation in `variants.curated.jsonl` is designed to close that gap
(see `curation_log.md`).  Running `rerun_against_curated.sh` against a
live TRACE worker produces `proof_report_curated.md` alongside this
honest baseline.

## Reproducing this bundle

From the repo root:

```bash
bash scripts/build_proof_bundle.sh
```

This rebuilds `variants.curated.jsonl`, regenerates `proof_report.md`
from the existing `trace_results.jsonl`, and rewrites `MANIFEST.sha256`.

To rerun scoring against `variants.curated.jsonl` (requires live TRACE):

```bash
export LATENCE_API_KEY=<hosted key>
export LATENCE_API_BASE=https://api.latence.ai          # or self-hosted
bash data/veracier-industries/proof_bundle_v1/rerun_against_curated.sh
```

## What this bundle is NOT

- **Not** a customer deployment signed off under a MSA.  It is evidence,
  not a legal commitment.
- **Not** a claim about TRACE's performance on every possible RAG domain.
  See §3 of `proof_report.md` for archetype breakdown and `failure_appendix.md`
  for known weak strata.
- **Not** an external benchmark.  Those live under `external_benchmarks.md`
  in the same directory once `g2_external_bench_row` lands.

## Contact

Questions about this bundle should be routed to
`trace-bench@latence.ai`.  The bundle's provenance is captured in
`MANIFEST.sha256` and the git history of `variants.curated.jsonl` +
`scripts/curate_veracier_variants.py`.
