#!/usr/bin/env bash
# Rerun TRACE against the curated variants in this bundle.
#
# Requires a live TRACE endpoint.  Two paths:
#
# 1. Hosted:  export LATENCE_API_KEY=<hosted-key>; use default endpoint
# 2. Self-hosted:  export LATENCE_API_BASE=https://your-runpod-url; export LATENCE_API_KEY=<local jwt>
#
# After this completes, re-run the report stage to produce a curated-proof
# report in this same directory.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
# ^ $BUNDLE/rerun_against_curated.sh -> latence-trace repo root

BUNDLE=data/veracier-industries/proof_bundle_v1
RUN=data/veracier-industries/trace_bench_runs/veracier_trace_validation_curated

mkdir -p "$RUN"
# Stage curated variants in a sibling run directory so the existing
# harness trace stage picks them up unchanged.
cp "$BUNDLE/variants.curated.jsonl" "$RUN/variants.jsonl"
cp "$BUNDLE/evidence_manifest.json" "$RUN/evidence_manifest.json" 2>/dev/null || true
cp "$BUNDLE/corpus_summary.json" "$RUN/corpus_summary.json" 2>/dev/null || true
cp "$BUNDLE/refinement_summary.json" "$RUN/refinement_summary.json" 2>/dev/null || true
cp -r "$BUNDLE/evidence_packs/." "$RUN/evidence_packs/" 2>/dev/null || true

python3 scripts/bench_veracier_rag_validation.py \
    --stage=trace \
    --output-dir "$RUN" \
    --scope=full-use-cases \
    --trace-concurrency=32 \
    --trace-profile=both

python3 scripts/bench_veracier_rag_validation.py \
    --stage=report \
    --output-dir "$RUN" \
    --scope=full-use-cases

cp "$RUN/proof_report.md" "$BUNDLE/proof_report_curated.md"
cp "$RUN/validation_summary.json" "$BUNDLE/validation_summary_curated.json"
cp "$RUN/trace_results.jsonl" "$BUNDLE/trace_results_curated.jsonl"

( cd "$BUNDLE" && find . -type f ! -name 'MANIFEST.sha256' \
    | sort | xargs sha256sum > MANIFEST.sha256 )

echo
echo "Curated run complete. Bundle updated at $BUNDLE"
