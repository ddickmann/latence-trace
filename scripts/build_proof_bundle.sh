#!/usr/bin/env bash
# Build the v1 proof bundle deterministically.
#
# Produces ``data/veracier-industries/proof_bundle_v1/`` with:
#   - variants.curated.jsonl (committed, hand-curated)
#   - variants.jsonl (original, for diffing)
#   - trace_results.jsonl (honest baseline against the uncurated variants)
#   - proof_report.md + validation_summary.json (regenerated with red recall
#     and worker_timeout annotations)
#   - evidence_manifest.json, corpus_summary.json, refinement_summary.json
#   - evidence_packs/ (evidence snippets by use-case)
#   - curation_log.md
#   - MANIFEST.sha256 for everything above
#   - README.md and REPRODUCE.md
#
# To rerun against the curated variants (requires live TRACE worker):
#
#   export LATENCE_API_KEY=<hosted key or leave blank with --endpoint local>
#   bash data/veracier-industries/proof_bundle_v1/rerun_against_curated.sh
#
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

BUNDLE=data/veracier-industries/proof_bundle_v1
RUN=data/veracier-industries/trace_bench_runs/veracier_trace_validation_full_use_cases_20260428

mkdir -p "$BUNDLE" "$BUNDLE/evidence_packs"

echo "[1/6] Curating variants..."
python3 scripts/curate_veracier_variants.py >/dev/null
python3 scripts/append_marketing_use_cases.py >/dev/null

echo "[2/6] Verifying curated variants..."
python3 scripts/verify_curated_variants.py >/dev/null

echo "[3/6] Regenerating honest baseline report..."
python3 scripts/bench_veracier_rag_validation.py \
    --stage=report \
    --output-dir "$RUN" \
    --scope=full-use-cases >/dev/null

echo "[4/6] Copying provenance artefacts into bundle..."
cp "$RUN/variants.jsonl" "$BUNDLE/variants.jsonl"
cp "$RUN/trace_results.jsonl" "$BUNDLE/trace_results.jsonl"
cp "$RUN/proof_report.md" "$BUNDLE/proof_report.md"
cp "$RUN/validation_summary.json" "$BUNDLE/validation_summary.json"
cp "$RUN/evidence_manifest.json" "$BUNDLE/evidence_manifest.json" 2>/dev/null || true
cp "$RUN/corpus_summary.json" "$BUNDLE/corpus_summary.json" 2>/dev/null || true
cp "$RUN/refinement_summary.json" "$BUNDLE/refinement_summary.json" 2>/dev/null || true
cp -r "$RUN/evidence_packs/." "$BUNDLE/evidence_packs/" 2>/dev/null || true

echo "[5/6] Computing SHA-256 manifest..."
( cd "$BUNDLE" && find . -type f ! -name 'MANIFEST.sha256' \
    | sort \
    | xargs sha256sum > MANIFEST.sha256 )

echo "[6/6] Proof bundle written to $BUNDLE"
echo
echo "Files:"
ls -1 "$BUNDLE"
