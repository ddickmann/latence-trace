"""Append the 4 marketing use cases to the curated variants file.

Reads ``marketing_use_cases.json`` and appends 12 extra rows (4 use
cases x 3 variants) to ``variants.curated.jsonl`` under the bundle.

The rows carry the same schema as the Veracier rows:

- ``example_id`` = ``{use_case_id}:{variant}``
- ``use_case_id`` / ``variant`` / ``expected_band``
- ``query_text`` / ``response_text``
- ``generation_archetype`` = marketing_claims
- ``raw_context`` built from ``selected_documents``
- ``curated_by`` + ``curation_reason`` + ``verticals``
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "data" / "veracier-industries" / "proof_bundle_v1"
MKT_PATH = BUNDLE / "marketing_use_cases.json"
CURATED = BUNDLE / "variants.curated.jsonl"

CURATOR = "latence-domain-review@v1"


def _build_raw_context(docs):
    parts = []
    for doc in docs:
        prefix = f"[{doc['doc_id']} | {doc['filename']}]"
        parts.append(f"{prefix}\n{doc['text']}")
    return "\n\n".join(parts)


def main() -> int:
    pack = json.loads(MKT_PATH.read_text(encoding="utf-8"))
    archetype = pack["archetype"]
    rows_out = []
    for uc in pack["use_cases"]:
        raw_ctx = _build_raw_context(uc["selected_documents"])
        doc_ids = [d["doc_id"] for d in uc["selected_documents"]]
        filenames = [d["filename"] for d in uc["selected_documents"]]
        for variant in uc["variants"]:
            example_id = f"{uc['use_case_id']}:{variant['variant']}"
            row = {
                "example_id": example_id,
                "use_case_id": uc["use_case_id"],
                "variant": variant["variant"],
                "question_id": uc["use_case_id"],
                "query_text": uc["question"],
                "response_text": variant["response_text"],
                "expected_band": variant["expected_band"],
                "expected_groundedness_range": {
                    "green": [0.80, 1.00],
                    "amber": [0.55, 0.79],
                    "red": [0.0, 0.54],
                }[variant["expected_band"]],
                "raw_context": raw_ctx,
                "context_doc_ids": doc_ids,
                "context_filenames": filenames,
                "generation_archetype": archetype,
                "mutation_type": variant["variant"],
                "claims": [
                    {
                        "claim_text": uc["selected_documents"][0]["text"].split(".")[0].strip() + ".",
                        "expected_label": "supported" if variant["variant"] == "perfect" else ("hedged" if variant["variant"] == "ambiguous" else "contradicted"),
                        "rationale": variant["curation_reason"],
                        "source_doc_ids": [doc_ids[0]],
                        "source_filenames": [filenames[0]],
                    }
                ],
                "coverage_notes": "Synthetic marketing-claim use case, additive to the Veracier corpus.",
                "utilization_notes": "Both documents are cited by the perfect variant.",
                "curated_by": CURATOR,
                "curation_reason": variant["curation_reason"],
                "curated_evidence_doc_ids": doc_ids,
                "verticals": uc.get("verticals", ["marketing"]),
                "synthetic_marketing": True,
            }
            rows_out.append(row)

    existing_lines = []
    if CURATED.exists():
        with CURATED.open("r", encoding="utf-8") as fh:
            existing_lines = [line for line in fh if line.strip()]

    # Drop any previous marketing rows so the append is idempotent.
    filtered = []
    for line in existing_lines:
        row = json.loads(line)
        if not row.get("synthetic_marketing"):
            filtered.append(line.rstrip("\n"))

    with CURATED.open("w", encoding="utf-8") as fh:
        for line in filtered:
            fh.write(line + "\n")
        for row in rows_out:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"appended {len(rows_out)} marketing rows to {CURATED}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
