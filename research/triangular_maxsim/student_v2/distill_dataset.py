"""Build distillation dataset for the TRACE v2 biaffine student.

Reads v1 teacher signals from three sources:

1. **Audit log JSONL** - per-call teacher scores captured by
   ``latence_trace.middleware.audit_log``.  These carry the per-channel
   scores, the top-k support attributions, and the final band, which is
   exactly the signal the student needs.
2. **Veracier curated variants** - gold bands from
   ``proof_bundle_v1/variants.curated.jsonl`` for the 118 audited
   enterprise cases.
3. **External benchmarks** - HaluEval QA/Summ + RAGTruth QA/Summ rows
   that went through the v1 API, as stored under
   ``proof_bundle_v1/external_bench/*_rows.jsonl``.  These give us both
   the v1 teacher output *and* the gold label, so the student can be
   pulled toward the gold on the weak cases the teacher got wrong.

Output is a JSONL shard per split where each row carries:

    {
      "response_text": "...",
      "evidence_text": "...",           # flattened raw_context
      "turn_band": "green" | "amber" | "red",
      "turn_score": float,              # v1 teacher head
      "gold_band": "green" | "red" | null,
      "support_units": [...],           # v1 teacher top-k positions
      "source": "audit_log" | "veracier" | "halueval_qa" | ...,
      "pair_id": optional[str],         # shared across grounded/hallucinated
      "phi_channels": {                 # cheap lexical features
        "exact_overlap": float,
        "numeric_overlap": float,
        "identifier_overlap": float,
        "source_type": str,
      }
    }

Pair IDs are emitted for HaluEval QA (grounded_answer + hallucinated_answer
share a question) and for Veracier (perfect + wrong share a case_id) so
the training loop can build ranking pairs without extra bookkeeping.

The script is intentionally boring - no GPU, no torch dep, no model
weights touched.  Run it whenever the audit log rolls over and you want
to rebuild the student corpus.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import re
from collections import defaultdict
from typing import Iterator


ROOT = pathlib.Path(__file__).resolve().parents[3]
PROOF_DIR = ROOT / "data/veracier-industries/proof_bundle_v1"


# ---- Lexical feature helpers -------------------------------------------

_NUM = re.compile(r"[-+]?\d+(?:[.,]\d+)?")
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*")
_WORD = re.compile(r"[A-Za-z0-9_]+")


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _WORD.findall(text or "")]


def _overlap(a: list[str], b: list[str]) -> float:
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    inter = len(sa & sb)
    return inter / max(len(sa), 1)


def _numeric_overlap(a: str, b: str) -> float:
    aa = set(_NUM.findall(a or ""))
    bb = set(_NUM.findall(b or ""))
    if not aa:
        return 0.0
    return len(aa & bb) / len(aa)


def _identifier_overlap(a: str, b: str) -> float:
    aa = set(_IDENT.findall(a or ""))
    bb = set(_IDENT.findall(b or ""))
    if not aa:
        return 0.0
    return len(aa & bb) / len(aa)


def _detect_source_type(text: str) -> str:
    if "```" in text or "def " in text:
        return "code"
    if text.strip().startswith(("##", "###", "# ")):
        return "markdown"
    if "passage " in text.lower():
        return "passage_enum"
    return "prose"


def _phi(response: str, evidence: str) -> dict:
    r_tok = _tokens(response)
    e_tok = _tokens(evidence)
    return {
        "exact_overlap": round(_overlap(r_tok, e_tok), 4),
        "numeric_overlap": round(_numeric_overlap(response, evidence), 4),
        "identifier_overlap": round(_identifier_overlap(response, evidence), 4),
        "source_type": _detect_source_type(evidence),
    }


# ---- Readers ------------------------------------------------------------


def iter_audit_log(path: pathlib.Path) -> Iterator[dict]:
    if not path.exists():
        return
    with path.open() as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            req = rec.get("request") or {}
            resp = rec.get("response") or {}
            response_text = req.get("response_text") or ""
            raw_context = req.get("raw_context") or ""
            if isinstance(raw_context, list):
                raw_context = "\n".join(str(x) for x in raw_context)
            band = resp.get("band") or resp.get("risk_band") or "unknown"
            if isinstance(band, dict):
                band = band.get("value", "unknown")
            yield {
                "response_text": response_text,
                "evidence_text": raw_context,
                "turn_band": str(band).lower(),
                "turn_score": resp.get("groundedness") or resp.get("groundedness_v2"),
                "gold_band": None,
                "support_units": resp.get("support_units", []),
                "source": "audit_log",
                "pair_id": None,
                "phi_channels": _phi(response_text, raw_context),
            }


def iter_veracier(path: pathlib.Path) -> Iterator[dict]:
    if not path.exists():
        return
    with path.open() as fh:
        for line in fh:
            try:
                v = json.loads(line)
            except json.JSONDecodeError:
                continue
            response_text = v.get("response_text") or v.get("answer") or ""
            raw_context = v.get("raw_context") or v.get("context") or ""
            if isinstance(raw_context, list):
                raw_context = "\n".join(str(x) for x in raw_context)
            gold = (v.get("expected_band") or v.get("label") or "").lower()
            case_id = v.get("case_id") or v.get("row_id")
            yield {
                "response_text": response_text,
                "evidence_text": raw_context,
                "turn_band": gold,
                "turn_score": None,
                "gold_band": gold,
                "support_units": [],
                "source": "veracier",
                "pair_id": case_id,
                "phi_channels": _phi(response_text, raw_context),
            }


def iter_external(path: pathlib.Path) -> Iterator[dict]:
    if not path.exists():
        return
    with path.open() as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            response_text = r.get("response_text") or ""
            raw_context = r.get("raw_context") or ""
            if isinstance(raw_context, list):
                raw_context = "\n".join(str(x) for x in raw_context)
            band = (r.get("band") or "").lower()
            gold = (r.get("expected_band") or r.get("expected_label") or "").lower()
            if gold == "hallucinated":
                gold = "red"
            if gold == "grounded" or gold == "faithful":
                gold = "green"
            row_id = r.get("row_id") or ""
            # HaluEval: grounded + hallucinated share a numeric prefix
            pair_id = row_id.replace("_halluc", "")
            yield {
                "response_text": response_text,
                "evidence_text": raw_context,
                "turn_band": band,
                "turn_score": r.get("score"),
                "gold_band": gold,
                "support_units": [],
                "source": r.get("bench") or path.stem,
                "pair_id": pair_id,
                "phi_channels": _phi(response_text, raw_context),
            }


# ---- Splitter -----------------------------------------------------------


def split_rows(rows: list[dict], seed: int) -> tuple[list[dict], list[dict], list[dict]]:
    """Pair-aware train/val/test split.  All rows sharing a ``pair_id``
    go to the same split so the ranking loss does not leak."""
    rng = random.Random(seed)
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        key = r.get("pair_id") or f"_solo_{id(r)}"
        buckets[key].append(r)
    keys = list(buckets.keys())
    rng.shuffle(keys)
    n_test = max(1, int(0.10 * len(keys)))
    n_val = max(1, int(0.10 * len(keys)))
    test_k = set(keys[:n_test])
    val_k = set(keys[n_test : n_test + n_val])
    train: list[dict] = []
    val: list[dict] = []
    test: list[dict] = []
    for k in keys:
        dest = test if k in test_k else (val if k in val_k else train)
        dest.extend(buckets[k])
    return train, val, test


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-log", type=pathlib.Path, default=None)
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=ROOT / "research/triangular_maxsim/student_v2/data",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []

    if args.audit_log:
        all_rows.extend(iter_audit_log(args.audit_log))

    all_rows.extend(iter_veracier(PROOF_DIR / "variants.curated.jsonl"))
    for bench_file in sorted((PROOF_DIR / "external_bench").glob("*_rows.jsonl")):
        all_rows.extend(iter_external(bench_file))

    print(f"collected rows: {len(all_rows)}")
    train, val, test = split_rows(all_rows, args.seed)
    for name, rows in (("train", train), ("val", val), ("test", test)):
        path = args.out / f"{name}.jsonl"
        with path.open("w") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {path}: {len(rows)} rows")


if __name__ == "__main__":
    main()
