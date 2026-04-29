"""Combine synthetic + teacher-labeled rows and re-run the pair splitter.

After ``teacher_label`` finishes producing the per-benchmark JSONL
files, this script fuses them with the existing synthetic corpus and
writes a fresh ``data/train.jsonl`` / ``val.jsonl`` / ``test.jsonl``
that the training loop consumes directly.

HaluEval remains deliberately quarantined — we do not import it from
any source here.

Usage
-----

    python -m research.triangular_maxsim.student_v2.merge_corpus \
        --synthetic-train research/triangular_maxsim/student_v2/data/train.jsonl \
        --teacher-jsonls \
            research/triangular_maxsim/student_v2/data/teacher_ragtruth_train.jsonl \
            research/triangular_maxsim/student_v2/data/teacher_veracier.jsonl \
        --out-dir research/triangular_maxsim/student_v2/data_v2 \
        --seed 7

The output directory mirrors the layout expected by ``train.py``:
``train.jsonl``, ``val.jsonl``, ``test.jsonl``, ``ood_eval.jsonl``,
``manifest.json``.

Dedup
-----
We key on ``pair_row_id``; when a row appears in more than one source
file, the first one wins. This makes re-runs idempotent.
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
from collections import Counter
from typing import Iterable

from research.triangular_maxsim.student_v2.distill_dataset import split_rows

logger = logging.getLogger("trace.v2.merge_corpus")


_INT_TO_BAND = {0: "green", 1: "amber", 2: "red"}


def _normalize_row(row: dict) -> dict:
    """Harmonize the row shape so ``collate.build_example`` accepts it.

    Teacher-labeled rows emitted by older revisions of ``teacher_label``
    stored ``turn_band`` as an int (0/1/2). The synthetic lane and the
    downstream collate use the string name ("green"/"amber"/"red"). We
    rewrite the int form on the fly so both corpora merge cleanly.
    """

    tb = row.get("turn_band")
    if isinstance(tb, int):
        row = {**row, "turn_band": _INT_TO_BAND.get(tb, "amber")}
    gb = row.get("gold_band")
    if isinstance(gb, int):
        row["gold_band"] = _INT_TO_BAND.get(gb, "amber")
    return row


def _iter_jsonl(path: pathlib.Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield _normalize_row(json.loads(line))


def _load_all(paths: list[pathlib.Path]) -> list[dict]:
    seen: set[str] = set()
    rows: list[dict] = []
    per_source = Counter()
    for p in paths:
        if not p.exists():
            logger.warning("skipping missing file: %s", p)
            continue
        kept = 0
        for row in _iter_jsonl(p):
            rid = row.get("pair_row_id") or row.get("pair_id")
            if rid in seen:
                continue
            seen.add(rid)
            rows.append(row)
            kept += 1
        per_source[str(p)] = kept
        logger.info("  %s -> %d rows", p.name, kept)
    logger.info("loaded %d unique rows across %d files", len(rows), len(paths))
    return rows


def _write_split(split: list[dict], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in split:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    logger.info("wrote %s (%d rows)", path, len(split))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--synthetic-train",
        type=pathlib.Path,
        required=True,
        help="existing synthetic train.jsonl (already teacher-labeled by the synthetic lane)",
    )
    parser.add_argument(
        "--synthetic-ood",
        type=pathlib.Path,
        default=None,
        help="optional synthetic ood_eval.jsonl; carried through unchanged",
    )
    parser.add_argument(
        "--teacher-jsonls",
        type=pathlib.Path,
        nargs="+",
        required=True,
        help="one or more teacher-labeled jsonls to fold into the corpus",
    )
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    logger.info("merging synthetic + teacher-labeled rows")
    rows = _load_all([args.synthetic_train, *args.teacher_jsonls])

    if args.synthetic_ood is not None and args.synthetic_ood.exists():
        ood_rows = list(_iter_jsonl(args.synthetic_ood))
        for r in ood_rows:
            r.setdefault("split_hint", "ood_eval")
        logger.info("carrying %d OOD rows through from %s", len(ood_rows), args.synthetic_ood)
        rows.extend(ood_rows)

    # Source-level summary so the user can spot imbalances.
    by_source = Counter(r.get("source", "unknown") for r in rows)
    logger.info("by source: %s", dict(by_source.most_common()))
    by_band = Counter(r.get("gold_band", "unknown") for r in rows)
    logger.info("by gold_band: %s", dict(by_band.most_common()))

    splits = split_rows(rows, seed=args.seed)

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    _write_split(splits["train"], out / "train.jsonl")
    _write_split(splits["val"], out / "val.jsonl")
    _write_split(splits["test"], out / "test.jsonl")
    _write_split(splits["ood_eval"], out / "ood_eval.jsonl")

    manifest = {
        "seed": args.seed,
        "sources": dict(by_source),
        "bands": dict(by_band),
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "inputs": {
            "synthetic_train": str(args.synthetic_train),
            "synthetic_ood": str(args.synthetic_ood) if args.synthetic_ood else None,
            "teacher_jsonls": [str(p) for p in args.teacher_jsonls],
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info("wrote manifest: %s", out / "manifest.json")
    logger.info("split sizes: %s", manifest["split_sizes"])


if __name__ == "__main__":
    main()
