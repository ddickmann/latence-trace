"""Analyse external-bench weak cases and print failure-mode breakdown."""

from __future__ import annotations

import json
import pathlib
from collections import Counter
from collections.abc import Iterable
from typing import Any

BENCH_DIR = pathlib.Path(
    "data/veracier-industries/proof_bundle_v1/external_bench"
)


def rows(path: pathlib.Path) -> Iterable[dict[str, Any]]:
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def classify(r: dict[str, Any]) -> str:
    """Return the failure mode for this row, or 'ok' if correctly scored."""
    expected = (r.get("expected_band") or r.get("expected_label") or "").lower()
    band = (r.get("band") or "").lower()
    if expected == "grounded":
        expected = "green"
    if expected == "hallucinated":
        expected = "red"
    if band == expected:
        return "ok"
    if expected == "green" and band == "red":
        return "fp_red_on_grounded"
    if expected == "green" and band == "amber":
        return "amber_on_grounded"
    if expected == "red" and band == "green":
        return "fn_green_on_hallucinated"
    if expected == "red" and band == "amber":
        return "amber_on_hallucinated"
    return f"other_{expected}_{band}"


def summarize(file: pathlib.Path) -> None:
    counts: Counter[str] = Counter()
    failures: list[dict[str, Any]] = []
    for r in rows(file):
        mode = classify(r)
        counts[mode] += 1
        if mode != "ok":
            failures.append(r)

    total = sum(counts.values())
    print(f"\n{'='*80}")
    print(f"FILE: {file.name}    total={total}")
    print(f"{'='*80}")
    for mode, n in counts.most_common():
        pct = 100 * n / total if total else 0
        print(f"  {mode:<32s} {n:4d}  ({pct:5.1f}%)")

    # Print sample of each non-ok mode (limited)
    for mode in counts:
        if mode == "ok":
            continue
        print(f"\n--- examples of {mode} (up to 3) ---")
        samples = [f for f in failures if classify(f) == mode][:3]
        for s in samples:
            print(f"  score={s.get('score'):.3f} band={s.get('band')} expected={s.get('expected_band')}")
            q = (s.get("question") or "")[:120]
            a = (s.get("response_text") or "")[:140]
            ctx = (s.get("raw_context") or "")
            if isinstance(ctx, list):
                ctx = " | ".join(str(c) for c in ctx[:2])
            ctx = str(ctx)[:200]
            print(f"    Q: {q}")
            print(f"    A: {a}")
            print(f"    C: {ctx}")


if __name__ == "__main__":
    for file in sorted(BENCH_DIR.glob("*_rows.jsonl")):
        summarize(file)
