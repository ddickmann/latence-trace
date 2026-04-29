"""Run the OOD probe against the live dev_app and emit a markdown report.

Usage:

    python3 scripts/run_ood_probe.py \
        --url http://127.0.0.1:8091 \
        --out data/corpus_classifier/proof_bundle_v3/OOD_PROBE.md

The probe pulls the out-of-distribution corpus from
``tests/integration/test_corpus_router_ood.py`` (single source of truth -
the test-time contract and the proof-bundle evidence stay aligned).

Each row is shown with:

* expected class
* predicted class
* decision source (``rule`` / ``classifier`` / ``explicit`` /
  ``fallback``)
* rule reason (if any) or classifier posterior
* correct / incorrect marker

Overall and per-class accuracy are emitted at the top of the report
alongside the gate thresholds, so a reader can decide at a glance
whether the router generalises.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

from tests.integration.test_corpus_router_ood import (  # type: ignore[import-not-found]
    OOD_CORPUS,
    _EXPLICIT_CORPUS_TYPE_CLASSES,
    OVERALL_GATE,
    PER_CLASS_GATE,
)


def _post(url: str, body: dict, timeout: int = 120) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8091")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    results: list[dict] = []
    started = time.time()
    for row in OOD_CORPUS:
        payload: dict[str, object] = {
            "query": row.query,
            "response": row.response,
            "raw_context": row.raw_context,
        }
        explicit = row.class_key in _EXPLICIT_CORPUS_TYPE_CLASSES
        if explicit:
            payload["corpus_type"] = row.class_key
        out = _post(f"{args.url}/runsync", {"input": payload})
        route = out.get("corpus_route") or {}
        predicted = route.get("corpus_type") or "unknown"
        results.append(
            {
                "expected": row.class_key,
                "predicted": predicted,
                "correct": predicted == row.class_key,
                "source": route.get("source"),
                "confidence": route.get("confidence"),
                "rule_reason": route.get("rule_reason"),
                "note": row.note,
                "explicit_corpus_type": explicit,
            }
        )
    duration_s = time.time() - started

    by_class: dict[str, list[dict]] = {}
    for r in results:
        by_class.setdefault(r["expected"], []).append(r)

    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    acc = correct / total if total else 0.0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        f.write("# Out-of-distribution probe\n\n")
        f.write(
            "All rows are hand-written, disjoint from the training corpora "
            "(Veracier, HaluEval, RAGTruth, Data2Text, transcripts_v2). "
            "Enterprise rows carry ``corpus_type`` explicitly because that "
            "class has no reliable text-level structural signature; all "
            "other rows arrive unmarked so the router's rule overlay and "
            "LR classifier are what decide the bundle.\n\n"
        )
        f.write(f"- Overall accuracy: **{correct}/{total} = {acc:.3f}** ")
        f.write(f"(gate: {OVERALL_GATE:.2f})\n")
        f.write(f"- Wall time: {duration_s:.2f}s against `{args.url}`\n")
        f.write(f"- Per-class gate: {PER_CLASS_GATE:.2f}\n\n")
        f.write("## Per-class accuracy\n\n")
        f.write("| class | correct | total | accuracy |\n")
        f.write("|---|---|---|---|\n")
        for cls, rows in sorted(by_class.items()):
            c = sum(1 for r in rows if r["correct"])
            n = len(rows)
            f.write(f"| `{cls}` | {c} | {n} | {c/n:.3f} |\n")
        f.write("\n## Row-level decisions\n\n")
        f.write("| expected | predicted | source | confidence | rule reason | note |\n")
        f.write("|---|---|---|---|---|---|\n")
        for r in results:
            tick = "OK" if r["correct"] else "FAIL"
            conf = (
                f"{r['confidence']:.3f}"
                if isinstance(r["confidence"], (float, int))
                else "n/a"
            )
            reason = r["rule_reason"] or "-"
            f.write(
                f"| `{r['expected']}` | `{r['predicted']}` ({tick}) | "
                f"`{r['source']}` | {conf} | `{reason}` | {r['note']} |\n"
            )
    print(
        f"wrote {args.out} (overall {correct}/{total} = {acc:.3f}, "
        f"duration {duration_s:.2f}s)"
    )


if __name__ == "__main__":
    main()
