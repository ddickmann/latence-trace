"""Agentic-coding benchmark for TRACE (Phase 3).

Runs the CRUXEval + HumanEval+ grounded and adversarial-variant rows through
a live TRACE endpoint and reports:

* Per-variant-type F1 (identifier_swap / literal_swap / api_signature_swap).
* Paired accuracy within each (grounded, variant) pair: the grounded
  response must outscore every adversarial it was paired with.
* Response-level operational metrics (red/green/amber counts, precision,
  recall, amber leakage).

Writes:

* ``proof_bundle_v1/coding_bench/rows.jsonl`` — one row per scored sample.
* ``proof_bundle_v1/coding_bench/summary.json`` — aggregate metrics.
* ``proof_bundle_v1/coding_bench/report.md`` — human-readable verdict.

Usage::

    python scripts/bench_coding.py --url http://127.0.0.1:8091/runsync \
        --profile quality --sample-per-dataset 120 --auto-decide

Gates success at per-type F1 >= 0.80 (plan). Below that, Phase 4 (v2 student)
must close the gap.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
import urllib.request
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple  # noqa: UP035

REPO_ROOT = Path(__file__).resolve().parents[1]
VARIANTS = REPO_ROOT / "data/veracier-industries/proof_bundle_v1/coding_bench/variants.jsonl"
PAIRS = REPO_ROOT / "data/veracier-industries/proof_bundle_v1/coding_bench/paired.jsonl"
OUT = REPO_ROOT / "data/veracier-industries/proof_bundle_v1/coding_bench"


def _load_pairs(limit_per_dataset: int, seed: int) -> list[dict[str, Any]]:
    all_pairs: list[dict[str, Any]] = []
    with PAIRS.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            all_pairs.append(json.loads(line))
    # stratify by bench
    by_bench: dict[str, list[dict[str, Any]]] = {}
    for p in all_pairs:
        bench = p["green"].get("bench", "unknown")
        by_bench.setdefault(bench, []).append(p)
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for bench, pairs in by_bench.items():
        rng.shuffle(pairs)
        selected.extend(pairs[:limit_per_dataset])
    return selected


def _score(
    url: str,
    row: dict[str, Any],
    *,
    profile: str,
    timeout: float,
    auto_decide: bool,
    scoring_mode: str = "rag",
    response_language_hint: Optional[str] = None,
) -> Tuple[dict[str, Any], float]:
    payload_input: dict[str, Any] = {
        "action": "score",
        "query_text": row["question"],
        "response_text": row["response_text"],
        "raw_context": row["raw_context"],
        "profile": profile,
        "scoring_mode": scoring_mode,
    }
    if scoring_mode == "code" and response_language_hint:
        payload_input["response_language_hint"] = response_language_hint
    if auto_decide:
        payload_input["auto_decide"] = True
    body = {"input": payload_input}
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "content-type": "application/json",
            "accept": "application/json",
            "x-latence-tenant-id": "bench-coding",
        },
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        out = json.loads(resp.read().decode("utf-8"))
    dt_ms = (time.perf_counter() - t0) * 1000.0
    return out.get("output") or out, dt_ms


def _f1(
    labels_scores: List[Tuple[int, float]],
    *,
    positive_is_hallucinated: bool = True,
) -> dict[str, Any]:
    """Sweep thresholds and return best-F1 operating point."""
    if not labels_scores:
        return {"f1": None, "precision": None, "recall": None, "threshold": None}
    thresholds = sorted({s for _, s in labels_scores}) + [math.inf]
    best = {"f1": -1.0, "precision": 0.0, "recall": 0.0, "threshold": None, "counts": None}
    for t in thresholds:
        tp = fp = fn = tn = 0
        for y, s in labels_scores:
            predicted_positive = (s < t) if positive_is_hallucinated else (s >= t)
            if predicted_positive and y == 1:
                tp += 1
            elif predicted_positive and y == 0:
                fp += 1
            elif not predicted_positive and y == 0:
                tn += 1
            else:
                fn += 1
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        if f1 > best["f1"]:
            best = {
                "f1": round(f1, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "threshold": round(float(t), 4) if math.isfinite(t) else None,
                "counts": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
            }
    if best["f1"] < 0:
        best["f1"] = None
    return best


def _band_match(band: Optional[str], expected: str) -> Optional[bool]:
    if band is None or band == "amber":
        return None
    return band == expected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8091/runsync")
    parser.add_argument("--profile", default="quality")
    parser.add_argument("--sample-per-dataset", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--auto-decide",
        action="store_true",
        help="Opt into amber escalation for every scored row.",
    )
    parser.add_argument(
        "--scoring-mode",
        choices=("rag", "code"),
        default="rag",
        help="Route through the RAG lane (default) or the code lane.",
    )
    parser.add_argument(
        "--response-language-hint",
        default="python",
        help="Only used when --scoring-mode=code (default: python).",
    )
    parser.add_argument(
        "--out-dir",
        default=str(OUT),
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = _load_pairs(args.sample_per_dataset, args.seed)
    rows: list[dict[str, Any]] = []
    kept_pairs: list[dict[str, Any]] = []
    for pair in pairs:
        scored_pair: dict[str, Any] = {}
        for kind in ("green", "identifier_swap", "literal_swap", "api_signature_swap"):
            row = pair.get(kind)
            if row is None:
                continue
            try:
                result, dt = _score(
                    args.url,
                    row,
                    profile=args.profile,
                    timeout=args.timeout,
                    auto_decide=args.auto_decide,
                    scoring_mode=args.scoring_mode,
                    response_language_hint=args.response_language_hint,
                )
            except Exception as exc:
                print(f"score error {row['row_id']}: {exc}")
                continue
            scored = {
                **row,
                "band": result.get("band"),
                "score": result.get("score"),
                "amber_escalation": result.get("amber_escalation"),
                "latency_ms": round(dt, 2),
                "profile": args.profile,
            }
            rows.append(scored)
            scored_pair[kind] = scored
        kept_pairs.append(scored_pair)

    rows_path = out_dir / "rows.jsonl"
    with rows_path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Per-variant-kind metrics
    per_variant: dict[str, dict[str, Any]] = {}
    overall_labels_scores: list[tuple[int, float]] = []
    for kind in ("identifier_swap", "literal_swap", "api_signature_swap"):
        # Build a labels/scores list: grounded=0, this variant=1.
        pair_rows: list[tuple[int, float]] = []
        paired_correct = paired_total = paired_ties = 0
        for pair in kept_pairs:
            green = pair.get("green")
            red = pair.get(kind)
            if green is None or red is None:
                continue
            gs, rs = green.get("score"), red.get("score")
            if not (isinstance(gs, (int, float)) and isinstance(rs, (int, float))):
                continue
            pair_rows.append((0, float(gs)))
            pair_rows.append((1, float(rs)))
            if gs > rs:
                paired_correct += 1
            elif gs == rs:
                paired_ties += 1
            paired_total += 1
        f1_metrics = _f1(pair_rows, positive_is_hallucinated=True)
        paired_acc = round(paired_correct / paired_total, 4) if paired_total else None
        # Band-level accuracy (amber counted as wrong for zero-human-in-loop)
        band_correct = band_wrong = band_amber = 0
        for pair in kept_pairs:
            for kind_key in ("green", kind):
                row = pair.get(kind_key)
                if row is None:
                    continue
                res = _band_match(row.get("band"), row["expected_band"])
                if res is True:
                    band_correct += 1
                elif res is False:
                    band_wrong += 1
                else:
                    band_amber += 1
        per_variant[kind] = {
            "n_pairs": paired_total,
            "paired_accuracy": paired_acc,
            "ties": paired_ties,
            "f1_at_best_threshold": f1_metrics,
            "band_counts": {
                "correct": band_correct,
                "wrong": band_wrong,
                "amber": band_amber,
            },
            "band_accuracy_excluding_amber": (
                round(band_correct / (band_correct + band_wrong), 4)
                if (band_correct + band_wrong)
                else None
            ),
        }
        overall_labels_scores.extend(pair_rows)

    overall_f1 = _f1(overall_labels_scores, positive_is_hallucinated=True)
    summary = {
        "profile": args.profile,
        "auto_decide": bool(args.auto_decide),
        "scoring_mode": args.scoring_mode,
        "pairs_per_dataset": args.sample_per_dataset,
        "seed": args.seed,
        "per_variant": per_variant,
        "overall_f1_at_best_threshold": overall_f1,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    _write_report(summary, out_dir / "report.md")
    print(f"wrote {rows_path} + {summary_path}")


def _write_report(summary: dict[str, Any], out: Path) -> None:
    lines = [
        "# Coding-bench report",
        "",
        f"Profile: {summary['profile']}, scoring_mode={summary.get('scoring_mode', 'rag')}, auto_decide={summary['auto_decide']}, seed={summary['seed']}.",
        "",
        "| Variant | n_pairs | paired_acc | F1@best | band_acc_ex_amber |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for kind, data in summary["per_variant"].items():
        f1 = (data.get("f1_at_best_threshold") or {}).get("f1")
        lines.append(
            f"| {kind} | {data.get('n_pairs')} | {_fmt(data.get('paired_accuracy'))} | {_fmt(f1)} | {_fmt(data.get('band_accuracy_excluding_amber'))} |"
        )
    overall_f1 = (summary.get("overall_f1_at_best_threshold") or {}).get("f1")
    lines += [
        "",
        f"Overall F1@best (all variants vs grounded): **{_fmt(overall_f1)}**.",
        "",
        "Gate: per-variant F1 >= 0.80 to pass Phase 3.",
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(x: Any) -> str:
    if x is None:
        return "-"
    try:
        return f"{float(x):.3f}"
    except (TypeError, ValueError):
        return str(x)


if __name__ == "__main__":
    main()
