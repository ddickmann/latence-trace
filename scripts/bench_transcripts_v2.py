"""Agentic-coding bench on the transcripts_v2 corpus.

The transcripts_v2 corpus lives in
``research/triangular_maxsim/coding/cases_transcripts_v2.yaml`` and
contains 180 real assistant turns mined from Cursor agent sessions: 60
correct (grounded), 60 wrong (fabricated-namespace hallucinations), and
60 ambiguous variants per base scenario. Each case ships with the full
context-file bundle from the original agent turn.

This bench pushes each case through a live TRACE endpoint and reports:

* paired-ranking accuracy within each base scenario (correct vs wrong;
  correct vs ambiguous).
* F1@best-threshold across correct-vs-wrong and correct-vs-ambiguous.
* Red / green precision under the hosted bands (amber counted as wrong
  for zero-human-in-the-loop framing).

Usage::

    python scripts/bench_transcripts_v2.py \
        --url http://127.0.0.1:8091/runsync \
        --profile quality --scoring-mode code \
        --out-dir data/veracier-industries/proof_bundle_v1/transcripts_v2

Writes ``rows.jsonl``, ``summary.json``, and ``report.md``.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple  # noqa: UP035

REPO_ROOT = Path(__file__).resolve().parents[1]

# Ensure research package is importable.
import sys
sys.path.insert(0, str(REPO_ROOT))

from research.triangular_maxsim.coding.transcript_cases_v2 import (  # noqa: E402
    load_transcript_cases_v2,
)


def _pack_context(case: Any, *, max_files: int, max_chars: int) -> str:
    """Pack the case's context files into a single ``raw_context`` blob.

    Uses a simple delimiter format that tree-sitter / the RAG lane can
    both parse: each file is prefixed with ``=== <path> ===`` so the
    AST and literal-novelty detectors can anchor spans to paths.
    """
    chunks: list[str] = []
    used = 0
    for f in (case.context_files or [])[:max_files]:
        header = f"=== {f.path} ===\n"
        body = (f.content or "")
        blob = header + body + "\n"
        if used + len(blob) > max_chars:
            remaining = max_chars - used - len(header) - 8
            if remaining > 256:
                blob = header + body[:remaining] + "\n...[truncated]...\n"
                chunks.append(blob)
            break
        chunks.append(blob)
        used += len(blob)
    return "".join(chunks)


def _score(
    url: str,
    *,
    query_text: str,
    response_text: str,
    raw_context: str,
    profile: str,
    scoring_mode: str,
    response_language_hint: Optional[str],
    timeout: float,
) -> Tuple[dict[str, Any], float]:
    payload_input: dict[str, Any] = {
        "action": "score",
        "query_text": query_text,
        "response_text": response_text,
        "raw_context": raw_context,
        "profile": profile,
        "scoring_mode": scoring_mode,
    }
    if scoring_mode == "code" and response_language_hint:
        payload_input["response_language_hint"] = response_language_hint
    body = {"input": payload_input}
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "content-type": "application/json",
            "accept": "application/json",
            "x-latence-tenant-id": "bench-transcripts-v2",
        },
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        out = json.loads(resp.read().decode("utf-8"))
    dt_ms = (time.perf_counter() - t0) * 1000.0
    return out.get("output") or out, dt_ms


def _f1_sweep(
    labels_scores: List[Tuple[int, float]],
    *,
    positive_is_hallucinated: bool = True,
) -> dict[str, Any]:
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8091/runsync")
    parser.add_argument("--profile", default="quality")
    parser.add_argument(
        "--scoring-mode",
        choices=("rag", "code"),
        default="code",
        help="Code lane by default — this corpus is code-focused.",
    )
    parser.add_argument("--response-language-hint", default="python")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument(
        "--max-files",
        type=int,
        default=20,
        help="Cap the number of context files we pack per case.",
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=60000,
        help="Hard cap on raw_context size per request.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "data/veracier-industries/proof_bundle_v1/transcripts_v2"),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Stop after N base scenarios (0 = all).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = load_transcript_cases_v2()
    scenarios: dict[str, dict[str, Any]] = defaultdict(dict)
    for case in cases:
        base_id = str(case.metadata.get("base_scenario_id") or case.id)
        scenarios[base_id][case.subcategory] = case

    base_ids = sorted(scenarios.keys())
    if args.limit:
        base_ids = base_ids[: args.limit]

    rows: list[dict[str, Any]] = []
    scored_pairs: list[dict[str, Any]] = []
    n_total = 0
    for base_id in base_ids:
        bundle = scenarios[base_id]
        case_any = next(iter(bundle.values()))
        raw_context = _pack_context(
            case_any,
            max_files=args.max_files,
            max_chars=args.max_context_chars,
        )
        pair: dict[str, Any] = {"base_id": base_id}
        for kind in ("correct", "wrong", "ambiguous"):
            case = bundle.get(kind)
            if case is None:
                continue
            try:
                result, dt = _score(
                    args.url,
                    query_text=case.query or "",
                    response_text=case.response or "",
                    raw_context=raw_context,
                    profile=args.profile,
                    scoring_mode=args.scoring_mode,
                    response_language_hint=args.response_language_hint,
                    timeout=args.timeout,
                )
            except Exception as exc:
                logging.error("score error %s/%s: %s", base_id, kind, exc)
                continue
            row = {
                "base_id": base_id,
                "case_id": case.id,
                "subcategory": kind,
                "label": case.label,
                "band": result.get("band"),
                "score": result.get("score"),
                "composite_score": (result.get("code_lane") or {}).get("composite_score"),
                "phantom_verdict": (result.get("code_lane") or {}).get("phantom_verdict"),
                "latency_ms": round(dt, 2),
                "scoring_mode": args.scoring_mode,
                "profile": args.profile,
            }
            rows.append(row)
            pair[kind] = row
            n_total += 1
        scored_pairs.append(pair)
        print(f"  scored {base_id}: {sorted(set(bundle.keys()))}")

    rows_path = out_dir / "rows.jsonl"
    with rows_path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Pairwise metrics
    def _paired(bundle_key: str) -> dict[str, Any]:
        correct_wins = wrong_wins = ties = total = 0
        pair_rows: list[tuple[int, float]] = []
        for p in scored_pairs:
            c = p.get("correct")
            w = p.get(bundle_key)
            if c is None or w is None:
                continue
            cs = c.get("score")
            ws = w.get("score")
            if cs is None or ws is None:
                continue
            cs, ws = float(cs), float(ws)
            pair_rows.append((0, cs))
            pair_rows.append((1, ws))
            if cs > ws:
                correct_wins += 1
            elif cs < ws:
                wrong_wins += 1
            else:
                ties += 1
            total += 1
        f1 = _f1_sweep(pair_rows, positive_is_hallucinated=True)
        return {
            "n_pairs": total,
            "correct_wins": correct_wins,
            "wrong_wins": wrong_wins,
            "ties": ties,
            "paired_accuracy": round(correct_wins / total, 4) if total else None,
            "f1_at_best_threshold": f1,
        }

    paired_correct_vs_wrong = _paired("wrong")
    paired_correct_vs_ambiguous = _paired("ambiguous")

    # Band-level summary
    band_counts = {"correct": {"green": 0, "amber": 0, "red": 0, "other": 0},
                   "wrong":   {"green": 0, "amber": 0, "red": 0, "other": 0},
                   "ambiguous": {"green": 0, "amber": 0, "red": 0, "other": 0}}
    for r in rows:
        sub = r["subcategory"]
        b = r.get("band") or "other"
        if b not in band_counts[sub]:
            b = "other"
        band_counts[sub][b] += 1

    # Red/green precision in zero-human framing (amber counts as wrong)
    def _precision_recall(sub: str, expected_band: str) -> dict[str, Any]:
        expected = expected_band
        correct = wrong = 0
        for r in rows:
            if r["subcategory"] != sub:
                continue
            b = r.get("band")
            if b == expected:
                correct += 1
            elif b is not None and b != "amber":
                wrong += 1
        tot = correct + wrong
        return {
            "correct": correct,
            "wrong_or_amber": wrong + band_counts[sub].get("amber", 0),
            "precision_excluding_amber": round(correct / tot, 4) if tot else None,
        }

    summary = {
        "n_scenarios": len(base_ids),
        "n_rows": n_total,
        "profile": args.profile,
        "scoring_mode": args.scoring_mode,
        "paired": {
            "correct_vs_wrong": paired_correct_vs_wrong,
            "correct_vs_ambiguous": paired_correct_vs_ambiguous,
        },
        "band_counts": band_counts,
        "green_precision_on_correct": _precision_recall("correct", "green"),
        "red_precision_on_wrong": _precision_recall("wrong", "red"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_report(summary, out_dir / "report.md")
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {rows_path} + {out_dir / 'summary.json'} + {out_dir / 'report.md'}")


def _write_report(summary: dict[str, Any], out: Path) -> None:
    cw = summary["paired"]["correct_vs_wrong"]
    ca = summary["paired"]["correct_vs_ambiguous"]
    lines = [
        "# Transcripts-v2 agentic-coding bench",
        "",
        f"n_scenarios: {summary['n_scenarios']}  |  scoring_mode: {summary['scoring_mode']}  |  profile: {summary['profile']}",
        "",
        "## Paired ranking",
        "",
        "| Matchup | n_pairs | paired_acc | ties | F1@best |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| correct vs wrong      | {cw['n_pairs']} | {_fmt(cw['paired_accuracy'])} | {cw['ties']} | {_fmt((cw['f1_at_best_threshold'] or {}).get('f1'))} |",
        f"| correct vs ambiguous  | {ca['n_pairs']} | {_fmt(ca['paired_accuracy'])} | {ca['ties']} | {_fmt((ca['f1_at_best_threshold'] or {}).get('f1'))} |",
        "",
        "## Band distribution",
        "",
        "| Subcategory | green | amber | red | other |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for sub in ("correct", "wrong", "ambiguous"):
        bc = summary["band_counts"][sub]
        lines.append(f"| {sub} | {bc['green']} | {bc['amber']} | {bc['red']} | {bc['other']} |")
    lines += [
        "",
        "## Zero-human-in-the-loop precision",
        "",
        "Amber counted as incorrect for the zero-human framing (auto-decide would need to collapse it).",
        "",
        f"* green_precision on correct cases (amber excluded): **{_fmt(summary['green_precision_on_correct']['precision_excluding_amber'])}**",
        f"* red_precision on wrong cases (amber excluded):     **{_fmt(summary['red_precision_on_wrong']['precision_excluding_amber'])}**",
        "",
        "Gate target: paired_acc (correct vs wrong) >= 0.80 and red_precision (excluding amber) >= 0.90.",
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
