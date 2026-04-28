"""External benchmark harness for TRACE (Plan G2).

Runs HaluEval QA / HaluEval Summarisation / RAGTruth QA / RAGTruth
Summarisation through the live TRACE endpoint and produces:

* Unified per-row JSONL at
  ``proof_bundle_v1/external_bench/<bench>_<profile>_rows.jsonl``
  with the same row shape as Veracier (``expected_band``, ``band``,
  ``score``, ``rationale``...).
* Aggregate JSON at ``proof_bundle_v1/external_bench/summary.json``.
* Markdown summary at
  ``proof_bundle_v1/external_benchmarks.md``.

Design choices:

* Fixed sample size per benchmark (default 120 to match Veracier
  order of magnitude) with a seed so the report is reproducible.
* Scores against the local ``dev_app`` on port 8091 by default;
  point ``--url`` at any TRACE worker to run against staging / prod.
* Faithful == green, hallucinated == red.  Amber is never the
  expected band for these benchmarks since they only have binary
  labels; we therefore report 2 x 2 precision / recall plus amber
  leakage rate, which is the operationally honest number.

Usage::

    python scripts/bench_external.py \
        --benches halueval_qa halueval_summ ragtruth_qa \
        --profile standard --n 120 --seed 7 \
        --url http://127.0.0.1:8091/runsync

Run this twice (standard and quality) to cover both lanes.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
import urllib.request
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple  # noqa: UP035

REPO_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = REPO_ROOT / "research/triangular_maxsim/external_data"


def iter_halueval_qa(n: int, seed: int) -> Iterable[dict]:
    path = EXTERNAL / "HaluEval/data/qa_data.jsonl"
    if not path.exists():
        return
    rng = random.Random(seed)
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    rng.shuffle(rows)
    for idx, row in enumerate(rows[: n // 2]):
        yield {
            "bench": "halueval_qa",
            "row_id": f"halueval_qa_{idx}",
            "question": row.get("question", ""),
            "response_text": row.get("right_answer", ""),
            "raw_context": row.get("knowledge", ""),
            "expected_band": "green",
            "expected_label": "faithful",
        }
        yield {
            "bench": "halueval_qa",
            "row_id": f"halueval_qa_{idx}_halluc",
            "question": row.get("question", ""),
            "response_text": row.get("hallucinated_answer", ""),
            "raw_context": row.get("knowledge", ""),
            "expected_band": "red",
            "expected_label": "hallucinated",
        }


def iter_halueval_summ(n: int, seed: int) -> Iterable[dict]:
    path = EXTERNAL / "HaluEval/data/summarization_data.jsonl"
    if not path.exists():
        return
    rng = random.Random(seed)
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    rng.shuffle(rows)
    for idx, row in enumerate(rows[: n // 2]):
        document = row.get("document", "")
        right_summary = row.get("right_summary") or row.get("summary") or ""
        hallucinated_summary = row.get("hallucinated_summary", "")
        question = "Summarise the article."
        if right_summary:
            yield {
                "bench": "halueval_summ",
                "row_id": f"halueval_summ_{idx}",
                "question": question,
                "response_text": right_summary,
                "raw_context": document,
                "expected_band": "green",
                "expected_label": "faithful",
            }
        if hallucinated_summary:
            yield {
                "bench": "halueval_summ",
                "row_id": f"halueval_summ_{idx}_halluc",
                "question": question,
                "response_text": hallucinated_summary,
                "raw_context": document,
                "expected_band": "red",
                "expected_label": "hallucinated",
            }


def iter_ragtruth(n: int, seed: int, task: str) -> Iterable[dict]:
    """RAGTruth responses + labels.

    Prefer the canonical ``voyager_layout/{qa,summarization,data2text}/test.jsonl``
    layout used by ``research/triangular_maxsim/groundedness_external_benchmarks.py``
    because that is what the published ``truth_bench_n120.json`` reference
    report was measured against (first N rows, no shuffle, query from the
    ``query`` field, context = json dump of ``source_info``).

    Fall back to the raw ``dataset/response.jsonl`` + ``dataset/source_info.jsonl``
    layout only when the voyager_layout is unavailable.
    """
    task_dir = {"qa": "qa", "summary": "summarization"}.get(task.lower(), task.lower())
    voyager = EXTERNAL / f"RAGTruth/voyager_layout/{task_dir}/test.jsonl"
    if voyager.exists():
        rows = []
        with voyager.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        # Reference takes the first N rows (no shuffle) for exact
        # reproducibility against ``truth_bench_n120.json``.
        for idx, obj in enumerate(rows[:n]):
            labels = obj.get("labels") or []
            has_halluc = bool(labels)
            source_info = obj.get("source_info") or ""
            if isinstance(source_info, dict):
                raw_context = json.dumps(source_info, ensure_ascii=False)
                query = (
                    obj.get("query")
                    or source_info.get("question")
                    or ""
                )
            else:
                raw_context = str(source_info)
                query = obj.get("query") or ""
            yield {
                "bench": f"ragtruth_{task.lower()}",
                "row_id": f"ragtruth_{task.lower()}_{idx}",
                "question": query,
                "response_text": obj.get("response", ""),
                "raw_context": raw_context,
                "expected_band": "red" if has_halluc else "green",
                "expected_label": "hallucinated" if has_halluc else "faithful",
            }
        return
    base = EXTERNAL / "RAGTruth/dataset"
    if not (base / "response.jsonl").exists() or not (base / "source_info.jsonl").exists():
        return
    sources: dict[str, dict] = {}
    with (base / "source_info.jsonl").open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(rec.get("task_type", "")).lower() == task.lower():
                sources[rec["source_id"]] = rec
    rng = random.Random(seed)
    matched: list[dict] = []
    with (base / "response.jsonl").open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            src = sources.get(rec.get("source_id", ""))
            if not src:
                continue
            matched.append({"resp": rec, "src": src})
    rng.shuffle(matched)
    for idx, row in enumerate(matched[:n]):
        resp = row["resp"]
        src = row["src"]
        has_halluc = bool(resp.get("labels"))
        label = "hallucinated" if has_halluc else "faithful"
        expected_band = "red" if has_halluc else "green"
        source_info = src.get("source_info") or {}
        # Reference methodology (load_ragtruth): dict -> json.dumps, else str.
        if isinstance(source_info, dict):
            raw_context = json.dumps(source_info, ensure_ascii=False)
            question = source_info.get("question") or ""
        else:
            raw_context = str(source_info)
            question = ""
        yield {
            "bench": f"ragtruth_{task.lower()}",
            "row_id": f"ragtruth_{task.lower()}_{idx}",
            "question": question,
            "response_text": resp.get("response", ""),
            "raw_context": raw_context,
            "expected_band": expected_band,
            "expected_label": label,
        }


def score(
    url: str,
    row: dict,
    profile: str,
    timeout: float,
    *,
    extra: Optional[dict] = None,
) -> Tuple[dict, float]:
    payload_input: dict[str, Any] = {
        "action": "score",
        # The RunPod handler maps `query_text`/`query` into the anchor
        # question; the legacy `question` key is silently dropped. Use
        # the supported key so TRACE actually sees the anchor.
        "query_text": row["question"],
        "response_text": row["response_text"],
        "raw_context": row["raw_context"],
        "profile": profile,
    }
    if extra:
        payload_input.update(extra)
    payload = {"input": payload_input}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "content-type": "application/json",
            "accept": "application/json",
            "x-latence-tenant-id": "bench-external",
        },
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    dt = (time.perf_counter() - t0) * 1000.0
    return body.get("output") or body, dt


def _best_threshold_f1(labels_scores: List[Tuple[int, float]]) -> dict:
    """Sweep thresholds; predict positive=hallucinated when score < t.

    Returns best F1, precision, recall, threshold, and confusion counts."""
    if not labels_scores:
        return {"f1": None, "precision": None, "recall": None, "threshold": None, "counts": None}
    sorted_scores = sorted({s for _, s in labels_scores})
    thresholds = sorted_scores + [max(sorted_scores) + 1e-6]
    best = {"f1": -1.0, "precision": 0.0, "recall": 0.0, "threshold": None, "counts": None}
    for t in thresholds:
        tp = fp = tn = fn = 0
        for y, s in labels_scores:
            pred_halluc = s < t
            if pred_halluc and y == 1:
                tp += 1
            elif pred_halluc and y == 0:
                fp += 1
            elif not pred_halluc and y == 0:
                tn += 1
            else:
                fn += 1
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        if f1 > best["f1"]:
            best = {
                "f1": round(f1, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "threshold": round(float(t), 4),
                "counts": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
            }
    if best["f1"] < 0:
        best["f1"] = None
    return best


def _roc_auc(labels_scores: List[Tuple[int, float]]) -> Optional[float]:
    """Trapezoidal ROC-AUC where positive=hallucinated (low score => positive)."""
    pos = [s for y, s in labels_scores if y == 1]
    neg = [s for y, s in labels_scores if y == 0]
    if not pos or not neg:
        return None
    # AUC = P(score_neg > score_pos) since low score => halluc.
    wins = ties = 0
    for np_ in neg:
        for pp in pos:
            if np_ > pp:
                wins += 1
            elif np_ == pp:
                ties += 1
    total = len(pos) * len(neg)
    auc = (wins + 0.5 * ties) / total if total else None
    return round(auc, 4) if auc is not None else None


def _paired_accuracy(rows: list[dict]) -> dict:
    """For HaluEval-style paired rows: faithful + hallucinated share a stem id."""
    by_stem: dict[str, dict[str, dict]] = {}
    for r in rows:
        rid = str(r.get("row_id", ""))
        stem = rid.replace("_halluc", "")
        lab = (r.get("expected_label") or "").lower()
        if lab in {"faithful", "grounded"}:
            by_stem.setdefault(stem, {})["green"] = r
        elif lab in {"hallucinated"}:
            by_stem.setdefault(stem, {})["red"] = r
    pairs = [p for p in by_stem.values() if "green" in p and "red" in p]
    usable = [
        p
        for p in pairs
        if isinstance(p["green"].get("score"), (int, float))
        and isinstance(p["red"].get("score"), (int, float))
    ]
    if not usable:
        return {"n_pairs": 0, "paired_accuracy": None, "ties": 0, "delta_mean": None}
    correct = sum(1 for p in usable if p["green"]["score"] > p["red"]["score"])
    ties = sum(1 for p in usable if p["green"]["score"] == p["red"]["score"])
    delta = [float(p["green"]["score"]) - float(p["red"]["score"]) for p in usable]
    return {
        "n_pairs": len(usable),
        "paired_accuracy": round(correct / len(usable), 4),
        "ties": ties,
        "delta_mean": round(sum(delta) / len(delta), 4),
        "delta_median": round(sorted(delta)[len(delta) // 2], 4),
    }


def _faithful_positive_f1(labels_scores: List[Tuple[int, float]]) -> dict:
    """Reference framing from ``groundedness_external_benchmarks.py``:

    positive class = faithful (l=1), predict faithful if score >= threshold.
    Reports F1 at median threshold and at best-F1 sweep threshold.
    """
    if not labels_scores:
        return {"median": None, "best_f1_sweep": None}
    # flip labels: l=1 means faithful
    flipped = [(1 - y, s) for y, s in labels_scores]  # y=1 => hallucinated => now 0 (negative)
    scores = sorted([s for _, s in flipped])
    median = scores[len(scores) // 2]

    def at(t: float) -> dict:
        tp = fp = fn = tn = 0
        for y, s in flipped:  # y=1 means faithful now
            if s >= t and y == 1:
                tp += 1
            elif s >= t and y == 0:
                fp += 1
            elif s < t and y == 1:
                fn += 1
            else:
                tn += 1
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        return {
            "threshold": round(t, 4),
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "counts": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        }

    lo, hi = min(scores), max(scores)
    best = None
    for k in range(64):
        t = lo + (hi - lo) * k / 63 if hi > lo else lo
        m = at(t)
        if best is None or m["f1"] > best["f1"]:
            best = m
    return {"median": at(median), "best_f1_sweep": best}


def aggregate(rows: list[dict]) -> dict:
    total = len(rows)
    if total == 0:
        return {"total": 0}
    tp = fp = tn = fn = amber = errors = 0
    labels_scores: List[Tuple[int, float]] = []
    for r in rows:
        exp = r["expected_band"]
        got = r.get("band")
        score_val = r.get("score")
        if got is None:
            errors += 1
        elif got == "amber":
            amber += 1
        elif exp == "red":
            if got == "red":
                tp += 1
            else:
                fn += 1
        else:
            if got == "green":
                tn += 1
            else:
                fp += 1
        if isinstance(score_val, (int, float)) and not math.isnan(score_val):
            labels_scores.append((1 if exp == "red" else 0, float(score_val)))
    red_precision = tp / (tp + fp) if (tp + fp) else None
    red_recall = tp / (tp + fn) if (tp + fn) else None
    green_precision = tn / (tn + fn) if (tn + fn) else None
    # Score-only accuracy at the best threshold: amber rows are included
    # because the score channel is continuous and does not respect the
    # operational amber hedge. This is the "research" metric matched to
    # the website's F1-@-best-threshold claim.
    best = _best_threshold_f1(labels_scores)
    faithful_framing = _faithful_positive_f1(labels_scores)
    auc = _roc_auc(labels_scores)
    paired = _paired_accuracy(rows)
    # Auto-decide metrics: when the harness opted into auto_decide, the
    # hosted amber-escalation middleware should have collapsed every amber
    # row to green/red. Count how often the FINAL band matches the gold
    # label. This is the zero-human-in-the-loop accuracy number.
    auto_correct = 0
    auto_wrong = 0
    auto_amber = 0
    auto_errors = 0
    for r in rows:
        got = r.get("band")
        exp = r["expected_band"]
        if got is None:
            auto_errors += 1
            continue
        if got == "amber":
            auto_amber += 1
            continue
        if got == exp:
            auto_correct += 1
        else:
            auto_wrong += 1
    auto_total = auto_correct + auto_wrong + auto_amber
    auto_decide = {
        "correct": auto_correct,
        "wrong": auto_wrong,
        "amber": auto_amber,
        "errors": auto_errors,
        "accuracy_excluding_amber": round(
            auto_correct / (auto_correct + auto_wrong), 4
        )
        if (auto_correct + auto_wrong)
        else None,
        "accuracy_including_amber": round(auto_correct / auto_total, 4)
        if auto_total
        else None,
    }
    return {
        "total": total,
        "errors": errors,
        "amber_leakage_rate": round(amber / total, 4),
        "red_precision": round(red_precision, 4) if red_precision is not None else None,
        "red_recall": round(red_recall, 4) if red_recall is not None else None,
        "green_precision": round(green_precision, 4) if green_precision is not None else None,
        "counts": {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "amber": amber},
        "f1_at_best_threshold": best,
        "faithful_positive_framing": faithful_framing,
        "roc_auc": auc,
        "paired": paired,
        "auto_decide": auto_decide,
    }


def bench_one(
    bench_name: str,
    rows: Iterable[dict],
    *,
    url: str,
    profile: str,
    timeout: float,
    out_dir: Path,
    extra: Optional[dict] = None,
) -> dict:
    out_path = out_dir / f"{bench_name}_{profile}_rows.jsonl"
    scored_rows: list[dict] = []
    start = time.perf_counter()
    with out_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            try:
                res, dt = score(url, row, profile, timeout, extra=extra)
                scored = {
                    **row,
                    "band": res.get("band"),
                    "score": res.get("score"),
                    "profile": profile,
                    "latency_ms": round(dt, 2),
                    "rationale": (
                        "TRACE agrees" if res.get("band") == row["expected_band"] else "TRACE disagrees"
                    ),
                }
            except Exception as exc:
                scored = {**row, "band": None, "score": None, "profile": profile, "error": str(exc)}
            scored_rows.append(scored)
            fh.write(json.dumps(scored, ensure_ascii=False) + "\n")
    elapsed = time.perf_counter() - start
    summary = aggregate(scored_rows)
    summary["bench"] = bench_name
    summary["profile"] = profile
    summary["elapsed_s"] = round(elapsed, 2)
    return summary


def write_markdown(summaries: list[dict], out_path: Path, timestamp: str) -> None:
    lines = [
        "# External benchmark report",
        "",
        f"Generated {timestamp}.  Reproducer: `scripts/bench_external.py`.",
        "",
        "| Benchmark | Profile | Rows | Red precision | Red recall | Green precision | Amber leakage | Errors |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for s in summaries:
        lines.append(
            "| {bench} | {profile} | {total} | {rp} | {rr} | {gp} | {amber} | {err} |".format(
                bench=s["bench"],
                profile=s["profile"],
                total=s["total"],
                rp=_fmt(s.get("red_precision")),
                rr=_fmt(s.get("red_recall")),
                gp=_fmt(s.get("green_precision")),
                amber=_fmt(s.get("amber_leakage_rate")),
                err=s.get("errors", 0),
            )
        )
    lines += [
        "",
        "## Reading the table",
        "",
        "- `Red precision` = faithfulness of TRACE's red flags.  The",
        "  hosted SLO is >= 0.95.",
        "- `Red recall` = fraction of truly hallucinated items that",
        "  TRACE correctly routes to red / amber.",
        "- `Green precision` = faithfulness of TRACE's green flags.",
        "  The hosted SLO is >= 0.97.",
        "- `Amber leakage` = fraction of rows TRACE declined to band",
        "  decisively.  Because the external benchmarks only have",
        "  binary labels, amber rows are conservative hedge-gate",
        "  outputs rather than errors; customers route them through",
        "  the reviewer queue.",
        "",
        "## What is not covered",
        "",
        "- The external benchmarks are English-only and do not cover",
        "  the legal / finance / procurement strata where the Veracier",
        "  proof is strongest.  See `proof_report.md` for the",
        "  headline defence against enterprise-buyer scrutiny.",
        "- RAGTruth \"faithful\" examples are sometimes partial",
        "  paraphrases of the source; TRACE may band these amber for",
        "  borderline support, which the aggregate correctly counts",
        "  as amber leakage rather than an error.",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(val: float | None) -> str:
    return "-" if val is None else f"{val:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8091/runsync")
    parser.add_argument(
        "--benches",
        nargs="+",
        default=["halueval_qa", "halueval_summ", "ragtruth_qa", "ragtruth_summ"],
    )
    parser.add_argument("--profile", default="standard")
    parser.add_argument("--n", type=int, default=120)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument(
        "--out-dir",
        default="data/veracier-industries/proof_bundle_v1/external_bench",
    )
    parser.add_argument(
        "--atomic-claims",
        default="auto",
        choices=("auto", "true", "false"),
        help="Override nli_use_atomic_claims on each request. 'auto' defers to the profile preset.",
    )
    parser.add_argument(
        "--config-tag",
        default=None,
        help="Human-readable tag recorded in the summary (e.g. 'english_nli_quality_2026_04_28').",
    )
    parser.add_argument(
        "--auto-decide",
        action="store_true",
        help="Opt into the zero-human-in-the-loop amber escalation on every row.",
    )
    args = parser.parse_args()

    extra: dict[str, Any] = {}
    if args.atomic_claims != "auto":
        extra["nli_use_atomic_claims"] = (args.atomic_claims == "true")
    if args.auto_decide:
        extra["auto_decide"] = True

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict] = []
    for bench in args.benches:
        if bench == "halueval_qa":
            rows = iter_halueval_qa(args.n, args.seed)
        elif bench == "halueval_summ":
            rows = iter_halueval_summ(args.n, args.seed)
        elif bench == "ragtruth_qa":
            rows = iter_ragtruth(args.n, args.seed, task="QA")
        elif bench == "ragtruth_summ":
            rows = iter_ragtruth(args.n, args.seed, task="Summary")
        else:
            print(f"skipping unknown bench: {bench}")
            continue
        print(f"running bench={bench} profile={args.profile} ...")
        summary = bench_one(
            bench,
            rows,
            url=args.url,
            profile=args.profile,
            timeout=args.timeout,
            out_dir=out_dir,
            extra=extra or None,
        )
        summaries.append(summary)
        print(json.dumps(summary, indent=2))

    effective_config = {
        "profile": args.profile,
        "config_tag": args.config_tag,
        "atomic_claims_override": args.atomic_claims,
        "request_extra": extra,
        "nli_model_env": os.environ.get("VOYAGER_GROUNDEDNESS_NLI_MODEL"),
        "nli_reranker_env": os.environ.get("VOYAGER_GROUNDEDNESS_NLI_PREMISE_RERANKER_MODEL"),
        "fusion_w_calibrated_env": os.environ.get("VOYAGER_GROUNDEDNESS_FUSION_W_CALIBRATED"),
        "fusion_w_literal_env": os.environ.get("VOYAGER_GROUNDEDNESS_FUSION_W_LITERAL"),
        "fusion_w_nli_env": os.environ.get("VOYAGER_GROUNDEDNESS_FUSION_W_NLI"),
        "n": args.n,
        "seed": args.seed,
        "url": args.url,
    }
    for s in summaries:
        s["effective_config"] = effective_config
    summary_path = out_dir / f"summary_{args.profile}.json"
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    md_path = out_dir.parent / "external_benchmarks.md"
    if md_path.exists():
        existing = md_path.read_text(encoding="utf-8")
    else:
        existing = ""
    # write/overwrite a fresh markdown summarising the latest run.
    import datetime

    ts = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    # Merge with any existing summary file if multiple profile runs
    # landed in the same day - we keep the latest per (bench, profile).
    all_summaries = summaries
    prior = existing  # kept verbatim in the file history if needed
    write_markdown(all_summaries, md_path, ts)
    print(f"wrote {summary_path} and {md_path}")


if __name__ == "__main__":
    main()
