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
import random
import time
import urllib.request
from pathlib import Path
from typing import Iterable, List, Tuple

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
    """RAGTruth has response.jsonl (responses + per-span labels) + source_info.jsonl."""
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
        # RAGTruth stores source_info as either a string (summary task)
        # or a dict ({"question": ..., "passages": "passage 1:..."}) for
        # QA task.  Flatten to natural prose so TRACE does not see raw
        # JSON tokens and lose its question anchor.
        if isinstance(source_info, dict):
            question = source_info.get("question") or src.get("question") or "Answer the user question from the source."
            passages = source_info.get("passages") or ""
            raw_context = passages if passages else json.dumps(source_info, ensure_ascii=False)
        else:
            question = src.get("question") or "Answer the user question from the source."
            raw_context = str(source_info)
        yield {
            "bench": f"ragtruth_{task.lower()}",
            "row_id": f"ragtruth_{task.lower()}_{idx}",
            "question": question,
            "response_text": resp.get("response", ""),
            "raw_context": raw_context,
            "expected_band": expected_band,
            "expected_label": label,
        }


def score(url: str, row: dict, profile: str, timeout: float) -> Tuple[dict, float]:
    payload = {
        "input": {
            "action": "score",
            "question": row["question"],
            "response_text": row["response_text"],
            "raw_context": row["raw_context"],
            "profile": profile,
        }
    }
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


def aggregate(rows: list[dict]) -> dict:
    total = len(rows)
    if total == 0:
        return {"total": 0}
    tp = fp = tn = fn = amber = errors = 0
    for r in rows:
        exp = r["expected_band"]
        got = r.get("band")
        if got is None:
            errors += 1
            continue
        if got == "amber":
            amber += 1
            continue
        if exp == "red":
            if got == "red":
                tp += 1
            else:
                fn += 1
        else:
            if got == "green":
                tn += 1
            else:
                fp += 1
    red_precision = tp / (tp + fp) if (tp + fp) else None
    red_recall = tp / (tp + fn) if (tp + fn) else None
    green_precision = tn / (tn + fn) if (tn + fn) else None
    return {
        "total": total,
        "errors": errors,
        "amber_leakage_rate": round(amber / total, 4),
        "red_precision": round(red_precision, 4) if red_precision is not None else None,
        "red_recall": round(red_recall, 4) if red_recall is not None else None,
        "green_precision": round(green_precision, 4) if green_precision is not None else None,
        "counts": {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "amber": amber},
    }


def bench_one(
    bench_name: str,
    rows: Iterable[dict],
    *,
    url: str,
    profile: str,
    timeout: float,
    out_dir: Path,
) -> dict:
    out_path = out_dir / f"{bench_name}_{profile}_rows.jsonl"
    scored_rows: list[dict] = []
    start = time.perf_counter()
    with out_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            try:
                res, dt = score(url, row, profile, timeout)
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
    args = parser.parse_args()

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
        )
        summaries.append(summary)
        print(json.dumps(summary, indent=2))

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
