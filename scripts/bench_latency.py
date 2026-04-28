"""Isolated + concurrent latency benchmark (Plan F1 + F2).

Runs a small, reproducible fixture set through the local ``dev_app``
(or any TRACE endpoint) at configurable concurrency and reports
per-profile p50/p95/p99 with queue-time vs score-time split.

Two lanes:

* ``--lane isolated`` — concurrency=1, every request waits for the
  previous one to finish.  This is the number customers see when
  only one call is in flight.  Use it in sales conversations.

* ``--lane sustained --concurrency N`` — N workers hammering the
  worker with a shared request queue.  Queue-time is measured as
  ``wall_time - score_time`` per row where ``score_time`` comes
  from the ``took_ms`` the worker reports.  Use it for capacity
  planning.

Example::

    python scripts/bench_latency.py --lane isolated --profile standard
    python scripts/bench_latency.py --lane sustained --concurrency 8 \
        --profile quality

Output is a JSON manifest at ``latency_bench_<lane>_<profile>.json``
and a one-line summary to stdout.

Intentionally small (15 rows) so the whole thing runs in < 2 min at
isolated concurrency; the full Veracier-scale dual-lane report lives
in ``data/veracier-industries/proof_bundle_v1/``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

FIXTURES = [
    {
        "question": "What was the 2023 ARR?",
        "response_text": "ARR reached 12.4M USD in 2023.",
        "raw_context": "FY23 shareholder letter: ARR ended 2023 at 12.4M USD.",
    },
    {
        "question": "Which NIST control framework is referenced?",
        "response_text": "The policy cites NIST SP 800-53 rev. 5.",
        "raw_context": (
            "Security policy \u00a73.2: 'Controls map to NIST SP 800-53 rev. 5 families "
            "AC, AU, and IA.'"
        ),
    },
    {
        "question": "When was the Berlin Wall torn down?",
        "response_text": "The Berlin Wall was torn down in 1989.",
        "raw_context": "History note: the Berlin Wall fell on the 9th of November 1989.",
    },
    {
        "question": "What is the hedge fund's target allocation for 2024?",
        "response_text": "Target allocation for 2024 is 60% equities, 30% bonds, 10% cash.",
        "raw_context": (
            "Q1 2024 investment committee memo: 'Target allocation: 60% equities, "
            "30% fixed income, 10% cash reserves.'"
        ),
    },
    {
        "question": "What are the return policy terms?",
        "response_text": "Returns accepted within 30 days of purchase with a receipt.",
        "raw_context": (
            "Returns policy v2.1: 'Customers may return items within 30 calendar "
            "days of purchase, provided an original receipt is presented.'"
        ),
    },
    {
        "question": "Quel est le chiffre d'affaires 2023?",
        "response_text": "Le chiffre d'affaires 2023 est de 45.2M EUR.",
        "raw_context": (
            "Rapport annuel 2023 : 'Le chiffre d\u2019affaires consolid\u00e9 s\u2019\u00e9l\u00e8ve "
            "\u00e0 45,2 M\u20ac, en hausse de 8% par rapport \u00e0 2022.'"
        ),
    },
    {
        "question": "What is the HIPAA audit controls standard?",
        "response_text": "HIPAA requires audit controls under 45 CFR \u00a7 164.312(b).",
        "raw_context": (
            "HIPAA Security Rule, 45 CFR \u00a7 164.312(b) Audit controls: 'Implement "
            "hardware, software, and/or procedural mechanisms that record and examine activity.'"
        ),
    },
    {
        "question": "What is the default Kubernetes resource quota?",
        "response_text": "Default quota is 4 CPU, 8Gi memory per namespace.",
        "raw_context": (
            "Cluster config: 'Per-namespace default ResourceQuota: cpu=4, memory=8Gi, "
            "pods=50.'"
        ),
    },
    {
        "question": "Wie hoch war der Jahresumsatz 2023?",
        "response_text": "Der Jahresumsatz betrug 45,2 Mio. EUR.",
        "raw_context": (
            "Gesch\u00e4ftsbericht 2023: 'Der Konzernumsatz stieg um 8% auf 45,2 Millionen Euro.'"
        ),
    },
    {
        "question": "What does the SLA guarantee for uptime?",
        "response_text": "The SLA guarantees 99.5% monthly uptime.",
        "raw_context": (
            "Hosted SLA v1: 'Monthly uptime target of 99.5% measured as successful "
            "score responses divided by all score attempts.'"
        ),
    },
    {
        "question": "What is the maximum context window?",
        "response_text": "The maximum context is 4096 tokens.",
        "raw_context": (
            "Model card: 'Context window: 4096 tokens including system and user messages.'"
        ),
    },
    {
        "question": "What languages does TRACE support?",
        "response_text": "TRACE supports English, French, German, Spanish, and Italian.",
        "raw_context": (
            "docs/guides/multilingual.md: 'First-class support for English, French, "
            "German, Spanish, and Italian as of v1.'"
        ),
    },
    {
        "question": "When does the free tier reset?",
        "response_text": "The free tier resets on the first of every calendar month.",
        "raw_context": (
            "Pricing FAQ: 'Free tier quotas reset at 00:00 UTC on the first day "
            "of each calendar month.'"
        ),
    },
    {
        "question": "What is the amber-band SLA for finance?",
        "response_text": "Reviewers must clear amber items within 4 business hours.",
        "raw_context": (
            "Per-vertical amber SLA table: 'Finance: 4 business hours from emission "
            "to reviewer decision.'"
        ),
    },
    {
        "question": "Which auditor performed the SOC 2 Type I review?",
        "response_text": "The SOC 2 Type I review was performed by Prescient Assurance.",
        "raw_context": (
            "Trust Center 2026-04-01: 'SOC 2 Type I auditor of record: Prescient "
            "Assurance. Type II window opens 2026-09-01.'"
        ),
    },
]


def score_one(url: str, row: dict, profile: str, timeout: float) -> tuple[float, float, str]:
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
            "x-latence-tenant-id": "bench-latency",
        },
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    t1 = time.perf_counter()
    wall_ms = (t1 - t0) * 1000.0
    # dev_app returns {"output": {..., "took_ms": X}} per the runpod shape.
    inner = body.get("output") or body
    score_ms = float(
        inner.get("took_ms")
        or inner.get("latency_ms")
        or inner.get("took")
        or wall_ms
    )
    band = str(inner.get("band") or "unknown")
    return wall_ms, score_ms, band


def _pct(vals: list[float], q: float) -> float:
    if not vals:
        return 0.0
    vals = sorted(vals)
    k = max(0, min(len(vals) - 1, int(round(q * (len(vals) - 1)))))
    return vals[k]


def run(args: argparse.Namespace) -> dict:
    wall: list[float] = []
    score: list[float] = []
    bands: dict[str, int] = {}
    errors = 0
    rows = FIXTURES * max(1, args.repeats)
    started = time.perf_counter()
    if args.lane == "isolated":
        for row in rows:
            try:
                w, s, b = score_one(args.url, row, args.profile, args.timeout)
                wall.append(w)
                score.append(s)
                bands[b] = bands.get(b, 0) + 1
            except Exception as exc:
                errors += 1
                print(f"ERR isolated: {exc}")
    else:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [pool.submit(score_one, args.url, row, args.profile, args.timeout) for row in rows]
            for fut in as_completed(futures):
                try:
                    w, s, b = fut.result()
                    wall.append(w)
                    score.append(s)
                    bands[b] = bands.get(b, 0) + 1
                except Exception as exc:
                    errors += 1
                    print(f"ERR sustained: {exc}")
    elapsed = time.perf_counter() - started

    queue_ms = [max(0.0, w - s) for w, s in zip(wall, score)]
    summary = {
        "lane": args.lane,
        "profile": args.profile,
        "concurrency": 1 if args.lane == "isolated" else args.concurrency,
        "total_requests": len(rows),
        "successful_requests": len(wall),
        "errors": errors,
        "elapsed_s": round(elapsed, 3),
        "rps": round(len(wall) / elapsed, 3) if elapsed > 0 else 0.0,
        "wall_ms": {
            "p50": round(_pct(wall, 0.50), 2),
            "p95": round(_pct(wall, 0.95), 2),
            "p99": round(_pct(wall, 0.99), 2),
            "mean": round(statistics.fmean(wall), 2) if wall else 0.0,
        },
        "score_ms": {
            "p50": round(_pct(score, 0.50), 2),
            "p95": round(_pct(score, 0.95), 2),
            "p99": round(_pct(score, 0.99), 2),
            "mean": round(statistics.fmean(score), 2) if score else 0.0,
        },
        "queue_ms": {
            "p50": round(_pct(queue_ms, 0.50), 2),
            "p95": round(_pct(queue_ms, 0.95), 2),
            "p99": round(_pct(queue_ms, 0.99), 2),
            "mean": round(statistics.fmean(queue_ms), 2) if queue_ms else 0.0,
        },
        "band_distribution": bands,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000/runsync")
    parser.add_argument("--lane", choices=["isolated", "sustained"], required=True)
    parser.add_argument("--profile", choices=["standard", "quality", "code"], default="standard")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=1, help="how many times to loop FIXTURES")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--output-dir", default="data/latency_bench")
    args = parser.parse_args()

    summary = run(args)
    print(json.dumps(summary, indent=2))

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = f"latency_{args.lane}_{args.profile}_c{summary['concurrency']}.json"
    path = outdir / stem
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
