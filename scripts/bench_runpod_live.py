"""Small live benchmark against the latence-trace RunPod serverless endpoint.

Submits the 20-case handcrafted suite (research/triangular_maxsim/cases.py) to
an async RunPod /run endpoint, polls /status until terminal, and prints:

    * per-case score / primary metric / band / coverage / unused / latency
    * Mann-Whitney AUROC on grounded-vs-ungrounded using `score`
    * p50/p95 server latency (handler reports "latency_ms") and wall clock

Usage:

    python scripts/bench_runpod_live.py \
        --endpoint-id campegd1dctnx2 \
        --api-key  $RUNPOD_API_KEY \
        --concurrency 4

A subset can be run with --cases G1,G3,U1,U4,A7 for quick smoke tests.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import httpx

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from research.triangular_maxsim.cases import CASES  # noqa: E402


def _auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    pairs = sorted(zip(scores, labels))
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = [0.0] * len(pairs)
    idx = 0
    while idx < len(pairs):
        jdx = idx + 1
        while jdx < len(pairs) and pairs[jdx][0] == pairs[idx][0]:
            jdx += 1
        avg_rank = (idx + jdx + 1) / 2.0
        for k in range(idx, jdx):
            ranks[k] = avg_rank
        idx = jdx
    sum_pos = sum(rank for rank, (_s, label) in zip(ranks, pairs) if label == 1)
    return float((sum_pos - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg))


def _percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return float("nan")
    data = sorted(values)
    k = (len(data) - 1) * pct
    lo = int(k)
    hi = min(lo + 1, len(data) - 1)
    return data[lo] + (data[hi] - data[lo]) * (k - lo)


async def _submit(
    client: httpx.AsyncClient,
    endpoint_id: str,
    headers: dict[str, str],
    case,
    verbose: bool,
) -> dict[str, Any]:
    payload = {
        "input": {
            "query": case.query,
            "context": case.context,
            "response": case.response,
            "include_triangular_diagnostics": True,
            "evidence_limit": 4,
            "verbose": verbose,
        }
    }
    submit_url = f"https://api.runpod.ai/v2/{endpoint_id}/run"
    status_url_template = f"https://api.runpod.ai/v2/{endpoint_id}/status/{{job_id}}"

    wall_start = time.perf_counter()
    resp = await client.post(submit_url, json=payload, headers=headers)
    resp.raise_for_status()
    job = resp.json()
    job_id = job["id"]

    while True:
        await asyncio.sleep(0.5)
        status_resp = await client.get(
            status_url_template.format(job_id=job_id), headers=headers
        )
        status_resp.raise_for_status()
        status_body = status_resp.json()
        status = status_body.get("status", "")
        if status in {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}:
            wall_ms = (time.perf_counter() - wall_start) * 1000.0
            return {
                "case": case,
                "status": status,
                "output": status_body.get("output"),
                "error": status_body.get("error"),
                "wall_ms": wall_ms,
                "job_id": job_id,
            }


async def _runsync_submit(
    client: httpx.AsyncClient,
    endpoint_id: str,
    headers: dict[str, str],
    case,
    verbose: bool,
) -> dict[str, Any]:
    """Use /runsync so the RunPod API blocks until terminal (simpler path)."""
    payload = {
        "input": {
            "query": case.query,
            "context": case.context,
            "response": case.response,
            "include_triangular_diagnostics": True,
            "evidence_limit": 4,
            "verbose": verbose,
        }
    }
    url = f"https://api.runpod.ai/v2/{endpoint_id}/runsync"

    wall_start = time.perf_counter()
    resp = await client.post(url, json=payload, headers=headers)
    resp.raise_for_status()
    body = resp.json()
    wall_ms = (time.perf_counter() - wall_start) * 1000.0
    status = body.get("status", "")
    if status not in {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}:
        status_url = f"https://api.runpod.ai/v2/{endpoint_id}/status/{body['id']}"
        while True:
            await asyncio.sleep(0.5)
            status_resp = await client.get(status_url, headers=headers)
            status_resp.raise_for_status()
            body = status_resp.json()
            status = body.get("status", "")
            if status in {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}:
                wall_ms = (time.perf_counter() - wall_start) * 1000.0
                break
    return {
        "case": case,
        "status": status,
        "output": body.get("output"),
        "error": body.get("error"),
        "wall_ms": wall_ms,
        "job_id": body.get("id"),
    }


async def _run(
    endpoint_id: str,
    api_key: str,
    cases: Sequence,
    concurrency: int,
    verbose: bool,
    use_runsync: bool,
    timeout_s: float,
) -> list[dict[str, Any]]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    limits = httpx.Limits(max_connections=concurrency * 2, max_keepalive_connections=concurrency * 2)
    timeout = httpx.Timeout(timeout_s, connect=10.0)
    sem = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        async def _worker(case):
            async with sem:
                submit_fn = _runsync_submit if use_runsync else _submit
                return await submit_fn(client, endpoint_id, headers, case, verbose)

        return await asyncio.gather(*[_worker(c) for c in cases])


def _label_to_int(label: str) -> int | None:
    if label == "grounded":
        return 1
    if label == "ungrounded":
        return 0
    return None


def _print_report(results: list[dict[str, Any]], *, verbose_dump: Path | None) -> int:
    rows = []
    g_u_scores: list[tuple[float, int]] = []
    server_latencies_ms: list[float] = []
    wall_latencies_ms: list[float] = []
    errors: list[dict[str, Any]] = []

    for entry in results:
        case = entry["case"]
        status = entry["status"]
        if status != "COMPLETED":
            errors.append(
                {
                    "id": case.id,
                    "label": case.label,
                    "status": status,
                    "error": entry.get("error"),
                    "wall_ms": round(entry["wall_ms"], 1),
                }
            )
            continue
        output = entry["output"] or {}
        if not output.get("success"):
            errors.append(
                {
                    "id": case.id,
                    "label": case.label,
                    "status": "ERROR_RESPONSE",
                    "error": output.get("error") or output.get("error_code"),
                    "wall_ms": round(entry["wall_ms"], 1),
                }
            )
            continue
        score = float(output.get("score", float("nan")))
        server_latency = float(output.get("latency_ms", float("nan")))
        server_latencies_ms.append(server_latency)
        wall_latencies_ms.append(entry["wall_ms"])

        label_int = _label_to_int(case.label)
        if label_int is not None:
            g_u_scores.append((score, label_int))

        rows.append(
            {
                "id": case.id,
                "label": case.label,
                "subcategory": case.subcategory,
                "score": score,
                "primary_metric": output.get("primary_metric"),
                "band": output.get("band"),
                "coverage": output.get("context_coverage_ratio"),
                "unused": output.get("context_unused_ratio"),
                "uncertain": output.get("context_uncertain_ratio"),
                "usage_used": (output.get("support_units_usage") or {}).get("used"),
                "usage_unused": (output.get("support_units_usage") or {}).get("unused"),
                "usage_uncertain": (output.get("support_units_usage") or {}).get("uncertain"),
                "server_latency_ms": server_latency,
                "wall_ms": entry["wall_ms"],
                "job_id": entry.get("job_id"),
            }
        )

    print()
    print("=" * 120)
    print("Per-case results")
    print("=" * 120)
    header = (
        f"{'id':<4} {'label':<11} {'sub':<14} "
        f"{'score':>7} {'band':<12} {'cov':>6} {'unused':>7} {'uncert':>7} "
        f"{'U/N/?':>9} {'srv_ms':>8} {'wall_ms':>9}"
    )
    print(header)
    print("-" * len(header))
    for row in sorted(rows, key=lambda r: (r["label"], r["id"])):
        sub = row["subcategory"] or "-"
        cov = f"{row['coverage']:.3f}" if row["coverage"] is not None else "-"
        unused = f"{row['unused']:.3f}" if row["unused"] is not None else "-"
        uncert = f"{row['uncertain']:.3f}" if row["uncertain"] is not None else "-"
        used_n = row.get("usage_used")
        unused_n = row.get("usage_unused")
        uncert_n = row.get("usage_uncertain")
        usage = (
            f"{used_n}/{unused_n}/{uncert_n}"
            if None not in (used_n, unused_n, uncert_n)
            else "-"
        )
        print(
            f"{row['id']:<4} {row['label']:<11} {sub:<14} "
            f"{row['score']:>7.4f} {str(row['band']):<12} "
            f"{cov:>6} {unused:>7} {uncert:>7} {usage:>9} "
            f"{row['server_latency_ms']:>8.1f} {row['wall_ms']:>9.1f}"
        )

    auc = _auroc([s for s, _ in g_u_scores], [label for _, label in g_u_scores])
    p50 = _percentile(server_latencies_ms, 0.5)
    p95 = _percentile(server_latencies_ms, 0.95)
    wall_p50 = _percentile(wall_latencies_ms, 0.5)
    wall_p95 = _percentile(wall_latencies_ms, 0.95)

    grounded_mean = statistics.fmean(s for s, lbl in g_u_scores if lbl == 1) if any(lbl == 1 for _s, lbl in g_u_scores) else float("nan")
    ungrounded_mean = statistics.fmean(s for s, lbl in g_u_scores if lbl == 0) if any(lbl == 0 for _s, lbl in g_u_scores) else float("nan")
    ambiguous_rows = [r for r in rows if r["label"] == "ambiguous"]
    ambiguous_mean = statistics.fmean(r["score"] for r in ambiguous_rows) if ambiguous_rows else float("nan")

    print()
    print("=" * 120)
    print("Aggregate")
    print("=" * 120)
    print(f"  completed                      : {len(rows)}")
    print(f"  errors                         : {len(errors)}")
    print(f"  grounded-vs-ungrounded AUROC   : {auc:.4f}")
    print(f"  mean score grounded  (n={sum(1 for _s, lbl in g_u_scores if lbl == 1):>2}) : {grounded_mean:.4f}")
    print(f"  mean score ungrounded(n={sum(1 for _s, lbl in g_u_scores if lbl == 0):>2}) : {ungrounded_mean:.4f}")
    print(f"  mean score ambiguous (n={len(ambiguous_rows):>2}) : {ambiguous_mean:.4f}")
    print(f"  server latency p50 / p95 (ms)  : {p50:>7.1f} / {p95:>7.1f}")
    print(f"  wall  latency p50 / p95 (ms)   : {wall_p50:>7.1f} / {wall_p95:>7.1f}")

    if errors:
        print()
        print("=" * 120)
        print(f"Errors ({len(errors)})")
        print("=" * 120)
        for err in errors:
            print(f"  {err['id']:<4} {err['label']:<11} {err['status']:<18} wall={err['wall_ms']} ms  err={err['error']}")

    if verbose_dump is not None:
        dump = {
            "cases": rows,
            "errors": errors,
            "aggregate": {
                "auroc_grounded_vs_ungrounded": auc,
                "mean_score_grounded": grounded_mean,
                "mean_score_ungrounded": ungrounded_mean,
                "mean_score_ambiguous": ambiguous_mean,
                "server_p50_ms": p50,
                "server_p95_ms": p95,
                "wall_p50_ms": wall_p50,
                "wall_p95_ms": wall_p95,
            },
        }
        verbose_dump.write_text(json.dumps(dump, indent=2, sort_keys=True, default=str))
        print(f"\n  wrote detailed report -> {verbose_dump}")

    return 0 if not errors else 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint-id", default=os.environ.get("RUNPOD_ENDPOINT_ID"))
    parser.add_argument("--api-key", default=os.environ.get("RUNPOD_API_KEY"))
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout-s", type=float, default=600.0)
    parser.add_argument("--cases", default=None, help="comma-separated case IDs (default: all 20)")
    parser.add_argument("--verbose", action="store_true", help="request full response payloads")
    parser.add_argument("--use-run-poll", action="store_true",
                        help="use /run + /status polling instead of /runsync")
    parser.add_argument("--dump", default=None, help="path to write a detailed JSON report")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.endpoint_id:
        print("error: --endpoint-id (or RUNPOD_ENDPOINT_ID) is required", file=sys.stderr)
        return 2
    if not args.api_key:
        print("error: --api-key (or RUNPOD_API_KEY) is required", file=sys.stderr)
        return 2

    case_bank = {c.id: c for c in CASES}
    if args.cases:
        requested_ids = [cid.strip() for cid in args.cases.split(",") if cid.strip()]
        missing = [cid for cid in requested_ids if cid not in case_bank]
        if missing:
            print(f"error: unknown case IDs: {missing}", file=sys.stderr)
            return 2
        selected = [case_bank[cid] for cid in requested_ids]
    else:
        selected = list(CASES)

    print(f"Running {len(selected)} cases against endpoint {args.endpoint_id} "
          f"(concurrency={args.concurrency}, transport={'/run+poll' if args.use_run_poll else '/runsync'})")
    results = asyncio.run(
        _run(
            endpoint_id=args.endpoint_id,
            api_key=args.api_key,
            cases=selected,
            concurrency=args.concurrency,
            verbose=args.verbose,
            use_runsync=not args.use_run_poll,
            timeout_s=args.timeout_s,
        )
    )

    dump_path = Path(args.dump) if args.dump else None
    return _print_report(results, verbose_dump=dump_path)


if __name__ == "__main__":
    raise SystemExit(main())
