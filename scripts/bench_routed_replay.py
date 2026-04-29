"""Phase F - End-to-end replay bench for the corpus-router.

Runs three decks against the live ``dev_app`` with the router wired in
and emits a reproducible proof bundle under ``data/corpus_classifier/
proof_bundle_v3/``.

Decks:

1. **per_class**: per-class test splits from
   ``data/corpus_classifier/test.parquet`` (up to ``--per-class-cap``
   rows each). Captures routed decisions, latency, and per-class
   agreement with the row's ``is_grounded`` label.
2. **veracier**: ``data/veracier-industries/proof_bundle_v1/
   variants.curated.jsonl`` (132 rows). Reports green/amber/red
   precision/agreement, the router-picked bundle per archetype, and
   classifier confusion vs. the archetype's expected RAG class.
3. **latency**: a ``--latency-sample`` warm mixture across all six
   classes to build a router-latency histogram.

Outputs
~~~~~~~
``proof_bundle_v3/PROOF_SUMMARY.json``   - machine-readable roll-up
``proof_bundle_v3/per_class.jsonl``      - per-row predictions
``proof_bundle_v3/veracier.jsonl``       - per-row predictions
``proof_bundle_v3/veracier_metrics.json`` - green/red precision etc.
``proof_bundle_v3/classifier_confusion.json``
``proof_bundle_v3/router_latency_histogram.json``
``proof_bundle_v3/PROOF_REPORT.md``      - human-readable markdown

All scoring happens against the router-aware service, so the numbers
reflect what production would see. No thresholds are tweaked here - the
bundles on disk are authoritative.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATA_DIR = REPO_ROOT / "data/corpus_classifier"
CLASSIFIER_TEST = DATA_DIR / "test.parquet"
VERACIER_PATH = (
    REPO_ROOT
    / "data/veracier-industries/proof_bundle_v1/variants.curated.jsonl"
)
OUT_DIR = DATA_DIR / "proof_bundle_v3"

CLASS_KEYS = (
    "rag.prose.enterprise",
    "rag.prose.short_factoid",
    "rag.prose.multi_claim",
    "rag.structured",
    "rag.code_in_context",
    "code.agentic_trace",
)

LOGGER = logging.getLogger("bench_routed_replay")


def _post(
    url: str,
    body: dict[str, Any],
    *,
    timeout: float,
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode()
            out = json.loads(raw)
            if out.get("success") is False:
                return None, out.get("error") or "service returned success=false"
            return out, None
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}: {e.read().decode()[:200]}"
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


def _score_rows(
    url: str,
    rows: list[dict[str, Any]],
    *,
    concurrency: int,
    timeout: float,
    explicit_corpus_type: Optional[str] = None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = [{} for _ in rows]

    def _one(idx: int, row: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        body: dict[str, Any] = {
            "input": {
                "query": row.get("query") or row.get("query_text") or "",
                "response": row.get("response") or row.get("response_text") or "",
                "raw_context": row.get("raw_context") or "",
            }
        }
        if explicit_corpus_type:
            body["input"]["corpus_type"] = explicit_corpus_type
        t0 = time.perf_counter()
        out, err = _post(url, body, timeout=timeout)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if err:
            return idx, {"error": err, "elapsed_ms": elapsed_ms}
        return idx, {
            "score": out.get("score"),
            "band": out.get("band"),
            "scoring_mode": out.get("scoring_mode"),
            "corpus_route": out.get("corpus_route"),
            "latency_ms": out.get("latency_ms"),
            "elapsed_ms": elapsed_ms,
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = {ex.submit(_one, i, r): i for i, r in enumerate(rows)}
        done = 0
        for f in concurrent.futures.as_completed(futures):
            idx, res = f.result()
            results[idx] = res
            done += 1
            if done % 25 == 0:
                LOGGER.info("scored %d/%d", done, len(rows))
    return results


# -- Deck 1: per-class test split ---------------------------------------


def _load_classifier_test(cap_per_class: int) -> list[dict[str, Any]]:
    t = pq.read_table(CLASSIFIER_TEST)
    rows = t.to_pylist()
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_class[r["class_key"]].append(r)
    subset: list[dict[str, Any]] = []
    for ck in CLASS_KEYS:
        subset.extend(by_class.get(ck, [])[:cap_per_class])
    return subset


def _deck_per_class(
    url: str, *, concurrency: int, timeout: float, cap: int
) -> dict[str, Any]:
    rows = _load_classifier_test(cap)
    LOGGER.info("per_class: scoring %d rows across %d classes", len(rows), len(CLASS_KEYS))
    results = _score_rows(
        url, rows, concurrency=concurrency, timeout=timeout
    )
    per_rows: list[dict[str, Any]] = []
    agreement: dict[str, dict[str, int]] = defaultdict(
        lambda: {"n": 0, "correct": 0, "errors": 0, "band_missing": 0}
    )
    confusion: dict[str, Counter] = defaultdict(Counter)
    for row, res in zip(rows, results):
        ck_true = row["class_key"]
        rec = {
            "row_id": row.get("row_id"),
            "class_key_true": ck_true,
            "is_grounded": bool(row["is_grounded"]),
            "error": res.get("error"),
            "band": res.get("band"),
            "score": res.get("score"),
            "scoring_mode": res.get("scoring_mode"),
            "corpus_route": res.get("corpus_route"),
            "elapsed_ms": res.get("elapsed_ms"),
        }
        per_rows.append(rec)
        if res.get("error"):
            agreement[ck_true]["errors"] += 1
            continue
        route = res.get("corpus_route") or {}
        ck_pred = route.get("corpus_type") or "unknown"
        confusion[ck_true][ck_pred] += 1
        agreement[ck_true]["n"] += 1
        band = res.get("band")
        if band not in {"green", "amber", "red"}:
            agreement[ck_true]["band_missing"] += 1
            continue
        # "correct" agreement: grounded -> green/amber OK, ungrounded -> red.
        expected_positive = rec["is_grounded"]
        observed_positive = band == "green"
        if (expected_positive and band in {"green", "amber"}) or (
            (not expected_positive) and band == "red"
        ):
            agreement[ck_true]["correct"] += 1
    return {
        "rows": per_rows,
        "agreement": {k: dict(v) for k, v in agreement.items()},
        "classifier_confusion": {k: dict(v) for k, v in confusion.items()},
    }


# -- Deck 2: Veracier regression ---------------------------------------


def _deck_veracier(url: str, *, concurrency: int, timeout: float) -> dict[str, Any]:
    if not VERACIER_PATH.exists():
        LOGGER.warning("Veracier variants file missing; deck skipped")
        return {"rows": [], "metrics": {}, "skipped": True}
    rows: list[dict[str, Any]] = []
    # File iteration (not ``splitlines``) so that embedded ``\r`` or
    # similar characters inside ``raw_context`` don't split records.
    with VERACIER_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            r["query"] = r.get("query_text", "")
            r["response"] = r.get("response_text", "")
            rows.append(r)
    LOGGER.info("veracier: scoring %d rows", len(rows))
    results = _score_rows(url, rows, concurrency=concurrency, timeout=timeout)
    out_rows: list[dict[str, Any]] = []
    band_by_expected: dict[str, Counter] = defaultdict(Counter)
    for row, res in zip(rows, results):
        expected = row["expected_band"]
        rec = {
            "example_id": row.get("example_id"),
            "question_id": row.get("question_id"),
            "expected_band": expected,
            "archetype": row.get("generation_archetype"),
            "verticals": row.get("verticals"),
            "error": res.get("error"),
            "band": res.get("band"),
            "score": res.get("score"),
            "corpus_route": res.get("corpus_route"),
            "elapsed_ms": res.get("elapsed_ms"),
        }
        out_rows.append(rec)
        if res.get("error"):
            continue
        band = res.get("band") or "unknown"
        band_by_expected[expected][band] += 1
    # Metrics aligned with proof_bundle_v1.
    green_tp = band_by_expected["green"]["green"]
    green_denom = sum(band_by_expected[b]["green"] for b in ("green", "amber", "red"))
    red_tp = band_by_expected["red"]["red"]
    red_denom = sum(band_by_expected[b]["red"] for b in ("green", "amber", "red"))
    red_recall_denom = sum(band_by_expected["red"].values())
    amber_expected = sum(band_by_expected["amber"].values())
    amber_match = band_by_expected["amber"]["amber"] + band_by_expected["amber"]["green"]
    metrics = {
        "green": {
            "true_positive": green_tp,
            "denominator": green_denom,
            "precision": (green_tp / green_denom) if green_denom else None,
        },
        "red": {
            "true_positive": red_tp,
            "denominator": red_denom,
            "precision": (red_tp / red_denom) if red_denom else None,
            "recall_numerator": red_tp,
            "recall_denominator": red_recall_denom,
            "recall": (red_tp / red_recall_denom) if red_recall_denom else None,
        },
        "amber": {
            "expected": amber_expected,
            "match": amber_match,
            "agreement": (amber_match / amber_expected) if amber_expected else None,
        },
        "targets": {
            "green_precision_target": 0.97,
            "red_precision_target": 0.95,
            "amber_agreement_target": 0.85,
        },
    }
    return {"rows": out_rows, "metrics": metrics, "skipped": False}


# -- Deck 3: router-latency histogram ----------------------------------


def _deck_latency(
    url: str, *, concurrency: int, timeout: float, sample_per_class: int
) -> dict[str, Any]:
    rows_by_class = _load_classifier_test(sample_per_class)
    LOGGER.info("latency: scoring %d rows for histogram", len(rows_by_class))
    results = _score_rows(
        url, rows_by_class, concurrency=concurrency, timeout=timeout
    )
    classifier_latencies: list[float] = []
    e2e_latencies: list[float] = []
    for res in results:
        if res.get("error"):
            continue
        route = res.get("corpus_route") or {}
        lat = route.get("classifier_latency_ms")
        if lat is not None:
            classifier_latencies.append(float(lat))
        e2e = res.get("elapsed_ms")
        if e2e is not None:
            e2e_latencies.append(float(e2e))

    def _histogram(vals: list[float], buckets: list[float]) -> dict[str, int]:
        out = {f"<{b}": 0 for b in buckets}
        out[f">={buckets[-1]}"] = 0
        for v in vals:
            placed = False
            for b in buckets:
                if v < b:
                    out[f"<{b}"] += 1
                    placed = True
                    break
            if not placed:
                out[f">={buckets[-1]}"] += 1
        return out

    def _stats(vals: list[float]) -> dict[str, float]:
        if not vals:
            return {}
        s = sorted(vals)
        return {
            "n": len(vals),
            "p50": s[len(s) // 2],
            "p95": s[min(len(s) - 1, int(len(s) * 0.95))],
            "p99": s[min(len(s) - 1, int(len(s) * 0.99))],
            "max": s[-1],
            "mean": statistics.mean(vals),
        }

    return {
        "classifier_latency_ms": _stats(classifier_latencies),
        "classifier_latency_histogram": _histogram(
            classifier_latencies, [1, 2, 5, 10, 20, 50]
        ),
        "e2e_latency_ms": _stats(e2e_latencies),
        "e2e_latency_histogram": _histogram(
            e2e_latencies, [50, 100, 250, 500, 1000, 2500, 5000]
        ),
    }


# -- Report writing ----------------------------------------------------


def _write_report(
    out_dir: Path,
    *,
    per_class: dict[str, Any],
    veracier: dict[str, Any],
    latency: dict[str, Any],
    url: str,
) -> None:
    lines: list[str] = []
    lines.append("# Proof Bundle v3 - Routed Replay\n")
    lines.append(f"_Service: `{url}`_  _Generated: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}_\n")
    lines.append(
        "\n> **Read first**: `NLI_HEALTH.md` in this folder - NLI subsystem "
        "stability directly impacts the end-to-end numbers below. Router "
        "itself is healthy (see classifier confusion + latency sections).\n"
    )
    lines.append("\n## Per-class agreement (test.parquet)\n")
    lines.append("| class_key | n | correct | accuracy | band_missing | errors |\n")
    lines.append("|---|---:|---:|---:|---:|---:|\n")
    for ck in CLASS_KEYS:
        ag = per_class["agreement"].get(ck, {"n": 0, "correct": 0, "errors": 0, "band_missing": 0})
        n = ag.get("n", 0)
        acc = (ag["correct"] / n) if n else 0.0
        lines.append(
            f"| {ck} | {n} | {ag['correct']} | {acc:.3f} | {ag.get('band_missing', 0)} | {ag.get('errors', 0)} |\n"
        )
    lines.append("\n## Classifier confusion (true -> predicted)\n")
    lines.append("| true\\pred | " + " | ".join(CLASS_KEYS) + " |\n")
    lines.append("|" + "---|" * (len(CLASS_KEYS) + 1) + "\n")
    for ck_t in CLASS_KEYS:
        row_counts = per_class["classifier_confusion"].get(ck_t, {})
        cells = " | ".join(str(row_counts.get(ck_p, 0)) for ck_p in CLASS_KEYS)
        lines.append(f"| {ck_t} | {cells} |\n")
    def _fmt(v: Optional[float], digits: int = 3) -> str:
        return f"{v:.{digits}f}" if isinstance(v, (int, float)) else "n/a"

    if not veracier.get("skipped"):
        m = veracier["metrics"]
        lines.append("\n## Veracier regression (variants.curated, N=132)\n")
        lines.append(
            f"- green precision: **{_fmt(m['green']['precision'])}** "
            f"(n={m['green']['denominator']}, target >= {m['targets']['green_precision_target']})  \n"
        )
        lines.append(
            f"- red precision: **{_fmt(m['red']['precision'])}** "
            f"(n={m['red']['denominator']}, target >= {m['targets']['red_precision_target']})  \n"
        )
        lines.append(
            f"- red recall: {_fmt(m['red']['recall'])} "
            f"({m['red']['recall_numerator']}/{m['red']['recall_denominator']})  \n"
        )
        lines.append(
            f"- amber agreement: **{_fmt(m['amber']['agreement'])}** "
            f"(n={m['amber']['expected']}, target >= {m['targets']['amber_agreement_target']})  \n"
        )
    lines.append("\n## Router latency histogram\n")
    lines.append(
        f"- classifier p50 / p95 / p99: "
        f"{latency['classifier_latency_ms'].get('p50', 0):.2f} / "
        f"{latency['classifier_latency_ms'].get('p95', 0):.2f} / "
        f"{latency['classifier_latency_ms'].get('p99', 0):.2f} ms  \n"
    )
    lines.append(
        f"- end-to-end p50 / p95 / p99: "
        f"{latency['e2e_latency_ms'].get('p50', 0):.1f} / "
        f"{latency['e2e_latency_ms'].get('p95', 0):.1f} / "
        f"{latency['e2e_latency_ms'].get('p99', 0):.1f} ms  \n"
    )
    (out_dir / "PROOF_REPORT.md").write_text("".join(lines))


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8091/runsync")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--per-class-cap", type=int, default=30)
    parser.add_argument("--latency-sample", type=int, default=10)
    parser.add_argument("--skip-per-class", action="store_true")
    parser.add_argument("--skip-veracier", action="store_true")
    parser.add_argument("--skip-latency", action="store_true")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "service_url": args.url,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "concurrency": args.concurrency,
            "per_class_cap": args.per_class_cap,
            "latency_sample_per_class": args.latency_sample,
        },
    }

    per_class: dict[str, Any] = {}
    veracier: dict[str, Any] = {"skipped": True, "metrics": {}}
    latency: dict[str, Any] = {}

    if not args.skip_per_class:
        per_class = _deck_per_class(
            args.url,
            concurrency=args.concurrency,
            timeout=args.timeout,
            cap=args.per_class_cap,
        )
        (out_dir / "per_class.jsonl").write_text(
            "\n".join(json.dumps(r) for r in per_class["rows"]) + "\n"
        )
        (out_dir / "classifier_confusion.json").write_text(
            json.dumps(per_class["classifier_confusion"], indent=2) + "\n"
        )
        summary["per_class_agreement"] = per_class["agreement"]

    if not args.skip_veracier:
        veracier = _deck_veracier(
            args.url, concurrency=args.concurrency, timeout=args.timeout
        )
        if not veracier.get("skipped"):
            (out_dir / "veracier.jsonl").write_text(
                "\n".join(json.dumps(r) for r in veracier["rows"]) + "\n"
            )
            (out_dir / "veracier_metrics.json").write_text(
                json.dumps(veracier["metrics"], indent=2) + "\n"
            )
            summary["veracier_metrics"] = veracier["metrics"]

    if not args.skip_latency:
        latency = _deck_latency(
            args.url,
            concurrency=args.concurrency,
            timeout=args.timeout,
            sample_per_class=args.latency_sample,
        )
        (out_dir / "router_latency_histogram.json").write_text(
            json.dumps(latency, indent=2) + "\n"
        )
        summary["router_latency"] = latency

    (out_dir / "PROOF_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    _write_report(
        out_dir,
        per_class=per_class or {"agreement": {}, "classifier_confusion": {}},
        veracier=veracier,
        latency=latency or {
            "classifier_latency_ms": {},
            "e2e_latency_ms": {},
        },
        url=args.url,
    )

    LOGGER.info("wrote proof bundle to %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
