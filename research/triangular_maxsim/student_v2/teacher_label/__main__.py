"""CLI orchestrator: teacher-label real benchmark rows for student v2.

Usage
-----

    python -m research.triangular_maxsim.student_v2.teacher_label \
        --source ragtruth \
        --out research/triangular_maxsim/student_v2/data/teacher_ragtruth.jsonl \
        [--limit 1000] [--checkpoint-every 500]

    python -m research.triangular_maxsim.student_v2.teacher_label \
        --source veracier \
        --out research/triangular_maxsim/student_v2/data/teacher_veracier.jsonl

Each emitted row follows the same schema as the synthetic-lane rows in
``data/train.jsonl`` so the existing collate / pair-aware sampler /
training loop can consume them without modification.

Hygiene
-------
HaluEval is never loaded here. It stays quarantined for reporting-day
evaluation so we retain the right to publish absolute scores on it.

Resumability
------------
The writer keeps an append-only JSONL + a sibling ``.done`` manifest
of already-labelled ``pair_row_id``s. If interrupted, re-running the
same command picks up exactly where it stopped.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import pathlib
import signal
import threading
import time
from typing import Iterator

from research.triangular_maxsim.student_v2.teacher_label.extract import (
    extract_labels,
)
from research.triangular_maxsim.student_v2.teacher_label.loaders import (
    iter_ragtruth_train,
    iter_veracier_curated,
)
from research.triangular_maxsim.student_v2.teacher_label.service import (
    TeacherConfig,
    build_teacher,
    score_row,
)

logger = logging.getLogger("trace.v2.teacher_label")


def _load_done(done_path: pathlib.Path) -> set[str]:
    if not done_path.exists():
        return set()
    with done_path.open(encoding="utf-8") as fh:
        return {line.strip() for line in fh if line.strip()}


def _iter_source(source: str) -> Iterator[dict]:
    if source == "ragtruth":
        yield from iter_ragtruth_train()
    elif source == "veracier":
        yield from iter_veracier_curated()
    else:
        raise ValueError(f"unknown --source {source!r}")


def _emit(row: dict, labels: dict) -> dict:
    """Merge the normalized row with teacher-extracted labels."""

    out = dict(row)
    out.update(labels)
    return out


def _configure_logging(verbose: bool) -> None:
    level = logging.INFO if not verbose else logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, choices=["ragtruth", "veracier"])
    parser.add_argument("--out", required=True, type=pathlib.Path)
    parser.add_argument("--limit", type=int, default=0, help="0 = no limit")
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument(
        "--max-failures",
        type=int,
        default=100,
        help="abort if the teacher fails on this many rows in a row",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--profile",
        default="quality",
        choices=["quality", "standard"],
        help="quality = full NLI cascade + atomic claims (slower, richer)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="number of concurrent teacher requests (vLLM handles ~128)",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    _configure_logging(args.verbose)

    out_path: pathlib.Path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_path = out_path.with_suffix(out_path.suffix + ".done")

    already = _load_done(done_path)
    logger.info(
        "resuming: %d rows already labelled (existing %s)",
        len(already),
        out_path.name,
    )

    cfg = TeacherConfig(device=args.device)
    service, cfg = build_teacher(cfg)

    shutdown = {"requested": False}

    def _on_sigint(signum, frame):  # pragma: no cover - interactive
        logger.warning("signal %s received; flushing before exit", signum)
        shutdown["requested"] = True

    signal.signal(signal.SIGINT, _on_sigint)
    signal.signal(signal.SIGTERM, _on_sigint)

    # Gather the input rows that still need labelling so we can fan
    # them out onto a thread pool. We only read rows that are not
    # already in `already`, so re-runs after interruption are cheap.
    pending: list[dict] = []
    for row in _iter_source(args.source):
        if args.limit and len(pending) >= args.limit:
            break
        if row["pair_row_id"] in already:
            continue
        pending.append(row)
    logger.info("pending: %d rows to label (concurrency=%d)", len(pending), args.concurrency)

    write_lock = threading.Lock()
    kept = 0
    consecutive_failures = 0
    t0 = time.perf_counter()
    last_ckpt = t0

    def _work(row: dict) -> tuple[dict, dict | None, Exception | None]:
        try:
            resp = score_row(service, cfg, row)
            return row, resp, None
        except Exception as exc:  # noqa: BLE001 - robust outer loop
            return row, None, exc

    with (
        out_path.open("a", encoding="utf-8") as out_fh,
        done_path.open("a", encoding="utf-8") as done_fh,
        concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, args.concurrency)
        ) as pool,
    ):
        futures = [pool.submit(_work, row) for row in pending]
        for fut in concurrent.futures.as_completed(futures):
            if shutdown["requested"]:
                break
            row, resp, exc = fut.result()
            rid = row["pair_row_id"]
            if exc is not None:
                consecutive_failures += 1
                logger.warning(
                    "teacher failed on row %s: %r (consec=%d)",
                    rid,
                    exc,
                    consecutive_failures,
                )
                if consecutive_failures >= args.max_failures:
                    logger.error("too many consecutive failures; aborting")
                    break
                continue
            consecutive_failures = 0
            labels = extract_labels(resp, gold_band=row.get("gold_band"))
            merged = _emit(row, labels)
            with write_lock:
                out_fh.write(json.dumps(merged, ensure_ascii=False) + "\n")
                done_fh.write(rid + "\n")
                kept += 1
                now = time.perf_counter()
                if kept % args.checkpoint_every == 0 or (now - last_ckpt) > 60:
                    out_fh.flush()
                    done_fh.flush()
                    rate = kept / max(1e-6, now - t0)
                    logger.info(
                        "progress: kept=%d/%d rate=%.2f rows/s elapsed=%.1fs",
                        kept,
                        len(pending),
                        rate,
                        now - t0,
                    )
                    last_ckpt = now

    elapsed = time.perf_counter() - t0
    logger.info(
        "teacher-label done: source=%s kept=%d elapsed=%.1fs rate=%.2f rows/s",
        args.source,
        kept,
        elapsed,
        kept / max(1e-6, elapsed),
    )


if __name__ == "__main__":
    main()
