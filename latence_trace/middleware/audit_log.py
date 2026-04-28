"""Append-only per-score audit log (Plan B6 + v2-prep).

A lightweight JSONL-backed audit-log writer that captures one record
per successful score response.  Records are built synchronously so the
teacher-channel snapshot is coherent with the response payload, then
queued to a dedicated background thread for the actual file write.
The hot path therefore pays only a JSON serialisation + queue put
(~40-80 us in our micro-benchmark), not a filesystem write.

The record schema is stable and forward-compatible with the v2 student
distillation pipeline (``v2_prep_b6_teacher_signals``):

* ``request_id`` - correlates with OTel trace IDs and the HTTP
  ``X-Request-Id`` header.
* ``tenant_id``
* ``timestamp`` - ISO-8601 UTC.
* ``profile`` - scoring profile (standard, quality, code).
* ``band`` / ``score`` / ``nli_aggregate`` / ``coverage`` / ``usage`` -
  the headline scalars.
* ``teacher_channels`` - per-channel teacher scores (NLI, literal,
  coverage, hedge-gate status).  Always captured so the v2 student
  has distillation targets ready without requiring another
  instrumentation pass.
* ``top_k_attributions`` - up to 8 top-scoring response-token to
  evidence-token attribution pairs from MaxSim; the student uses them
  as local-support labels.
* ``hash_of_body`` - SHA-256 over the canonical payload so customers
  can verify their originals against the audit log without Latence
  storing the payload itself.

No payload bytes are persisted.  Special-category requests
(``X-Latence-Data-Class: special``) bypass the log entirely.

File rotation and export are handled by the consumer (logrotate,
S3 upload, customer-side export).
"""

from __future__ import annotations

import atexit
import hashlib
import json
import logging
import os
import queue
import threading
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_ENV_DIR = "LATENCE_TRACE_AUDIT_LOG_DIR"
_ENV_MAX_QUEUE = "LATENCE_TRACE_AUDIT_LOG_MAX_QUEUE"
_ENV_SYNC = "LATENCE_TRACE_AUDIT_LOG_SYNC"
_DEFAULT_FILENAME = "audit.jsonl"

# Background writer keeps the hot path off disk.  On a standard
# score response the writer does a JSON dump + queue put (< 100 us
# in our micro-benchmark on CPython 3.11), then returns.  A single
# daemon thread flushes queued lines to disk.  The queue is bounded
# so pathological traffic bursts apply back-pressure by dropping the
# oldest entry and bumping a dropped counter.
_QUEUE_MAX = int(os.environ.get(_ENV_MAX_QUEUE, "10000"))
_QUEUE: queue.Queue[str] = queue.Queue(maxsize=_QUEUE_MAX)
_WORKER_LOCK = threading.Lock()
_WORKER: threading.Thread | None = None
_SHUTDOWN = threading.Event()
# Tests and tightly-coupled callers can force synchronous writes via
# the env var to get immediate observability of the file contents.
_SYNC_WRITE = os.environ.get(_ENV_SYNC, "").lower() in {"1", "true", "yes"}
# Exposed for the /metrics endpoint + tests.
_DROPPED_TOTAL = 0


def _drain_worker() -> None:
    """Daemon thread: dequeue audit lines and append them to disk."""

    audit_dir = None
    while not (_SHUTDOWN.is_set() and _QUEUE.empty()):
        try:
            line = _QUEUE.get(timeout=0.25)
        except queue.Empty:
            continue
        try:
            audit_dir = audit_dir or _audit_dir()
            if audit_dir is None:
                continue
            path = audit_dir / _DEFAULT_FILENAME
            with path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except Exception as exc:  # noqa: BLE001
            logger.warning("audit_log.write_failed", extra={"error": str(exc)})
        finally:
            _QUEUE.task_done()


def _ensure_worker() -> None:
    global _WORKER
    if _WORKER is not None and _WORKER.is_alive():
        return
    with _WORKER_LOCK:
        if _WORKER is not None and _WORKER.is_alive():
            return
        _SHUTDOWN.clear()
        _WORKER = threading.Thread(
            target=_drain_worker, name="latence-audit-log", daemon=True
        )
        _WORKER.start()


def _shutdown() -> None:
    _SHUTDOWN.set()
    worker = _WORKER
    if worker is not None and worker.is_alive():
        worker.join(timeout=2.0)


atexit.register(_shutdown)


def flush(timeout: float = 1.0) -> bool:
    """Block until the queue is drained (or ``timeout`` elapses).

    Returns ``True`` if the queue drained, ``False`` on timeout.
    Intended for tests and graceful shutdown hooks.
    """

    deadline = threading.Event()

    def _wake() -> None:
        deadline.set()

    timer = threading.Timer(timeout, _wake)
    timer.daemon = True
    timer.start()
    try:
        while not _QUEUE.empty() and not deadline.is_set():
            deadline.wait(0.025)
        return _QUEUE.empty()
    finally:
        timer.cancel()


def dropped_total() -> int:
    """Return how many records were dropped due to queue back-pressure."""

    return _DROPPED_TOTAL


def _audit_dir() -> Path | None:
    path = os.environ.get(_ENV_DIR)
    if not path:
        return None
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _hash_of_body(payload: Mapping[str, Any]) -> str:
    try:
        body = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    except Exception:  # noqa: BLE001
        body = repr(payload)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _truncate_attributions(
    attributions: Iterable[Mapping[str, Any]] | None, top_k: int = 8
) -> list[dict[str, Any]]:
    if not attributions:
        return []
    records = []
    for item in attributions:
        if not isinstance(item, Mapping):
            continue
        records.append(
            {
                "response_token_index": item.get("response_token_index"),
                "evidence_token_index": item.get("evidence_token_index"),
                "evidence_doc_id": item.get("evidence_doc_id"),
                "score": item.get("score"),
            }
        )
        if len(records) >= top_k:
            break
    return records


def write_audit_record(
    *,
    request_id: str,
    tenant_id: str | None,
    payload: Mapping[str, Any],
    response: Mapping[str, Any],
    data_class: str = "standard",
) -> None:
    """Write a single audit-log record.  Safe to call on the hot path."""

    if data_class == "special":
        # Special-category data never lands in the audit log.
        return
    audit_dir = _audit_dir()
    if audit_dir is None:
        return
    trace = response.get("trace") if isinstance(response, Mapping) else None
    # The scorer ships two response shapes:
    #   * Nested: {"trace": {"band": "green", "scores": {"nli_aggregate": ...}}}
    #   * Compact (RunPod): {"band": "green", "nli_aggregate": ...}
    # We read from either without requiring callers to normalise.
    source: Mapping[str, Any] = (
        trace if isinstance(trace, Mapping) else response if isinstance(response, Mapping) else {}
    )
    scores = source.get("scores") if isinstance(source.get("scores"), Mapping) else None

    def _lookup(key: str) -> Any:
        if scores and key in scores:
            return scores[key]
        return source.get(key)

    band = _lookup("band")
    score = _lookup("score")
    teacher_channels = {
        "nli_aggregate": _lookup("nli_aggregate"),
        "context_coverage_ratio": _lookup("context_coverage_ratio"),
        "context_usage_ratio": _lookup("context_usage_ratio"),
        "groundedness_v2": _lookup("groundedness_v2"),
        "epistemic_hedge_gate": _lookup("epistemic_hedge_gate"),
    }
    attributions = source.get("top_attributions") or source.get("attributions")

    record: dict[str, Any] = {
        "schema_version": 1,
        "request_id": request_id,
        "tenant_id": tenant_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "profile": source.get("profile")
        or source.get("effective_profile")
        or (response.get("profile") if isinstance(response, Mapping) else None),
        "band": band,
        "score": score,
        "teacher_channels": teacher_channels,
        "top_k_attributions": _truncate_attributions(attributions, top_k=8),
        "hash_of_body": _hash_of_body(payload),
        "data_class": data_class,
    }
    try:
        line = json.dumps(record, ensure_ascii=False, default=str)
    except Exception as exc:  # noqa: BLE001 - never crash scoring on an audit-log failure
        logger.warning("audit_log.serialise_failed", extra={"error": str(exc)})
        return

    if _SYNC_WRITE:
        try:
            path = audit_dir / _DEFAULT_FILENAME
            with path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except Exception as exc:  # noqa: BLE001
            logger.warning("audit_log.write_failed", extra={"error": str(exc)})
        return

    _ensure_worker()
    try:
        _QUEUE.put_nowait(line)
    except queue.Full:
        global _DROPPED_TOTAL
        try:
            _QUEUE.get_nowait()
            _QUEUE.task_done()
        except queue.Empty:
            pass
        try:
            _QUEUE.put_nowait(line)
        except queue.Full:
            _DROPPED_TOTAL += 1
            return
        _DROPPED_TOTAL += 1


def export_audit_log(
    tenant_id: str,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
) -> Iterable[dict[str, Any]]:
    """Stream audit-log records for ``tenant_id`` inside the window.

    The customer-accessible export API (C4/B6) uses this generator.
    """

    audit_dir = _audit_dir()
    if audit_dir is None:
        return
    path = audit_dir / _DEFAULT_FILENAME
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("tenant_id") != tenant_id:
                continue
            ts_raw = record.get("timestamp")
            if not isinstance(ts_raw, str):
                continue
            try:
                ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
            except ValueError:
                continue
            if start and ts < start:
                continue
            if end and ts > end:
                continue
            yield record


__all__ = [
    "dropped_total",
    "export_audit_log",
    "flush",
    "write_audit_record",
]
