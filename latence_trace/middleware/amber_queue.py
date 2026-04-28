"""Amber reviewer queue API (Plan C4 + v2-prep).

The amber band is for answers where TRACE is not confident enough to
send to the user but not red enough to fail closed.  Reviewers (SMEs,
risk / compliance owners) need to see those answers, accept / edit /
reject them, and have the decision logged alongside the original
scoring response.

This module provides:

* :func:`list_amber_records` - iterate audit-log records whose band
  is ``amber`` for a given tenant, within a time window.
* :func:`write_reviewer_decision` - append a decision record to a
  sibling ``reviewer_decisions.jsonl`` file.  The decision carries
  the reviewer identity, the accept/edit/reject action, the
  optional corrected answer, and the original ``request_id`` so it
  joins back to the score record in downstream analytics.
* :func:`list_reviewer_decisions` - read the decisions.

Reviewer decisions feed directly into the v2 student distillation
pipeline: accept == positive label, reject == negative label, edit
== soft label plus the corrected answer is used as an additional
faithful example.

The portal (Next.js) uses the REST endpoints mounted by
:func:`latence_trace.api.routes.create_router`; self-hosted operators
can also use them from the CLI.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from latence_trace.middleware.audit_log import export_audit_log

logger = logging.getLogger(__name__)


_DECISIONS_FILENAME = "reviewer_decisions.jsonl"
_ENV_DIR = "LATENCE_TRACE_AUDIT_LOG_DIR"

ReviewerAction = Literal["accept", "edit", "reject"]


def _audit_dir() -> Path | None:
    raw = os.environ.get(_ENV_DIR)
    if not raw:
        return None
    return Path(raw).expanduser()


def list_amber_records(
    tenant_id: str,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
    limit: int | None = 200,
) -> list[dict[str, Any]]:
    """Return up to ``limit`` amber-band audit records for a tenant.

    Records are returned newest-first.  ``limit=None`` returns the
    entire amber set (used by CSV export).
    """

    records = [
        r
        for r in export_audit_log(tenant_id, start=start, end=end)
        if str(r.get("band", "")).lower() == "amber"
    ]
    records.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
    if limit is not None:
        return records[:limit]
    return records


def write_reviewer_decision(
    *,
    request_id: str,
    tenant_id: str,
    reviewer_id: str,
    action: ReviewerAction,
    comment: str | None = None,
    corrected_response: str | None = None,
) -> dict[str, Any]:
    """Append a reviewer decision record; returns the record."""

    audit_dir = _audit_dir()
    if audit_dir is None:
        raise RuntimeError(
            "reviewer decisions require the audit log dir. "
            "Set LATENCE_TRACE_AUDIT_LOG_DIR."
        )
    audit_dir.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "request_id": request_id,
        "tenant_id": tenant_id,
        "reviewer_id": reviewer_id,
        "action": action,
    }
    if comment:
        record["comment"] = comment
    if corrected_response:
        record["corrected_response"] = corrected_response
    line = json.dumps(record, separators=(",", ":"), ensure_ascii=False)
    path = audit_dir / _DECISIONS_FILENAME
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    return record


def list_reviewer_decisions(
    tenant_id: str,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
) -> Iterable[dict[str, Any]]:
    audit_dir = _audit_dir()
    if audit_dir is None:
        return
    path = audit_dir / _DECISIONS_FILENAME
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
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
    "list_amber_records",
    "list_reviewer_decisions",
    "write_reviewer_decision",
]
