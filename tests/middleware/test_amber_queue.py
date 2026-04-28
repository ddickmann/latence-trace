"""Unit tests for the amber reviewer queue (C4 + v2 prep)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from latence_trace.middleware import amber_queue as aq
from latence_trace.middleware import audit_log as al


@pytest.fixture
def audit_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("LATENCE_TRACE_AUDIT_LOG_DIR", str(tmp_path))
    monkeypatch.setenv("LATENCE_TRACE_AUDIT_LOG_SYNC", "1")
    yield tmp_path


def _write_audit_row(tmp: Path, **kwargs: object) -> None:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "request_id": kwargs.get("request_id", "req-1"),
        "tenant_id": kwargs.get("tenant_id", "acme"),
        "band": kwargs.get("band", "amber"),
        "profile": "standard",
        "score": 0.72,
    }
    with (tmp / "audit.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def test_list_amber_records_filters_by_tenant_and_band(audit_dir: Path) -> None:
    _write_audit_row(audit_dir, request_id="r1", tenant_id="acme", band="amber")
    _write_audit_row(audit_dir, request_id="r2", tenant_id="acme", band="green")
    _write_audit_row(audit_dir, request_id="r3", tenant_id="other", band="amber")

    results = aq.list_amber_records("acme")
    assert len(results) == 1
    assert results[0]["request_id"] == "r1"


def test_write_and_list_reviewer_decision(audit_dir: Path) -> None:
    aq.write_reviewer_decision(
        request_id="req-1",
        tenant_id="acme",
        reviewer_id="alice@acme.example",
        action="accept",
        comment="verified against Q4 shareholder letter",
    )
    aq.write_reviewer_decision(
        request_id="req-2",
        tenant_id="acme",
        reviewer_id="alice@acme.example",
        action="edit",
        corrected_response="ARR ended 2023 at 12.4M USD.",
    )
    aq.write_reviewer_decision(
        request_id="req-3",
        tenant_id="other",
        reviewer_id="bob@other.example",
        action="reject",
    )

    results = list(aq.list_reviewer_decisions("acme"))
    assert len(results) == 2
    actions = {r["action"] for r in results}
    assert actions == {"accept", "edit"}
    edit = next(r for r in results if r["action"] == "edit")
    assert edit["corrected_response"].startswith("ARR ended 2023")


def test_reviewer_decision_requires_audit_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("LATENCE_TRACE_AUDIT_LOG_DIR", raising=False)
    with pytest.raises(RuntimeError):
        aq.write_reviewer_decision(
            request_id="r", tenant_id="t", reviewer_id="u", action="accept"
        )
