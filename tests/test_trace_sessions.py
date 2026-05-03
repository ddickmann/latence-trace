from __future__ import annotations

from pathlib import Path

from latence_trace.memory.models import MemoryPolicy
from latence_trace.sessions.models import (
    TraceSessionCreateRequest,
    TraceSessionEvent,
    TraceSessionEventRequest,
)
from latence_trace.sessions.service import TraceSessionService
from scripts.run_stateful_trace_session_proof import run_proof


def test_trace_session_event_updates_stateful_memory() -> None:
    service = TraceSessionService(groundedness_service=object())
    created = service.create(
        TraceSessionCreateRequest(
            kind="code",
            memory_policy=MemoryPolicy(hot_token_budget=80, warm_token_budget=240),
        )
    )

    response = service.append_event(
        created.session.session_id,
        TraceSessionEventRequest(
            memory_domain="code",
            event=TraceSessionEvent(
                event_type="file_read",
                content="Opened src/cache.py and found CacheClient.get_many.",
                raw_context="src/cache.py CacheClient.get_many must remain stable.",
                idempotency_key="evt-1",
            ),
        ),
    )

    assert response.session.event_count == 1
    assert response.session.memory_state is not None
    assert "src/cache.py" in response.hot_context

    duplicate = service.append_event(
        created.session.session_id,
        TraceSessionEventRequest(
            memory_domain="code",
            event=TraceSessionEvent(
                event_type="file_read",
                content="Duplicate",
                idempotency_key="evt-1",
            ),
        ),
    )
    assert duplicate.session.event_count == 1


def test_stateful_session_proof_harness_writes_report(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    summary = run_proof(steps=30, output=report)

    assert report.exists()
    assert summary["rows"]
    assert summary["mean_anchor_preservation"] >= 0.75
