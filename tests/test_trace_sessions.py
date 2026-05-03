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


def test_trace_session_masks_pii_and_skips_noisy_exact_indexes() -> None:
    service = TraceSessionService(groundedness_service=object())
    created = service.create(
        TraceSessionCreateRequest(
            kind="rag",
            memory_policy=MemoryPolicy(hot_token_budget=96, warm_token_budget=256),
        )
    )

    service.append_event(
        created.session.session_id,
        TraceSessionEventRequest(
            memory_domain="grounding",
            event=TraceSessionEvent(
                content=(
                    "Customer asks if simvastatin can be taken with clarithromycin. "
                    "Critical: contraindicated combination; risk of rhabdomyolysis."
                ),
                raw_context=(
                    "PII-like noisy chat: Jane Roe phone 555-0100 asks for exact dose. "
                    "Medication fact: clarithromycin increases simvastatin exposure."
                ),
            ),
        ),
    )

    hot = service.context(created.session.session_id).hot_context
    assert "simvastatin" in hot
    assert "clarithromycin" in hot
    assert "rhabdomyolysis" in hot
    assert "555-0100" not in hot
    assert "[phone]" in hot


def test_trace_session_supersedes_old_tool_status() -> None:
    service = TraceSessionService(groundedness_service=object())
    created = service.create(
        TraceSessionCreateRequest(
            kind="rag",
            memory_policy=MemoryPolicy(hot_token_budget=72, warm_token_budget=256),
        )
    )

    for event in [
        "Support ticket T-771 initial tool result: refund_status=pending replacement_order=none.",
        "New tool result supersedes prior state: refund_status=approved replacement_order=R-90017 ship_by=2026-05-06.",
    ]:
        service.append_event(
            created.session.session_id,
            TraceSessionEventRequest(
                memory_domain="tool",
                event=TraceSessionEvent(content=event, raw_context=event),
            ),
        )

    hot = service.context(created.session.session_id).hot_context
    assert "refund_status=approved" in hot
    assert "R-90017" in hot
    assert "refund_status=pending" not in hot
