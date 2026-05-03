from __future__ import annotations

from pathlib import Path

from latence_trace.memory.models import MemoryPolicy
from latence_trace.sessions.models import (
    TraceSessionCreateRequest,
    TraceSessionEvent,
    TraceSessionEventRequest,
    TraceSessionRepairRequest,
    TraceSessionScoreRequest,
)
from latence_trace.sessions.service import TraceSessionService
from scripts.run_history_vault_repair_proof import run_proof as run_repair_proof
from scripts.run_stateful_trace_session_proof import run_proof


class _DummyTraceResponse:
    next_session_state = None
    next_memory_state = None
    memory_diagnostics = None

    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def model_dump(self, *args, **kwargs) -> dict:
        return dict(self._payload)


class _RiskyGroundednessService:
    def groundedness(self, _request) -> _DummyTraceResponse:
        return _DummyTraceResponse(
            {
                "risk_band": "red",
                "runtime_decision": {"action": "auto_repair"},
                "scores": {"groundedness_v2": 0.21},
            }
        )


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


def test_history_vault_repair_proof_harness_passes(tmp_path: Path) -> None:
    summary = run_repair_proof(output_dir=tmp_path, step_counts=(20, 40), max_tokens=160)

    assert (tmp_path / "live_metrics.jsonl").exists()
    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "scorecard.md").exists()
    assert summary["promotion_gate"]["passed"]
    assert summary["promotion_gate"]["repair_recall"] >= 0.95
    assert summary["promotion_gate"]["no_raw_pii_leakage"]


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


def test_source_vault_attaches_redacted_source_pointers() -> None:
    service = TraceSessionService(groundedness_service=object())
    created = service.create(
        TraceSessionCreateRequest(
            kind="rag",
            memory_policy=MemoryPolicy(hot_token_budget=96, warm_token_budget=256),
        )
    )

    response = service.append_event(
        created.session.session_id,
        TraceSessionEventRequest(
            memory_domain="rag",
            event=TraceSessionEvent(
                content="Legal matter LGL-042 final clause requires 14-day cure period.",
                raw_context="Counsel phone 555-0199 confirmed LGL-042 is the governing source.",
            ),
        ),
    )

    assert response.source_pointer is not None
    memory_state = response.session.memory_state
    assert memory_state is not None
    assert any(
        span.provenance.get("source_pointer", {}).get("source_id")
        == response.source_pointer.source_id
        for span in memory_state.spans
    )
    redacted = service.source(created.session.session_id, response.source_pointer.source_id)
    assert redacted.source is not None
    assert "555-0199" not in (redacted.source.raw_context or "")
    assert "[phone]" in (redacted.source.raw_context or "")


def test_manual_repair_packet_uses_original_vault_without_pii_leakage() -> None:
    service = TraceSessionService(groundedness_service=object())
    created = service.create(TraceSessionCreateRequest(kind="rag"))
    service.append_event(
        created.session.session_id,
        TraceSessionEventRequest(
            memory_domain="rag",
            event=TraceSessionEvent(
                content="Finance close source FIN-778 says accrual owner is Elena.",
                raw_context="Private contact elena@example.com confirms FIN-778 accrual is $44,200.",
            ),
        ),
    )

    packet = service.repair(
        created.session.session_id,
        TraceSessionRepairRequest(missing_terms=["FIN-778", "$44,200"], max_tokens=80),
    ).repair_packet

    assert packet.triggered
    assert packet.repaired_terms
    assert packet.token_count <= 80
    assert "FIN-778" in " ".join(packet.repaired_terms)
    joined = " ".join(excerpt.text for excerpt in packet.excerpts)
    assert "$44,200" in joined
    assert "elena@example.com" not in joined
    assert "[email]" in joined


def test_score_repair_gate_returns_packet_for_risky_lost_anchor() -> None:
    service = TraceSessionService(groundedness_service=_RiskyGroundednessService())
    created = service.create(
        TraceSessionCreateRequest(
            kind="code",
            memory_policy=MemoryPolicy(hot_token_budget=32, warm_token_budget=96),
        )
    )
    service.append_event(
        created.session.session_id,
        TraceSessionEventRequest(
            memory_domain="code",
            event=TraceSessionEvent(
                content="Read src/lib.rs and found deploy_guard::verify_release for bug RUST-991.",
                raw_context="src/lib.rs deploy_guard::verify_release failed before patch RUST-991.",
            ),
        ),
    )

    response = service.score(
        created.session.session_id,
        TraceSessionScoreRequest(
            lane="code",
            trace_request={
                "query_text": "Did RUST-991 update src/lib.rs?",
                "response_text": "RUST-991 is fixed but evidence is missing.",
                "raw_context": "unrelated sparse trace",
            },
            force_original_on_trigger=True,
        ),
    )

    assert response.repair_packet is not None
    assert response.repair_packet.suggested_action in {
        "re_score",
        "ask_tool_again",
        "block_until_regrounded",
    }
    repair_text = " ".join(excerpt.text for excerpt in response.repair_packet.excerpts)
    assert "RUST-991" in repair_text
