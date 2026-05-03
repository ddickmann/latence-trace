#!/usr/bin/env python
"""Stateful TRACE session proof harness.

This harness is intentionally runnable in CI with synthetic fixtures while
keeping the command shape needed for the large held-out proof. Real dataset
adapters can feed the same event schema.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from latence_trace.memory.models import MemoryPolicy
from latence_trace.sessions.models import (
    TraceSessionCreateRequest,
    TraceSessionEvent,
    TraceSessionEventRequest,
)
from latence_trace.sessions.service import TraceSessionService


def _synthetic_events(domain: str, steps: int) -> list[TraceSessionEvent]:
    anchors = {
        "code": "src/payments/retry_policy.rs RetryPolicy::next_delay TEST_PAYMENT_42",
        "rag": "policy_clause_14_2 citation:kb/audit-log-export customer_plan=enterprise",
        "tool": "reservation_id=R-8821 latest_status=confirmed available_seats=2",
        "compliance": "patient_id=PX-1042 drug=atorvastatin contraindication=grapefruit",
    }
    anchor = anchors.get(domain, anchors["rag"])
    events = []
    for idx in range(steps):
        if idx % 7 == 0:
            content = f"Stable anchor for {domain}: {anchor}"
        elif idx % 5 == 0:
            content = f"Superseded noisy result {idx}: tmp_{idx}.log generated_{idx}.rs HAT{idx:03d}"
        else:
            content = f"Working step {idx} in {domain}; current task evidence remains {anchor}"
        events.append(
            TraceSessionEvent(
                event_type="tool_result" if domain == "tool" else "observation",
                content=content,
                raw_context=content,
                metadata={"domain": domain, "step": idx},
            )
        )
    return events


def run_proof(*, steps: int, output: Path, live_log: Path | None = None) -> dict[str, Any]:
    service = TraceSessionService(groundedness_service=object())
    domains = ["code", "rag", "tool", "compliance"]
    rows = []
    for domain in domains:
        session = service.create(
            TraceSessionCreateRequest(
                kind="code" if domain == "code" else "rag",
                memory_policy=MemoryPolicy(hot_token_budget=112, warm_token_budget=2048),
            )
        )
        full_tokens = 0
        for idx, event in enumerate(_synthetic_events(domain, steps), start=1):
            full_tokens += len((event.content or "").split())
            response = service.append_event(
                session.session.session_id,
                TraceSessionEventRequest(event=event, memory_domain=domain),
            )
            if live_log and idx % max(1, steps // 10) == 0:
                _append_jsonl(
                    live_log,
                    {
                        "event": "session_progress",
                        "domain": domain,
                        "step": idx,
                        "hot_tokens": response.memory_diagnostics.hot_tokens
                        if response.memory_diagnostics
                        else 0,
                    },
                )
        context = service.context(session.session.session_id)
        hot_tokens = context.hot_tokens
        reduction = 1.0 - (hot_tokens / max(1, full_tokens))
        hot = context.hot_context
        expected_anchor = _synthetic_events(domain, 1)[0].content.split(": ", 1)[-1]
        anchor_terms = expected_anchor.split()
        preservation = sum(1 for term in anchor_terms if term in hot) / max(1, len(anchor_terms))
        rows.append(
            {
                "domain": domain,
                "steps": steps,
                "full_tokens": full_tokens,
                "hot_tokens": hot_tokens,
                "live_context_reduction": round(reduction, 4),
                "anchor_preservation": round(preservation, 4),
            }
        )
    summary = {
        "proof": "stateful_trace_session_synthetic",
        "steps": steps,
        "domains": domains,
        "mean_live_context_reduction": round(mean(row["live_context_reduction"] for row in rows), 4),
        "mean_anchor_preservation": round(mean(row["anchor_preservation"] for row in rows), 4),
        "promotion_gate": {
            "target_live_context_reduction": 0.90,
            "target_anchor_preservation": 0.90,
            "passed": all(
                row["live_context_reduction"] >= 0.90 and row["anchor_preservation"] >= 0.90
                for row in rows
            ),
        },
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("research/trace_sessions/proof/report.json"))
    parser.add_argument("--live-log", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(run_proof(steps=args.steps, output=args.output, live_log=args.live_log), indent=2))


if __name__ == "__main__":
    main()
