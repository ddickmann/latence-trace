"""Qualitative bare-user smoke test for TRACE autodiscovery.

The scenarios intentionally omit ``corpus_type`` and runtime feature maps.
They exercise the user experience where product callers send only text and
expect TRACE to route, synthesize eligible features, and emit an auditable
runtime decision.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts.bench_runpod_360 import Transport  # noqa: E402


@dataclass(frozen=True)
class Scenario:
    user: str
    class_key: str
    expected_actions: tuple[str, ...]
    payload: Dict[str, Any]


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        user="compliance analyst at a bank",
        class_key="rag.prose.enterprise",
        expected_actions=("auto_repair", "block"),
        payload={
            "scoring_mode": "rag",
            "query_text": "Can a wire transfer over $25,000 be approved by a single branch manager?",
            "raw_context": (
                "Operations manual section 8.4: Wire transfers over $25,000 require "
                "dual approval: one branch manager and one operations supervisor. "
                "Branch managers may singly approve wires up to $25,000 only."
            ),
            "response_text": (
                "A single branch manager can approve a $40,000 wire if the customer "
                "is known to the branch."
            ),
        },
    ),
    Scenario(
        user="customer-success rep answering a quick fact question",
        class_key="rag.prose.short_factoid",
        expected_actions=("allow",),
        payload={
            "scoring_mode": "rag",
            "query_text": "What is the battery warranty on the Orion X2 sensor?",
            "raw_context": (
                "Orion X2 field sensor datasheet: battery warranty: 18 months from "
                "ship date; enclosure warranty: 3 years; calibration interval: annual."
            ),
            "response_text": "The Orion X2 battery warranty is 18 months from the ship date.",
        },
    ),
    Scenario(
        user="healthcare operations manager summarizing several rules",
        class_key="rag.prose.multi_claim",
        expected_actions=("block", "auto_repair"),
        payload={
            "scoring_mode": "rag",
            "query_text": "Summarize the new flu-shot clinic rules for contractors.",
            "raw_context": (
                "Clinic memo: contractors may attend Tuesday or Thursday clinics. "
                "Bring a badge and signed consent form. The clinic is on floor 4. "
                "Walk-ins are not allowed after 3:30 PM."
            ),
            "response_text": (
                "Contractors can attend on Tuesday or Friday, should bring a badge, "
                "and may walk in any time until 5 PM on floor 4."
            ),
        },
    ),
    Scenario(
        user="finance analyst checking a table-to-text dashboard answer",
        class_key="rag.structured",
        expected_actions=("allow",),
        payload={
            "scoring_mode": "rag",
            "content_type": "application/json+schema",
            "structured_verification": "on",
            "query_text": "Which region had the highest Q4 renewal revenue?",
            "raw_context": (
                '[{"region":"Nordics","quarter":"Q4","renewal_revenue_usd":1240000},'
                '{"region":"DACH","quarter":"Q4","renewal_revenue_usd":980000},'
                '{"region":"Iberia","quarter":"Q4","renewal_revenue_usd":760000}]'
            ),
            "response_text": (
                "The Nordics region had the highest Q4 renewal revenue at $1,240,000."
            ),
        },
    ),
    Scenario(
        user="backend engineer asking about code in docs/context",
        class_key="rag.code_in_context",
        expected_actions=("block", "auto_repair"),
        payload={
            "scoring_mode": "rag",
            "query_text": "How do I retry a failed invoice sync with the SDK?",
            "raw_context": (
                "billing_sdk.py exposes retry_invoice_sync(invoice_id: str, *, "
                "idempotency_key: str). There is no resendInvoice or "
                "auto_retry_invoice function."
            ),
            "response_text": (
                "Use billing.resendInvoice(invoice_id) and the SDK will auto-retry it for you."
            ),
        },
    ),
    Scenario(
        user="AI coding agent editing a production service",
        class_key="code.agentic_trace",
        expected_actions=("allow",),
        payload={
            "scoring_mode": "code",
            "response_language_hint": "python",
            "query_text": (
                "Add a timeout to the existing invoice export client and keep current "
                "retry behavior."
            ),
            "raw_context": (
                "exporter/client.py defines class InvoiceExporter with method "
                "export_batch(timeout_s: float = 30.0). tests/test_exporter.py has "
                "test_export_batch_respects_timeout. Last command: pytest "
                "tests/test_exporter.py -q passed."
            ),
            "response_text": (
                "Updated InvoiceExporter.export_batch to pass timeout_s through to "
                "the HTTP client and kept the existing retry loop. Verified with "
                "pytest tests/test_exporter.py -q."
            ),
        },
    ),
)


async def run(endpoint_id: str, api_key: str, *, dump: Optional[Path] = None) -> int:
    transport = Transport(endpoint_id=endpoint_id, api_key=api_key)
    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=15.0)) as client:
        for scenario in SCENARIOS:
            body = await transport.submit(client, {"input": scenario.payload})
            output = body.get("output") or {}
            route = output.get("corpus_route") or {}
            decision = output.get("runtime_decision") or {}
            row = {
                "user": scenario.user,
                "expected_class": scenario.class_key,
                "expected_actions": list(scenario.expected_actions),
                "status": body.get("status"),
                "success": output.get("success"),
                "route": route.get("corpus_type"),
                "route_source": route.get("source"),
                "action": decision.get("action"),
                "head_id": decision.get("head_id"),
                "head_enabled": decision.get("head_enabled"),
                "runtime_feature_source": output.get("runtime_feature_source"),
                "missing_groups": output.get("runtime_feature_missing_groups") or [],
                "score": output.get("score"),
                "band": output.get("band"),
            }
            row["passed"] = (
                row["status"] == "COMPLETED"
                and row["success"] is True
                and row["route"] == scenario.class_key
                and row["action"] in scenario.expected_actions
                and bool(decision)
            )
            rows.append(row)
            if not row["passed"]:
                failures.append(row)
            print(
                f"{scenario.class_key:24s} route={row['route']} "
                f"action={row['action']} features={row['runtime_feature_source']} "
                f"head={row['head_id']} pass={row['passed']}"
            )
    report = {"passed": not failures, "rows": rows, "failures": failures}
    if dump:
        dump.parent.mkdir(parents=True, exist_ok=True)
        dump.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return 0 if not failures else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint-id", default=os.environ.get("RUNPOD_ENDPOINT_ID"))
    parser.add_argument("--api-key", default=os.environ.get("RUNPOD_API_KEY"))
    parser.add_argument("--dump", type=Path, default=None)
    args = parser.parse_args()
    if not args.endpoint_id or not args.api_key:
        raise SystemExit("set RUNPOD_ENDPOINT_ID/RUNPOD_API_KEY or pass --endpoint-id/--api-key")
    return asyncio.run(run(args.endpoint_id, args.api_key, dump=args.dump))


if __name__ == "__main__":
    raise SystemExit(main())
