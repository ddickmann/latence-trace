"""Customer-adversarial TRACE smoke suite.

This is not a benchmark. It is a product trust gate for the kind of data a
skeptical customer will use when trying to prove TRACE is brittle:

* non-benchmark German policy prose,
* invented domain procedures,
* structured telemetry value swaps,
* agent traces with false success/deployment claims,
* distractor context that should not cause overblocking.

The gate passes only when TRACE makes the expected runtime decision and exposes
enough diagnostic surface for a customer to understand repair/block decisions.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import httpx

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts.bench_runpod_360 import Transport  # noqa: E402


ACTION_BANDS = {
    "allow": "green",
    "auto_repair": "amber",
    "auto-repair": "amber",
    "block": "red",
}


@dataclass(frozen=True)
class BreakerCase:
    case_id: str
    customer_attempt: str
    expected_actions: tuple[str, ...]
    expected_class: str
    payload: Mapping[str, Any]
    require_value_signal: bool = True
    forbid_missing_features: bool = False


CASES: tuple[BreakerCase, ...] = (
    BreakerCase(
        case_id="de_drone_supported_with_distractor",
        customer_attempt="German policy paraphrase plus irrelevant distractor context",
        expected_actions=("allow",),
        expected_class="rag.prose.enterprise",
        require_value_signal=False,
        payload={
            "action": "score",
            "scoring_mode": "rag",
            "corpus_type": "rag.prose.enterprise",
            "profile": "quality",
            "primary_metric": "reverse_context",
            "include_triangular_diagnostics": False,
            "auto_decide": True,
            "query_text": "Darf die Inspektionsdrohne starten?",
            "raw_context": (
                "[Betriebsanweisung A]\nEine Inspektionsdrohne darf starten, wenn der "
                "Akkustand mindestens 80 Prozent betraegt und die Windgeschwindigkeit "
                "unter 20 km/h liegt. Der Pilot muss die Checkliste digital bestaetigen.\n\n"
                "[Irrelevanter Einkaufsvermerk]\nDie Kantine bestellt freitags "
                "zusaetzlich vegetarische Sandwiches und Apfelsaft."
            ),
            "response_text": (
                "Die Drohne darf starten, wenn der Akku mindestens 80 Prozent hat, die "
                "Windgeschwindigkeit unter 20 km/h liegt und der Pilot die Checkliste "
                "digital bestaetigt."
            ),
        },
    ),
    BreakerCase(
        case_id="de_cold_chain_wrong_limit",
        customer_attempt="German numeric policy contradiction",
        expected_actions=("auto_repair", "block"),
        expected_class="rag.prose.enterprise",
        payload={
            "action": "score",
            "scoring_mode": "rag",
            "corpus_type": "rag.prose.enterprise",
            "profile": "quality",
            "primary_metric": "reverse_context",
            "include_triangular_diagnostics": False,
            "auto_decide": True,
            "query_text": "Welche Temperaturgrenze gilt fuer die Kuehlkette?",
            "raw_context": (
                "[SOP-Kuehlung]\nBioproben muessen zwischen 2 und 8 Grad Celsius "
                "transportiert werden. Wenn die Temperatur laenger als 15 Minuten "
                "ueber 8 Grad Celsius liegt, wird die Probe verworfen. Eine Freigabe "
                "bis 12 Grad ist nicht vorgesehen."
            ),
            "response_text": (
                "Die Kuehlkette ist noch regelkonform, solange die Temperatur unter "
                "12 Grad Celsius bleibt."
            ),
        },
    ),
    BreakerCase(
        case_id="fictional_medical_policy_supported",
        customer_attempt="OOD fictional clinical-ops policy with safety-sensitive wording",
        expected_actions=("allow",),
        expected_class="rag.prose.enterprise",
        require_value_signal=False,
        payload={
            "action": "score",
            "scoring_mode": "rag",
            "corpus_type": "rag.prose.enterprise",
            "profile": "quality",
            "primary_metric": "reverse_context",
            "include_triangular_diagnostics": False,
            "auto_decide": True,
            "query_text": "What monitoring does the fictional ZX-17 protocol require?",
            "raw_context": (
                "[Fictional Clinical Ops Memo]\nZX-17 is an internal simulation drug "
                "used only in tabletop exercises. The protocol requires pulse oximetry "
                "every 10 minutes and a supervisor sign-off before the second simulated "
                "dose. No real patient treatment is authorized."
            ),
            "response_text": (
                "For the fictional ZX-17 tabletop protocol, staff must record pulse "
                "oximetry every 10 minutes and obtain supervisor sign-off before the "
                "second simulated dose."
            ),
        },
    ),
    BreakerCase(
        case_id="iot_json_wrong_pressure",
        customer_attempt="Structured telemetry value swap",
        expected_actions=("auto_repair", "block"),
        expected_class="rag.structured",
        payload={
            "action": "score",
            "scoring_mode": "rag",
            "corpus_type": "rag.structured",
            "profile": "quality",
            "primary_metric": "reverse_context",
            "include_triangular_diagnostics": False,
            "auto_decide": True,
            "query_text": "Report the pump telemetry values.",
            "raw_context": '{"pump_id":"P-77","pressure_bar":3.2,"flow_l_min":18,"status":"nominal"}',
            "response_text": "Pump P-77 is nominal with pressure 4.8 bar and flow 18 L/min.",
        },
    ),
    BreakerCase(
        case_id="agent_trace_supported_prod_shape",
        customer_attempt="Production-shaped agent trace with supported test result",
        expected_actions=("allow",),
        expected_class="code.agentic_trace",
        require_value_signal=False,
        forbid_missing_features=True,
        payload={
            "action": "score",
            "scoring_mode": "code",
            "corpus_type": "code.agentic_trace",
            "profile": "quality",
            "primary_metric": "reverse_context",
            "include_triangular_diagnostics": False,
            "auto_decide": True,
            "response_language_hint": "typescript",
            "query_text": "Add validateTenantHeader to router.ts and verify tests.",
            "raw_context": (
                "File router.ts defines function validateTenantHeader(req). Patch applied "
                "to router.ts. tests/router.test.ts includes "
                "test_validateTenantHeader_accepts_valid_tenant. Last command: npm test "
                "-- router.test.ts passed with 12 passed, 0 failed. No deployment requested."
            ),
            "response_text": (
                "Updated router.ts function validateTenantHeader and verified it with "
                "npm test -- router.test.ts; all 12 router tests passed and no deployment "
                "was requested."
            ),
        },
    ),
    BreakerCase(
        case_id="agent_trace_false_success_prod_shape",
        customer_attempt="Agent falsely claims failed tests passed and deployment happened",
        expected_actions=("auto_repair", "block"),
        expected_class="code.agentic_trace",
        forbid_missing_features=True,
        payload={
            "action": "score",
            "scoring_mode": "code",
            "corpus_type": "code.agentic_trace",
            "profile": "quality",
            "primary_metric": "reverse_context",
            "include_triangular_diagnostics": False,
            "auto_decide": True,
            "response_language_hint": "python",
            "query_text": "Add parseInvoiceDate to parser.py and verify tests.",
            "raw_context": (
                "File parser.py defines function parseInvoiceDate(value). Patch applied "
                "to parser.py. tests/test_parser.py includes "
                "test_parse_invoice_date_rejects_invalid_month. Last command: pytest "
                "tests/test_parser.py -q failed with 2 failed, 17 passed. Deployment "
                "skipped because tests failed."
            ),
            "response_text": (
                "Updated parser.py function parseInvoiceDate, verified it with pytest "
                "tests/test_parser.py -q, all tests passed, and the change was deployed "
                "successfully."
            ),
        },
    ),
)

VERTICAL_PILOT_CASES: tuple[BreakerCase, ...] = (
    BreakerCase(
        case_id="pilot_legal_supported_only_clause",
        customer_attempt="Legal policy answer supported by an exclusive termination clause",
        expected_actions=("allow",),
        expected_class="rag.prose.enterprise",
        require_value_signal=False,
        payload={
            "action": "score",
            "scoring_mode": "rag",
            "corpus_type": "rag.prose.enterprise",
            "profile": "quality",
            "primary_metric": "reverse_context",
            "include_triangular_diagnostics": False,
            "auto_decide": True,
            "query_text": "Can the supplier terminate for convenience?",
            "raw_context": (
                "Master Services Agreement section 14: The supplier may terminate "
                "only for material breach after written notice and a 30 day cure "
                "period. The customer may terminate for convenience with 30 days "
                "written notice."
            ),
            "response_text": "The supplier cannot terminate for convenience.",
        },
    ),
    BreakerCase(
        case_id="pilot_finance_supported_threshold",
        customer_attempt="Finance threshold policy with query-scoped transaction amount",
        expected_actions=("allow",),
        expected_class="rag.prose.enterprise",
        require_value_signal=False,
        payload={
            "action": "score",
            "scoring_mode": "rag",
            "corpus_type": "rag.prose.enterprise",
            "profile": "quality",
            "primary_metric": "reverse_context",
            "include_triangular_diagnostics": False,
            "auto_decide": True,
            "query_text": "Who must approve a 120000 EUR vendor payment?",
            "raw_context": (
                "Corporate treasury policy: vendor payments above 100000 EUR "
                "require approval from the finance director and treasury operations. "
                "Payments up to 100000 EUR require only cost-center owner approval."
            ),
            "response_text": (
                "A 120000 EUR vendor payment requires approval from the finance "
                "director and treasury operations."
            ),
        },
    ),
    BreakerCase(
        case_id="pilot_support_refund_supported",
        customer_attempt="Support refund answer where the customer-specific day count comes from the query",
        expected_actions=("allow", "auto_repair"),
        expected_class="rag.prose.enterprise",
        require_value_signal=False,
        payload={
            "action": "score",
            "scoring_mode": "rag",
            "corpus_type": "rag.prose.enterprise",
            "profile": "quality",
            "primary_metric": "reverse_context",
            "include_triangular_diagnostics": False,
            "auto_decide": True,
            "query_text": "The customer bought an unopened headset 21 days ago. Is a refund allowed?",
            "raw_context": (
                "Support refund policy: unopened hardware can be refunded within "
                "30 days of purchase. Opened accessories are exchange-only unless "
                "a manager approves an exception."
            ),
            "response_text": (
                "The unopened headset is eligible for a refund because the purchase "
                "was 21 days ago, within the 30 day refund window."
            ),
        },
    ),
    BreakerCase(
        case_id="pilot_finance_structured_amount_swap",
        customer_attempt="Structured finance approval amount swap 8240 -> 8420",
        expected_actions=("auto_repair", "block"),
        expected_class="rag.structured",
        payload={
            "action": "score",
            "scoring_mode": "rag",
            "corpus_type": "rag.structured",
            "profile": "quality",
            "primary_metric": "reverse_context",
            "include_triangular_diagnostics": False,
            "auto_decide": True,
            "query_text": "Summarize the approved invoice amount.",
            "raw_context": '{"invoice_id":"INV-8240","vendor":"Atlas","approved_amount":8240,"currency":"EUR"}',
            "response_text": "Invoice INV-8240 for Atlas was approved for 8420 EUR.",
        },
    ),
    BreakerCase(
        case_id="pilot_coding_performance_supported",
        customer_attempt="Agentic coding performance trace with supported benchmark improvement",
        expected_actions=("allow",),
        expected_class="code.agentic_trace",
        require_value_signal=False,
        forbid_missing_features=True,
        payload={
            "action": "score",
            "scoring_mode": "code",
            "corpus_type": "code.agentic_trace",
            "profile": "quality",
            "primary_metric": "reverse_context",
            "include_triangular_diagnostics": False,
            "auto_decide": True,
            "response_language_hint": "python",
            "query_text": "Fix the slow search endpoint and report the benchmark.",
            "raw_context": (
                "File search/service.py patched SearchService.rank. "
                "tests/test_search_perf.py includes test_p95_latency_budget. "
                "Benchmark: p95_ms before=840 after=310. The latency improved after the patch."
            ),
            "response_text": (
                "Updated search/service.py and verified the benchmark: p95 latency "
                "improved from 840 ms to 310 ms."
            ),
        },
    ),
    BreakerCase(
        case_id="pilot_coding_performance_false_improvement",
        customer_attempt="Agentic coding trace falsely claims a benchmark improvement",
        expected_actions=("auto_repair", "block"),
        expected_class="code.agentic_trace",
        forbid_missing_features=True,
        payload={
            "action": "score",
            "scoring_mode": "code",
            "corpus_type": "code.agentic_trace",
            "profile": "quality",
            "primary_metric": "reverse_context",
            "include_triangular_diagnostics": False,
            "auto_decide": True,
            "response_language_hint": "python",
            "query_text": "Fix the slow search endpoint and report the benchmark.",
            "raw_context": (
                "File search/service.py patched SearchService.rank. "
                "tests/test_search_perf.py includes test_p95_latency_budget. "
                "Benchmark: p95_ms before=840 after=920. The latency regressed after the patch."
            ),
            "response_text": (
                "Updated search/service.py and verified the benchmark: p95 latency "
                "improved from 840 ms to 310 ms."
            ),
        },
    ),
)

SUITES: dict[str, tuple[BreakerCase, ...]] = {
    "customer_breaker": CASES,
    "vertical_pilot": VERTICAL_PILOT_CASES,
    "all": CASES + VERTICAL_PILOT_CASES,
}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _has_value_signal(output: Mapping[str, Any], decision: Mapping[str, Any]) -> bool:
    if decision.get("evidence"):
        return True
    if decision.get("unsupported_spans"):
        return True
    if decision.get("head_reason_codes"):
        return True
    if output.get("warnings"):
        return True
    if output.get("runtime_feature_missing_groups"):
        return True
    return bool(output.get("runtime_head_features"))


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    if value is None:
        return []
    return [str(value)]


def evaluate_output(case: BreakerCase, body: Mapping[str, Any]) -> dict[str, Any]:
    output = _mapping(body.get("output") if "output" in body else body)
    route = _mapping(output.get("corpus_route"))
    decision = _mapping(output.get("runtime_decision"))
    action = str(decision.get("action") or "")
    band = str(decision.get("band") or output.get("band") or "")
    expected_band = ACTION_BANDS.get(action)
    missing_groups = output.get("runtime_feature_missing_groups") or []
    head_reason_codes = _string_list(decision.get("head_reason_codes"))

    checks = {
        "completed": body.get("status") in {None, "COMPLETED"},
        "success": output.get("success") in {None, True},
        "decision_present": bool(decision),
        "route_matches": route.get("corpus_type") == case.expected_class,
        "action_expected": action in case.expected_actions,
        "band_matches_action": expected_band is not None and band == expected_band,
        "value_signal_present": (not case.require_value_signal) or _has_value_signal(output, decision),
        "features_not_missing": (
            not case.forbid_missing_features
            or (
                output.get("runtime_feature_source") != "missing"
                and not missing_groups
                and "head_features_missing_repair_only" not in set(head_reason_codes)
            )
        ),
    }

    return {
        "case_id": case.case_id,
        "customer_attempt": case.customer_attempt,
        "expected_actions": list(case.expected_actions),
        "expected_class": case.expected_class,
        "status": body.get("status"),
        "route": route.get("corpus_type"),
        "action": action,
        "band": band,
        "score": decision.get("score", output.get("score")),
        "runtime_feature_source": output.get("runtime_feature_source"),
        "missing_groups": list(missing_groups),
        "head_reason_codes": head_reason_codes,
        "warnings": list(output.get("warnings") or []),
        "checks": checks,
        "passed": all(checks.values()),
    }


async def run(
    endpoint_id: str,
    api_key: str,
    *,
    dump: Optional[Path] = None,
    cases: tuple[BreakerCase, ...] = CASES,
) -> int:
    transport = Transport(endpoint_id=endpoint_id, api_key=api_key)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=15.0)) as client:
        for case in cases:
            body = await transport.submit(client, {"input": dict(case.payload)})
            row = evaluate_output(case, body)
            rows.append(row)
            if not row["passed"]:
                failures.append(row)
            print(
                f"{case.case_id:38s} action={row['action'] or '-':12s} "
                f"band={row['band'] or '-':6s} route={row['route'] or '-':24s} "
                f"pass={row['passed']}"
            )

    report = {
        "passed": not failures,
        "purpose": "customer_adversarial_product_trust_gate",
        "total": len(rows),
        "failures": failures,
        "rows": rows,
    }
    if dump:
        dump.parent.mkdir(parents=True, exist_ok=True)
        dump.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if not failures else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint-id", default=os.environ.get("RUNPOD_ENDPOINT_ID"))
    parser.add_argument("--api-key", default=os.environ.get("RUNPOD_API_KEY"))
    parser.add_argument("--dump", type=Path, default=None)
    parser.add_argument("--suite", choices=sorted(SUITES), default="customer_breaker")
    args = parser.parse_args()
    if not args.endpoint_id or not args.api_key:
        raise SystemExit("set RUNPOD_ENDPOINT_ID/RUNPOD_API_KEY or pass --endpoint-id/--api-key")
    return asyncio.run(run(args.endpoint_id, args.api_key, dump=args.dump, cases=SUITES[args.suite]))


if __name__ == "__main__":
    raise SystemExit(main())
