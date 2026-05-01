from __future__ import annotations

from scripts.customer_breaker_smoke import CASES, SUITES, VERTICAL_PILOT_CASES, evaluate_output


def _case(case_id: str):
    return next(case for case in CASES if case.case_id == case_id)


def _body(*, action: str, band: str, class_key: str, extra_output: dict | None = None):
    output = {
        "success": True,
        "corpus_route": {"corpus_type": class_key},
        "runtime_decision": {
            "action": action,
            "band": band,
            "score": 0.9,
            "head_reason_codes": ["head_score_channel:test"],
            "evidence": [{"text": "source evidence"}],
        },
        "warnings": [],
    }
    if extra_output:
        output.update(extra_output)
    return {"status": "COMPLETED", "output": output}


def test_customer_breaker_gate_accepts_expected_supported_case() -> None:
    case = _case("fictional_medical_policy_supported")

    row = evaluate_output(
        case,
        _body(action="allow", band="green", class_key="rag.prose.enterprise"),
    )

    assert row["passed"] is True


def test_customer_breaker_gate_rejects_band_action_mismatch() -> None:
    case = _case("iot_json_wrong_pressure")

    row = evaluate_output(
        case,
        _body(action="auto_repair", band="green", class_key="rag.structured"),
    )

    assert row["passed"] is False
    assert row["checks"]["band_matches_action"] is False


def test_customer_breaker_gate_requires_diagnostic_value_for_repairs() -> None:
    case = _case("de_cold_chain_wrong_limit")
    body = _body(action="auto_repair", band="amber", class_key="rag.prose.enterprise")
    output = body["output"]
    output["runtime_decision"]["head_reason_codes"] = []
    output["runtime_decision"]["evidence"] = []

    row = evaluate_output(case, body)

    assert row["passed"] is False
    assert row["checks"]["value_signal_present"] is False


def test_customer_breaker_gate_rejects_missing_agentic_features() -> None:
    case = _case("agent_trace_supported_prod_shape")

    row = evaluate_output(
        case,
        _body(
            action="allow",
            band="green",
            class_key="code.agentic_trace",
            extra_output={
                "runtime_feature_source": "missing",
                "runtime_feature_missing_groups": ["trajectory_code_evidence_missing"],
            },
        ),
    )

    assert row["passed"] is False
    assert row["checks"]["features_not_missing"] is False


def test_vertical_pilot_suite_keeps_customer_breaker_default_separate() -> None:
    assert SUITES["customer_breaker"] == CASES
    assert SUITES["vertical_pilot"] == VERTICAL_PILOT_CASES
    assert len(SUITES["all"]) == len(CASES) + len(VERTICAL_PILOT_CASES)
    assert {case.case_id for case in VERTICAL_PILOT_CASES} >= {
        "pilot_legal_supported_only_clause",
        "pilot_finance_structured_amount_swap",
        "pilot_coding_performance_false_improvement",
    }
