from __future__ import annotations

from types import SimpleNamespace

from latence_trace.core import runtime_decision
from latence_trace.core.runtime_feature_synthesizer import synthesize_runtime_features


def _response(class_key: str, *, score: float = 0.95, nli: float = 0.95, reverse: float | None = None):
    reverse = score if reverse is None else reverse
    return SimpleNamespace(
        corpus_route=SimpleNamespace(corpus_type=class_key),
        scores=SimpleNamespace(
            primary_score=score,
            reverse_context=reverse,
            reverse_context_calibrated=reverse,
            literal_guarded=score,
            nli_aggregate=nli,
            groundedness_v2=score,
            context_coverage_ratio=1.0,
            context_unused_ratio=0.0,
            context_uncertain_ratio=0.0,
            dead_weight_ratio=0.0,
        ),
    )


def _request(query: str, context: str, response: str):
    return SimpleNamespace(query_text=query, raw_context=context, response_text=response)


def test_synthesizes_structured_features_from_json_value_alignment() -> None:
    result = synthesize_runtime_features(
        _request(
            "Which region had the highest Q4 renewal revenue?",
            '[{"region":"Nordics","quarter":"Q4","renewal_revenue_usd":1240000}]',
            "The Nordics region had the highest Q4 renewal revenue at $1,240,000.",
        ),
        _response("rag.structured", score=0.5, nli=0.5),
    )

    assert result.source == "synthesized"
    assert result.features is not None
    assert result.features["numeric_coverage"] == 1.0
    assert result.features["cell_provenance_match"] > 0.0


def test_structured_synthesis_refuses_without_structured_source() -> None:
    result = synthesize_runtime_features(
        _request("q", "plain prose context", "plain prose response"),
        _response("rag.structured"),
    )

    assert result.features is None
    assert result.source == "missing"
    assert "structured_source_missing" in result.missing_groups


def test_synthesizes_agentic_trace_only_with_command_evidence() -> None:
    result = synthesize_runtime_features(
        _request(
            "Add a timeout to the invoice export client.",
            (
                "exporter/client.py defines class InvoiceExporter. "
                "tests/test_exporter.py has test_export_batch_respects_timeout. "
                "Last command: pytest tests/test_exporter.py -q passed."
            ),
            "Updated InvoiceExporter.export_batch and verified with pytest tests/test_exporter.py -q.",
        ),
        _response("code.agentic_trace", score=0.95, nli=0.95),
    )

    assert result.source == "synthesized"
    assert result.features is not None
    assert result.features["test_outcome_alignment"] == 1.0
    assert result.features["missing_command_evidence"] == 0.0


def test_agentic_trace_synthesis_stays_partial_without_command_evidence() -> None:
    result = synthesize_runtime_features(
        _request(
            "Add a timeout.",
            "exporter/client.py defines class InvoiceExporter.",
            "Updated InvoiceExporter.export_batch.",
        ),
        _response("code.agentic_trace", score=0.95, nli=0.95),
    )

    assert result.features is None
    assert result.source in {"missing", "partial"}


def test_agentic_trace_synthesis_uses_reverse_evidence_when_outcomes_align(monkeypatch) -> None:
    monkeypatch.setenv("LATENCE_TRACE_RUNTIME_DECISION_ENABLED", "1")
    runtime_decision.reset_policy_cache_for_tests()
    result = synthesize_runtime_features(
        _request(
            "Add validateTenantHeader to router.ts and verify tests.",
            (
                "File router.ts defines function validateTenantHeader(req). Patch applied to router.ts. "
                "tests/router.test.ts includes test_validateTenantHeader_accepts_valid_tenant. "
                "Last command: npm test -- router.test.ts passed with 12 passed, 0 failed. "
                "No deployment requested."
            ),
            (
                "Updated router.ts function validateTenantHeader and verified it with npm test -- "
                "router.test.ts; all 12 router tests passed and no deployment was requested."
            ),
        ),
        _response("code.agentic_trace", score=0.68, reverse=0.95),
    )
    assert result.source == "synthesized"
    assert result.features is not None
    assert result.features["test_outcome_alignment"] == 1.0
    assert result.features["groundedness_v2"] == result.features["reverse_context"]

    response = _response("code.agentic_trace", score=0.68, reverse=0.95)
    response.runtime_head_features = result.features
    decision = runtime_decision.build_runtime_decision(response)

    assert decision is not None
    assert decision["action"] == "allow"


def test_agentic_trace_synthesis_penalizes_test_and_deploy_contradictions(monkeypatch) -> None:
    monkeypatch.setenv("LATENCE_TRACE_RUNTIME_DECISION_ENABLED", "1")
    runtime_decision.reset_policy_cache_for_tests()
    result = synthesize_runtime_features(
        _request(
            "Add parseInvoiceDate to parser.py and verify tests.",
            (
                "File parser.py defines function parseInvoiceDate(value). Patch applied to parser.py. "
                "tests/test_parser.py includes test_parse_invoice_date_rejects_invalid_month. "
                "Last command: pytest tests/test_parser.py -q failed with 2 failed, 17 passed. "
                "Deployment skipped because tests failed."
            ),
            (
                "Updated parser.py function parseInvoiceDate, verified it with pytest "
                "tests/test_parser.py -q, all tests passed, and the change was deployed successfully."
            ),
        ),
        _response("code.agentic_trace", score=0.67, reverse=0.94),
    )
    assert result.source == "synthesized"
    assert result.features is not None
    assert result.features["test_outcome_alignment"] == 0.0
    assert result.features["temporal_order_alignment"] == 0.0

    response = _response("code.agentic_trace", score=0.67, reverse=0.94)
    response.runtime_head_features = result.features
    decision = runtime_decision.build_runtime_decision(response)

    assert decision is not None
    assert decision["action"] == "block"
