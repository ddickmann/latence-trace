from __future__ import annotations

from types import SimpleNamespace

from latence_trace.core.runtime_feature_synthesizer import synthesize_runtime_features


def _response(class_key: str, *, score: float = 0.95, nli: float = 0.95):
    return SimpleNamespace(
        corpus_route=SimpleNamespace(corpus_type=class_key),
        scores=SimpleNamespace(
            primary_score=score,
            reverse_context=score,
            reverse_context_calibrated=score,
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
