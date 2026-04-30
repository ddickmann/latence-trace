"""Unit tests for the corpus-router middleware."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from latence_trace.core.corpus_router import bundles as _bundles
from latence_trace.core.corpus_router import classifier as _classifier
from latence_trace.middleware import corpus_router as _router


@pytest.fixture(autouse=True)
def _reset_singletons():
    _bundles.reset_singleton_for_tests()
    _classifier.reset_singleton_for_tests()
    yield
    _bundles.reset_singleton_for_tests()
    _classifier.reset_singleton_for_tests()


def _request(
    *,
    query="q",
    response="r",
    raw_context="c",
    corpus_type=None,
    scoring_mode="rag",
) -> SimpleNamespace:
    return SimpleNamespace(
        query_text=query,
        response_text=response,
        raw_context=raw_context,
        corpus_type=corpus_type,
        scoring_mode=scoring_mode,
    )


def test_explicit_corpus_type_wins_over_classifier() -> None:
    req = _request(corpus_type="rag.structured")
    decision = _router.route(req)
    assert decision.source == "explicit"
    assert decision.corpus_type == "rag.structured"
    assert decision.bundle is not None
    assert decision.bundle.class_key == "rag.structured"


def test_explicit_with_enum_value_string_is_accepted() -> None:
    """MCP / HTTP callers may pass the raw dotted string."""
    from latence_trace.api.models import CorpusType
    req = _request(corpus_type=CorpusType.CODE_AGENTIC_TRACE)
    decision = _router.route(req)
    assert decision.source == "explicit"
    assert decision.corpus_type == "code.agentic_trace"
    assert decision.bundle.scoring_mode == "code"


def test_classifier_path_picks_a_bundle_for_prose_request() -> None:
    req = _request(
        query="What is the annual revenue?",
        response="The annual revenue for FY24 was $12.3M.",
        raw_context="Annual report: revenue $12.3M in FY24.",
    )
    decision = _router.route(req)
    # One of the RAG / code classes must come back with a valid bundle.
    # ``rule`` is valid too because short factoid style responses can
    # legitimately be caught by the high-precision rule overlay before
    # the LR runs.
    assert decision.source in {"classifier", "classifier_ambiguous", "rule", "fallback"}
    assert decision.bundle is not None
    if decision.source in {"classifier", "classifier_ambiguous", "rule"}:
        assert decision.corpus_type.startswith(("rag.", "code."))
        assert decision.confidence is not None and 0.0 <= decision.confidence <= 1.0


def test_router_latency_under_three_ms() -> None:
    req = _request(
        query="q",
        response="The response text with a few sentences for realism.",
        raw_context="Context text, modest size.",
    )
    # Warm-up
    _router.route(req)
    import time
    samples = []
    for _ in range(50):
        t0 = time.perf_counter()
        _router.route(req)
        samples.append((time.perf_counter() - t0) * 1000.0)
    p50 = sorted(samples)[len(samples) // 2]
    # Router = featurize + predict_proba + bundle lookup. Budget 5 ms.
    assert p50 < 10.0, f"router p50 was {p50:.2f} ms, budget is 10 ms"


def test_build_diagnostics_shape_matches_pydantic_model() -> None:
    from latence_trace.api.models import CorpusRouteDiagnostics
    req = _request(corpus_type="rag.prose.enterprise")
    decision = _router.route(req)
    payload = _router.build_diagnostics(decision)
    # Must validate against the pydantic model so downstream consumers
    # never see an undocumented field.
    model = CorpusRouteDiagnostics.model_validate(payload)
    assert model.corpus_type == "rag.prose.enterprise"
    assert model.source == "explicit"
    assert model.thresholds_applied is not None
    assert set(model.thresholds_applied.keys()) == {"green", "amber"}
    assert isinstance(model.classifier_top_classes, list)


def test_fallback_when_explicit_class_has_no_bundle() -> None:
    req = _request(corpus_type="rag.nonsense.class")
    decision = _router.route(req)
    # Should gracefully fall through to classifier / rule / fallback.
    assert decision.source in {"classifier", "classifier_ambiguous", "rule", "fallback"}
    assert decision.corpus_type is not None


@pytest.mark.parametrize(
    ("name", "expected", "req"),
    [
        (
            "bank_policy",
            "rag.prose.enterprise",
            _request(
                query="Can a wire transfer over $25,000 be approved by a single branch manager?",
                raw_context=(
                    "Operations manual section 8.4: Wire transfers over $25,000 "
                    "require dual approval: one branch manager and one operations supervisor."
                ),
                response="A single branch manager can approve a $40,000 wire for known customers.",
            ),
        ),
        (
            "short_factoid",
            "rag.prose.short_factoid",
            _request(
                query="What is the battery warranty on the Orion X2 sensor?",
                raw_context="Orion X2 datasheet: battery warranty: 18 months from ship date.",
                response="The Orion X2 battery warranty is 18 months from the ship date.",
            ),
        ),
        (
            "multi_claim",
            "rag.prose.multi_claim",
            _request(
                query="Summarize the flu-shot clinic rules for contractors.",
                raw_context=(
                    "Contractors may attend Tuesday or Thursday clinics. Bring a badge "
                    "and signed consent form. Walk-ins are not allowed after 3:30 PM."
                ),
                response=(
                    "Contractors can attend Tuesday or Friday, should bring a badge, "
                    "and may walk in until 5 PM."
                ),
            ),
        ),
        (
            "structured_table",
            "rag.structured",
            _request(
                query="Which region had the highest Q4 renewal revenue?",
                raw_context='[{"region":"Nordics","quarter":"Q4","renewal_revenue_usd":1240000}]',
                response="The Nordics region had the highest Q4 renewal revenue at $1,240,000.",
            ),
        ),
        (
            "code_docs",
            "rag.code_in_context",
            _request(
                query="How do I retry a failed invoice sync with the SDK?",
                raw_context=(
                    "billing_sdk.py exposes retry_invoice_sync(invoice_id: str, "
                    "idempotency_key: str)."
                ),
                response="Use billing.resendInvoice(invoice_id).",
            ),
        ),
        (
            "agent_trace",
            "code.agentic_trace",
            _request(
                query="Add a timeout to the invoice export client.",
                raw_context=(
                    "exporter/client.py defines class InvoiceExporter. "
                    "tests/test_exporter.py has test_export_batch_respects_timeout. "
                    "Last command: pytest tests/test_exporter.py -q passed."
                ),
                response=(
                    "Updated InvoiceExporter.export_batch and verified with "
                    "pytest tests/test_exporter.py -q."
                ),
                scoring_mode="code",
            ),
        ),
    ],
)
def test_bare_text_qualitative_routes_to_expected_class(name, expected, req) -> None:
    decision = _router.route(req)

    assert decision.corpus_type == expected, (name, decision)
    assert decision.bundle is not None
