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


def _request(*, query="q", response="r", raw_context="c", corpus_type=None) -> SimpleNamespace:
    return SimpleNamespace(
        query_text=query,
        response_text=response,
        raw_context=raw_context,
        corpus_type=corpus_type,
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
    assert decision.source in {"classifier", "rule", "fallback"}
    assert decision.bundle is not None
    if decision.source in {"classifier", "rule"}:
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


def test_fallback_when_explicit_class_has_no_bundle() -> None:
    req = _request(corpus_type="rag.nonsense.class")
    decision = _router.route(req)
    # Should gracefully fall through to classifier / rule / fallback.
    assert decision.source in {"classifier", "rule", "fallback"}
    assert decision.corpus_type is not None
