"""Routing middleware for the corpus-type classifier.

Given a :class:`GroundednessRequest`, decides which of the six corpus
classes the request should be scored under, loads the matching
:class:`~latence_trace.core.corpus_router.bundles.CalibrationBundle`
and returns a :class:`CorpusRouteDecision`. The scoring service calls
:func:`route` at the top of :meth:`score` before dispatching to the
RAG / code lane; if a bundle is produced, the service layers the
bundle's ``fusion_weights`` + ``thresholds`` on top of the runtime
profile.

Design goals:

* Pure function — no I/O after module import.
* Deterministic — identical requests always return identical decisions.
* <= 3 ms p50 on CPU - featurizer is fixed-width, classifier is LR over
  20 features.
* Respects ``request.corpus_type`` as a tenant-declared override.
* Safe fallback when the classifier artefact is missing: returns a
  decision with ``source="fallback"`` pointing at the enterprise bundle.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

from latence_trace.core.corpus_router import bundles as _bundles
from latence_trace.core.corpus_router import classifier as _classifier
from latence_trace.core.corpus_router import rules as _rules
from latence_trace.core.language_detector import resolve_language as _resolve_language

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CorpusRouteDecision:
    """Outcome of :func:`route`.

    ``bundle`` is ``None`` only when the requested corpus type has no
    on-disk calibration file AND the fallback also fails (misconfigured
    deployment). Callers should treat a ``None`` bundle as "skip the
    router, scoring proceeds with the default runtime profile".

    ``language`` records the language used to look up the calibration
    bundle (``en`` for the historical baseline, ``de`` for German, etc.).
    ``language_source`` distinguishes ``"request"`` (caller specified
    explicitly), ``"auto"`` (langdetect resolved), or ``"fallback_en"``
    (no usable probe text -- defaults to English).
    """

    corpus_type: Optional[str]
    source: str  # "classifier" | "classifier_ambiguous" | "explicit" | "fallback" | "disabled" | "rule"
    confidence: Optional[float]
    classifier_latency_ms: float
    artefact_sha256: Optional[str]
    bundle: Optional[_bundles.CalibrationBundle]
    rule_reason: Optional[str] = None
    classifier_top_classes: tuple[tuple[str, float], ...] = ()
    classifier_probabilities: Optional[dict[str, float]] = None
    language: str = "en"
    language_source: str = "fallback_en"


def _request_field(request: Any, name: str, default: Any = None) -> Any:
    """Duck-typed field access so tests can pass plain dataclasses.

    ``GroundednessRequest`` is a pydantic model; older callers may pass a
    ``SimpleNamespace`` or bare dict-like object.
    """
    if isinstance(request, dict):
        return request.get(name, default)
    return getattr(request, name, default)


def _request_text(request: Any, field: str) -> str:
    value = _request_field(request, field)
    if value is None:
        return ""
    return str(value)


def _resolve_request_language(request: Any) -> tuple[str, str]:
    """Resolve ``(language, language_source)`` from a routing request.

    Mirrors ``service.py::_apply_language_defaults`` so the router and
    the scoring layer see the *same* language for the same request --
    they're independent code paths but they MUST agree, otherwise the
    bundle thresholds (router) and the NLI defaults (scoring) drift apart
    on the same scoring call.
    """

    explicit = _request_field(request, "language")
    explicit_value = explicit if explicit in {"de", "en"} else None
    return _resolve_language(
        explicit=explicit_value,
        response_text=_request_text(request, "response_text") or None,
        query_text=(
            _request_text(request, "query_text") or _request_text(request, "query") or None
        ),
        raw_context=_request_text(request, "raw_context") or None,
    )


def route(request: Any) -> CorpusRouteDecision:
    """Infer (or accept) the corpus class for a scoring request.

    Explicit override wins. Otherwise the lazy-loaded classifier singleton
    predicts the class from featurised request payload. If the classifier
    artefact is missing, falls back to the enterprise bundle.
    """
    # Resolve the effective language up front so every load_bundle call
    # below picks the matching per-language artefact (or logs a
    # bundle_language_fallback warning if it has to fall back).
    language, language_source = _resolve_request_language(request)

    # Explicit override path — accepts the CorpusType enum value or a
    # bare string so MCP / HTTP callers without the enum are honoured.
    explicit = _request_field(request, "corpus_type")
    if explicit is not None:
        class_key = getattr(explicit, "value", explicit)
        bundle = _bundles.load_bundle(str(class_key), language)
        if bundle is not None:
            return CorpusRouteDecision(
                corpus_type=str(class_key),
                source="explicit",
                confidence=None,
                classifier_latency_ms=0.0,
                artefact_sha256=None,
                bundle=bundle,
                language=language,
                language_source=language_source,
            )
        logger.warning(
            "corpus_router: explicit corpus_type=%s has no bundle; falling back",
            class_key,
        )

    t0 = time.perf_counter()
    query = _request_text(request, "query_text") or _request_text(request, "query")
    response = _request_text(request, "response_text")
    raw_context = _request_text(request, "raw_context")

    # Rule overlay runs first - high-precision structural rules bypass
    # the learned classifier when the evidence is unambiguous. This
    # dramatically improves out-of-distribution behaviour because the
    # rules are dataset-agnostic (JSON root, markdown table, multi-file
    # code bundle) whereas the LR classifier can overfit to training-
    # time surface features.
    rule_decision = _rules.apply_rules(
        query=query, response=response, raw_context=raw_context
    )
    if (
        rule_decision.corpus_type is not None
        and rule_decision.confidence >= _rules.MIN_RULE_CONFIDENCE
    ):
        bundle = _bundles.load_bundle(rule_decision.corpus_type, language)
        if bundle is not None:
            total_ms = (time.perf_counter() - t0) * 1000.0
            return CorpusRouteDecision(
                corpus_type=rule_decision.corpus_type,
                source="rule",
                confidence=rule_decision.confidence,
                classifier_latency_ms=total_ms,
                artefact_sha256=None,
                bundle=bundle,
                rule_reason=rule_decision.reason,
                language=language,
                language_source=language_source,
            )

    clf_result = _classifier.classify(
        query=query,
        response=response,
        raw_context=raw_context,
    )
    if clf_result.corpus_type is not None:
        selected_class = clf_result.corpus_type
        source = "classifier"
        if _is_ambiguous_prose_result(clf_result):
            selected_class = _bundles.DEFAULT_FALLBACK_CLASS
            source = "classifier_ambiguous"
        bundle = _bundles.load_bundle(selected_class, language)
        if bundle is not None:
            return CorpusRouteDecision(
                corpus_type=selected_class,
                source=source,
                confidence=clf_result.confidence,
                classifier_latency_ms=clf_result.latency_ms,
                artefact_sha256=clf_result.artefact_sha256,
                bundle=bundle,
                classifier_top_classes=clf_result.top_classes,
                classifier_probabilities=clf_result.probabilities,
                language=language,
                language_source=language_source,
            )
        logger.warning(
            "corpus_router: classifier chose %s but no bundle is loaded; falling back",
            clf_result.corpus_type,
        )

    # Fallback path — always return the enterprise bundle when available
    # so scoring proceeds with a sane, production-tested configuration.
    fallback = _bundles.load_bundle(_bundles.DEFAULT_FALLBACK_CLASS, language)
    total_ms = (time.perf_counter() - t0) * 1000.0
    return CorpusRouteDecision(
        corpus_type=_bundles.DEFAULT_FALLBACK_CLASS if fallback else None,
        source="fallback",
        confidence=None,
        classifier_latency_ms=total_ms,
        artefact_sha256=None,
        bundle=fallback,
        language=language,
        language_source=language_source,
    )


def build_diagnostics(decision: CorpusRouteDecision) -> dict:
    """Shape the decision as a dict matching
    :class:`latence_trace.api.models.CorpusRouteDiagnostics`.

    Kept as a helper so service integration tests can assert on the exact
    dict shape without importing pydantic.
    """
    bundle = decision.bundle
    return {
        "corpus_type": decision.corpus_type or _bundles.DEFAULT_FALLBACK_CLASS,
        "source": decision.source,
        "confidence": decision.confidence,
        "classifier_latency_ms": round(decision.classifier_latency_ms, 3),
        "artefact_sha256": decision.artefact_sha256,
        "rule_reason": decision.rule_reason,
        "classifier_top_classes": [
            {"corpus_type": class_key, "probability": probability}
            for class_key, probability in decision.classifier_top_classes
        ],
        "classifier_probabilities": dict(decision.classifier_probabilities or {}),
        "fusion_weights_applied": dict(bundle.fusion_weights) if bundle else None,
        "thresholds_applied": dict(bundle.thresholds) if bundle else None,
        "scoring_mode_applied": bundle.scoring_mode if bundle else None,
        "bundle_metric": bundle.metric if bundle else None,
        "bundle_metric_value": bundle.metric_value if bundle else None,
    }


def _is_ambiguous_prose_result(result: _classifier.CorpusClassificationResult) -> bool:
    """Conservative guard for adjacent prose classes.

    Structural rules catch the obvious OOD shapes. When the learned model
    remains uncertain between prose bundles, enterprise is the safest default
    because it uses the broadest calibrated policy and does not require a
    feature-gated runtime head.
    """

    if not result.top_classes or not result.corpus_type:
        return False
    prose = {
        "rag.prose.enterprise",
        "rag.prose.short_factoid",
        "rag.prose.multi_claim",
    }
    if result.corpus_type not in prose:
        return False
    top = result.top_classes[0][1]
    if len(result.top_classes) < 2:
        return top < 0.72
    second_class, second = result.top_classes[1]
    return top < 0.72 or (second_class in prose and (top - second) < 0.18)
