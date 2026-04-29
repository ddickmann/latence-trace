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

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CorpusRouteDecision:
    """Outcome of :func:`route`.

    ``bundle`` is ``None`` only when the requested corpus type has no
    on-disk calibration file AND the fallback also fails (misconfigured
    deployment). Callers should treat a ``None`` bundle as "skip the
    router, scoring proceeds with the default runtime profile".
    """

    corpus_type: Optional[str]
    source: str  # "classifier" | "explicit" | "fallback" | "disabled" | "rule"
    confidence: Optional[float]
    classifier_latency_ms: float
    artefact_sha256: Optional[str]
    bundle: Optional[_bundles.CalibrationBundle]
    rule_reason: Optional[str] = None


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


def route(request: Any) -> CorpusRouteDecision:
    """Infer (or accept) the corpus class for a scoring request.

    Explicit override wins. Otherwise the lazy-loaded classifier singleton
    predicts the class from featurised request payload. If the classifier
    artefact is missing, falls back to the enterprise bundle.
    """
    # Explicit override path — accepts the CorpusType enum value or a
    # bare string so MCP / HTTP callers without the enum are honoured.
    explicit = _request_field(request, "corpus_type")
    if explicit is not None:
        class_key = getattr(explicit, "value", explicit)
        bundle = _bundles.load_bundle(str(class_key))
        if bundle is not None:
            return CorpusRouteDecision(
                corpus_type=str(class_key),
                source="explicit",
                confidence=None,
                classifier_latency_ms=0.0,
                artefact_sha256=None,
                bundle=bundle,
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
        bundle = _bundles.load_bundle(rule_decision.corpus_type)
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
            )

    clf_result = _classifier.classify(
        query=query,
        response=response,
        raw_context=raw_context,
    )
    if clf_result.corpus_type is not None:
        bundle = _bundles.load_bundle(clf_result.corpus_type)
        if bundle is not None:
            return CorpusRouteDecision(
                corpus_type=clf_result.corpus_type,
                source="classifier",
                confidence=clf_result.confidence,
                classifier_latency_ms=clf_result.latency_ms,
                artefact_sha256=clf_result.artefact_sha256,
                bundle=bundle,
            )
        logger.warning(
            "corpus_router: classifier chose %s but no bundle is loaded; falling back",
            clf_result.corpus_type,
        )

    # Fallback path — always return the enterprise bundle when available
    # so scoring proceeds with a sane, production-tested configuration.
    fallback = _bundles.load_bundle(_bundles.DEFAULT_FALLBACK_CLASS)
    total_ms = (time.perf_counter() - t0) * 1000.0
    return CorpusRouteDecision(
        corpus_type=_bundles.DEFAULT_FALLBACK_CLASS if fallback else None,
        source="fallback",
        confidence=None,
        classifier_latency_ms=total_ms,
        artefact_sha256=None,
        bundle=fallback,
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
        "fusion_weights_applied": dict(bundle.fusion_weights) if bundle else None,
        "thresholds_applied": dict(bundle.thresholds) if bundle else None,
        "scoring_mode_applied": bundle.scoring_mode if bundle else None,
        "bundle_metric": bundle.metric if bundle else None,
        "bundle_metric_value": bundle.metric_value if bundle else None,
    }
