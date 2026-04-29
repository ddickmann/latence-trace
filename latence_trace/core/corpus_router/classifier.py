"""Runtime singleton for the corpus-type classifier.

The classifier is a small scikit-learn pipeline persisted to
``latence_trace/data/corpus_classifier.joblib`` by
``scripts/train_corpus_classifier.py``. This module loads it once per
process (thread-safe lazy init) and exposes a single pure function
:func:`classify` that callers in the router middleware invoke.

Artefact shape (produced by the training script)::

    {
        "schema_version": int,
        "model_kind": "logistic_regression" | "gradient_boosting",
        "feature_names": [ ... ],
        "classes": [ ... ],
        "scaler_mean": [ ... ],
        "scaler_scale": [ ... ],
        "model": sklearn.BaseEstimator,
        "top1_accuracy_heldout": float,
        "trained_on_manifest_sha": str,
    }

Loading is defensive: a missing or invalid artefact makes
:func:`classify` return ``CorpusClassificationResult(None, ...)`` so the
middleware falls back to a safe default (Veracier enterprise bundle).
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from latence_trace.core.corpus_router import features as _features

logger = logging.getLogger(__name__)

_ARTEFACT_PATH = Path(__file__).resolve().parents[2] / "data/corpus_classifier.joblib"


@dataclass(frozen=True)
class ClassifierBundle:
    schema_version: int
    model_kind: str
    feature_names: Tuple[str, ...]
    classes: Tuple[str, ...]
    scaler_mean: Tuple[float, ...]
    scaler_scale: Tuple[float, ...]
    model: Any
    artefact_sha256: str
    top1_accuracy_heldout: Optional[float]


@dataclass(frozen=True)
class CorpusClassificationResult:
    """Output of :func:`classify`.

    ``corpus_type`` is ``None`` when the classifier is unavailable (e.g.
    artefact missing on disk). In that case, the middleware falls back to
    the ``rag.prose.enterprise`` default bundle.
    """

    corpus_type: Optional[str]
    confidence: Optional[float]
    latency_ms: float
    source: str  # "classifier" or "unavailable"
    artefact_sha256: Optional[str]


_LOCK = threading.Lock()
_BUNDLE: Optional[ClassifierBundle] = None
_LOAD_ERROR: Optional[str] = None


def _load_bundle(path: Path = _ARTEFACT_PATH) -> Optional[ClassifierBundle]:
    global _BUNDLE, _LOAD_ERROR
    if _BUNDLE is not None:
        return _BUNDLE
    with _LOCK:
        if _BUNDLE is not None:
            return _BUNDLE
        if not path.exists():
            _LOAD_ERROR = f"artefact missing: {path}"
            logger.warning("corpus_classifier: %s", _LOAD_ERROR)
            return None
        try:
            import joblib  # imported lazily so test-only imports don't pay the cost
            raw: Dict[str, Any] = joblib.load(path)
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            _BUNDLE = ClassifierBundle(
                schema_version=int(raw.get("schema_version", 0)),
                model_kind=str(raw["model_kind"]),
                feature_names=tuple(raw["feature_names"]),
                classes=tuple(raw["classes"]),
                scaler_mean=tuple(float(x) for x in raw["scaler_mean"]),
                scaler_scale=tuple(float(x) for x in raw["scaler_scale"]),
                model=raw["model"],
                artefact_sha256=sha,
                top1_accuracy_heldout=raw.get("top1_accuracy_heldout"),
            )
            # Schema sanity check: if the on-disk feature names diverge
            # from the runtime featurizer, refuse to load so we don't
            # silently map the wrong vector.
            if _BUNDLE.feature_names != _features.FEATURE_NAMES:
                _LOAD_ERROR = "feature schema mismatch between runtime and artefact"
                logger.error("corpus_classifier: %s", _LOAD_ERROR)
                _BUNDLE = None
                return None
            logger.info(
                "corpus_classifier loaded: kind=%s top1=%.4f sha=%s",
                _BUNDLE.model_kind,
                _BUNDLE.top1_accuracy_heldout or -1.0,
                sha[:12],
            )
            return _BUNDLE
        except Exception as exc:  # pragma: no cover - defensive I/O
            _LOAD_ERROR = f"failed to load: {exc!r}"
            logger.exception("corpus_classifier: failed to load artefact")
            return None


def _standardise(vec: List[float], mean: Tuple[float, ...], scale: Tuple[float, ...]) -> List[float]:
    return [
        (v - m) / s if s != 0 else 0.0
        for v, m, s in zip(vec, mean, scale)
    ]


def classify(
    *,
    query: str,
    response: str,
    raw_context: str,
) -> CorpusClassificationResult:
    """Infer the corpus type from a request payload.

    Pure function modulo the singleton classifier bundle. Deterministic:
    identical inputs always return the same output. Returns ``None`` for
    ``corpus_type`` if the artefact is not loaded — middleware should
    fall back to the enterprise default bundle in that case.
    """
    t0 = time.perf_counter()
    bundle = _load_bundle()
    vec = _features.featurize(query=query, response=response, raw_context=raw_context)
    if bundle is None:
        return CorpusClassificationResult(
            corpus_type=None,
            confidence=None,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            source="unavailable",
            artefact_sha256=None,
        )
    standardised = [
        _standardise(list(vec.values), bundle.scaler_mean, bundle.scaler_scale)
    ]
    # Both LogisticRegression and GradientBoostingClassifier expose the
    # same predict_proba / predict API, so we treat them uniformly.
    try:
        proba = bundle.model.predict_proba(standardised)[0]
    except Exception:  # pragma: no cover - estimator lacks predict_proba
        preds = bundle.model.predict(standardised)
        corpus_type = str(preds[0])
        return CorpusClassificationResult(
            corpus_type=corpus_type,
            confidence=None,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            source="classifier",
            artefact_sha256=bundle.artefact_sha256,
        )
    idx = int(max(range(len(proba)), key=lambda i: proba[i]))
    corpus_type = str(bundle.classes[idx])
    confidence = float(proba[idx])
    return CorpusClassificationResult(
        corpus_type=corpus_type,
        confidence=confidence,
        latency_ms=(time.perf_counter() - t0) * 1000.0,
        source="classifier",
        artefact_sha256=bundle.artefact_sha256,
    )


def reset_singleton_for_tests() -> None:
    """Hook used by unit tests to force the bundle to be re-loaded."""
    global _BUNDLE, _LOAD_ERROR
    with _LOCK:
        _BUNDLE = None
        _LOAD_ERROR = None


def get_load_error() -> Optional[str]:
    return _LOAD_ERROR
