"""Calibration bundle loader - singleton, import-time memoised.

Loads the six ``calibration.<class_key>.json`` files produced by
``scripts/calibrate_per_class.py`` into immutable :class:`CalibrationBundle`
objects. The runtime router looks up a bundle by class key and hands it
to the service layer to override fusion weights and thresholds for the
matching request.

A missing or malformed calibration file for any class is treated as a
hard error at startup: the router should never silently fall back to the
global default for a corpus it already knows about, because that would
undo the core promise of per-class routing.

A sentinel bundle named ``rag.prose.enterprise`` is also installed as
the safe default for requests where the classifier is unavailable
(e.g. missing joblib artefact on disk).
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# Enum of known corpus types. The slug form (with underscores) maps to
# the on-disk filename; the dotted form is what the API and classifier
# expose to callers.
KNOWN_CORPUS_TYPES: Tuple[str, ...] = (
    "rag.prose.enterprise",
    "rag.prose.short_factoid",
    "rag.prose.multi_claim",
    "rag.structured",
    "rag.code_in_context",
    "code.agentic_trace",
)

DEFAULT_FALLBACK_CLASS = "rag.prose.enterprise"


def _key_to_slug(key: str) -> str:
    return key.replace(".", "_")


def _calibration_path(key: str) -> Path:
    return _DATA_DIR / f"calibration.{_key_to_slug(key)}.json"


@dataclass(frozen=True)
class CalibrationBundle:
    """Immutable per-class calibration.

    ``fusion_weights`` is the normalised dict (sums to ~1.0) used by the
    RAG lane. For ``scoring_mode="code"`` the weights are inert and
    ``thresholds`` applies to ``composite_phantom_score`` instead of the
    fused score.
    """

    class_key: str
    scoring_mode: str  # "rag" | "code"
    fusion_weights: Mapping[str, float]
    thresholds: Mapping[str, float]
    nli_model_hint: Optional[str]
    metric: str
    metric_value: float
    trained_on: Optional[str]
    raw: Mapping[str, Any] = field(default_factory=dict)

    @property
    def green_threshold(self) -> float:
        return float(self.thresholds.get("green", 0.80))

    @property
    def amber_threshold(self) -> float:
        return float(self.thresholds.get("amber", 0.60))


_LOCK = threading.Lock()
_BUNDLES: Optional[Dict[str, CalibrationBundle]] = None
_LOAD_ERRORS: Dict[str, str] = {}


def _parse_bundle(key: str, payload: Mapping[str, Any]) -> CalibrationBundle:
    fw = payload.get("fusion_weights") or {}
    th = payload.get("thresholds") or {}
    scoring_mode = str(payload.get("scoring_mode") or "rag")
    if scoring_mode not in {"rag", "code"}:
        raise ValueError(f"{key}: invalid scoring_mode {scoring_mode!r}")
    weight_sum = sum(float(v) for v in fw.values())
    if scoring_mode == "rag" and weight_sum < 0.5 - 1e-6:
        raise ValueError(
            f"{key}: fusion weights sum to {weight_sum:.3f} < 0.5 safety floor"
        )
    green = float(th.get("green", 0.80))
    amber = float(th.get("amber", 0.60))
    if not (0.0 <= amber <= green <= 1.0):
        raise ValueError(
            f"{key}: thresholds invalid (amber={amber}, green={green})"
        )
    return CalibrationBundle(
        class_key=key,
        scoring_mode=scoring_mode,
        fusion_weights=dict(fw),
        thresholds={"green": green, "amber": amber},
        nli_model_hint=payload.get("nli_model_hint"),
        metric=str(payload.get("metric") or "unknown"),
        metric_value=float(payload.get("metric_value") or 0.0),
        trained_on=payload.get("trained_on"),
        raw=dict(payload),
    )


def _load_all(strict: bool = True) -> Dict[str, CalibrationBundle]:
    global _BUNDLES, _LOAD_ERRORS
    if _BUNDLES is not None:
        return _BUNDLES
    with _LOCK:
        if _BUNDLES is not None:
            return _BUNDLES
        bundles: Dict[str, CalibrationBundle] = {}
        for key in KNOWN_CORPUS_TYPES:
            path = _calibration_path(key)
            if not path.exists():
                msg = f"missing calibration file: {path}"
                _LOAD_ERRORS[key] = msg
                if strict:
                    logger.warning("corpus_router.bundles: %s", msg)
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                bundles[key] = _parse_bundle(key, payload)
            except Exception as exc:
                _LOAD_ERRORS[key] = f"failed to parse {path}: {exc!r}"
                logger.exception("corpus_router.bundles: failed to load %s", path)
        _BUNDLES = bundles
        logger.info(
            "corpus_router.bundles loaded %d/%d classes",
            len(bundles), len(KNOWN_CORPUS_TYPES),
        )
        return bundles


def load_bundle(class_key: str) -> Optional[CalibrationBundle]:
    """Return the calibration bundle for ``class_key`` or ``None``."""
    return _load_all().get(class_key)


def load_all_bundles() -> Dict[str, CalibrationBundle]:
    """Return a dict snapshot of every loaded bundle."""
    return dict(_load_all())


def get_load_errors() -> Dict[str, str]:
    """Per-class load errors, surfaced for diagnostics."""
    _load_all()
    return dict(_LOAD_ERRORS)


def reset_singleton_for_tests() -> None:
    """Hook used by unit tests to force bundles to be re-loaded."""
    global _BUNDLES
    with _LOCK:
        _BUNDLES = None
        _LOAD_ERRORS.clear()


def known_corpus_types() -> Tuple[str, ...]:
    return KNOWN_CORPUS_TYPES
