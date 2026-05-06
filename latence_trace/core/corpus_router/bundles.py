"""Calibration bundle loader - singleton, import-time memoised.

Loads the per-class ``calibration.<class_key>[.<language>].json`` files
produced by ``scripts/calibrate_per_class.py`` into immutable
:class:`CalibrationBundle` objects. The runtime router looks up a bundle
by ``(class_key, language)`` and hands it to the service layer to
override fusion weights and thresholds for the matching request.

Filename convention:

* ``calibration.<class>.json`` -- the historical English bundle. Treated
  as the ``en`` bundle for that class.
* ``calibration.<class>.<lang>.json`` -- a per-language bundle. Currently
  only ``de`` ships post Phase C; additional languages can drop in here
  with no code change.

Resolution order for ``load_bundle(class_key, language)``:

1. ``calibration.<class>.<language>.json`` (per-language artefact).
2. ``calibration.<class>.json`` (English fallback). When this fires for
   a non-English language we log a one-shot ``bundle_language_fallback``
   warning so the operator can audit silent fallbacks in observability.
3. ``None`` -- caller must handle missing bundle (the router falls back
   to the enterprise default).

A missing or malformed calibration file for any class is treated as a
hard error at startup *for the English bundle*: the router should never
silently fall back to the global default for a corpus it already knows
about, because that would undo the core promise of per-class routing.
Per-language overrides are *optional* by design: a German request for a
class whose German bundle has not shipped falls back to the English
bundle and emits the warning above.

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
from typing import Any, Dict, Mapping, Optional, Tuple

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

# Languages with potential per-language calibration. ``en`` is the
# universal baseline (the historical bundles), ``de`` is the German
# release shipped in Phase C. Adding a new language here is enough to
# turn on per-language lookup -- no other code change required.
SUPPORTED_LANGUAGES: Tuple[str, ...] = ("en", "de")
DEFAULT_LANGUAGE: str = "en"


def _key_to_slug(key: str) -> str:
    return key.replace(".", "_")


def _calibration_path(key: str, language: str = DEFAULT_LANGUAGE) -> Path:
    """Resolve the on-disk path for a (class_key, language) pair.

    For the default language we keep the historical filename
    ``calibration.<class>.json`` so the existing English bundles are
    found without renaming. For any other language the filename has the
    language suffix: ``calibration.<class>.<lang>.json``.
    """

    slug = _key_to_slug(key)
    if language == DEFAULT_LANGUAGE:
        return _DATA_DIR / f"calibration.{slug}.json"
    return _DATA_DIR / f"calibration.{slug}.{language}.json"


@dataclass(frozen=True)
class CalibrationBundle:
    """Immutable per-class calibration.

    ``fusion_weights`` is the normalised dict (sums to ~1.0) used by the
    RAG lane. For ``scoring_mode="code"`` the weights are inert and
    ``thresholds`` applies to ``composite_phantom_score`` instead of the
    fused score.

    ``language`` records which language artefact this bundle was loaded
    from (``en`` for the historical baseline, ``de`` for the German
    release, etc.). Surfaced in profile_diagnostics so observability can
    detect cases where a request expected a per-language bundle but
    actually got the English fallback.
    """

    class_key: str
    scoring_mode: str  # "rag" | "code"
    fusion_weights: Mapping[str, float]
    thresholds: Mapping[str, float]
    nli_model_hint: Optional[str]
    metric: str
    metric_value: float
    trained_on: Optional[str]
    language: str = DEFAULT_LANGUAGE
    raw: Mapping[str, Any] = field(default_factory=dict)

    @property
    def green_threshold(self) -> float:
        return float(self.thresholds.get("green", 0.80))

    @property
    def amber_threshold(self) -> float:
        return float(self.thresholds.get("amber", 0.60))


_LOCK = threading.Lock()
_BUNDLES: Dict[Tuple[str, str], Optional[CalibrationBundle]] = {}
_LOAD_ERRORS: Dict[Tuple[str, str], str] = {}
_FALLBACK_WARNED: set[Tuple[str, str]] = set()


def _parse_bundle(
    key: str,
    payload: Mapping[str, Any],
    language: str = DEFAULT_LANGUAGE,
) -> CalibrationBundle:
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
        language=language,
        raw=dict(payload),
    )


def _load_one_locked(
    class_key: str, language: str
) -> Optional[CalibrationBundle]:
    """Load a single bundle from disk. Caller must hold ``_LOCK``."""

    cache_key = (class_key, language)
    if cache_key in _BUNDLES:
        return _BUNDLES[cache_key]

    path = _calibration_path(class_key, language)
    if not path.exists():
        if language != DEFAULT_LANGUAGE:
            # Try the English fallback. We deliberately do NOT cache this
            # under (class, non_default_lang) so a later language-bundle
            # drop-in (Phase C deliverable) is picked up without a
            # process restart.
            fallback = _load_one_locked(class_key, DEFAULT_LANGUAGE)
            if fallback is not None:
                if cache_key not in _FALLBACK_WARNED:
                    _FALLBACK_WARNED.add(cache_key)
                    logger.warning(
                        "bundle_language_fallback",
                        extra={
                            "class_key": class_key,
                            "requested_language": language,
                            "loaded_language": DEFAULT_LANGUAGE,
                            "missing_path": str(path),
                        },
                    )
                # Re-tag the bundle as having loaded under the requested
                # language so the diagnostics field reads honestly. We
                # return a copy with language replaced so the caller can
                # display "language: de (uncalibrated)" while the actual
                # thresholds come from the English bundle.
                rebadged = CalibrationBundle(
                    class_key=fallback.class_key,
                    scoring_mode=fallback.scoring_mode,
                    fusion_weights=fallback.fusion_weights,
                    thresholds=fallback.thresholds,
                    nli_model_hint=fallback.nli_model_hint,
                    metric=fallback.metric,
                    metric_value=fallback.metric_value,
                    trained_on=fallback.trained_on,
                    # Keep the original (English) language on the bundle
                    # so callers know the bundle is *not* a German one.
                    # The router-side diagnostics still record what the
                    # caller requested.
                    language=fallback.language,
                    raw=fallback.raw,
                )
                _BUNDLES[cache_key] = rebadged
                return rebadged
        msg = f"missing calibration file: {path}"
        _LOAD_ERRORS[cache_key] = msg
        if language == DEFAULT_LANGUAGE:
            logger.warning("corpus_router.bundles: %s", msg)
        _BUNDLES[cache_key] = None
        return None

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        bundle = _parse_bundle(class_key, payload, language=language)
        _BUNDLES[cache_key] = bundle
        return bundle
    except Exception as exc:
        _LOAD_ERRORS[cache_key] = f"failed to parse {path}: {exc!r}"
        logger.exception("corpus_router.bundles: failed to load %s", path)
        _BUNDLES[cache_key] = None
        return None


def load_bundle(
    class_key: str, language: str = DEFAULT_LANGUAGE
) -> Optional[CalibrationBundle]:
    """Return the calibration bundle for ``(class_key, language)`` or ``None``.

    Resolution order:

    1. ``calibration.<class>.<language>.json`` if present.
    2. English fallback ``calibration.<class>.json`` with a one-shot
       ``bundle_language_fallback`` warning so the operator can spot
       silent fallbacks.
    3. ``None`` -- the caller (typically the corpus router) handles a
       missing bundle by falling through to the enterprise default.

    The result is memoised per ``(class_key, language)``; call
    :func:`reset_singleton_for_tests` to invalidate the cache between
    test cases.
    """

    if not language:
        language = DEFAULT_LANGUAGE
    with _LOCK:
        return _load_one_locked(class_key, language)


def load_all_bundles(
    language: str = DEFAULT_LANGUAGE,
) -> Dict[str, CalibrationBundle]:
    """Eager-load every known class for the given language.

    Used at startup so the router can log the (class, language) coverage
    matrix and operators can spot missing bundles before the first
    request lands.
    """

    out: Dict[str, CalibrationBundle] = {}
    with _LOCK:
        for key in KNOWN_CORPUS_TYPES:
            bundle = _load_one_locked(key, language)
            if bundle is not None:
                out[key] = bundle
    if language == DEFAULT_LANGUAGE:
        logger.info(
            "corpus_router.bundles loaded %d/%d classes (%s)",
            len(out),
            len(KNOWN_CORPUS_TYPES),
            language,
        )
    return out


def get_load_errors() -> Dict[str, str]:
    """Per-class load errors, surfaced for diagnostics.

    Keys collapse to the class name when only the English bundle was
    loaded (legacy callers). Per-language errors are exposed under
    ``"<class>:<language>"``.
    """

    load_all_bundles()  # Ensure default bundles loaded so legacy keys exist.
    out: Dict[str, str] = {}
    for (class_key, language), msg in _LOAD_ERRORS.items():
        if language == DEFAULT_LANGUAGE:
            out[class_key] = msg
        else:
            out[f"{class_key}:{language}"] = msg
    return out


def reset_singleton_for_tests() -> None:
    """Hook used by unit tests to force bundles to be re-loaded."""
    with _LOCK:
        _BUNDLES.clear()
        _LOAD_ERRORS.clear()
        _FALLBACK_WARNED.clear()


def known_corpus_types() -> Tuple[str, ...]:
    return KNOWN_CORPUS_TYPES


def supported_languages() -> Tuple[str, ...]:
    return SUPPORTED_LANGUAGES
