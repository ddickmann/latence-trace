"""Runtime loader + classifier for the Phase H risk-band system.

The calibration pipeline (``research/triangular_maxsim/calibrate_thresholds.py``)
writes a JSON artefact that maps failure strata to per-stratum green/amber
score thresholds. At runtime we load that artefact once, fall back to a
conservative default if it is missing, and expose a single helper
``classify_risk_band`` that maps a headline score to one of
``{"green", "amber", "red"}``.

The stratum concept is intentionally simple here: callers can attach a
``stratum`` hint via the request payload; otherwise the classifier
applies a conservative "default" threshold that is the max across all
calibrated strata so the green band stays honest for the worst known
failure mode.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_PATH_ENV = "VOYAGER_GROUNDEDNESS_THRESHOLDS_PATH"
_DEFAULT_FILENAME = "thresholds.json"

# Product-default thresholds used for normal RAG calls when callers do not
# provide a failure-mode stratum. Explicit strata preserve precision-first
# thresholds for known failure modes; the default band should be intuitive for
# users reading a 0-1 groundedness score.
_PRODUCT_DEFAULT_THRESHOLDS: Dict[str, float] = {
    "green_min": 0.80,
    "amber_min": 0.60,
}

# Conservative fallback thresholds used when no calibration artefact is
# available. Stratum-specific entries still err on the side of rejecting
# borderline responses, but the product default maps high groundedness scores
# to the user-facing green band.
_FALLBACK_PAYLOAD: Dict[str, Any] = {
    "schema_version": 1,
    "headline": "groundedness_v2",
    "precision_target": 0.75,
    "nli_enabled": True,
    "pair_count": 0,
    "strata": {
        "default": dict(_PRODUCT_DEFAULT_THRESHOLDS),
        "entity_swap": {"green_min": 0.70, "amber_min": 0.55},
        "date_swap": {"green_min": 0.72, "amber_min": 0.60},
        "number_swap": {"green_min": 0.70, "amber_min": 0.55},
        "unit_swap": {"green_min": 0.70, "amber_min": 0.55},
        "negation": {"green_min": 0.75, "amber_min": 0.60},
        "role_swap": {"green_min": 0.75, "amber_min": 0.60},
        "partial": {"green_min": 0.72, "amber_min": 0.58},
    },
    "source": "fallback_default",
}


@dataclass
class RiskBandPolicy:
    """In-memory view of the calibrated risk-band thresholds."""

    headline: str = "groundedness_v2"
    precision_target: float = 0.75
    nli_enabled: bool = True
    strata: Dict[str, Dict[str, float]] = field(default_factory=dict)
    source: str = "fallback_default"
    schema_version: int = 1

    def threshold_for(self, stratum: Optional[str]) -> Dict[str, float]:
        if stratum and stratum in self.strata:
            entry = self.strata[stratum]
        else:
            entry = self.strata.get("default", {})
        return {
            "green_min": float(
                entry.get("green_min", _PRODUCT_DEFAULT_THRESHOLDS["green_min"])
            ),
            "amber_min": float(
                entry.get("amber_min", _PRODUCT_DEFAULT_THRESHOLDS["amber_min"])
            ),
        }


def _artefact_path() -> Path:
    env = os.environ.get(_DEFAULT_PATH_ENV)
    if env:
        return Path(env)
    return Path(__file__).resolve().parent / _DEFAULT_FILENAME


def _resolve_default_stratum(strata: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Ensure a product-default ``default`` entry exists.

    Stratum-specific thresholds are calibrated for precision on explicit
    failure modes (number swaps, negation, hard dialogue, etc.). For ordinary
    RAG calls without ``risk_band_stratum`` we need a stable, intuitive
    headline band: high groundedness should be green, mid confidence amber,
    and low support red. Do not synthesize the default from the hardest
    stratum, because a single pathological stratum can make a 0.94 score red.
    """

    if "default" in strata:
        return strata
    strata = dict(strata)
    strata["default"] = dict(_PRODUCT_DEFAULT_THRESHOLDS)
    return strata


def load_risk_band_policy(path: Optional[Path] = None) -> RiskBandPolicy:
    """Load the calibrated thresholds, falling back to defaults on error."""

    artefact_path = path or _artefact_path()
    payload: Dict[str, Any] = dict(_FALLBACK_PAYLOAD)
    try:
        if artefact_path.exists():
            with artefact_path.open("r", encoding="utf-8") as fp:
                parsed = json.load(fp)
            if isinstance(parsed, dict) and isinstance(parsed.get("strata"), dict):
                payload = parsed
                payload["source"] = str(artefact_path)
            else:
                logger.warning(
                    "groundedness_thresholds_invalid",
                    extra={"path": str(artefact_path)},
                )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "groundedness_thresholds_load_failed",
            extra={"error": str(exc), "path": str(artefact_path)},
        )

    strata_raw = payload.get("strata", {}) or {}
    strata: Dict[str, Dict[str, float]] = {
        str(name): {
            "green_min": float(entry.get("green_min", 0.70)),
            "amber_min": float(entry.get("amber_min", 0.55)),
        }
        for name, entry in strata_raw.items()
        if isinstance(entry, dict)
    }
    strata = _resolve_default_stratum(strata)
    return RiskBandPolicy(
        headline=str(payload.get("headline", "groundedness_v2")),
        precision_target=float(payload.get("precision_target", 0.75)),
        nli_enabled=bool(payload.get("nli_enabled", True)),
        strata=strata,
        source=str(payload.get("source", "fallback_default")),
        schema_version=int(payload.get("schema_version", 1)),
    )


_POLICY_CACHE: Optional[RiskBandPolicy] = None


def get_risk_band_policy(*, refresh: bool = False) -> RiskBandPolicy:
    """Cached accessor for the global risk-band policy."""

    global _POLICY_CACHE
    if _POLICY_CACHE is None or refresh:
        _POLICY_CACHE = load_risk_band_policy()
    return _POLICY_CACHE


def classify_risk_band(
    headline_score: Optional[float],
    *,
    stratum: Optional[str] = None,
    policy: Optional[RiskBandPolicy] = None,
) -> str:
    """Map ``headline_score`` to one of ``{"green", "amber", "red"}``.

    When the score is missing, returns ``"red"`` to err on the side of
    caution. Callers should pass the dominant failure stratum they are
    worried about (e.g. ``negation``); when omitted we use the hardest
    threshold from the calibration artefact.
    """

    if headline_score is None:
        return "red"
    policy = policy or get_risk_band_policy()
    thresholds = policy.threshold_for(stratum)
    score = float(headline_score)
    if score >= thresholds["green_min"]:
        return "green"
    if score >= thresholds["amber_min"]:
        return "amber"
    return "red"


def thresholds_summary_for(policy: Optional[RiskBandPolicy] = None) -> Dict[str, Any]:
    """Return a JSON-safe summary for the API response/diagnostics."""

    policy = policy or get_risk_band_policy()
    return {
        "headline": policy.headline,
        "precision_target": policy.precision_target,
        "nli_enabled": policy.nli_enabled,
        "source": policy.source,
        "strata": {
            name: dict(thresholds) for name, thresholds in policy.strata.items()
        },
    }


def list_known_strata(policy: Optional[RiskBandPolicy] = None) -> List[str]:
    policy = policy or get_risk_band_policy()
    return sorted(policy.strata.keys())


__all__ = [
    "RiskBandPolicy",
    "classify_risk_band",
    "get_risk_band_policy",
    "load_risk_band_policy",
    "list_known_strata",
    "thresholds_summary_for",
]
