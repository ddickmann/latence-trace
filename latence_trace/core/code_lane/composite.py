"""Composite phantom-guard score.

Two implementations are shipped:

- :class:`LinearComposite` — the v2 weighted sum of
  ``reverse_context``, ``per_token_p10``, and ``literal_guard`` that
  proved robust on the transcripts_v2 bank. Kept as a deterministic
  fallback so the lane still works when no calibration artefact has
  been loaded.
- :class:`LogisticComposite` — a calibrated :class:`sklearn` logistic
  regression with interaction terms. Trained on pooled v1+v2 data and
  persisted alongside this module as ``composite_logistic_v3.json``;
  loaded lazily at ``default_composite()`` time. When the artefact is
  not present (fresh checkouts, offline sandboxes) the linear fallback
  keeps the lane fully functional. The artefact can be re-fit offline
  via :func:`fit_logistic_from_payloads`.

Both implementations share the same :class:`CompositeFeatures` input,
so swapping is a one-line change in the orchestrator.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


_MODULE_DIR = Path(__file__).resolve().parent
_COMPOSITE_ARTIFACT = _MODULE_DIR / "composite_logistic_v3.json"


@dataclass
class CompositeFeatures:
    """Signals fed into the composite phantom-guard score.

    Every field has a sensible default so callers can build a partial
    bag and still get a valid composite (useful for the ablation
    harness). Missing signals are imputed to the neutral midpoint
    rather than 0 so they do not spuriously flag a grounded turn.
    """

    reverse_context: float = 0.75
    per_token_p10: float = 0.5
    literal_guard: float = 1.0
    literal_novelty_min: float = 0.75
    ast_phantom_symbol_count: int = 0
    ast_literal_drift_count: int = 0
    nli_contradiction_prob_max: float = 0.0
    semantic_entropy_aggregate: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "reverse_context": float(self.reverse_context),
            "per_token_p10": float(self.per_token_p10),
            "literal_guard": float(self.literal_guard),
            "literal_novelty_min": float(self.literal_novelty_min),
            "ast_phantom_symbol_count": int(self.ast_phantom_symbol_count),
            "ast_literal_drift_count": int(self.ast_literal_drift_count),
            "nli_contradiction_prob_max": float(self.nli_contradiction_prob_max),
            "semantic_entropy_aggregate": (
                None
                if self.semantic_entropy_aggregate is None
                else float(self.semantic_entropy_aggregate)
            ),
        }


@dataclass
class CompositeResult:
    """Scalar output plus the exact features + weights used."""

    composite_score: float
    phantom_probability: float
    verdict: bool
    kind: str  # "linear" | "logistic"
    threshold: float
    features: CompositeFeatures
    contributions: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "composite_score": float(self.composite_score),
            "phantom_probability": float(self.phantom_probability),
            "verdict": bool(self.verdict),
            "kind": self.kind,
            "threshold": float(self.threshold),
            "features": self.features.as_dict(),
            "contributions": {k: float(v) for k, v in self.contributions.items()},
        }


# ---------------------------------------------------------------------------
# Linear composite (default / fallback)
# ---------------------------------------------------------------------------


_LINEAR_DEFAULT_WEIGHTS: Dict[str, float] = {
    "reverse_context": 0.35,
    "per_token_p10": 0.35,
    "literal_guard": 0.20,
    "literal_novelty_min": 0.10,
}


class LinearComposite:
    """Deterministic weighted-sum composite.

    Weights are normalised so the positive contributors sum to 1.
    Negative features (phantom / drift counts, NLI contradiction, SE
    entropy) subtract from the headline score with fixed coefficients.
    """

    def __init__(
        self,
        *,
        weights: Optional[Mapping[str, float]] = None,
        threshold: float = 0.65,
        ast_penalty: float = 0.15,
        drift_penalty: float = 0.02,
        nli_penalty: float = 0.6,
        se_penalty: float = 0.3,
    ) -> None:
        self.weights = dict(weights or _LINEAR_DEFAULT_WEIGHTS)
        w_sum = sum(v for v in self.weights.values() if v > 0) or 1.0
        self.weights = {k: v / w_sum for k, v in self.weights.items()}
        self.threshold = float(threshold)
        self.ast_penalty = float(ast_penalty)
        self.drift_penalty = float(drift_penalty)
        self.nli_penalty = float(nli_penalty)
        self.se_penalty = float(se_penalty)

    def score(self, features: CompositeFeatures) -> CompositeResult:
        contributions: Dict[str, float] = {}
        base = 0.0
        for name, w in self.weights.items():
            value = float(getattr(features, name))
            term = float(w) * value
            contributions[name] = term
            base += term
        ast_term = -self.ast_penalty * float(features.ast_phantom_symbol_count)
        drift_term = -self.drift_penalty * float(features.ast_literal_drift_count)
        nli_term = -self.nli_penalty * float(features.nli_contradiction_prob_max)
        se_term = 0.0
        if features.semantic_entropy_aggregate is not None:
            # ``semantic_entropy_aggregate`` is the *positive* consistency
            # score (see :mod:`latence_trace.core.semantic_entropy`). We
            # reward agreement and penalise divergence around 0.5.
            se_term = self.se_penalty * (float(features.semantic_entropy_aggregate) - 0.5)
        composite = base + ast_term + drift_term + nli_term + se_term
        composite = max(0.0, min(1.0, composite))
        contributions.update(
            ast_phantom=ast_term,
            ast_drift=drift_term,
            nli_contradiction=nli_term,
            semantic_entropy=se_term,
        )
        # Map to a probability via a logistic on (threshold - score).
        phantom_prob = 1.0 / (1.0 + math.exp(6.0 * (composite - self.threshold)))
        return CompositeResult(
            composite_score=float(composite),
            phantom_probability=float(phantom_prob),
            verdict=bool(composite < self.threshold),
            kind="linear",
            threshold=self.threshold,
            features=features,
            contributions=contributions,
        )


# ---------------------------------------------------------------------------
# Logistic composite (production)
# ---------------------------------------------------------------------------


@dataclass
class LogisticArtifact:
    """Serialised logistic-regression coefficients."""

    feature_order: List[str]
    interactions: List[Tuple[str, str]]
    coef: List[float]
    intercept: float
    threshold: float
    auroc_pooled: Optional[float] = None
    trained_on: Optional[str] = None
    version: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_order": list(self.feature_order),
            "interactions": [[a, b] for a, b in self.interactions],
            "coef": list(self.coef),
            "intercept": float(self.intercept),
            "threshold": float(self.threshold),
            "auroc_pooled": self.auroc_pooled,
            "trained_on": self.trained_on,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LogisticArtifact":
        return cls(
            feature_order=list(payload["feature_order"]),
            interactions=[tuple(pair) for pair in payload.get("interactions", [])],
            coef=list(payload["coef"]),
            intercept=float(payload["intercept"]),
            threshold=float(payload.get("threshold", 0.5)),
            auroc_pooled=payload.get("auroc_pooled"),
            trained_on=payload.get("trained_on"),
            version=payload.get("version"),
        )


class LogisticComposite:
    """Logistic-regression composite with interaction terms.

    The coefficients live in a JSON artefact so we can re-fit offline
    without rebuilding the container. ``score()`` is deterministic and
    thread-safe (coefficients are read-only after construction).
    """

    def __init__(self, artifact: LogisticArtifact) -> None:
        self.artifact = artifact

    @classmethod
    def default_path(cls) -> Path:
        return _COMPOSITE_ARTIFACT

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "LogisticComposite":
        path = path or cls.default_path()
        with path.open() as handle:
            payload = json.load(handle)
        return cls(LogisticArtifact.from_dict(payload))

    @classmethod
    def try_load(cls, path: Optional[Path] = None) -> Optional["LogisticComposite"]:
        try:
            return cls.load(path)
        except FileNotFoundError:
            return None
        except Exception as exc:
            logger.warning("logistic_artifact_load_failed", extra={"error": str(exc)})
            return None

    def _feature_vector(self, features: CompositeFeatures) -> List[float]:
        base: List[float] = []
        for name in self.artifact.feature_order:
            value = getattr(features, name, 0.0)
            base.append(float(value) if value is not None else 0.0)
        for a, b in self.artifact.interactions:
            va = float(getattr(features, a, 0.0) or 0.0)
            vb = float(getattr(features, b, 0.0) or 0.0)
            base.append(va * vb)
        return base

    def score(self, features: CompositeFeatures) -> CompositeResult:
        vec = self._feature_vector(features)
        z = self.artifact.intercept + sum(
            c * v for c, v in zip(self.artifact.coef, vec)
        )
        phantom_prob = 1.0 / (1.0 + math.exp(-z))
        composite = 1.0 - phantom_prob
        contributions: Dict[str, float] = {}
        base_len = len(self.artifact.feature_order)
        for idx, name in enumerate(self.artifact.feature_order):
            contributions[name] = float(self.artifact.coef[idx] * vec[idx])
        for jdx, (a, b) in enumerate(self.artifact.interactions):
            contributions[f"{a}*{b}"] = float(
                self.artifact.coef[base_len + jdx] * vec[base_len + jdx]
            )
        return CompositeResult(
            composite_score=float(composite),
            phantom_probability=float(phantom_prob),
            verdict=bool(phantom_prob > self.artifact.threshold),
            kind="logistic",
            threshold=self.artifact.threshold,
            features=features,
            contributions=contributions,
        )


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def default_composite() -> Any:
    """Return the best available composite.

    Resolution order:

    1. If ``LATENCE_TRACE_COMPOSITE`` is set to ``linear`` (or ``force_linear``),
       return :class:`LinearComposite`. This is an operator escape hatch for
       rolling back to the deterministic weighted-sum composite without a
       redeploy, e.g. if a freshly fit logistic artefact regresses in the wild.
    2. Otherwise, if the calibrated logistic artefact exists *and* can be
       loaded, use it.
    3. Otherwise fall back to the linear composite with the v2 weights and
       emit a ``WARNING`` so the lane-quality audit surfaces that the
       signed-off logistic is not active (typical cause: missing
       ``composite_logistic_v3.json`` in the wheel or container).
    """
    override = (os.getenv("LATENCE_TRACE_COMPOSITE") or "").strip().lower()
    if override in {"linear", "force_linear"}:
        logger.info("composite_override_linear", extra={"env": "LATENCE_TRACE_COMPOSITE"})
        return LinearComposite()
    logistic = LogisticComposite.try_load()
    if logistic is not None:
        logger.info(
            "composite_active_logistic",
            extra={
                "artifact_path": str(LogisticComposite.default_path()),
                "feature_order": logistic.artifact.feature_order,
                "auroc_pooled": logistic.artifact.auroc_pooled,
                "version": logistic.artifact.version,
            },
        )
        return logistic
    logger.warning(
        "composite_fallback_linear",
        extra={
            "artifact_path": str(LogisticComposite.default_path()),
            "reason": "logistic_artifact_missing_or_unreadable",
        },
    )
    return LinearComposite()


# ---------------------------------------------------------------------------
# Offline fit helper (used by the ablation harness)
# ---------------------------------------------------------------------------


def fit_logistic_from_payloads(
    samples: Sequence[Tuple[CompositeFeatures, int]],
    *,
    feature_order: Sequence[str] = (
        "reverse_context",
        "per_token_p10",
        "literal_guard",
        "literal_novelty_min",
        "ast_phantom_symbol_count",
        "ast_literal_drift_count",
        "nli_contradiction_prob_max",
    ),
    interactions: Sequence[Tuple[str, str]] = (
        ("per_token_p10", "literal_guard"),
        ("reverse_context", "literal_novelty_min"),
        ("nli_contradiction_prob_max", "literal_guard"),
    ),
    threshold: float = 0.5,
    trained_on: Optional[str] = None,
    version: Optional[str] = None,
) -> LogisticArtifact:
    """Fit the logistic composite on ``(features, phantom_label)`` pairs.

    ``phantom_label == 1`` means the turn is a phantom / hallucination;
    ``0`` means grounded.
    """
    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
    except Exception as exc:  # pragma: no cover - import-time guard
        raise RuntimeError(
            "scikit-learn is required to fit the logistic composite"
        ) from exc

    rows: List[List[float]] = []
    labels: List[int] = []
    for feats, label in samples:
        row: List[float] = []
        for name in feature_order:
            row.append(float(getattr(feats, name, 0.0) or 0.0))
        for a, b in interactions:
            row.append(
                float(getattr(feats, a, 0.0) or 0.0)
                * float(getattr(feats, b, 0.0) or 0.0)
            )
        rows.append(row)
        labels.append(int(label))

    X = np.asarray(rows, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    model = LogisticRegression(
        max_iter=500,
        class_weight="balanced",
        C=1.0,
        solver="lbfgs",
    )
    model.fit(X, y)
    coef = model.coef_[0].tolist()
    intercept = float(model.intercept_[0])
    try:
        from sklearn.metrics import roc_auc_score

        y_score = model.predict_proba(X)[:, 1]
        auroc = float(roc_auc_score(y, y_score))
    except Exception:
        auroc = None
    return LogisticArtifact(
        feature_order=list(feature_order),
        interactions=list(interactions),
        coef=coef,
        intercept=intercept,
        threshold=float(threshold),
        auroc_pooled=auroc,
        trained_on=trained_on,
        version=version,
    )


def persist_artifact(artifact: LogisticArtifact, *, path: Optional[Path] = None) -> Path:
    path = path or _COMPOSITE_ARTIFACT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact.to_dict(), indent=2))
    return path
