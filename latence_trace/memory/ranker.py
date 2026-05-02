"""Transparent learned baseline ranker for TRACE Memory labels."""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, Field

_LABEL_TARGET = {"remove": 0.0, "superseded": 0.2, "demote": 0.35, "keep": 0.75, "anchor": 1.0}
_FEATURES = (
    "salience",
    "relevance",
    "attribution",
    "exact_critical",
    "dead_weight",
    "redundancy",
    "domain_decay_pressure",
    "is_code_domain",
    "is_rag_or_search_domain",
    "is_tool_domain",
)


class RankerArtifact(BaseModel):
    model_type: str = "mean_difference_linear_ranker"
    weights: dict[str, float]
    label_counts: dict[str, int]
    training_rows: int
    diagnostics: dict[str, Any] = Field(default_factory=dict)


def rows_from_labels(labels: list[Any]) -> list[dict[str, Any]]:
    rows = []
    for item in labels:
        audit = getattr(item, "audit", {}) or item.get("audit", {})
        label = getattr(item, "label", None) or item.get("label")
        domain = str(audit.get("trajectory_domain", "")).lower()
        if label not in _LABEL_TARGET:
            continue
        rows.append(
            {
                "span_id": getattr(item, "span_id", None) or item.get("span_id"),
                "label": label,
                "target": _LABEL_TARGET[label],
                "features": features_from_scores(audit, domain=domain),
            }
        )
    return rows


def train_ranker(rows: list[dict[str, Any]]) -> RankerArtifact:
    if not rows:
        raise ValueError("Cannot train memory ranker without survival labels")
    positives = [row for row in rows if row["target"] >= 0.75]
    negatives = [row for row in rows if row["target"] < 0.75]
    weights: dict[str, float] = {}
    for feature in _FEATURES:
        pos_mean = sum(row["features"].get(feature, 0.0) for row in positives) / max(1, len(positives))
        neg_mean = sum(row["features"].get(feature, 0.0) for row in negatives) / max(1, len(negatives))
        weights[feature] = round(pos_mean - neg_mean, 6)
    return RankerArtifact(
        weights=weights,
        label_counts=_counts(row["label"] for row in rows),
        training_rows=len(rows),
        diagnostics={
            "positive_rows": len(positives),
            "negative_rows": len(negatives),
            "domain_counts": _counts(row.get("domain", "unknown") for row in rows),
            "long_horizon_training_rows": sum(
                1 for row in rows if int(row.get("turn_count", 0) or 0) >= 100
            ),
            "max_training_turn_count": max((int(row.get("turn_count", 0) or 0) for row in rows), default=0),
            "exact_critical_guard": "hard_rule_retained",
        },
    )


def features_from_scores(scores: Any, *, domain: str) -> dict[str, float]:
    return {
        "salience": _score_value(scores, "salience"),
        "relevance": _score_value(scores, "relevance"),
        "attribution": _score_value(scores, "attribution"),
        "exact_critical": _score_value(scores, "exact_critical"),
        "dead_weight": _score_value(scores, "dead_weight"),
        "redundancy": _score_value(scores, "redundancy"),
        "domain_decay_pressure": _score_value(scores, "domain_decay_pressure"),
        "is_code_domain": 1.0 if domain == "code" else 0.0,
        "is_rag_or_search_domain": 1.0 if domain in {"rag", "search", "grounding", "chat"} else 0.0,
        "is_tool_domain": 1.0 if domain in {"tool", "workflow"} else 0.0,
    }


def score_features(features: dict[str, float], weights: dict[str, float]) -> float:
    if not weights:
        return 0.0
    raw = sum(float(weights.get(feature, 0.0)) * features.get(feature, 0.0) for feature in _FEATURES)
    return max(0.0, min(1.0, 0.5 + 0.5 * math.tanh(raw)))


def score_span(span: Any, weights: dict[str, float], *, domain: str) -> float:
    return score_features(features_from_scores(span.scores, domain=domain), weights)


def _score_value(scores: Any, field: str) -> float:
    if isinstance(scores, dict):
        return float(scores.get(field, 0.0) or 0.0)
    return float(getattr(scores, field, 0.0) or 0.0)


def _counts(labels: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return counts
