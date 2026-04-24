"""Regression guards for the shipped composite phantom-guard.

The v3 quality-boost sprint signed off shipping the calibrated logistic
composite as the default request-path score. If the
``composite_logistic_v3.json`` artefact ever goes missing from the wheel
(e.g. forgotten ``package-data`` entry, stale Docker layer), the lane
silently degrades to :class:`LinearComposite`. These tests surface the
regression in CI rather than in production metrics.
"""

from __future__ import annotations

import os

import pytest

from latence_trace.core.code_lane.composite import (
    CompositeFeatures,
    LinearComposite,
    LogisticComposite,
    default_composite,
)


def test_logistic_artifact_is_shipped_and_loads() -> None:
    """The packaged artefact must exist next to ``composite.py``."""
    path = LogisticComposite.default_path()
    assert path.exists(), (
        "composite_logistic_v3.json is missing next to composite.py. "
        "Re-fit via scripts/fit_composite_logistic_v3.py before shipping."
    )
    logistic = LogisticComposite.load()
    assert logistic.artifact.feature_order, "logistic artefact missing feature_order"
    assert logistic.artifact.coef, "logistic artefact missing coefficients"


def test_default_composite_is_logistic() -> None:
    """Without any override, production must use the calibrated logistic."""
    prev = os.environ.pop("LATENCE_TRACE_COMPOSITE", None)
    try:
        composite = default_composite()
        assert isinstance(composite, LogisticComposite), (
            "default_composite() returned "
            f"{type(composite).__name__}; "
            "the calibrated logistic is the signed-off production composite."
        )
    finally:
        if prev is not None:
            os.environ["LATENCE_TRACE_COMPOSITE"] = prev


def test_operator_can_force_linear_via_env() -> None:
    """The linear escape hatch must be honoured so operators can roll back."""
    prev = os.environ.get("LATENCE_TRACE_COMPOSITE")
    os.environ["LATENCE_TRACE_COMPOSITE"] = "linear"
    try:
        composite = default_composite()
        assert isinstance(composite, LinearComposite)
    finally:
        if prev is None:
            os.environ.pop("LATENCE_TRACE_COMPOSITE", None)
        else:
            os.environ["LATENCE_TRACE_COMPOSITE"] = prev


def test_logistic_produces_bounded_probability() -> None:
    """Sanity: the logistic head must return a valid probability and verdict."""
    logistic = LogisticComposite.load()
    grounded = CompositeFeatures(
        reverse_context=0.95, per_token_p10=0.80, literal_guard=1.0
    )
    phantom = CompositeFeatures(
        reverse_context=0.30, per_token_p10=0.10, literal_guard=0.2
    )
    good = logistic.score(grounded)
    bad = logistic.score(phantom)
    assert 0.0 <= good.phantom_probability <= 1.0
    assert 0.0 <= bad.phantom_probability <= 1.0
    assert good.kind == "logistic"
    assert bad.kind == "logistic"
    # A strongly-grounded feature bag must score higher than a phantom bag on
    # every coefficient sign, regardless of the exact training set.
    assert good.composite_score > bad.composite_score, (
        f"logistic composite failed monotonicity: grounded={good.composite_score:.3f} "
        f"phantom={bad.composite_score:.3f}"
    )
