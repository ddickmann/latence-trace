"""Unit tests for the fusion missing-channel substitution hook.

The corpus-router enables ``substitute_missing_channels_threshold`` for
every routed request so a crashed NLI engine cannot silently collapse
the headline score onto the calibrated channel. These tests pin the
math of that substitution and the backward-compat default (0.0).
"""

from __future__ import annotations

from latence_trace.core.nli import fuse_groundedness_v2


def test_default_drop_and_renormalise_behaviour_preserved() -> None:
    """With ``substitute_missing_channels_threshold=0.0`` (default) the
    fusion drops the missing channel and renormalises the others. This
    is the pre-router behaviour and must stay exact so legacy callers
    see bitwise-identical scores."""

    fused = fuse_groundedness_v2(
        reverse_context_calibrated=0.98,
        literal_guarded=None,
        nli_aggregate=None,
        weights={"calibrated": 0.625, "literal": 0.0, "nli": 0.375},
    )
    assert fused is not None and abs(fused - 0.98) < 1e-6


def test_substitute_missing_nli_with_uncertainty_prior() -> None:
    """When NLI is missing but the bundle relied on it, substituting
    0.5 pulls the fused score down toward amber so we never leak a
    calibrated-only 'green'. Closed form:

        0.625 * 0.98 + 0.375 * 0.5 = 0.8
    """

    fused = fuse_groundedness_v2(
        reverse_context_calibrated=0.98,
        literal_guarded=None,
        nli_aggregate=None,
        weights={"calibrated": 0.625, "literal": 0.0, "nli": 0.375},
        substitute_missing_channels_threshold=0.2,
    )
    assert fused is not None
    assert abs(fused - 0.8) < 1e-6


def test_substitution_ignores_zero_weight_channels() -> None:
    """Channels the bundle explicitly zeroed out must not trigger the
    substitution, regardless of the missing-value threshold."""

    fused = fuse_groundedness_v2(
        reverse_context_calibrated=0.90,
        literal_guarded=None,
        nli_aggregate=None,
        semantic_entropy=None,
        structured_source_guarded=None,
        weights={
            "calibrated": 1.0,
            "literal": 0.0,
            "nli": 0.0,
            "semantic_entropy": 0.0,
            "structured": 0.0,
        },
        substitute_missing_channels_threshold=0.2,
    )
    assert fused == 0.90


def test_substitution_with_custom_prior() -> None:
    """Callers may pin a more conservative prior (e.g. 0.3) to dampen
    harder."""

    fused = fuse_groundedness_v2(
        reverse_context_calibrated=0.98,
        literal_guarded=None,
        nli_aggregate=None,
        weights={"calibrated": 0.5, "nli": 0.5},
        substitute_missing_channels_threshold=0.2,
        substitute_missing_channels_prior=0.3,
    )
    assert fused is not None
    # 0.5 * 0.98 + 0.5 * 0.3 = 0.64
    assert abs(fused - 0.64) < 1e-6


def test_substitution_with_all_channels_present_is_noop() -> None:
    """When every channel with positive weight has a real value, the
    substitution hook must have zero effect."""

    weights = {"calibrated": 0.6, "nli": 0.4}
    without = fuse_groundedness_v2(
        reverse_context_calibrated=0.9,
        literal_guarded=None,
        nli_aggregate=0.8,
        weights=weights,
    )
    with_sub = fuse_groundedness_v2(
        reverse_context_calibrated=0.9,
        literal_guarded=None,
        nli_aggregate=0.8,
        weights=weights,
        substitute_missing_channels_threshold=0.2,
    )
    assert without == with_sub
