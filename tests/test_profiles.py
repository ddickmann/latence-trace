"""Tests for the Pareto-optimal default profile loader.

These tests focus on the behaviour callers actually depend on:

* The env preset for each profile is applied as documented in
  ``latence_trace/api/service.py``.
* Operator-set environment variables always win over the preset
  (the loader is non-destructive by design).
* The bundled per-profile JSON artefacts under
  ``latence_trace/data/`` are loadable and shape-compatible with the
  runtime risk-band classifier.
* Invalid profile names raise a ``ValueError`` with a helpful list of
  the supported names.
* The fusion-weight channel mask in
  ``latence_trace.core.nli.fuse_groundedness_v2`` renormalises
  cleanly when a profile turns a channel off (e.g. NLI is disabled
  in ``fast``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from latence_trace.api.service import (
    DEFAULT_PROFILE,
    PROFILE_ENV_PRESETS,
    PROFILE_NAMES,
    ProfileApplication,
    apply_profile,
)
from latence_trace.core.nli import fuse_groundedness_v2
from latence_trace.core.thresholds import (
    RiskBandPolicy,
    classify_risk_band,
    load_risk_band_policy,
)


_DATA_DIR = Path(__file__).resolve().parent.parent / "latence_trace" / "data"


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch):
    """Each test runs against a clean env-overlay slate.

    The profile loader mutates ``os.environ`` directly. We snapshot
    every key it might touch and restore it after each test so the
    suite stays order-independent.
    """

    keys = set()
    for preset in PROFILE_ENV_PRESETS.values():
        keys.update(preset.keys())
    keys.add("LATENCE_TRACE_ACTIVE_PROFILE")
    keys.add("LATENCE_TRACE_PROFILE")
    for key in keys:
        monkeypatch.delenv(key, raising=False)
    yield


def test_profile_names_match_preset_table():
    assert set(PROFILE_NAMES) == set(PROFILE_ENV_PRESETS.keys())
    assert DEFAULT_PROFILE in PROFILE_NAMES


def test_apply_profile_none_is_noop():
    result = apply_profile(None)
    assert isinstance(result, ProfileApplication)
    assert result.profile == "none"
    assert result.applied == {}
    assert result.skipped == {}


@pytest.mark.parametrize("profile", list(PROFILE_NAMES))
def test_apply_profile_sets_documented_env_keys(profile, monkeypatch):
    env = {}
    result = apply_profile(profile, env=env, refresh_thresholds=False)

    assert result.profile == profile
    expected_keys = set(PROFILE_ENV_PRESETS[profile].keys())
    assert set(result.applied.keys()) == expected_keys
    for key, value in PROFILE_ENV_PRESETS[profile].items():
        assert env[key] == value
    assert env["LATENCE_TRACE_ACTIVE_PROFILE"] == profile


@pytest.mark.parametrize("profile", list(PROFILE_NAMES))
def test_existing_env_vars_are_never_overridden(profile):
    custom_value = "definitely-not-the-preset"
    env = {
        "VOYAGER_GROUNDEDNESS_NLI_ENABLED": custom_value,
        "VOYAGER_GROUNDEDNESS_THRESHOLDS_PATH": "/tmp/operator-supplied.json",
    }

    result = apply_profile(profile, env=env, refresh_thresholds=False)

    assert env["VOYAGER_GROUNDEDNESS_NLI_ENABLED"] == custom_value
    assert env["VOYAGER_GROUNDEDNESS_THRESHOLDS_PATH"] == "/tmp/operator-supplied.json"
    assert result.skipped["VOYAGER_GROUNDEDNESS_NLI_ENABLED"] == custom_value
    assert result.skipped["VOYAGER_GROUNDEDNESS_THRESHOLDS_PATH"] == "/tmp/operator-supplied.json"
    assert "VOYAGER_GROUNDEDNESS_NLI_ENABLED" not in result.applied
    assert "VOYAGER_GROUNDEDNESS_THRESHOLDS_PATH" not in result.applied


def test_unknown_profile_raises_value_error_with_choices():
    with pytest.raises(ValueError) as exc:
        apply_profile("ludicrous", env={}, refresh_thresholds=False)
    msg = str(exc.value)
    assert "ludicrous" in msg
    for name in PROFILE_NAMES:
        assert name in msg


def test_apply_profile_overrides_take_precedence_over_preset():
    env = {}
    overrides = {"VOYAGER_GROUNDEDNESS_FUSION_W_NLI": "0.42"}
    result = apply_profile(
        "balanced", env=env, overrides=overrides, refresh_thresholds=False
    )
    assert env["VOYAGER_GROUNDEDNESS_FUSION_W_NLI"] == "0.42"
    assert result.applied["VOYAGER_GROUNDEDNESS_FUSION_W_NLI"] == "0.42"


@pytest.mark.parametrize("profile", list(PROFILE_NAMES))
def test_per_profile_thresholds_json_is_loadable(profile):
    path = _DATA_DIR / "thresholds.{0}.json".format(profile)
    assert path.exists(), "Missing threshold artefact for profile '{0}'".format(profile)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["profile"] == profile
    assert payload["headline"] in {
        "groundedness_v2",
        "groundedness_v2_no_nli",
        "reverse_context_calibrated",
    }
    assert isinstance(payload["strata"], dict) and payload["strata"], (
        "Profile '{0}' has no calibrated strata".format(profile)
    )
    for stratum, bounds in payload["strata"].items():
        assert "green_min" in bounds and "amber_min" in bounds, stratum
        assert bounds["green_min"] >= bounds["amber_min"], stratum


@pytest.mark.parametrize("profile", list(PROFILE_NAMES))
def test_per_profile_fusion_weights_json_is_loadable(profile):
    path = _DATA_DIR / "fusion_weights.{0}.json".format(profile)
    assert path.exists(), "Missing fusion-weight artefact for profile '{0}'".format(profile)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["profile"] == profile
    weights = payload["best"]["weights"]
    expected_channels = {"calibrated", "literal", "nli", "semantic_entropy", "structured"}
    assert set(weights.keys()) == expected_channels
    total = sum(weights.values())
    assert total > 0, "Profile '{0}' fusion weights sum to zero".format(profile)


@pytest.mark.parametrize("profile", list(PROFILE_NAMES))
def test_runtime_loads_per_profile_threshold_artefact(profile, monkeypatch):
    """Round-trip: apply_profile points the runtime loader at the right file."""

    env = {}
    apply_profile(profile, env=env, refresh_thresholds=False)
    path = Path(env["VOYAGER_GROUNDEDNESS_THRESHOLDS_PATH"])
    policy = load_risk_band_policy(path=path)
    assert isinstance(policy, RiskBandPolicy)
    assert policy.source == str(path)
    assert "default" in policy.strata, (
        "Profile '{0}' policy is missing a default stratum".format(profile)
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert policy.headline == payload["headline"]
    assert policy.nli_enabled == payload["nli_enabled"]


@pytest.mark.parametrize(
    ("profile", "expected_green", "expected_amber"),
    [
        ("fast", 0.95, 0.85),
        ("balanced", 0.80, 0.60),
        ("quality", 0.80, 0.60),
    ],
)
def test_profile_threshold_artefacts_define_product_default(
    profile, expected_green, expected_amber
):
    """Bundled artefacts must not derive default from the hardest stratum."""

    policy = load_risk_band_policy(path=_DATA_DIR / f"thresholds.{profile}.json")
    assert policy.strata["default"]["green_min"] == pytest.approx(expected_green)
    assert policy.strata["default"]["amber_min"] == pytest.approx(expected_amber)


def test_classify_risk_band_uses_profile_thresholds(monkeypatch):
    """End-to-end: a quality-profile policy should rate easy positives green."""

    policy = load_risk_band_policy(path=_DATA_DIR / "thresholds.balanced.json")
    assert classify_risk_band(0.947, policy=policy) == "green"
    assert classify_risk_band(0.839, policy=policy) == "green"
    assert classify_risk_band(0.70, policy=policy) == "amber"
    assert classify_risk_band(0.50, policy=policy) == "red"
    # The balanced policy is calibrated against minimal pairs so a
    # very high headline score must hit the green band on a known
    # stratum.
    assert classify_risk_band(0.99, stratum="entity_swap", policy=policy) == "green"
    # Explicit strata remain stricter than the user-facing default.
    assert classify_risk_band(0.839, stratum="unit_swap", policy=policy) == "amber"
    assert classify_risk_band(0.947, stratum="hard_dialogue_distributed", policy=policy) == "red"
    # A clearly ungrounded score must end up in the red band.
    assert classify_risk_band(0.05, stratum="entity_swap", policy=policy) == "red"
    # Missing scores always degrade to red so the classifier is safe
    # to call on unverifiable responses.
    assert classify_risk_band(None, policy=policy) == "red"


def _profile_weights(profile: str) -> dict:
    preset = PROFILE_ENV_PRESETS[profile]
    return {
        "calibrated": float(preset["VOYAGER_GROUNDEDNESS_FUSION_W_CALIBRATED"]),
        "literal": float(preset["VOYAGER_GROUNDEDNESS_FUSION_W_LITERAL"]),
        "nli": float(preset["VOYAGER_GROUNDEDNESS_FUSION_W_NLI"]),
        "semantic_entropy": float(
            preset["VOYAGER_GROUNDEDNESS_FUSION_W_SEMANTIC_ENTROPY"]
        ),
        "structured": float(preset["VOYAGER_GROUNDEDNESS_FUSION_W_STRUCTURED"]),
    }


def test_fast_profile_fusion_uses_literal_only_when_available():
    """The fast preset zeros calibrated/NLI/SE/structured, so only the
    literal channel contributes when the response carries literals."""

    weights = _profile_weights("fast")
    fused = fuse_groundedness_v2(
        reverse_context_calibrated=0.42,
        literal_guarded=0.91,
        nli_aggregate=None,
        semantic_entropy=None,
        structured_source_guarded=None,
        weights=weights,
    )
    assert fused == pytest.approx(0.91)


def test_fast_profile_fusion_returns_none_when_literal_channel_absent():
    """When all weighted channels are missing, ``fuse_groundedness_v2``
    must return None so the headline-resolver falls back to the raw
    calibrated reverse-context score (preserving the documented
    fast-profile fallback semantics)."""

    weights = _profile_weights("fast")
    fused = fuse_groundedness_v2(
        reverse_context_calibrated=0.42,
        literal_guarded=None,
        nli_aggregate=None,
        semantic_entropy=None,
        structured_source_guarded=None,
        weights=weights,
    )
    assert fused is None


def test_balanced_profile_uses_nli_when_only_nli_is_available():
    """The balanced preset lives or dies on NLI - if it is the only
    available channel the fused score must equal the NLI score."""

    weights = _profile_weights("balanced")
    fused = fuse_groundedness_v2(
        reverse_context_calibrated=0.5,
        literal_guarded=None,
        nli_aggregate=0.83,
        semantic_entropy=None,
        structured_source_guarded=None,
        weights=weights,
    )
    assert fused == pytest.approx(0.83)


def test_quality_profile_uses_only_calibrated_channels_by_default():
    """The quality preset zeros every channel except literal and NLI,
    matching the L7 sweep winner exactly. With both contributing the
    fused score is the weighted sum without renormalisation - SE is
    not in the preset so there is nothing to drop or rescale by
    default. This preserves the audit trail: the per-stratum F1 in
    fusion_weights.quality.json applies to what ships."""

    weights = _profile_weights("quality")
    fused = fuse_groundedness_v2(
        reverse_context_calibrated=0.6,
        literal_guarded=0.9,
        nli_aggregate=0.8,
        semantic_entropy=None,
        structured_source_guarded=None,
        weights=weights,
    )
    expected = 0.2 * 0.9 + 0.8 * 0.8
    assert fused == pytest.approx(expected)


def test_quality_profile_includes_semantic_entropy_when_provided():
    """When the caller passes ensemble samples we expect SE to
    contribute proportionally - the quality preset zeros SE by default
    so the renormalisation yields literal=0.2/(0.2+0.8) and
    nli=0.8/(0.2+0.8) on the contributing channels. SE shows up only
    when the operator explicitly opts in via env override."""

    weights = _profile_weights("quality")
    fused = fuse_groundedness_v2(
        reverse_context_calibrated=0.6,
        literal_guarded=0.9,
        nli_aggregate=0.8,
        semantic_entropy=0.7,
        structured_source_guarded=None,
        weights=weights,
    )
    expected = 0.2 * 0.9 + 0.8 * 0.8
    assert fused == pytest.approx(expected)


@pytest.mark.parametrize("profile", list(PROFILE_NAMES))
def test_preset_weights_match_calibration_artefacts(profile):
    """Lock the runtime preset, the threshold artefact, and the fusion
    sweep artefact in sync.

    If any of the three drifts the runtime can no longer claim the
    per-stratum F1 numbers from the calibration report apply to what
    actually ships. This test catches that the moment it happens.
    """

    preset = PROFILE_ENV_PRESETS[profile]
    preset_weights = {
        "calibrated": float(preset["VOYAGER_GROUNDEDNESS_FUSION_W_CALIBRATED"]),
        "literal": float(preset["VOYAGER_GROUNDEDNESS_FUSION_W_LITERAL"]),
        "nli": float(preset["VOYAGER_GROUNDEDNESS_FUSION_W_NLI"]),
        "semantic_entropy": float(
            preset["VOYAGER_GROUNDEDNESS_FUSION_W_SEMANTIC_ENTROPY"]
        ),
        "structured": float(preset["VOYAGER_GROUNDEDNESS_FUSION_W_STRUCTURED"]),
    }

    th_path = _DATA_DIR / "thresholds.{0}.json".format(profile)
    th = json.loads(th_path.read_text(encoding="utf-8"))
    assert "fusion_weights" in th, (
        "Threshold artefact for '{0}' is missing the self-describing "
        "fusion_weights block.".format(profile)
    )
    th_weights = {k: float(v) for k, v in th["fusion_weights"].items()}
    assert th_weights == preset_weights, (
        "Threshold artefact disagrees with PROFILE_ENV_PRESETS for "
        "'{0}'. Re-run scripts/calibrate_thresholds.py with the "
        "current preset env vars exported.".format(profile)
    )

    fw_path = _DATA_DIR / "fusion_weights.{0}.json".format(profile)
    fw = json.loads(fw_path.read_text(encoding="utf-8"))
    fw_weights = {k: float(v) for k, v in fw["best"]["weights"].items()}
    assert fw_weights == preset_weights, (
        "Fusion-sweep winner for '{0}' (artefact) disagrees with "
        "PROFILE_ENV_PRESETS. Either the preset is wrong or the sweep "
        "needs to be re-run.".format(profile)
    )
