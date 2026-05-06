"""Unit tests for the calibration bundle loader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from latence_trace.core.corpus_router import bundles as _bundles


@pytest.fixture(autouse=True)
def _reset_singletons():
    _bundles.reset_singleton_for_tests()
    yield
    _bundles.reset_singleton_for_tests()


def test_known_corpus_types_cover_six_classes() -> None:
    assert len(_bundles.KNOWN_CORPUS_TYPES) == 6
    assert "rag.prose.enterprise" in _bundles.KNOWN_CORPUS_TYPES
    assert "code.agentic_trace" in _bundles.KNOWN_CORPUS_TYPES


def test_load_all_bundles_succeeds_for_shipped_artefacts() -> None:
    loaded = _bundles.load_all_bundles()
    # Every class ships with a calibration file produced by
    # scripts/calibrate_per_class.py. Missing any of them means the
    # shipped artefacts drifted.
    for key in _bundles.KNOWN_CORPUS_TYPES:
        assert key in loaded, f"missing calibration for {key}"
        b = loaded[key]
        assert b.class_key == key
        assert b.scoring_mode in {"rag", "code"}
        assert 0.0 <= b.amber_threshold <= b.green_threshold <= 1.0


def test_bundle_rejects_sum_below_safety_floor(tmp_path: Path, monkeypatch) -> None:
    bad = {
        "scoring_mode": "rag",
        "fusion_weights": {"calibrated": 0.1, "literal": 0.1, "nli": 0.0, "semantic_entropy": 0.0, "structured": 0.0},
        "thresholds": {"green": 0.8, "amber": 0.6},
        "metric": "f1_at_best_threshold",
        "metric_value": 0.5,
    }
    with pytest.raises(ValueError, match="fusion weights sum"):
        _bundles._parse_bundle("rag.prose.enterprise", bad)


def test_bundle_rejects_inverted_thresholds() -> None:
    bad = {
        "scoring_mode": "rag",
        "fusion_weights": {"calibrated": 1.0, "literal": 0.0, "nli": 0.0, "semantic_entropy": 0.0, "structured": 0.0},
        "thresholds": {"green": 0.5, "amber": 0.7},
        "metric": "f1_at_best_threshold",
        "metric_value": 0.5,
    }
    with pytest.raises(ValueError, match="thresholds invalid"):
        _bundles._parse_bundle("rag.prose.enterprise", bad)


def test_load_bundle_returns_none_for_unknown_class() -> None:
    assert _bundles.load_bundle("rag.unknown.class") is None


def test_green_amber_properties() -> None:
    bundle = _bundles.CalibrationBundle(
        class_key="rag.prose.enterprise",
        scoring_mode="rag",
        fusion_weights={"calibrated": 1.0, "literal": 0.0, "nli": 0.0, "semantic_entropy": 0.0, "structured": 0.0},
        thresholds={"green": 0.9, "amber": 0.7},
        nli_model_hint="en",
        metric="f1_at_best_threshold",
        metric_value=0.9,
        trained_on=None,
    )
    assert bundle.green_threshold == pytest.approx(0.9)
    assert bundle.amber_threshold == pytest.approx(0.7)


# --------------------------------------------------------------------------- #
# Phase C: language-aware bundle loading
# --------------------------------------------------------------------------- #


def test_default_language_is_en_and_returns_existing_bundle() -> None:
    bundle = _bundles.load_bundle("rag.prose.multi_claim")
    assert bundle is not None
    assert bundle.language == _bundles.DEFAULT_LANGUAGE
    assert bundle.language == "en"


def test_load_bundle_with_explicit_en_matches_default() -> None:
    a = _bundles.load_bundle("rag.prose.multi_claim")
    b = _bundles.load_bundle("rag.prose.multi_claim", "en")
    # Same file → same content; cache may or may not return identical
    # object refs but values must match.
    assert a is not None and b is not None
    assert a.thresholds == b.thresholds
    assert a.fusion_weights == b.fusion_weights


def test_load_bundle_de_falls_back_to_en_with_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # No German bundle ships in this branch yet (Phase C deliverable);
    # the loader must fall back to the English one and emit
    # ``bundle_language_fallback`` exactly once per (class, lang) pair.
    caplog.set_level("WARNING")
    de_bundle = _bundles.load_bundle("rag.prose.multi_claim", "de")
    assert de_bundle is not None, (
        "German request must fall back to the English bundle, not return None."
    )
    # The bundle is the English fallback — language tag stays English so
    # the runtime can record `bundle_language_fallback` honestly while
    # still scoring the request.
    assert de_bundle.language == "en"

    # Second call must NOT emit the warning again — the loader keys the
    # warning by (class, language) so a steady stream of de requests
    # does not flood the log.
    caplog.clear()
    _bundles.load_bundle("rag.prose.multi_claim", "de")
    fallback_warnings = [
        r for r in caplog.records if "bundle_language_fallback" in r.getMessage()
    ]
    assert fallback_warnings == [], (
        "bundle_language_fallback must be one-shot per (class, language)."
    )


def test_calibration_path_uses_language_suffix_for_non_default() -> None:
    en_path = _bundles._calibration_path("rag.prose.multi_claim", "en")
    de_path = _bundles._calibration_path("rag.prose.multi_claim", "de")
    assert en_path.name == "calibration.rag_prose_multi_claim.json"
    assert de_path.name == "calibration.rag_prose_multi_claim.de.json"


def test_load_bundle_picks_up_de_artefact_when_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Drop a fake German bundle under the data directory and assert the
    loader picks it up over the English fallback. Mirrors the flow that
    Phase C will follow when ``scripts/calibrate_per_class.py --language de``
    writes the per-class artefacts."""

    fake_de = {
        "scoring_mode": "rag",
        "fusion_weights": {
            "calibrated": 0.4,
            "literal": 0.2,
            "nli": 0.4,
            "semantic_entropy": 0.0,
            "structured": 0.0,
        },
        "thresholds": {"green": 0.85, "amber": 0.55},
        "metric": "f1_at_best_threshold",
        "metric_value": 0.62,
        "nli_model_hint": "de",
        "trained_on": "data/corpus_classifier/german_translation/rag_prose_multi_claim.jsonl",
    }
    target = _bundles._DATA_DIR / "calibration.rag_prose_multi_claim.de.json"
    target.write_text(json.dumps(fake_de), encoding="utf-8")
    try:
        _bundles.reset_singleton_for_tests()
        bundle = _bundles.load_bundle("rag.prose.multi_claim", "de")
        assert bundle is not None
        assert bundle.language == "de"
        assert bundle.green_threshold == pytest.approx(0.85)
        assert bundle.amber_threshold == pytest.approx(0.55)
        assert bundle.nli_model_hint == "de"
    finally:
        target.unlink(missing_ok=True)
        _bundles.reset_singleton_for_tests()


def test_supported_languages_includes_de_and_en() -> None:
    langs = _bundles.supported_languages()
    assert "en" in langs
    assert "de" in langs
