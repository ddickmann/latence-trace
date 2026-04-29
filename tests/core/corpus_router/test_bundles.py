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
