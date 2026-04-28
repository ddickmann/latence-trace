"""Tests for per-tenant threshold routing + hot reload (Plan B2)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from latence_trace.core import thresholds as mod


def _write(path: Path, green: float, amber: float) -> None:
    payload = {
        "schema_version": 1,
        "headline": "groundedness_v2",
        "precision_target": 0.75,
        "nli_enabled": True,
        "pair_count": 0,
        "strata": {"default": {"green_min": green, "amber_min": amber}},
        "source": str(path),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture()
def isolated_thresholds(tmp_path, monkeypatch):
    # Clear both caches and reset the global env.
    mod._POLICY_CACHE = None
    mod._POLICY_CACHE_MTIME = None
    mod._TENANT_CACHE.clear()
    monkeypatch.setenv("VOYAGER_GROUNDEDNESS_THRESHOLDS_PATH", str(tmp_path / "thresholds.json"))
    monkeypatch.setenv(mod._TENANT_DIR_ENV, str(tmp_path))
    _write(tmp_path / "thresholds.json", 0.80, 0.60)
    yield tmp_path


def test_default_policy_loads_once(isolated_thresholds):
    policy = mod.get_risk_band_policy()
    assert policy.threshold_for(None)["green_min"] == 0.80
    # second call hits the cache
    policy2 = mod.get_risk_band_policy()
    assert policy is policy2


def test_per_tenant_policy_overrides_default(isolated_thresholds, tmp_path):
    _write(tmp_path / "thresholds.acme.json", 0.90, 0.70)
    default = mod.get_risk_band_policy_for_tenant(None)
    acme = mod.get_risk_band_policy_for_tenant("acme")
    assert default.threshold_for(None)["green_min"] == 0.80
    assert acme.threshold_for(None)["green_min"] == 0.90


def test_tenant_policy_hot_reload(isolated_thresholds, tmp_path):
    _write(tmp_path / "thresholds.acme.json", 0.90, 0.70)
    first = mod.get_risk_band_policy_for_tenant("acme")
    # Ensure mtime moves by at least 1s so the cache invalidates even on
    # filesystems with coarse mtime granularity.
    time.sleep(1.05)
    _write(tmp_path / "thresholds.acme.json", 0.85, 0.65)
    reloaded = mod.get_risk_band_policy_for_tenant("acme")
    assert first.threshold_for(None)["green_min"] == 0.90
    assert reloaded.threshold_for(None)["green_min"] == 0.85


def test_unknown_tenant_falls_back_to_global(isolated_thresholds):
    policy = mod.get_risk_band_policy_for_tenant("unknown-tenant-id")
    assert policy.threshold_for(None)["green_min"] == 0.80


def test_global_policy_hot_reload(isolated_thresholds, tmp_path):
    first = mod.get_risk_band_policy()
    time.sleep(1.05)
    _write(tmp_path / "thresholds.json", 0.75, 0.55)
    reloaded = mod.get_risk_band_policy()
    assert first.threshold_for(None)["green_min"] == 0.80
    assert reloaded.threshold_for(None)["green_min"] == 0.75
