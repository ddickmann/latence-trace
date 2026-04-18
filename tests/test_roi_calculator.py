"""Tests for the pilot ROI calculator (commercial/pilot-kit).

The calculator is shipped as a stand-alone, dependency-free script.
We import it dynamically from the commercial directory so the test
exercises the same source the AE / SE team uses.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType

import pytest

CALC_PATH = (
    Path(__file__).resolve().parent.parent
    / "commercial"
    / "pilot-kit"
    / "roi-calculator.py"
)


def _load_calculator() -> ModuleType:
    spec = importlib.util.spec_from_file_location("roi_calculator", CALC_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # ``dataclass`` resolves PEP 563 string annotations against the module
    # registered in ``sys.modules``; register the module before executing.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def calc() -> ModuleType:
    return _load_calculator()


def test_worked_example_matches_readme(calc: ModuleType) -> None:
    """Reproduces the mid-market RAG worked example from the README.

    Avoiding silent drift between docs and code is the whole point of
    keeping this test."""
    inputs = calc.RoiInputs(
        queries=12_000_000,
        hallucination_rate=0.04,
        cost_per_bad_answer=25.0,
        reduction=0.70,
        false_positive_rate=0.02,
        cost_per_review=1.50,
        subscription_fee=120_000.0,
        integration_cost=40_000.0,
    )
    result = calc.compute_roi(inputs)
    assert result.avoided_losses == pytest.approx(8_400_000.0)
    assert result.false_positive_cost == pytest.approx(360_000.0)
    assert result.net_annual_benefit == pytest.approx(8_000_000.0)
    assert result.roi_percent == pytest.approx(6_566.6666666, rel=1e-6)
    expected_payback = 120_000.0 / (8_000_000.0 / 12.0)
    assert result.payback_months == pytest.approx(expected_payback, rel=1e-9)


def test_invalid_rate_rejected(calc: ModuleType) -> None:
    bad = calc.RoiInputs(
        queries=10.0,
        hallucination_rate=1.5,  # invalid
        cost_per_bad_answer=10.0,
        reduction=0.5,
        false_positive_rate=0.01,
        cost_per_review=1.0,
        subscription_fee=10_000.0,
        integration_cost=0.0,
    )
    with pytest.raises(ValueError):
        calc.compute_roi(bad)


def test_zero_subscription_fee_rejected(calc: ModuleType) -> None:
    bad = calc.RoiInputs(
        queries=10.0,
        hallucination_rate=0.1,
        cost_per_bad_answer=10.0,
        reduction=0.5,
        false_positive_rate=0.01,
        cost_per_review=1.0,
        subscription_fee=0.0,
        integration_cost=0.0,
    )
    with pytest.raises(ValueError):
        calc.compute_roi(bad)


def test_negative_net_benefit_yields_infinite_payback(calc: ModuleType) -> None:
    inputs = calc.RoiInputs(
        queries=1_000.0,
        hallucination_rate=0.001,
        cost_per_bad_answer=1.0,
        reduction=0.5,
        false_positive_rate=0.5,
        cost_per_review=10.0,
        subscription_fee=100_000.0,
        integration_cost=10_000.0,
    )
    result = calc.compute_roi(inputs)
    assert result.net_annual_benefit < 0
    assert math.isinf(result.payback_months)


def test_cli_emits_json(monkeypatch, capsys, calc: ModuleType) -> None:
    rc = calc.main(
        [
            "--queries",
            "12000000",
            "--hallucination-rate",
            "0.04",
            "--cost-per-bad-answer",
            "25",
            "--reduction",
            "0.70",
            "--false-positive-rate",
            "0.02",
            "--cost-per-review",
            "1.5",
            "--subscription-fee",
            "120000",
            "--integration-cost",
            "40000",
            "--format",
            "json",
        ]
    )
    assert rc == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["avoided_losses"] == pytest.approx(8_400_000.0)
    assert payload["net_annual_benefit"] == pytest.approx(8_000_000.0)
    assert payload["inputs"]["queries"] == 12_000_000
