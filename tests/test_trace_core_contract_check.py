from __future__ import annotations

from scripts.trace_core_contract_check import run_contract_check


def test_trace_core_contract_manifest_matches_product_surface() -> None:
    assert run_contract_check() == []
