from __future__ import annotations

from scripts.eval_trace_memory import run_case, synthetic_cases


def test_eval_harness_reports_preservation_and_reduction() -> None:
    result = run_case(synthetic_cases()[0])

    assert result["token_reduction"] > 0.0
    assert result["exact_critical_preservation"] >= 0.5
    assert "label_counts" in result
    assert result["label_counts"].get("anchor", 0) >= 1
