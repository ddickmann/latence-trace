from __future__ import annotations

from scripts.eval_trace_memory import synthetic_cases
from scripts.train_trace_memory_baseline import (
    collect_training_rows,
    compare_baselines,
    train_baseline_ranker,
)


def test_baseline_training_requires_and_uses_rule_labels() -> None:
    rows = collect_training_rows(synthetic_cases())
    weights = train_baseline_ranker(rows)
    comparison = compare_baselines(synthetic_cases())

    assert rows
    assert "exact_critical" in weights
    assert "is_code_domain" in weights
    assert "is_rag_or_search_domain" in weights
    assert comparison["trace_memory_rule_based"]["status"] == "measured"
    assert comparison["sliding_window"]["exact_critical_preservation"] == 1.0
