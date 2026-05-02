from __future__ import annotations

import json

from latence_trace.memory.baselines import evaluate_baselines
from latence_trace.memory.datasets import (
    adapt_garage,
    adapt_longmemeval,
    adapt_mtrag,
    adapt_ragtruth,
    adapt_swe_agent,
    adapt_swe_bench_verified,
    adapt_tau_bench,
    stitch_long_code_trajectory,
)
from latence_trace.memory.labels import label_trajectory
from latence_trace.memory.ranker import train_ranker
from latence_trace.memory.replay import replay_trace_signals
from latence_trace.memory.scorecard import build_scorecard, promotion_decision
from latence_trace.memory.trajectory import coerce_trajectory
from scripts.eval_trace_memory import run_case, synthetic_cases
from scripts.train_trace_memory_baseline import collect_training_rows


def test_canonical_schema_accepts_legacy_and_records_critical_terms() -> None:
    trajectory = coerce_trajectory(
        {
            "id": "case-1",
            "domain": "code",
            "turns": [{"turn_text": "Keep src/app.py and value 120 ms."}],
            "critical_terms": ["src/app.py", "120 ms"],
        }
    )

    assert trajectory.case_id == "case-1"
    assert trajectory.domain == "code"
    assert "src/app.py" in trajectory.critical_terms


def test_all_serious_v1_adapters_emit_canonical_trajectories() -> None:
    rows = [
        adapt_longmemeval({"id": "l", "messages": [{"content": "My dog is Ada."}], "question": "Dog?"}),
        adapt_mtrag({"id": "m", "turns": [{"query": "Refund?", "contexts": ["Refunds take 14 days."]}]}),
        adapt_ragtruth({"id": "r", "prompt": "Q", "response": "A", "source": "A is supported."}),
        adapt_garage({"id": "g", "question": "Q", "answer": "A", "passages": ["A evidence."]}),
        adapt_swe_agent(
            {
                "id": "sa",
                "problem_statement": "Fix src/app.py",
                "trajectory": [{"thought": "Run pytest", "observation": "pytest failed"}],
            }
        ),
        adapt_swe_bench_verified(
            {
                "instance_id": "sv",
                "problem_statement": "Fix src/app.py",
                "patch": "diff --git a/src/app.py b/src/app.py",
            }
        ),
        adapt_tau_bench(
            {
                "task_id": "t",
                "turns": [{"user": "Book order", "tool_outputs": [{"content": "order_id=123"}]}],
            }
        ),
    ]

    assert {row.dataset for row in rows} == {
        "LongMemEval",
        "MTRAG",
        "RAGTruth",
        "GaRAGe",
        "SWE-agent trajectories",
        "SWE-bench Verified",
        "tau-bench",
    }
    assert all(row.turns for row in rows)


def test_domain_exact_terms_and_long_code_stitching_are_preserved() -> None:
    tau = adapt_tau_bench(
        {
            "task_id": "tau",
            "turns": [
                {
                    "user": "Never ship before payment clears.",
                    "tool_outputs": [{"content": '{"order_id": "ORD-123", "status": "paid"}'}],
                }
            ],
        }
    )
    code = stitch_long_code_trajectory(
        [
            {
                "id": "sa1",
                "problem_statement": "Fix src/cache.py and test_cache_hit",
                "trajectory": [{"thought": "Run pytest tests/test_cache.py::test_cache_hit", "observation": "AssertionError"}]
                * 120,
            }
        ],
        target_steps=100,
    )

    assert "ORD-123" in tau.critical_terms
    assert "src/cache.py" in code.critical_terms
    assert len(code.turns) == 100
    assert code.turns[0].metadata["stitched_source_case_id"] == "sa1"


def test_tau_tool_structured_facts_are_preserved_under_hot_budget() -> None:
    trajectory = replay_trace_signals(
        adapt_tau_bench(
            {
                "task_id": "tau",
                "messages": [
                    {"role": "user", "content": "Cancel reservation ORD-123."},
                    {
                        "role": "tool",
                        "content": (
                            '{"reservation_id": "ORD-123", "status": "cancelled", '
                            '"payment_methods": {"gift_card_555": {"amount": 42, "id": "gift_card_555"}}, '
                            '"flights": [{"flight_number": "HAT123", "scheduled_departure_time_est": "13:00:00"}]}'
                        ),
                    },
                ],
            }
        )
    )

    metrics = evaluate_baselines(trajectory)
    preserved = metrics["trace_memory_rule_based"]["preserved_terms"]

    assert metrics["trace_memory_rule_based"]["exact_critical_preservation"] == 1.0
    assert {"ORD-123", "cancelled", "gift_card_555", "HAT123"} <= set(preserved)


def test_replay_labels_ranker_scorecard_and_promotion_are_wired() -> None:
    trajectory = replay_trace_signals(synthetic_cases()[0])
    state, labels, hot_context = label_trajectory(trajectory)
    rows = collect_training_rows([trajectory])
    artifact = train_ranker(rows)
    scorecard = build_scorecard([trajectory], learned_weights=artifact.weights)
    decision = promotion_decision(scorecard)
    eval_row = run_case(trajectory)

    assert state.spans
    assert labels
    assert "src/cache.py" in hot_context
    assert artifact.training_rows == len(rows)
    assert artifact.diagnostics["domain_counts"]["code"] > 0
    assert scorecard["case_count"] == 1
    assert "matched_budget_baselines" in scorecard
    assert "long_horizon" in scorecard
    assert decision["recommended_mode"] in {"shadow", "opt_in_production"}
    assert eval_row["baseline_comparison"]["trace_memory_rule_based"]["status"] == "measured"


def test_scorecard_is_json_serializable() -> None:
    trajectory = replay_trace_signals(synthetic_cases()[1])
    scorecard = build_scorecard([trajectory])

    assert json.loads(json.dumps(scorecard))["case_count"] == 1
