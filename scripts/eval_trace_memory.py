"""Offline TRACE Memory evaluation harness.

The harness starts with rule-based labels. Public datasets and internal traces
can be normalized into JSONL rows with ``turns``; synthetic fixtures are used
when no path is supplied so CI can guard the core metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from latence_trace.memory.baselines import evaluate_baselines
from latence_trace.memory.datasets import load_dataset
from latence_trace.memory.labels import label_trajectory
from latence_trace.memory.models import MemoryPolicy
from latence_trace.memory.replay import replay_trace_signals
from latence_trace.memory.trajectory import CanonicalTrajectory, coerce_trajectory


def _tokens(text: str) -> int:
    return len(text.split())


def synthetic_cases() -> list[CanonicalTrajectory]:
    return [
        coerce_trajectory(
            {
                "id": "code_exact_critical",
                "domain": "code",
                "turns": [
                    {
                        "query_text": "Fix cache latency regression.",
                        "turn_text": "Always preserve public API CacheClient.get_many.",
                        "response_text": "Initial benchmark is 120 ms in src/cache.py.",
                        "raw_context": "src/cache.py contains CacheClient.get_many.",
                    },
                    {
                        "query_text": "Continue the fix.",
                        "turn_text": "The failed branch is resolved; keep the API unchanged.",
                        "response_text": "Updated src/cache.py and improved benchmark to 84 ms.",
                        "raw_context": "pytest tests/test_cache.py passed.",
                    },
                ],
                "critical_terms": ["CacheClient.get_many", "src/cache.py", "84 ms"],
            }
        ),
        coerce_trajectory(
            {
                "id": "rag_stale_evidence",
                "domain": "rag",
                "turns": [
                    {
                        "query_text": "What is the refund policy?",
                        "turn_text": "Old retrieved chunk says refunds take 30 days.",
                        "response_text": "Refunds take 30 days.",
                    },
                    {
                        "query_text": "Update with latest policy.",
                        "turn_text": "New policy supersedes the old one: refunds take 14 days.",
                        "response_text": "Refunds take 14 days.",
                    },
                ],
                "critical_terms": ["14 days"],
            }
        ),
    ]


def load_cases(path: Path | None, *, dataset: str | None = None) -> list[CanonicalTrajectory]:
    if path is None:
        return synthetic_cases()
    if dataset:
        return load_dataset(path, dataset=dataset)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return [coerce_trajectory(row) for row in rows]


def run_case(case: dict[str, Any] | CanonicalTrajectory) -> dict[str, Any]:
    trajectory = replay_trace_signals(
        case if isinstance(case, CanonicalTrajectory) else coerce_trajectory(case)
    )
    policy = MemoryPolicy(hot_token_budget=40, warm_token_budget=160)
    state, labels, hot_context = label_trajectory(trajectory, policy=policy)
    full_history_tokens = _tokens(trajectory.full_history_text())
    hot_tokens = _tokens(hot_context)
    critical_terms = trajectory.critical_terms
    preserved = [term for term in critical_terms if term in hot_context]
    exact_preservation = len(preserved) / max(1, len(critical_terms))
    token_reduction = 1.0 - (hot_tokens / max(1, full_history_tokens))
    baselines = evaluate_baselines(trajectory, policy=policy)
    return {
        "id": trajectory.case_id,
        "dataset": trajectory.dataset,
        "domain": trajectory.domain,
        "full_history_tokens": full_history_tokens,
        "hot_tokens": hot_tokens,
        "token_reduction": round(token_reduction, 4),
        "exact_critical_preservation": round(exact_preservation, 4),
        "preserved_terms": preserved,
        "label_counts": _counts(labels),
        "baseline_comparison": baselines,
        "span_count": len(state.spans),
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(results)
    return {
        "case_count": count,
        "mean_token_reduction": round(sum(row["token_reduction"] for row in results) / max(1, count), 4),
        "mean_exact_critical_preservation": round(
            sum(row["exact_critical_preservation"] for row in results) / max(1, count),
            4,
        ),
        "results": results,
    }


def _counts(labels: list[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in labels:
        label = row.label if hasattr(row, "label") else row["label"]
        counts[label] = counts.get(label, 0) + 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    report = summarize([run_case(case) for case in load_cases(args.input_jsonl, dataset=args.dataset)])
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
