"""Train a tiny baseline ranker from TRACE Memory rule labels.

This script is intentionally modest: it refuses to "train" without labels
from the counterfactual/rule-label harness and produces a comparison report.
Learned production hazard models should replace this only after real replay
labels exist.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from latence_trace.memory.baselines import evaluate_baselines
from latence_trace.memory.labels import label_trajectory
from latence_trace.memory.ranker import features_from_scores, train_ranker
from latence_trace.memory.replay import replay_trace_signals
from scripts.eval_trace_memory import load_cases


def collect_training_rows(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        trajectory = replay_trace_signals(case)
        state, labels, _ = label_trajectory(trajectory)
        label_by_id = {label.span_id: label for label in labels}
        for span in state.spans:
            label = label_by_id.get(span.id)
            if label is None:
                continue
            rows.append(
                {
                    "span_id": span.id,
                    "domain": trajectory.domain,
                    "turn_count": len(trajectory.turns),
                    "features": features_from_scores(span.scores, domain=trajectory.domain),
                    "label": label.label,
                    "target": {"remove": 0.0, "demote": 0.35, "superseded": 0.2, "keep": 0.75, "anchor": 1.0}[label.label],
                }
            )
    if not rows:
        raise ValueError("No rule labels available; run the eval harness first or provide labeled cases.")
    return rows


def train_baseline_ranker(rows: list[dict[str, Any]]) -> dict[str, float]:
    return train_ranker(rows).weights


def compare_baselines(cases: list[dict[str, Any]]) -> dict[str, Any]:
    rows = collect_training_rows(cases)
    weights = train_baseline_ranker(rows)
    aggregate: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        for name, metrics in evaluate_baselines(replay_trace_signals(case), learned_weights=weights).items():
            aggregate.setdefault(name, []).append(metrics)
    return {
        name: {
            "status": "measured",
            "mean_token_reduction": round(sum(row["token_reduction"] for row in values) / len(values), 4),
            "mean_exact_critical_preservation": round(
                sum(row["exact_critical_preservation"] for row in values) / len(values),
                4,
            ),
            "token_reduction": round(sum(row["token_reduction"] for row in values) / len(values), 4),
            "exact_critical_preservation": round(
                sum(row["exact_critical_preservation"] for row in values) / len(values),
                4,
            ),
        }
        for name, values in aggregate.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    cases = load_cases(args.input_jsonl, dataset=args.dataset)
    rows = collect_training_rows(cases)
    report = {
        "training_rows": len(rows),
        "label_counts": _counts(row["label"] for row in rows),
        "baseline_ranker_weights": train_baseline_ranker(rows),
        "baseline_comparison": compare_baselines(cases),
        "production_note": (
            "This is a label-readiness baseline, not the production hazard model. "
            "Promotion still requires replayed counterfactual labels."
        ),
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def _counts(labels: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return counts


if __name__ == "__main__":
    main()
