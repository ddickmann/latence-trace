"""Generate TRACE Memory baseline, holdout, and ablation scorecards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from latence_trace.memory.replay import replay_trace_signals
from latence_trace.memory.scorecard import build_scorecard
from latence_trace.memory.trajectory import coerce_trajectory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--ranker-json", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    weights = None
    if args.ranker_json:
        artifact = json.loads(args.ranker_json.read_text(encoding="utf-8"))
        weights = artifact.get("baseline_ranker_weights") or artifact.get("weights")
    trajectories = []
    with args.input_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                trajectories.append(replay_trace_signals(coerce_trajectory(json.loads(line))))
    scorecard = build_scorecard(trajectories, learned_weights=weights)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(scorecard, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
