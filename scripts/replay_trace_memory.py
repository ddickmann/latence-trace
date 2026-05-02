"""Attach TRACE replay signals to canonical TRACE Memory trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from latence_trace.memory.replay import replay_trace_signals
from latence_trace.memory.trajectory import coerce_trajectory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    args = parser.parse_args()
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.input_jsonl.open("r", encoding="utf-8") as source, args.output_jsonl.open(
        "w", encoding="utf-8"
    ) as dest:
        for line in source:
            if not line.strip():
                continue
            trajectory = replay_trace_signals(coerce_trajectory(json.loads(line)))
            dest.write(json.dumps(trajectory.model_dump(mode="json"), sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
