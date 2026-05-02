"""Normalize serious-v1 datasets into canonical TRACE Memory JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from latence_trace.memory.datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    args = parser.parse_args()
    trajectories = load_dataset(args.input, dataset=args.dataset)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for trajectory in trajectories:
            handle.write(json.dumps(trajectory.model_dump(mode="json"), sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
