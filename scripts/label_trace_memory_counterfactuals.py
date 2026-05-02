"""Generate audited TRACE Memory counterfactual survival labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from latence_trace.memory.labels import label_trajectory
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
            trajectory = coerce_trajectory(json.loads(line))
            state, labels, hot_context = label_trajectory(trajectory)
            dest.write(
                json.dumps(
                    {
                        "dataset": trajectory.dataset,
                        "case_id": trajectory.case_id,
                        "domain": trajectory.domain,
                        "hot_context": hot_context,
                        "memory_state": state.model_dump(mode="json"),
                        "labels": [label.model_dump(mode="json") for label in labels],
                    },
                    sort_keys=True,
                )
                + "\n"
            )


if __name__ == "__main__":
    main()
