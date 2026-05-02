"""Run a bounded real-data InfiniMem bootstrap over the serious-v1 stack."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from datasets import load_dataset

from latence_trace.memory.datasets import adapter_for, stitch_long_code_trajectory
from latence_trace.memory.labels import label_trajectory
from latence_trace.memory.ranker import features_from_scores, train_ranker
from latence_trace.memory.replay import replay_trace_signals
from latence_trace.memory.scorecard import build_scorecard, promotion_decision

DEFAULT_DATASETS = {
    "longmemeval": {
        "hf_id": "xiaowu0162/longmemeval-cleaned",
        "split": "longmemeval_s_cleaned",
        "adapter": "longmemeval",
    },
    "mtrag": {
        "hf_id": "mapauk/mtrag-dialogs-annotated-2026-04-27",
        "split": "train",
        "adapter": "mtrag",
    },
    "ragtruth": {
        "hf_id": "wandb/RAGTruth-processed",
        "split": "train",
        "adapter": "ragtruth",
    },
    "garage": {
        "hf_id": "AmazonScience/GaRAGe",
        "split": "train",
        "adapter": "garage",
    },
    "swe_agent": {
        "hf_id": "nebius/SWE-agent-trajectories",
        "split": "train",
        "adapter": "swe-agent-trajectories",
    },
    "swe_bench_verified": {
        "hf_id": "SWE-bench/SWE-bench_Verified",
        "split": "test",
        "adapter": "swe-bench-verified",
    },
    "tau_bench": {
        "hf_id": "jkazdan/taubench_traces_training_data",
        "split": "train",
        "adapter": "tau-bench",
    },
}


def load_real_trajectories(
    *,
    limit_per_dataset: int,
    limits: dict[str, int] | None = None,
    dataset_names: list[str] | None = None,
    max_steps_per_trajectory: int | None = None,
    long_code_stress_steps: list[int] | None = None,
    checkpoint_dir: Path | None = None,
    checkpoint_every: int = 50,
    progress: ProgressLogger | None = None,
) -> tuple[list[Any], list[dict[str, Any]]]:
    trajectories = []
    failures = []
    swe_rows: list[dict[str, Any]] = []
    selected = dataset_names or list(DEFAULT_DATASETS)
    for name in selected:
        config = DEFAULT_DATASETS[name]
        adapter = adapter_for(config["adapter"])
        target_limit = (limits or {}).get(name, limit_per_dataset)
        if progress is not None:
            progress.write(
                {
                    "event": "dataset_start",
                    "dataset": name,
                    "target_limit": target_limit,
                    "trajectory_count": len(trajectories),
                    "failure_count": len(failures),
                }
            )
        try:
            stream = load_dataset(config["hf_id"], split=config["split"], streaming=True)
            count = 0
            for row in stream:
                try:
                    if name == "swe_agent":
                        swe_rows.append(row)
                        trajectory = adapter(row, max_steps=max_steps_per_trajectory)
                    else:
                        trajectory = adapter(row)
                    trajectory = replay_trace_signals(trajectory)
                    trajectories.append(trajectory)
                    count += 1
                    if checkpoint_dir is not None and checkpoint_every > 0 and count % checkpoint_every == 0:
                        _write_checkpoint(
                            checkpoint_dir,
                            dataset=name,
                            count=count,
                            trajectories=trajectories,
                            failures=failures,
                        )
                    if progress is not None and checkpoint_every > 0 and count % checkpoint_every == 0:
                        progress.write(
                            {
                                "event": "dataset_progress",
                                "dataset": name,
                                "count": count,
                                "target_limit": target_limit,
                                "trajectory_count": len(trajectories),
                                "failure_count": len(failures),
                            }
                        )
                except Exception as exc:  # noqa: BLE001 - keep bootstrap moving
                    failures.append(
                        {
                            "dataset": name,
                            "stage": "adapt_or_replay",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                if count >= target_limit:
                    break
            if count == 0:
                failures.append({"dataset": name, "stage": "load", "error": "no usable rows"})
            if progress is not None:
                progress.write(
                    {
                        "event": "dataset_complete",
                        "dataset": name,
                        "count": count,
                        "target_limit": target_limit,
                        "trajectory_count": len(trajectories),
                        "failure_count": len(failures),
                    }
                )
        except Exception as exc:  # noqa: BLE001 - report access blockers
            failures.append(
                {
                    "dataset": name,
                    "stage": "load_dataset",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            if progress is not None:
                progress.write(
                    {
                        "event": "dataset_error",
                        "dataset": name,
                        "error": f"{type(exc).__name__}: {exc}",
                        "failure_count": len(failures),
                    }
                )
    for steps in long_code_stress_steps or []:
        if progress is not None:
            progress.write({"event": "long_code_stress_start", "target_steps": steps})
        try:
            trajectories.append(
                replay_trace_signals(
                    stitch_long_code_trajectory(
                        swe_rows,
                        target_steps=steps,
                        case_id=f"swe_agent_stress_{steps}",
                    )
                )
            )
            if progress is not None:
                progress.write(
                    {
                        "event": "long_code_stress_complete",
                        "target_steps": steps,
                        "trajectory_count": len(trajectories),
                    }
                )
        except Exception as exc:  # noqa: BLE001
            failures.append(
                {
                    "dataset": "swe_agent",
                    "stage": "long_code_stress",
                    "error": f"{type(exc).__name__}: {exc}",
                    "target_steps": steps,
                }
            )
            if progress is not None:
                progress.write(
                    {
                        "event": "long_code_stress_error",
                        "target_steps": steps,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    return trajectories, failures


class ProgressLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.started_at = time.time()

    def write(self, payload: dict[str, Any]) -> None:
        event = {
            "elapsed_s": round(time.time() - self.started_at, 3),
            **payload,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")


def train_from_trajectories(trajectories: list[Any]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    label_failures = []
    for trajectory in trajectories:
        try:
            state, labels, _ = label_trajectory(trajectory)
            label_by_id = {label.span_id: label for label in labels}
            for span in state.spans:
                label = label_by_id.get(span.id)
                if label is None:
                    continue
                rows.append(
                    {
                        "span_id": span.id,
                        "dataset": trajectory.dataset,
                        "case_id": trajectory.case_id,
                        "domain": trajectory.domain,
                        "turn_count": len(trajectory.turns),
                        "split": _split_for(trajectory.dataset, trajectory.case_id),
                        "label": label.label,
                        "target": {
                            "remove": 0.0,
                            "superseded": 0.2,
                            "demote": 0.35,
                            "keep": 0.75,
                            "anchor": 1.0,
                        }[label.label],
                        "features": features_from_scores(span.scores, domain=trajectory.domain),
                    }
                )
        except Exception as exc:  # noqa: BLE001
            label_failures.append(
                {
                    "dataset": trajectory.dataset,
                    "case_id": trajectory.case_id,
                    "stage": "label",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    if not rows:
        raise ValueError("No survival labels produced from real trajectories")
    return train_ranker(rows).model_dump(mode="json"), label_failures, rows


def _load_ranker_artifact(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "weights" not in payload or not isinstance(payload["weights"], dict):
        raise ValueError(f"Ranker artifact at {path} must contain a weights object")
    payload = dict(payload)
    payload.setdefault("diagnostics", {})
    payload["diagnostics"] = {
        **payload["diagnostics"],
        "source": "frozen_artifact",
        "artifact_path": str(path),
    }
    return payload


def run_bootstrap(
    limit_per_dataset: int,
    dataset_names: list[str] | None = None,
    *,
    limits: dict[str, int] | None = None,
    max_steps_per_trajectory: int | None = None,
    long_code_stress_steps: list[int] | None = None,
    artifact_dir: Path | None = None,
    checkpoint_every: int = 50,
    ranker_artifact: Path | None = None,
    ranker_train_split: str = "train",
    score_split: str = "eval",
    progress_log: Path | None = None,
    progress_every: int = 25,
) -> dict[str, Any]:
    progress = ProgressLogger(progress_log) if progress_log is not None else None
    if progress is not None:
        progress.write(
            {
                "event": "run_start",
                "limit_per_dataset": limit_per_dataset,
                "dataset_names": dataset_names or list(DEFAULT_DATASETS),
                "score_split": score_split,
                "ranker_train_split": ranker_train_split,
                "ranker_artifact": str(ranker_artifact) if ranker_artifact else None,
                "long_code_stress_steps": long_code_stress_steps or [],
            }
        )
    trajectories, load_failures = load_real_trajectories(
        limit_per_dataset=limit_per_dataset,
        limits=limits,
        dataset_names=dataset_names,
        max_steps_per_trajectory=max_steps_per_trajectory,
        long_code_stress_steps=long_code_stress_steps,
        checkpoint_dir=artifact_dir,
        checkpoint_every=checkpoint_every,
        progress=progress,
    )
    if progress is not None:
        progress.write(
            {
                "event": "load_complete",
                "trajectory_count": len(trajectories),
                "split_counts": _split_counts(trajectories),
                "failure_count": len(load_failures),
            }
        )
    if ranker_artifact is not None:
        ranker = _load_ranker_artifact(ranker_artifact)
        label_failures = []
        ranker_rows = []
        ranker_training_trajectories: list[Any] = []
    else:
        ranker_training_trajectories = [
            trajectory
            for trajectory in trajectories
            if _split_for(trajectory.dataset, trajectory.case_id) == ranker_train_split
        ]
        if not ranker_training_trajectories:
            ranker_training_trajectories = list(trajectories)
        if progress is not None:
            progress.write(
                {
                    "event": "ranker_training_start",
                    "training_trajectory_count": len(ranker_training_trajectories),
                    "training_split_counts": _split_counts(ranker_training_trajectories),
                }
            )
        ranker, label_failures, ranker_rows = train_from_trajectories(ranker_training_trajectories)
        ranker.setdefault("diagnostics", {})
        ranker["diagnostics"] = {
            **ranker["diagnostics"],
            "source": "in_run_split",
            "train_split": ranker_train_split,
        }
    if progress is not None:
        progress.write(
            {
                "event": "ranker_ready",
                "training_rows": len(ranker_rows),
                "diagnostics": ranker.get("diagnostics", {}),
                "label_failures": len(label_failures),
            }
        )
    score_trajectories = _score_trajectories(trajectories, score_split=score_split)
    if progress is not None:
        progress.write(
            {
                "event": "heldout_score_start",
                "score_split": score_split,
                "scored_trajectory_count": len(score_trajectories),
                "score_split_counts": _split_counts(score_trajectories),
            }
        )
    scorecard = build_scorecard(
        score_trajectories,
        learned_weights=ranker["weights"],
        progress_callback=progress.write if progress is not None else None,
        progress_every=progress_every,
    )
    decision = promotion_decision(scorecard)
    if progress is not None:
        progress.write(
            {
                "event": "heldout_score_complete",
                "promotion_decision": decision,
            }
        )
        progress.write(
            {
                "event": "all_score_start",
                "trajectory_count": len(trajectories),
            }
        )
    all_scorecard = build_scorecard(
        trajectories,
        learned_weights=ranker["weights"],
        progress_callback=progress.write if progress is not None else None,
        progress_every=progress_every,
    )
    if progress is not None:
        progress.write(
            {
                "event": "all_score_complete",
                "trace_memory_rule_based": all_scorecard["by_domain"],
            }
        )
    if artifact_dir is not None:
        _write_artifacts(
            artifact_dir=artifact_dir,
            trajectories=trajectories,
            ranker=ranker,
            ranker_rows=ranker_rows,
            scorecard=scorecard,
            failures=[*load_failures, *label_failures],
            checkpoint_every=checkpoint_every,
        )
    return {
        "dataset_configs": {
            name: DEFAULT_DATASETS[name]
            for name in (dataset_names or list(DEFAULT_DATASETS))
            if name in DEFAULT_DATASETS
        },
        "limit_per_dataset": limit_per_dataset,
        "limits": limits or {},
        "max_steps_per_trajectory": max_steps_per_trajectory,
        "long_code_stress_steps": long_code_stress_steps or [],
        "checkpoint_every": checkpoint_every,
        "split_counts": _split_counts(trajectories),
        "ranker_training_split_counts": _split_counts(ranker_training_trajectories),
        "score_split": score_split,
        "score_split_counts": _split_counts(score_trajectories),
        "trajectory_count": len(trajectories),
        "scored_trajectory_count": len(score_trajectories),
        "ranker": ranker,
        "ranker_rows": len(ranker_rows),
        "scorecard": scorecard,
        "all_scorecard": all_scorecard,
        "promotion_decision": decision,
        "failures": [*load_failures, *label_failures],
        "interpretation": _interpret(decision),
    }


def _score_trajectories(trajectories: list[Any], *, score_split: str) -> list[Any]:
    if score_split == "all":
        return list(trajectories)
    if score_split == "eval":
        selected = [
            trajectory
            for trajectory in trajectories
            if _split_for(trajectory.dataset, trajectory.case_id) != "train"
        ]
    else:
        selected = [
            trajectory
            for trajectory in trajectories
            if _split_for(trajectory.dataset, trajectory.case_id) == score_split
        ]
    return selected or list(trajectories)


def _interpret(decision: dict[str, Any]) -> str:
    if decision.get("approved_for_production_coupling"):
        return "Real bootstrap supports opt-in production coupling, pending larger full-dataset run."
    return (
        "Real bootstrap completed but does not approve production coupling yet. "
        "Use this as evidence/debug signal, then scale dataset limits and inspect failures."
    )


def _parse_dataset_names(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    names = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [name for name in names if name not in DEFAULT_DATASETS]
    if unknown:
        raise ValueError(f"Unknown dataset names: {unknown}. Known: {sorted(DEFAULT_DATASETS)}")
    return names


def _parse_limits(raw: str | None) -> dict[str, int] | None:
    if not raw:
        return None
    limits: dict[str, int] = {}
    for item in raw.split(","):
        if not item.strip():
            continue
        key, _, value = item.partition("=")
        if not key or not value:
            raise ValueError("Limits must use name=value pairs")
        if key not in DEFAULT_DATASETS:
            raise ValueError(f"Unknown dataset in limits: {key}")
        limits[key] = int(value)
    return limits


def _parse_steps(raw: str | None) -> list[int] | None:
    if not raw:
        return None
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def _write_artifacts(
    *,
    artifact_dir: Path,
    trajectories: list[Any],
    ranker: dict[str, Any],
    ranker_rows: list[dict[str, Any]],
    scorecard: dict[str, Any],
    failures: list[dict[str, Any]],
    checkpoint_every: int,
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    with (artifact_dir / "canonical_replayed.jsonl").open("w", encoding="utf-8") as handle:
        for trajectory in trajectories:
            handle.write(json.dumps(trajectory.model_dump(mode="json"), sort_keys=True) + "\n")
    (artifact_dir / "ranker.json").write_text(json.dumps(ranker, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (artifact_dir / "ranker_rows.jsonl").open("w", encoding="utf-8") as handle:
        for row in ranker_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    (artifact_dir / "scorecard.json").write_text(json.dumps(scorecard, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (artifact_dir / "failures.json").write_text(json.dumps(failures, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    split_rows = [
        {
            "dataset": trajectory.dataset,
            "case_id": trajectory.case_id,
            "domain": trajectory.domain,
            "turn_count": len(trajectory.turns),
            "split": _split_for(trajectory.dataset, trajectory.case_id),
        }
        for trajectory in trajectories
    ]
    with (artifact_dir / "splits.jsonl").open("w", encoding="utf-8") as handle:
        for row in split_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    (artifact_dir / "checkpoint_manifest.json").write_text(
        json.dumps(
            {
                "checkpoint_every": checkpoint_every,
                "trajectory_count": len(trajectories),
                "failure_count": len(failures),
                "split_counts": _counts(row["split"] for row in split_rows),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_checkpoint(
    checkpoint_dir: Path,
    *,
    dataset: str,
    count: int,
    trajectories: list[Any],
    failures: list[dict[str, Any]],
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"checkpoint_{dataset}_{count}.json"
    path.write_text(
        json.dumps(
            {
                "dataset": dataset,
                "count": count,
                "trajectory_count": len(trajectories),
                "failure_count": len(failures),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _split_for(dataset: str, case_id: str) -> str:
    digest = hashlib.sha256(f"{dataset}:{case_id}".encode()).hexdigest()
    bucket = int(digest[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "dev"
    return "test"


def _split_counts(trajectories: list[Any]) -> dict[str, int]:
    return _counts(_split_for(trajectory.dataset, trajectory.case_id) for trajectory in trajectories)


def _counts(items: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-per-dataset", type=int, default=2)
    parser.add_argument("--limits", default=None, help="Comma-separated per-dataset limits, e.g. ragtruth=100,swe_agent=20.")
    parser.add_argument("--datasets", default=None, help="Comma-separated subset of known dataset keys.")
    parser.add_argument("--max-steps-per-trajectory", type=int, default=None)
    parser.add_argument("--long-code-stress", default=None, help="Comma-separated stitched SWE-agent step targets, e.g. 100,250.")
    parser.add_argument("--artifact-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--ranker-artifact", type=Path, default=None, help="Frozen ranker.json to use without training on this run.")
    parser.add_argument("--ranker-train-split", default="train", choices=["train", "validation", "test"])
    parser.add_argument("--score-split", default="eval", choices=["eval", "train", "validation", "test", "all"])
    parser.add_argument("--progress-log", type=Path, default=None, help="JSONL file for live progress and partial metrics.")
    parser.add_argument("--progress-every", type=int, default=25, help="Emit partial metrics every N scored trajectories.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_bootstrap(
        limit_per_dataset=args.limit_per_dataset,
        dataset_names=_parse_dataset_names(args.datasets),
        limits=_parse_limits(args.limits),
        max_steps_per_trajectory=args.max_steps_per_trajectory,
        long_code_stress_steps=_parse_steps(args.long_code_stress),
        artifact_dir=args.artifact_dir,
        checkpoint_every=args.checkpoint_every,
        ranker_artifact=args.ranker_artifact,
        ranker_train_split=args.ranker_train_split,
        score_split=args.score_split,
        progress_log=args.progress_log,
        progress_every=args.progress_every,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "trajectory_count": report["trajectory_count"], "failures": len(report["failures"])}, sort_keys=True))


if __name__ == "__main__":
    main()
