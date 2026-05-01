"""Fit an offline LM-AggreFact calibration artifact for Granite parity.

The script consumes streamed rows produced by ``bench_granite_parity.py`` and
selects a production-safe threshold rule using only TRACE output features. It
does not call the live service and it does not read LM-AggreFact test labels
unless the caller explicitly passes a test rows file for evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence


RAG_ORDER = (
    "AggreFact-CNN",
    "AggreFact-XSum",
    "ClaimVerify",
    "ExpertQA",
    "FactCheck-GPT",
    "LFQA",
    "RAGTruth",
    "Reveal",
    "TofuEval-MediaS",
    "TofuEval-MeetB",
    "Wice",
)

GRANITE_TARGET_MACRO = 0.765
SCORE_FIELDS = (
    "score",
    "runtime_decision.score",
    "runtime_decision.head_score",
    "score_channels.groundedness_v2",
    "score_channels.primary",
    "scores.groundedness_v2",
)
GROUP_FIELDS: tuple[Optional[str], ...] = (
    None,
    "runtime_decision.class_key",
    "corpus_route.corpus_type",
)


@dataclass(frozen=True)
class Row:
    dataset: str
    row_id: str
    supported: bool
    payload: Mapping[str, Any]


def _nested_get(payload: Mapping[str, Any], path: str) -> Any:
    current: Any = payload
    for part in path.split("."):
        if not isinstance(current, Mapping):
            return None
        current = current.get(part)
    return current


def _output_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    audit = row.get("audit") if isinstance(row.get("audit"), Mapping) else {}
    return {
        "score": row.get("score"),
        "runtime_decision": audit.get("runtime_decision"),
        "corpus_route": audit.get("corpus_route"),
        "score_channels": audit.get("score_channels"),
        "scores": audit.get("scores"),
        "band": audit.get("band"),
    }


def _value(row: Row, field: str) -> Optional[float]:
    payload = _output_payload(row.payload)
    value = _nested_get(payload, field)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _group(row: Row, field: Optional[str]) -> str:
    if field is None:
        return "global"
    if field == "dataset":
        return row.dataset
    payload = _output_payload(row.payload)
    return str(_nested_get(payload, field) or "missing")


def _load_rows(path: Path) -> list[Row]:
    rows: list[Row] = []
    seen: set[tuple[str, str]] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            key = (str(payload.get("dataset")), str(payload.get("row_id")))
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                Row(
                    dataset=key[0],
                    row_id=key[1],
                    supported=bool(payload.get("supported")),
                    payload=payload,
                )
            )
    return rows


def _balanced_accuracy(rows: Sequence[tuple[Row, Optional[bool]]]) -> dict[str, Any]:
    supported_total = unsupported_total = 0
    supported_correct = unsupported_correct = 0
    errors = 0
    for row, pred in rows:
        if pred is None:
            errors += 1
            continue
        if row.supported:
            supported_total += 1
            supported_correct += int(pred)
        else:
            unsupported_total += 1
            unsupported_correct += int(not pred)
    supported_recall = supported_correct / supported_total if supported_total else None
    unsupported_recall = unsupported_correct / unsupported_total if unsupported_total else None
    bacc = (
        (supported_recall + unsupported_recall) / 2.0
        if supported_recall is not None and unsupported_recall is not None
        else None
    )
    return {
        "n": len(rows),
        "errors": errors,
        "supported_total": supported_total,
        "unsupported_total": unsupported_total,
        "supported_recall": supported_recall,
        "unsupported_recall": unsupported_recall,
        "balanced_accuracy": bacc,
    }


def _macro_bacc(rows: Sequence[Row], predictions: Mapping[tuple[str, str], Optional[bool]]) -> dict[str, Any]:
    datasets: dict[str, Any] = {}
    values: list[float] = []
    for dataset in sorted({row.dataset for row in rows}):
        pairs = [
            (row, predictions.get((row.dataset, row.row_id)))
            for row in rows
            if row.dataset == dataset
        ]
        metric = _balanced_accuracy(pairs)
        datasets[dataset] = metric
        if metric["balanced_accuracy"] is not None:
            values.append(float(metric["balanced_accuracy"]))
    overall = _balanced_accuracy(
        [(row, predictions.get((row.dataset, row.row_id))) for row in rows]
    )
    return {
        "overall": overall,
        "macro_balanced_accuracy": statistics.fmean(values) if values else None,
        "datasets": datasets,
    }


def _thresholds(values: Sequence[float], *, max_points: int = 201) -> list[float]:
    unique = sorted(set(values))
    if not unique:
        return [0.5]
    if len(unique) <= max_points:
        return unique + [max(unique) + 1e-9]
    indices = {
        round(i * (len(unique) - 1) / float(max_points - 1))
        for i in range(max_points)
    }
    sampled = [unique[int(idx)] for idx in sorted(indices)]
    return sampled + [max(unique) + 1e-9]


def _fit_threshold(rows: Sequence[Row], score_field: str) -> Optional[float]:
    scored = [(row, _value(row, score_field)) for row in rows]
    usable = [(row, value) for row, value in scored if value is not None]
    if not usable:
        return None
    best_threshold: Optional[float] = None
    best = -1.0
    for threshold in _thresholds([float(value) for _, value in usable]):
        preds = {
            (row.dataset, row.row_id): bool(float(value) >= threshold)
            for row, value in usable
        }
        metric = _macro_bacc([row for row, _ in usable], preds)
        score = float(metric["macro_balanced_accuracy"] or -1.0)
        if score > best:
            best = score
            best_threshold = float(threshold)
    return best_threshold


def _fit_candidate(
    train: Sequence[Row],
    *,
    score_field: str,
    group_field: Optional[str],
) -> Optional[dict[str, Any]]:
    fallback = _fit_threshold(train, score_field)
    if fallback is None:
        return None
    thresholds: dict[str, float] = {}
    if group_field is not None:
        groups = sorted({_group(row, group_field) for row in train})
        for group in groups:
            subset = [row for row in train if _group(row, group_field) == group]
            if len(subset) < 20:
                continue
            threshold = _fit_threshold(subset, score_field)
            if threshold is not None:
                thresholds[group] = threshold
    return {
        "kind": "threshold_rule",
        "score_field": score_field,
        "group_field": group_field,
        "fallback_threshold": fallback,
        "thresholds": thresholds,
        "production_safe": group_field != "dataset",
    }


def _predict(rows: Sequence[Row], candidate: Mapping[str, Any]) -> dict[tuple[str, str], Optional[bool]]:
    score_field = str(candidate["score_field"])
    group_field = candidate.get("group_field")
    fallback = float(candidate["fallback_threshold"])
    thresholds = candidate.get("thresholds") if isinstance(candidate.get("thresholds"), Mapping) else {}
    predictions: dict[tuple[str, str], Optional[bool]] = {}
    for row in rows:
        value = _value(row, score_field)
        if value is None:
            predictions[(row.dataset, row.row_id)] = None
            continue
        group = _group(row, str(group_field) if group_field else None)
        threshold = float(thresholds.get(group, fallback))
        predictions[(row.dataset, row.row_id)] = bool(value >= threshold)
    return predictions


def _split_train_validation(
    rows: Sequence[Row],
    *,
    holdout_fraction: float,
    seed: int,
) -> tuple[list[Row], list[Row]]:
    rng = random.Random(seed)
    train: list[Row] = []
    validation: list[Row] = []
    buckets: dict[tuple[str, bool], list[Row]] = {}
    for row in rows:
        buckets.setdefault((row.dataset, row.supported), []).append(row)
    for bucket in buckets.values():
        shuffled = list(bucket)
        rng.shuffle(shuffled)
        n_val = max(1, int(round(len(shuffled) * holdout_fraction))) if len(shuffled) >= 4 else 0
        validation.extend(shuffled[:n_val])
        train.extend(shuffled[n_val:])
    return train, validation


def _stored_predictions(rows: Sequence[Row]) -> dict[tuple[str, str], Optional[bool]]:
    predictions: dict[tuple[str, str], Optional[bool]] = {}
    for row in rows:
        value = row.payload.get("predicted_supported")
        predictions[(row.dataset, row.row_id)] = value if isinstance(value, bool) else None
    return predictions


def _passes_dataset_guardrails(
    candidate_metric: Mapping[str, Any],
    baseline_metric: Mapping[str, Any],
    *,
    min_dataset_bacc: float,
    min_dataset_delta: float,
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    candidate_datasets = candidate_metric.get("datasets")
    baseline_datasets = baseline_metric.get("datasets")
    if not isinstance(candidate_datasets, Mapping):
        return False, ["candidate has no per-dataset metrics"]
    if not isinstance(baseline_datasets, Mapping):
        baseline_datasets = {}
    for dataset, metric in candidate_datasets.items():
        if not isinstance(metric, Mapping):
            continue
        bacc = metric.get("balanced_accuracy")
        if bacc is None:
            continue
        bacc_f = float(bacc)
        if bacc_f < min_dataset_bacc:
            failures.append(
                f"{dataset}: balanced_accuracy {bacc_f:.3f} < min_dataset_bacc {min_dataset_bacc:.3f}"
            )
        baseline = baseline_datasets.get(dataset)
        if isinstance(baseline, Mapping) and baseline.get("balanced_accuracy") is not None:
            baseline_bacc = float(baseline["balanced_accuracy"])
            if bacc_f < baseline_bacc + min_dataset_delta:
                failures.append(
                    f"{dataset}: balanced_accuracy {bacc_f:.3f} < baseline {baseline_bacc:.3f} "
                    f"+ min_dataset_delta {min_dataset_delta:.3f}"
                )
    return not failures, failures


def fit_calibrator(
    rows: Sequence[Row],
    *,
    holdout_fraction: float,
    seed: int,
    min_dataset_bacc: float = 0.0,
    min_dataset_delta: float = -1.0,
) -> dict[str, Any]:
    train, validation = _split_train_validation(
        rows,
        holdout_fraction=holdout_fraction,
        seed=seed,
    )
    baseline_train = _macro_bacc(train, _stored_predictions(train))
    baseline_validation = _macro_bacc(validation, _stored_predictions(validation))
    candidates: list[dict[str, Any]] = []
    for score_field in SCORE_FIELDS:
        for group_field in GROUP_FIELDS:
            candidate = _fit_candidate(train, score_field=score_field, group_field=group_field)
            if candidate is None:
                continue
            val_metric = _macro_bacc(validation, _predict(validation, candidate))
            train_metric = _macro_bacc(train, _predict(train, candidate))
            candidate["train"] = train_metric
            candidate["validation"] = val_metric
            passes, failures = _passes_dataset_guardrails(
                val_metric,
                baseline_validation,
                min_dataset_bacc=min_dataset_bacc,
                min_dataset_delta=min_dataset_delta,
            )
            candidate["guardrails"] = {
                "passed": passes,
                "failures": failures,
                "min_dataset_bacc": min_dataset_bacc,
                "min_dataset_delta": min_dataset_delta,
            }
            candidates.append(candidate)
    if not candidates:
        raise SystemExit("no calibration candidate could be fit")
    guarded = [candidate for candidate in candidates if candidate["guardrails"]["passed"]]
    selection_pool = guarded or candidates
    selection_pool.sort(
        key=lambda item: (
            float((item["validation"].get("macro_balanced_accuracy") or -1.0)),
            bool(item.get("production_safe")),
        ),
        reverse=True,
    )
    selected = selection_pool[0]
    return {
        "schema": "granite_parity_calibrator.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target_macro_bacc": GRANITE_TARGET_MACRO,
        "rows": len(rows),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "guardrail_selection": {
            "min_dataset_bacc": min_dataset_bacc,
            "min_dataset_delta": min_dataset_delta,
            "guarded_candidates": len(guarded),
            "fallback_to_unguarded": not bool(guarded),
        },
        "baseline_train": baseline_train,
        "baseline_validation": baseline_validation,
        "source_rows_sha256": hashlib.sha256(
            "\n".join(f"{row.dataset}\t{row.row_id}" for row in rows).encode("utf-8")
        ).hexdigest(),
        "selected_candidate": {
            key: selected[key]
            for key in (
                "kind",
                "score_field",
                "group_field",
                "fallback_threshold",
                "thresholds",
                "production_safe",
                "guardrails",
            )
        },
        "selected_train": selected["train"],
        "selected_validation": selected["validation"],
        "candidates": candidates,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--holdout-fraction", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=20260501)
    parser.add_argument(
        "--min-dataset-bacc",
        type=float,
        default=0.0,
        help="Reject candidates whose validation balanced accuracy is below this for any dataset.",
    )
    parser.add_argument(
        "--min-dataset-delta",
        type=float,
        default=-1.0,
        help=(
            "Reject candidates whose validation balanced accuracy is below the stored-decision "
            "baseline plus this delta for any dataset. Negative values allow bounded regressions."
        ),
    )
    args = parser.parse_args(argv)

    rows = _load_rows(args.rows)
    if not rows:
        raise SystemExit("no rows loaded")
    artifact = fit_calibrator(
        rows,
        holdout_fraction=args.holdout_fraction,
        seed=args.seed,
        min_dataset_bacc=args.min_dataset_bacc,
        min_dataset_delta=args.min_dataset_delta,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / "granite_parity_calibrator.json"
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    selected = artifact["selected_candidate"]
    validation = artifact["selected_validation"]
    print(f"rows: {len(rows)}")
    print(f"artifact: {path}")
    print(
        "selected: field={field} group={group} production_safe={safe}".format(
            field=selected["score_field"],
            group=selected.get("group_field"),
            safe=selected["production_safe"],
        )
    )
    print(
        "validation_macro_bacc: {:.3f}".format(
            float(validation.get("macro_balanced_accuracy") or 0.0)
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
