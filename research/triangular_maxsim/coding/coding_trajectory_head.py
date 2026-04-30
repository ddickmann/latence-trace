"""Train and evaluate the agentic coding trajectory head.

The trajectory head uses native trace facts (files, symbols, command outcomes,
patch state, and event order) rather than relying only on MaxSim-style
similarity. The legacy transcript score banks are still reported as diagnostics,
but promotion is gated on the trajectory-native held-out banks.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.triangular_maxsim.coding.trajectory_native_data import (
    DEFAULT_OUT_DIR as NATIVE_OUT_DIR,
    generate_rows as generate_native_rows,
    write_banks as write_native_banks,
)


ARTIFACTS = ROOT / "research/triangular_maxsim/coding/artifacts"

DEFAULT_BANKS = {
    "native_train": NATIVE_OUT_DIR / "trajectory_native_train.json",
    "native_val": NATIVE_OUT_DIR / "trajectory_native_val.json",
    "native_test": NATIVE_OUT_DIR / "trajectory_native_test.json",
    "transcripts_v1": ARTIFACTS / "trace_stabilization_transcripts_v1.json",
    "transcripts_v2": ARTIFACTS / "trace_stabilization_transcripts_v2.json",
    "both": ARTIFACTS / "trace_stabilization_both.json",
}

DEFAULT_CELL = "sentence_packed|gte_only"
PROMOTION_GATE = {
    "required_auroc": 0.90,
    "required_false_allow_rate": 0.02,
    "required_false_block_rate": 0.05,
    "required_decision_coverage": 0.10,
}

FEATURES = (
    "file_alignment",
    "symbol_alignment",
    "test_outcome_alignment",
    "patch_alignment",
    "temporal_order_alignment",
    "claim_atom_coverage",
    "unsupported_atom_rate",
    "phantom_symbol_rate",
    "missing_command_evidence",
    "literal_match_rate",
    "literal_mismatch_rate",
    "identifier_query_overlap",
    "identifier_query_absent_rate",
    "warning_identifier_rate",
    "reverse_context",
    "consensus_hardened",
    "groundedness_v2",
    "triangular",
    "context_attribution_ratio",
    "context_uncertain_ratio",
    "dead_weight_ratio",
    "support_usage_rate",
    "context_token_log",
    "multi_cell_reverse_min",
    "multi_cell_reverse_max",
    "multi_cell_reverse_std",
)


@dataclass(frozen=True)
class TrajectoryRow:
    row_id: str
    label: int
    tier: str
    features: dict[str, float]
    base_scenario_id: str
    source_schema: str


def ensure_native_banks(banks: dict[str, Path] | None = None) -> dict[str, Path]:
    banks = dict(banks or DEFAULT_BANKS)
    native_paths = [banks.get(name) for name in ("native_train", "native_val", "native_test")]
    if all(path is not None and path.exists() for path in native_paths):
        return banks
    generated = write_native_banks(generate_native_rows(), NATIVE_OUT_DIR)
    banks.update(
        {
            "native_train": generated["train"],
            "native_val": generated["val"],
            "native_test": generated["test"],
        }
    )
    return banks


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _row_label(row: dict[str, Any]) -> int | None:
    label = str(row.get("label") or "").lower()
    tier = str(row.get("tier") or "").lower()
    if label == "grounded" or tier == "correct":
        return 1
    if label == "ungrounded" or tier == "wrong":
        return 0
    return None


def _tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z_][A-Za-z0-9_\.]*|\d+(?:\.\d+)?", text or "")


def _code_identifiers(text: str) -> list[str]:
    identifiers: list[str] = []
    for token in _tokens(text):
        if len(token) < 3:
            continue
        if "_" in token or "." in token or any(ch.isupper() for ch in token[1:]):
            identifiers.append(token)
    return identifiers


def _overlap_rate(claimed: Iterable[str], observed: Iterable[str]) -> float:
    claimed_set = {str(item).lower() for item in claimed if str(item)}
    observed_set = {str(item).lower() for item in observed if str(item)}
    if not claimed_set:
        return 1.0
    return len(claimed_set & observed_set) / len(claimed_set)


def _native_features(row: dict[str, Any]) -> dict[str, float]:
    expected = row.get("expected") or {}
    observed = row.get("observed") or {}
    claim_files = expected.get("claim_files") or expected.get("files") or []
    claim_symbols = expected.get("claim_symbols") or expected.get("symbols") or []
    observed_files = observed.get("files") or []
    observed_symbols = observed.get("symbols") or []
    claim_order = expected.get("claim_order") or expected.get("order") or []
    observed_order = observed.get("order") or []
    atoms = [
        _overlap_rate(claim_files, observed_files),
        _overlap_rate(claim_symbols, observed_symbols),
        1.0 if bool(expected.get("claim_test_passed", expected.get("test_passed"))) == bool(observed.get("test_passed")) else 0.0,
        1.0 if bool(expected.get("claim_patch_applied", expected.get("patch_applied"))) == bool(observed.get("patch_applied")) else 0.0,
        1.0 if list(claim_order) == list(observed_order) else 0.0,
    ]
    unsupported = [1.0 - atom for atom in atoms]
    return {
        "file_alignment": atoms[0],
        "symbol_alignment": atoms[1],
        "test_outcome_alignment": atoms[2],
        "patch_alignment": atoms[3],
        "temporal_order_alignment": atoms[4],
        "claim_atom_coverage": float(np.mean(atoms)),
        "unsupported_atom_rate": float(np.mean(unsupported)),
        "phantom_symbol_rate": 1.0 - atoms[1],
        "missing_command_evidence": 0.0 if observed.get("test_command") or expected.get("test_command") else 1.0,
        "literal_match_rate": 1.0,
        "literal_mismatch_rate": 1.0 - float(np.mean(atoms)),
        "identifier_query_overlap": atoms[1],
        "identifier_query_absent_rate": 1.0 - atoms[1],
        "warning_identifier_rate": 1.0 - atoms[1],
        "reverse_context": 0.80 + 0.15 * float(np.mean(atoms)),
        "consensus_hardened": 0.80 + 0.15 * float(np.mean(atoms)),
        "groundedness_v2": 0.80 + 0.15 * float(np.mean(atoms)),
        "triangular": 0.80 + 0.15 * float(np.mean(atoms)),
        "context_attribution_ratio": float(np.mean(atoms)),
        "context_uncertain_ratio": 1.0 - float(np.mean(atoms)),
        "dead_weight_ratio": 1.0 - float(np.mean(atoms)),
        "support_usage_rate": float(np.mean(atoms)),
        "context_token_log": math.log1p(sum(len(str(event.get("text") or "")) for event in row.get("trace_events") or [])),
        "multi_cell_reverse_min": 0.80 + 0.15 * float(np.mean(atoms)),
        "multi_cell_reverse_max": 0.80 + 0.15 * float(np.mean(atoms)),
        "multi_cell_reverse_std": 0.0,
    }


def _legacy_cell_features(row: dict[str, Any]) -> dict[str, float]:
    scores = row.get("scores") or {}
    literal_total = _safe_float(scores.get("literal_total_count"))
    literal_match = _safe_float(scores.get("literal_match_count"))
    literal_mismatch = _safe_float(scores.get("literal_mismatch_count"))
    support_total = _safe_float(scores.get("support_units_total"))
    support_used = _safe_float(scores.get("support_units_usage_used"))
    response = str(row.get("response") or "")
    query_lower = str(row.get("query") or "").lower()
    identifiers = _code_identifiers(response)
    identifier_overlap = sum(1 for item in identifiers if item.lower() in query_lower)
    warning_ids = re.findall(r"identifier=([^,\s]+)", " ".join(str(w) for w in row.get("warnings") or []))
    identifier_total = len(identifiers) or 1
    return {
        "file_alignment": identifier_overlap / identifier_total,
        "symbol_alignment": identifier_overlap / identifier_total,
        "test_outcome_alignment": 0.5,
        "patch_alignment": 0.5,
        "temporal_order_alignment": 0.5,
        "claim_atom_coverage": identifier_overlap / identifier_total,
        "unsupported_atom_rate": 1.0 - (identifier_overlap / identifier_total),
        "phantom_symbol_rate": 1.0 - (identifier_overlap / identifier_total),
        "missing_command_evidence": 0.0 if "test" in response.lower() or "pytest" in response.lower() else 1.0,
        "literal_match_rate": literal_match / literal_total if literal_total else 0.0,
        "literal_mismatch_rate": literal_mismatch / literal_total if literal_total else 0.0,
        "identifier_query_overlap": identifier_overlap / identifier_total,
        "identifier_query_absent_rate": 1.0 - (identifier_overlap / identifier_total),
        "warning_identifier_rate": len(warning_ids) / max(1, len(warning_ids) + identifier_overlap),
        "reverse_context": _safe_float(scores.get("reverse_context")),
        "consensus_hardened": _safe_float(scores.get("consensus_hardened")),
        "groundedness_v2": _safe_float(scores.get("groundedness_v2")),
        "triangular": _safe_float(scores.get("triangular")),
        "context_attribution_ratio": _safe_float(scores.get("context_attribution_ratio")),
        "context_uncertain_ratio": _safe_float(scores.get("context_uncertain_ratio")),
        "dead_weight_ratio": _safe_float(scores.get("dead_weight_ratio")),
        "support_usage_rate": support_used / support_total if support_total else 0.0,
        "context_token_log": math.log1p(_safe_float(row.get("context_token_count"))),
    }


def _merge_cell_features(cell_rows: list[dict[str, Any]]) -> dict[str, float]:
    feature_rows = [_legacy_cell_features(row) for row in cell_rows]
    merged = dict(feature_rows[0])
    reverse_values = [features.get("reverse_context", 0.0) for features in feature_rows]
    for name in FEATURES:
        vals = [features.get(name) for features in feature_rows if name in features]
        if vals:
            merged[name] = float(np.mean(vals))
    merged["multi_cell_reverse_min"] = float(np.min(reverse_values)) if reverse_values else 0.0
    merged["multi_cell_reverse_max"] = float(np.max(reverse_values)) if reverse_values else 0.0
    merged["multi_cell_reverse_std"] = float(np.std(reverse_values)) if reverse_values else 0.0
    return merged


def load_rows(path: Path, *, cell: str | None = None) -> list[TrajectoryRow]:
    payload = _read_json(path)
    schema = str(payload.get("schema") or "")
    rows: list[TrajectoryRow] = []
    if schema == "trace_trajectory_native_bank.v1":
        for raw in payload.get("rows") or []:
            label = _row_label(raw)
            if label is None:
                continue
            rows.append(
                TrajectoryRow(
                    row_id=str(raw.get("row_id") or raw.get("id")),
                    label=label,
                    tier=str(raw.get("tier") or ""),
                    features=_native_features(raw),
                    base_scenario_id=str(raw.get("base_scenario_id") or raw.get("row_id")),
                    source_schema=schema,
                )
            )
        return rows

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    wanted_chunker: str | None = None
    wanted_scorer: str | None = None
    if cell:
        wanted_chunker, wanted_scorer = cell.split("|", 1)
    for raw in payload.get("rows") or []:
        if wanted_chunker and raw.get("chunker") != wanted_chunker:
            continue
        if wanted_scorer and raw.get("scorer_config") != wanted_scorer:
            continue
        label = _row_label(raw)
        if label is None:
            continue
        grouped[str(raw.get("id"))].append(raw)

    for row_id, cell_rows in grouped.items():
        first = cell_rows[0]
        metadata = first.get("metadata") or {}
        label = _row_label(first)
        if label is None:
            continue
        rows.append(
            TrajectoryRow(
                row_id=row_id,
                label=label,
                tier=str(first.get("tier") or ""),
                features=_merge_cell_features(cell_rows),
                base_scenario_id=str(first.get("base_scenario_id") or metadata.get("base_scenario_id") or row_id),
                source_schema="trace_stabilization_legacy",
            )
        )
    return rows


def _matrix(rows: list[TrajectoryRow], feature_names: Iterable[str] = FEATURES) -> np.ndarray:
    names = list(feature_names)
    return np.array([[row.features.get(name, 0.0) for name in names] for row in rows], dtype=float)


def _labels(rows: list[TrajectoryRow]) -> np.ndarray:
    return np.array([row.label for row in rows], dtype=int)


def _standardize(rows: list[TrajectoryRow]) -> tuple[dict[str, float], dict[str, float]]:
    means: dict[str, float] = {}
    scales: dict[str, float] = {}
    for name in FEATURES:
        vals = [row.features.get(name, 0.0) for row in rows]
        mean = sum(vals) / len(vals) if vals else 0.0
        var = sum((v - mean) ** 2 for v in vals) / len(vals) if vals else 0.0
        means[name] = mean
        scales[name] = math.sqrt(var) or 1.0
    return means, scales


def train_ranker(rows: list[TrajectoryRow]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot train trajectory head without rows")
    means, scales = _standardize(rows)
    labels = [row.label for row in rows]
    y_mean = sum(labels) / len(labels)
    y_scale = math.sqrt(sum((y - y_mean) ** 2 for y in labels) / len(labels)) or 1.0
    weights: dict[str, float] = {}
    for name in FEATURES:
        cov = sum(
            ((row.features.get(name, 0.0) - means[name]) / scales[name])
            * ((row.label - y_mean) / y_scale)
            for row in rows
        ) / len(rows)
        weights[name] = cov
    return {
        "model_type": "trajectory_symbolic_linear_ranker",
        "feature_names": list(FEATURES),
        "means": means,
        "scales": scales,
        "weights": weights,
        "train_rows": len(rows),
    }


def train_logreg(rows: list[TrajectoryRow]) -> dict[str, Any]:
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=23)),
        ]
    )
    model.fit(_matrix(rows), _labels(rows))
    scaler = model.named_steps["scale"]
    clf = model.named_steps["clf"]
    return {
        "model_type": "trajectory_logistic_regression",
        "feature_names": list(FEATURES),
        "means": {name: float(value) for name, value in zip(FEATURES, scaler.mean_)},
        "scales": {name: float(value) if float(value) else 1.0 for name, value in zip(FEATURES, scaler.scale_)},
        "weights": {name: float(value) for name, value in zip(FEATURES, clf.coef_[0])},
        "intercept": float(clf.intercept_[0]),
        "train_rows": len(rows),
    }


def _train_tree_candidate(rows: list[TrajectoryRow], model_type: str) -> Any:
    if model_type == "trajectory_extra_trees":
        model = ExtraTreesClassifier(n_estimators=200, max_depth=6, class_weight="balanced", random_state=23)
    else:
        model = GradientBoostingClassifier(random_state=23)
    model.fit(_matrix(rows), _labels(rows))
    return model


def score_row(row: TrajectoryRow, model: dict[str, Any]) -> float:
    names = list(model.get("feature_names") or FEATURES)
    means = {k: float(v) for k, v in (model.get("means") or {}).items()}
    scales = {k: float(v) or 1.0 for k, v in (model.get("scales") or {}).items()}
    weights = {k: float(v) for k, v in (model.get("weights") or {}).items()}
    raw = float(model.get("intercept") or 0.0)
    for name in names:
        raw += weights.get(name, 0.0) * ((row.features.get(name, 0.0) - means.get(name, 0.0)) / scales.get(name, 1.0))
    if str(model.get("model_type")) == "trajectory_logistic_regression":
        return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, raw))))
    return raw


def _best_threshold(scores: list[float], labels: list[int]) -> float:
    candidates = sorted(set(scores))
    if not candidates:
        return 0.0
    best = candidates[0]
    best_balanced = -1.0
    for threshold in candidates:
        metrics = _forced_metrics(scores, labels, threshold)
        balanced = (metrics["grounded_recall"] + metrics["ungrounded_recall"]) / 2.0
        if balanced > best_balanced:
            best_balanced = balanced
            best = threshold
    return best


def _calibrate_policy(scores: list[float], labels: list[int]) -> tuple[float, float]:
    candidates = sorted(set(scores))
    if not candidates:
        return 1.0, 0.0
    allow_threshold = max(candidates)
    best_allow: tuple[float, float] | None = None
    for allow_threshold in candidates:
        allowed = [label for score, label in zip(scores, labels) if score >= allow_threshold]
        if not allowed:
            continue
        false_allow = sum(1 for label in allowed if label == 0) / len(allowed)
        if false_allow <= PROMOTION_GATE["required_false_allow_rate"]:
            key = (float(len(allowed)), -float(allow_threshold))
            if best_allow is None or key > best_allow:
                best_allow = key
                selected_allow = allow_threshold
    allow_threshold = locals().get("selected_allow", max(candidates))

    block_threshold = min(candidates)
    best_block: tuple[float, float] | None = None
    for candidate in candidates:
        blocked = [label for score, label in zip(scores, labels) if score <= candidate]
        if not blocked:
            continue
        false_block = sum(1 for label in blocked if label == 1) / len(blocked)
        if false_block <= PROMOTION_GATE["required_false_block_rate"]:
            key = (float(len(blocked)), float(candidate))
            if best_block is None or key > best_block:
                best_block = key
                block_threshold = candidate
    if block_threshold >= allow_threshold:
        lower = [candidate for candidate in candidates if candidate < allow_threshold]
        block_threshold = max(lower) if lower else min(candidates)
    return float(allow_threshold), float(block_threshold)


def _forced_metrics(scores: list[float], labels: list[int], threshold: float) -> dict[str, float]:
    tp = tn = fp = fn = 0
    for score, label in zip(scores, labels):
        pred = 1 if score >= threshold else 0
        if pred == 1 and label == 1:
            tp += 1
        elif pred == 0 and label == 0:
            tn += 1
        elif pred == 1 and label == 0:
            fp += 1
        else:
            fn += 1
    n = len(labels) or 1
    grounded = tp + fn
    ungrounded = tn + fp
    return {
        "accuracy": (tp + tn) / n,
        "forced_false_allow_rate": fp / ungrounded if ungrounded else 0.0,
        "forced_false_block_rate": fn / grounded if grounded else 0.0,
        "grounded_recall": tp / grounded if grounded else 0.0,
        "ungrounded_recall": tn / ungrounded if ungrounded else 0.0,
    }


def _policy_metrics(
    scores: list[float],
    labels: list[int],
    allow_threshold: float,
    block_threshold: float,
) -> dict[str, float]:
    allowed = blocked = false_allow = false_block = 0
    for score, label in zip(scores, labels):
        if score >= allow_threshold:
            allowed += 1
            false_allow += 1 if label == 0 else 0
        elif score <= block_threshold:
            blocked += 1
            false_block += 1 if label == 1 else 0
    decided = allowed + blocked
    total = len(labels) or 1
    return {
        "allowed": float(allowed),
        "blocked": float(blocked),
        "auto_repair": float(len(labels) - decided),
        "decision_coverage": decided / total,
        "false_allow_rate": false_allow / allowed if allowed else 0.0,
        "false_block_rate": false_block / blocked if blocked else 0.0,
        "allow_threshold": float(allow_threshold),
        "block_threshold": float(block_threshold),
    }


def auc(scores: list[float], labels: list[int]) -> float | None:
    positives = [s for s, y in zip(scores, labels) if y == 1]
    negatives = [s for s, y in zip(scores, labels) if y == 0]
    if not positives or not negatives:
        return None
    wins = 0.0
    for pos in positives:
        for neg in negatives:
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def evaluate_ranker(model: dict[str, Any], rows: list[TrajectoryRow]) -> dict[str, Any]:
    scores = [score_row(row, model) for row in rows]
    labels = [row.label for row in rows]
    threshold = float(model.get("threshold") or _best_threshold(scores, labels))
    allow_threshold = float(model.get("allow_threshold", threshold))
    block_threshold = float(model.get("block_threshold", threshold))
    metrics = _forced_metrics(scores, labels, threshold)
    metrics.update(_policy_metrics(scores, labels, allow_threshold, block_threshold))
    metrics.update(
        {
            "rows": len(rows),
            "auroc": auc(scores, labels),
            "threshold": threshold,
            "grounded_count": sum(labels),
            "ungrounded_count": len(labels) - sum(labels),
        }
    )
    return metrics


def _evaluate_tree(model: Any, rows: list[TrajectoryRow], thresholds: tuple[float, float, float]) -> dict[str, Any]:
    scores = [float(score) for score in model.predict_proba(_matrix(rows))[:, 1]]
    labels = [row.label for row in rows]
    threshold, allow_threshold, block_threshold = thresholds
    metrics = _forced_metrics(scores, labels, threshold)
    metrics.update(_policy_metrics(scores, labels, allow_threshold, block_threshold))
    metrics.update(
        {
            "rows": len(rows),
            "auroc": auc(scores, labels),
            "threshold": threshold,
            "grounded_count": sum(labels),
            "ungrounded_count": len(labels) - sum(labels),
        }
    )
    return metrics


def _passes(metrics: dict[str, Any]) -> bool:
    return bool(
        metrics.get("auroc") is not None
        and float(metrics["auroc"]) >= PROMOTION_GATE["required_auroc"]
        and float(metrics["false_allow_rate"]) <= PROMOTION_GATE["required_false_allow_rate"]
        and float(metrics["false_block_rate"]) <= PROMOTION_GATE["required_false_block_rate"]
        and float(metrics.get("decision_coverage", 0.0)) >= PROMOTION_GATE["required_decision_coverage"]
    )


def _calibrate_model(model: dict[str, Any], rows: list[TrajectoryRow]) -> dict[str, Any]:
    scores = [score_row(row, model) for row in rows]
    labels = [row.label for row in rows]
    threshold = _best_threshold(scores, labels)
    allow_threshold, block_threshold = _calibrate_policy(scores, labels)
    return {**model, "threshold": threshold, "allow_threshold": allow_threshold, "block_threshold": block_threshold}


def run(
    *,
    train_bank: str = "native_train",
    calibration_bank: str | None = "native_val",
    eval_banks: Iterable[str] = ("native_test",),
    legacy_eval_banks: Iterable[str] = ("transcripts_v2", "both"),
    banks: dict[str, Path] | None = None,
) -> dict[str, Any]:
    banks = ensure_native_banks(banks)
    train_rows = load_rows(banks[train_bank])
    calibration_rows = (
        load_rows(banks[calibration_bank])
        if calibration_bank and calibration_bank in banks
        else train_rows
    )
    candidates: list[dict[str, Any]] = []

    for name, model in (
        ("symbolic_linear", train_ranker(train_rows)),
        ("trajectory_logreg", train_logreg(train_rows)),
    ):
        calibrated = _calibrate_model(model, calibration_rows)
        candidates.append(
            {
                "candidate": name,
                "model": calibrated,
                "calibration_metrics": evaluate_ranker(calibrated, calibration_rows),
            }
        )

    for name, model_type in (("extra_trees", "trajectory_extra_trees"), ("gbdt", "trajectory_gbdt")):
        tree = _train_tree_candidate(train_rows, model_type)
        val_scores = [float(score) for score in tree.predict_proba(_matrix(calibration_rows))[:, 1]]
        val_labels = [row.label for row in calibration_rows]
        thresholds = (_best_threshold(val_scores, val_labels), *_calibrate_policy(val_scores, val_labels))
        candidates.append(
            {
                "candidate": name,
                "model": {"model_type": model_type, "feature_names": list(FEATURES), "not_runtime_serialized": True},
                "tree_model": tree,
                "thresholds": thresholds,
                "calibration_metrics": _evaluate_tree(tree, calibration_rows, thresholds),
            }
        )

    eval_results: dict[str, dict[str, dict[str, Any]]] = {}
    for candidate in candidates:
        per_bank: dict[str, dict[str, Any]] = {}
        for bank in eval_banks:
            rows = load_rows(banks[bank])
            if "tree_model" in candidate:
                per_bank[bank] = _evaluate_tree(candidate["tree_model"], rows, candidate["thresholds"])
            else:
                per_bank[bank] = evaluate_ranker(candidate["model"], rows)
        candidate["eval"] = per_bank
        eval_results[candidate["candidate"]] = per_bank

    passing = [
        candidate
        for candidate in candidates
        if all(_passes(metrics) for metrics in candidate["eval"].values())
    ]
    selected = sorted(
        passing,
        key=lambda candidate: (
            0 if candidate["candidate"] == "trajectory_logreg" else 1,
            -min(float(metrics.get("decision_coverage", 0.0)) for metrics in candidate["eval"].values()),
        ),
    )[0] if passing else None

    legacy_eval: dict[str, Any] = {}
    diagnostic_model = selected["model"] if selected and "tree_model" not in selected else candidates[0]["model"]
    for bank in legacy_eval_banks:
        path = banks.get(bank)
        if path and path.exists():
            legacy_eval[bank] = evaluate_ranker(diagnostic_model, load_rows(path))

    report = {
        "schema": "trace_coding_trajectory_head.v2",
        "train_bank": train_bank,
        "calibration_bank": calibration_bank,
        "eval_banks": list(eval_banks),
        "legacy_eval_banks": list(legacy_eval_banks),
        "feature_names": list(FEATURES),
        "promotion_gate": PROMOTION_GATE,
        "train_rows": len(train_rows),
        "calibration_rows": len(calibration_rows),
        "candidate_bakeoff": [
            {k: v for k, v in candidate.items() if k not in {"tree_model"}}
            for candidate in candidates
        ],
        "selected_candidate": selected["candidate"] if selected else None,
        "model": selected["model"] if selected else candidates[0]["model"],
        "train_metrics": (
            _evaluate_tree(selected["tree_model"], train_rows, selected["thresholds"])
            if selected and "tree_model" in selected
            else evaluate_ranker(selected["model"], train_rows)
            if selected
            else evaluate_ranker(candidates[0]["model"], train_rows)
        ),
        "eval": selected["eval"] if selected else candidates[0]["eval"],
        "legacy_eval": legacy_eval,
    }
    report["promotion_decision"] = "promote" if selected else "do_not_promote_keep_code_agentic_trace_repair_only"
    return report


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Coding Agent Trajectory Head",
        "",
        f"Trained on `{report['train_bank']}`, calibrated on `{report.get('calibration_bank')}`.",
        "",
        "## Selected Evaluation",
        "",
        "| split | rows | AUROC | accuracy | false allow | false block | decision coverage |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split, metrics in {"train": report["train_metrics"], **report["eval"]}.items():
        auroc_value = metrics["auroc"]
        lines.append(
            f"| `{split}` | {metrics['rows']} | "
            f"{'n/a' if auroc_value is None else round(float(auroc_value), 4)} | "
            f"{round(metrics['accuracy'], 4)} | "
            f"{round(metrics['false_allow_rate'], 4)} | "
            f"{round(metrics['false_block_rate'], 4)} | "
            f"{round(metrics.get('decision_coverage', 0.0), 4)} |"
        )
    lines.extend(["", "## Candidate Bake-Off", ""])
    for candidate in report.get("candidate_bakeoff") or []:
        lines.append(f"- `{candidate['candidate']}`")
        for split, metrics in (candidate.get("eval") or {}).items():
            auroc_value = metrics["auroc"]
            lines.append(
                f"  - `{split}`: AUROC={'n/a' if auroc_value is None else round(float(auroc_value), 4)}, "
                f"false_allow={round(metrics['false_allow_rate'], 4)}, "
                f"false_block={round(metrics['false_block_rate'], 4)}"
            )
    if report.get("legacy_eval"):
        lines.extend(["", "## Legacy Similarity-Bank Diagnostic", ""])
        for split, metrics in report["legacy_eval"].items():
            auroc_value = metrics["auroc"]
            lines.append(
                f"- `{split}`: AUROC={'n/a' if auroc_value is None else round(float(auroc_value), 4)}, "
                f"false_allow={round(metrics['false_allow_rate'], 4)}, "
                f"false_block={round(metrics['false_block_rate'], 4)}"
            )
    lines.extend(
        [
            "",
            f"Promotion decision: `{report['promotion_decision']}`.",
            "",
            "Promotion is based on trajectory-native held-out banks. Legacy "
            "similarity-only transcript banks are retained as diagnostics because "
            "they do not contain the file/test/order facts needed by the runtime head.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-bank", default="native_train", choices=sorted(DEFAULT_BANKS))
    parser.add_argument("--calibration-bank", default="native_val", choices=sorted(DEFAULT_BANKS))
    parser.add_argument(
        "--eval-bank",
        action="append",
        dest="eval_banks",
        choices=sorted(DEFAULT_BANKS),
        help="Evaluation bank. May be passed multiple times.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(ARTIFACTS / "trajectory_head_root_cause_v1"),
        help="Directory for trajectory-head artifacts.",
    )
    args = parser.parse_args()
    eval_banks = tuple(args.eval_banks or ["native_test"])
    report = run(train_bank=args.train_bank, calibration_bank=args.calibration_bank, eval_banks=eval_banks)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "coding_trajectory_head.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    (out_dir / "coding_trajectory_head.md").write_text(render_markdown(report), encoding="utf-8")
    print(out_dir / "coding_trajectory_head.json")
    print(out_dir / "coding_trajectory_head.md")


if __name__ == "__main__":
    main()
