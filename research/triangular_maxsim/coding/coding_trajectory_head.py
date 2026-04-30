"""Train and evaluate a lightweight coding-agent trajectory head.

The manufactured coding trajectory banks are the correct gate for
`code.agentic_trace`. This script trains a small symbolic-plus-linear ranker on
one trajectory bank and evaluates it on held-out trajectory banks. It is meant
to answer whether a cheap code-specific head can be promoted; it is not
promoted unless held-out false-allow/false-block gates pass.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = ROOT / "research/triangular_maxsim/coding/artifacts"

DEFAULT_BANKS = {
    "transcripts_v1": ARTIFACTS / "trace_stabilization_transcripts_v1.json",
    "transcripts_v2": ARTIFACTS / "trace_stabilization_transcripts_v2.json",
    "both": ARTIFACTS / "trace_stabilization_both.json",
}

DEFAULT_CELL = "sentence_packed|gte_only"

FEATURES = (
    "reverse_context",
    "consensus_hardened",
    "groundedness_v2",
    "triangular",
    "literal_match_rate",
    "literal_mismatch_rate",
    "context_attribution_ratio",
    "context_unused_ratio",
    "context_uncertain_ratio",
    "dead_weight_ratio",
    "max_top_evidence_score",
    "phantom_flagged",
    "support_usage_rate",
    "context_token_log",
)


@dataclass(frozen=True)
class TrajectoryRow:
    row_id: str
    label: int
    tier: str
    features: dict[str, float]
    base_scenario_id: str


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


def _features(row: dict[str, Any]) -> dict[str, float]:
    scores = row.get("scores") or {}
    literal_total = _safe_float(scores.get("literal_total_count"))
    literal_match = _safe_float(scores.get("literal_match_count"))
    literal_mismatch = _safe_float(scores.get("literal_mismatch_count"))
    support_total = _safe_float(scores.get("support_units_total"))
    support_used = _safe_float(scores.get("support_units_usage_used"))
    context_tokens = _safe_float(row.get("context_token_count"))
    return {
        "reverse_context": _safe_float(scores.get("reverse_context")),
        "consensus_hardened": _safe_float(scores.get("consensus_hardened")),
        "groundedness_v2": _safe_float(scores.get("groundedness_v2")),
        "triangular": _safe_float(scores.get("triangular")),
        "literal_match_rate": literal_match / literal_total if literal_total else 0.0,
        "literal_mismatch_rate": literal_mismatch / literal_total if literal_total else 0.0,
        "context_attribution_ratio": _safe_float(scores.get("context_attribution_ratio")),
        "context_unused_ratio": _safe_float(scores.get("context_unused_ratio")),
        "context_uncertain_ratio": _safe_float(scores.get("context_uncertain_ratio")),
        "dead_weight_ratio": _safe_float(scores.get("dead_weight_ratio")),
        "max_top_evidence_score": _safe_float(row.get("max_top_evidence_score")),
        "phantom_flagged": 1.0 if row.get("phantom_flagged") else 0.0,
        "support_usage_rate": support_used / support_total if support_total else 0.0,
        "context_token_log": math.log1p(context_tokens),
    }


def load_rows(path: Path, *, cell: str = DEFAULT_CELL) -> list[TrajectoryRow]:
    payload = _read_json(path)
    chunker, scorer_config = cell.split("|", 1)
    rows: list[TrajectoryRow] = []
    for raw in payload.get("rows") or []:
        if raw.get("chunker") != chunker or raw.get("scorer_config") != scorer_config:
            continue
        label = _row_label(raw)
        if label is None:
            continue
        metadata = raw.get("metadata") or {}
        rows.append(
            TrajectoryRow(
                row_id=str(raw.get("id")),
                label=label,
                tier=str(raw.get("tier") or ""),
                features=_features(raw),
                base_scenario_id=str(raw.get("base_scenario_id") or metadata.get("base_scenario_id") or raw.get("id")),
            )
        )
    return rows


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

    train_scores = [score_row(row, weights, means, scales) for row in rows]
    threshold = _best_threshold(train_scores, labels)
    return {
        "model_type": "trajectory_symbolic_linear_ranker",
        "cell": DEFAULT_CELL,
        "feature_names": list(FEATURES),
        "means": means,
        "scales": scales,
        "weights": weights,
        "threshold": threshold,
        "train_rows": len(rows),
    }


def score_row(
    row: TrajectoryRow,
    weights: dict[str, float],
    means: dict[str, float],
    scales: dict[str, float],
) -> float:
    return sum(
        weights.get(name, 0.0) * ((row.features.get(name, 0.0) - means[name]) / scales[name])
        for name in FEATURES
    )


def _best_threshold(scores: list[float], labels: list[int]) -> float:
    candidates = sorted(set(scores))
    if not candidates:
        return 0.0
    best = candidates[0]
    best_balanced = -1.0
    for threshold in candidates:
        metrics = _decision_metrics(scores, labels, threshold)
        balanced = (metrics["grounded_recall"] + metrics["ungrounded_recall"]) / 2.0
        if balanced > best_balanced:
            best_balanced = balanced
            best = threshold
    return best


def _decision_metrics(scores: list[float], labels: list[int], threshold: float) -> dict[str, float]:
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
        "false_allow_rate": fp / ungrounded if ungrounded else 0.0,
        "false_block_rate": fn / grounded if grounded else 0.0,
        "grounded_recall": tp / grounded if grounded else 0.0,
        "ungrounded_recall": tn / ungrounded if ungrounded else 0.0,
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
    weights = {k: float(v) for k, v in (model.get("weights") or {}).items()}
    means = {k: float(v) for k, v in (model.get("means") or {}).items()}
    scales = {k: float(v) for k, v in (model.get("scales") or {}).items()}
    threshold = float(model.get("threshold") or 0.0)
    scores = [score_row(row, weights, means, scales) for row in rows]
    labels = [row.label for row in rows]
    metrics = _decision_metrics(scores, labels, threshold)
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


def run(
    *,
    train_bank: str = "transcripts_v1",
    eval_banks: Iterable[str] = ("transcripts_v2", "both"),
    banks: dict[str, Path] | None = None,
) -> dict[str, Any]:
    banks = banks or DEFAULT_BANKS
    train_rows = load_rows(banks[train_bank])
    model = train_ranker(train_rows)
    report = {
        "schema": "trace_coding_trajectory_head.v1",
        "train_bank": train_bank,
        "cell": DEFAULT_CELL,
        "model": model,
        "train_metrics": evaluate_ranker(model, train_rows),
        "eval": {},
        "promotion_gate": {
            "required_auroc": 0.90,
            "required_false_allow_rate": 0.02,
            "required_false_block_rate": 0.05,
        },
    }
    promoted = True
    for bank in eval_banks:
        metrics = evaluate_ranker(model, load_rows(banks[bank]))
        report["eval"][bank] = metrics
        promoted = promoted and bool(
            metrics["auroc"] is not None
            and metrics["auroc"] >= report["promotion_gate"]["required_auroc"]
            and metrics["false_allow_rate"] <= report["promotion_gate"]["required_false_allow_rate"]
            and metrics["false_block_rate"] <= report["promotion_gate"]["required_false_block_rate"]
        )
    report["promotion_decision"] = (
        "promote" if promoted else "do_not_promote_keep_code_agentic_trace_repair_only"
    )
    return report


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Coding Agent Trajectory Head",
        "",
        f"Trained on `{report['train_bank']}` using `{report['cell']}` rows.",
        "",
        "| split | rows | AUROC | accuracy | false allow | false block |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split, metrics in {"train": report["train_metrics"], **report["eval"]}.items():
        auroc = metrics["auroc"]
        lines.append(
            f"| `{split}` | {metrics['rows']} | "
            f"{'n/a' if auroc is None else round(float(auroc), 4)} | "
            f"{round(metrics['accuracy'], 4)} | "
            f"{round(metrics['false_allow_rate'], 4)} | "
            f"{round(metrics['false_block_rate'], 4)} |"
        )
    lines.extend(
        [
            "",
            f"Promotion decision: `{report['promotion_decision']}`.",
            "",
            "This head is intentionally separated from production promotion. "
            "It becomes eligible only if held-out manufactured trajectory banks "
            "clear AUROC and false-decision gates.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-bank", default="transcripts_v1", choices=sorted(DEFAULT_BANKS))
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
    eval_banks = tuple(args.eval_banks or [b for b in DEFAULT_BANKS if b != args.train_bank])
    report = run(train_bank=args.train_bank, eval_banks=eval_banks)
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
