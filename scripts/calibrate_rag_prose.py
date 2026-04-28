"""Threshold calibration for the ``rag_prose`` stratum.

This entry point produces customer-defensible green/amber/red thresholds for
prose RAG answers by fitting to a labeled calibration set instead of the
minimal-pair calibrator (which targets structured/entity/number swaps).

Subcommands:

* ``build-seed`` - assemble a first-pass calibration file from a completed
  Veracier benchmark run (variants.jsonl + trace_results.jsonl).
* ``fit`` - call TRACE RAG on the labeled set (both profiles) and write the
  fitted ``rag_prose`` stratum back into
  ``latence_trace/data/thresholds.balanced.json`` and
  ``latence_trace/data/thresholds.quality.json``.

The core fitting logic is a pure function (``fit_rag_prose_thresholds``) so it
can be unit-tested without the SDK / live TRACE service.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

DEFAULT_CALIBRATION_PATH = (
    Path(__file__).resolve().parent.parent
    / "latence_trace"
    / "data"
    / "calibration"
    / "rag_prose.jsonl"
)
DEFAULT_BALANCED_THRESHOLDS = (
    Path(__file__).resolve().parent.parent
    / "latence_trace"
    / "data"
    / "thresholds.balanced.json"
)
DEFAULT_QUALITY_THRESHOLDS = (
    Path(__file__).resolve().parent.parent
    / "latence_trace"
    / "data"
    / "thresholds.quality.json"
)

VALID_LABELS = {"green", "amber", "red"}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class CalibrationExample:
    id: str
    archetype: str
    language: str
    label: str
    query_text: str
    response_text: str
    raw_context: str
    source_doc_ids: List[str] = field(default_factory=list)
    notes: str = ""

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "CalibrationExample":
        missing = [
            field_name
            for field_name in ("id", "archetype", "label", "query_text", "response_text", "raw_context")
            if not payload.get(field_name)
        ]
        if missing:
            raise ValueError(
                f"calibration row {payload.get('id') or '?'} is missing fields: {missing}"
            )
        label = str(payload["label"]).strip().lower()
        if label not in VALID_LABELS:
            raise ValueError(
                f"calibration row {payload['id']} has invalid label {label!r}"
            )
        return cls(
            id=str(payload["id"]).strip(),
            archetype=str(payload["archetype"]).strip(),
            language=str(payload.get("language") or "fr").strip(),
            label=label,
            query_text=str(payload["query_text"]),
            response_text=str(payload["response_text"]),
            raw_context=str(payload["raw_context"]),
            source_doc_ids=list(payload.get("source_doc_ids") or []),
            notes=str(payload.get("notes") or ""),
        )


@dataclass
class ScoredExample:
    example: CalibrationExample
    score: float
    band: Optional[str]


def load_calibration_set(path: Path) -> List[CalibrationExample]:
    rows: List[CalibrationExample] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as fp:
        for line_no, line in enumerate(fp, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_no}: {exc}") from exc
            example = CalibrationExample.from_json(payload)
            if example.id in seen_ids:
                raise ValueError(f"duplicate calibration id: {example.id}")
            seen_ids.add(example.id)
            rows.append(example)
    return rows


# ---------------------------------------------------------------------------
# Seed builder
# ---------------------------------------------------------------------------


def _jsonl_iter(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _expected_label_from_variant(mutation_type: str, expected_band: str) -> Optional[str]:
    mutation = (mutation_type or "").strip().lower()
    expected = (expected_band or "").strip().lower()
    if mutation == "perfect" and expected == "green":
        return "green"
    if mutation == "ambiguous" and expected == "amber":
        return "amber"
    if mutation == "wrong" and expected == "red":
        return "red"
    return None


def build_seed_from_bench(
    *,
    benchmark_run: Path,
    out_path: Path,
    label_from: str = "expected-band",
    keep_unstable_amber: bool = False,
) -> Dict[str, Any]:
    """Promote high-confidence benchmark rows into the calibration file.

    A row is kept when the expected band matches the live TRACE band (i.e.
    the scorer and the benchmark already agree), which is the cheapest proxy
    for a human-approved label.
    """

    variants_path = benchmark_run / "variants.jsonl"
    trace_path = benchmark_run / "trace_results.jsonl"
    if not variants_path.exists():
        raise FileNotFoundError(f"missing variants.jsonl under {benchmark_run}")

    variants = {row["example_id"]: row for row in _jsonl_iter(variants_path)}
    trace_rows: List[Dict[str, Any]] = []
    if trace_path.exists():
        trace_rows = list(_jsonl_iter(trace_path))

    # Pick the standard-profile trace row as the confirmation source when
    # available - matches the refine-ambiguous sandbox profile.
    standard_by_example: Dict[str, Dict[str, Any]] = {}
    for row in trace_rows:
        if row.get("profile") != "standard":
            continue
        standard_by_example[row["example_id"]] = row

    kept: List[CalibrationExample] = []
    skipped: List[Dict[str, str]] = []
    for example_id, variant in variants.items():
        label = _expected_label_from_variant(
            variant.get("mutation_type", ""),
            variant.get("expected_band", ""),
        )
        if label is None:
            skipped.append({"id": example_id, "reason": "unrecognized mutation/expected pair"})
            continue
        if label == "amber" and variant.get("ambiguous_unstable") and not keep_unstable_amber:
            skipped.append({"id": example_id, "reason": "ambiguous_unstable"})
            continue
        trace_row = standard_by_example.get(example_id)
        if trace_row is not None and label_from == "expected-band":
            live_band = (trace_row.get("trace") or {}).get("band")
            if live_band is not None and live_band != label:
                skipped.append(
                    {"id": example_id, "reason": f"live band {live_band} != expected {label}"}
                )
                continue
        archetype_payload = variant.get("generation_archetype") or {}
        archetype = ""
        if isinstance(archetype_payload, dict):
            archetype = str(archetype_payload.get("id") or "")
        if not archetype:
            archetype = "default_enterprise"
        kept.append(
            CalibrationExample(
                id=str(example_id),
                archetype=archetype,
                language=str(variant.get("language") or "fr"),
                label=label,
                query_text=str(variant.get("query_text") or ""),
                response_text=str(variant.get("response_text") or ""),
                raw_context=str(variant.get("raw_context") or ""),
                source_doc_ids=list(variant.get("context_doc_ids") or []),
                notes=f"seed from benchmark {benchmark_run.name}",
            )
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fp:
        for example in kept:
            fp.write(json.dumps(example.__dict__, ensure_ascii=False) + "\n")

    return {
        "kept": len(kept),
        "skipped": len(skipped),
        "out_path": str(out_path),
        "skipped_rows": skipped[:20],
    }


# ---------------------------------------------------------------------------
# Fit logic (pure, testable)
# ---------------------------------------------------------------------------


def _precision_recall_at_threshold(
    positive_scores: Sequence[float],
    negative_scores: Sequence[float],
    threshold: float,
) -> Tuple[float, float]:
    tp = sum(1 for s in positive_scores if s >= threshold)
    fp = sum(1 for s in negative_scores if s >= threshold)
    fn = sum(1 for s in positive_scores if s < threshold)
    precision = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
    return float(precision), float(recall)


def _select_green_min(
    green_scores: Sequence[float],
    non_green_scores: Sequence[float],
    *,
    precision_target: float,
    grid_step: float = 0.005,
) -> Tuple[float, float, float]:
    """Smallest threshold hitting ``precision_target`` on the green class."""

    if not green_scores or not non_green_scores:
        return 1.0, 0.0, 0.0
    lo = min(min(green_scores), min(non_green_scores))
    hi = max(max(green_scores), max(non_green_scores))
    if lo == hi:
        return float(hi), 0.0, 0.0
    grid: List[float] = []
    x = lo
    while x <= hi + 1e-9:
        grid.append(round(float(x), 6))
        x += grid_step
    candidates: List[Tuple[float, float, float]] = []
    for thr in grid:
        precision, recall = _precision_recall_at_threshold(
            green_scores, non_green_scores, thr
        )
        if precision >= precision_target:
            candidates.append((thr, precision, recall))
    if candidates:
        candidates.sort(key=lambda item: (-item[2], item[0]))
        return candidates[0]
    # Fall back to the tightest threshold we have when precision cannot be met.
    return float(hi), 0.0, 0.0


def _select_amber_min(
    non_red_scores: Sequence[float],
    red_scores: Sequence[float],
    *,
    red_precision_target: float,
    grid_step: float = 0.005,
) -> Tuple[float, float, float]:
    """Largest threshold such that below it we hit the red precision floor.

    Red precision is the fraction of ``< amber_min`` predictions that are
    genuinely red. Since ``score < amber_min`` triggers the red band,
    red precision = TP_r / (TP_r + FP_r) where TP_r counts red examples with
    ``score < threshold`` and FP_r counts non-red examples with
    ``score < threshold``. We pick the largest threshold that still satisfies
    the precision floor so the amber band is as wide as possible.
    """

    if not red_scores or not non_red_scores:
        return 0.0, 0.0, 0.0
    lo = min(min(red_scores), min(non_red_scores))
    hi = max(max(red_scores), max(non_red_scores))
    if lo == hi:
        return float(lo), 0.0, 0.0
    grid: List[float] = []
    x = lo
    while x <= hi + 1e-9:
        grid.append(round(float(x), 6))
        x += grid_step
    candidates: List[Tuple[float, float, float]] = []
    for thr in grid:
        red_tp = sum(1 for s in red_scores if s < thr)
        red_fp = sum(1 for s in non_red_scores if s < thr)
        red_fn = sum(1 for s in red_scores if s >= thr)
        red_precision = (
            red_tp / float(red_tp + red_fp) if (red_tp + red_fp) > 0 else 0.0
        )
        red_recall = red_tp / float(red_tp + red_fn) if (red_tp + red_fn) > 0 else 0.0
        if red_precision >= red_precision_target:
            candidates.append((thr, red_precision, red_recall))
    if candidates:
        candidates.sort(key=lambda item: (-item[2], -item[0]))
        return candidates[0]
    return float(lo), 0.0, 0.0


@dataclass
class FitResult:
    profile: str
    green_min: float
    amber_min: float
    precision_at_green: float
    recall_at_green: float
    red_precision_at_amber: float
    red_recall_at_amber: float
    sample_count: int
    holdout_sample_count: int
    class_counts: Dict[str, int]
    positive_median: float
    negative_median: float

    def to_threshold_entry(self) -> Dict[str, Any]:
        return {
            "amber_min": float(self.amber_min),
            "green_min": float(self.green_min),
            "precision_at_green": float(self.precision_at_green),
            "recall_at_green": float(self.recall_at_green),
            "red_precision_at_amber": float(self.red_precision_at_amber),
            "red_recall_at_amber": float(self.red_recall_at_amber),
            "sample_count": int(self.sample_count),
            "holdout_sample_count": int(self.holdout_sample_count),
            "positive_median": float(self.positive_median),
            "negative_median": float(self.negative_median),
        }


def fit_rag_prose_thresholds(
    examples: Sequence[CalibrationExample],
    scores: Mapping[str, float],
    *,
    profile: str,
    green_precision_target: float = 0.97,
    red_precision_target: float = 0.95,
    holdout_fraction: float = 0.3,
    seed: int = 42,
) -> FitResult:
    """Fit ``green_min`` / ``amber_min`` for a single profile.

    ``scores`` maps ``example.id`` to the fused ``groundedness_v2`` score
    reported by TRACE for this profile.
    """

    if not examples:
        raise ValueError("no calibration examples supplied")
    if not scores:
        raise ValueError("no TRACE scores supplied")

    rng = random.Random(seed)
    shuffled = list(examples)
    rng.shuffle(shuffled)
    holdout_size = max(1, int(round(len(shuffled) * holdout_fraction)))
    train = shuffled[holdout_size:]
    holdout = shuffled[:holdout_size]
    if not train:
        train = shuffled
        holdout = shuffled

    def _bucket(items: Sequence[CalibrationExample]) -> Dict[str, List[float]]:
        b: Dict[str, List[float]] = {"green": [], "amber": [], "red": []}
        for item in items:
            if item.id not in scores:
                continue
            b[item.label].append(float(scores[item.id]))
        return b

    train_buckets = _bucket(train)
    green_scores = train_buckets["green"]
    non_green_scores = train_buckets["amber"] + train_buckets["red"]
    red_scores = train_buckets["red"]
    non_red_scores = train_buckets["green"] + train_buckets["amber"]

    green_min, _train_green_precision, _train_green_recall = _select_green_min(
        green_scores,
        non_green_scores,
        precision_target=green_precision_target,
    )
    amber_min, _train_red_precision, _train_red_recall = _select_amber_min(
        non_red_scores,
        red_scores,
        red_precision_target=red_precision_target,
    )
    if amber_min >= green_min:
        # Monotonicity: amber must sit strictly below green. When the fit
        # produces an inverted ordering we squash amber to 80 % of green.
        amber_min = max(0.0, green_min * 0.80)

    # Evaluate on the held-out split.
    holdout_buckets = _bucket(holdout)
    h_green = holdout_buckets["green"]
    h_amber = holdout_buckets["amber"]
    h_red = holdout_buckets["red"]
    precision_at_green, recall_at_green = _precision_recall_at_threshold(
        h_green, h_amber + h_red, green_min
    )
    if h_red or h_green or h_amber:
        red_tp = sum(1 for s in h_red if s < amber_min)
        red_fp = sum(1 for s in h_green + h_amber if s < amber_min)
        red_fn = sum(1 for s in h_red if s >= amber_min)
        red_precision = (
            red_tp / float(red_tp + red_fp) if (red_tp + red_fp) > 0 else 0.0
        )
        red_recall = red_tp / float(red_tp + red_fn) if (red_tp + red_fn) > 0 else 0.0
    else:
        red_precision = 0.0
        red_recall = 0.0

    all_scores = [float(scores[e.id]) for e in shuffled if e.id in scores]
    positives = [float(scores[e.id]) for e in shuffled if e.label == "green" and e.id in scores]
    negatives = [
        float(scores[e.id])
        for e in shuffled
        if e.label in {"amber", "red"} and e.id in scores
    ]
    class_counts = {
        "green": sum(1 for e in shuffled if e.label == "green"),
        "amber": sum(1 for e in shuffled if e.label == "amber"),
        "red": sum(1 for e in shuffled if e.label == "red"),
    }
    return FitResult(
        profile=profile,
        green_min=float(green_min),
        amber_min=float(amber_min),
        precision_at_green=float(precision_at_green),
        recall_at_green=float(recall_at_green),
        red_precision_at_amber=float(red_precision),
        red_recall_at_amber=float(red_recall),
        sample_count=len(all_scores),
        holdout_sample_count=len(holdout),
        class_counts=class_counts,
        positive_median=float(statistics.median(positives)) if positives else 0.0,
        negative_median=float(statistics.median(negatives)) if negatives else 0.0,
    )


def update_thresholds_file(path: Path, fit: FitResult) -> Dict[str, Any]:
    """Write ``fit`` into the ``rag_prose`` stratum of a thresholds file."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    strata = payload.setdefault("strata", {})
    strata["rag_prose"] = fit.to_threshold_entry()
    payload["rag_prose_fit"] = {
        "profile": fit.profile,
        "green_precision_target": 0.97,
        "red_precision_target": 0.95,
        "holdout_fraction": (
            fit.holdout_sample_count / float(fit.sample_count)
            if fit.sample_count
            else 0.0
        ),
        "class_counts": fit.class_counts,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return {"path": str(path), "entry": strata["rag_prose"]}


# ---------------------------------------------------------------------------
# Live TRACE scoring (SDK path)
# ---------------------------------------------------------------------------


def _score_examples_with_sdk(
    examples: Sequence[CalibrationExample],
    *,
    profile: str,
    structured_verification: str = "auto",
    sdk_timeout: float = 180.0,
) -> Dict[str, float]:
    from latence import Latence  # imported lazily so tests do not need the SDK

    api_key = os.environ.get("LATENCE_API_KEY") or os.environ.get("LATENCE_GATEWAY_KEY")
    if not api_key:
        raise RuntimeError("set LATENCE_API_KEY before running calibrate_rag_prose fit")

    scores: Dict[str, float] = {}
    client = Latence(
        api_key=api_key,
        base_url=os.environ.get("LATENCE_BASE_URL"),
        timeout=sdk_timeout,
    )
    try:
        for example in examples:
            response = client.experimental.trace.rag(
                response_text=example.response_text,
                query_text=example.query_text,
                raw_context=example.raw_context,
                primary_metric="triangular",
                segmentation_mode="sentence_packed",
                heatmap_format="none",
                profile=profile,  # type: ignore[arg-type]
                structured_verification=structured_verification,
                verbose=False,
            )
            dumped = (
                response.model_dump(mode="python", exclude_none=True)
                if hasattr(response, "model_dump")
                else dict(response)  # type: ignore[arg-type]
            )
            score = dumped.get("score")
            if score is None:
                raise RuntimeError(f"TRACE returned no score for {example.id}")
            scores[example.id] = float(score)
    finally:
        client.close()
    return scores


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli_build_seed(args: argparse.Namespace) -> int:
    summary = build_seed_from_bench(
        benchmark_run=args.benchmark_run,
        out_path=args.out,
        label_from=args.label_from,
        keep_unstable_amber=args.keep_unstable_amber,
    )
    print(json.dumps(summary, indent=2))
    return 0


def _cli_fit(args: argparse.Namespace) -> int:
    examples = load_calibration_set(args.calibration_jsonl)
    if not examples:
        raise SystemExit(f"calibration file {args.calibration_jsonl} is empty")

    profiles = args.profiles
    report_rows: Dict[str, Any] = {}
    for profile in profiles:
        print(f"scoring {len(examples)} examples with profile={profile}")
        scorer: Callable[[Sequence[CalibrationExample]], Dict[str, float]] = (
            lambda pool, p=profile: _score_examples_with_sdk(
                pool,
                profile=p,
                structured_verification=args.structured_verification,
                sdk_timeout=args.sdk_timeout,
            )
        )
        scores = scorer(examples)
        fit = fit_rag_prose_thresholds(
            examples,
            scores,
            profile=profile,
            green_precision_target=args.green_precision_target,
            red_precision_target=args.red_precision_target,
            holdout_fraction=args.holdout_fraction,
            seed=args.seed,
        )
        target_path = (
            args.quality_thresholds
            if profile == "quality"
            else args.balanced_thresholds
        )
        update_result = update_thresholds_file(target_path, fit)
        report_rows[profile] = {
            "fit": fit.to_threshold_entry(),
            "target_file": update_result["path"],
        }

    out_report = args.calibration_jsonl.with_name("rag_prose_fit.json")
    out_report.write_text(
        json.dumps({"profiles": report_rows}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps({"report_path": str(out_report), "profiles": report_rows}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    seed = sub.add_parser(
        "build-seed",
        help="Seed the calibration file from a completed Veracier benchmark run.",
    )
    seed.add_argument("--benchmark-run", type=Path, required=True)
    seed.add_argument("--out", type=Path, default=DEFAULT_CALIBRATION_PATH)
    seed.add_argument(
        "--label-from",
        choices=["expected-band", "expected-only"],
        default="expected-band",
    )
    seed.add_argument("--keep-unstable-amber", action="store_true")
    seed.set_defaults(func=_cli_build_seed)

    fit = sub.add_parser("fit", help="Fit thresholds against a labeled calibration set.")
    fit.add_argument(
        "--calibration-jsonl", type=Path, default=DEFAULT_CALIBRATION_PATH
    )
    fit.add_argument(
        "--balanced-thresholds", type=Path, default=DEFAULT_BALANCED_THRESHOLDS
    )
    fit.add_argument(
        "--quality-thresholds", type=Path, default=DEFAULT_QUALITY_THRESHOLDS
    )
    fit.add_argument("--green-precision-target", type=float, default=0.97)
    fit.add_argument("--red-precision-target", type=float, default=0.95)
    fit.add_argument("--holdout-fraction", type=float, default=0.3)
    fit.add_argument("--seed", type=int, default=42)
    fit.add_argument("--structured-verification", choices=["auto", "on", "off"], default="auto")
    fit.add_argument("--sdk-timeout", type=float, default=180.0)
    fit.add_argument(
        "--profiles",
        nargs="+",
        choices=["standard", "quality"],
        default=["standard", "quality"],
    )
    fit.set_defaults(func=_cli_fit)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help(sys.stderr)
        return 2
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
