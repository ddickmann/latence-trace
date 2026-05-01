"""Build the TRACE launch scorecard from frozen benchmark artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional


GRANITE_41_NON_THINK_AVG = 0.760
GRANITE_41_THINK_AVG = 0.764


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_get(payload: Mapping[str, Any], path: str, default: Any = None) -> Any:
    current: Any = payload
    for part in path.split("."):
        if not isinstance(current, Mapping):
            return default
        current = current.get(part)
    return current if current is not None else default


def _fmt(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def _pct(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def build_scorecard(args: argparse.Namespace) -> dict[str, Any]:
    calibrator = _load_json(args.calibrator)
    eval_report = _load_json(args.eval_report)
    granite_resolution = _load_json(args.granite_resolution)
    coverage = _load_json(args.coverage_report)
    unused = _load_json(args.unused_context_report)
    response_chunking = _load_json(args.response_chunking_report)

    macro_bacc = _safe_get(eval_report, "aggregate.macro_balanced_accuracy")
    overall_bacc = _safe_get(eval_report, "aggregate.overall.balanced_accuracy")
    rows = _safe_get(eval_report, "aggregate.total_rows")
    full_cache_rows = _count_lines(args.full_cache_rows)

    product_value = {
        "unused_context": {
            "gate_status": unused.get("gate_status"),
            "heldout_accuracy": _safe_get(unused, "heldout.accuracy"),
            "heldout_unused_precision": _safe_get(unused, "heldout.unused_precision"),
            "heldout_unused_recall": _safe_get(unused, "heldout.unused_recall"),
        },
        "coverage_attribution": {
            "grounded_context_coverage_mean": _safe_get(
                coverage, "grounded.context_coverage_ratio.mean"
            ),
            "grounded_context_attribution_mean": _safe_get(
                coverage, "grounded.context_attribution_ratio.mean"
            ),
            "overfetch_context_coverage_mean": _safe_get(
                coverage, "overfetch.context_coverage_ratio.mean"
            ),
            "overfetch_support_units_mean": _safe_get(
                coverage, "overfetch.support_units_total.mean"
            ),
            "latency_p95_ms": _safe_get(coverage, "latency_ms_per_request.p95"),
        },
        "response_chunking": response_chunking.get("regimes", []),
        "granite_resolution_counterexample": {
            "row_accuracy": _safe_get(granite_resolution, "case_level.row.accuracy"),
            "matrix_accuracy": _safe_get(granite_resolution, "case_level.matrix.accuracy"),
            "rag_matrix_accuracy": _safe_get(
                granite_resolution, "case_level.matrix.by_lane.rag.accuracy"
            ),
            "prompt_multiplier": (
                _safe_get(granite_resolution, "quality_latency.prompt_count", 0)
                / max(1, _safe_get(granite_resolution, "dataset.case_count", 1))
            ),
            "throughput_prompts_per_s": _safe_get(
                granite_resolution, "quality_latency.throughput_prompts_per_s"
            ),
        },
    }

    coarse_gap_non_think = (
        macro_bacc - GRANITE_41_NON_THINK_AVG if isinstance(macro_bacc, (int, float)) else None
    )
    coarse_gap_think = (
        macro_bacc - GRANITE_41_THINK_AVG if isinstance(macro_bacc, (int, float)) else None
    )
    launch_recommendation = (
        "Do not claim Granite-parity classification yet. Launch positioning should lead with "
        "TRACE's actionable verification surface: localization, attribution, heatmaps, repair "
        "signals, and runtime policy decisions. Keep the frozen threshold artifact as an "
        "evaluation artifact only until full-cache held-out results improve materially."
    )

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "coarse_benchmark": {
            "source_eval_report": str(args.eval_report),
            "rows": rows,
            "macro_balanced_accuracy": macro_bacc,
            "overall_balanced_accuracy": overall_bacc,
            "granite_41_non_think_avg": GRANITE_41_NON_THINK_AVG,
            "granite_41_think_avg": GRANITE_41_THINK_AVG,
            "gap_to_granite_non_think": coarse_gap_non_think,
            "gap_to_granite_think": coarse_gap_think,
            "full_cache_rows_streamed": full_cache_rows,
            "full_cache_target_rows": 29320,
        },
        "calibrator": {
            "source": str(args.calibrator),
            "selected_candidate": calibrator.get("selected_candidate"),
            "selected_validation_macro_bacc": _safe_get(
                calibrator, "selected_validation.macro_balanced_accuracy"
            ),
            "guardrail_selection": calibrator.get("guardrail_selection"),
        },
        "product_value": product_value,
        "decision": {
            "replace_trace_with_granite": False,
            "ship_trace_as_black_box_classifier": False,
            "wire_calibrator_to_production_now": False,
            "continue_full_cache_scoring": full_cache_rows < 29320,
            "recommendation": launch_recommendation,
        },
    }


def write_markdown(scorecard: Mapping[str, Any], path: Path) -> None:
    coarse = scorecard["coarse_benchmark"]
    product = scorecard["product_value"]
    unused = product["unused_context"]
    coverage = product["coverage_attribution"]
    granite_res = product["granite_resolution_counterexample"]
    calibrator = scorecard["calibrator"]
    lines = [
        "# TRACE Launch Scorecard",
        "",
        f"Created: `{scorecard['created_at']}`",
        "",
        "## Decision",
        "",
        str(scorecard["decision"]["recommendation"]),
        "",
        "| Decision Item | Value |",
        "| --- | ---: |",
        f"| Replace TRACE with Granite | `{scorecard['decision']['replace_trace_with_granite']}` |",
        f"| Ship TRACE as black-box classifier | `{scorecard['decision']['ship_trace_as_black_box_classifier']}` |",
        f"| Wire calibrator to production now | `{scorecard['decision']['wire_calibrator_to_production_now']}` |",
        f"| Continue full-cache scoring | `{scorecard['decision']['continue_full_cache_scoring']}` |",
        "",
        "## Coarse Benchmark",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Held-out rows evaluated | `{coarse['rows']}` |",
        f"| TRACE macro balanced accuracy | `{_fmt(coarse['macro_balanced_accuracy'])}` |",
        f"| TRACE overall balanced accuracy | `{_fmt(coarse['overall_balanced_accuracy'])}` |",
        f"| Granite 4.1 non-think AVG | `{_fmt(coarse['granite_41_non_think_avg'])}` |",
        f"| Granite 4.1 think AVG | `{_fmt(coarse['granite_41_think_avg'])}` |",
        f"| Gap to Granite non-think | `{_fmt(coarse['gap_to_granite_non_think'])}` |",
        f"| Gap to Granite think | `{_fmt(coarse['gap_to_granite_think'])}` |",
        f"| Full cache streamed rows | `{coarse['full_cache_rows_streamed']}/{coarse['full_cache_target_rows']}` |",
        "",
        "## Frozen Calibrator",
        "",
        f"- Source: `{calibrator['source']}`",
        f"- Selected field: `{_safe_get(calibrator, 'selected_candidate.score_field')}`",
        f"- Group field: `{_safe_get(calibrator, 'selected_candidate.group_field')}`",
        f"- Dev validation macro BAcc: `{_fmt(calibrator['selected_validation_macro_bacc'])}`",
        f"- Guardrails: `{calibrator.get('guardrail_selection')}`",
        "",
        "## TRACE-Only Production Value",
        "",
        "| Product Signal | Result |",
        "| --- | ---: |",
        f"| Unused-context gate | `{unused['gate_status']}` |",
        f"| Held-out unused precision | `{_pct(unused['heldout_unused_precision'])}` |",
        f"| Held-out unused recall | `{_pct(unused['heldout_unused_recall'])}` |",
        f"| Grounded context coverage mean | `{_pct(coverage['grounded_context_coverage_mean'])}` |",
        f"| Grounded context attribution mean | `{_pct(coverage['grounded_context_attribution_mean'])}` |",
        f"| Overfetch context coverage mean | `{_pct(coverage['overfetch_context_coverage_mean'])}` |",
        f"| Coverage bench p95 latency | `{_fmt(coverage['latency_p95_ms'])} ms` |",
        "",
        "## Granite Resolution Counterexample",
        "",
        "| Granite Mode | Result |",
        "| --- | ---: |",
        f"| Whole-row accuracy | `{_pct(granite_res['row_accuracy'])}` |",
        f"| High-resolution matrix accuracy | `{_pct(granite_res['matrix_accuracy'])}` |",
        f"| High-resolution RAG matrix accuracy | `{_pct(granite_res['rag_matrix_accuracy'])}` |",
        f"| Prompt multiplier for heatmap-style scoring | `{_fmt(granite_res['prompt_multiplier'])}x` |",
        f"| Throughput | `{_fmt(granite_res['throughput_prompts_per_s'])} prompts/s` |",
        "",
        "## Launch Language",
        "",
        "Use: `Granite can classify; TRACE verifies, localizes, explains, and enables runtime intervention.`",
        "",
        "Do not claim Granite-equivalent coarse classification until the full-cache held-out score supports it.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibrator", type=Path, required=True)
    parser.add_argument("--eval-report", type=Path, required=True)
    parser.add_argument("--granite-resolution", type=Path, required=True)
    parser.add_argument("--coverage-report", type=Path, required=True)
    parser.add_argument("--unused-context-report", type=Path, required=True)
    parser.add_argument("--response-chunking-report", type=Path, required=True)
    parser.add_argument("--full-cache-rows", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    scorecard = build_scorecard(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(scorecard, indent=2, sort_keys=True), encoding="utf-8")
    md_path = args.out.with_suffix(".md")
    write_markdown(scorecard, md_path)
    print(f"scorecard: {args.out}")
    print(f"markdown:  {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
