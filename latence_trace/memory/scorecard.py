"""Scorecards and promotion gates for TRACE Memory."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import Any

from latence_trace.memory.baselines import evaluate_baselines
from latence_trace.memory.trajectory import CanonicalTrajectory


def build_scorecard(
    trajectories: list[CanonicalTrajectory],
    *,
    learned_weights: dict[str, float] | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    progress_every: int = 0,
) -> dict[str, Any]:
    rows = []
    total = len(trajectories)
    for idx, trajectory in enumerate(trajectories, start=1):
        baselines = evaluate_baselines(trajectory, learned_weights=learned_weights)
        rows.append(
            {
                "dataset": trajectory.dataset,
                "case_id": trajectory.case_id,
                "domain": trajectory.domain,
                "turn_count": len(trajectory.turns),
                "baselines": baselines,
                "horizon_diagnostics": _horizon_diagnostics(baselines),
            }
        )
        if progress_callback is not None and (
            idx == total or (progress_every > 0 and idx % progress_every == 0)
        ):
            progress_callback(
                {
                    "event": "scorecard_progress",
                    "processed": idx,
                    "total": total,
                    "trace_memory_rule_based": _mean_metric(rows, "trace_memory_rule_based"),
                    "plain_compression": _mean_metric(rows, "plain_compression"),
                    "vector_memory": _mean_metric(rows, "vector_memory"),
                    "summary_memory": _mean_metric(rows, "summary_memory"),
                    "by_domain": _group_summary(rows, "domain"),
                }
            )
    return {
        "case_count": len(rows),
        "by_dataset": _group_summary(rows, "dataset"),
        "by_domain": _group_summary(rows, "domain"),
        "holdouts": _holdout_summary(rows),
        "ablations": _ablation_summary(rows),
        "long_horizon": _long_horizon_summary(rows),
        "matched_budget_baselines": _matched_budget_summary(rows),
        "domain_horizon_thesis": _domain_horizon_thesis(rows),
        "rows": rows,
    }


def promotion_decision(scorecard: dict[str, Any]) -> dict[str, Any]:
    trace = _mean_metric(scorecard["rows"], "trace_memory_rule_based")
    compression = _mean_metric(scorecard["rows"], "plain_compression")
    vector = _mean_metric(scorecard["rows"], "vector_memory")
    summary = _mean_metric(scorecard["rows"], "summary_memory")
    long_horizon = scorecard.get("long_horizon", {})
    by_domain = scorecard.get("by_domain", {})
    exact_ok = trace["exact_critical_preservation"] >= 0.98
    reduction_ok = trace["token_reduction"] > 0.15
    beats_generic = (
        trace["exact_critical_preservation"] >= compression["exact_critical_preservation"]
        and trace["exact_critical_preservation"] >= vector["exact_critical_preservation"]
        and trace["exact_critical_preservation"] >= summary["exact_critical_preservation"]
    )
    beats_by_domain = _beats_compression_by_domain(by_domain)
    long_horizon_ok = bool(long_horizon.get("coding_anchor_advantage_holds", False))
    approved = bool(
        exact_ok
        and reduction_ok
        and beats_generic
        and beats_by_domain
        and long_horizon_ok
        and scorecard.get("case_count", 0) > 0
    )
    return {
        "approved_for_production_coupling": approved,
        "recommended_mode": "shadow" if not approved else "opt_in_production",
        "reasons": {
            "exact_critical_gate": exact_ok,
            "token_reduction_gate": reduction_ok,
            "beats_generic_memory_gate": beats_generic,
            "beats_plain_compression_by_domain_gate": beats_by_domain,
            "long_horizon_coding_gain_gate": long_horizon_ok,
        },
        "trace_memory_rule_based": trace,
        "plain_compression": compression,
        "vector_memory": vector,
        "summary_memory": summary,
    }


def _group_summary(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return {
        group: {
            "case_count": len(values),
            "trace_memory_rule_based": _mean_metric(values, "trace_memory_rule_based"),
            "plain_compression": _mean_metric(values, "plain_compression"),
            "vector_memory": _mean_metric(values, "vector_memory"),
            "summary_memory": _mean_metric(values, "summary_memory"),
        }
        for group, values in grouped.items()
    }


def _holdout_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    datasets = sorted({row["dataset"] for row in rows})
    return {
        dataset: {
            "train_cases": len([row for row in rows if row["dataset"] != dataset]),
            "holdout_cases": len([row for row in rows if row["dataset"] == dataset]),
            "holdout_trace_memory_rule_based": _mean_metric(
                [row for row in rows if row["dataset"] == dataset],
                "trace_memory_rule_based",
            ),
        }
        for dataset in datasets
    }


def _ablation_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    trace = _mean_metric(rows, "trace_memory_rule_based")
    return {
        "without_attribution": {"expected_delta": "negative", "reference": trace},
        "without_dead_weight": {"expected_delta": "negative", "reference": trace},
        "without_exactness": {"expected_delta": "severe_exact_loss", "reference": trace},
        "without_code_lane_signals": {"expected_delta": "code_holdout_negative", "reference": trace},
    }


def _long_horizon_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    long_code = [row for row in rows if row["domain"] == "code" and row.get("turn_count", 0) >= 100]
    code_rows = [row for row in rows if row["domain"] == "code"]
    non_code_rows = [row for row in rows if row["domain"] != "code"]
    code_anchor = _mean_horizon_ratio(code_rows)
    non_code_anchor = _mean_horizon_ratio(non_code_rows)
    return {
        "code_case_count": len(code_rows),
        "non_code_case_count": len(non_code_rows),
        "long_code_case_count": len(long_code),
        "code_anchor_or_keep_ratio": code_anchor,
        "non_code_anchor_or_keep_ratio": non_code_anchor,
        "coding_anchor_advantage_holds": code_anchor > non_code_anchor if code_rows and non_code_rows else False,
    }


def _matched_budget_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    baseline_names = sorted(
        {
            name
            for row in rows
            for name in row["baselines"]
            if name.endswith(("8000", "32000", "128000", "200000"))
        }
    )
    return {name: _mean_metric(rows, name) for name in baseline_names}


def _horizon_diagnostics(baselines: dict[str, dict[str, Any]]) -> dict[str, Any]:
    labels = (
        baselines.get("trace_memory_rule_based", {})
        .get("diagnostics", {})
        .get("label_counts", {})
    )
    total = sum(labels.values()) or 1
    return {
        "anchor_or_keep_ratio": round((labels.get("anchor", 0) + labels.get("keep", 0)) / total, 4),
        "demote_or_remove_ratio": round((labels.get("demote", 0) + labels.get("remove", 0)) / total, 4),
        "label_counts": labels,
    }


def _mean_horizon_ratio(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    return round(sum(row["horizon_diagnostics"]["anchor_or_keep_ratio"] for row in rows) / len(rows), 4)


def _domain_horizon_thesis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_domain[str(row["domain"])].append(row)
    domain_ratios = {
        domain: round(
            sum(item["horizon_diagnostics"]["anchor_or_keep_ratio"] for item in values)
            / max(1, len(values)),
            4,
        )
        for domain, values in by_domain.items()
    }
    return {
        "claim": "agentic_search_rag_spans_should_have_shorter_survival_than_agentic_coding_spans",
        "anchor_or_keep_ratio_by_domain": domain_ratios,
        "supported_by_current_scorecard": (
            domain_ratios.get("code", 0.0)
            > max(domain_ratios.get("rag", 0.0), domain_ratios.get("search", 0.0))
            if "code" in domain_ratios and ("rag" in domain_ratios or "search" in domain_ratios)
            else None
        ),
    }


def _mean_metric(rows: list[dict[str, Any]], baseline: str) -> dict[str, float]:
    if not rows:
        return {"token_reduction": 0.0, "exact_critical_preservation": 0.0}
    metrics = [row["baselines"][baseline] for row in rows if baseline in row["baselines"]]
    if not metrics:
        return {"token_reduction": 0.0, "exact_critical_preservation": 0.0}
    return {
        "token_reduction": round(sum(item["token_reduction"] for item in metrics) / len(metrics), 4),
        "exact_critical_preservation": round(
            sum(item["exact_critical_preservation"] for item in metrics) / len(metrics),
            4,
        ),
    }


def _beats_compression_by_domain(by_domain: dict[str, Any]) -> bool:
    if not by_domain:
        return False
    for summary in by_domain.values():
        trace = summary["trace_memory_rule_based"]["exact_critical_preservation"]
        compression = summary["plain_compression"]["exact_critical_preservation"]
        if trace < compression:
            return False
    return True
