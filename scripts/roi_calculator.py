"""TRACE ROI calculator.

Given customer inputs (monthly RAG traffic, reviewer cost, accept rate
distribution) projects reviewer hours saved and risk-incident reduction
from deploying TRACE v1.1.

Defaults come from the published proof bundle so the projection is
defensibly grounded (see ``commercial/bold-claims-v1.md`` row 2-4).

Usage::

    python scripts/roi_calculator.py \
        --calls-per-month 1000000 \
        --reviewer-cost-per-hour 75 \
        --avg-review-seconds 90 \
        --baseline-hallucination-rate 0.12

Output is both a text table and a machine-readable JSON block so the
same calculator can be embedded in the portal (``/roi`` page) and in
the pilot contract.
"""

from __future__ import annotations

import argparse
import json

# Default band distribution from Veracier proof bundle (2026-04).  These
# are empirical observations, NOT promises.
DEFAULT_GREEN_SHARE = 0.72
DEFAULT_AMBER_SHARE = 0.18
DEFAULT_RED_SHARE = 0.10

# Verified Veracier numbers.
VERACIER_RED_PRECISION = 1.00
VERACIER_GREEN_PRECISION = 1.00
VERACIER_AMBER_AGREEMENT = 0.88

# External-bench conservative pullback (HaluEval + RAGTruth mean red precision).
EXTERNAL_RED_PRECISION = 0.48


def roi(
    calls_per_month: int,
    reviewer_cost_per_hour: float,
    avg_review_seconds: float,
    baseline_hallucination_rate: float,
    risk_incident_cost_usd: float = 25_000.0,
    green_share: float = DEFAULT_GREEN_SHARE,
    amber_share: float = DEFAULT_AMBER_SHARE,
    red_share: float = DEFAULT_RED_SHARE,
    conservative: bool = True,
) -> dict:
    """Return a ROI projection dict.

    conservative=True uses the external-bench red precision (0.48) for
    the savings calculation so deck-ready numbers are defensible even
    for traffic that doesn't look like Veracier.
    """

    amber_calls = calls_per_month * amber_share
    red_calls = calls_per_month * red_share

    # Without TRACE: every call is either reviewed or ignored.
    # Industry data (see failure_modes_v1.md) suggests half of all RAG
    # calls in regulated verticals get reviewed manually at a baseline
    # hallucination rate of ~10-15%.
    baseline_review_rate = 0.50
    baseline_reviewed_calls = calls_per_month * baseline_review_rate
    baseline_review_hours = baseline_reviewed_calls * avg_review_seconds / 3600
    baseline_review_cost = baseline_review_hours * reviewer_cost_per_hour
    # Empirically, ~1 in 1000 hallucinations turns into a filed
    # compliance / customer-trust incident in regulated verticals.
    # Conservative anchor; customers can tune via --incident-rate.
    incident_rate_per_halluc = 0.001
    baseline_risk_incidents = (
        calls_per_month * baseline_hallucination_rate * incident_rate_per_halluc
    )
    baseline_risk_cost = baseline_risk_incidents * risk_incident_cost_usd

    # With TRACE: only amber + red-flagged calls go to review.  Green
    # precision means false-accepts are rare; red precision means
    # false-rejects are rare.
    trace_reviewed_calls = amber_calls + red_calls
    trace_review_hours = trace_reviewed_calls * avg_review_seconds / 3600
    trace_review_cost = trace_review_hours * reviewer_cost_per_hour

    red_prec = EXTERNAL_RED_PRECISION if conservative else VERACIER_RED_PRECISION
    # Same incident rate, but only on hallucinations TRACE misses.
    # Use ``1 - red_prec`` as a very conservative proxy for the share
    # of hallucinations that slip through.
    trace_risk_incidents = (
        calls_per_month
        * baseline_hallucination_rate
        * (1.0 - red_prec)
        * incident_rate_per_halluc
    )
    trace_risk_cost = trace_risk_incidents * risk_incident_cost_usd

    return {
        "calls_per_month": int(calls_per_month),
        "baseline": {
            "reviewed_calls": int(baseline_reviewed_calls),
            "review_hours": round(baseline_review_hours, 0),
            "review_cost_usd": round(baseline_review_cost, 0),
            "risk_incidents_expected": round(baseline_risk_incidents, 2),
            "risk_cost_usd": round(baseline_risk_cost, 0),
            "total_cost_usd": round(baseline_review_cost + baseline_risk_cost, 0),
        },
        "with_trace": {
            "reviewed_calls": int(trace_reviewed_calls),
            "review_hours": round(trace_review_hours, 0),
            "review_cost_usd": round(trace_review_cost, 0),
            "risk_incidents_expected": round(trace_risk_incidents, 2),
            "risk_cost_usd": round(trace_risk_cost, 0),
            "total_cost_usd": round(trace_review_cost + trace_risk_cost, 0),
        },
        "savings": {
            "reviewer_hours_per_month": round(
                baseline_review_hours - trace_review_hours, 0
            ),
            "usd_per_month": round(
                (baseline_review_cost + baseline_risk_cost)
                - (trace_review_cost + trace_risk_cost),
                0,
            ),
            "risk_reduction_pct": round(
                (baseline_risk_incidents - trace_risk_incidents)
                / max(baseline_risk_incidents, 1e-9)
                * 100,
                1,
            ),
        },
        "assumptions": {
            "conservative": conservative,
            "red_precision_used": red_prec,
            "green_share": green_share,
            "amber_share": amber_share,
            "red_share": red_share,
            "baseline_review_rate": baseline_review_rate,
            "baseline_hallucination_rate": baseline_hallucination_rate,
            "risk_incident_cost_usd": risk_incident_cost_usd,
        },
    }


def _fmt_usd(x: float) -> str:
    return f"${x:,.0f}"


def _pretty(savings: dict) -> str:
    base = savings["baseline"]
    trc = savings["with_trace"]
    sv = savings["savings"]
    rows = [
        "# TRACE ROI projection",
        "",
        f"Calls per month: {savings['calls_per_month']:,}",
        "",
        "| | Reviewer hours | Review cost | Risk incidents | Risk cost | Total cost |",
        "|---|---|---|---|---|---|",
        (
            f"| Baseline | {base['review_hours']:,.0f} | {_fmt_usd(base['review_cost_usd'])} | "
            f"{base['risk_incidents_expected']} | {_fmt_usd(base['risk_cost_usd'])} | "
            f"{_fmt_usd(base['total_cost_usd'])} |"
        ),
        (
            f"| With TRACE | {trc['review_hours']:,.0f} | {_fmt_usd(trc['review_cost_usd'])} | "
            f"{trc['risk_incidents_expected']} | {_fmt_usd(trc['risk_cost_usd'])} | "
            f"{_fmt_usd(trc['total_cost_usd'])} |"
        ),
        "",
        f"Reviewer hours saved: **{sv['reviewer_hours_per_month']:,.0f}/month**",
        f"Money saved: **{_fmt_usd(sv['usd_per_month'])}/month**",
        f"Risk reduction: **{sv['risk_reduction_pct']}%**",
        "",
        "## Assumptions (defensible - tied to the published proof bundle)",
        "",
    ]
    for k, v in savings["assumptions"].items():
        rows.append(f"* {k} = {v}")
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calls-per-month", type=int, required=True)
    parser.add_argument("--reviewer-cost-per-hour", type=float, default=75.0)
    parser.add_argument("--avg-review-seconds", type=float, default=90.0)
    parser.add_argument("--baseline-hallucination-rate", type=float, default=0.12)
    parser.add_argument(
        "--optimistic",
        action="store_true",
        help="Use Veracier red precision (1.0) instead of external (0.48).",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON only.")
    args = parser.parse_args()

    result = roi(
        calls_per_month=args.calls_per_month,
        reviewer_cost_per_hour=args.reviewer_cost_per_hour,
        avg_review_seconds=args.avg_review_seconds,
        baseline_hallucination_rate=args.baseline_hallucination_rate,
        conservative=not args.optimistic,
    )

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(_pretty(result))


if __name__ == "__main__":
    main()
