#!/usr/bin/env python3
"""ROI calculator for latence-trace deployments.

Pure-stdlib CLI; no third-party dependencies. Mirrors the model in
``roi-calculator.md`` so the worksheet, the script, and the README
formulas always agree.

Usage::

    python roi-calculator.py \
        --queries 12_000_000 \
        --hallucination-rate 0.04 \
        --cost-per-bad-answer 25 \
        --reduction 0.70 \
        --false-positive-rate 0.02 \
        --cost-per-review 1.50 \
        --subscription-fee 120_000 \
        --integration-cost 40_000

Outputs a JSON object with the components and the headline numbers,
and a human-readable summary on stderr.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RoiInputs:
    """Inputs to the ROI model.

    Attributes:
        queries: Annual queries (RAG / agent calls).
        hallucination_rate: Baseline rate of bad answers (0-1).
        cost_per_bad_answer: Cost (EUR) when one bad answer reaches a user.
        reduction: Fractional reduction in undetected bad answers (0-1).
        false_positive_rate: Rate at which the chosen risk band wrongly
            flags a good answer (0-1).
        cost_per_review: Cost (EUR) of one human review of a flagged answer.
        subscription_fee: Annual subscription fee (EUR).
        integration_cost: One-time + annual integration / ops cost (EUR).
    """

    queries: float
    hallucination_rate: float
    cost_per_bad_answer: float
    reduction: float
    false_positive_rate: float
    cost_per_review: float
    subscription_fee: float
    integration_cost: float


@dataclass(frozen=True)
class RoiResult:
    """Computed ROI components and headline numbers."""

    avoided_losses: float
    false_positive_cost: float
    net_annual_benefit: float
    roi_percent: float
    payback_months: float
    inputs: RoiInputs

    def to_dict(self) -> dict:
        out = asdict(self)
        out["inputs"] = asdict(self.inputs)
        return out


def compute_roi(inputs: RoiInputs) -> RoiResult:
    """Compute the ROI components from a set of inputs.

    The model is intentionally simple and conservative -- see
    ``roi-calculator.md`` for the rationale and worked examples.
    """

    if inputs.queries < 0:
        raise ValueError("queries must be non-negative")
    for rate_name in (
        "hallucination_rate",
        "reduction",
        "false_positive_rate",
    ):
        rate = getattr(inputs, rate_name)
        if not (0.0 <= rate <= 1.0):
            raise ValueError(f"{rate_name} must be in [0, 1], got {rate}")
    if inputs.subscription_fee <= 0:
        raise ValueError("subscription_fee must be > 0")

    avoided_losses = (
        inputs.queries
        * inputs.hallucination_rate
        * inputs.reduction
        * inputs.cost_per_bad_answer
    )
    false_positive_cost = (
        inputs.queries
        * inputs.false_positive_rate
        * inputs.cost_per_review
    )
    net_annual_benefit = (
        avoided_losses - false_positive_cost - inputs.integration_cost
    )

    roi_percent = (
        (net_annual_benefit - inputs.subscription_fee)
        / inputs.subscription_fee
        * 100.0
    )

    if net_annual_benefit <= 0:
        payback_months = math.inf
    else:
        payback_months = inputs.subscription_fee / (net_annual_benefit / 12.0)

    return RoiResult(
        avoided_losses=avoided_losses,
        false_positive_cost=false_positive_cost,
        net_annual_benefit=net_annual_benefit,
        roi_percent=roi_percent,
        payback_months=payback_months,
        inputs=inputs,
    )


def _format_eur(value: float) -> str:
    if value == math.inf:
        return "infinity"
    return f"EUR {value:,.0f}"


def _summary(result: RoiResult) -> str:
    payback = (
        f"{result.payback_months:.2f} months"
        if math.isfinite(result.payback_months)
        else "never (negative net benefit)"
    )
    return "\n".join(
        [
            "latence-trace ROI summary",
            "-------------------------",
            f"Avoided losses (B):      {_format_eur(result.avoided_losses)}",
            f"False-positive cost:     {_format_eur(result.false_positive_cost)}",
            f"Integration cost:        {_format_eur(result.inputs.integration_cost)}",
            f"Net annual benefit:      {_format_eur(result.net_annual_benefit)}",
            f"Subscription fee:        {_format_eur(result.inputs.subscription_fee)}",
            f"ROI:                     {result.roi_percent:,.1f}%",
            f"Payback:                 {payback}",
        ]
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="roi-calculator",
        description="Compute ROI for a latence-trace deployment.",
    )
    parser.add_argument("--queries", type=float, required=True)
    parser.add_argument("--hallucination-rate", type=float, required=True)
    parser.add_argument("--cost-per-bad-answer", type=float, required=True)
    parser.add_argument("--reduction", type=float, default=0.70)
    parser.add_argument("--false-positive-rate", type=float, default=0.02)
    parser.add_argument("--cost-per-review", type=float, default=1.50)
    parser.add_argument("--subscription-fee", type=float, required=True)
    parser.add_argument("--integration-cost", type=float, default=40_000.0)
    parser.add_argument(
        "--format",
        choices=("json", "summary", "both"),
        default="both",
        help="Output format. 'both' prints summary on stderr and JSON on stdout.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    inputs = RoiInputs(
        queries=args.queries,
        hallucination_rate=args.hallucination_rate,
        cost_per_bad_answer=args.cost_per_bad_answer,
        reduction=args.reduction,
        false_positive_rate=args.false_positive_rate,
        cost_per_review=args.cost_per_review,
        subscription_fee=args.subscription_fee,
        integration_cost=args.integration_cost,
    )
    try:
        result = compute_roi(inputs)
    except ValueError as exc:
        parser.error(str(exc))
        return 2

    if args.format in ("summary", "both"):
        print(_summary(result), file=sys.stderr)
    if args.format in ("json", "both"):
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
