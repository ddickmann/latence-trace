"""Produce the TRACE Memory novelty and production-readiness report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from latence_trace.memory.scorecard import promotion_decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scorecard", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scorecard = json.loads(args.scorecard.read_text(encoding="utf-8"))
    decision = promotion_decision(scorecard)
    report = {
        "decision": decision,
        "novelty_interpretation": _novelty_interpretation(decision),
        "production_rule": (
            "Keep TRACE Memory shadow-only unless approved_for_production_coupling is true. "
            "Even then, production coupling must remain opt-in and reversible."
        ),
        "scorecard_summary": {
            "case_count": scorecard.get("case_count", 0),
            "by_dataset": scorecard.get("by_dataset", {}),
            "by_domain": scorecard.get("by_domain", {}),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _novelty_interpretation(decision: dict) -> str:
    if decision.get("approved_for_production_coupling"):
        return "TRACE-specific span survival shows enough incremental value for opt-in production coupling."
    reasons = decision.get("reasons", {})
    if not reasons.get("beats_generic_memory_gate"):
        return "Not novel yet: TRACE Memory has not beaten generic compression/vector/summary baselines."
    if not reasons.get("exact_critical_gate"):
        return "Not production-ready: exact-critical preservation is below gate."
    if not reasons.get("token_reduction_gate"):
        return "Not product-worthy yet: token reduction is below gate."
    return "Not approved: scorecard gates are incomplete."


if __name__ == "__main__":
    main()
