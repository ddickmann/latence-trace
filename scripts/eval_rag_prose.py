#!/usr/bin/env python3
"""Lightweight regression gate for the RAG-prose root-cause fix.

This harness exercises the deterministic decision layer of the RAG
scoring pipeline — structured-source detection, fusion, verbatim /
lexical rescue, risk-band classification — against a curated 50-case
fixture. No embedding model or NLI provider is required: per-case
*simulated* channel scores (ColBERT calibrated, literal, NLI) replace
the GPU-backed signals so the gate can run in CI on CPU in under a
second.

Usage
-----
Run directly:

    python scripts/eval_rag_prose.py

Run with verbose per-case output:

    python scripts/eval_rag_prose.py --verbose

Regenerate the golden summary (after an intentional change to the
classifier):

    python scripts/eval_rag_prose.py --regenerate

Acceptance
----------
The script exits non-zero if less than 95% of cases match their
expected band, or if a ``critical`` case (prose case expected green /
true-table case expected red) is misclassified.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))

from latence_trace.core.groundedness import (  # noqa: E402  isort: skip
    _lexical_rescue_floor,
    _resolve_effective_stratum,
    _verbatim_support_floor,
)
from latence_trace.core.nli import fuse_groundedness_v2  # noqa: E402  isort: skip
from latence_trace.core.structured import (  # noqa: E402  isort: skip
    detect_source_format,
)
from latence_trace.core.thresholds import (  # noqa: E402  isort: skip
    classify_risk_band,
    load_risk_band_policy,
)


FIXTURE_PATH = _REPO / "tests" / "fixtures" / "rag_prose_cases.jsonl"


@dataclass
class CaseResult:
    id: str
    category: str
    language: str
    expected_band: str
    actual_band: str
    source_format: Optional[str]
    stratum: Optional[str]
    fused_score: Optional[float]
    critical: bool
    structured_verification: Optional[str]


_FUSION_WEIGHTS = {
    "calibrated": 0.0,
    "literal": 0.0,
    "nli": 1.0,
    "semantic_entropy": 0.0,
    "structured": 0.0,
}


def _score_case(case: Dict[str, Any]) -> Tuple[str, Optional[float], Optional[str], Optional[str]]:
    support_text: str = case["support_text"]
    response_text: str = case["response_text"]
    structured_verification = case.get("structured_verification")
    sim_rc = float(case.get("sim_reverse_context_calibrated", 0.5))
    sim_literal = float(case.get("sim_literal_guarded", 0.5))
    sim_nli = float(case.get("sim_nli_aggregate", 0.5))
    typed_structured = case.get("sim_typed_structured")
    typed_claims_matched = int(case.get("sim_typed_claims_matched", 0))

    mode = (structured_verification or "auto").lower()
    if mode == "off":
        source_format: Optional[str] = None
    else:
        support_for_detect = support_text
        source_format = detect_source_format(
            support_for_detect,
            content_type=case.get("content_type"),
            response_text=response_text,
        )
    support_units = [{"text": support_text}]

    verbatim = _verbatim_support_floor(response_text, support_units)
    if verbatim is not None:
        sim_nli = max(sim_nli, verbatim)
    rescue = _lexical_rescue_floor(
        response_text,
        support_units,
        reverse_context_calibrated=sim_rc,
        literal_guarded=sim_literal,
        nli_aggregate=sim_nli,
    )
    if rescue is not None:
        sim_nli = max(sim_nli, rescue)

    fused = fuse_groundedness_v2(
        reverse_context_calibrated=sim_rc,
        literal_guarded=sim_literal,
        nli_aggregate=sim_nli,
        typed_structured=typed_structured,
        typed_structured_gate=True,
        source_format=source_format,
        typed_claims_matched=typed_claims_matched,
        weights=_FUSION_WEIGHTS,
    )
    stratum = _resolve_effective_stratum(
        risk_band_stratum=case.get("risk_band_stratum"),
        structured_source_format=source_format,
        structured_verification=structured_verification,
    )
    policy = load_risk_band_policy()
    band = classify_risk_band(fused, stratum=stratum, policy=policy)
    return band, fused, source_format, stratum


def run(cases: Iterable[Dict[str, Any]]) -> List[CaseResult]:
    results: List[CaseResult] = []
    for case in cases:
        band, fused, source_format, stratum = _score_case(case)
        results.append(
            CaseResult(
                id=str(case["id"]),
                category=str(case["category"]),
                language=str(case.get("language", "en")),
                expected_band=str(case["expected_band"]),
                actual_band=band,
                source_format=source_format,
                stratum=stratum,
                fused_score=fused,
                critical=bool(case.get("critical", False)),
                structured_verification=case.get("structured_verification"),
            )
        )
    return results


def summarize(results: List[CaseResult]) -> Dict[str, Any]:
    total = len(results)
    matched = sum(1 for r in results if r.actual_band == r.expected_band)
    by_category: Dict[str, Dict[str, int]] = {}
    critical_fails: List[CaseResult] = []
    fails: List[CaseResult] = []
    for r in results:
        cat = by_category.setdefault(r.category, {"total": 0, "matched": 0})
        cat["total"] += 1
        if r.actual_band == r.expected_band:
            cat["matched"] += 1
        else:
            fails.append(r)
            if r.critical:
                critical_fails.append(r)
    return {
        "total": total,
        "matched": matched,
        "match_rate": matched / float(total) if total else 1.0,
        "by_category": by_category,
        "fails": [r.__dict__ for r in fails],
        "critical_fails": [r.__dict__ for r in critical_fails],
    }


def load_cases(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Fixture not found: {path}")
    cases: List[Dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            cases.append(json.loads(stripped))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON ({exc})") from exc
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=FIXTURE_PATH)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--min-match-rate",
        type=float,
        default=0.95,
        help="Minimum fraction of cases that must match expected band.",
    )
    args = parser.parse_args()

    cases = load_cases(args.fixture)
    results = run(cases)
    summary = summarize(results)

    if args.verbose:
        for r in results:
            status = "OK" if r.actual_band == r.expected_band else "FAIL"
            print(
                f"[{status}] {r.id:<28} category={r.category:<14} "
                f"lang={r.language} expected={r.expected_band} "
                f"actual={r.actual_band} score="
                f"{'-' if r.fused_score is None else f'{r.fused_score:.3f}'} "
                f"source_format={r.source_format} stratum={r.stratum}"
            )

    print("---")
    print(
        f"Matched {summary['matched']}/{summary['total']} "
        f"({summary['match_rate'] * 100:.1f}%)"
    )
    for cat, stats in sorted(summary["by_category"].items()):
        print(f"  {cat}: {stats['matched']}/{stats['total']}")
    if summary["fails"]:
        print("Failures:")
        for fail in summary["fails"]:
            print(
                f"  - {fail['id']} ({fail['category']}, {fail['language']}): "
                f"expected={fail['expected_band']} actual={fail['actual_band']}"
            )

    exit_code = 0
    if summary["critical_fails"]:
        print("CRITICAL FAILURES present — fail immediately.")
        exit_code = 2
    elif summary["match_rate"] < args.min_match_rate:
        print(
            f"Match rate {summary['match_rate']:.1%} below "
            f"threshold {args.min_match_rate:.0%}."
        )
        exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
