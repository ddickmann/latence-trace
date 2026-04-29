"""Production verifier - 5 calls against api.latence.ai.

Hits the public gateway with realistic inputs covering five lanes:

    EX1 - rag.prose.enterprise (Veracier-shaped earnings summary)
    EX2 - rag.structured (tabular context, JSON response)
    EX3 - rag.prose.multi_claim (RAGTruth-shaped summary)
    EX4 - code.agentic_trace (3-file multi-file bundle with one grounded + two dead)
    EX5 - rollup over the above four turns

Reference values live in
``data/corpus_classifier/proof_bundle_v3_final/recal_ab/RECALIBRATION_NOTE.md``
and in the inline ``_expected`` block below. The script asserts (soft-
tolerant) that each call matches its reference and prints a pass/fail
summary. No secrets are baked in — pass the API key via
``LATENCE_API_KEY`` (or ``--api-key``).

Usage::

    LATENCE_API_KEY=lat_... python scripts/prod_verifier_5_call.py
    python scripts/prod_verifier_5_call.py --api-key lat_... --verbose
    python scripts/prod_verifier_5_call.py --session-id sess_prod_verifier_042
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    from latence import Latence
except ImportError:  # pragma: no cover - dev convenience
    print(
        "ERROR: latence-python SDK not installed. "
        "Install with `pip install latence` or "
        "`pip install -e /workspace/latence-python`.",
        file=sys.stderr,
    )
    raise


# ---------------------------------------------------------------------------
# Reference values (pinned against RECALIBRATION_NOTE.md + post-A0a fixes)
# ---------------------------------------------------------------------------
#
# Tolerances are intentionally generous because production load, NLI
# jitter, and small cache effects can move the absolute score by a few
# percent per call. The gate is: the band is correct and the key
# diagnostics land in the expected region.


@dataclass
class Expected:
    corpus_type: str
    source_prefix: str  # "rule:", "classifier:", "explicit:"
    band_in: List[str]  # set of acceptable bands
    score_min: Optional[float] = None
    score_max: Optional[float] = None
    extra_checks: List[str] = field(default_factory=list)


EXPECTED: Dict[str, Expected] = {
    "EX1_enterprise_summary": Expected(
        corpus_type="rag.prose.enterprise",
        source_prefix="classifier:",
        band_in=["green", "amber"],
        score_min=0.80,
        extra_checks=["score>=0.80"],
    ),
    "EX2_structured_table": Expected(
        corpus_type="rag.structured",
        source_prefix="rule:",
        band_in=["green", "amber"],
        score_min=0.50,
    ),
    "EX3_ragtruth_multi_claim": Expected(
        corpus_type="rag.prose.multi_claim",
        source_prefix="rule:",
        band_in=["green", "amber", "red"],
    ),
    "EX4_agentic_code_multi_file": Expected(
        corpus_type="code.agentic_trace",
        source_prefix="rule:",
        band_in=["green", "amber", "red"],
        extra_checks=[
            "code_lane.composite_score is not None",
            "file_attribution.n_files >= 3",
            "'src/db.py' in file_attribution.dead_weight_files "
            "or 'src/api.py' in file_attribution.dead_weight_files",
        ],
    ),
    "EX5_rollup": Expected(
        corpus_type="n/a",
        source_prefix="n/a",
        band_in=["n/a"],
    ),
}


# ---------------------------------------------------------------------------
# Call payloads
# ---------------------------------------------------------------------------


EX1_RAG_PROSE_ENTERPRISE = dict(
    query_text="Summarize Apple's Q3 FY2024 earnings performance.",
    raw_context=(
        "Apple announced third quarter revenue of $85.8 billion, up 5 percent "
        "year over year, with services revenue hitting an all-time high of "
        "$24.2 billion (up 14 percent). Operating margin expanded to 29.4 "
        "percent and diluted earnings per share rose to $1.40, a 33 percent "
        "increase year over year. CEO Tim Cook noted strong growth in "
        "emerging markets and a record installed base of more than 2.2 "
        "billion active devices across all product categories."
    ),
    response_text=(
        "Apple reported Q3 FY2024 revenue of $85.8 billion, up 5 percent "
        "year over year, with services setting a record at $24.2 billion. "
        "Operating margin reached 29.4 percent and EPS grew 33 percent to "
        "$1.40, reflecting strong emerging-market demand and a 2.2 billion "
        "active-device installed base."
    ),
    content_type="text",
)


EX2_RAG_STRUCTURED = dict(
    query_text="What are the failure rates and mean recovery times for each system?",
    raw_context=(
        "| system | failure_rate | mean_recovery_minutes |\n"
        "|---|---:|---:|\n"
        "| auth | 0.0023 | 4.1 |\n"
        "| checkout | 0.0008 | 2.9 |\n"
        "| search | 0.0041 | 6.4 |\n"
        "| notifications | 0.0012 | 3.3 |\n"
    ),
    response_text=(
        "Auth had a 0.23% failure rate with a 4.1 minute mean recovery, "
        "checkout 0.08% and 2.9 minutes, search 0.41% and 6.4 minutes, "
        "and notifications 0.12% and 3.3 minutes."
    ),
    content_type="text",
)


EX3_RAGTRUTH_MULTI_CLAIM = dict(
    query_text="Summarize the key points of the article.",
    raw_context=(
        "The 2024 EU AI Act introduces tiered obligations for general-"
        "purpose AI providers across the European Union. Providers must "
        "publish a model card summarising training data and capabilities, "
        "and maintain a technical documentation package ready for "
        "inspection. A copyright opt-out mechanism for text and data "
        "mining is mandatory for all covered models. Models above the "
        "compute threshold face additional red-teaming obligations and "
        "serious incident reporting duties within fifteen days of "
        "detection. Penalties scale up to 7 percent of worldwide "
        "turnover for the most severe infringements."
    ),
    response_text=(
        "The 2024 EU AI Act introduces tiered obligations for general-"
        "purpose AI providers across the European Union. Providers must "
        "publish a model card and maintain technical documentation. A "
        "copyright opt-out for text and data mining is mandatory. Models "
        "above the compute threshold face additional red-teaming "
        "obligations and serious incident reporting duties within "
        "fifteen days. Penalties scale up to 7 percent of worldwide "
        "turnover."
    ),
    content_type="text",
)


EX4_AGENTIC_CODE = dict(
    query_text="Add input validation to the login endpoint.",
    raw_context=(
        "# file: src/auth.py\n"
        "from typing import Optional\n"
        "def login(username: str, password: str) -> Optional[dict]:\n"
        "    user = find_user(username)\n"
        "    if user and verify_password(user, password):\n"
        "        return make_session(user)\n"
        "    return None\n"
        "\n"
        "# file: src/db.py\n"
        "class Db:\n"
        "    def connect(self) -> None: ...\n"
        "    def execute(self, sql: str) -> list[dict]: ...\n"
        "    def close(self) -> None: ...\n"
        "\n"
        "# file: src/api.py\n"
        "def register_routes(app):\n"
        "    app.route('/health', lambda: 'ok')\n"
        "    app.route('/metrics', lambda: metrics_dump())\n"
        "    return app\n"
    ),
    response_text=(
        "I'll add basic validation to the login flow in src/auth.py:\n"
        "\n"
        "```python\n"
        "def login(username: str, password: str) -> Optional[dict]:\n"
        "    if not username or not password:\n"
        "        return None\n"
        "    if len(username) > 128 or len(password) > 256:\n"
        "        return None\n"
        "    user = find_user(username)\n"
        "    if user and verify_password(user, password):\n"
        "        return make_session(user)\n"
        "    return None\n"
        "```\n"
        "\n"
        "This keeps the existing success path unchanged and rejects "
        "empty or oversized inputs before any DB lookup."
    ),
    content_type="code",
)


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


def _fmt_band(resp: Any) -> str:
    return f"{getattr(resp, 'band', '?')} (score={getattr(resp, 'score', '?')})"


def _corpus_route(resp: Any) -> Dict[str, Any]:
    route = getattr(resp, "corpus_route", None)
    if route is None:
        return {}
    return route.model_dump() if hasattr(route, "model_dump") else dict(route)


def _check(label: str, expected: Expected, resp: Any, verbose: bool) -> bool:
    route = _corpus_route(resp)
    band = getattr(resp, "band", None)
    score = getattr(resp, "score", None)

    problems: List[str] = []

    if expected.corpus_type != "n/a":
        corpus_type = route.get("corpus_type") if route else None
        if corpus_type != expected.corpus_type:
            problems.append(
                f"corpus_type={corpus_type!r} != expected {expected.corpus_type!r}"
            )
        source = (route.get("source") or "") if route else ""
        if not source.startswith(expected.source_prefix):
            problems.append(
                f"source={source!r} does not start with {expected.source_prefix!r}"
            )

    if expected.band_in != ["n/a"]:
        if band not in expected.band_in:
            problems.append(f"band={band!r} not in {expected.band_in}")

    if expected.score_min is not None and score is not None and score < expected.score_min:
        problems.append(f"score={score:.3f} < min {expected.score_min}")
    if expected.score_max is not None and score is not None and score > expected.score_max:
        problems.append(f"score={score:.3f} > max {expected.score_max}")

    for check in expected.extra_checks:
        try:
            if not eval(check, {"response": resp, **resp.__dict__}):  # noqa: S307
                problems.append(f"extra_check failed: {check}")
        except Exception as exc:  # noqa: BLE001
            problems.append(f"extra_check error: {check} -> {exc}")

    status = "PASS" if not problems else "FAIL"
    print(f"  [{status}] {label}: {_fmt_band(resp)}")
    if route:
        print(
            f"          route: {route.get('corpus_type')} "
            f"via {route.get('source')} conf={route.get('confidence')}"
        )
    if verbose:
        print(f"          raw: {json.dumps(route, indent=2, default=str)}")
    for p in problems:
        print(f"          - {p}")
    return not problems


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="TRACE production 5-call verifier.")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("LATENCE_API_KEY"),
        help="API key (default: $LATENCE_API_KEY).",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("LATENCE_BASE_URL", "https://api.latence.ai"),
    )
    parser.add_argument(
        "--session-id",
        default=f"prod-verifier-{int(time.time())}",
        help="Session id shared across all five calls.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    if not args.api_key:
        print("ERROR: set LATENCE_API_KEY or pass --api-key.", file=sys.stderr)
        return 2

    client = Latence(api_key=args.api_key, base_url=args.base_url)

    print("TRACE prod verifier - 5 calls")
    print(f"  base_url   = {args.base_url}")
    print(f"  session_id = {args.session_id}")
    print()

    results: Dict[str, bool] = {}

    print("EX1 - rag.prose.enterprise (earnings summary)")
    r1 = client.trace.rag(
        session_id=args.session_id,
        turn_index=0,
        **EX1_RAG_PROSE_ENTERPRISE,
    )
    results["EX1_enterprise_summary"] = _check(
        "EX1", EXPECTED["EX1_enterprise_summary"], r1, args.verbose
    )

    print("EX2 - rag.structured (tabular context)")
    r2 = client.trace.rag(
        session_id=args.session_id,
        turn_index=1,
        **EX2_RAG_STRUCTURED,
    )
    results["EX2_structured_table"] = _check(
        "EX2", EXPECTED["EX2_structured_table"], r2, args.verbose
    )

    print("EX3 - rag.prose.multi_claim (RAGTruth-shaped)")
    r3 = client.trace.rag(
        session_id=args.session_id,
        turn_index=2,
        **EX3_RAGTRUTH_MULTI_CLAIM,
    )
    results["EX3_ragtruth_multi_claim"] = _check(
        "EX3", EXPECTED["EX3_ragtruth_multi_claim"], r3, args.verbose
    )

    print("EX4 - code.agentic_trace (3-file multi-file bundle)")
    r4 = client.trace.code(
        session_id=args.session_id,
        turn_index=3,
        **EX4_AGENTIC_CODE,
    )
    results["EX4_agentic_code_multi_file"] = _check(
        "EX4", EXPECTED["EX4_agentic_code_multi_file"], r4, args.verbose
    )

    print("EX5 - rollup across all four turns")
    try:
        r5 = client.trace.rollup(session_id=args.session_id)
        print(f"  [PASS] EX5 rollup: turns_processed={r5.turns_processed}")
        results["EX5_rollup"] = True
    except Exception as exc:  # noqa: BLE001
        print(f"  [FAIL] EX5 rollup: {exc}")
        results["EX5_rollup"] = False

    print()
    n_pass = sum(results.values())
    n_total = len(results)
    print(f"SUMMARY: {n_pass}/{n_total} passed")
    for label, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
