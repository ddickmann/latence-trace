"""Live RunPod endpoint smoke test (skipped unless credentials are present).

Runs a *small subset* of ``scripts/bench_runpod_360.py`` against the
serverless endpoint the user provisions, as a sanity check that the
deployed service answers on every lane.

This test is intentionally **gated by environment variables** so CI
and local unit runs stay offline:

    RUNPOD_ENDPOINT_ID  — e.g. ``campegd1dctnx2``
    RUNPOD_API_KEY      — user-supplied bearer token

Examples::

    RUNPOD_ENDPOINT_ID=campegd1dctnx2 \\
    RUNPOD_API_KEY=rpa_... \\
    pytest -k live_endpoint -s

For a full 360-degree sweep, invoke the bench script directly::

    python scripts/bench_runpod_360.py \\
        --endpoint-id $RUNPOD_ENDPOINT_ID \\
        --api-key $RUNPOD_API_KEY \\
        --dump reports/runpod_360.json
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_ENDPOINT = os.environ.get("RUNPOD_ENDPOINT_ID")
_API_KEY = os.environ.get("RUNPOD_API_KEY")
_SKIP_REASON = "set RUNPOD_ENDPOINT_ID and RUNPOD_API_KEY to run live endpoint tests"

live_only = pytest.mark.skipif(
    not (_ENDPOINT and _API_KEY),
    reason=_SKIP_REASON,
)


def _import_bench():
    # Delayed import so local runs without httpx still collect cleanly.
    from scripts import bench_runpod_360  # type: ignore[import-not-found]

    return bench_runpod_360


@live_only
def test_live_endpoint_rag_smoke() -> None:
    """RAG lane returns a COMPLETED score for the G1 handcrafted case."""

    bench = _import_bench()
    import asyncio

    async def run() -> None:
        import httpx

        async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=15.0)) as client:
            transport = bench.Transport(endpoint_id=_ENDPOINT, api_key=_API_KEY)
            from research.triangular_maxsim.cases import CASES

            g1 = next(c for c in CASES if c.id == "G1")
            dim = await bench._dim_rag(client, transport, [g1], concurrency=1)
            assert dim["errors"] == [], dim["errors"]
            row = dim["rows"][0]
            assert row["id"] == "G1"
            assert row["score"] is not None

    asyncio.run(run())


@live_only
def test_live_endpoint_code_smoke() -> None:
    """Code lane returns a COMPLETED phantom diagnostics for a grounded case."""

    bench = _import_bench()
    import asyncio

    async def run() -> None:
        import httpx

        async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=15.0)) as client:
            transport = bench.Transport(endpoint_id=_ENDPOINT, api_key=_API_KEY)
            from research.triangular_maxsim.coding.code_cases import HANDCRAFTED_CASES

            grounded = next(c for c in HANDCRAFTED_CASES if c.label == "grounded")
            dim = await bench._dim_code(client, transport, [grounded], concurrency=1)
            assert dim["errors"] == [], dim["errors"]
            row = dim["rows"][0]
            assert row["id"] == grounded.id
            assert row["composite_score"] is not None
            # Grounded cases must not trip the AST phantom verdict.
            assert row["ast_phantom_verdict"] is not True

    asyncio.run(run())


@live_only
def test_live_endpoint_session_smoke() -> None:
    """Session-state round-trip produces monotonic total_turns across 2 turns."""

    bench = _import_bench()
    import asyncio

    async def run() -> None:
        import httpx

        async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=15.0)) as client:
            transport = bench.Transport(endpoint_id=_ENDPOINT, api_key=_API_KEY)
            dim = await bench._dim_session(client, transport, turns=2)
            assert dim["errors"] == [], dim["errors"]
            assert dim["monotonic_turns"], dim["turn_records"]
            assert dim["produced_next_state"], dim["turn_records"]

    asyncio.run(run())
