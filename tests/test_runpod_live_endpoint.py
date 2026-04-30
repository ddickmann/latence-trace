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


def _import_bare_user_smoke():
    from scripts import qualitative_bare_user_smoke  # type: ignore[import-not-found]

    return qualitative_bare_user_smoke


def _good_runtime_features() -> dict[str, float]:
    return {
        "v1_score": 0.99,
        "v1_nli_aggregate": 0.9,
        "reverse_context": 0.99,
        "groundedness_v2": 0.99,
        "literal_guarded": 0.99,
        "literal_mismatch_count": 0.0,
        "context_coverage_ratio": 1.0,
        "context_unused_ratio": 0.0,
        "dead_weight_ratio": 0.0,
        "token_mean": 0.99,
        "token_bottom10": 0.99,
        "token_saturation_rate": 0.0,
        "calibrated_mean": 0.99,
        "nli_token_mean": 0.9,
        "literal_coverage": 1.0,
        "atom_match": 1.0,
        "claim_count": 1.0,
        "unsupported_claim_fraction": 0.0,
        "schema_match": 1.0,
        "api_symbol_match": 1.0,
        "trajectory_order_match": 1.0,
        "test_outcome_match": 1.0,
        "numeric_coverage": 1.0,
        "numeric_count": 1.0,
        "coverage_label_mean": 1.0,
        "dead_weight_label_mean": 0.0,
        "support_unit_label_mean": 1.0,
        "min_claim_coverage": 1.0,
        "row_alignment": 1.0,
        "column_alignment": 1.0,
        "value_alignment": 1.0,
        "unit_alignment": 1.0,
        "date_alignment": 1.0,
        "numeric_tolerance_match": 1.0,
        "schema_alias_match": 1.0,
        "cell_provenance_match": 1.0,
        "identifier_coverage": 1.0,
        "identifier_count": 1.0,
        "file_alignment": 1.0,
        "symbol_alignment": 1.0,
        "test_alignment": 1.0,
        "patch_alignment": 1.0,
        "temporal_order_alignment": 1.0,
        "claim_atom_coverage": 1.0,
        "unsupported_atom_rate": 0.0,
        "missing_command_evidence": 0.0,
        "api_call_alignment": 1.0,
        "edit_intent_alignment": 1.0,
    }


def _bad_runtime_features() -> dict[str, float]:
    features = {key: 0.0 for key in _good_runtime_features()}
    features.update(
        {
            "literal_mismatch_count": 10.0,
            "context_unused_ratio": 1.0,
            "dead_weight_ratio": 1.0,
            "token_saturation_rate": 1.0,
            "unsupported_claim_fraction": 1.0,
            "claim_count": 1.0,
            "numeric_count": 1.0,
            "identifier_count": 1.0,
            "missing_command_evidence": 1.0,
        }
    )
    return features


def _structured_allow_features() -> dict[str, float]:
    features = _good_runtime_features()
    features.update(
        {
            "v1_score": 0.0,
            "reverse_context": 0.0,
            "groundedness_v2": 0.0,
            "literal_guarded": 0.0,
            "claim_count": 3.0,
        }
    )
    return features


def _trajectory_allow_features() -> dict[str, float]:
    return {
        "file_alignment": 1.0,
        "symbol_alignment": 1.0,
        "test_outcome_alignment": 1.0,
        "patch_alignment": 1.0,
        "temporal_order_alignment": 1.0,
        "claim_atom_coverage": 1.0,
        "unsupported_atom_rate": 0.0,
        "phantom_symbol_rate": 0.0,
        "missing_command_evidence": 0.0,
        "literal_match_rate": 1.0,
        "literal_mismatch_rate": 0.0,
        "identifier_query_overlap": 1.0,
        "identifier_query_absent_rate": 0.0,
        "warning_identifier_rate": 0.0,
        "reverse_context": 0.95,
        "consensus_hardened": 0.95,
        "groundedness_v2": 0.95,
        "triangular": 0.95,
        "context_attribution_ratio": 1.0,
        "context_uncertain_ratio": 0.0,
        "dead_weight_ratio": 0.0,
        "support_usage_rate": 1.0,
        "context_token_log": 5.0,
        "multi_cell_reverse_min": 0.95,
        "multi_cell_reverse_max": 0.95,
        "multi_cell_reverse_std": 0.0,
    }


def _trajectory_block_features() -> dict[str, float]:
    features = {key: 1.0 - value for key, value in _trajectory_allow_features().items()}
    features["context_token_log"] = 5.0
    features["missing_command_evidence"] = 0.0
    return features


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


@live_only
def test_live_endpoint_runtime_decision_feature_maps() -> None:
    """Explicit v2 feature maps produce autonomous allow and block records."""

    bench = _import_bench()
    import asyncio

    async def run() -> None:
        import httpx

        async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=15.0)) as client:
            transport = bench.Transport(endpoint_id=_ENDPOINT, api_key=_API_KEY)
            cases = [
                (
                    "rag.structured",
                    "runtime_head_features",
                    "rag",
                    _structured_allow_features(),
                    _bad_runtime_features(),
                ),
                (
                    "code.agentic_trace",
                    "trajectory_features",
                    "code",
                    _trajectory_allow_features(),
                    _trajectory_block_features(),
                ),
            ]
            for class_key, feature_key, mode, allow_features, block_features in cases:
                for label, expected_action, features in [
                    ("good", "allow", allow_features),
                    ("bad", "block", block_features),
                ]:
                    payload_input = {
                        "scoring_mode": mode,
                        "corpus_type": class_key,
                        "query_text": "verify runtime decision",
                        "raw_context": (
                            "The exact supported answer is present in this evidence. "
                            "customer_1001 amount_usd=1200 USD status approved."
                        ),
                        "response_text": "The exact supported answer is present in this evidence.",
                        feature_key: features,
                    }
                    if mode == "code":
                        payload_input["response_language_hint"] = "python"

                    body = await transport.submit(client, {"input": payload_input})
                    assert body.get("status") == "COMPLETED", (class_key, label, body)
                    output = body.get("output") or {}
                    decision = output.get("runtime_decision")
                    assert decision is not None, (class_key, label, output)
                    assert decision["class_key"] == class_key
                    assert decision["action"] == expected_action
                    assert decision["head_enabled"] is True
                    assert decision["head_score"] is not None

    asyncio.run(run())


@live_only
def test_live_endpoint_bare_text_autodiscovery() -> None:
    """Bare user text routes to all six classes without caller feature maps."""

    smoke = _import_bare_user_smoke()
    import asyncio

    async def run() -> None:
        rc = await smoke.run(_ENDPOINT, _API_KEY)
        assert rc == 0

    asyncio.run(run())
