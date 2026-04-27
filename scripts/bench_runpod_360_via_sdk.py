"""Run the 360-degree bench through the ``latence-python`` SDK.

Monkey-patches ``Transport.submit`` in :mod:`scripts.bench_runpod_360` so that
every dimension (RAG / unused / code / session / rag_attribution / heatmap /
rollup / burst) goes through ``client.experimental.trace.{rag,code,rollup}``
instead of raw ``httpx`` to RunPod. The bench's pass/fail gates stay identical;
this just swaps the wire path.

Usage (same flags as the parent bench)::

    export LATENCE_API_KEY=lat_...
    python scripts/bench_runpod_360_via_sdk.py --skip concurrency --concurrency 2

The SDK client points at ``$LATENCE_BASE_URL`` (default
``https://api.latence.ai``), so the gateway is in the path. The bench's own
``--endpoint-id`` / ``--api-key`` / ``--gateway-url`` flags are accepted for
backwards compatibility but unused when routing through the SDK.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import httpx

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts import bench_runpod_360 as bench  # noqa: E402

from latence import AsyncLatence  # noqa: E402
from latence._exceptions import APIError as LatenceAPIError  # noqa: E402


_GATEWAY_KEY_REWRITES = bench._GATEWAY_KEY_REWRITES  # query→query_text, etc.

# Fields the bench sends that the SDK accepts as-is (after key rewrites).
_RAG_KEYS = {
    "response_text", "query_text", "raw_context", "chunk_ids", "support_units",
    "primary_metric", "evidence_limit", "coverage_threshold",
    "segmentation_mode", "raw_context_chunk_tokens", "response_chunk_tokens",
    "attribution_mode", "include_triangular_diagnostics", "heatmap_format",
    "verification_samples", "content_type", "risk_band_stratum", "model",
    "query_prompt_name", "document_prompt_name", "debug_dense_matrices",
    "session_id", "verbose",
}
_CODE_KEYS = _RAG_KEYS | {
    "response_language_hint", "emit_chunk_ownership", "session_state",
}
_ROLLUP_KEYS = {"turns", "session_id", "heatmap_format"}


def _translate_flat(flat: Dict[str, Any]) -> Dict[str, Any]:
    """Apply query/context/response -> canonical gateway field names."""
    out: Dict[str, Any] = {}
    for key, value in flat.items():
        out[_GATEWAY_KEY_REWRITES.get(key, key)] = value
    return out


def _dump_model(obj: Any) -> Any:
    """Recursively convert pydantic models to JSON-safe dicts/lists.

    The bench reads the ``output`` dict with ``.get(...)``; pydantic
    models also expose attributes but nested collections of models
    must be normalized to plain dicts/lists for downstream code paths
    that iterate them.
    """
    if hasattr(obj, "model_dump"):
        return obj.model_dump(exclude_none=False, mode="python")
    if isinstance(obj, list):
        return [_dump_model(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _dump_model(v) for k, v in obj.items()}
    return obj


async def _sdk_submit(
    self: bench.Transport,  # noqa: ARG001
    _client: httpx.AsyncClient,  # unused — the SDK owns its own httpx
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    """Route a bench payload through the Latence SDK.

    Re-wraps the flat gateway response in ``{"status", "output", "id"}``
    so the bench's analysis code (which expects the RunPod job shape)
    keeps working unchanged.
    """
    wall_start = time.perf_counter()
    raw_input = dict(payload.get("input") or {})
    # Always route on ``action`` / ``scoring_mode`` — the bench's payload
    # shape is the single source of truth for which lane to hit.
    action = (raw_input.pop("action", None) or "").lower()
    scoring_mode = (raw_input.pop("scoring_mode", None) or "rag").lower()

    flat = _translate_flat(raw_input)

    try:
        if action == "rollup":
            kwargs = {k: v for k, v in flat.items() if k in _ROLLUP_KEYS and v is not None}
            resp = await _SDK.experimental.trace.rollup(**kwargs)
        elif scoring_mode == "code":
            kwargs = {k: v for k, v in flat.items() if k in _CODE_KEYS and v is not None}
            resp = await _SDK.experimental.trace.code(**kwargs)
        else:
            kwargs = {k: v for k, v in flat.items() if k in _RAG_KEYS and v is not None}
            resp = await _SDK.experimental.trace.rag(**kwargs)
    except LatenceAPIError as exc:
        # Map API errors into the bench's FAILED-job envelope so the
        # per-dimension error tables render something useful.
        wall_ms = (time.perf_counter() - wall_start) * 1000.0
        return {
            "status": "FAILED",
            "output": {"success": False, "error": str(exc)},
            "id": "sdk",
            "_wall_ms": wall_ms,
        }

    output = _dump_model(resp)
    if isinstance(output, dict) and "success" not in output:
        output["success"] = True
    wall_ms = (time.perf_counter() - wall_start) * 1000.0
    return {"status": "COMPLETED", "output": output, "id": "sdk", "_wall_ms": wall_ms}


_SDK: AsyncLatence  # set in ``main()`` before the bench's _async_main runs


def main() -> int:
    api_key = os.environ.get("LATENCE_API_KEY") or os.environ.get("LATENCE_GATEWAY_KEY")
    if not api_key:
        print(
            "error: set LATENCE_API_KEY (or LATENCE_GATEWAY_KEY) to the bearer "
            "token the SDK should present to the gateway",
            file=sys.stderr,
        )
        return 2

    base_url = os.environ.get("LATENCE_BASE_URL") or "https://api.latence.ai"

    # Parse args with the bench's own parser so the flag surface stays 1:1.
    args = bench._parse_args()
    # The bench short-circuits on missing endpoint-id/api-key; give it
    # placeholders since we no longer hit RunPod directly.
    if not args.endpoint_id:
        args.endpoint_id = "via-sdk"
    if not args.api_key:
        args.api_key = "via-sdk"

    # Monkey-patch Transport.submit so every dimension goes through the SDK.
    bench.Transport.submit = _sdk_submit  # type: ignore[method-assign]

    print(f"=> routing bench through latence-python SDK (base_url={base_url})\n", flush=True)

    # Allow callers to bump the per-request timeout via env; the SDK's
    # default (30s) is far too short for cold starts on heavy lanes.
    try:
        sdk_timeout = float(os.environ.get("LATENCE_SDK_TIMEOUT", "180"))
    except ValueError:
        sdk_timeout = 180.0

    async def _runner() -> int:
        global _SDK
        async with AsyncLatence(
            api_key=api_key,
            base_url=base_url,
            timeout=sdk_timeout,
        ) as sdk:
            _SDK = sdk
            rc = await bench._async_main(args)
        return rc

    return asyncio.run(_runner())


if __name__ == "__main__":
    raise SystemExit(main())
