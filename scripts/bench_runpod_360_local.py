"""Run the 360-degree RunPod benchmark against ``runpod/dev_app.py``.

This keeps ``bench_runpod_360.py`` focused on real RunPod/gateway targets
while letting release work prove the same dimensions against a local
production-mode wrapper started with managed vLLM services.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts import bench_runpod_360 as bench  # noqa: E402


async def _local_submit(
    self: bench.Transport,
    client: httpx.AsyncClient,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Post RunPod-shaped payloads to the local dev wrapper."""

    del self
    url = os.environ.get("LATENCE_TRACE_LOCAL_RUNPOD_URL", "http://127.0.0.1:8091/runsync")
    wall_start = time.perf_counter()
    response = await client.post(url, json=payload, headers={"Content-Type": "application/json"})
    response.raise_for_status()
    raw = response.json()
    if isinstance(raw, dict) and "status" in raw and "output" in raw:
        body = raw
    else:
        body = {"status": "COMPLETED", "output": raw, "id": "local-dev-app"}
    body["_wall_ms"] = (time.perf_counter() - wall_start) * 1000.0
    return body


def main() -> int:
    args = bench._parse_args()
    args.endpoint_id = args.endpoint_id or "local-dev-app"
    args.api_key = args.api_key or "local-dev-app"
    args.gateway_url = None
    bench.Transport.submit = _local_submit  # type: ignore[method-assign]
    print(
        "=> routing bench through local runpod/dev_app.py "
        f"({os.environ.get('LATENCE_TRACE_LOCAL_RUNPOD_URL', 'http://127.0.0.1:8091/runsync')})\n",
        flush=True,
    )
    return asyncio.run(bench._async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
