"""PA4 concurrency tests.

Verifies that the FastAPI ``/groundedness`` handler:

1. Off-loads the synchronous, CPU/GPU-bound scoring call to the
   threadpool so other requests (e.g. ``/healthz``) and concurrent
   ``/groundedness`` calls keep making progress.
2. Bounds inflight scoring work with a semaphore so a burst can't
   exceed the documented max-concurrency (the load-test budget).
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from typing import List, Sequence

import httpx
import pytest
import torch
from fastapi import FastAPI

from latence_trace.api.routes import create_router
from latence_trace.api.service import GroundednessService


class _SlowDeterministicEncoder:
    """Encoder stub that sleeps to simulate the encoder/NLI/reranker
    runtime so concurrency wins are observable without booting the
    real models. Tracks peak inflight calls so the inflight semaphore
    test can assert the cap.
    """

    def __init__(self, sleep_ms: int = 80, hidden: int = 32):
        self._sleep_ms = sleep_ms
        self._hidden = hidden
        self.inflight = 0
        self.peak_inflight = 0
        self._lock = threading.Lock()

    @property
    def model_name_or_path(self) -> str:
        return "slow-deterministic-encoder"

    @property
    def model_name(self) -> str:
        return self.model_name_or_path

    def encode(
        self,
        sentences: Sequence[str],
        *,
        is_query: bool = False,
        prompt_name: str | None = None,
        **kwargs,
    ) -> List[torch.Tensor]:
        with self._lock:
            self.inflight += 1
            self.peak_inflight = max(self.peak_inflight, self.inflight)
        try:
            time.sleep(self._sleep_ms / 1000.0)
            outs: List[torch.Tensor] = []
            for sentence in sentences:
                token_count = max(2, min(8, len((sentence or "x").split())))
                seed = abs(hash(sentence)) % (2**31 - 1)
                generator = torch.Generator().manual_seed(seed)
                outs.append(torch.randn(token_count, self._hidden, generator=generator))
            return outs
        finally:
            with self._lock:
                self.inflight -= 1


def _build_app(encoder: _SlowDeterministicEncoder, *, max_inflight: int = 4) -> FastAPI:
    os.environ["LATENCE_TRACE_MAX_INFLIGHT"] = str(max_inflight)
    os.environ["VOYAGER_GROUNDEDNESS_NLI_ENABLED"] = "0"
    service = GroundednessService(encoder_factory=lambda _name: encoder)
    app = FastAPI()
    app.include_router(create_router(lambda: service))
    return app


def _payload() -> dict:
    return {
        "query_text": "Where was Heinrich born?",
        "raw_context": "Heinrich was born in 1851 in Augsburg.",
        "response_text": "Heinrich was born in Augsburg in 1851.",
        "primary_metric": "reverse_context",
    }


def _async_client(app: FastAPI) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


def test_groundedness_handler_runs_on_threadpool_unblocking_async_routes():
    encoder = _SlowDeterministicEncoder(sleep_ms=120)
    app = _build_app(encoder, max_inflight=4)

    async def _run() -> tuple[float, float, int]:
        async with _async_client(app) as client:
            warm = await client.post("/groundedness", json=_payload(), timeout=20.0)
            assert warm.status_code == 200
            loop = asyncio.get_running_loop()
            start = loop.time()
            score_task = asyncio.create_task(
                client.post("/groundedness", json=_payload(), timeout=20.0)
            )
            await asyncio.sleep(0.02)
            health_start = loop.time()
            health_resp = await client.get("/healthz", timeout=5.0)
            health_elapsed = loop.time() - health_start
            score_resp = await score_task
            total_elapsed = loop.time() - start
        # The healthz response itself is informational; we assert below
        # on the elapsed time to prove the threadpool offload works.
        assert health_resp.status_code == 200
        return total_elapsed, health_elapsed, score_resp.status_code

    total, health_elapsed, status = asyncio.run(_run())
    assert status == 200
    assert health_elapsed < 0.05, (
        f"/healthz blocked for {health_elapsed*1000:.1f} ms while /groundedness ran; "
        "the scoring handler is starving the event loop"
    )
    assert total < 1.5


def test_inflight_semaphore_bounds_concurrent_scoring_requests():
    encoder = _SlowDeterministicEncoder(sleep_ms=60)
    cap = 3
    app = _build_app(encoder, max_inflight=cap)

    async def _run() -> int:
        async with _async_client(app) as client:
            warm = await client.post("/groundedness", json=_payload(), timeout=20.0)
            assert warm.status_code == 200
            encoder.peak_inflight = 0
            burst = [
                client.post("/groundedness", json=_payload(), timeout=30.0)
                for _ in range(cap * 3)
            ]
            responses = await asyncio.gather(*burst)
            for r in responses:
                assert r.status_code == 200
        return encoder.peak_inflight

    peak = asyncio.run(_run())
    assert peak <= cap, (
        f"peak inflight encoder calls = {peak}, but LATENCE_TRACE_MAX_INFLIGHT={cap}; "
        "the inflight semaphore is not enforcing the cap"
    )


def test_readyz_reports_max_inflight_for_operators():
    encoder = _SlowDeterministicEncoder(sleep_ms=5)
    app = _build_app(encoder, max_inflight=7)
    from latence_trace.kernels.warmup import warm_all
    warm_all("balanced", force=True)

    async def _run():
        async with _async_client(app) as client:
            return await client.get("/readyz", timeout=5.0)

    response = asyncio.run(_run())
    assert response.status_code == 200
    body = response.json()
    assert body["max_inflight"] == 7


@pytest.fixture(autouse=True)
def _restore_env():
    saved_inflight = os.environ.get("LATENCE_TRACE_MAX_INFLIGHT")
    saved_nli = os.environ.get("VOYAGER_GROUNDEDNESS_NLI_ENABLED")
    yield
    if saved_inflight is None:
        os.environ.pop("LATENCE_TRACE_MAX_INFLIGHT", None)
    else:
        os.environ["LATENCE_TRACE_MAX_INFLIGHT"] = saved_inflight
    if saved_nli is None:
        os.environ.pop("VOYAGER_GROUNDEDNESS_NLI_ENABLED", None)
    else:
        os.environ["VOYAGER_GROUNDEDNESS_NLI_ENABLED"] = saved_nli
