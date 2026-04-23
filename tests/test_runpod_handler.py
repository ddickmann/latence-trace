from __future__ import annotations

import asyncio
import importlib.util
import sys
import threading
import time
from pathlib import Path

from latence_trace.api.models import (
    AttributionMode,
    CollectionKind,
    GroundednessEligibility,
    GroundednessResponse,
    GroundednessScores,
)

_RUNPOD_DIR = Path(__file__).resolve().parents[1] / "runpod"
if str(_RUNPOD_DIR) not in sys.path:
    sys.path.insert(0, str(_RUNPOD_DIR))

_SPEC = importlib.util.spec_from_file_location(
    "latence_trace_runpod_handler",
    _RUNPOD_DIR / "handler.py",
)
assert _SPEC is not None and _SPEC.loader is not None
runpod_handler = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = runpod_handler
_SPEC.loader.exec_module(runpod_handler)


class _SlowService:
    def __init__(self, sleep_ms: int = 80) -> None:
        self._sleep_ms = sleep_ms
        self._lock = threading.Lock()
        self.inflight = 0
        self.peak_inflight = 0
        self.calls = 0

    def groundedness(self, request) -> GroundednessResponse:
        with self._lock:
            self.calls += 1
            self.inflight += 1
            self.peak_inflight = max(self.peak_inflight, self.inflight)
        try:
            time.sleep(self._sleep_ms / 1000.0)
            return GroundednessResponse(
                collection="latence-trace",
                mode="raw_context",
                model="test-model",
                scores=GroundednessScores(
                    primary_name="reverse_context",
                    primary_score=0.9,
                    reverse_context=0.9,
                    risk_band="low",
                ),
                response_tokens=[],
                support_units=[],
                top_evidence=[],
                eligibility=GroundednessEligibility(
                    collection_kind=CollectionKind.LATE_INTERACTION,
                    vector_source="encoded_raw_context",
                    storage_compression=None,
                    quantization_mode=None,
                    dequantized=True,
                    user_facing_supported=True,
                    warnings=[],
                ),
                time_ms=float(self._sleep_ms),
                attribution_mode=AttributionMode.CLOSED_BOOK,
            )
        finally:
            with self._lock:
                self.inflight -= 1


def _config(*, profile: str = "quality", max_concurrency: int = 2):
    return runpod_handler.WorkerConfig(
        profile=profile,
        version="test",
        request_timeout_s=5,
        max_concurrency=max_concurrency,
        collection_label="latence-trace",
        service_device="cpu",
        docs_url="",
        colbert_model="lightonai/LateOn",
        colbert_port=18001,
        colbert_gpu_mem=0.34,
        colbert_max_model_len=8192,
        colbert_max_num_seqs=192,
        colbert_max_batched_tokens=32768,
        nli_model="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
        nli_port=18002,
        nli_gpu_mem=0.24,
        nli_max_model_len=512,
        nli_max_num_seqs=256,
        nli_max_batched_tokens=8192,
    )


async def _burst(size: int) -> list[dict]:
    payload = {
        "input": {
            "query_text": "Where was Heinrich born?",
            "raw_context": "Heinrich was born in 1851 in Augsburg.",
            "response_text": "Heinrich was born in Augsburg in 1851.",
        }
    }
    return await asyncio.gather(*(runpod_handler.handler(payload) for _ in range(size)))


def test_create_config_uses_fixed_runpod_max_concurrency(monkeypatch) -> None:
    monkeypatch.setenv("MAX_CONCURRENCY", "7")
    monkeypatch.setenv("LATENCE_TRACE_PROFILE", "quality")
    assert runpod_handler.create_config().max_concurrency == 32


def test_build_startup_warmup_requests_exercises_three_triangular_probes() -> None:
    requests = runpod_handler._build_startup_warmup_requests()
    assert len(requests) == 3
    assert all(request.include_triangular_diagnostics for request in requests)
    assert all(request.query_text for request in requests)
    assert all(request.raw_context for request in requests)
    assert all(request.response_text for request in requests)


def test_prime_service_runtime_issues_all_startup_requests() -> None:
    service = _SlowService(sleep_ms=0)
    runpod_handler._prime_service_runtime(service)
    assert service.calls == 3


def test_runpod_handler_bounds_inflight_requests(monkeypatch) -> None:
    service = _SlowService(sleep_ms=80)
    config = _config(max_concurrency=2)

    monkeypatch.setattr(runpod_handler, "initialize", lambda: None)
    runpod_handler._initialized = True
    runpod_handler._config = config
    runpod_handler._service = service
    runpod_handler._servers = {}
    runpod_handler._request_executor = None
    runpod_handler._request_semaphore = None
    runpod_handler._request_semaphore_loop = None

    results = asyncio.run(_burst(6))

    try:
        assert all(item["success"] for item in results)
        assert service.peak_inflight <= config.max_concurrency
    finally:
        runpod_handler.shutdown()
