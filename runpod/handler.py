"""RunPod serverless entrypoint for latence-trace."""

from __future__ import annotations

import asyncio
import atexit
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError as PydanticValidationError

from latence_trace import __version__
from latence_trace.api.models import GroundednessRequest, GroundednessResponse
from latence_trace.api.service import (
    GroundednessService,
    ServiceError,
    ValidationError,
    apply_profile,
)
from latence_trace.kernels.warmup import warm_all
from server import ManagedVllmServer

try:  # pragma: no cover - optional in local dev
    import runpod
except Exception:  # pragma: no cover - local smoke tests import this module
    runpod = None

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=os.environ.get("LATENCE_TRACE_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _detect_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


_STARTUP_WARMUP_REQUEST_COUNT = 3


@dataclass(frozen=True)
class WorkerConfig:
    profile: str
    version: str
    request_timeout_s: int
    max_concurrency: int
    collection_label: str
    service_device: str
    docs_url: str
    colbert_model: str
    colbert_port: int
    colbert_gpu_mem: float
    colbert_max_model_len: int
    colbert_max_num_seqs: int
    colbert_max_batched_tokens: int
    nli_model: str
    nli_port: int
    nli_gpu_mem: float
    nli_max_model_len: int
    nli_max_num_seqs: int
    nli_max_batched_tokens: int


def create_config() -> WorkerConfig:
    profile = (
        os.environ.get("LATENCE_TRACE_PROFILE")
        or os.environ.get("LATENCE_TRACE_ACTIVE_PROFILE")
        or "quality"
    ).strip().lower()
    colbert_model = os.environ.get("LATENCE_TRACE_COLBERT_MODEL", "lightonai/LateOn")
    nli_model = os.environ.get(
        "LATENCE_TRACE_NLI_MODEL",
        os.environ.get(
            "VOYAGER_GROUNDEDNESS_NLI_MODEL",
            "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
        ),
    )
    return WorkerConfig(
        profile=profile,
        version=os.environ.get("LATENCE_TRACE_RUNPOD_VERSION", __version__),
        request_timeout_s=_env_int("LATENCE_TRACE_RUNPOD_REQUEST_TIMEOUT", 120),
        max_concurrency=64,
        collection_label=os.environ.get("LATENCE_TRACE_COLLECTION_LABEL", "latence-trace"),
        service_device=os.environ.get("LATENCE_TRACE_SERVICE_DEVICE", _detect_device()),
        docs_url=os.environ.get("LATENCE_TRACE_DOCS_URL", ""),
        colbert_model=colbert_model,
        colbert_port=_env_int("LATENCE_TRACE_COLBERT_PORT", 8001),
        colbert_gpu_mem=_env_float("LATENCE_TRACE_COLBERT_GPU_MEM", 0.34),
        colbert_max_model_len=_env_int("LATENCE_TRACE_COLBERT_MAX_MODEL_LEN", 8192),
        colbert_max_num_seqs=_env_int("LATENCE_TRACE_COLBERT_MAX_NUM_SEQS", 128),
        colbert_max_batched_tokens=_env_int("LATENCE_TRACE_COLBERT_MAX_BATCHED_TOKENS", 8192),
        nli_model=nli_model,
        nli_port=_env_int("LATENCE_TRACE_NLI_PORT", 8002),
        nli_gpu_mem=_env_float("LATENCE_TRACE_NLI_GPU_MEM", 0.24),
        nli_max_model_len=_env_int("LATENCE_TRACE_NLI_MAX_MODEL_LEN", 512),
        nli_max_num_seqs=_env_int("LATENCE_TRACE_NLI_MAX_NUM_SEQS", 128),
        nli_max_batched_tokens=_env_int("LATENCE_TRACE_NLI_MAX_BATCHED_TOKENS", 8192),
    )


_initialized = False
_config: WorkerConfig | None = None
_servers: dict[str, ManagedVllmServer] = {}
_service: GroundednessService | None = None
_initialize_lock = threading.Lock()
_request_executor: ThreadPoolExecutor | None = None
_request_executor_lock = threading.Lock()
_request_semaphore: asyncio.Semaphore | None = None
_request_semaphore_loop: asyncio.AbstractEventLoop | None = None


def _get_request_executor(config: WorkerConfig) -> ThreadPoolExecutor:
    global _request_executor
    if _request_executor is not None:
        return _request_executor
    with _request_executor_lock:
        if _request_executor is None:
            _request_executor = ThreadPoolExecutor(
                max_workers=config.max_concurrency,
                thread_name_prefix="latence-trace-runpod",
            )
        return _request_executor


def _get_request_semaphore(limit: int) -> asyncio.Semaphore:
    global _request_semaphore, _request_semaphore_loop
    loop = asyncio.get_running_loop()
    if _request_semaphore is not None and _request_semaphore_loop is loop:
        return _request_semaphore
    _request_semaphore = asyncio.Semaphore(limit)
    _request_semaphore_loop = loop
    return _request_semaphore


def _make_startup_sentence(prefix: str, index: int) -> str:
    return (
        f"{prefix} record {index} states the reference sample remained internally "
        f"consistent on day {index} with calibration value {100 + index}."
    )


def _build_startup_warmup_requests() -> list[GroundednessRequest]:
    query_text = "Which recorded days remained internally consistent in the reference sample?"
    shapes = (
        ("alpha", 8, 4),
        ("beta", 24, 8),
        ("gamma", 48, 16),
    )
    requests: list[GroundednessRequest] = []
    for prefix, context_count, response_count in shapes:
        context_sentences = [_make_startup_sentence(prefix, idx) for idx in range(1, context_count + 1)]
        response_sentences = [_make_startup_sentence(prefix, idx) for idx in range(1, response_count + 1)]
        requests.append(
            GroundednessRequest(
                query_text=query_text,
                raw_context=" ".join(context_sentences),
                response_text=" ".join(response_sentences),
                include_triangular_diagnostics=True,
                evidence_limit=4,
            )
        )
    return requests


def _ensure_kernel_warmup(profile: str) -> None:
    result = warm_all(profile)
    if result.ok:
        return
    raise RuntimeError(
        "Triton kernel warmup failed for profile '{profile}' on {device}: {error}".format(
            profile=result.profile,
            device=result.device,
            error=result.error or "unknown error",
        )
    )


def _prime_service_runtime(service: GroundednessService) -> None:
    requests = _build_startup_warmup_requests()
    for idx, request in enumerate(requests, start=1):
        response = service.groundedness(request)
        logger.info(
            "startup warmup request %s/%s complete: primary=%s score=%.4f latency_ms=%.2f",
            idx,
            len(requests),
            response.scores.primary_name,
            response.scores.primary_score,
            response.time_ms,
        )


def _build_servers(config: WorkerConfig) -> dict[str, ManagedVllmServer]:
    return {
        "colbert": ManagedVllmServer(
            name="colbert",
            model=config.colbert_model,
            port=config.colbert_port,
            io_processor_plugin="moderncolbert_batched_io",
            gpu_memory_utilization=config.colbert_gpu_mem,
            max_model_len=config.colbert_max_model_len,
            max_num_seqs=config.colbert_max_num_seqs,
            max_num_batched_tokens=config.colbert_max_batched_tokens,
            plugins=["moderncolbert", "moderncolbert_batched_io"],
            enforce_eager=False,
        ),
        "nli": ManagedVllmServer(
            name="nli",
            model=config.nli_model,
            port=config.nli_port,
            io_processor_plugin="nli_mdeberta",
            gpu_memory_utilization=config.nli_gpu_mem,
            max_model_len=config.nli_max_model_len,
            max_num_seqs=config.nli_max_num_seqs,
            max_num_batched_tokens=config.nli_max_batched_tokens,
            plugins=["nli_mdeberta"],
            enforce_eager=False,
        ),
    }


def initialize() -> None:
    global _initialized, _config, _servers, _service
    if _initialized:
        return

    with _initialize_lock:
        if _initialized:
            return

        config = create_config()
        _config = config

        os.environ.setdefault("LATENCE_TRACE_PROFILE", config.profile)
        apply_profile(config.profile)

        servers = _build_servers(config)

        try:
            with ThreadPoolExecutor(max_workers=len(servers)) as pool:
                futures = [pool.submit(server.start) for server in servers.values()]
                for future in futures:
                    future.result()

            os.environ["VOYAGER_GROUNDEDNESS_VLLM_ENDPOINT"] = servers["colbert"].base_url
            os.environ["VOYAGER_GROUNDEDNESS_VLLM_MODEL"] = config.colbert_model
            os.environ["VOYAGER_GROUNDEDNESS_VLLM_MAX_CONCURRENCY"] = str(config.max_concurrency)
            os.environ["LATENCE_TRACE_NLI_VLLM_ENDPOINT"] = servers["nli"].base_url
            os.environ["LATENCE_TRACE_NLI_VLLM_MODEL"] = config.nli_model
            os.environ["LATENCE_TRACE_NLI_VLLM_MAX_CONCURRENCY"] = str(config.max_concurrency)
            os.environ["VOYAGER_GROUNDEDNESS_NLI_MODEL"] = config.nli_model
            os.environ.setdefault("VOYAGER_GROUNDEDNESS_NLI_ENABLED", "1")
            os.environ.setdefault(
                "VOYAGER_GROUNDEDNESS_NLI_PREMISE_RERANKER_MODEL",
                "BAAI/bge-reranker-v2-m3",
            )

            service = GroundednessService(
                device=config.service_device,
                collection_label=config.collection_label,
            )
            _ensure_kernel_warmup(config.profile)
            _prime_service_runtime(service)
            _get_request_executor(config)

            _servers = servers
            _service = service
            _initialized = True
            logger.info(
                "latence-trace RunPod worker ready: profile=%s colbert=%s nli=%s",
                config.profile,
                servers["colbert"].base_url,
                servers["nli"].base_url,
            )
        except Exception:
            for server in servers.values():
                server.stop()
            raise


def shutdown() -> None:
    global _initialized, _servers, _service, _request_executor, _request_semaphore, _request_semaphore_loop
    for server in _servers.values():
        try:
            server.stop()
        except Exception:
            logger.exception("failed to stop server %s", server.name)
    _servers = {}
    _service = None
    _initialized = False
    with _request_executor_lock:
        if _request_executor is not None:
            _request_executor.shutdown(wait=False, cancel_futures=True)
            _request_executor = None
    _request_semaphore = None
    _request_semaphore_loop = None


def _health_payload() -> dict[str, Any]:
    return {
        "success": True,
        "version": _config.version if _config else __version__,
        "profile": _config.profile if _config else None,
        "service_device": _config.service_device if _config else None,
        "max_concurrency": _config.max_concurrency if _config else None,
        "startup_warmup_requests": _STARTUP_WARMUP_REQUEST_COUNT,
        "servers": {name: server.health() for name, server in _servers.items()},
    }


def _build_request(input_data: dict[str, Any]) -> tuple[GroundednessRequest, bool]:
    raw_context = input_data.get("raw_context")
    if raw_context is None:
        raw_context = input_data.get("context")
    query_text = input_data.get("query_text")
    if query_text is None:
        query_text = input_data.get("query")
    response_text = input_data.get("response_text")
    if response_text is None:
        response_text = input_data.get("response")

    payload: dict[str, Any] = {
        "query_text": query_text,
        "raw_context": raw_context,
        "response_text": response_text,
    }

    passthrough_keys = (
        "chunk_ids",
        "support_units",
        "attribution_mode",
        "segmentation_mode",
        "raw_context_chunk_tokens",
        "response_chunk_tokens",
        "coverage_threshold",
        "primary_metric",
        "evidence_limit",
        "debug_dense_matrices",
        "include_triangular_diagnostics",
        "model",
        "query_prompt_name",
        "document_prompt_name",
        "verification_samples",
        "content_type",
        "risk_band_stratum",
    )
    for key in passthrough_keys:
        if key in input_data:
            payload[key] = input_data[key]

    verbose = bool(input_data.get("verbose", False))
    return GroundednessRequest.model_validate(payload), verbose


def _compact_response(response: GroundednessResponse, *, verbose: bool) -> dict[str, Any]:
    scores = response.scores
    primary_metric = "groundedness_v2" if scores.groundedness_v2 is not None else scores.primary_name
    score = scores.groundedness_v2 if scores.groundedness_v2 is not None else scores.primary_score
    result: dict[str, Any] = {
        "success": True,
        "score": float(score),
        "primary_metric": primary_metric,
        "band": scores.risk_band,
        "structured_score": scores.structured_source,
        "nli_aggregate": scores.nli_aggregate,
        "context_coverage_ratio": scores.context_coverage_ratio,
        "context_unused_ratio": scores.context_unused_ratio,
        "context_uncertain_ratio": scores.context_uncertain_ratio,
        "context_usage_ratio": scores.context_usage_ratio,
        "support_units_usage": {
            "used": scores.support_units_usage_used,
            "unused": scores.support_units_unused,
            "uncertain": scores.support_units_uncertain,
        },
        "support_units": [
            {
                "support_id": unit.support_id,
                "usage_state": unit.usage_state.value,
                "usage_confidence": unit.usage_confidence,
                "unused_confidence": unit.unused_confidence,
                "coverage_score": unit.coverage_score,
                "used": unit.used,
            }
            for unit in response.support_units
        ],
        "latency_ms": float(response.time_ms),
        "version": _config.version if _config else __version__,
    }
    if response.reason:
        result["reason"] = response.reason
    if response.warnings:
        result["warnings"] = list(response.warnings)
    if verbose:
        result["full"] = response.model_dump(mode="json")
    return result


def _service_error_payload(
    message: str,
    *,
    error_code: str,
    hint: str | None = None,
    status_code: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "success": False,
        "error": message,
        "error_code": error_code,
    }
    if hint:
        payload["hint"] = hint
    if status_code is not None:
        payload["status_code"] = status_code
    if _config and _config.docs_url:
        payload["docs_url"] = _config.docs_url
    return payload


async def handler(job: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(job, dict):
        return _service_error_payload(
            f"Invalid event type: {type(job).__name__}",
            error_code="invalid_event",
            hint="Send a JSON object with an 'input' field.",
            status_code=400,
        )

    initialize()

    input_data = job.get("input")
    if not isinstance(input_data, dict):
        return _service_error_payload(
            "Missing or invalid 'input' field",
            error_code="invalid_input",
            hint="Wrap the request payload under input={query, context, response}.",
            status_code=400,
        )

    if input_data.get("endpoint_id") == "_health" or bool(input_data.get("health")):
        return _health_payload()

    try:
        request, verbose = _build_request(input_data)
    except PydanticValidationError as exc:
        return _service_error_payload(
            str(exc),
            error_code="validation_error",
            hint=ValidationError.hint,
            status_code=400,
        )
    except Exception as exc:
        return _service_error_payload(
            str(exc),
            error_code="validation_error",
            hint=ValidationError.hint,
            status_code=400,
        )

    try:
        config = _config
        service = _service
        if config is None or service is None:
            raise RuntimeError("RunPod worker was not initialized")
        executor = _get_request_executor(config)
        async with _get_request_semaphore(config.max_concurrency):
            loop = asyncio.get_running_loop()
            response = await asyncio.wait_for(
                loop.run_in_executor(executor, service.groundedness, request),
                timeout=config.request_timeout_s,
            )
            return _compact_response(response, verbose=verbose)
    except asyncio.TimeoutError:
        return _service_error_payload(
            f"Job exceeded {_config.request_timeout_s if _config else 120}s execution timeout",
            error_code="job_timeout",
            hint="Retry with a smaller request or increase LATENCE_TRACE_RUNPOD_REQUEST_TIMEOUT.",
            status_code=504,
        )
    except ServiceError as exc:
        return _service_error_payload(
            str(exc),
            error_code=exc.error_code,
            hint=getattr(exc, "hint", None),
            status_code=getattr(exc, "status_code", None),
        )
    except Exception as exc:  # pragma: no cover - runtime safeguard
        logger.exception("RunPod request failed")
        return _service_error_payload(
            str(exc),
            error_code="service_error",
            hint="Inspect worker logs; both vLLM lanes and the reranker are process-wide singletons.",
            status_code=500,
        )


__all__ = ["create_config", "handler", "initialize", "shutdown"]


if __name__ == "__main__":  # pragma: no cover - exercised in container
    if runpod is None:
        raise RuntimeError("runpod package is required to start the serverless worker")
    initialize()
    atexit.register(shutdown)
    runpod.serverless.start(
        {
            "handler": handler,
            "concurrency_modifier": lambda _current: _config.max_concurrency if _config else 64,
        }
    )
