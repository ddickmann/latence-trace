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
        max_concurrency=_env_int("MAX_CONCURRENCY", 32),
        collection_label=os.environ.get("LATENCE_TRACE_COLLECTION_LABEL", "latence-trace"),
        service_device=os.environ.get("LATENCE_TRACE_SERVICE_DEVICE", _detect_device()),
        docs_url=os.environ.get("LATENCE_TRACE_DOCS_URL", ""),
        colbert_model=colbert_model,
        colbert_port=_env_int("LATENCE_TRACE_COLBERT_PORT", 8001),
        colbert_gpu_mem=_env_float("LATENCE_TRACE_COLBERT_GPU_MEM", 0.34),
        colbert_max_model_len=_env_int("LATENCE_TRACE_COLBERT_MAX_MODEL_LEN", 8192),
        colbert_max_num_seqs=_env_int("LATENCE_TRACE_COLBERT_MAX_NUM_SEQS", 192),
        colbert_max_batched_tokens=_env_int("LATENCE_TRACE_COLBERT_MAX_BATCHED_TOKENS", 32768),
        nli_model=nli_model,
        nli_port=_env_int("LATENCE_TRACE_NLI_PORT", 8002),
        nli_gpu_mem=_env_float("LATENCE_TRACE_NLI_GPU_MEM", 0.24),
        nli_max_model_len=_env_int("LATENCE_TRACE_NLI_MAX_MODEL_LEN", 512),
        nli_max_num_seqs=_env_int("LATENCE_TRACE_NLI_MAX_NUM_SEQS", 256),
        nli_max_batched_tokens=_env_int("LATENCE_TRACE_NLI_MAX_BATCHED_TOKENS", 8192),
    )


_initialized = False
_config: WorkerConfig | None = None
_servers: dict[str, ManagedVllmServer] = {}
_service: GroundednessService | None = None
_initialize_lock = threading.Lock()


def _start_kernel_warmup(profile: str) -> tuple[threading.Thread, dict[str, Any]]:
    result: dict[str, Any] = {"error": None}

    def _runner() -> None:
        try:
            warm_all(profile)
        except Exception as exc:  # pragma: no cover - best-effort warmup
            logger.warning("kernel warmup failed: %s", exc)
            result["error"] = exc

    thread = threading.Thread(target=_runner, name="latence-kernel-warmup", daemon=True)
    thread.start()
    return thread, result


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
            plugins=["moderncolbert"],
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

        warmup_thread, _warmup_result = _start_kernel_warmup(config.profile)
        servers = _build_servers(config)

        try:
            with ThreadPoolExecutor(max_workers=len(servers)) as pool:
                futures = [pool.submit(server.start) for server in servers.values()]
                for future in futures:
                    future.result()

            os.environ["VOYAGER_GROUNDEDNESS_VLLM_ENDPOINT"] = servers["colbert"].base_url
            os.environ["VOYAGER_GROUNDEDNESS_VLLM_MODEL"] = config.colbert_model
            os.environ["LATENCE_TRACE_NLI_VLLM_ENDPOINT"] = servers["nli"].base_url
            os.environ["LATENCE_TRACE_NLI_VLLM_MODEL"] = config.nli_model
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

            warmup_thread.join()

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
    global _initialized, _servers, _service
    for server in _servers.values():
        try:
            server.stop()
        except Exception:
            logger.exception("failed to stop server %s", server.name)
    _servers = {}
    _service = None
    _initialized = False


def _health_payload() -> dict[str, Any]:
    return {
        "success": True,
        "version": _config.version if _config else __version__,
        "profile": _config.profile if _config else None,
        "service_device": _config.service_device if _config else None,
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
        response = await asyncio.wait_for(
            asyncio.to_thread(_service.groundedness, request),
            timeout=_config.request_timeout_s if _config else 120,
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
            "concurrency_modifier": lambda _current: _config.max_concurrency if _config else 32,
        }
    )
