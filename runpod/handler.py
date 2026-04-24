"""RunPod serverless entrypoint for latence-trace.

The worker exposes two scoring lanes on one endpoint:

- ``scoring_mode="rag"`` (default) — the long-standing enterprise RAG
  groundedness pipeline. Unchanged semantics, unchanged response shape.
- ``scoring_mode="code"`` — a code-aware lane built on top of the same
  ColBERT MaxSim backbone with AST drift, literal novelty, an
  ambiguity-gated NLI cascade, semantic entropy, a calibrated composite
  phantom score, and per-file / per-unit attribution. Designed for IDE
  plugins (Cursor, Claude Code, OpenAI Codex, OpenCode, aider) that
  want a real-time dashboard tracing the quality of their agent.

Concurrency model
-----------------
Each lane gets its own inflight semaphore so a burst of code-lane calls
never starves in-flight RAG calls (and vice versa). The total worker
concurrency is still bounded by ``max_concurrency``; the per-lane
budget defaults to ``ceil(max_concurrency / 2)`` each so one lane can
burst through the shared ceiling when the other is idle.

Warmup
------
Two warmup requests run at boot: one RAG request through the full
pipeline (primes ColBERT + NLI vLLM + null bank) and one code-lane
request (primes ``GPUScorer`` streams + pre-loads tree-sitter grammars
+ exercises the composite model). Boot is a no-op once the singletons
are primed.

Observability
-------------
One structured log line per request (``groundedness_turn``) captures
``{lane, session_id, cascade_fired, phantom_verdict, nli_ms, ast_ms,
composite_ms, total_ms}``. No PII is logged — only hashed session
labels.
"""

from __future__ import annotations

import asyncio
import atexit
import logging
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError as PydanticValidationError

from latence_trace import __version__
from latence_trace.api.models import (
    GroundednessRequest,
    GroundednessResponse,
    ScoringMode,
)
from latence_trace.api.service import (
    GroundednessService,
    ServiceError,
    ValidationError,
    apply_profile,
)
from latence_trace.kernels.warmup import warm_all, warm_code_lane
from latence_trace.observability.metrics import (
    BUDGET_EXCEEDED_COUNT,
    CASCADE_FIRE_COUNT,
    LANE_REQUEST_COUNT,
    PHANTOM_VERDICT_COUNT,
)
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


_STARTUP_WARMUP_REQUEST_COUNT = 4


@dataclass(frozen=True)
class WorkerConfig:
    profile: str
    version: str
    request_timeout_s: int
    code_request_timeout_s: float
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
        # Per-lane timeout: code lane ships a tight 150ms p95 SLO; a 2s
        # ceiling protects the tail while still catching hung requests.
        code_request_timeout_s=_env_float(
            "LATENCE_TRACE_CODE_REQUEST_TIMEOUT_S", 2.0
        ),
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
_lane_semaphores: dict[ScoringMode, asyncio.Semaphore] = {}
_lane_semaphores_loop: asyncio.AbstractEventLoop | None = None


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


def _lane_budget(config: WorkerConfig, lane: ScoringMode) -> int:
    """Per-lane inflight budget.

    Defaults to ``ceil(max_concurrency / 2)`` each so one lane can
    burst through the shared ceiling when the other is idle. Callers
    can pin the budget explicitly via
    ``LATENCE_TRACE_RAG_CONCURRENCY`` /
    ``LATENCE_TRACE_CODE_CONCURRENCY``.
    """
    env_key = (
        "LATENCE_TRACE_RAG_CONCURRENCY"
        if lane == ScoringMode.RAG
        else "LATENCE_TRACE_CODE_CONCURRENCY"
    )
    override = _env_int(env_key, 0)
    if override > 0:
        return override
    return max(1, int(math.ceil(config.max_concurrency / 2)))


def _get_lane_semaphore(config: WorkerConfig, lane: ScoringMode) -> asyncio.Semaphore:
    """Return (and lazily build) the per-lane semaphore for the active loop.

    The semaphore is pinned to the asyncio loop that owns it; if the
    loop flips (RunPod cold start, test reuse), we rebuild both
    semaphores against the new loop.
    """
    global _lane_semaphores, _lane_semaphores_loop
    loop = asyncio.get_running_loop()
    if _lane_semaphores_loop is not loop:
        _lane_semaphores = {
            ScoringMode.RAG: asyncio.Semaphore(_lane_budget(config, ScoringMode.RAG)),
            ScoringMode.CODE: asyncio.Semaphore(_lane_budget(config, ScoringMode.CODE)),
        }
        _lane_semaphores_loop = loop
    return _lane_semaphores[lane]


def _make_startup_sentence(prefix: str, index: int) -> str:
    return (
        f"{prefix} record {index} states the reference sample remained internally "
        f"consistent on day {index} with calibration value {100 + index}."
    )


_CODE_LANE_WARMUP_RESPONSE = (
    "```python\n"
    "from httpx import AsyncClient\n\n"
    "async def fetch_user(id: int):\n"
    "    async with AsyncClient() as client:\n"
    "        return await client.get(f'/users/{id}')\n"
    "```"
)

_CODE_LANE_WARMUP_CONTEXT = (
    "# fetch_user.py\n"
    "async def fetch_user(id: int):\n"
    "    async with httpx.AsyncClient() as client:\n"
    "        response = await client.get(f'/users/{id}')\n"
    "        return response.json()\n"
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
    # One code-lane warmup turn primes GPUScorer streams, pre-loads the
    # tree-sitter grammars, and touches the composite model.
    requests.append(
        GroundednessRequest(
            scoring_mode=ScoringMode.CODE,
            session_id="warmup-code-lane",
            response_language_hint="python",
            query_text="Add retry logic to fetch_user",
            raw_context=_CODE_LANE_WARMUP_CONTEXT,
            response_text=_CODE_LANE_WARMUP_RESPONSE,
            emit_chunk_ownership=False,
            evidence_limit=4,
        )
    )
    return requests


def _ensure_kernel_warmup(profile: str) -> None:
    result = warm_all(profile)
    if not result.ok:
        raise RuntimeError(
            "Triton kernel warmup failed for profile '{profile}' on {device}: {error}".format(
                profile=result.profile,
                device=result.device,
                error=result.error or "unknown error",
            )
        )
    # Code lane warmup is now a hard gate: tree-sitter grammars are a
    # first-class production dependency and a missing grammar would
    # silently degrade the AST phantom / drift signals to a regex
    # fallback. We refuse to serve traffic in that state.
    code_result = warm_code_lane()
    if not code_result.ok:
        raise RuntimeError(
            "Code-lane warmup failed on {device}: {error} (details={details})".format(
                device=code_result.device,
                error=code_result.error,
                details=code_result.details,
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
    global _initialized, _servers, _service, _request_executor
    global _lane_semaphores, _lane_semaphores_loop
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
    _lane_semaphores = {}
    _lane_semaphores_loop = None


def _code_lane_health() -> dict[str, Any]:
    """Snapshot code-lane readiness for the /healthz probe.

    Reports the current tree-sitter parser backend, which grammars
    loaded, and whether the code-lane warmup pass succeeded. Allows
    operators and alerting to instantly flag a deploy that slipped
    into the regex fallback path.
    """

    payload: dict[str, Any] = {
        "parser_backend": None,
        "loaded_grammars": [],
        "required_grammars": [],
        "warmup_ok": None,
        "warmup_error": None,
    }
    try:
        from latence_trace.core.code_lane.ast_grounding import (
            AstSymbolExtractor,
            SUPPORTED_LANGUAGES,
        )
        from latence_trace.kernels.warmup import warmup_state

        payload["required_grammars"] = sorted(SUPPORTED_LANGUAGES)
        payload["parser_backend"] = AstSymbolExtractor(
            enabled=True
        ).parser_backend
        payload["loaded_grammars"] = AstSymbolExtractor.available_languages()
        code_result = warmup_state().get("code_lane")
        if code_result is not None:
            payload["warmup_ok"] = bool(code_result.ok)
            payload["warmup_error"] = code_result.error
    except Exception as exc:  # pragma: no cover - defensive
        payload["warmup_error"] = str(exc)
    return payload


def _health_payload() -> dict[str, Any]:
    return {
        "success": True,
        "version": _config.version if _config else __version__,
        "profile": _config.profile if _config else None,
        "service_device": _config.service_device if _config else None,
        "max_concurrency": _config.max_concurrency if _config else None,
        "startup_warmup_requests": _STARTUP_WARMUP_REQUEST_COUNT,
        "servers": {name: server.health() for name, server in _servers.items()},
        "code_lane": _code_lane_health(),
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
        # Code-lane fields (ignored by the RAG path).
        "scoring_mode",
        "session_id",
        "response_language_hint",
        "emit_chunk_ownership",
        "session_state",
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
        "scoring_mode": response.scoring_mode.value if response.scoring_mode else None,
        "session_id": response.session_id,
    }
    if response.scoring_mode == ScoringMode.CODE:
        result["code_lane"] = {
            "composite_score": scores.composite_phantom_score,
            "phantom_probability": scores.composite_phantom_probability,
            "phantom_verdict": scores.composite_phantom_verdict,
            "literal_novelty_min": scores.literal_novelty_min,
            "literal_novelty_missing_count": scores.literal_novelty_missing_count,
            "ast_phantom_symbol_count": scores.ast_phantom_symbol_count,
            "ast_literal_drift_count": scores.ast_literal_drift_count,
            "ast_phantom_verdict": scores.ast_phantom_verdict,
            "nli_contradiction_prob_max": scores.nli_contradiction_prob_max,
            "nli_cascade_triggered": scores.nli_cascade_triggered,
            "dead_weight_ratio": scores.dead_weight_ratio,
            "dead_weight_file_count": scores.dead_weight_file_count,
        }
        if response.code_lane_diagnostics is not None:
            diag = response.code_lane_diagnostics
            result["code_lane"]["diagnostics"] = {
                "config": diag.config,
                "total_latency_ms": diag.total_latency_ms,
                "component_latency_ms": diag.component_latency_ms,
                "file_attribution": (
                    diag.file_attribution.model_dump() if diag.file_attribution else None
                ),
                "ast": diag.ast.model_dump() if diag.ast else None,
                "literal_novelty": (
                    diag.literal_novelty.model_dump() if diag.literal_novelty else None
                ),
                "nli_cascade": (
                    diag.nli_cascade.model_dump() if diag.nli_cascade else None
                ),
                "composite": diag.composite.model_dump() if diag.composite else None,
            }
        # Opt-in caller-portable session blob + derived signals. The
        # client is expected to round-trip ``next_session_state`` verbatim
        # on the next turn; ``session_signals`` is advisory.
        if response.next_session_state is not None:
            result["next_session_state"] = response.next_session_state.model_dump(
                mode="json"
            )
        if response.session_signals is not None:
            result["session_signals"] = response.session_signals.model_dump(
                mode="json"
            )
    if response.reason:
        result["reason"] = response.reason
    if response.warnings:
        result["warnings"] = list(response.warnings)
    if verbose:
        result["full"] = response.model_dump(mode="json")
    return result


def _log_turn_event(
    *,
    request: GroundednessRequest,
    response: GroundednessResponse,
    duration_ms: float,
) -> None:
    """Emit one structured per-turn log line (no PII, PII-safe labels only).

    The payload is the minimal set of fields an IDE-plugin dashboard
    needs to plot lane latency, cascade fire rate, and phantom verdicts
    over time. ``session_id`` is echoed verbatim so plugins can bucket
    turn events client-side; callers must supply a hashed token rather
    than a raw user identifier (documented on the request model).
    """
    diag = response.code_lane_diagnostics
    scores = response.scores
    component = dict((diag.component_latency_ms if diag else {}) or {})
    lane = response.scoring_mode.value if response.scoring_mode else "rag"
    cascade_fired = bool(scores.nli_cascade_triggered)
    phantom_verdict = scores.composite_phantom_verdict
    ast_parser_backend: str | None = None
    ast_language: str | None = None
    if diag and diag.ast is not None:
        ast_parser_backend = getattr(diag.ast, "parser_backend", None)
        ast_language = getattr(diag.ast, "language", None)
    # Caller-carried session signals (opt-in, derived from the pure
    # transform in ``latence_trace.core.code_lane.session``). We log
    # the *signals*, not the opaque state blob, so the log volume stays
    # bounded and no per-file paths leak into the metric store.
    signals = response.session_signals
    logger.info(
        "groundedness_turn",
        extra={
            "lane": lane,
            "session_id": response.session_id,
            "cascade_fired": cascade_fired,
            "phantom_verdict": phantom_verdict,
            "phantom_probability": scores.composite_phantom_probability,
            "dead_weight_ratio": scores.dead_weight_ratio,
            "ast_parser_backend": ast_parser_backend,
            "ast_language": ast_language,
            "nli_ms": component.get("nli_ms"),
            "ast_ms": component.get("ast_ms"),
            "scorer_ms": component.get("scorer_ms"),
            "composite_ms": component.get("composite_ms"),
            "file_attribution_ms": component.get("file_attribution_ms"),
            "literal_novelty_ms": component.get("literal_novelty_ms"),
            "total_ms": float(duration_ms),
            "risk_band": scores.risk_band,
            "session_total_turns": (signals.total_turns if signals else None),
            "session_drift_z": (signals.drift_z_score if signals else None),
            "session_ema_groundedness": (
                signals.ema_groundedness if signals else None
            ),
            "session_cascade_density": (
                signals.cascade_density if signals else None
            ),
            "session_phantom_rate": (signals.phantom_rate if signals else None),
            "session_red_streak": (signals.red_streak if signals else None),
            "session_recommendation": (
                signals.recommendation if signals else None
            ),
        },
    )
    # Loud WARNING when a code-lane request landed on the regex
    # fallback for a language tree-sitter was supposed to cover. This
    # must never fire in production — if it does, grammars are not
    # installed and the AST drift signal has degraded in quality.
    if (
        lane == "code"
        and ast_parser_backend is not None
        and ast_parser_backend != "tree_sitter"
    ):
        logger.warning(
            "ast_regex_fallback_on_supported_lang",
            extra={
                "lane": lane,
                "ast_parser_backend": ast_parser_backend,
                "ast_language": ast_language,
                "session_id": response.session_id,
            },
        )
    # Prometheus counters — keep cardinality bounded; labels are
    # enum-like so no PII leaks into the metric store.
    try:
        LANE_REQUEST_COUNT.labels(lane=lane).inc()
        if cascade_fired:
            CASCADE_FIRE_COUNT.labels(lane=lane).inc()
        if phantom_verdict is not None:
            PHANTOM_VERDICT_COUNT.labels(
                verdict="true" if phantom_verdict else "false"
            ).inc()
    except Exception:  # pragma: no cover - metrics must never fail a turn
        logger.exception("groundedness_turn_metrics_failed")


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

    # Pre-compute the lane so exception handlers below can refer to it
    # even if initialisation failed mid-flight.
    lane: ScoringMode = request.scoring_mode or ScoringMode.RAG

    try:
        config = _config
        service = _service
        if config is None or service is None:
            raise RuntimeError("RunPod worker was not initialized")
        executor = _get_request_executor(config)
        semaphore = _get_lane_semaphore(config, lane)
        started = time.perf_counter()
        if semaphore.locked():
            # The lane is at capacity — surface as a metric so the
            # operator dashboard can alert on sustained backpressure.
            try:
                BUDGET_EXCEEDED_COUNT.labels(lane=lane.value).inc()
            except Exception:  # pragma: no cover
                logger.exception("budget_exceeded_metric_failed")
        # Code-lane calls get a much tighter ceiling (150ms p95 SLO);
        # RAG stays on the long-running timeout so document-heavy
        # requests still fit.
        effective_timeout = (
            config.code_request_timeout_s
            if lane == ScoringMode.CODE
            else config.request_timeout_s
        )
        async with semaphore:
            loop = asyncio.get_running_loop()
            response = await asyncio.wait_for(
                loop.run_in_executor(executor, service.groundedness, request),
                timeout=effective_timeout,
            )
            duration_ms = (time.perf_counter() - started) * 1000.0
            try:
                _log_turn_event(
                    request=request, response=response, duration_ms=duration_ms
                )
            except Exception:  # pragma: no cover - logging must never fail a turn
                logger.exception("groundedness_turn_log_failed")
            return _compact_response(response, verbose=verbose)
    except asyncio.TimeoutError:
        if lane == ScoringMode.CODE:
            timeout_value = _config.code_request_timeout_s if _config else 2.0
            hint = (
                "Retry with a smaller request or increase "
                "LATENCE_TRACE_CODE_REQUEST_TIMEOUT_S."
            )
        else:
            timeout_value = _config.request_timeout_s if _config else 120
            hint = (
                "Retry with a smaller request or increase "
                "LATENCE_TRACE_RUNPOD_REQUEST_TIMEOUT."
            )
        return _service_error_payload(
            f"Job exceeded {timeout_value}s execution timeout",
            error_code="job_timeout",
            hint=hint,
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
