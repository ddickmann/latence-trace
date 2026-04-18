"""FastAPI router for the standalone latence-trace Groundedness Tracker.

The router exposes a single ``POST /groundedness`` endpoint plus the
operational health/readiness probes used by Kubernetes and the Helm
chart (PA2 + L5):

- ``GET /healthz`` returns 200 immediately as long as the process is
  alive (liveness).
- ``GET /readyz`` returns 200 only after the Triton-kernel warmup has
  completed - so traffic only reaches a worker that already paid the
  JIT cost and the first user request matches steady-state p95.
"""

from __future__ import annotations

import asyncio
import os
from typing import Callable, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from latence_trace.api.models import GroundednessRequest, GroundednessResponse
from latence_trace.api.service import (
    DEFAULT_PROFILE,
    PROFILE_NAMES,
    GroundednessService,
    NotFoundError,
    ServiceError,
    ValidationError,
)
from latence_trace.kernels.warmup import is_warm, warmup_state


def _resolve_inflight_limit() -> int:
    """Resolve the bounded inflight concurrency for groundedness scoring.

    PA4: groundedness scoring is GPU- and CPU-heavy (encoder, NLI,
    reranker, Triton kernel). Letting an unbounded number of requests
    run concurrently inside the FastAPI threadpool causes GPU OOM and
    starvation. We cap the number of in-flight scoring calls with a
    semaphore. The default keeps GPU memory pressure inside the 24 GB
    VRAM budget the README documents; operators can override via
    ``LATENCE_TRACE_MAX_INFLIGHT`` after their own load test.
    """

    raw = os.environ.get("LATENCE_TRACE_MAX_INFLIGHT", "").strip()
    if not raw:
        return 8
    try:
        value = int(raw)
    except ValueError:
        return 8
    return max(1, value)


def _raise_service_error(exc: ServiceError) -> None:
    """Raise a FastAPI ``HTTPException`` with the v1 structured envelope.

    Every non-200 response from ``/groundedness`` carries the same
    ``{code, message, hint, docs_url}`` shape so AI agents can
    branch on the machine-readable ``code`` and surface ``hint`` to a
    human (or to themselves) without having to parse the prose
    ``message``. The envelope is the public PA7 contract.
    """

    status_code = getattr(exc, "status_code", 500)
    detail = {
        "code": getattr(exc, "error_code", "service_error"),
        "message": str(exc),
        "hint": getattr(exc, "hint", None)
        or "See /agent-help for the canonical request shape.",
        "docs_url": "https://latence.ai/trace/docs",
    }
    raise HTTPException(status_code=status_code, detail=detail)


def create_router(service_provider: Callable[[], GroundednessService]) -> APIRouter:
    """Build a FastAPI router bound to a ``GroundednessService`` provider.

    The provider callable is wrapped in :func:`fastapi.Depends` so embedders
    can swap the service implementation at runtime (e.g. mock it in tests or
    inject a chunk-resolver-aware variant in production).
    """

    router = APIRouter(tags=["Groundedness"])
    # PA4: bound inflight scoring requests so a burst can't OOM the GPU
    # or starve the FastAPI threadpool. The semaphore is created lazily
    # on the first request because asyncio primitives must be bound to
    # the running event loop. Asyncio is single-threaded, so the lazy
    # init below is race-free even though it looks like a check-then-set.
    _inflight_limit = _resolve_inflight_limit()
    _semaphore_holder: list[Optional[asyncio.Semaphore]] = [None]

    def _inflight_semaphore() -> asyncio.Semaphore:
        sem = _semaphore_holder[0]
        if sem is None:
            sem = asyncio.Semaphore(_inflight_limit)
            _semaphore_holder[0] = sem
        return sem

    def get_service() -> GroundednessService:
        return service_provider()

    @router.post(
        "/groundedness",
        response_model=GroundednessResponse,
        summary="Groundedness / hallucination detection",
        operation_id="score_groundedness",
        description=(
            "Score a generated response for groundedness against caller-supplied "
            "evidence. Three premise lanes are supported: ``chunk_ids`` (vector "
            "fast path with a chunk_resolver), ``raw_context`` (text segmented "
            "into token-budgeted windows), and ``support_units[]`` (structured, "
            "multi-source/multi-speaker premises with per-unit attribution). "
            "Returns heatmap-ready token-level signals, top evidence links, NLI "
            "diagnostics, and a calibrated risk band. Closed-book by default: "
            "requests with no premises return risk_band='unknown'."
        ),
    )
    async def groundedness(
        request: GroundednessRequest,
        service: GroundednessService = Depends(get_service),
    ) -> GroundednessResponse:
        # PA4: route the synchronous, CPU/GPU-bound service call through
        # FastAPI's threadpool so other async handlers (healthz, readyz,
        # metrics) and concurrent /groundedness calls keep making
        # progress, gated by an inflight semaphore so the GPU/encoder
        # batch queues stay inside their documented limit.
        async with _inflight_semaphore():
            try:
                return await run_in_threadpool(service.groundedness, request)
            except (ValidationError, NotFoundError, ServiceError) as exc:
                _raise_service_error(exc)
                raise  # pragma: no cover - _raise_service_error always raises

    @router.get(
        "/health",
        tags=["Health"],
        summary="Backwards-compatible health alias",
        operation_id="legacy_health",
    )
    async def health() -> dict:
        return {"status": "ok"}

    @router.get(
        "/healthz",
        tags=["Health"],
        summary="Liveness probe",
        operation_id="healthz",
    )
    async def healthz() -> dict:
        return {"status": "ok"}

    @router.get(
        "/readyz",
        tags=["Health"],
        summary="Readiness probe (gated on Triton kernel warmup)",
        operation_id="readyz",
    )
    async def readyz() -> JSONResponse:
        if is_warm():
            warmup = {
                key: {
                    "ok": result.ok,
                    "profile": result.profile,
                    "device": result.device,
                    "elapsed_ms": round(result.elapsed_ms, 2),
                    "shape_count": len(result.shapes),
                    "error": result.error,
                }
                for key, result in warmup_state().items()
            }
            return JSONResponse(
                status_code=200,
                content={
                    "status": "ready",
                    "warmup": warmup,
                    "max_inflight": _inflight_limit,
                },
            )
        return JSONResponse(
            status_code=503,
            content={
                "status": "warming",
                "hint": "Triton kernel JIT cache is still priming. Retry after warmup.",
            },
        )

    @router.get(
        "/agent-help",
        tags=["Discovery"],
        summary="Self-describing agent contract",
        operation_id="agent_help",
        description=(
            "Returns a compact, machine-readable description of the latence-trace "
            "API surface so AI agents can discover the canonical request shape, "
            "premise lanes, and refusal contract without scraping the OpenAPI "
            "schema."
        ),
    )
    async def agent_help(request: Request) -> JSONResponse:
        # Resolution order, most-authoritative first:
        #   1. ``request.app.state.active_profile`` -- set by ``create_app``
        #      so per-app selection (CLI flag / embedded apply_profile call)
        #      wins even when the process eagerly built another app first.
        #   2. ``LATENCE_TRACE_ACTIVE_PROFILE`` -- exported by ``apply_profile``
        #      for routers built outside of ``server.main.create_app``.
        #   3. ``LATENCE_TRACE_PROFILE`` -- the operator-facing selection env.
        #   4. ``DEFAULT_PROFILE`` -- static fallback.
        state_profile = getattr(request.app.state, "active_profile", None)
        active_profile = (
            state_profile
            or os.environ.get("LATENCE_TRACE_ACTIVE_PROFILE")
            or os.environ.get("LATENCE_TRACE_PROFILE")
            or DEFAULT_PROFILE
        )
        return JSONResponse(
            status_code=200,
            content={
                "service": {
                    "name": "latence-trace",
                    "version": "1.0.0",
                    "summary": (
                        "Calibrated, auditable groundedness scoring for RAG and "
                        "evidence-bearing LLM outputs."
                    ),
                },
                "endpoints": {
                    "score": {
                        "method": "POST",
                        "path": "/groundedness",
                        "operation_id": "score_groundedness",
                    },
                    "agent_help": {"method": "GET", "path": "/agent-help"},
                    "ai_plugin": {
                        "method": "GET",
                        "path": "/.well-known/ai-plugin.json",
                    },
                    "openapi": {"method": "GET", "path": "/openapi.json"},
                    "docs": {"method": "GET", "path": "/docs"},
                    "healthz": {"method": "GET", "path": "/healthz"},
                    "readyz": {"method": "GET", "path": "/readyz"},
                },
                "premise_lanes": [
                    {
                        "name": "chunk_ids",
                        "description": "Stable chunk ids resolved by a caller-provided chunk_resolver.",
                    },
                    {
                        "name": "raw_context",
                        "description": "Free-form premise text segmented into token-budgeted windows.",
                    },
                    {
                        "name": "support_units",
                        "description": "Structured premises with per-unit attribution (source_id, speaker, timestamp, metadata).",
                    },
                ],
                "attribution_modes": [
                    {
                        "value": "closed_book",
                        "description": (
                            "Default. Refuses to score zero-evidence inputs and returns "
                            "risk_band='unknown', reason='no_premise_supplied'."
                        ),
                    },
                    {
                        "value": "open_domain",
                        "description": (
                            "Reserved for the post-v1 retrieval-callback lane. Currently "
                            "returns risk_band='unsupported' so callers can detect the "
                            "lane is not yet enabled."
                        ),
                    },
                ],
                "profiles": {
                    "active": active_profile,
                    "default": DEFAULT_PROFILE,
                    "available": list(PROFILE_NAMES),
                    "selection_env": "LATENCE_TRACE_PROFILE",
                    "description": (
                        "Pareto-optimal default presets. Override the env var "
                        "before starting the server, or pass --profile to "
                        "`latence-trace serve` / `latence-trace score`."
                    ),
                },
                "error_envelope": {
                    "code": "machine-readable error code (e.g. 'validation_error')",
                    "message": "human-readable description of what went wrong",
                    "hint": "actionable guidance for the agent",
                    "docs_url": "deep link to the relevant doc page",
                },
                "limits": {
                    "max_inflight": _inflight_limit,
                    "max_inflight_env": "LATENCE_TRACE_MAX_INFLIGHT",
                },
                "docs_url": "https://latence.ai/trace/docs",
            },
        )

    @router.get(
        "/.well-known/ai-plugin.json",
        tags=["Discovery"],
        summary="Well-known descriptor for AI agent runtimes",
        operation_id="ai_plugin_descriptor",
        description=(
            "Standard /.well-known descriptor used by ChatGPT plugins, Claude "
            "tools, and other AI agent runtimes to discover the API surface "
            "and authentication policy."
        ),
    )
    async def ai_plugin_descriptor() -> JSONResponse:
        return JSONResponse(
            status_code=200,
            content={
                "schema_version": "v1",
                "name_for_human": "latence-trace",
                "name_for_model": "latence_trace",
                "description_for_human": (
                    "Calibrated, auditable groundedness scoring for RAG and "
                    "evidence-bearing LLM outputs."
                ),
                "description_for_model": (
                    "Use score_groundedness to verify that a model-generated "
                    "response is grounded in caller-supplied evidence. Returns "
                    "calibrated probabilities, an ordinal risk band, NLI claim "
                    "verdicts, and a token-level heatmap. Closed-book by default - "
                    "empty premises return risk_band='unknown'."
                ),
                "auth": {"type": "none"},
                "api": {
                    "type": "openapi",
                    "url": "/openapi.json",
                },
                "logo_url": "https://latence.ai/trace/logo.png",
                "contact_email": "support@latence.ai",
                "legal_info_url": "https://latence.ai/trace/legal",
            },
        )

    return router


__all__ = ["create_router"]
