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
from collections.abc import Callable

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from latence_trace.api.models import (
    GroundednessRequest,
    GroundednessResponse,
    RollupRequest,
    RollupResponse,
)
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
    _semaphore_holder: list[asyncio.Semaphore | None] = [None]

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

    @router.post(
        "/groundedness/rollup",
        response_model=RollupResponse,
        summary="Stateless session-level rollup",
        operation_id="rollup_groundedness",
        description=(
            "Aggregate a sequence of per-turn records into conversation-level "
            "metrics (noise %, model drift %, retrieval waste %, reason-code "
            "histogram, top dead files, risk-band trail, recommendations). "
            "Purely stateless — nothing is persisted. CPU-only; no GPU or "
            "model calls."
        ),
    )
    async def groundedness_rollup(
        request: RollupRequest,
        service: GroundednessService = Depends(get_service),
    ) -> RollupResponse:
        try:
            return service.rollup(request)
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
                    "rollup": {
                        "method": "POST",
                        "path": "/groundedness/rollup",
                        "operation_id": "rollup_groundedness",
                    },
                    "compliance_redact": {
                        "method": "POST",
                        "path": "/v1/compliance/redact",
                        "operation_id": "v1_compliance_redact",
                    },
                    "compliance_schema": {
                        "method": "GET",
                        "path": "/v1/compliance/schema",
                        "operation_id": "v1_compliance_schema",
                    },
                    "compliance_healthz": {
                        "method": "GET",
                        "path": "/v1/compliance/healthz",
                        "operation_id": "v1_compliance_healthz",
                    },
                    "compression": {
                        "method": "POST",
                        "path": "/v1/compression",
                        "operation_id": "compression_compress",
                    },
                    "memory_update": {
                        "method": "POST",
                        "path": "/v1/memory/update",
                        "operation_id": "memory_update",
                    },
                    "trace_session_create": {
                        "method": "POST",
                        "path": "/v1/trace/sessions",
                        "operation_id": "trace_session_create",
                    },
                    "trace_session_get": {
                        "method": "GET",
                        "path": "/v1/trace/sessions/{session_id}",
                        "operation_id": "trace_session_get",
                    },
                    "trace_session_event": {
                        "method": "POST",
                        "path": "/v1/trace/sessions/{session_id}/events",
                        "operation_id": "trace_session_event",
                    },
                    "trace_session_memory_update": {
                        "method": "POST",
                        "path": "/v1/trace/sessions/{session_id}/memory/update",
                        "operation_id": "trace_session_memory_update",
                    },
                    "trace_session_score": {
                        "method": "POST",
                        "path": "/v1/trace/sessions/{session_id}/score",
                        "operation_id": "trace_session_score",
                    },
                    "trace_session_context": {
                        "method": "GET",
                        "path": "/v1/trace/sessions/{session_id}/context",
                        "operation_id": "trace_session_context",
                    },
                    "trace_session_source": {
                        "method": "GET",
                        "path": "/v1/trace/sessions/{session_id}/sources/{source_id}",
                        "operation_id": "trace_session_source",
                    },
                    "trace_session_source_post": {
                        "method": "POST",
                        "path": "/v1/trace/sessions/{session_id}/sources/{source_id}",
                        "operation_id": "trace_session_source_post",
                    },
                    "trace_session_repair": {
                        "method": "POST",
                        "path": "/v1/trace/sessions/{session_id}/repair",
                        "operation_id": "trace_session_repair",
                    },
                    "trace_session_rollup": {
                        "method": "POST",
                        "path": "/v1/trace/sessions/{session_id}/rollup",
                        "operation_id": "trace_session_rollup",
                    },
                    "trace_session_close": {
                        "method": "DELETE",
                        "path": "/v1/trace/sessions/{session_id}",
                        "operation_id": "trace_session_close",
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
                "state_model": {
                    "stateless_compute": [
                        "score",
                        "rollup",
                        "compliance_redact",
                        "compression",
                    ],
                    "caller_carried_state": ["memory_update"],
                    "server_stateful": [
                        "trace_session_create",
                        "trace_session_get",
                        "trace_session_event",
                        "trace_session_memory_update",
                        "trace_session_score",
                        "trace_session_context",
                        "trace_session_source",
                        "trace_session_repair",
                        "trace_session_rollup",
                        "trace_session_close",
                    ],
                    "note": (
                        "Default TRACE sessions use process-local state. Production "
                        "RunPod workflows should use caller-carried memory state or "
                        "a deployment with durable SessionStore persistence."
                    ),
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
                "observability": {
                    "context_coverage": {
                        "request_field": "coverage_threshold",
                        "default": 0.5,
                        "range": [0.0, 1.0],
                        "response_globals": [
                            "scores.context_coverage_ratio",
                            "scores.context_coverage_threshold",
                            "scores.support_units_used",
                            "scores.support_units_total",
                            "scores.context_attribution_ratio",
                            "scores.context_attribution_used_count",
                        ],
                        "response_per_unit": [
                            "support_units[*].coverage_score",
                            "support_units[*].used",
                        ],
                        "description": (
                            "Retrieval-efficiency observability. Per-unit "
                            "coverage_score is the max similarity any response "
                            "token had to that unit; used=False flags chunks the "
                            "retriever pulled but the response did not lean on. "
                            "context_coverage_ratio is used_count/total — a "
                            "ratio of 0.4 means 60% of the retrieved chunks were "
                            "dead weight."
                        ),
                    },
                    "unused_context": {
                        "response_globals": [
                            "scores.support_units_usage_used",
                            "scores.support_units_unused",
                            "scores.support_units_uncertain",
                            "scores.context_usage_ratio",
                            "scores.context_unused_ratio",
                            "scores.context_uncertain_ratio",
                        ],
                        "response_per_unit": [
                            "support_units[*].usage_state",
                            "support_units[*].usage_confidence",
                            "support_units[*].unused_confidence",
                        ],
                        "description": (
                            "Precision-first unused-context contract. "
                            "``usage_state=unused`` is emitted only for high-"
                            "confidence negatives; semantically overlapping, "
                            "partially used, or otherwise ambiguous units are "
                            "surfaced as ``uncertain`` instead."
                        ),
                    },
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

    _mount_amber_queue(router)

    return router


def _mount_amber_queue(router: APIRouter) -> None:
    """Mount the amber-queue review + decision endpoints (Plan C4).

    Served off the main FastAPI router so self-hosted operators and
    the hosted gateway share the same contract.  Access control is
    delegated to the caller (Cloudflare gateway / LicenseMiddleware);
    this layer only enforces that the tenant id on the URL matches
    the one the upstream resolved.
    """

    try:
        from latence_trace.middleware.amber_queue import (
            list_amber_records,
            list_reviewer_decisions,
            write_reviewer_decision,
        )
    except Exception:  # pragma: no cover - missing optional deps
        return

    @router.get("/v1/amber/{tenant_id}")
    def _list_amber(tenant_id: str, limit: int = 200) -> JSONResponse:
        records = list_amber_records(tenant_id, limit=limit)
        return JSONResponse(content={"tenant_id": tenant_id, "records": records})

    @router.post("/v1/amber/{tenant_id}/decisions")
    async def _record_decision(tenant_id: str, request: Request) -> JSONResponse:
        try:
            body = await request.json()
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "invalid_json",
                    "message": f"request body was not valid JSON: {exc}",
                    "hint": (
                        "POST a JSON object with request_id, reviewer_id, "
                        "action, and optionally comment / corrected_response."
                    ),
                    "docs_url": "https://latence.ai/trace/docs",
                },
            ) from exc
        action = str(body.get("action", "")).lower()
        if action not in {"accept", "edit", "reject"}:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "invalid_action",
                    "message": "action must be one of accept, edit, reject.",
                    "hint": "Use 'edit' when the reviewer supplies a corrected_response.",
                    "docs_url": "https://latence.ai/trace/docs",
                },
            )
        try:
            record = write_reviewer_decision(
                request_id=str(body.get("request_id", "")),
                tenant_id=tenant_id,
                reviewer_id=str(body.get("reviewer_id", "")),
                action=action,  # type: ignore[arg-type]
                comment=body.get("comment"),
                corrected_response=body.get("corrected_response"),
            )
        except RuntimeError as exc:
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "audit_log_disabled",
                    "message": str(exc),
                    "hint": "Set LATENCE_TRACE_AUDIT_LOG_DIR in the worker env.",
                    "docs_url": "https://latence.ai/trace/docs",
                },
            ) from exc
        return JSONResponse(content=record)

    @router.get("/v1/amber/{tenant_id}/decisions")
    def _list_decisions(tenant_id: str) -> JSONResponse:
        records = list(list_reviewer_decisions(tenant_id))
        return JSONResponse(content={"tenant_id": tenant_id, "decisions": records})


__all__ = ["create_router"]
