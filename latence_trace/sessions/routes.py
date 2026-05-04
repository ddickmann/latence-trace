"""FastAPI routes for stateful TRACE sessions."""

from __future__ import annotations

from collections.abc import Callable

from fastapi import APIRouter, Depends, HTTPException

from latence_trace.api.service import ServiceError
from latence_trace.sessions.models import (
    TraceSessionCloseResponse,
    TraceSessionContextResponse,
    TraceSessionCreateRequest,
    TraceSessionCreateResponse,
    TraceSessionEventRequest,
    TraceSessionEventResponse,
    TraceSessionGetResponse,
    TraceSessionRepairRequest,
    TraceSessionRepairResponse,
    TraceSessionRollupRequest,
    TraceSessionRollupResponse,
    TraceSessionScoreRequest,
    TraceSessionScoreResponse,
    TraceSessionSourceRequest,
    TraceSessionSourceResponse,
)
from latence_trace.sessions.service import TraceSessionService


def create_trace_session_router(
    service_provider: Callable[[], TraceSessionService],
    *,
    prefix: str = "/v1/trace/sessions",
) -> APIRouter:
    router = APIRouter(prefix=prefix, tags=["Trace Sessions"])

    def get_service() -> TraceSessionService:
        return service_provider()

    @router.post("", response_model=TraceSessionCreateResponse, operation_id="trace_session_create")
    async def create_session(
        request: TraceSessionCreateRequest,
        service: TraceSessionService = Depends(get_service),
    ) -> TraceSessionCreateResponse:
        return _call(lambda: service.create(request))

    @router.get("/{session_id}", response_model=TraceSessionGetResponse, operation_id="trace_session_get")
    async def get_session(
        session_id: str,
        service: TraceSessionService = Depends(get_service),
    ) -> TraceSessionGetResponse:
        return _call(lambda: service.get(session_id))

    @router.post(
        "/{session_id}/events",
        response_model=TraceSessionEventResponse,
        operation_id="trace_session_event",
    )
    async def append_event(
        session_id: str,
        request: TraceSessionEventRequest,
        service: TraceSessionService = Depends(get_service),
    ) -> TraceSessionEventResponse:
        return _call(lambda: service.append_event(session_id, request))

    @router.post(
        "/{session_id}/score",
        response_model=TraceSessionScoreResponse,
        operation_id="trace_session_score",
    )
    async def score_session(
        session_id: str,
        request: TraceSessionScoreRequest,
        service: TraceSessionService = Depends(get_service),
    ) -> TraceSessionScoreResponse:
        return _call(lambda: service.score(session_id, request))

    @router.post(
        "/{session_id}/memory/update",
        response_model=TraceSessionEventResponse,
        operation_id="trace_session_memory_update",
    )
    async def update_memory(
        session_id: str,
        request: TraceSessionEventRequest,
        service: TraceSessionService = Depends(get_service),
    ) -> TraceSessionEventResponse:
        return _call(lambda: service.append_event(session_id, request))

    @router.get(
        "/{session_id}/context",
        response_model=TraceSessionContextResponse,
        operation_id="trace_session_context",
    )
    async def context(
        session_id: str,
        service: TraceSessionService = Depends(get_service),
    ) -> TraceSessionContextResponse:
        return _call(lambda: service.context(session_id))

    @router.get(
        "/{session_id}/sources/{source_id}",
        response_model=TraceSessionSourceResponse,
        operation_id="trace_session_source",
    )
    async def source(
        session_id: str,
        source_id: str,
        include_raw: bool = False,
        service: TraceSessionService = Depends(get_service),
    ) -> TraceSessionSourceResponse:
        return _call(lambda: service.source(session_id, source_id, include_raw=include_raw))

    @router.post(
        "/{session_id}/sources/{source_id}",
        response_model=TraceSessionSourceResponse,
        operation_id="trace_session_source_post",
    )
    async def source_post(
        session_id: str,
        source_id: str,
        request: TraceSessionSourceRequest,
        service: TraceSessionService = Depends(get_service),
    ) -> TraceSessionSourceResponse:
        return _call(lambda: service.source(session_id, source_id, include_raw=request.include_raw))

    @router.post(
        "/{session_id}/repair",
        response_model=TraceSessionRepairResponse,
        operation_id="trace_session_repair",
    )
    async def repair(
        session_id: str,
        request: TraceSessionRepairRequest,
        service: TraceSessionService = Depends(get_service),
    ) -> TraceSessionRepairResponse:
        return _call(lambda: service.repair(session_id, request))

    @router.post(
        "/{session_id}/rollup",
        response_model=TraceSessionRollupResponse,
        operation_id="trace_session_rollup",
    )
    async def rollup(
        session_id: str,
        request: TraceSessionRollupRequest,
        service: TraceSessionService = Depends(get_service),
    ) -> TraceSessionRollupResponse:
        return _call(lambda: service.rollup(session_id, request))

    @router.delete(
        "/{session_id}",
        response_model=TraceSessionCloseResponse,
        operation_id="trace_session_close",
    )
    async def close(
        session_id: str,
        service: TraceSessionService = Depends(get_service),
    ) -> TraceSessionCloseResponse:
        return _call(lambda: service.close(session_id))

    return router


def _call(fn):
    try:
        return fn()
    except ServiceError as exc:
        raise HTTPException(
            status_code=getattr(exc, "status_code", 500),
            detail={
                "code": getattr(exc, "error_code", "service_error"),
                "message": str(exc),
                "hint": getattr(exc, "hint", None),
                "docs_url": getattr(exc, "docs_url", None),
            },
        ) from exc
