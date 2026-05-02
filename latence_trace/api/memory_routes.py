"""FastAPI routes for TRACE Memory."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from latence_trace.memory.models import MemoryUpdateRequest, MemoryUpdateResponse
from latence_trace.memory.service import update_memory


def create_memory_router(*, prefix: str = "/v1/memory") -> APIRouter:
    router = APIRouter(prefix=prefix, tags=["Memory"])

    @router.post(
        "/update",
        response_model=MemoryUpdateResponse,
        operation_id="memory_update",
        summary="Update caller-carried TRACE Memory state",
    )
    async def memory_update(request: MemoryUpdateRequest) -> MemoryUpdateResponse:
        try:
            return update_memory(request)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return router
