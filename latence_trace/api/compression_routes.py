"""FastAPI routes for standalone compression."""

from __future__ import annotations

from typing import Callable

from fastapi import APIRouter, Depends, HTTPException

from latence_trace.api.compression_models import CompressionRequest, CompressionResponse
from latence_trace.api.compression_service import CompressionService


def create_compression_router(
    service_provider: Callable[[], CompressionService],
    *,
    prefix: str = "/v1/compression",
    operation_id_prefix: str = "compression",
) -> APIRouter:
    router = APIRouter(prefix=prefix, tags=["Compression"])

    def get_service() -> CompressionService:
        return service_provider()

    @router.post(
        "",
        response_model=CompressionResponse,
        operation_id=f"{operation_id_prefix}_compress",
        summary="Compress text or chat messages for TRACE Memory",
    )
    async def compress(
        request: CompressionRequest,
        service: CompressionService = Depends(get_service),
    ) -> CompressionResponse:
        try:
            return await service.compress(request)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return router
