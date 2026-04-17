"""FastAPI router for the standalone latence-trace Groundedness Tracker.

The router exposes a single ``POST /groundedness`` endpoint that mirrors the
historical voyager-index ``/collections/{name}/groundedness`` payload but
without the per-collection namespace - latence-trace owns no collections.
"""

from __future__ import annotations

from typing import Callable

from fastapi import APIRouter, Depends, HTTPException

from latence_trace.api.models import GroundednessRequest, GroundednessResponse
from latence_trace.api.service import (
    GroundednessService,
    NotFoundError,
    ServiceError,
    ValidationError,
)


def _raise_service_error(exc: ServiceError) -> None:
    code = getattr(exc, "status_code", 500)
    detail = {"error_code": getattr(exc, "error_code", "service_error"), "message": str(exc)}
    raise HTTPException(status_code=code, detail=detail)


def create_router(service_provider: Callable[[], GroundednessService]) -> APIRouter:
    """Build a FastAPI router bound to a ``GroundednessService`` provider.

    The provider callable is wrapped in :func:`fastapi.Depends` so embedders
    can swap the service implementation at runtime (e.g. mock it in tests or
    inject a chunk-resolver-aware variant in production).
    """

    router = APIRouter(tags=["Groundedness"])

    def get_service() -> GroundednessService:
        return service_provider()

    @router.post(
        "/groundedness",
        response_model=GroundednessResponse,
        summary="Groundedness / hallucination detection (Beta)",
        description=(
            "Post-generation groundedness scoring over final chunk_ids or raw_context. "
            "Returns heatmap-ready support signals and evidence links. "
            "Beta feature: useful for groundedness and evidence tracing, but not a final factuality oracle."
        ),
    )
    async def groundedness(
        request: GroundednessRequest,
        service: GroundednessService = Depends(get_service),
    ) -> GroundednessResponse:
        try:
            return service.groundedness(request)
        except (ValidationError, NotFoundError, ServiceError) as exc:
            _raise_service_error(exc)
            raise  # pragma: no cover - _raise_service_error always raises

    @router.get("/health", tags=["Health"])
    async def health() -> dict:
        return {"status": "ok"}

    return router


__all__ = ["create_router"]
