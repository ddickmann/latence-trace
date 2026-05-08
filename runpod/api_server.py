"""Production FastAPI server with RunPod action parity.

This module lives beside ``handler.py`` and reuses its initialization lifecycle
and service instances. Product endpoints return the native API payloads, while
``/run`` and ``/runsync`` preserve the RunPod-shaped envelope for compatibility.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import handler as runpod_handler
from fastapi import FastAPI, Request

from latence_trace.api.compliance_routes import create_compliance_router
from latence_trace.api.compression_routes import create_compression_router
# TRACE Retrieval-Only Pivot: memory router removed from user-facing paths.
# from latence_trace.api.memory_routes import create_memory_router
from latence_trace.api.routes import create_router


def _groundedness_service():
    runpod_handler.initialize()
    if runpod_handler._service is None:
        raise RuntimeError("Groundedness service is not initialized")
    return runpod_handler._service


def _compliance_service():
    runpod_handler.initialize()
    if runpod_handler._compliance_service is None:
        raise RuntimeError("Compliance service is not initialized")
    return runpod_handler._compliance_service


def _compression_service():
    runpod_handler.initialize()
    if runpod_handler._compression_service is None:
        raise RuntimeError("Compression service is not initialized")
    return runpod_handler._compression_service


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    runpod_handler.initialize()
    try:
        yield
    finally:
        runpod_handler.shutdown()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Latence TRACE API",
        version=runpod_handler.__version__,
        description="Production FastAPI server for the stateless TRACE runtime.",
        lifespan=lifespan,
    )
    app.include_router(create_router(_groundedness_service))
    app.include_router(
        create_compliance_router(
            _compliance_service,
            prefix="/v1/compliance",
            operation_id_prefix="v1_compliance",
        )
    )
    app.include_router(create_compression_router(_compression_service))
    # TRACE Retrieval-Only Pivot: memory router removed.
    # app.include_router(create_memory_router())

    @app.get("/healthz/runpod-runtime", include_in_schema=False)
    async def runpod_runtime_health() -> dict[str, Any]:
        return runpod_handler._health_payload()

    @app.post("/run", include_in_schema=False)
    @app.post("/runsync", include_in_schema=False)
    async def runpod_compat(request: Request) -> dict[str, Any]:
        body = await request.json()
        job = body if isinstance(body, dict) and isinstance(body.get("input"), dict) else {"input": body}
        return await runpod_handler.handler(job)

    return app


app = create_app()


__all__ = ["app", "create_app"]
