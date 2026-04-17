"""Standalone uvicorn entry point for the latence-trace Groundedness Tracker.

Mounts the :mod:`latence_trace.api.routes` router at ``POST /groundedness``
and exposes a ``/health`` probe. The service uses the default encoder
factory which honours ``VOYAGER_GROUNDEDNESS_VLLM_ENDPOINT`` for the
production vLLM-factory deployment and falls back to a local pylate model
otherwise.
"""

from __future__ import annotations

import argparse
import logging
import os

from fastapi import FastAPI

from latence_trace.api.routes import create_router
from latence_trace.api.service import GroundednessService

logger = logging.getLogger(__name__)

_service: GroundednessService | None = None


def _get_service() -> GroundednessService:
    global _service
    if _service is None:
        _service = GroundednessService(
            device=os.environ.get("LATENCE_TRACE_DEVICE", "cpu"),
            collection_label=os.environ.get("LATENCE_TRACE_COLLECTION_LABEL", "latence-trace"),
        )
    return _service


def create_app() -> FastAPI:
    app = FastAPI(
        title="latence-trace Groundedness Tracker (Beta)",
        version="0.1.0",
        description=(
            "Calibrated, auditable groundedness scoring for RAG and evidence-bearing "
            "LLM outputs. Part of the latence.ai product family."
        ),
    )
    app.include_router(create_router(_get_service))
    return app


app = create_app()


def run() -> None:
    parser = argparse.ArgumentParser(description="latence-trace Groundedness Tracker server")
    parser.add_argument("--host", default=os.environ.get("LATENCE_TRACE_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("LATENCE_TRACE_PORT", "8090")))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("LATENCE_TRACE_WORKERS", "1")))
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(
        "server.main:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
    )


if __name__ == "__main__":
    run()
