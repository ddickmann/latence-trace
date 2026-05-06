"""FastAPI routes for the compliance redaction runtime."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from latence_trace.api.compliance_models import (
    ComplianceRedactionRequest,
    ComplianceRedactionResponse,
    compliance_schema_metadata,
)
from latence_trace.api.compliance_service import (
    ComplianceRedactionService,
    ComplianceServiceError,
)


def _resolve_compliance_inflight_limit() -> int:
    raw = os.environ.get("LATENCE_TRACE_COMPLIANCE_MAX_INFLIGHT", "").strip()
    if not raw:
        raw = os.environ.get("LATENCE_TRACE_COMPLIANCE_MAX_CONCURRENCY", "").strip()
    try:
        return max(1, int(raw)) if raw else 8
    except ValueError:
        return 8


def _raise_compliance_error(exc: ComplianceServiceError) -> None:
    raise HTTPException(
        status_code=getattr(exc, "status_code", 500),
        detail={
            "code": getattr(exc, "error_code", "compliance_service_error"),
            "message": str(exc),
            "hint": getattr(exc, "hint", None)
            or "Inspect /v1/compliance/schema for valid redaction labels and modes.",
            "docs_url": "https://latence.ai/trace/docs",
        },
    )


def create_compliance_router(
    service_provider: Callable[[], ComplianceRedactionService],
    *,
    prefix: str = "/compliance",
    operation_id_prefix: str = "compliance",
) -> APIRouter:
    router = APIRouter(prefix=prefix, tags=["Compliance"])
    inflight_limit = _resolve_compliance_inflight_limit()
    semaphore_holder: list[asyncio.Semaphore | None] = [None]

    def _semaphore() -> asyncio.Semaphore:
        sem = semaphore_holder[0]
        if sem is None:
            sem = asyncio.Semaphore(inflight_limit)
            semaphore_holder[0] = sem
        return sem

    def get_service() -> ComplianceRedactionService:
        return service_provider()

    @router.post(
        "/redact",
        response_model=ComplianceRedactionResponse,
        operation_id=f"{operation_id_prefix}_redact",
        summary="Detect and redact GDPR PII",
        description=(
            "Detect GDPR/enterprise PII with GLiNER, optional custom regex labels, "
            "recall-preserving boundary cleanup, advisory format metadata, and "
            "mask or replacement redaction. "
            "The service returns spans, labels, timings, and privacy-safe usage "
            "metadata without logging raw PII."
        ),
    )
    async def redact(
        request: ComplianceRedactionRequest,
        service: ComplianceRedactionService = Depends(get_service),
    ) -> ComplianceRedactionResponse:
        async with _semaphore():
            try:
                return await run_in_threadpool(service.redact, request)
            except ComplianceServiceError as exc:
                _raise_compliance_error(exc)
                raise

    @router.get(
        "/schema",
        operation_id=f"{operation_id_prefix}_schema",
        summary="Compliance redaction label schema",
    )
    async def schema() -> dict:
        return compliance_schema_metadata()

    @router.get(
        "/healthz",
        operation_id=f"{operation_id_prefix}_healthz",
        summary="Compliance runtime readiness",
    )
    async def healthz(
        service: ComplianceRedactionService = Depends(get_service),
    ) -> JSONResponse:
        try:
            health = await run_in_threadpool(service.provider.healthcheck)
            return JSONResponse(
                status_code=200,
                content={
                    "status": "ready",
                    "provider": health,
                    "max_inflight": inflight_limit,
                    "max_text_tokens": service.max_text_tokens,
                    "max_model_len": service.max_model_len,
                    "replacement_dataset": service.replacement_dataset_stats(),
                },
            )
        except Exception as exc:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unavailable",
                    "error": str(exc),
                    "max_inflight": inflight_limit,
                },
            )

    return router
