"""OpenTelemetry tracing helpers.

Importing this module never opens a network connection: when the OTel
SDK is not configured we return a no-op tracer, so library code can
freely sprinkle ``with span("...")`` calls without making the
in-process / CI deployment depend on an OTLP collector.

Operators opt in by exporting ``OTEL_EXPORTER_OTLP_ENDPOINT`` (the
standard env var); :func:`configure_tracing` then wires the OTLP HTTP
exporter, sets the resource attributes, and instruments FastAPI +
HTTPX. The function is idempotent so reload-style restarts do not
register the same instrumentation twice.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Iterator, Optional

from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.trace import Span, Tracer

logger = logging.getLogger(__name__)

_TRACER_NAME = "latence-trace"
_INSTRUMENTED = False


def configure_tracing(
    app: FastAPI,
    *,
    service_name: str = "latence-trace",
    service_version: str = "1.0.0",
) -> bool:
    """Wire OTel exporters + auto-instrumentation. Returns True on success.

    Required env: ``OTEL_EXPORTER_OTLP_ENDPOINT``. Optional env:
    ``OTEL_SERVICE_NAME``, ``OTEL_RESOURCE_ATTRIBUTES``. When the
    endpoint is unset the function is a no-op and returns False so the
    caller can decide whether to log it.
    """

    global _INSTRUMENTED
    if _INSTRUMENTED:
        return True

    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        return False

    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError as exc:  # pragma: no cover - extras-only path
        logger.warning(
            "otel_extras_missing",
            extra={
                "hint": (
                    "install opentelemetry-exporter-otlp + "
                    "opentelemetry-instrumentation-fastapi to enable tracing"
                ),
                "error": str(exc),
            },
        )
        return False

    resource = Resource.create({
        "service.name": os.environ.get("OTEL_SERVICE_NAME", service_name),
        "service.version": service_version,
        "deployment.environment": os.environ.get("LATENCE_TRACE_ENV", "production"),
    })

    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    trace.set_tracer_provider(provider)

    FastAPIInstrumentor.instrument_app(app)
    HTTPXClientInstrumentor().instrument()

    _INSTRUMENTED = True
    logger.info(
        "tracing_configured",
        extra={"endpoint": endpoint, "service": service_name},
    )
    return True


def get_tracer(name: Optional[str] = None) -> Tracer:
    """Return a tracer scoped to the latence-trace package."""

    return trace.get_tracer(name or _TRACER_NAME)


@contextmanager
def span(name: str, **attributes) -> Iterator[Span]:
    """Tiny convenience wrapper so library modules can do
    ``with span("encode.context"): ...`` without importing OTel
    boilerplate. When tracing is not configured the underlying tracer
    is a no-op and the span overhead is a few function calls."""

    tracer = get_tracer()
    with tracer.start_as_current_span(name, attributes=attributes or None) as current:
        yield current
