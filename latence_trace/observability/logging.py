"""Structured JSON log formatter + request-id correlation filter.

Every log line carries:

- ``ts``: ISO-8601 timestamp with microsecond resolution.
- ``level``: log level name.
- ``logger``: logger module path.
- ``message``: rendered message body.
- ``request_id``: copied from ``X-Request-ID`` header when present, else
  a uuid4 generated on demand. This is the canonical correlation key
  across logs / metrics labels / OTel trace ids.
- ``trace_id`` and ``span_id``: hex-encoded current span IDs when an
  OTel span is active, so log entries link back to the trace view.
- everything passed via ``logger.info(..., extra={...})``.

Production deployments call :func:`configure_json_logging` from the
entrypoint exactly once; CI / dev keeps the default human-readable
format.
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
import time
import uuid
from typing import Any, Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

REQUEST_ID_HEADER = "x-request-id"
_REQUEST_ID: contextvars.ContextVar[str] = contextvars.ContextVar(
    "latence_trace_request_id", default=""
)


def current_request_id() -> str:
    return _REQUEST_ID.get()


class JsonFormatter(logging.Formatter):
    """Render log records as one JSON object per line."""

    _IGNORED = {
        "args", "asctime", "created", "exc_info", "exc_text", "filename",
        "funcName", "levelname", "levelno", "lineno", "module", "msecs",
        "message", "msg", "name", "pathname", "process", "processName",
        "relativeCreated", "stack_info", "thread", "threadName",
        "taskName",
    }

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": _iso_timestamp(record.created),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        rid = current_request_id()
        if rid:
            payload["request_id"] = rid
        try:
            from opentelemetry import trace

            span = trace.get_current_span()
            ctx = span.get_span_context() if span else None
            if ctx and ctx.is_valid:
                payload["trace_id"] = format(ctx.trace_id, "032x")
                payload["span_id"] = format(ctx.span_id, "016x")
        except Exception:  # pragma: no cover - OTel optional
            pass
        for key, value in record.__dict__.items():
            if key in self._IGNORED:
                continue
            if key.startswith("_"):
                continue
            try:
                json.dumps(value)
            except TypeError:
                value = repr(value)
            payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, separators=(",", ":"), default=str)


class RequestIdFilter(logging.Filter):
    """Attach the current request id to every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = current_request_id()
        return True


def configure_json_logging(level: int = logging.INFO) -> None:
    """Replace the root handler with a JSON-formatted stderr handler.

    Idempotent: calling twice does not stack handlers. Uvicorn's own
    loggers are coerced to the same handler so the wire-level access
    log is also JSON-formatted -- one log shape end-to-end.
    """

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter())
    handler.addFilter(RequestIdFilter())

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level)

    for logger_name in (
        "uvicorn",
        "uvicorn.access",
        "uvicorn.error",
        "fastapi",
        "latence_trace",
    ):
        log = logging.getLogger(logger_name)
        log.handlers.clear()
        log.propagate = True


def install_request_id_filter() -> RequestIdFilter:
    """Install the request-id filter on the root logger when callers
    don't want full JSON formatting (e.g. tests with caplog)."""

    flt = RequestIdFilter()
    logging.getLogger().addFilter(flt)
    return flt


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Echo the X-Request-ID header back, generating one if missing."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        rid = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex
        token = _REQUEST_ID.set(rid)
        try:
            response = await call_next(request)
        finally:
            _REQUEST_ID.reset(token)
        response.headers[REQUEST_ID_HEADER] = rid
        return response


def _iso_timestamp(epoch: float) -> str:
    seconds, frac = divmod(epoch, 1.0)
    base = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(seconds))
    return f"{base}.{int(frac * 1_000_000):06d}Z"
