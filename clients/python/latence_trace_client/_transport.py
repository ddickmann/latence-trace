"""Shared retry / backoff / envelope-decoding helpers.

The sync and async clients differ only in how they perform the HTTP
call. The retry policy, backoff schedule, OTel propagation, and error
decoding live here so the two implementations stay byte-for-byte
identical in observable behaviour.
"""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from latence_trace_client.errors import (
    LatenceTraceAPIError,
    LatenceTraceAuthError,
    LatenceTraceRateLimited,
    LatenceTraceServerError,
    LatenceTraceValidationError,
    _Envelope,
)


DEFAULT_USER_AGENT = "latence-trace-client/1.0.0"
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_RETRIES = 4
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


@dataclass(frozen=True)
class RetryPolicy:
    """Capped exponential backoff with full jitter.

    Sleep on attempt ``k`` is ``random.uniform(0, min(cap, base * 2**k))``
    seconds, plus the server-supplied ``Retry-After`` header when the
    response is a 429. ``base`` defaults to 0.25 s; ``cap`` to 8 s.
    """

    max_retries: int = DEFAULT_MAX_RETRIES
    base_seconds: float = 0.25
    cap_seconds: float = 8.0

    def sleep_for(self, attempt: int, retry_after_header: Optional[float]) -> float:
        if retry_after_header is not None:
            return max(0.0, retry_after_header)
        backoff = min(self.cap_seconds, self.base_seconds * math.pow(2, attempt))
        return random.uniform(0.0, backoff)


def default_headers(api_key: Optional[str], extra: Optional[Mapping[str, str]] = None) -> dict:
    """Compose request headers with bearer auth + UA + W3C trace context."""

    headers: dict = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "User-Agent": DEFAULT_USER_AGENT,
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    inject_trace_context(headers)
    if extra:
        headers.update(extra)
    return headers


def inject_trace_context(headers: dict) -> None:
    """Best-effort W3C traceparent injection.

    When the OTel SDK is configured the current span context is encoded
    into ``traceparent`` so the server log lines / spans correlate to
    the caller's trace. When the SDK is missing or no span is active
    this is a no-op and the SDK still works.
    """

    try:
        from opentelemetry import trace
        from opentelemetry.propagate import inject

        span = trace.get_current_span()
        ctx = span.get_span_context() if span else None
        if ctx and ctx.is_valid:
            inject(headers)
    except Exception:  # pragma: no cover - OTel optional
        pass


def parse_retry_after(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def decode_error(status: int, body: Any, request_id: Optional[str]) -> LatenceTraceAPIError:
    envelope = _envelope_from_body(body)
    message = envelope.message if envelope and envelope.message else f"HTTP {status}"
    if status in (401, 402, 403):
        return LatenceTraceAuthError(message, status=status, envelope=envelope, request_id=request_id)
    if status == 422:
        return LatenceTraceValidationError(message, status=status, envelope=envelope, request_id=request_id)
    if status == 429:
        return LatenceTraceRateLimited(message, status=status, envelope=envelope, request_id=request_id)
    if 500 <= status < 600:
        return LatenceTraceServerError(message, status=status, envelope=envelope, request_id=request_id)
    return LatenceTraceAPIError(message, status=status, envelope=envelope, request_id=request_id)


def _envelope_from_body(body: Any) -> Optional[_Envelope]:
    if not isinstance(body, Mapping):
        return None
    detail = body.get("detail")
    if isinstance(detail, Mapping):
        body = detail
    if not isinstance(body, Mapping):
        return None
    code = body.get("code")
    if not isinstance(code, str):
        return None
    return _Envelope(
        code=code,
        message=str(body.get("message") or ""),
        hint=body.get("hint") if isinstance(body.get("hint"), str) else None,
        docs_url=body.get("docs_url") if isinstance(body.get("docs_url"), str) else None,
        extra={
            k: v
            for k, v in body.items()
            if k not in {"code", "message", "hint", "docs_url"}
        },
    )


def coerce_base_url(base_url: Optional[str]) -> str:
    base = base_url or os.environ.get("LATENCE_TRACE_URL", "http://localhost:8090")
    return base.rstrip("/")


def coerce_api_key(api_key: Optional[str]) -> Optional[str]:
    return api_key or os.environ.get("LATENCE_TRACE_API_KEY")


__all__: Sequence[str] = (
    "DEFAULT_TIMEOUT_SECONDS",
    "DEFAULT_USER_AGENT",
    "RetryPolicy",
    "RETRYABLE_STATUS",
    "coerce_api_key",
    "coerce_base_url",
    "decode_error",
    "default_headers",
    "parse_retry_after",
)
