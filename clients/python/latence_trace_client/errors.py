"""Typed exceptions surfaced by the SDK.

Every error carries the structured envelope the server returns
(``code``, ``message``, ``hint``, ``docs_url``) plus the HTTP status
so calling code can branch on a stable machine-readable token without
parsing prose.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass
class _Envelope:
    code: str
    message: str
    hint: str | None = None
    docs_url: str | None = None
    extra: Mapping[str, Any] = ()  # type: ignore[assignment]

    def as_dict(self) -> dict:
        return {
            "code": self.code,
            "message": self.message,
            "hint": self.hint,
            "docs_url": self.docs_url,
            "extra": dict(self.extra) if self.extra else {},
        }


class LatenceTraceAPIError(Exception):
    """Base error for any non-2xx response from the API."""

    def __init__(
        self,
        message: str,
        *,
        status: int,
        envelope: _Envelope | None = None,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.envelope = envelope
        self.request_id = request_id
        self.code = envelope.code if envelope else "unknown_error"
        self.hint = envelope.hint if envelope else None
        self.docs_url = envelope.docs_url if envelope else None

    @property
    def status_code(self) -> int:
        """Compatibility alias used by integration adapters."""

        return self.status


class LatenceTraceAuthError(LatenceTraceAPIError):
    """Raised on 401 / 402 / 403 (license or auth failure)."""


class LatenceTraceValidationError(LatenceTraceAPIError):
    """Raised on 422 (request schema rejection)."""


class LatenceTraceRateLimited(LatenceTraceAPIError):
    """Raised on 429 after retries are exhausted."""

    def __init__(
        self,
        message: str,
        *,
        status: int = 429,
        envelope: _Envelope | None = None,
        request_id: str | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(
            message,
            status=status,
            envelope=envelope,
            request_id=request_id,
        )
        self.retry_after = retry_after


class LatenceTraceServerError(LatenceTraceAPIError):
    """Raised on persistent 5xx after retries are exhausted."""


class LatenceTraceTimeout(LatenceTraceAPIError):
    """Raised when the request never completed before the timeout."""

    def __init__(self, message: str, *, request_id: str | None = None) -> None:
        super().__init__(message, status=0, envelope=None, request_id=request_id)
