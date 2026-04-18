"""ASGI middleware that enforces the L3 commercial license per request.

The middleware loads the license once at startup (cheap signature
verification on a small JWT) and refuses traffic whenever it is
missing, expired or fingerprint-mismatched. Health probes (``/healthz``,
``/readyz``) are deliberately exempt so Kubernetes can keep restarting
a pod with an expired license -- the operator still needs the readiness
signal to know the service was reachable.

The error envelope follows the project-wide
``{code, message, hint, docs_url}`` shape so AI agents and humans
branch on a stable ``code`` instead of parsing prose.
"""

from __future__ import annotations

import logging
import os
from typing import Awaitable, Callable, Iterable, Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp

from latence_trace.auth.license import (
    LicenseClaims,
    LicenseError,
    LicenseExpired,
    LicenseMissing,
    LicenseNotYetValid,
    load_license_from_env,
)

logger = logging.getLogger(__name__)

DEFAULT_EXEMPT_PATHS: tuple[str, ...] = (
    "/healthz",
    "/health",
    "/readyz",
    "/livez",
    "/metrics",
    "/agent-help",
    "/.well-known/ai-plugin.json",
    "/openapi.json",
    "/docs",
    "/redoc",
    "/docs/oauth2-redirect",
)


class LicenseMiddleware(BaseHTTPMiddleware):
    """Enforce the JWT license on protected routes.

    Two failure modes:

    1. License is required (``LATENCE_TRACE_LICENSE_REQUIRE != false``)
       and missing / expired / invalid -> every protected request gets
       the structured error envelope with the appropriate HTTP status
       (402 for missing/expired, 403 for invalid).
    2. License is *not* required (dev mode) -> requests pass through but
       a single warning is logged at startup so an operator does not
       accidentally ship a license-less binary to production.

    A successful license is exposed via ``request.state.license`` so
    downstream middleware (rate limiting, metrics, audit logging) can
    branch on the customer subject and tier.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        license_claims: Optional[LicenseClaims] = None,
        require: Optional[bool] = None,
        exempt_paths: Iterable[str] = DEFAULT_EXEMPT_PATHS,
    ) -> None:
        super().__init__(app)
        self._exempt = tuple(exempt_paths)
        self._require = require if require is not None else _require_from_env()
        self._claims = license_claims
        self._load_error: Optional[LicenseError] = None

        if self._claims is None:
            try:
                self._claims = load_license_from_env(require=self._require)
            except LicenseError as exc:
                # Hold on to the failure so every protected request can
                # surface it; do not crash the process on startup, since
                # ``/healthz`` and ``/readyz`` should still answer (so an
                # operator can SSH into the pod, fix the env var, and
                # restart deterministically).
                self._load_error = exc
                logger.error(
                    "license_load_failed",
                    extra={"code": exc.code, "error_detail": str(exc)},
                )

        if self._claims is not None:
            if self._claims.is_expiring_soon():
                logger.warning(
                    "license_expiring_soon",
                    extra={
                        "subject": self._claims.subject,
                        "tier": self._claims.tier,
                        "days_until_expiry": round(self._claims.days_until_expiry, 2),
                    },
                )
            else:
                logger.info(
                    "license_loaded",
                    extra={
                        "subject": self._claims.subject,
                        "tier": self._claims.tier,
                        "expires_at": self._claims.expires_at,
                        "features": list(self._claims.features),
                    },
                )
        elif not self._require:
            logger.warning(
                "license_enforcement_disabled",
                extra={"hint": "set LATENCE_TRACE_LICENSE_REQUIRE=true in production"},
            )

    @property
    def claims(self) -> Optional[LicenseClaims]:
        """Expose the loaded license for tests + the /agent-help surface."""

        return self._claims

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable["JSONResponse"]],
    ):
        path = request.url.path
        if path in self._exempt or any(path.startswith(p + "/") for p in self._exempt):
            return await call_next(request)

        if self._load_error is not None:
            return _envelope(self._load_error)

        if self._claims is None:
            if self._require:
                return _envelope(LicenseMissing("no license token configured"))
            # Dev mode: leave request.state.license unset, log a single
            # warning above so production deploys notice quickly.
            return await call_next(request)

        # Re-check expiry on every request so a long-running process
        # does not silently keep serving traffic for hours after the
        # token expires.
        try:
            self._claims = self._reverify_if_stale(self._claims)
        except LicenseError as exc:
            return _envelope(exc)

        request.state.license = self._claims
        return await call_next(request)

    def _reverify_if_stale(self, claims: LicenseClaims) -> LicenseClaims:
        import time

        now = time.time()
        if now >= claims.expires_at:
            raise LicenseExpired("license has expired during this process lifetime")
        if claims.not_before is not None and now < claims.not_before:
            raise LicenseNotYetValid("license nbf is still in the future")
        return claims


def license_required(request: Request) -> LicenseClaims:
    """FastAPI ``Depends`` helper: returns the license or 401s.

    Useful for routes that need to branch on the loaded license (e.g.
    feature gates, customer audit logging) without re-running the JWT
    verification.
    """

    claims = getattr(request.state, "license", None)
    if not isinstance(claims, LicenseClaims):
        from fastapi import HTTPException

        raise HTTPException(
            status_code=402,
            detail={
                "code": "license_missing",
                "message": "no license attached to this request",
                "hint": "ensure LATENCE_TRACE_LICENSE is set; restart the server.",
                "docs_url": "https://latence.ai/trace/docs/operations/licensing",
            },
        )
    return claims


def _envelope(exc: LicenseError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.http_status,
        content={
            "detail": {
                "code": exc.code,
                "message": str(exc),
                "hint": (
                    "Set LATENCE_TRACE_LICENSE to your customer-issued JWT, or "
                    "place it at /etc/latence-trace/license. Contact "
                    "support@latence.ai for renewal."
                ),
                "docs_url": "https://latence.ai/trace/docs/operations/licensing",
            }
        },
    )


def _require_from_env() -> bool:
    raw = os.environ.get("LATENCE_TRACE_LICENSE_REQUIRE", "true").strip().lower()
    return raw in {"1", "true", "yes", "on"}
