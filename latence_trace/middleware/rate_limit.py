"""Token-bucket rate limiter middleware (L5).

Algorithm:

- Each "key" (license subject by default; falls back to client IP for
  unlicensed dev deploys) gets its own bucket with capacity = burst,
  refilled at rate tokens/second.
- ``rate`` defaults to ``min(license.max_qps or env, env)``, so the
  contractual cap from the JWT is always honored even if an operator
  forgets to set the env.
- ``burst`` defaults to ``2 * rate`` (ceil), matching the "burst is
  twice the steady state" rule of thumb.
- When the bucket is empty the middleware returns 429 with
  ``Retry-After`` (seconds, rounded up to the next int) so well-behaved
  clients back off correctly.

Bucket state lives in-process so a multi-worker deployment shares
*nothing* across workers; the per-worker burst caps are summed. This is
intentional -- a serious global limiter would need Redis and we don't
want to ship a Redis dependency for the default config. Operators with
multi-worker deployments size their burst per-worker accordingly, or
plug their own gateway-level limiter in front of the API.
"""

from __future__ import annotations

import math
import os
import threading
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Iterable, Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from latence_trace.auth.license import LicenseClaims

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
)


@dataclass
class TokenBucket:
    """Classic refill-on-read token bucket.

    Thread-safe via the ``_lock`` member; we only spend a few
    instructions inside the critical section so contention stays
    irrelevant even under high RPS.
    """

    capacity: float
    rate_per_sec: float
    tokens: float
    updated: float
    _lock: threading.Lock

    @classmethod
    def new(cls, *, capacity: float, rate_per_sec: float) -> "TokenBucket":
        now = time.monotonic()
        return cls(
            capacity=float(capacity),
            rate_per_sec=float(rate_per_sec),
            tokens=float(capacity),
            updated=now,
            _lock=threading.Lock(),
        )

    def take(self) -> tuple[bool, float]:
        """Try to take one token. Returns (allowed, retry_after_seconds)."""

        with self._lock:
            now = time.monotonic()
            elapsed = now - self.updated
            self.updated = now
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate_per_sec)
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return True, 0.0
            deficit = 1.0 - self.tokens
            retry_after = deficit / max(self.rate_per_sec, 1e-6)
            return False, retry_after


def resolve_rate_from_license(
    claims: Optional[LicenseClaims],
    *,
    env_default: Optional[float] = None,
) -> Optional[float]:
    """Return the effective steady-state RPS, or None for unlimited.

    The license claim wins over the env (the contract is authoritative);
    the env wins over the implicit default of "no limit". Returning None
    disables the rate limiter entirely.
    """

    license_qps = float(claims.max_qps) if claims and claims.max_qps else None
    env_qps: Optional[float] = env_default
    raw = os.environ.get("LATENCE_TRACE_RATE_LIMIT_RPS")
    if raw:
        try:
            env_qps = float(raw)
        except ValueError:
            env_qps = env_default

    if license_qps is None and env_qps is None:
        return None
    if license_qps is None:
        return env_qps
    if env_qps is None:
        return license_qps
    return min(license_qps, env_qps)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Enforce a token-bucket per identity key on protected routes."""

    def __init__(
        self,
        app,
        *,
        rate_per_sec: Optional[float] = None,
        burst: Optional[float] = None,
        exempt_paths: Iterable[str] = DEFAULT_EXEMPT_PATHS,
    ) -> None:
        super().__init__(app)
        self._exempt = tuple(exempt_paths)
        # If neither the license nor the env set a rate, the limiter is a
        # pass-through. We still install the middleware so operators can
        # SIGHUP-reload a license with max_qps and pick it up at runtime.
        self._default_rate = rate_per_sec
        self._default_burst = burst
        self._buckets: dict[str, TokenBucket] = {}
        self._lock = threading.Lock()

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[JSONResponse]],
    ):
        path = request.url.path
        if path in self._exempt or any(path.startswith(p + "/") for p in self._exempt):
            return await call_next(request)

        claims: Optional[LicenseClaims] = getattr(request.state, "license", None)
        rate = resolve_rate_from_license(claims, env_default=self._default_rate)
        if rate is None or rate <= 0:
            return await call_next(request)

        burst = self._default_burst
        if burst is None:
            burst = max(1.0, math.ceil(rate * 2.0))

        key = self._identity_key(claims, request)
        bucket = self._get_or_create_bucket(key, capacity=burst, rate_per_sec=rate)
        allowed, retry_after = bucket.take()
        if allowed:
            return await call_next(request)

        retry_seconds = max(1, math.ceil(retry_after))
        response = JSONResponse(
            status_code=429,
            content={
                "detail": {
                    "code": "rate_limited",
                    "message": (
                        f"rate limit exceeded for {key} "
                        f"({rate:g} req/s, burst {int(burst)})"
                    ),
                    "hint": (
                        f"retry after {retry_seconds}s; raise the limit by "
                        "issuing a license with a higher max_qps or by setting "
                        "LATENCE_TRACE_RATE_LIMIT_RPS."
                    ),
                    "docs_url": "https://latence.ai/trace/docs/operations/rate-limits",
                }
            },
        )
        response.headers["Retry-After"] = str(retry_seconds)
        response.headers["X-RateLimit-Limit"] = f"{rate:g}"
        response.headers["X-RateLimit-Burst"] = str(int(burst))
        return response

    def _identity_key(self, claims: Optional[LicenseClaims], request: Request) -> str:
        if claims and claims.subject:
            return f"license:{claims.subject}"
        client = request.client
        if client and client.host:
            return f"ip:{client.host}"
        return "anonymous"

    def _get_or_create_bucket(
        self, key: str, *, capacity: float, rate_per_sec: float
    ) -> TokenBucket:
        bucket = self._buckets.get(key)
        if bucket is not None:
            # Honor a license update (rate may have grown / shrunk after a
            # SIGHUP-style reload). Rebuild only when the documented
            # capacity changes so steady-state buckets keep their tokens.
            if bucket.capacity != capacity or bucket.rate_per_sec != rate_per_sec:
                with self._lock:
                    bucket = TokenBucket.new(capacity=capacity, rate_per_sec=rate_per_sec)
                    self._buckets[key] = bucket
            return bucket
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = TokenBucket.new(capacity=capacity, rate_per_sec=rate_per_sec)
                self._buckets[key] = bucket
            return bucket
