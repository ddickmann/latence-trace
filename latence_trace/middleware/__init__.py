"""Cross-cutting ASGI middleware (rate limiting, request id).

The rate limiter (L5) is a per-key token bucket sized from either the
license ``max_qps`` claim or the ``LATENCE_TRACE_RATE_LIMIT_RPS`` env
var. Requests beyond the bucket capacity get a 429 with a
``Retry-After`` header and the project-wide error envelope.

The request-id middleware lives in ``latence_trace.observability.logging``
because the JSON formatter pulls the id from the same context var; we
re-export it here for callers who only care about the wiring side.
"""

from latence_trace.middleware.rate_limit import (
    RateLimitMiddleware,
    TokenBucket,
    resolve_rate_from_license,
)
from latence_trace.observability.logging import RequestIdMiddleware

__all__ = [
    "RateLimitMiddleware",
    "RequestIdMiddleware",
    "TokenBucket",
    "resolve_rate_from_license",
]
