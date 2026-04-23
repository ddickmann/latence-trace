"""Prometheus metrics + ASGI middleware.

Operators wire this into the FastAPI app via ``create_app`` and the
Helm chart's ``ServiceMonitor``. The exposed metrics give a Fortune 500
ops team everything they need for a basic Grafana board:

- ``latence_trace_requests_total{route,method,status,license_subject}``
- ``latence_trace_request_duration_seconds{route,method}`` (histogram
  with 50/90/95/99 percentile buckets aligned to the 118 ms p95 SLO).
- ``latence_trace_request_inflight{route}``
- ``latence_trace_encode_seconds{stage}`` -- per-stage encoder timing.
- ``latence_trace_nli_seconds`` -- NLI verifier timing.
- ``latence_trace_score{kind="groundedness_v2"|...}`` -- score histogram
  for drift monitoring.
- ``latence_trace_active_profile{profile}`` -- gauge that lets
  alerting catch unexpected profile downgrades.
- ``latence_trace_license_days_until_expiry`` -- gauge that lets
  alerting catch licenses about to lapse.
- ``latence_trace_build_info{version,profile,torch_version}`` -- info
  metric used to correlate alerts with deploys.

The middleware is mounted *after* the license middleware so the metric
labels can include the ``license_subject`` (or ``"unlicensed"`` if the
deployment is intentionally running without a license, e.g. CI).
"""

from __future__ import annotations

import os
import time
from typing import Awaitable, Callable, Sequence

from fastapi import APIRouter
from fastapi.responses import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Info,
    REGISTRY,
    generate_latest,
)
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

# We register collectors against the default global REGISTRY because
# prometheus_client exporters expect that. Tests can pass a private
# registry through ``PrometheusMiddleware(registry=...)`` for isolation.

NAMESPACE = "latence_trace"
LATENCY_BUCKETS_SECONDS: Sequence[float] = (
    0.005, 0.010, 0.025, 0.050, 0.075, 0.100, 0.118,  # p95 SLO bucket
    0.150, 0.200, 0.250, 0.500, 1.000, 2.000, 5.000,
)
SCORE_BUCKETS: Sequence[float] = (
    0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
    0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 1.00,
)


def _counter(name: str, documentation: str, labelnames: Sequence[str]) -> Counter:
    full = f"{NAMESPACE}_{name}"
    existing = REGISTRY._names_to_collectors.get(full)  # type: ignore[attr-defined]
    if isinstance(existing, Counter):
        return existing
    return Counter(full, documentation, labelnames=tuple(labelnames))


def _histogram(
    name: str,
    documentation: str,
    labelnames: Sequence[str],
    buckets: Sequence[float],
) -> Histogram:
    full = f"{NAMESPACE}_{name}"
    existing = REGISTRY._names_to_collectors.get(full)  # type: ignore[attr-defined]
    if isinstance(existing, Histogram):
        return existing
    return Histogram(
        full,
        documentation,
        labelnames=tuple(labelnames),
        buckets=tuple(buckets),
    )


def _gauge(name: str, documentation: str, labelnames: Sequence[str]) -> Gauge:
    full = f"{NAMESPACE}_{name}"
    existing = REGISTRY._names_to_collectors.get(full)  # type: ignore[attr-defined]
    if isinstance(existing, Gauge):
        return existing
    return Gauge(full, documentation, labelnames=tuple(labelnames))


def _info(name: str, documentation: str) -> Info:
    full = f"{NAMESPACE}_{name}"
    existing = REGISTRY._names_to_collectors.get(full)  # type: ignore[attr-defined]
    if isinstance(existing, Info):
        return existing
    return Info(full, documentation)


REQUEST_COUNT = _counter(
    "requests_total",
    "Total HTTP requests handled by the latence-trace API.",
    ("route", "method", "status", "license_subject"),
)
REQUEST_LATENCY = _histogram(
    "request_duration_seconds",
    "End-to-end HTTP request latency, including license + middleware.",
    ("route", "method"),
    LATENCY_BUCKETS_SECONDS,
)
REQUEST_INFLIGHT = _gauge(
    "request_inflight",
    "Concurrent HTTP requests currently being served.",
    ("route",),
)
ENCODE_LATENCY = _histogram(
    "encode_seconds",
    "ColBERT-style encoder latency (per stage: query / context / response).",
    ("stage",),
    LATENCY_BUCKETS_SECONDS,
)
SCORE_LATENCY = _histogram(
    "score_seconds",
    "MaxSim + literal + NLI scoring latency (excluding encode).",
    ("profile",),
    LATENCY_BUCKETS_SECONDS,
)
NLI_LATENCY = _histogram(
    "nli_seconds",
    "NLI verifier latency (entailment classification + reranker).",
    ("model",),
    LATENCY_BUCKETS_SECONDS,
)
SCORE_GROUNDEDNESS = _histogram(
    "score",
    "Distribution of headline scores -- drift detector.",
    ("kind",),
    SCORE_BUCKETS,
)
PROFILE_GAUGE = _gauge(
    "active_profile",
    "1 for the currently active profile, 0 otherwise.",
    ("profile",),
)
LICENSE_DAYS_UNTIL_EXPIRY = _gauge(
    "license_days_until_expiry",
    "Days remaining on the loaded license (negative = expired).",
    ("subject", "tier"),
)
BUILD_INFO = _info(
    "build_info",
    "Version, profile and torch backend reported by this process.",
)
# --- Code-lane counters (sprint: code_lane_quality_boost) --------------
# Per-turn counters so IDE-plugin dashboards (Claude Code, Cursor,
# Codex, OpenCode) and on-call operators can plot lane mix, cascade
# fire rate, backpressure events, and phantom verdicts without having
# to parse JSONL logs. Labels are PII-safe: only enum-like values
# (``rag``/``code``, ``true``/``false``) appear.
LANE_REQUEST_COUNT = _counter(
    "lane_requests_total",
    "Total groundedness requests bucketed by scoring lane.",
    ("lane",),
)
CASCADE_FIRE_COUNT = _counter(
    "code_lane_cascade_fires_total",
    "Number of times the code-lane NLI/semantic-entropy cascade fired.",
    ("lane",),
)
BUDGET_EXCEEDED_COUNT = _counter(
    "lane_budget_exceeded_total",
    "Requests that waited on a full per-lane inflight semaphore.",
    ("lane",),
)
PHANTOM_VERDICT_COUNT = _counter(
    "code_lane_phantom_verdicts_total",
    "Code-lane composite phantom verdicts (verdict=true|false).",
    ("verdict",),
)


def register_default_collectors(*, profile: str, version: str, torch_version: str = "") -> None:
    """Stamp the build-info gauge and select the active profile gauge.

    Called once at startup. Subsequent calls re-set the values so a
    process that switches profile via SIGHUP-style reload cleanly
    updates the gauge without leaking stale labels.
    """

    BUILD_INFO.info({
        "version": version,
        "profile": profile,
        "torch_version": torch_version,
    })
    for known in ("fast", "balanced", "quality"):
        PROFILE_GAUGE.labels(profile=known).set(1.0 if known == profile else 0.0)


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Increment the request counter / latency histogram on every call."""

    def __init__(self, app, *, registry: CollectorRegistry = REGISTRY) -> None:
        super().__init__(app)
        self._registry = registry

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        route = _normalise_route(request)
        method = request.method
        license_subject = _license_subject(request)

        REQUEST_INFLIGHT.labels(route=route).inc()
        start = time.perf_counter()
        status_code = "500"
        try:
            response = await call_next(request)
            status_code = str(response.status_code)
            return response
        finally:
            elapsed = time.perf_counter() - start
            REQUEST_INFLIGHT.labels(route=route).dec()
            REQUEST_LATENCY.labels(route=route, method=method).observe(elapsed)
            REQUEST_COUNT.labels(
                route=route,
                method=method,
                status=status_code,
                license_subject=license_subject,
            ).inc()


def _normalise_route(request: Request) -> str:
    """Use the FastAPI route template (e.g. ``/groundedness``) when
    available so the metric cardinality stays bounded -- raw URL path
    blows up on routes with path parameters."""

    route = request.scope.get("route")
    if route is not None and getattr(route, "path", None):
        return str(route.path)
    return request.url.path


def _license_subject(request: Request) -> str:
    license_obj = getattr(request.state, "license", None)
    subject = getattr(license_obj, "subject", None)
    if isinstance(subject, str) and subject:
        return subject
    if os.environ.get("LATENCE_TRACE_LICENSE_REQUIRE", "true").strip().lower() in {
        "0", "false", "no", "off",
    }:
        return "unlicensed"
    return "unknown"


def create_metrics_router() -> APIRouter:
    """FastAPI router that exposes ``GET /metrics`` in the Prometheus
    text exposition format. Mounted unconditionally so scrapers always
    have something to read, even before the first request arrives."""

    router = APIRouter(tags=["Observability"])

    @router.get("/metrics", include_in_schema=False)
    async def metrics_endpoint() -> Response:
        payload = generate_latest(REGISTRY)
        return Response(content=payload, media_type=CONTENT_TYPE_LATEST)

    return router


def update_license_gauge(
    *,
    subject: str,
    tier: str,
    days_until_expiry: float,
) -> None:
    """Public accessor used by the license middleware on startup."""

    LICENSE_DAYS_UNTIL_EXPIRY.labels(subject=subject, tier=tier).set(days_until_expiry)
