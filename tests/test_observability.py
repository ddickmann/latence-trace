"""Smoke tests for the L4 observability layer."""

from __future__ import annotations

import logging
from io import StringIO

from fastapi import FastAPI
from fastapi.testclient import TestClient

from latence_trace.observability import (
    PrometheusMiddleware,
    create_metrics_router,
    register_default_collectors,
)
from latence_trace.observability.logging import (
    JsonFormatter,
    RequestIdMiddleware,
    current_request_id,
)


def _build_app() -> TestClient:
    app = FastAPI()
    app.add_middleware(PrometheusMiddleware)
    app.add_middleware(RequestIdMiddleware)
    app.include_router(create_metrics_router())

    @app.get("/echo")
    def echo() -> dict:
        return {"request_id": current_request_id()}

    return TestClient(app)


def test_metrics_endpoint_exposes_prometheus_text() -> None:
    register_default_collectors(profile="balanced", version="1.0.0")
    client = _build_app()
    client.get("/echo")
    body = client.get("/metrics").text
    assert "latence_trace_requests_total" in body
    assert "latence_trace_request_duration_seconds" in body
    assert 'latence_trace_active_profile{profile="balanced"} 1.0' in body


def test_request_id_round_trip_uses_supplied_header() -> None:
    client = _build_app()
    response = client.get("/echo", headers={"X-Request-ID": "rid-123"})
    assert response.json()["request_id"] == "rid-123"
    assert response.headers["x-request-id"] == "rid-123"


def test_request_id_generated_when_missing() -> None:
    client = _build_app()
    response = client.get("/echo")
    body = response.json()
    assert body["request_id"]
    assert response.headers["x-request-id"] == body["request_id"]


def test_lane_counters_exposed_on_metrics_endpoint() -> None:
    """The code-lane quality-boost sprint added counters for lane mix,
    cascade fires, lane-budget backpressure, and phantom verdicts. This
    guards that they are registered against the default registry and
    scrapeable through ``/metrics`` — any accidental rename or removal
    breaks the IDE-plugin Grafana board.
    """

    from latence_trace.observability.metrics import (
        BUDGET_EXCEEDED_COUNT,
        CASCADE_FIRE_COUNT,
        LANE_REQUEST_COUNT,
        PHANTOM_VERDICT_COUNT,
    )

    LANE_REQUEST_COUNT.labels(lane="rag").inc()
    LANE_REQUEST_COUNT.labels(lane="code").inc()
    CASCADE_FIRE_COUNT.labels(lane="code").inc()
    BUDGET_EXCEEDED_COUNT.labels(lane="code").inc()
    PHANTOM_VERDICT_COUNT.labels(verdict="false").inc()

    client = _build_app()
    body = client.get("/metrics").text
    assert "latence_trace_lane_requests_total" in body
    assert "latence_trace_code_lane_cascade_fires_total" in body
    assert "latence_trace_lane_budget_exceeded_total" in body
    assert "latence_trace_code_lane_phantom_verdicts_total" in body


def test_json_formatter_emits_one_line_per_record() -> None:
    buf = StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("latence_trace.test_json_logger")
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    logger.info("hello", extra={"customer": "acme"})
    line = buf.getvalue().strip()
    import json as _json

    parsed = _json.loads(line)
    assert parsed["message"] == "hello"
    assert parsed["customer"] == "acme"
    assert parsed["level"] == "INFO"
    assert "ts" in parsed
