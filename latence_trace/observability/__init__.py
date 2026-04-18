"""Observability layer (L4): Prometheus metrics + OpenTelemetry tracing
+ structured JSON logging.

The three sub-modules are independently importable so a deployment can
opt out of any one of them by simply not wiring it. Defaults are
production-safe:

- Prometheus collectors are *registered globally on first import* so
  any module that `from latence_trace.observability.metrics import
  REQUEST_COUNT` sees the same counter; this matches how
  ``prometheus_client`` is used in long-running services.
- OTel falls back to a no-op tracer when no exporter is configured,
  so importing the module never opens a network connection by accident.
- JSON logging is opt-in -- call :func:`configure_json_logging` once
  in your entrypoint to switch from the default human-readable
  formatter.
"""

from latence_trace.observability.logging import (
    configure_json_logging,
    install_request_id_filter,
)
from latence_trace.observability.metrics import (
    ENCODE_LATENCY,
    NLI_LATENCY,
    PROFILE_GAUGE,
    PrometheusMiddleware,
    REQUEST_COUNT,
    REQUEST_INFLIGHT,
    REQUEST_LATENCY,
    SCORE_GROUNDEDNESS,
    SCORE_LATENCY,
    create_metrics_router,
    register_default_collectors,
)
from latence_trace.observability.tracing import (
    configure_tracing,
    get_tracer,
    span,
)

__all__ = [
    "ENCODE_LATENCY",
    "NLI_LATENCY",
    "PROFILE_GAUGE",
    "PrometheusMiddleware",
    "REQUEST_COUNT",
    "REQUEST_INFLIGHT",
    "REQUEST_LATENCY",
    "SCORE_GROUNDEDNESS",
    "SCORE_LATENCY",
    "configure_json_logging",
    "configure_tracing",
    "create_metrics_router",
    "get_tracer",
    "install_request_id_filter",
    "register_default_collectors",
    "span",
]
