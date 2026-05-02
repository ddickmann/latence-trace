"""Standalone uvicorn entry point for the latence-trace Groundedness Tracker.

Mounts the :mod:`latence_trace.api.routes` router at ``POST /groundedness``
and exposes ``/healthz`` (liveness) plus ``/readyz`` (Triton-kernel
warmup gate) probes. The service uses the default encoder factory which
honours ``VOYAGER_GROUNDEDNESS_VLLM_ENDPOINT`` for the production
vLLM-Factory deployment and falls back to a local pylate model otherwise.

Operators select a Pareto-optimal default profile via either the
``LATENCE_TRACE_PROFILE`` environment variable or the ``--profile`` CLI
flag. Supported values are ``fast``, ``balanced`` (default) and
``quality``; see ``research/triangular_maxsim/reports/profile_pareto.md``
for the sweep that backs each knee. Any environment variable already
exported by the operator wins over the profile preset, so production
tuning decisions remain authoritative.

Triton-kernel warmup runs in the background on startup. By the time
``/readyz`` flips to 200 the JIT cache is primed and the first user
request matches steady-state p95.
"""

from __future__ import annotations

import argparse
import logging
import os
import threading

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from latence_trace.api.compliance_routes import create_compliance_router
from latence_trace.api.compliance_service import ComplianceRedactionService
from latence_trace.api.compression_routes import create_compression_router
from latence_trace.api.compression_service import CompressionService
from latence_trace.api.memory_routes import create_memory_router
from latence_trace.api.routes import create_router
from latence_trace.api.service import (
    DEFAULT_PROFILE,
    PROFILE_NAMES,
    GroundednessService,
    apply_profile,
)
from latence_trace.auth import LicenseMiddleware
from latence_trace.kernels.warmup import warm_all
from latence_trace.middleware import RateLimitMiddleware, RequestIdMiddleware
from latence_trace.observability import (
    PrometheusMiddleware,
    configure_json_logging,
    configure_tracing,
    create_metrics_router,
    register_default_collectors,
)
from latence_trace.observability.metrics import update_license_gauge

logger = logging.getLogger(__name__)


def _warmup_disabled() -> bool:
    raw = os.environ.get("LATENCE_TRACE_DISABLE_WARMUP", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _kick_off_warmup(profile: str | None) -> None:
    """Run Triton kernel warmup off the main thread.

    The warmup is a few hundred ms on CUDA and effectively free on CPU
    deploys (the reference path does the same shape grid). Either way we
    don't want to block uvicorn's startup hook; the readiness probe
    gates traffic until ``warm_all`` returns.
    """

    if _warmup_disabled():
        logger.info("kernel_warmup_skipped", extra={"reason": "LATENCE_TRACE_DISABLE_WARMUP"})
        # Still flip the warmup gate so /readyz reports ready - the
        # operator explicitly opted out of warmup, so blocking traffic
        # would be incorrect.
        from latence_trace.kernels.warmup import _WARMUP_DONE  # noqa: PLC0415

        _WARMUP_DONE.set()
        return

    target_profile = (profile or DEFAULT_PROFILE).strip().lower()

    def _worker() -> None:
        try:
            warm_all(target_profile)
        except Exception as exc:  # pragma: no cover - defensive logging only
            logger.warning(
                "kernel_warmup_thread_failed",
                extra={"profile": target_profile, "error": str(exc)},
            )

    thread = threading.Thread(target=_worker, name="latence-trace-warmup", daemon=True)
    thread.start()

_service: GroundednessService | None = None
_service_lock = threading.Lock()
_compliance_service: ComplianceRedactionService | None = None
_compliance_service_lock = threading.Lock()
_compression_service: CompressionService | None = None
_compression_service_lock = threading.Lock()


def _get_service() -> GroundednessService:
    """Lazily build the singleton ``GroundednessService``.

    The first request builds the encoder cache, NLI provider stubs, and
    null-bank scaffolding. We guard the construction with a lock so two
    concurrent in-flight requests do not race and double-instantiate.
    """

    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = GroundednessService(
                    compression_service=_get_compression_service(),
                    device=os.environ.get("LATENCE_TRACE_DEVICE", "cpu"),
                    collection_label=os.environ.get(
                        "LATENCE_TRACE_COLLECTION_LABEL", "latence-trace"
                    ),
                )
    return _service


def _get_compliance_service() -> ComplianceRedactionService:
    """Lazily build the singleton compliance redaction service."""

    global _compliance_service
    if _compliance_service is None:
        with _compliance_service_lock:
            if _compliance_service is None:
                _compliance_service = ComplianceRedactionService.from_env()
    return _compliance_service


def _get_compression_service() -> CompressionService:
    """Lazily build the singleton compression service."""

    global _compression_service
    if _compression_service is None:
        with _compression_service_lock:
            if _compression_service is None:
                _compression_service = CompressionService.from_env()
    return _compression_service


def _resolve_profile_from_env() -> str | None:
    raw = os.environ.get("LATENCE_TRACE_PROFILE")
    if raw is None:
        return DEFAULT_PROFILE
    label = raw.strip().lower()
    if not label or label == "none":
        return None
    return label


def create_app(profile: str | None = None) -> FastAPI:
    """Build the FastAPI app, applying the requested default profile.

    The profile env overlay is installed *before* the
    :class:`GroundednessService` is instantiated so the encoder factory,
    NLI provider resolver and risk-band classifier all see the same
    configuration. Operator-set environment variables are never
    overwritten.
    """

    selected = profile if profile is not None else _resolve_profile_from_env()
    applied_profile_name: str | None = None
    if selected:
        try:
            applied = apply_profile(selected)
            applied_profile_name = applied.profile
            logger.info(
                "groundedness_profile_active",
                extra={
                    "profile": applied.profile,
                    "applied_count": len(applied.applied),
                    "skipped_count": len(applied.skipped),
                },
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc

    app = FastAPI(
        title="latence-trace Groundedness Tracker",
        version="1.0.0",
        description=(
            "Calibrated, auditable groundedness scoring with **two lanes**:\n\n"
            "- **RAG lane** (`scoring_mode=rag`, default) — enterprise "
            "retrieval-augmented LLM apps. MaxSim + NLI + atomic claims + "
            "structured-evidence AND-gate, calibrated thresholds, per-claim "
            "evidence, `context_coverage_ratio` retrieval observability.\n"
            "- **Code lane** (`scoring_mode=code`) — coding agents "
            "(Claude Code, Cursor, Codex, OpenCode, …). AST-grounded "
            "literal matching, ambiguity-triggered NLI cascade, logistic "
            "composite, multi-turn per-file ownership with reason codes.\n\n"
            "Part of the latence.ai product family. Closed-book by default; "
            "supports `chunk_ids`, `raw_context`, and structured "
            "`support_units[]` premise lanes plus the bilingual EN + DE "
            "pipeline.\n\n"
            "**Discovery:**\n"
            "- `GET /agent-help` returns a compact agent-friendly contract "
            "(premise lanes, attribution modes, error envelope).\n"
            "- `GET /.well-known/ai-plugin.json` returns a standard tool "
            "descriptor for ChatGPT, Claude and other AI agent runtimes.\n"
            "- `GET /openapi.json` returns the full OpenAPI 3.1 schema "
            "(includes the code-lane diagnostics block)."
        ),
        summary=(
            "Dual-lane groundedness scoring: RAG for enterprises, "
            "Code lane for coding agents."
        ),
        openapi_url="/openapi.json",
        docs_url="/docs",
        redoc_url="/redoc",
        contact={"name": "latence.ai", "url": "https://latence.ai/trace"},
        license_info={
            "name": "Commercial",
            "url": "https://latence.ai/trace/legal",
        },
    )
    # Stamp the active profile on app.state so /agent-help reports the
    # per-app selection (CLI flag / embedded apply_profile call) even when
    # an earlier eager ``app = create_app()`` already wrote
    # ``LATENCE_TRACE_ACTIVE_PROFILE`` for the process.
    app.state.active_profile = applied_profile_name or DEFAULT_PROFILE

    # --- L4 observability ------------------------------------------------
    # Switch to JSON-formatted logging when the operator opts in. We
    # keep the human-readable default for ``LATENCE_TRACE_LOG_FORMAT``
    # absent so local dev / CI does not see machine-only output.
    log_format = os.environ.get("LATENCE_TRACE_LOG_FORMAT", "").strip().lower()
    if log_format == "json":
        configure_json_logging()
    register_default_collectors(
        profile=app.state.active_profile,
        version=app.version,
    )
    configure_tracing(app, service_version=app.version)

    # --- L3 license + L5 rate limit + observability middleware ----------
    # Eagerly load the license so the Prometheus gauge can publish the
    # days-until-expiry value before the first scrape. We pass the
    # already-validated claims into LicenseMiddleware so the per-request
    # path does no extra work.
    from latence_trace.auth.license import (  # noqa: PLC0415
        LicenseError,
        load_license_from_env,
    )

    license_claims = None
    try:
        license_claims = load_license_from_env()
    except LicenseError as exc:
        logger.error(
            "license_load_failed",
            extra={"code": exc.code, "error_detail": str(exc)},
        )

    if license_claims is not None:
        update_license_gauge(
            subject=license_claims.subject,
            tier=license_claims.tier,
            days_until_expiry=license_claims.days_until_expiry,
        )

    # Starlette processes ``add_middleware`` in LIFO order, so the
    # *last* call wraps the others outermost. We want
    # RequestId -> License -> RateLimit -> Prometheus -> route, so we
    # add them in REVERSE here:
    app.add_middleware(PrometheusMiddleware)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(
        LicenseMiddleware,
        license_claims=license_claims,
    )
    app.add_middleware(RequestIdMiddleware)

    app.include_router(create_metrics_router())
    app.include_router(create_router(_get_service))
    app.include_router(create_compliance_router(_get_compliance_service))
    app.include_router(
        create_compliance_router(
            _get_compliance_service,
            prefix="/v1/compliance",
            operation_id_prefix="v1_compliance",
        )
    )
    app.include_router(create_compression_router(_get_compression_service))
    app.include_router(create_memory_router())

    if os.environ.get("LATENCE_TRACE_ENABLE_MCP_HTTP", "0") in {"1", "true", "yes"}:
        try:
            from latence_trace.mcp.remote import _make_router as _make_mcp_router

            app.include_router(_make_mcp_router())
            logger.info("mounted remote MCP endpoint at /mcp")
        except Exception:  # noqa: BLE001
            logger.exception("failed to mount remote MCP endpoint")

    # PA7 polish: coerce FastAPI's default 422 validation responses into
    # the same {code, message, hint, docs_url} envelope that runtime
    # service errors use. Without this, agents have to handle two
    # different error shapes (the FastAPI list-of-pydantic-errors at the
    # schema layer, and our envelope at the service layer). Returning
    # one shape across the board keeps the agent contract honest.
    def _safe_validation_errors(raw_errors: list) -> list:
        """Strip non-JSON-serialisable members (e.g. ``ValueError`` instances
        in pydantic's ``ctx`` block) so the structured envelope can survive
        ``json.dumps`` no matter what the validator put in there."""

        cleaned: list = []
        for entry in raw_errors:
            safe_entry: dict = {}
            for key, value in entry.items():
                if key == "ctx" and isinstance(value, dict):
                    safe_entry[key] = {
                        ck: (str(cv) if isinstance(cv, BaseException) else cv)
                        for ck, cv in value.items()
                    }
                elif isinstance(value, BaseException):
                    safe_entry[key] = str(value)
                else:
                    safe_entry[key] = value
            cleaned.append(safe_entry)
        return cleaned

    @app.exception_handler(RequestValidationError)
    async def _handle_request_validation_error(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        raw_errors = exc.errors()
        primary = raw_errors[0] if raw_errors else {}
        message = primary.get("msg") or "Request validation failed"
        loc = primary.get("loc") or []
        loc_str = ".".join(str(part) for part in loc if part not in ("body",))
        if loc_str:
            message = f"{loc_str}: {message}"
        return JSONResponse(
            status_code=422,
            content={
                "detail": {
                    "code": "validation_error",
                    "message": message,
                    "hint": (
                        "Inspect /agent-help or the OpenAPI schema for the "
                        "canonical request shape; every field is documented "
                        "with the value range it accepts."
                    ),
                    "docs_url": "https://latence.ai/trace/docs",
                    "errors": _safe_validation_errors(raw_errors),
                }
            },
        )

    # Kick off Triton warmup right after the router is mounted so the
    # /readyz gate goes green as soon as the JIT cache is hot. Keep it
    # off the main thread so uvicorn does not block its startup hook.
    _kick_off_warmup(selected)
    return app


def app_factory() -> FastAPI:
    """Uvicorn factory hook so workers honour the configured profile.

    Using ``--factory`` ensures every worker (re-)applies the env preset
    on import even when the CLI flag mutated ``LATENCE_TRACE_PROFILE``
    after the parent process imported this module.
    """

    return create_app()


# Eagerly construct an ``app`` symbol for embedders that import
# ``server.main:app`` directly (e.g. ASGI hosts that do not support the
# factory pattern). The CLI entrypoint always uses ``app_factory`` so
# the per-worker profile selection stays consistent.
app = create_app()


def run() -> None:
    parser = argparse.ArgumentParser(description="latence-trace Groundedness Tracker server")
    parser.add_argument("--host", default=os.environ.get("LATENCE_TRACE_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("LATENCE_TRACE_PORT", "8090")))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("LATENCE_TRACE_WORKERS", "1")))
    parser.add_argument("--reload", action="store_true")
    parser.add_argument(
        "--profile",
        choices=list(PROFILE_NAMES) + ["none"],
        default=None,
        help=(
            "Pareto-optimal default profile to load before the service starts. "
            "Falls back to LATENCE_TRACE_PROFILE (default 'balanced'). Use "
            "'none' to opt out and rely entirely on env vars."
        ),
    )
    args = parser.parse_args()

    if args.profile is not None:
        os.environ["LATENCE_TRACE_PROFILE"] = args.profile

    import uvicorn

    uvicorn.run(
        "server.main:app_factory",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        factory=True,
    )


if __name__ == "__main__":
    run()
