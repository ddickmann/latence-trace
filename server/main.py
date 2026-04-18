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
from typing import Optional

from fastapi import FastAPI

from latence_trace.api.routes import create_router
from latence_trace.api.service import (
    DEFAULT_PROFILE,
    GroundednessService,
    PROFILE_NAMES,
    apply_profile,
)
from latence_trace.kernels.warmup import warm_all

logger = logging.getLogger(__name__)


def _warmup_disabled() -> bool:
    raw = os.environ.get("LATENCE_TRACE_DISABLE_WARMUP", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _kick_off_warmup(profile: Optional[str]) -> None:
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

_service: Optional[GroundednessService] = None
_service_lock = threading.Lock()


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
                    device=os.environ.get("LATENCE_TRACE_DEVICE", "cpu"),
                    collection_label=os.environ.get(
                        "LATENCE_TRACE_COLLECTION_LABEL", "latence-trace"
                    ),
                )
    return _service


def _resolve_profile_from_env() -> Optional[str]:
    raw = os.environ.get("LATENCE_TRACE_PROFILE")
    if raw is None:
        return DEFAULT_PROFILE
    label = raw.strip().lower()
    if not label or label == "none":
        return None
    return label


def create_app(profile: Optional[str] = None) -> FastAPI:
    """Build the FastAPI app, applying the requested default profile.

    The profile env overlay is installed *before* the
    :class:`GroundednessService` is instantiated so the encoder factory,
    NLI provider resolver and risk-band classifier all see the same
    configuration. Operator-set environment variables are never
    overwritten.
    """

    selected = profile if profile is not None else _resolve_profile_from_env()
    if selected:
        try:
            applied = apply_profile(selected)
            logger.info(
                "groundedness_profile_active",
                extra={
                    "profile": applied.profile,
                    "applied_count": len(applied.applied),
                    "skipped_count": len(applied.skipped),
                },
            )
        except ValueError as exc:
            raise SystemExit(str(exc))

    app = FastAPI(
        title="latence-trace Groundedness Tracker",
        version="1.0.0",
        description=(
            "Calibrated, auditable groundedness scoring for RAG and "
            "evidence-bearing LLM outputs. Part of the latence.ai product "
            "family. Closed-book by default; supports `chunk_ids`, "
            "`raw_context`, and structured `support_units[]` premise lanes "
            "plus the bilingual EN + DE pipeline.\n\n"
            "**Discovery:**\n"
            "- `GET /agent-help` returns a compact agent-friendly contract "
            "(premise lanes, attribution modes, error envelope).\n"
            "- `GET /.well-known/ai-plugin.json` returns a standard tool "
            "descriptor for ChatGPT, Claude and other AI agent runtimes."
        ),
        summary=(
            "Closed-book groundedness scoring + hallucination detection for "
            "RAG and evidence-bearing LLM outputs."
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
    app.include_router(create_router(_get_service))

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
