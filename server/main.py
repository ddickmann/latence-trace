"""Standalone uvicorn entry point for the latence-trace Groundedness Tracker.

Mounts the :mod:`latence_trace.api.routes` router at ``POST /groundedness``
and exposes a ``/health`` probe. The service uses the default encoder
factory which honours ``VOYAGER_GROUNDEDNESS_VLLM_ENDPOINT`` for the
production vLLM-factory deployment and falls back to a local pylate model
otherwise.

Operators select a Pareto-optimal default profile via either the
``LATENCE_TRACE_PROFILE`` environment variable or the ``--profile`` CLI
flag. Supported values are ``fast``, ``balanced`` (default) and
``quality``; see ``research/triangular_maxsim/reports/profile_pareto.md``
for the sweep that backs each knee. Any environment variable already
exported by the operator wins over the profile preset, so production
tuning decisions remain authoritative.
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

logger = logging.getLogger(__name__)

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
        title="latence-trace Groundedness Tracker (Beta)",
        version="0.1.0",
        description=(
            "Calibrated, auditable groundedness scoring for RAG and evidence-bearing "
            "LLM outputs. Part of the latence.ai product family."
        ),
    )
    app.include_router(create_router(_get_service))
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
