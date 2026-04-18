"""PA2 tests: Triton-kernel warmup singletons + readiness gate.

Validates that:
- ``warm_all`` runs to completion on CPU (where the reference path is
  exercised) without raising and reports ``ok=True``.
- The cached result is reused on subsequent calls so two concurrent
  workers do not double-warm.
- The ``is_warm`` flag flips to ``True`` after warmup completes.
- The ``/healthz`` endpoint always returns 200, while ``/readyz`` returns
  503 before warmup and 200 after.
- The ``LATENCE_TRACE_DISABLE_WARMUP`` env opt-out skips the heavy work
  but still flips the readiness gate so traffic is not blocked.
"""

from __future__ import annotations

import importlib

from fastapi import FastAPI
from fastapi.testclient import TestClient

from latence_trace.api.routes import create_router
from latence_trace.api.service import GroundednessService
from latence_trace.kernels import warmup as warmup_mod


class _StubProvider:
    """Encoder factory stub - the warmup never reaches here."""

    model_name = "stub"


def _service_factory():
    return GroundednessService(encoder_factory=lambda _name: _StubProvider())


def _build_app() -> FastAPI:
    app = FastAPI()
    app.include_router(create_router(_service_factory))
    return app


def test_warm_all_returns_ok_on_cpu_when_cuda_unavailable() -> None:
    warmup_mod.reset_warmup_state_for_tests()
    result = warmup_mod.warm_all("balanced")
    assert result.ok is True
    assert result.profile == "balanced"
    assert result.shapes
    assert warmup_mod.is_warm() is True


def test_warm_all_caches_result() -> None:
    warmup_mod.reset_warmup_state_for_tests()
    first = warmup_mod.warm_all("fast")
    assert first.ok is True
    second = warmup_mod.warm_all("fast")
    assert second is first


def test_warm_all_force_re_runs() -> None:
    warmup_mod.reset_warmup_state_for_tests()
    first = warmup_mod.warm_all("balanced")
    second = warmup_mod.warm_all("balanced", force=True)
    assert second is not first
    assert second.ok is True


def test_warm_all_unknown_profile_falls_back_to_balanced_grid() -> None:
    warmup_mod.reset_warmup_state_for_tests()
    result = warmup_mod.warm_all("custom-tenant-profile")
    assert result.ok is True
    # The unknown profile is recorded under its own key so
    # /readyz can surface it for diagnostics.
    assert "custom-tenant-profile" in warmup_mod.warmup_state()


def test_healthz_always_returns_200() -> None:
    warmup_mod.reset_warmup_state_for_tests()
    client = TestClient(_build_app())
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readyz_blocks_until_warmup_completes() -> None:
    warmup_mod.reset_warmup_state_for_tests()
    client = TestClient(_build_app())
    pre = client.get("/readyz")
    assert pre.status_code == 503
    assert pre.json()["status"] == "warming"

    warmup_mod.warm_all("balanced")
    post = client.get("/readyz")
    assert post.status_code == 200
    body = post.json()
    assert body["status"] == "ready"
    assert "balanced" in body["warmup"]
    assert body["warmup"]["balanced"]["ok"] is True


def test_disable_warmup_env_short_circuits_to_ready(monkeypatch) -> None:
    warmup_mod.reset_warmup_state_for_tests()
    monkeypatch.setenv("LATENCE_TRACE_DISABLE_WARMUP", "1")
    server_main = importlib.import_module("server.main")
    server_main._kick_off_warmup("balanced")
    # Opt-out path flips the gate immediately so traffic is not blocked.
    assert warmup_mod.is_warm() is True


def test_legacy_health_endpoint_still_works() -> None:
    warmup_mod.reset_warmup_state_for_tests()
    client = TestClient(_build_app())
    response = client.get("/health")
    assert response.status_code == 200
