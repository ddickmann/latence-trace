"""Token-bucket rate limit middleware tests."""

from __future__ import annotations

import time

from fastapi import FastAPI
from fastapi.testclient import TestClient

from latence_trace.middleware import RateLimitMiddleware, TokenBucket


def test_token_bucket_allows_burst_then_throttles() -> None:
    bucket = TokenBucket.new(capacity=3, rate_per_sec=1.0)
    assert bucket.take() == (True, 0.0)
    assert bucket.take() == (True, 0.0)
    assert bucket.take() == (True, 0.0)
    allowed, retry = bucket.take()
    assert not allowed
    # We owe one full token at 1 RPS, so retry_after ~= 1 second.
    assert 0.5 < retry <= 1.0


def test_token_bucket_refills_with_time() -> None:
    bucket = TokenBucket.new(capacity=2, rate_per_sec=10.0)
    bucket.take()
    bucket.take()
    assert bucket.take()[0] is False
    time.sleep(0.2)  # ~2 tokens at 10/s
    assert bucket.take()[0] is True


def _build_app(rate_per_sec: float, burst: float | None = None) -> TestClient:
    app = FastAPI()
    app.add_middleware(
        RateLimitMiddleware,
        rate_per_sec=rate_per_sec,
        burst=burst,
        exempt_paths=("/healthz",),
    )

    @app.get("/healthz")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/groundedness")
    def score() -> dict:
        return {"ok": True}

    return TestClient(app)


def test_rate_limit_returns_429_with_retry_after() -> None:
    client = _build_app(rate_per_sec=2.0, burst=2.0)
    assert client.get("/groundedness").status_code == 200
    assert client.get("/groundedness").status_code == 200
    response = client.get("/groundedness")
    assert response.status_code == 429
    assert response.headers["Retry-After"]
    body = response.json()
    assert body["detail"]["code"] == "rate_limited"


def test_rate_limit_exempts_health_probe() -> None:
    client = _build_app(rate_per_sec=1.0, burst=1.0)
    for _ in range(5):
        assert client.get("/healthz").status_code == 200


def test_rate_limit_disabled_when_no_rate_configured() -> None:
    client = _build_app(rate_per_sec=None)  # type: ignore[arg-type]
    for _ in range(5):
        assert client.get("/groundedness").status_code == 200
