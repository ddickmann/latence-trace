"""SDK smoke tests using httpx MockTransport."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from latence_trace_client import (
    AsyncLatenceTraceClient,
    AttributionMode,
    LatenceTraceAuthError,
    LatenceTraceClient,
    LatenceTraceRateLimited,
    LatenceTraceValidationError,
    SupportUnit,
)
from latence_trace_client._transport import RetryPolicy


SAMPLE_RESPONSE = {
    "risk_band": "green",
    "risk_reason": None,
    "scores": {
        "groundedness_v2": 0.91,
        "coverage_score_u": 0.74,
        "context_coverage_ratio": 0.5,
    },
    "response_tokens": [
        {"token": "Newton", "char_start": 0, "char_end": 6, "g_t": 0.93},
    ],
    "nli": [],
    "support_units": [],
}


def _mock_transport(handler):
    return httpx.MockTransport(handler)


def test_client_returns_typed_response_with_request_id() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/groundedness"
        body = json.loads(request.content)
        assert body["response_text"] == "Newton was born in 1643."
        assert body["raw_context"] == ["Newton was born in 1643."]
        return httpx.Response(
            200,
            json=SAMPLE_RESPONSE,
            headers={"x-request-id": "req-123"},
        )

    with LatenceTraceClient(transport=_mock_transport(handler)) as client:
        result = client.score_groundedness(
            response_text="Newton was born in 1643.",
            raw_context=["Newton was born in 1643."],
        )
    assert result.risk_band.value == "green"
    assert result.scores.groundedness_v2 == 0.91
    assert result.request_id == "req-123"


def test_client_retries_on_503_then_succeeds() -> None:
    counter = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        counter["n"] += 1
        if counter["n"] < 3:
            return httpx.Response(503, json={"detail": {"code": "warming"}})
        return httpx.Response(200, json=SAMPLE_RESPONSE)

    with LatenceTraceClient(
        transport=_mock_transport(handler),
        retry_policy=RetryPolicy(max_retries=4, base_seconds=0.0, cap_seconds=0.01),
    ) as client:
        client.score_groundedness(response_text="x", raw_context=["y"])
    assert counter["n"] == 3


def test_client_429_with_retry_after_then_raises_after_max() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            json={
                "detail": {
                    "code": "rate_limited",
                    "message": "slow down",
                    "hint": "back off",
                }
            },
            headers={"Retry-After": "0"},
        )

    with LatenceTraceClient(
        transport=_mock_transport(handler),
        retry_policy=RetryPolicy(max_retries=1, base_seconds=0.0, cap_seconds=0.01),
    ) as client:
        with pytest.raises(LatenceTraceRateLimited) as excinfo:
            client.score_groundedness(response_text="x", raw_context=["y"])
    assert excinfo.value.code == "rate_limited"
    assert excinfo.value.retry_after == 0.0


def test_client_402_raises_auth_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            402,
            json={
                "detail": {
                    "code": "license_missing",
                    "message": "no license configured",
                }
            },
        )

    with LatenceTraceClient(transport=_mock_transport(handler)) as client:
        with pytest.raises(LatenceTraceAuthError) as excinfo:
            client.score_groundedness(response_text="x", raw_context=["y"])
    assert excinfo.value.code == "license_missing"


def test_client_authorization_header_set_when_api_key_provided() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("Authorization")
        return httpx.Response(200, json=SAMPLE_RESPONSE)

    with LatenceTraceClient(
        api_key="lt_abc",
        transport=_mock_transport(handler),
    ) as client:
        client.score_groundedness(response_text="x", raw_context=["y"])
    assert seen["auth"] == "Bearer lt_abc"


def test_client_supports_support_units_with_attribution() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["support_units"][0]["source_id"] == "doc-1"
        assert body["attribution_mode"] == "closed_book"
        return httpx.Response(200, json=SAMPLE_RESPONSE)

    with LatenceTraceClient(transport=_mock_transport(handler)) as client:
        client.score_groundedness(
            response_text="x",
            support_units=[SupportUnit(text="hello", source_id="doc-1", speaker="A")],
            attribution_mode=AttributionMode.CLOSED_BOOK,
        )


def test_client_validation_error_is_caught_locally() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("server should never be called for client-side validation")

    with LatenceTraceClient(transport=_mock_transport(handler)) as client:
        with pytest.raises(LatenceTraceValidationError):
            client.score_groundedness(response_text=None, raw_context=["x"])  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_async_client_round_trip() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=SAMPLE_RESPONSE, headers={"x-request-id": "rid"})

    transport = httpx.MockTransport(handler)
    async with AsyncLatenceTraceClient(transport=transport) as client:
        result = await client.score_groundedness(response_text="x", raw_context=["y"])
    assert result.risk_band.value == "green"
    assert result.request_id == "rid"


@pytest.mark.asyncio
async def test_async_client_retries_on_500() -> None:
    counter = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        counter["n"] += 1
        if counter["n"] < 2:
            return httpx.Response(500, json={"detail": {"code": "boom"}})
        return httpx.Response(200, json=SAMPLE_RESPONSE)

    async with AsyncLatenceTraceClient(
        transport=httpx.MockTransport(handler),
        retry_policy=RetryPolicy(max_retries=3, base_seconds=0.0, cap_seconds=0.01),
    ) as client:
        await client.score_groundedness(response_text="x", raw_context=["y"])
    assert counter["n"] == 2
