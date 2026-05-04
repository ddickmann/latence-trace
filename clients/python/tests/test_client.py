"""SDK smoke tests using httpx MockTransport."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from latence_trace_client import (
    AsyncLatenceTraceClient,
    AttributionMode,
    ComplianceRedactionMode,
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

SAMPLE_COMPLIANCE_RESPONSE = {
    "success": True,
    "original_text": None,
    "entities": [
        {
            "start": 8,
            "end": 24,
            "text": "jane@example.com",
            "label": "email",
            "score": 0.99,
            "source": "model",
            "redacted_value": "[EMAIL]",
            "redaction_mode": "mask",
        }
    ],
    "entity_count": 1,
    "unique_labels": ["email"],
    "redacted_text": "Contact [EMAIL]",
    "chunks_processed": 1,
    "labels_used": ["email"],
    "label_mode": "category",
    "selected_categories": [],
    "processing_time_ms": 12.3,
    "timings_ms": {"vllm_request_ms": 10.1},
    "usage": {
        "chunks_processed": 1,
        "labels_used": 1,
        "mode": "category",
        "categories": [],
    },
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


def test_client_surfaces_context_trust_toggle() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["context_trust_enabled"] is False
        return httpx.Response(200, json=SAMPLE_RESPONSE)

    with LatenceTraceClient(transport=_mock_transport(handler)) as client:
        client.score_groundedness(
            response_text="Newton was born in 1643.",
            raw_context=["Newton was born in 1643."],
            context_trust_enabled=False,
        )


def test_client_retries_on_503_then_succeeds() -> None:
    counter = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
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
    def handler(_request: httpx.Request) -> httpx.Response:
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
    ) as client, pytest.raises(LatenceTraceRateLimited) as excinfo:
        client.score_groundedness(response_text="x", raw_context=["y"])
    assert excinfo.value.code == "rate_limited"
    assert excinfo.value.retry_after == 0.0


def test_client_402_raises_auth_error() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            402,
            json={
                "detail": {
                    "code": "license_missing",
                    "message": "no license configured",
                }
            },
        )

    with (
        LatenceTraceClient(transport=_mock_transport(handler)) as client,
        pytest.raises(LatenceTraceAuthError) as excinfo,
    ):
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


def test_client_redact_compliance_round_trip() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/compliance/redact"
        body = json.loads(request.content)
        assert body["text"] == "Contact jane@example.com"
        assert body["labels"] == ["email"]
        assert body["redaction_mode"] == "mask"
        assert body["include_original_text"] is False
        return httpx.Response(200, json=SAMPLE_COMPLIANCE_RESPONSE)

    with LatenceTraceClient(transport=_mock_transport(handler)) as client:
        result = client.redact_compliance(
            text="Contact jane@example.com",
            labels=["email"],
            redaction_mode=ComplianceRedactionMode.MASK,
        )

    assert result.entity_count == 1
    assert result.redacted_text == "Contact [EMAIL]"
    assert result.entities[0].label == "email"


def test_product_namespaces_route_to_canonical_paths() -> None:
    seen: list[tuple[str, dict[str, Any]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append((request.url.path, body))
        if request.url.path == "/groundedness":
            return httpx.Response(200, json=SAMPLE_RESPONSE)
        if request.url.path == "/v1/compression":
            return httpx.Response(
                200,
                json={"compressed_text": "short", "tokens_saved": 3},
            )
        if request.url.path == "/v1/memory/update":
            return httpx.Response(
                200,
                json={"next_memory_state": {"turn_index": 1}, "hot_context": "x"},
            )
        if request.url.path == "/groundedness/rollup":
            return httpx.Response(200, json={"session_id": "s", "turn_count": 1})
        raise AssertionError(f"unexpected path {request.url.path}")

    with LatenceTraceClient(transport=_mock_transport(handler)) as client:
        client.grounding.code(response_text="x", raw_context=["y"])
        client.compression.text("alpha beta gamma", compression_rate=0.4)
        client.compression.messages([{"role": "user", "content": "hello"}])
        client.memory.step(turn_text="remember this")
        client.rollup([{"risk_band": "green"}], session_id="s")

    assert seen[0][0] == "/groundedness"
    assert seen[0][1]["response_text"] == "x"
    assert seen[0][1]["raw_context"] == ["y"]
    assert seen[0][1]["scoring_mode"] == "code"
    assert seen[1][0] == "/v1/compression"
    assert seen[1][1]["compression_rate"] == 0.4
    assert seen[2][0] == "/v1/compression"
    assert seen[2][1]["action"] == "compress_messages"
    assert seen[2][1]["messages"] == [{"role": "user", "content": "hello"}]
    assert seen[3][0] == "/v1/memory/update"
    assert seen[3][1]["prior_memory_state"] is None
    assert seen[4][0] == "/groundedness/rollup"
    assert seen[4][1]["turns"] == [{"risk_band": "green"}]


def test_sdk_managed_session_round_trips_memory_state() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/memory/update"
        body = json.loads(request.content)
        assert body["prior_memory_state"] is None
        return httpx.Response(
            200,
            json={"next_memory_state": {"version": "infinimem.v1", "turn_index": 1}},
        )

    with LatenceTraceClient(transport=_mock_transport(handler)) as client:
        session = client.session(session_id="sess-1")
        result = session.memory_step(turn_text="User wants manual approvals preserved.")

    assert result["next_memory_state"]["turn_index"] == 1
    assert session.memory_state == {"version": "infinimem.v1", "turn_index": 1}


def test_sdk_managed_session_injects_state_into_grounding_calls() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/groundedness"
        body = json.loads(request.content)
        seen.append(body)
        return httpx.Response(200, json=SAMPLE_RESPONSE)

    with LatenceTraceClient(transport=_mock_transport(handler)) as client:
        session = client.session(
            session_id="sess-1",
            memory_state={"version": "infinimem.v1", "turn_index": 4},
        )
        session.rag(response_text="answer", raw_context=["context"])
        session.code(response_text="print('x')", raw_context=["print('x')"])

    assert seen[0]["scoring_mode"] == "rag"
    assert seen[0]["session_id"] == "sess-1"
    assert seen[0]["memory_state"]["turn_index"] == 4
    assert seen[1]["scoring_mode"] == "code"
    assert seen[1]["session_id"] == "sess-1"


def test_client_agent_help_pass_through() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/agent-help"
        return httpx.Response(
            200,
            json={"endpoints": {"score": {"path": "/groundedness"}}},
        )

    with LatenceTraceClient(transport=_mock_transport(handler)) as client:
        result = client.agent_help()

    assert result["endpoints"]["score"]["path"] == "/groundedness"


def test_client_validation_error_is_caught_locally() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("server should never be called for client-side validation")

    with (
        LatenceTraceClient(transport=_mock_transport(handler)) as client,
        pytest.raises(LatenceTraceValidationError),
    ):
        client.score_groundedness(response_text=None, raw_context=["x"])  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_async_client_round_trip() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=SAMPLE_RESPONSE, headers={"x-request-id": "rid"})

    transport = httpx.MockTransport(handler)
    async with AsyncLatenceTraceClient(transport=transport) as client:
        result = await client.score_groundedness(response_text="x", raw_context=["y"])
    assert result.risk_band.value == "green"
    assert result.request_id == "rid"


@pytest.mark.asyncio
async def test_async_client_redact_compliance_round_trip() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/compliance/redact"
        return httpx.Response(200, json=SAMPLE_COMPLIANCE_RESPONSE)

    async with AsyncLatenceTraceClient(transport=httpx.MockTransport(handler)) as client:
        result = await client.redact_compliance(
            text="Contact jane@example.com",
            labels=["email"],
        )

    assert result.entity_count == 1


@pytest.mark.asyncio
async def test_async_product_namespace_round_trip() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if request.url.path == "/groundedness":
            assert body["scoring_mode"] == "rag"
            return httpx.Response(200, json=SAMPLE_RESPONSE)
        if request.url.path == "/v1/memory/update":
            return httpx.Response(200, json={"next_memory_state": {"turn_index": 2}})
        raise AssertionError(f"unexpected path {request.url.path}")

    async with AsyncLatenceTraceClient(transport=httpx.MockTransport(handler)) as client:
        await client.grounding.rag(response_text="x", raw_context=["y"])
        session = client.session(session_id="sess-async")
        await session.memory_step(turn_text="remember this")

    assert session.memory_state == {"turn_index": 2}


@pytest.mark.asyncio
async def test_async_client_retries_on_500() -> None:
    counter = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
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
