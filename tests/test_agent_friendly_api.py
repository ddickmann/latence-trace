"""PA7 agent-friendly API tests.

Covers:
- ``/agent-help`` self-describing contract.
- ``/.well-known/ai-plugin.json`` standard descriptor.
- Structured error envelope on validation failures (code/message/hint/docs_url).
- OpenAPI schema exposes operation_ids that AI agent SDK code generators
  rely on.
"""

from __future__ import annotations

import os
from typing import Iterator

import httpx
import pytest
import torch
from fastapi import FastAPI

from latence_trace.api.routes import create_router
from latence_trace.api.service import GroundednessService


class _DeterministicEncoder:
    @property
    def model_name(self) -> str:
        return "agent-test-encoder"

    @property
    def model_name_or_path(self) -> str:
        return self.model_name

    def encode(self, sentences, *, is_query=False, prompt_name=None, **kwargs):
        outs = []
        for sentence in sentences:
            seed = abs(hash(sentence)) % (2**31 - 1)
            generator = torch.Generator().manual_seed(seed)
            token_count = max(2, min(8, len((sentence or "x").split())))
            outs.append(torch.randn(token_count, 32, generator=generator))
        return outs


def _build_app() -> FastAPI:
    os.environ["VOYAGER_GROUNDEDNESS_NLI_ENABLED"] = "0"
    service = GroundednessService(encoder_factory=lambda _name: _DeterministicEncoder())
    app = FastAPI(title="latence-trace test", version="1.0.0")
    app.include_router(create_router(lambda: service))
    return app


def _client(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


@pytest.fixture(autouse=True)
def _restore_env() -> Iterator[None]:
    saved = os.environ.get("VOYAGER_GROUNDEDNESS_NLI_ENABLED")
    yield
    if saved is None:
        os.environ.pop("VOYAGER_GROUNDEDNESS_NLI_ENABLED", None)
    else:
        os.environ["VOYAGER_GROUNDEDNESS_NLI_ENABLED"] = saved


@pytest.mark.anyio("asyncio")
async def test_agent_help_describes_premise_lanes_and_attribution_modes():
    app = _build_app()
    async with _client(app) as client:
        r = await client.get("/agent-help")
    assert r.status_code == 200
    body = r.json()

    assert body["service"]["name"] == "latence-trace"
    assert body["service"]["version"] == "1.0.0"

    lane_names = {lane["name"] for lane in body["premise_lanes"]}
    assert lane_names == {"chunk_ids", "raw_context", "support_units"}

    attribution_values = {m["value"] for m in body["attribution_modes"]}
    assert attribution_values == {"closed_book", "open_domain"}

    envelope = body["error_envelope"]
    for key in ("code", "message", "hint", "docs_url"):
        assert key in envelope, f"agent-help error envelope missing '{key}'"

    assert "max_inflight" in body["limits"]
    assert body["endpoints"]["score"]["operation_id"] == "score_groundedness"

    # PA7 polish: agent-help must point agents at every other discovery
    # surface so they can crawl one URL and find the rest.
    for key in ("agent_help", "ai_plugin", "openapi", "docs", "healthz", "readyz"):
        assert key in body["endpoints"], (
            f"agent-help endpoints map missing '{key}' so agents have to guess"
        )

    # The profiles block lets agents adjust expectations for latency vs
    # accuracy without scraping the docs.
    profiles = body["profiles"]
    assert profiles["default"] == "balanced"
    assert set(profiles["available"]) == {"fast", "balanced", "quality"}
    assert profiles["active"] in profiles["available"]
    assert profiles["selection_env"] == "LATENCE_TRACE_PROFILE"


@pytest.mark.anyio("asyncio")
async def test_well_known_ai_plugin_descriptor_is_self_consistent():
    app = _build_app()
    async with _client(app) as client:
        r = await client.get("/.well-known/ai-plugin.json")
    assert r.status_code == 200
    body = r.json()

    assert body["schema_version"] == "v1"
    assert body["name_for_model"] == "latence_trace"
    assert body["api"]["type"] == "openapi"
    assert body["api"]["url"] == "/openapi.json"
    assert body["auth"]["type"] == "none"


def _build_full_app() -> FastAPI:
    """Build the app via ``server.main.create_app`` so the global
    RequestValidationError handler is registered."""

    os.environ["VOYAGER_GROUNDEDNESS_NLI_ENABLED"] = "0"
    from server.main import create_app  # noqa: PLC0415

    return create_app()


@pytest.mark.anyio("asyncio")
async def test_validation_error_returns_structured_envelope():
    app = _build_full_app()
    async with _client(app) as client:
        # primary_metric=triangular requires query_text; sending an empty
        # string trips the model validator. The full server registers a
        # global RequestValidationError handler that coerces FastAPI's
        # default 422 list-of-pydantic-errors into our structured
        # envelope so every error class has the same shape.
        r = await client.post(
            "/groundedness",
            json={
                "raw_context": "Heinrich was born in 1851 in Augsburg.",
                "response_text": "Heinrich was born in 1851 in Augsburg.",
                "primary_metric": "triangular",
                "query_text": "",
            },
        )
    assert r.status_code == 422
    detail = r.json().get("detail")
    assert isinstance(detail, dict), (
        f"schema-level 422 must use the structured envelope, got {detail!r}"
    )
    for key in ("code", "message", "hint", "docs_url"):
        assert key in detail, f"validation envelope missing '{key}'"
    assert detail["code"] == "validation_error"
    # The raw pydantic error list is preserved under `errors` for
    # callers that want the full debug trail.
    assert isinstance(detail["errors"], list) and detail["errors"]


@pytest.mark.anyio("asyncio")
async def test_service_error_envelope_includes_code_and_hint():
    """Force a service-level ValidationError and assert the structured
    envelope (code/message/hint/docs_url) is what the client sees."""

    from latence_trace.api.service import ValidationError as SvcValidationError

    class _ExplodingService:
        """Stand-in service that always raises so we can probe the
        structured-error envelope without bringing up the encoder
        stack."""

        def groundedness(self, _request):
            raise SvcValidationError(
                "encoder unavailable in test environment",
            )

    app = FastAPI()
    app.include_router(create_router(lambda: _ExplodingService()))

    async with _client(app) as client:
        r = await client.post(
            "/groundedness",
            json={
                "raw_context": "Heinrich was born in 1851 in Augsburg.",
                "response_text": "Heinrich was born in Augsburg.",
                "primary_metric": "reverse_context",
            },
        )

    assert r.status_code == 400
    detail = r.json().get("detail")
    assert isinstance(detail, dict), f"expected structured envelope, got {detail!r}"
    for key in ("code", "message", "hint", "docs_url"):
        assert key in detail, f"service error envelope missing '{key}'"
    assert detail["code"] == "validation_error"
    assert detail["docs_url"].startswith("https://")


@pytest.mark.anyio("asyncio")
async def test_openapi_exposes_named_operation_ids_for_agent_codegen():
    app = _build_app()
    async with _client(app) as client:
        r = await client.get("/openapi.json")
    assert r.status_code == 200
    spec = r.json()

    operation_ids = set()
    for path, methods in spec.get("paths", {}).items():
        for verb, op in methods.items():
            if isinstance(op, dict) and "operationId" in op:
                operation_ids.add(op["operationId"])

    expected = {
        "score_groundedness",
        "agent_help",
        "ai_plugin_descriptor",
        "healthz",
        "readyz",
    }
    missing = expected - operation_ids
    assert not missing, f"OpenAPI spec missing operation_ids: {missing}"
