from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from latence_trace.api.models import (
    AttributionMode,
    CollectionKind,
    GroundednessEligibility,
    GroundednessResponse,
    GroundednessScores,
    ScoringMode,
    TraceRuntimeProfile,
)

RUNPOD_DIR = Path(__file__).resolve().parents[1] / "runpod"


def _load_api_server():
    if str(RUNPOD_DIR) not in sys.path:
        sys.path.insert(0, str(RUNPOD_DIR))
    spec = importlib.util.spec_from_file_location(
        "latence_trace_runpod_api_server",
        RUNPOD_DIR / "api_server.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _GroundednessService:
    def groundedness(self, request):
        return _groundedness_response(session_id=request.session_id)

    async def groundedness_async(self, request):
        return self.groundedness(request)

    def rollup(self, request):
        return {"turns": len(request.turns), "session_id": request.session_id}


def _groundedness_response(session_id: str | None = None) -> GroundednessResponse:
    return GroundednessResponse(
        collection="latence-trace",
        mode="raw_context",
        model="test-model",
        scores=GroundednessScores(
            primary_name="reverse_context",
            primary_score=0.91,
            reverse_context=0.91,
            risk_band="low",
        ),
        response_tokens=[],
        support_units=[],
        top_evidence=[],
        eligibility=GroundednessEligibility(
            collection_kind=CollectionKind.LATE_INTERACTION,
            vector_source="encoded_raw_context",
            storage_compression=None,
            quantization_mode=None,
            dequantized=True,
            user_facing_supported=True,
            warnings=[],
        ),
        time_ms=3.0,
        scoring_mode=ScoringMode.RAG,
        profile=None,
        effective_profile=TraceRuntimeProfile.STANDARD,
        session_id=session_id,
        attribution_mode=AttributionMode.CLOSED_BOOK,
    )


def test_generic_api_score_and_runpod_compat_parity(monkeypatch) -> None:
    api_server = _load_api_server()
    response = _groundedness_response(session_id="sess-1")

    async def fake_handler(_job):
        return {
            "success": True,
            "action": "score",
            "result": response.model_dump(mode="json"),
            "version": "test",
        }

    monkeypatch.setattr(api_server.runpod_handler, "initialize", lambda: None)
    monkeypatch.setattr(api_server.runpod_handler, "shutdown", lambda: None)
    monkeypatch.setattr(api_server.runpod_handler, "_service", _GroundednessService())
    monkeypatch.setattr(api_server.runpod_handler, "handler", fake_handler)

    app = api_server.create_app()
    with TestClient(app) as client:
        payload = {
            "query_text": "Where was Heinrich born?",
            "raw_context": "Heinrich was born in Augsburg.",
            "response_text": "Heinrich was born in Augsburg.",
            "session_id": "sess-1",
        }
        direct = client.post("/groundedness", json=payload).json()
        compat = client.post("/runsync", json={"input": payload}).json()

    assert direct["scores"]["primary_score"] == compat["result"]["scores"]["primary_score"]
    assert direct["session_id"] == compat["result"]["session_id"]
