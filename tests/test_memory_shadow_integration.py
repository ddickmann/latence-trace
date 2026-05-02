from __future__ import annotations

from latence_trace.api.compression_models import CompressionResponse
from latence_trace.api.models import (
    AttributionMode,
    CollectionKind,
    GroundednessEligibility,
    GroundednessRequest,
    GroundednessResponse,
    GroundednessScores,
    ScoringMode,
    TraceRuntimeProfile,
)
from latence_trace.api.service import GroundednessService
from latence_trace.memory.models import MemoryPolicy


def test_memory_shadow_attaches_code_lane_preview_without_scoring_rewrite() -> None:
    service = GroundednessService.__new__(GroundednessService)
    request = GroundednessRequest(
        scoring_mode=ScoringMode.CODE,
        enable_memory_shadow=True,
        memory_policy=MemoryPolicy(hot_token_budget=80),
        query_text="Fix the flaky test in tests/test_cache.py",
        raw_context="tests/test_cache.py failed with AssertionError on cache expiry.",
        response_text="Fixed tests/test_cache.py by preserving cache expiry semantics.",
    )
    response = GroundednessResponse(
        collection="latence-trace",
        mode="raw_context",
        model="test",
        scores=GroundednessScores(
            primary_name="reverse_context",
            primary_score=0.9,
            reverse_context=0.9,
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
        time_ms=1.0,
        scoring_mode=ScoringMode.CODE,
        effective_profile=TraceRuntimeProfile.STANDARD,
        attribution_mode=AttributionMode.CLOSED_BOOK,
        runtime_head_features={"file_alignment": 0.95, "symbol_alignment": 0.8},
    )

    service._maybe_attach_memory_shadow(request, response)

    assert response.next_memory_state is not None
    assert response.hot_context_preview is not None
    assert "tests/test_cache.py" in response.hot_context_preview
    assert response.memory_diagnostics is not None


class _FakeCompressionService:
    async def compress(self, request):
        assert "tests/test_cache.py" in request.force_tokens
        return CompressionResponse(
            compressed_text="compressed context keeps tests/test_cache.py and AssertionError",
            original_tokens=900,
            compressed_tokens=9,
            compression_ratio=0.01,
            preserved_terms=["tests/test_cache.py"],
            provider="fallback",
        )


def test_memory_shadow_compresses_large_context_before_memory_update() -> None:
    service = GroundednessService.__new__(GroundednessService)
    service._compression_service = _FakeCompressionService()
    request = GroundednessRequest(
        scoring_mode=ScoringMode.CODE,
        enable_memory_shadow=True,
        memory_policy=MemoryPolicy(hot_token_budget=120),
        query_text="Fix tests/test_cache.py",
        raw_context="tests/test_cache.py AssertionError " + "irrelevant generated log " * 700,
        response_text="Fixed tests/test_cache.py.",
    )
    response = GroundednessResponse(
        collection="latence-trace",
        mode="raw_context",
        model="test",
        scores=GroundednessScores(
            primary_name="reverse_context",
            primary_score=0.9,
            reverse_context=0.9,
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
        time_ms=1.0,
        scoring_mode=ScoringMode.CODE,
        effective_profile=TraceRuntimeProfile.STANDARD,
        attribution_mode=AttributionMode.CLOSED_BOOK,
    )

    service._maybe_attach_memory_shadow(request, response)

    assert response.hot_context_preview is not None
    assert "compressed context" in response.hot_context_preview
    assert "tests/test_cache.py" in response.hot_context_preview
    assert "irrelevant generated log" not in response.hot_context_preview
