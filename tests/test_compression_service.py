from __future__ import annotations

import pytest

from latence_trace.api.compression_models import CompressionRequest
from latence_trace.api.compression_service import CompressionService


@pytest.mark.asyncio
async def test_compression_fallback_preserves_exact_terms() -> None:
    service = CompressionService()
    request = CompressionRequest(
        text=(
            "The agent tried branch A and failed pytest. "
            "The active fix is in src/cache.py with benchmark value 120 ms. "
            "A duplicated note says the weather is sunny."
        ),
        target_token_ratio=0.35,
        preserve_exact=["src/cache.py", "120 ms"],
    )

    response = await service.compress(request)

    assert response.provider == "fallback"
    assert "src/cache.py" in response.compressed_text
    assert "120 ms" in response.compressed_text
    assert response.compressed_tokens <= response.original_tokens
    assert response.preserved_terms == ["120 ms", "src/cache.py"]
