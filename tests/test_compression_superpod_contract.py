from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from latence_trace.api.compression_models import CompressionRequest
from latence_trace.api.compression_service import CompressionProviderError, CompressionService


class _FakeProvider:
    def __init__(self, model: str) -> None:
        self.model = model

    async def keep_probabilities(self, text: str) -> list[float]:
        return [0.9 for _ in text.split()]


class _FailingProvider:
    model = "failing-model"

    async def keep_probabilities(self, _text: str) -> list[float]:
        raise RuntimeError("provider down")


class _FakePostprocessorConfig:
    def __init__(
        self,
        *,
        target_rate: float,
        force_preserve_digit: bool,
        force_tokens: list[str],
        fallback_mode: bool,
    ) -> None:
        self.target_rate = target_rate
        self.force_preserve_digit = force_preserve_digit
        self.force_tokens = force_tokens
        self.fallback_mode = fallback_mode


class _FakePreprocessor:
    def __init__(self, _chunk_size: int) -> None:
        pass

    def process_single_text(self, text: str) -> list[SimpleNamespace]:
        return [SimpleNamespace(start=0, end=len(text))]


class _FakePostprocessor:
    @classmethod
    def new_with_tokenizer(cls, _tokenizer_path: str) -> _FakePostprocessor:
        return cls()

    def process_batch_with_tokenization(self, chunks, _chunk_logprobs, config):
        compressed = " ".join([*config.force_tokens, chunks[0].split()[0]])
        return [SimpleNamespace(compressed_tokens=len(compressed.split()))], [
            (compressed, len(chunks[0].split()), len(compressed.split()))
        ]


class _DroppingPostprocessor:
    @classmethod
    def new_with_tokenizer(cls, _tokenizer_path):
        return cls()

    def process_batch_with_tokenization(self, chunks, _chunk_logprobs, _config):
        compressed = " ".join(chunks[0].split()[:3])
        return [SimpleNamespace(compressed_tokens=len(compressed.split()))], [
            (compressed, len(chunks[0].split()), len(compressed.split()))
        ]


@pytest.mark.asyncio
async def test_superpod_contract_uses_postprocessor_force_tokens(monkeypatch, tmp_path) -> None:
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text("{}", encoding="utf-8")
    fake_module = SimpleNamespace(
        FastPreprocessor=_FakePreprocessor,
        FastPostprocessor=_FakePostprocessor,
        PostprocessorConfig=_FakePostprocessorConfig,
    )
    monkeypatch.setitem(sys.modules, "text_processing", fake_module)

    service = CompressionService(provider=_FakeProvider(str(tokenizer)))
    response = await service.compress(
        CompressionRequest(
            text="noise before src/cache.py and order ORD-123 after",
            compression_rate=0.75,
            force_tokens=["src/cache.py", "ORD-123"],
            force_preserve_digit=True,
        )
    )

    assert response.provider == "superpod_vllm"
    assert "src/cache.py" in response.compressed_text
    assert "ORD-123" in response.compressed_text
    assert response.diagnostics["config_used"]["compression_rate"] == 0.75
    assert response.diagnostics["chunks_processed"] == 1


@pytest.mark.asyncio
async def test_superpod_contract_backfills_dropped_force_token_context(
    monkeypatch, tmp_path
) -> None:
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text("{}", encoding="utf-8")
    fake_module = SimpleNamespace(
        FastPreprocessor=_FakePreprocessor,
        FastPostprocessor=_DroppingPostprocessor,
        PostprocessorConfig=_FakePostprocessorConfig,
    )
    monkeypatch.setitem(sys.modules, "text_processing", fake_module)

    service = CompressionService(provider=_FakeProvider(str(tokenizer)))
    response = await service.compress(
        CompressionRequest(
            text=(
                "Short intro. The final safety gate is pytest tests/test_billing_retry.py -q "
                "before redeploying."
            ),
            compression_rate=0.9,
            force_tokens=["tests/test_billing_retry.py"],
            fallback_mode=False,
        )
    )

    assert "tests/test_billing_retry.py" in response.compressed_text
    assert response.preserved_terms == ["tests/test_billing_retry.py"]
    assert response.diagnostics["restored_force_tokens"] == ["tests/test_billing_retry.py"]


@pytest.mark.asyncio
async def test_managed_provider_failure_is_not_silently_downgraded() -> None:
    service = CompressionService(provider=_FailingProvider())

    with pytest.raises(CompressionProviderError):
        await service.compress(CompressionRequest(text="alpha beta gamma"))


@pytest.mark.asyncio
async def test_managed_provider_fallback_requires_explicit_opt_in() -> None:
    service = CompressionService(provider=_FailingProvider(), allow_provider_fallback=True)

    response = await service.compress(
        CompressionRequest(
            text="The active fix is in src/cache.py with benchmark value 120 ms.",
            preserve_exact=["src/cache.py", "120 ms"],
        )
    )

    assert response.provider == "fallback"
    assert "provider_error" in response.diagnostics
    assert "src/cache.py" in response.compressed_text
