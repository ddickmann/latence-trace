"""Standalone SuperPod-compatible LLMLingua2 compression service."""

from __future__ import annotations

import math
import os
import re
import uuid
from typing import Any

from latence_trace.api.compression_models import (
    CompressionRequest,
    CompressionResponse,
    CompressionSpan,
)
from latence_trace.providers.compression import VllmCompressionProvider

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def _token_count(text: str) -> int:
    return len(text.split())


def _message_text(messages: list[dict[str, Any]] | None) -> str:
    parts: list[str] = []
    for message in messages or []:
        role = str(message.get("role") or "").strip()
        content = str(message.get("content") or "").strip()
        if not content:
            continue
        parts.append(f"{role}: {content}" if role else content)
    return "\n".join(parts)


class CompressionProviderError(RuntimeError):
    """Raised when the managed LLMLingua2 provider is required but fails."""


class CompressionService:
    """Compress text with the SuperPod LLMLingua2 pipeline when configured.

    The vLLM path intentionally mirrors the SuperPod compression service:
    chunk text, obtain token-classification keep probabilities from vLLM,
    run Rust ``text_processing`` postprocessing with force-token and digit
    preservation, then reassemble.  The sentence fallback only exists to keep
    local/dev deployments available when vLLM or the Rust wheel is absent.
    """

    def __init__(
        self,
        provider: VllmCompressionProvider | None = None,
        *,
        allow_provider_fallback: bool = False,
    ) -> None:
        self.provider = provider
        self.allow_provider_fallback = bool(allow_provider_fallback)
        self._postprocessor: Any | None = None

    @classmethod
    def from_env(cls) -> CompressionService:
        endpoint = os.environ.get("LATENCE_TRACE_COMPRESSION_ENDPOINT", "").strip()
        if not endpoint:
            return cls()
        allow_provider_fallback = os.environ.get(
            "LATENCE_TRACE_COMPRESSION_PROVIDER_FALLBACK",
            "0",
        ).lower() in {"1", "true", "yes", "on"}
        return cls(
            VllmCompressionProvider(
                endpoint=endpoint,
                model=os.environ.get("LATENCE_TRACE_COMPRESSION_MODEL"),
                timeout=float(os.environ.get("LATENCE_TRACE_COMPRESSION_REQUEST_TIMEOUT_S", "30")),
                max_concurrency=int(os.environ.get("LATENCE_TRACE_COMPRESSION_CONCURRENCY", "16")),
            ),
            allow_provider_fallback=allow_provider_fallback,
        )

    async def compress(self, request: CompressionRequest) -> CompressionResponse:
        text = (request.text or _message_text(request.messages)).strip()
        if self.provider is not None:
            try:
                if request.action == "compress_messages" and request.messages:
                    return await self._compress_messages_with_superpod(request)
                return await self._compress_text_with_superpod(text, request)
            except Exception as exc:  # noqa: BLE001 - report provider failures unless explicitly relaxed
                if not self.allow_provider_fallback:
                    raise CompressionProviderError(
                        "Managed LLMLingua2 compression provider failed"
                    ) from exc
                fallback = self._fallback(text, request)
                fallback.diagnostics["provider_error"] = str(exc)
                return fallback
        return self._fallback(text, request)

    async def _compress_text_with_superpod(
        self,
        text: str,
        request: CompressionRequest,
    ) -> CompressionResponse:
        if self.provider is None:
            return self._fallback(text, request)
        text_processing = self._text_processing()
        processed_text, toon_applied = self._maybe_apply_toon(text, request)
        chunks = self._chunk_text(processed_text, request.chunk_size, text_processing)
        chunk_logprobs = []
        fallback_chunks = 0
        for chunk in chunks:
            probs = await self.provider.keep_probabilities(chunk)
            if probs is None:
                fallback_chunks += 1
                probs = [1.0] * max(1, _token_count(chunk))
            chunk_logprobs.append(probs)

        config = text_processing.PostprocessorConfig(
            target_rate=1.0 - request.effective_compression_rate,
            force_preserve_digit=request.force_preserve_digit,
            force_tokens=request.effective_force_tokens,
            fallback_mode=request.fallback_mode,
        )
        postprocessor = self._ensure_postprocessor(text_processing)
        compressed_results, compressed_data = postprocessor.process_batch_with_tokenization(
            chunks,
            chunk_logprobs,
            config,
        )
        compressed = " ".join(
            chunk_data[0]
            for chunk_data in compressed_data
            if isinstance(chunk_data, tuple) and chunk_data and chunk_data[0]
        )
        compressed, restored_force_tokens = _restore_missing_force_tokens(
            original=text,
            compressed=compressed,
            force_tokens=request.effective_force_tokens,
        )
        original_tokens = sum(len(probs) for probs in chunk_logprobs)
        compressed_tokens = max(
            sum(result.compressed_tokens for result in compressed_results),
            _token_count(compressed),
        )
        response = self._response_from_counts(
            original=text,
            compressed=compressed,
            original_tokens=max(1, original_tokens),
            compressed_tokens=compressed_tokens,
            provider="superpod_vllm",
        )
        response.preserved_terms = sorted(
            token for token in request.effective_force_tokens if token and token in compressed
        )
        response.diagnostics.update(
            {
                "request_id": f"comp_{uuid.uuid4().hex[:12]}",
                "chunks_processed": len(chunks),
                "fallback_chunks": fallback_chunks,
                "toon_applied": toon_applied,
                "restored_force_tokens": restored_force_tokens,
                "config_used": {
                    "compression_rate": request.effective_compression_rate,
                    "chunk_size": request.chunk_size,
                    "force_preserve_digit": request.force_preserve_digit,
                    "force_tokens": request.effective_force_tokens,
                    "fallback_mode": request.fallback_mode,
                },
            }
        )
        return response

    async def _compress_messages_with_superpod(
        self,
        request: CompressionRequest,
    ) -> CompressionResponse:
        messages = request.messages or []
        compressed_messages: list[dict[str, Any]] = []
        stats: list[dict[str, Any]] = []
        rates = _cosine_annealing_rates(
            len(messages),
            target_compression=request.target_compression,
            max_compression=request.max_compression,
        )
        original_total = 0
        compressed_total = 0
        for idx, message in enumerate(messages):
            content = str(message.get("content") or "")
            role = str(message.get("role") or "user")
            original_total += _token_count(content)
            if role == "system" or not content.strip():
                compressed_messages.append(dict(message))
                compressed_total += _token_count(content)
                stats.append({"index": idx, "role": role, "compression_rate": 0.0})
                continue
            nested = request.model_copy(
                update={
                    "action": "compress",
                    "text": content,
                    "messages": None,
                    "compression_rate": rates[idx],
                }
            )
            response = await self._compress_text_with_superpod(content, nested)
            compressed_messages.append({"role": role, "content": response.compressed_text})
            compressed_total += response.compressed_tokens
            stats.append(
                {
                    "index": idx,
                    "role": role,
                    "compression_rate": rates[idx],
                    "original_tokens": response.original_tokens,
                    "compressed_tokens": response.compressed_tokens,
                }
            )
        compressed_text = _message_text(compressed_messages)
        response = self._response_from_counts(
            original=_message_text(messages),
            compressed=compressed_text,
            original_tokens=max(1, original_total),
            compressed_tokens=compressed_total,
            provider="superpod_vllm",
        )
        response.compressed_messages = compressed_messages
        response.preserved_terms = sorted(
            token for token in request.effective_force_tokens if token and token in compressed_text
        )
        response.diagnostics.update(
            {
                "message_statistics": stats,
                "target_compression": request.target_compression,
                "max_compression": request.max_compression,
            }
        )
        return response

    def _fallback(self, text: str, request: CompressionRequest) -> CompressionResponse:
        sentences = [item.strip() for item in _SENTENCE_SPLIT_RE.split(text) if item.strip()]
        if not sentences:
            sentences = [text]
        target_tokens = max(1, int(round(_token_count(text) * request.target_token_ratio)))
        selected: list[str] = []
        total = 0
        preserved = {term for term in request.preserve_exact if term and term in text}
        for sentence in sentences:
            sentence_tokens = _token_count(sentence)
            must_keep = any(term in sentence for term in preserved)
            if selected and total + sentence_tokens > target_tokens and not must_keep:
                continue
            selected.append(sentence)
            total += sentence_tokens
            if total >= target_tokens and not preserved:
                break
        if not selected:
            selected = [sentences[0]]
        compressed = " ".join(selected)
        response = self._response_from_text(text, compressed, provider="fallback")
        response.preserved_terms = sorted(preserved)
        response.diagnostics["fallback_mode"] = "sentence_preserve_exact"
        return response

    def _response_from_text(
        self,
        original: str,
        compressed: str,
        *,
        provider: str,
    ) -> CompressionResponse:
        original_tokens = _token_count(original)
        compressed_tokens = _token_count(compressed)
        return self._response_from_counts(
            original=original,
            compressed=compressed,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            provider=provider,
        )

    def _response_from_counts(
        self,
        original: str,
        compressed: str,
        *,
        original_tokens: int,
        compressed_tokens: int,
        provider: str,
    ) -> CompressionResponse:
        ratio = compressed_tokens / original_tokens if original_tokens else 1.0
        start = original.find(compressed)
        spans: list[CompressionSpan] = []
        if start >= 0 and compressed:
            spans.append(
                CompressionSpan(
                    start=start,
                    end=start + len(compressed),
                    text=compressed,
                    keep_score=1.0,
                )
            )
        return CompressionResponse(
            compressed_text=compressed,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=ratio,
            compression_percentage=round((1.0 - ratio) * 100, 2),
            tokens_saved=original_tokens - compressed_tokens,
            spans=spans,
            provider=provider,  # type: ignore[arg-type]
        )

    def _text_processing(self) -> Any:
        import text_processing

        return text_processing

    def _ensure_postprocessor(self, text_processing: Any) -> Any:
        if self._postprocessor is not None:
            return self._postprocessor
        if self.provider is None:
            raise RuntimeError("Compression provider is not configured")
        tokenizer_path = _resolve_tokenizer_path(self.provider.model or "")
        self._postprocessor = text_processing.FastPostprocessor.new_with_tokenizer(tokenizer_path)
        return self._postprocessor

    def _chunk_text(self, text: str, chunk_size: int, text_processing: Any) -> list[str]:
        safe_chunk_size = min(max(512, int(chunk_size)), 7500)
        preprocessor = text_processing.FastPreprocessor(safe_chunk_size)
        raw_chunks = preprocessor.process_single_text(text)
        chunks = [text[chunk.start : chunk.end] for chunk in raw_chunks]
        return [chunk for chunk in chunks if chunk.strip()] or [text]

    def _maybe_apply_toon(self, text: str, request: CompressionRequest) -> tuple[str, bool]:
        if not (request.apply_toon or request.toon_encoding):
            return text, False
        try:
            import json

            parsed = json.loads(text)
        except Exception:
            return text, False
        return _toon_encode(parsed), True


def _restore_missing_force_tokens(
    *,
    original: str,
    compressed: str,
    force_tokens: list[str],
) -> tuple[str, list[str]]:
    """Preserve exact-critical force tokens even if the model/postprocessor drops them."""

    restored: list[str] = []
    output = compressed.strip()
    original_sentences = [item.strip() for item in _SENTENCE_SPLIT_RE.split(original) if item.strip()]
    for token in force_tokens:
        if not token or token not in original or token in output:
            continue
        carrier = next((sentence for sentence in original_sentences if token in sentence), token)
        if carrier not in output:
            output = f"{output} {carrier}".strip()
        restored.append(token)
    return output, restored


def _resolve_tokenizer_path(model_name: str) -> str:
    if model_name:
        if os.path.isfile(model_name):
            return model_name
        candidate = os.path.join(model_name, "tokenizer.json")
        if os.path.isfile(candidate):
            return candidate
    try:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(repo_id=model_name, filename="tokenizer.json")
    except Exception:
        pass
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        candidate = os.path.join(str(tokenizer.name_or_path), "tokenizer.json")
        if os.path.isfile(candidate):
            return candidate
    except Exception:
        pass
    for fallback in (
        "/workspace/latence-trace/runpod/compression_model/tokenizer.json",
        "/app/model/tokenizer.json",
    ):
        if os.path.isfile(fallback):
            return fallback
    raise RuntimeError(f"Tokenizer not found for compression model '{model_name}'")


def _toon_encode(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int | float):
        return str(value)
    if isinstance(value, str):
        return _toon_string(value)
    if isinstance(value, list):
        return "[" + ",".join(_toon_encode(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{" + ",".join(f"{_toon_string(str(key))}:{_toon_encode(item)}" for key, item in value.items()) + "}"
    return _toon_string(str(value))


def _toon_string(text: str) -> str:
    if not text:
        return '""'
    needs_quotes = (
        text in {"null", "true", "false"}
        or text[0].isdigit()
        or any(char in text for char in ['{', '}', '[', ']', ':', ',', '"', ' ', '\n', '\t', '\r'])
    )
    if not needs_quotes:
        return text
    escaped = (
        text.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\t", "\\t")
        .replace("\r", "\\r")
    )
    return f'"{escaped}"'


def _cosine_annealing_rates(
    num_messages: int,
    *,
    target_compression: float,
    max_compression: float,
) -> list[float]:
    if num_messages <= 0:
        return []
    if num_messages == 1:
        return [target_compression]
    rates = []
    for idx in range(num_messages):
        position = idx / (num_messages - 1)
        cosine_factor = (1 + math.cos(position * math.pi)) / 2
        rates.append(target_compression + (max_compression - target_compression) * cosine_factor)
    return rates
