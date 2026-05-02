"""Tokenizer-aware text chunking for compliance redaction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TextChunk:
    text: str
    start: int
    end: int
    chunk_index: int
    total_chunks: int
    token_count: int


def _fallback_chunks(text: str, *, max_chars: int) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    start = 0
    chunk_index = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            for sep in (". ", "! ", "? ", "\n\n", "\n", " "):
                split_at = text.rfind(sep, start, end)
                if split_at > start:
                    end = split_at + len(sep)
                    break
        chunks.append(
            TextChunk(
                text=text[start:end],
                start=start,
                end=end,
                chunk_index=chunk_index,
                total_chunks=0,
                token_count=max(1, len(text[start:end].split())),
            )
        )
        chunk_index += 1
        start = end
    total = len(chunks)
    return [
        TextChunk(
            text=chunk.text,
            start=chunk.start,
            end=chunk.end,
            chunk_index=chunk.chunk_index,
            total_chunks=total,
            token_count=chunk.token_count,
        )
        for chunk in chunks
    ]


def chunk_text_by_tokens(
    text: str,
    tokenizer: Any | None,
    *,
    max_text_tokens: int = 512,
) -> list[TextChunk]:
    """Split text into chunks capped by input-text tokens.

    The GLiNER request has a larger total sequence budget (768), but this
    function only budgets the user text portion. Labels and special tokens are
    accounted for by the model-side preprocessor.
    """

    if not text or not text.strip():
        return []
    max_text_tokens = max(1, int(max_text_tokens))
    if tokenizer is None:
        return _fallback_chunks(text, max_chars=max_text_tokens * 4)

    try:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
            padding=False,
            return_tensors=None,
        )
        offsets = list(encoded.get("offset_mapping") or [])
    except Exception:
        return _fallback_chunks(text, max_chars=max_text_tokens * 4)

    usable_offsets = [
        (int(start), int(end))
        for start, end in offsets
        if int(end) > int(start)
    ]
    if not usable_offsets:
        return _fallback_chunks(text, max_chars=max_text_tokens * 4)

    chunks: list[TextChunk] = []
    for token_start in range(0, len(usable_offsets), max_text_tokens):
        token_offsets = usable_offsets[token_start:token_start + max_text_tokens]
        start = token_offsets[0][0]
        end = token_offsets[-1][1]
        chunks.append(
            TextChunk(
                text=text[start:end],
                start=start,
                end=end,
                chunk_index=len(chunks),
                total_chunks=0,
                token_count=len(token_offsets),
            )
        )
    total = len(chunks)
    return [
        TextChunk(
            text=chunk.text,
            start=chunk.start,
            end=chunk.end,
            chunk_index=chunk.chunk_index,
            total_chunks=total,
            token_count=chunk.token_count,
        )
        for chunk in chunks
    ]
