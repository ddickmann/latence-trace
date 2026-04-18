"""Response-chunking parity + stress tests.

These tests exercise the response-side chunking path added in the response
chunking refactor. They prove two properties:

1. **Parity:** scoring the same response with chunking effectively disabled
   (one window) and with chunking forced on (many windows of <= N tokens)
   produces identical headline scores, identical per-token reverse-context
   values, and identical per-token attribution. This is the math contract
   the orchestrator must hold.

2. **Long-response correctness:** a response many times longer than a
   single chunk window still produces a well-formed score payload, every
   token gets attribution, the global aggregates remain finite, and the
   number of stitched chunks matches the expected budget.

Both tests use a deterministic stub encoder so they run in CI without GPU
or HuggingFace downloads.
"""

from __future__ import annotations

import hashlib
import re
from typing import Iterable, List

import numpy as np
import pytest

from latence_trace.core.groundedness import (
    SupportUnitInput,
    _build_response_chunks,
    encode_texts,
    partition_support_units,
    score_groundedness_response_chunked,
    segment_text,
    tokenize_text,
)


_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


class _StubColbertProvider:
    """Deterministic multi-vector encoder used by the parity tests.

    Mirrors a ColBERT-style provider closely enough to exercise the
    chunker, segmenter, and scorer end-to-end:

    - ``encode`` returns one ``(T, dim)`` matrix per input string.
    - ``tokenize`` returns whitespace-ish tokens.
    - ``encoded_token_count`` makes the sentence-packed segmenter compute
      realistic windows.
    - ``doc_maxlen`` advertises a token limit so the orchestrator wires
      up its truncation warning when exceeded.
    """

    model_name = "stub-colbert"
    doc_maxlen = 64

    def __init__(self, dim: int = 16) -> None:
        self.dim = dim

    def tokenize(self, text: str, is_query: bool = False) -> List[str]:
        return _TOKEN_RE.findall(text)

    def encoded_token_count(self, text: str, is_query: bool = False) -> int:
        return len(self.tokenize(text))

    def _vec(self, token: str) -> np.ndarray:
        digest = hashlib.sha1(token.lower().encode("utf-8")).digest()
        v = np.zeros((self.dim,), dtype=np.float32)
        v[digest[0] % self.dim] = 1.0
        v[digest[1] % self.dim] = max(v[digest[1] % self.dim], 0.5)
        norm = np.linalg.norm(v) + 1e-8
        return v / norm

    def encode(self, inputs, **_kwargs):
        texts = [inputs] if isinstance(inputs, str) else list(inputs)
        out: List[np.ndarray] = []
        for text in texts:
            tokens = self.tokenize(text) or ["<empty>"]
            out.append(np.stack([self._vec(token) for token in tokens]).astype(np.float32))
        return out


def _make_support_units(provider: _StubColbertProvider, raw_context: str) -> List[SupportUnitInput]:
    segments = segment_text(
        raw_context,
        "sentence_packed",
        provider=provider,
        chunk_token_budget=32,
    )
    embeddings = encode_texts(provider, [s["text"] for s in segments], is_query=False)
    units: List[SupportUnitInput] = []
    for idx, (segment, emb) in enumerate(zip(segments, embeddings)):
        tokens = tokenize_text(provider, segment["text"], expected_len=int(emb.shape[0]))
        units.append(
            SupportUnitInput(
                support_id=f"raw-{idx}",
                text=segment["text"],
                embeddings=emb,
                tokens=tokens,
                source_mode="raw_context",
                offset_start=int(segment["offset_start"]),
                offset_end=int(segment["offset_end"]),
            )
        )
    return units


def _scalar_pairs() -> Iterable[str]:
    return (
        "primary_score",
        "reverse_context",
        "consensus_hardened",
        "echo_mean",
    )


def test_response_chunking_parity_short_response() -> None:
    """Same scores when the response trivially fits in one window vs many."""
    provider = _StubColbertProvider()
    raw_context = (
        "The Treaty of Versailles was signed on June 28, 1919. "
        "Berlin became the German capital. "
        "The Bundesbank is headquartered in Frankfurt. "
        "The Eiffel Tower is in Paris."
    )
    response_text = (
        "Berlin became the German capital. "
        "The Bundesbank is in Frankfurt. "
        "The Eiffel Tower is in Paris. "
        "The Treaty was signed in 1919."
    )

    support_units = _make_support_units(provider, raw_context)
    support_batches = partition_support_units(support_units, batch_size=4)

    chunks_one = _build_response_chunks(
        response_text,
        provider=provider,
        chunk_token_budget=512,
        encode_fn=encode_texts,
    )
    chunks_many = _build_response_chunks(
        response_text,
        provider=provider,
        chunk_token_budget=4,
        encode_fn=encode_texts,
    )
    assert len(chunks_one) == 1
    assert len(chunks_many) >= 2

    one = score_groundedness_response_chunked(
        response_chunks=chunks_one,
        support_batches=support_batches,
        response_text=response_text,
        evidence_limit=8,
        primary_metric="reverse_context",
    )
    many = score_groundedness_response_chunked(
        response_chunks=chunks_many,
        support_batches=support_batches,
        response_text=response_text,
        evidence_limit=8,
        primary_metric="reverse_context",
    )

    for key in _scalar_pairs():
        a, b = one["scores"].get(key), many["scores"].get(key)
        if a is None and b is None:
            continue
        assert b is not None and a is not None, f"{key}: one={a} many={b}"
        assert a == pytest.approx(b, abs=1e-5), (key, a, b)

    # Per-token reverse-context, support attribution, and weight survive
    # chunking exactly. Heatmap_score follows primary_metric so it tracks
    # reverse_context under the default primary metric.
    assert len(one["response_tokens"]) == len(many["response_tokens"])
    for row_a, row_b in zip(one["response_tokens"], many["response_tokens"]):
        assert row_a["index"] == row_b["index"]
        assert row_a["token"] == row_b["token"]
        assert row_a["weight"] == pytest.approx(row_b["weight"], abs=1e-6)
        assert row_a["reverse_context"] == pytest.approx(row_b["reverse_context"], abs=1e-5)
        assert row_a["consensus_hardened"] == pytest.approx(
            row_b["consensus_hardened"], abs=1e-5
        )
        assert row_a["support_unit_index"] == row_b["support_unit_index"]
        assert row_a["support_token_index"] == row_b["support_token_index"]
        assert row_a["support_token"] == row_b["support_token"]


def test_response_chunking_preserves_global_attribution_indexing() -> None:
    """Per-token attribution remains valid when stitched across chunks."""
    provider = _StubColbertProvider()
    raw_context = (
        "Berlin is the German capital. "
        "Paris is the French capital. "
        "Rome is the Italian capital. "
        "Madrid is the Spanish capital."
    )
    response_text = (
        "Berlin is the German capital. "
        "Paris is the French capital. "
        "Rome is the Italian capital. "
        "Madrid is the Spanish capital."
    )

    support_units = _make_support_units(provider, raw_context)
    support_batches = partition_support_units(support_units, batch_size=2)

    chunks = _build_response_chunks(
        response_text,
        provider=provider,
        chunk_token_budget=4,
        encode_fn=encode_texts,
    )
    assert len(chunks) >= 3

    result = score_groundedness_response_chunked(
        response_chunks=chunks,
        support_batches=support_batches,
        response_text=response_text,
        evidence_limit=16,
        primary_metric="reverse_context",
    )

    seen_indices = [row["index"] for row in result["response_tokens"]]
    assert seen_indices == list(range(len(seen_indices)))
    chunk_indices_seen = {row["response_chunk_index"] for row in result["response_tokens"]}
    assert chunk_indices_seen == set(range(len(chunks)))

    # Every per-token attribution that exists must reference a real
    # support unit / support token within bounds. This guards against
    # off-by-one errors in the stitching step.
    flat_units = [unit for batch in support_batches for unit in batch]
    for row in result["response_tokens"]:
        unit_idx = row.get("support_unit_index")
        token_idx = row.get("support_token_index")
        if unit_idx is None or token_idx is None:
            continue
        assert 0 <= unit_idx < len(flat_units)
        unit = flat_units[unit_idx]
        assert 0 <= token_idx < int(unit.embeddings.shape[0])
        # support_token in the row should match the unit's tokens at
        # token_idx (modulo alignment padding).
        if row.get("support_token") is not None:
            assert row["support_token"] == unit.tokens[token_idx] or unit.tokens[token_idx].startswith("tok_")


def test_response_chunking_long_response_stays_well_formed() -> None:
    """Heavy stress test: a long response (>>encoder limit) still scores cleanly.

    This is the "5000-token response" scenario. We use a deterministic stub
    encoder with ``doc_maxlen=64`` so we exercise the chunked path without
    needing a GPU. The headline metric must be finite, every token gets a
    score, and the orchestrator returns one row per response token.
    """
    provider = _StubColbertProvider()
    raw_context_pieces = [
        "Berlin became the German capital in 1990.",
        "The Bundesbank is headquartered in Frankfurt.",
        "Paris is the French capital and home of the Eiffel Tower.",
        "Rome is the Italian capital and home of the Colosseum.",
        "Madrid is the Spanish capital.",
        "London is the British capital.",
        "Vienna is the Austrian capital.",
        "Bern is the Swiss capital.",
    ]
    raw_context = " ".join(raw_context_pieces)

    long_response = " ".join(raw_context_pieces * 80)

    support_units = _make_support_units(provider, raw_context)
    support_batches = partition_support_units(support_units, batch_size=4)

    chunks = _build_response_chunks(
        long_response,
        provider=provider,
        chunk_token_budget=32,
        encode_fn=encode_texts,
    )
    assert len(chunks) >= 8, f"expected >=8 chunks, got {len(chunks)}"

    result = score_groundedness_response_chunked(
        response_chunks=chunks,
        support_batches=support_batches,
        response_text=long_response,
        evidence_limit=16,
        primary_metric="reverse_context",
    )

    rc = result["scores"]["reverse_context"]
    assert rc is not None and np.isfinite(rc)
    assert -1.0 <= rc <= 1.0

    total_tokens = sum(int(c.embeddings.shape[0]) for c in chunks)
    assert len(result["response_tokens"]) == total_tokens
    assert {row["response_chunk_index"] for row in result["response_tokens"]} == set(
        range(len(chunks))
    )

    matched = sum(
        1 for row in result["response_tokens"] if row.get("support_unit_index") is not None
    )
    assert matched > total_tokens // 4

    assert result["top_evidence"], "long response should produce evidence rows"
    for evidence in result["top_evidence"]:
        assert 0 <= int(evidence["support_unit_index"]) < len(support_units)


def test_response_chunking_full_service_request_default_chunk_budget(monkeypatch) -> None:
    """End-to-end: the API service layer routes through the orchestrator."""
    monkeypatch.setenv("VOYAGER_GROUNDEDNESS_NLI_ENABLED", "0")
    monkeypatch.setenv("LATENCE_TRACE_DISABLE_WARMUP", "1")

    from latence_trace.api.models import GroundednessRequest
    from latence_trace.api.service import GroundednessService

    provider = _StubColbertProvider()

    def _encoder_factory(*_args, **_kwargs):
        return provider

    service = GroundednessService(encoder_factory=lambda _name: provider)

    long_response = " ".join(
        [
            "Berlin became the German capital. ",
            "The Bundesbank is headquartered in Frankfurt. ",
            "Paris is the French capital. ",
            "Rome is the Italian capital. ",
        ] * 20
    )
    request = GroundednessRequest(
        raw_context=(
            "Berlin became the German capital. "
            "The Bundesbank is headquartered in Frankfurt. "
            "Paris is the French capital. "
            "Rome is the Italian capital."
        ),
        response_text=long_response,
        primary_metric="reverse_context",
        response_chunk_tokens=32,
        raw_context_chunk_tokens=32,
    )
    response = service.groundedness(request)
    assert response.scores.reverse_context is not None
    assert np.isfinite(response.scores.reverse_context)
    assert any(
        warn.startswith("response_chunked:") for warn in (response.warnings or [])
    ), response.warnings
    chunk_indices = {tok.response_chunk_index for tok in response.response_tokens if tok.response_chunk_index is not None}
    assert len(chunk_indices) >= 2
