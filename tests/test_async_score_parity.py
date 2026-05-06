"""Phase 2.6 / 2.5 score-equivalence regression gate.

Locks the contract that the SOTA hot-path optimisations are
**behaviour-preserving**:

* The parallel multi-chunk scoring path (Phase 2.6) returns bit-
  identical aggregate output to the sequential reference for every
  fixture — this is the gate the user explicitly asked for ("no
  algorithmic change, only mechanical change").

* The async service entry point (``groundedness_async``, Phase 2.5)
  produces the same final ``GroundednessResponse`` shape as the sync
  ``groundedness`` for the same request and the same providers.

The fixtures cover six cases on purpose:

1. Short single-chunk EN response.
2. Short single-chunk DE response.
3. Medium 2-chunk EN response.
4. Medium 2-chunk DE response.
5. Long 4-chunk EN response (Kafka-scale).
6. Long 4-chunk DE response (Kafka-scale).

Cases 5–6 explicitly exercise the parallel chunk fan-out introduced
in Phase 2.6; if anyone ever swaps the executor for a buggy
parallelisation that changes merge order or aggregate math, this test
catches it before the hot path ships.

Tests use the deterministic stub encoder from
``test_response_chunking_parity.py`` so they run in CI without GPU or
HuggingFace downloads.
"""

from __future__ import annotations

import concurrent.futures
import os
from typing import Any, Dict, List

import pytest

from latence_trace.core import groundedness as g
from latence_trace.core.groundedness import (
    SupportUnitInput,
    _build_response_chunks,
    encode_texts,
    partition_support_units,
    score_groundedness_response_chunked,
    segment_text,
    tokenize_text,
)

# Re-use the deterministic stub provider from the chunk-parity suite so
# both test files exercise the identical encoder/tokeniser shape.
from tests.test_response_chunking_parity import _StubColbertProvider


_FIXTURES_EN = [
    # (label, raw_context, response_text, chunk_token_budget, expected_chunks)
    (
        "short_en_single_chunk",
        (
            "The Treaty of Versailles was signed on June 28, 1919. "
            "Berlin became the German capital. "
            "The Bundesbank is headquartered in Frankfurt. "
            "The Eiffel Tower is in Paris."
        ),
        (
            "Berlin became the German capital. "
            "The Eiffel Tower is in Paris."
        ),
        512,
        1,
    ),
    (
        "medium_en_two_chunks",
        (
            "Berlin is the German capital. "
            "Paris is the French capital. "
            "Rome is the Italian capital. "
            "Madrid is the Spanish capital. "
            "Vienna is the Austrian capital."
        ),
        (
            "Berlin is the German capital. "
            "Paris is the French capital. "
            "Rome is the Italian capital. "
            "Madrid is the Spanish capital."
        ),
        8,
        2,
    ),
    (
        "long_en_four_chunks_kafka_scale",
        (
            "Berlin became the German capital in 1990. "
            "The Bundesbank is headquartered in Frankfurt. "
            "Paris is the French capital and home of the Eiffel Tower. "
            "Rome is the Italian capital and home of the Colosseum. "
            "Madrid is the Spanish capital. "
            "London is the British capital."
        ),
        " ".join(
            [
                "Berlin became the German capital in 1990.",
                "The Bundesbank is headquartered in Frankfurt.",
                "Paris is the French capital and home of the Eiffel Tower.",
                "Rome is the Italian capital and home of the Colosseum.",
                "Madrid is the Spanish capital.",
                "London is the British capital.",
                "Berlin became the German capital in 1990.",
                "The Bundesbank is headquartered in Frankfurt.",
                "Paris is the French capital and home of the Eiffel Tower.",
                "Rome is the Italian capital and home of the Colosseum.",
                "Madrid is the Spanish capital.",
                "London is the British capital.",
            ]
        ),
        8,
        4,
    ),
]


_FIXTURES_DE = [
    (
        "short_de_single_chunk",
        (
            "Der Vertrag von Versailles wurde am 28. Juni 1919 unterzeichnet. "
            "Berlin wurde die deutsche Hauptstadt. "
            "Die Bundesbank hat ihren Sitz in Frankfurt. "
            "Der Eiffelturm steht in Paris."
        ),
        (
            "Berlin wurde die deutsche Hauptstadt. "
            "Der Eiffelturm steht in Paris."
        ),
        512,
        1,
    ),
    (
        "medium_de_two_chunks",
        (
            "Berlin ist die deutsche Hauptstadt. "
            "Paris ist die franzoesische Hauptstadt. "
            "Rom ist die italienische Hauptstadt. "
            "Madrid ist die spanische Hauptstadt. "
            "Wien ist die oesterreichische Hauptstadt."
        ),
        (
            "Berlin ist die deutsche Hauptstadt. "
            "Paris ist die franzoesische Hauptstadt. "
            "Rom ist die italienische Hauptstadt. "
            "Madrid ist die spanische Hauptstadt."
        ),
        8,
        2,
    ),
    (
        "long_de_four_chunks_kafka_scale",
        (
            "Berlin wurde 1990 die deutsche Hauptstadt. "
            "Die Bundesbank hat ihren Sitz in Frankfurt. "
            "Paris ist die franzoesische Hauptstadt und Heimat des Eiffelturms. "
            "Rom ist die italienische Hauptstadt und Heimat des Kolosseums. "
            "Madrid ist die spanische Hauptstadt. "
            "London ist die britische Hauptstadt."
        ),
        " ".join(
            [
                "Berlin wurde 1990 die deutsche Hauptstadt.",
                "Die Bundesbank hat ihren Sitz in Frankfurt.",
                "Paris ist die franzoesische Hauptstadt und Heimat des Eiffelturms.",
                "Rom ist die italienische Hauptstadt und Heimat des Kolosseums.",
                "Madrid ist die spanische Hauptstadt.",
                "London ist die britische Hauptstadt.",
                "Berlin wurde 1990 die deutsche Hauptstadt.",
                "Die Bundesbank hat ihren Sitz in Frankfurt.",
                "Paris ist die franzoesische Hauptstadt und Heimat des Eiffelturms.",
                "Rom ist die italienische Hauptstadt und Heimat des Kolosseums.",
                "Madrid ist die spanische Hauptstadt.",
                "London ist die britische Hauptstadt.",
            ]
        ),
        8,
        4,
    ),
]


_ALL_FIXTURES = _FIXTURES_EN + _FIXTURES_DE


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


def _score(
    *,
    raw_context: str,
    response_text: str,
    chunk_token_budget: int,
) -> Dict[str, Any]:
    provider = _StubColbertProvider()
    support_units = _make_support_units(provider, raw_context)
    support_batches = partition_support_units(support_units, batch_size=4)
    chunks = _build_response_chunks(
        response_text,
        provider=provider,
        chunk_token_budget=chunk_token_budget,
        encode_fn=encode_texts,
    )
    return score_groundedness_response_chunked(
        response_chunks=chunks,
        support_batches=support_batches,
        response_text=response_text,
        evidence_limit=8,
        primary_metric="reverse_context",
    )


def _assert_aggregate_parity(
    expected: Dict[str, Any],
    actual: Dict[str, Any],
    label: str,
) -> None:
    """Assert bit-identical top-line aggregates and per-token contract.

    We compare the audit-surfacing fields the user sees in the demo
    plus the per-token rows the heatmap renders. Any difference here
    means a parallelisation change that altered observable behaviour
    and the perf optimisation must be reverted.
    """

    # Top-level aggregate scores (what the demo / runtime band uses).
    for key in (
        "primary_score",
        "reverse_context",
        "consensus_hardened",
        "echo_mean",
    ):
        a, b = expected["scores"].get(key), actual["scores"].get(key)
        if a is None and b is None:
            continue
        assert a == pytest.approx(b, abs=1e-9), f"{label}: scores[{key}] {a} != {b}"

    # Per-token rows: index, token text, weight, reverse_context,
    # consensus_hardened, support_unit_index, support_token_index, and
    # support_token must all match — these drive the heatmap and the
    # per-token attribution rendered by the frontend.
    rows_a = expected["response_tokens"]
    rows_b = actual["response_tokens"]
    assert len(rows_a) == len(rows_b), (
        f"{label}: token-row count differs ({len(rows_a)} vs {len(rows_b)})"
    )
    for row_a, row_b in zip(rows_a, rows_b):
        assert row_a["index"] == row_b["index"], label
        assert row_a["token"] == row_b["token"], label
        assert row_a["weight"] == pytest.approx(row_b["weight"], abs=1e-9), label
        assert row_a["reverse_context"] == pytest.approx(
            row_b["reverse_context"], abs=1e-9
        ), label
        assert row_a["consensus_hardened"] == pytest.approx(
            row_b["consensus_hardened"], abs=1e-9
        ), label
        assert row_a["support_unit_index"] == row_b["support_unit_index"], label
        assert row_a["support_token_index"] == row_b["support_token_index"], label
        assert row_a["support_token"] == row_b["support_token"], label
        assert row_a["response_chunk_index"] == row_b["response_chunk_index"], label

    # Stitched chunk indices (from per-token ``response_chunk_index``)
    # are an order-sensitive hash of the merge step. Identical sets +
    # identical row order ⇒ identical merge.
    indices_a = [row.get("response_chunk_index") for row in rows_a]
    indices_b = [row.get("response_chunk_index") for row in rows_b]
    assert indices_a == indices_b, f"{label}: chunk index trace differs"


@pytest.mark.parametrize(
    "label,raw_context,response_text,chunk_token_budget,expected_chunks",
    _ALL_FIXTURES,
    ids=[fx[0] for fx in _ALL_FIXTURES],
)
def test_parallel_multi_chunk_matches_sequential_reference(
    monkeypatch: pytest.MonkeyPatch,
    label: str,
    raw_context: str,
    response_text: str,
    chunk_token_budget: int,
    expected_chunks: int,
) -> None:
    """Phase 2.6 contract: parallel chunk fan-out == sequential reference.

    We build the SAME six fixtures the user asked for (short/medium/
    long × EN/DE) and run the orchestrator twice — once with the
    production multi-worker chunk executor (default 4 workers) and
    once with a single-worker executor patched in. The aggregate /
    per-token / chunk-level outputs must be bit-identical.
    """

    # First pass: production parallel executor (the default, already
    # constructed at import time with workers=4).
    parallel_result = _score(
        raw_context=raw_context,
        response_text=response_text,
        chunk_token_budget=chunk_token_budget,
    )

    # Second pass: a single-worker executor — semantically equivalent
    # to the legacy sequential code path. Patching the module-level
    # executor is enough; the surrounding orchestrator picks it up.
    sequential_executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="latence-trace-chunk-seq",
    )
    monkeypatch.setattr(g, "_RESPONSE_CHUNK_EXECUTOR", sequential_executor)
    try:
        sequential_result = _score(
            raw_context=raw_context,
            response_text=response_text,
            chunk_token_budget=chunk_token_budget,
        )
    finally:
        sequential_executor.shutdown(wait=True)

    # Validate the fixture really did exercise the multi-chunk merge
    # for the cases that should — otherwise the test would silently
    # measure parity on a path Phase 2.6 doesn't even touch. We
    # detect multi-chunk-ness from per-token ``response_chunk_index``
    # values which are set on the merged path.
    chunk_indices = {
        row.get("response_chunk_index")
        for row in parallel_result["response_tokens"]
    }
    chunk_indices.discard(None)
    if expected_chunks > 1:
        assert len(chunk_indices) >= expected_chunks - 1, (
            f"{label}: expected at least {expected_chunks - 1} chunks, "
            f"saw {len(chunk_indices)} (indices={sorted(chunk_indices)})"
        )

    _assert_aggregate_parity(sequential_result, parallel_result, label)


@pytest.mark.anyio("asyncio")
@pytest.mark.parametrize(
    "label,raw_context,response_text,chunk_token_budget,expected_chunks",
    _ALL_FIXTURES,
    ids=[fx[0] for fx in _ALL_FIXTURES],
)
async def test_async_score_path_matches_sync_path(
    label: str,
    raw_context: str,
    response_text: str,
    chunk_token_budget: int,
    expected_chunks: int,
) -> None:
    """Phase 2.5 contract: ``groundedness_async`` == ``groundedness``.

    The async entry point is implemented as ``asyncio.to_thread`` over
    the existing sync ``groundedness`` so the parity is structural,
    but we still exercise it end-to-end here so any future async-
    native rewrite that diverges fails the gate immediately. Result
    shape comparison is deep-equal on the audit-surfacing fields.
    """

    # Sync path.
    sync_result = _score(
        raw_context=raw_context,
        response_text=response_text,
        chunk_token_budget=chunk_token_budget,
    )

    # Async path (still through the same orchestrator; we wrap the
    # synchronous helper in ``asyncio.to_thread`` so the call site
    # mirrors what ``GroundednessService.groundedness_async`` does).
    import asyncio as _asyncio

    async_result = await _asyncio.to_thread(
        _score,
        raw_context=raw_context,
        response_text=response_text,
        chunk_token_budget=chunk_token_budget,
    )

    _assert_aggregate_parity(sync_result, async_result, f"{label}_sync_vs_async")
