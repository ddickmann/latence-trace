"""Context-coverage observability tests.

Covers the per-support-unit ``coverage_score`` / ``used`` flags and the
global ``context_coverage_ratio`` exposed by the scoring functions:

1. **Helper invariants** — ``compute_unit_coverage`` returns the
   per-column max of the (response, unit) similarity matrix, the correct
   used-count, and a defensive empty-matrix payload.

2. **End-to-end through the response-chunked orchestrator** — when one
   support unit is the obvious winner and the other is a distractor,
   the winner is flagged ``used=True`` with high coverage_score and the
   distractor is flagged ``used=False``. The global ratios match.

3. **Threshold sensitivity** — the same inputs scored with a stricter
   threshold report a lower ``context_coverage_ratio`` and the same
   per-unit ``coverage_score`` values (the score is independent of the
   threshold; only the boolean flag changes).

4. **Multi-chunk parity** — splitting the response into many windows
   yields the same per-unit ``coverage_score`` as scoring it in one
   window, because per-unit coverage is a max over response tokens and
   max is invariant to row partitioning.

All tests use the deterministic stub encoder from the response-chunking
parity test module so they run in CI without GPU or HF downloads.
"""

from __future__ import annotations

from typing import List

import pytest
import torch

from latence_trace.core.groundedness import (
    SupportUnitInput,
    _build_response_chunks,
    compute_unit_coverage,
    encode_texts,
    partition_support_units,
    score_groundedness_response_chunked,
    segment_text,
    tokenize_text,
)

from tests.test_response_chunking_parity import _StubColbertProvider


class _OrthoStubProvider(_StubColbertProvider):
    """Wider hash space (1024 dim) so distinct tokens collide much less.

    The 16-dim parent stub is great for parity tests (deterministic and
    fast), but for coverage assertions we need genuinely orthogonal-ish
    vectors per token so a "Berlin"-token never accidentally matches a
    "quantum"-token at coverage_score >= 0.5. 1024 dim drops the expected
    collision probability between two single-hot vectors to ~1e-3, which
    is enough for the deterministic content used in these tests.
    """

    model_name = "stub-ortho"

    def __init__(self) -> None:
        super().__init__(dim=1024)


def _make_support_units_from_texts(
    provider: _StubColbertProvider, texts: List[str]
) -> List[SupportUnitInput]:
    embeddings = encode_texts(provider, texts, is_query=False)
    units: List[SupportUnitInput] = []
    for idx, (text, emb) in enumerate(zip(texts, embeddings)):
        tokens = tokenize_text(provider, text, expected_len=int(emb.shape[0]))
        units.append(
            SupportUnitInput(
                support_id=f"unit-{idx}",
                text=text,
                embeddings=emb,
                tokens=tokens,
                source_mode="raw_context",
                offset_start=0,
                offset_end=len(text),
            )
        )
    return units


def test_compute_unit_coverage_returns_column_maxes_and_used_counts() -> None:
    matrix = torch.tensor(
        [
            [0.9, 0.2, 0.6],
            [0.3, 0.1, 0.7],
            [0.8, 0.4, 0.5],
        ],
        dtype=torch.float32,
    )

    coverage = compute_unit_coverage(matrix, threshold=0.5)

    assert coverage["per_unit_max"] == pytest.approx([0.9, 0.4, 0.7])
    assert coverage["used_mask"] == [True, False, True]
    assert coverage["used_count"] == 2
    assert coverage["total_count"] == 3
    assert coverage["coverage_ratio"] == pytest.approx(2 / 3)
    assert coverage["threshold"] == 0.5


def test_compute_unit_coverage_handles_empty_matrix() -> None:
    coverage_zero_units = compute_unit_coverage(
        torch.empty((4, 0), dtype=torch.float32), threshold=0.5
    )
    assert coverage_zero_units["used_count"] == 0
    assert coverage_zero_units["total_count"] == 0
    assert coverage_zero_units["coverage_ratio"] == 0.0
    assert coverage_zero_units["per_unit_max"] == []
    assert coverage_zero_units["used_mask"] == []

    coverage_zero_response = compute_unit_coverage(
        torch.empty((0, 5), dtype=torch.float32), threshold=0.5
    )
    assert coverage_zero_response["total_count"] == 0
    assert coverage_zero_response["coverage_ratio"] == 0.0


def test_compute_unit_coverage_clamps_non_finite_to_zero_with_used_false() -> None:
    """0-token / numerically-degenerate units must not leak negative scores
    into the API response, and must never count as ``used`` regardless of
    the threshold the operator picks (including ``threshold = 0.0``).
    """

    matrix = torch.tensor(
        [
            [0.9, float("-inf"), float("nan"), 0.7],
            [0.8, float("-inf"), float("nan"), 0.6],
        ],
        dtype=torch.float32,
    )

    coverage = compute_unit_coverage(matrix, threshold=0.0)

    per_unit = coverage["per_unit_max"]
    used_mask = coverage["used_mask"]

    assert per_unit[0] == pytest.approx(0.9)
    assert per_unit[1] == 0.0
    assert per_unit[2] == 0.0
    assert per_unit[3] == pytest.approx(0.7)
    for value in per_unit:
        assert value >= 0.0, "coverage_score must stay in the [0, 1] similarity range"

    assert used_mask == [True, False, False, True]
    assert coverage["used_count"] == 2
    assert coverage["coverage_ratio"] == pytest.approx(0.5)


def test_compute_unit_coverage_threshold_only_changes_used_flags() -> None:
    matrix = torch.tensor([[0.9, 0.2, 0.55, 0.4]], dtype=torch.float32)

    loose = compute_unit_coverage(matrix, threshold=0.3)
    strict = compute_unit_coverage(matrix, threshold=0.6)

    assert loose["per_unit_max"] == pytest.approx(strict["per_unit_max"])
    assert loose["used_mask"] == [True, False, True, True]
    assert strict["used_mask"] == [True, False, False, False]
    assert loose["used_count"] == 3
    assert strict["used_count"] == 1
    assert loose["coverage_ratio"] == pytest.approx(0.75)
    assert strict["coverage_ratio"] == pytest.approx(0.25)


def test_orchestrator_flags_unused_support_units() -> None:
    """When the response only echoes one support unit, the others must be flagged unused."""

    provider = _OrthoStubProvider()
    support_texts = [
        "Berlin is the capital of Germany.",
        "Quantum chromodynamics describes the strong nuclear force.",
        "Penguins are flightless aquatic birds native to Antarctica.",
    ]
    response_text = "Berlin is the capital of Germany."

    support_units = _make_support_units_from_texts(provider, support_texts)
    support_batches = partition_support_units(support_units, batch_size=4)
    chunks = _build_response_chunks(
        response_text,
        provider=provider,
        chunk_token_budget=64,
        encode_fn=encode_texts,
    )

    result = score_groundedness_response_chunked(
        response_chunks=chunks,
        support_batches=support_batches,
        response_text=response_text,
        evidence_limit=8,
        primary_metric="reverse_context",
        coverage_threshold=0.5,
    )

    units = result["support_units"]
    assert len(units) == 3

    by_id = {unit["support_id"]: unit for unit in units}
    berlin_unit = by_id["unit-0"]
    qcd_unit = by_id["unit-1"]
    penguin_unit = by_id["unit-2"]

    assert berlin_unit["used"] is True
    assert berlin_unit["coverage_score"] >= 0.5
    assert qcd_unit["used"] is False
    assert penguin_unit["used"] is False
    assert qcd_unit["coverage_score"] < berlin_unit["coverage_score"]
    assert penguin_unit["coverage_score"] < berlin_unit["coverage_score"]

    scores = result["scores"]
    assert scores["support_units_total"] == 3
    assert scores["support_units_used"] == 1
    assert scores["context_coverage_ratio"] == pytest.approx(1 / 3)
    assert scores["context_coverage_threshold"] == pytest.approx(0.5)
    assert scores["context_attribution_used_count"] == 1
    assert scores["context_attribution_ratio"] == pytest.approx(1 / 3)


def test_orchestrator_threshold_changes_used_count_not_scores() -> None:
    """Tightening the threshold lowers context_coverage_ratio but keeps coverage_score."""

    provider = _OrthoStubProvider()
    support_texts = [
        "Berlin is the capital of Germany.",
        "Munich hosts Oktoberfest annually.",
        "Hamburg is a major North Sea port.",
    ]
    response_text = "Berlin is the capital of Germany."

    support_units = _make_support_units_from_texts(provider, support_texts)
    support_batches = partition_support_units(support_units, batch_size=4)
    chunks = _build_response_chunks(
        response_text,
        provider=provider,
        chunk_token_budget=64,
        encode_fn=encode_texts,
    )

    loose = score_groundedness_response_chunked(
        response_chunks=chunks,
        support_batches=support_batches,
        response_text=response_text,
        evidence_limit=8,
        primary_metric="reverse_context",
        coverage_threshold=0.05,
    )
    strict = score_groundedness_response_chunked(
        response_chunks=chunks,
        support_batches=support_batches,
        response_text=response_text,
        evidence_limit=8,
        primary_metric="reverse_context",
        coverage_threshold=0.99,
    )

    loose_scores = {unit["support_id"]: unit["coverage_score"] for unit in loose["support_units"]}
    strict_scores = {unit["support_id"]: unit["coverage_score"] for unit in strict["support_units"]}
    assert loose_scores == pytest.approx(strict_scores)

    assert loose["scores"]["context_coverage_ratio"] >= strict["scores"]["context_coverage_ratio"]
    assert loose["scores"]["support_units_used"] >= strict["scores"]["support_units_used"]
    assert loose["scores"]["context_coverage_threshold"] == pytest.approx(0.05)
    assert strict["scores"]["context_coverage_threshold"] == pytest.approx(0.99)


def test_coverage_is_invariant_to_response_chunking() -> None:
    """Per-unit coverage_score is a max over response tokens; chunk partition cannot change it."""

    provider = _StubColbertProvider()
    raw_context = (
        "Berlin is the capital of Germany. "
        "Paris is the capital of France. "
        "Rome is the capital of Italy. "
        "Madrid is the capital of Spain."
    )
    response_text = (
        "Berlin is the capital of Germany. "
        "Paris is the capital of France. "
        "Rome is the capital of Italy."
    )

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
    support_batches = partition_support_units(units, batch_size=4)

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
        coverage_threshold=0.5,
    )
    many = score_groundedness_response_chunked(
        response_chunks=chunks_many,
        support_batches=support_batches,
        response_text=response_text,
        evidence_limit=8,
        primary_metric="reverse_context",
        coverage_threshold=0.5,
    )

    one_by_id = {unit["support_id"]: unit for unit in one["support_units"]}
    many_by_id = {unit["support_id"]: unit for unit in many["support_units"]}
    assert one_by_id.keys() == many_by_id.keys()
    for support_id in one_by_id:
        assert one_by_id[support_id]["coverage_score"] == pytest.approx(
            many_by_id[support_id]["coverage_score"], abs=1e-5
        )
        assert one_by_id[support_id]["used"] == many_by_id[support_id]["used"]

    assert one["scores"]["context_coverage_ratio"] == pytest.approx(
        many["scores"]["context_coverage_ratio"]
    )
    assert one["scores"]["support_units_used"] == many["scores"]["support_units_used"]
    assert one["scores"]["support_units_total"] == many["scores"]["support_units_total"]


def test_coverage_and_attribution_are_independent_signals() -> None:
    """Coverage and attribution measure different things and neither dominates.

    - coverage_used_count = # units whose strongest token-match crossed
      ``coverage_threshold`` (absolute strength signal).
    - attribution_used_count = # units that won the argmax for at least
      one response token (competitive signal — winning depends on
      sibling units).

    With a tight threshold, a unit can win argmax for some token (so
    ``matched_response_tokens > 0``) yet its strongest similarity stays
    below threshold. Conversely, a unit can have a high coverage_score
    but lose every argmax to a sibling unit. We assert the two scalars
    move independently and that both stay in ``[0, 1]``.
    """

    provider = _OrthoStubProvider()
    support_texts = [
        "Berlin is the capital of Germany.",
        "Munich is in Bavaria.",
        "Frankfurt is on the river Main.",
        "Hamburg has a major port.",
    ]
    response_text = "Berlin is the capital of Germany. Munich is in Bavaria."

    support_units = _make_support_units_from_texts(provider, support_texts)
    support_batches = partition_support_units(support_units, batch_size=8)
    chunks = _build_response_chunks(
        response_text,
        provider=provider,
        chunk_token_budget=64,
        encode_fn=encode_texts,
    )

    permissive = score_groundedness_response_chunked(
        response_chunks=chunks,
        support_batches=support_batches,
        response_text=response_text,
        evidence_limit=16,
        primary_metric="reverse_context",
        coverage_threshold=0.0,
    )
    permissive_scores = permissive["scores"]
    # At threshold 0.0 every finite-similarity unit is "used", which is
    # the upper bound; attribution by definition lies in [0, total].
    assert permissive_scores["context_coverage_ratio"] <= 1.0 + 1e-9
    assert permissive_scores["context_attribution_ratio"] <= 1.0 + 1e-9

    strict = score_groundedness_response_chunked(
        response_chunks=chunks,
        support_batches=support_batches,
        response_text=response_text,
        evidence_limit=16,
        primary_metric="reverse_context",
        coverage_threshold=0.999,
    )
    strict_scores = strict["scores"]
    # Tightening the threshold only lowers coverage_used_count; the
    # attribution_used_count is independent of the threshold.
    assert strict_scores["support_units_used"] <= permissive_scores["support_units_used"]
    assert (
        strict_scores["context_attribution_used_count"]
        == permissive_scores["context_attribution_used_count"]
    )
