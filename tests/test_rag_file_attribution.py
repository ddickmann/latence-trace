"""Integration regression tests for RAG-lane file attribution.

Verifies that ``score_groundedness_response_chunked`` and the
``GroundednessService._score_rag`` top-level path surface the shared
attribution bundle, populate ``dead_weight_ratio`` /
``dead_weight_file_count`` on the scores payload, and echo the
bundle through to the new top-level ``file_attribution`` field on
``GroundednessResponse``.
"""

from __future__ import annotations

import os

import pytest

from latence_trace.api.models import (
    FileAttributionDiagnostics,
    GroundednessRequest,
    ScoringMode,
)
from latence_trace.api.service import GroundednessService
from latence_trace.core.attribution.file_attribution import FileAttributionResult
from latence_trace.core.groundedness import (
    SupportUnitInput,
    _build_response_chunks,
    encode_texts,
    partition_support_units,
    score_groundedness_response_chunked,
    tokenize_text,
)
from tests.test_context_coverage import (
    _OrthoStubProvider,
    _make_support_units_from_texts,
)


@pytest.fixture(autouse=True)
def _hermetic_voyager_env():
    snapshot = {
        key: value
        for key, value in os.environ.items()
        if key.startswith("VOYAGER_GROUNDEDNESS_") or key.startswith("LATENCE_TRACE_")
    }
    for key in list(os.environ.keys()):
        if (
            key.startswith("VOYAGER_GROUNDEDNESS_")
            or key.startswith("LATENCE_TRACE_")
        ) and key not in snapshot:
            del os.environ[key]
    yield
    for key in list(os.environ.keys()):
        if (
            key.startswith("VOYAGER_GROUNDEDNESS_")
            or key.startswith("LATENCE_TRACE_")
        ) and key not in snapshot:
            del os.environ[key]
    for key, value in snapshot.items():
        if os.environ.get(key) != value:
            os.environ[key] = value


def _units_with_metadata(provider, texts, *, path_for_idx=None, source_for_idx=None):
    """Build support units with optional ``metadata`` for grouping tests."""
    embeddings = encode_texts(provider, texts, is_query=False)
    units = []
    for idx, (text, emb) in enumerate(zip(texts, embeddings)):
        tokens = tokenize_text(provider, text, expected_len=int(emb.shape[0]))
        metadata = {}
        if path_for_idx is not None:
            path = path_for_idx(idx)
            if path is not None:
                metadata["path"] = path
        if source_for_idx is not None:
            source = source_for_idx(idx)
            if source is not None:
                metadata["source"] = source
        units.append(
            SupportUnitInput(
                support_id=f"unit-{idx}",
                text=text,
                embeddings=emb,
                tokens=tokens,
                source_mode="raw_context",
                offset_start=0,
                offset_end=len(text),
                metadata=metadata,
            )
        )
    return units


def _score(response_text, support_texts, *, path_for_idx=None, source_for_idx=None):
    provider = _OrthoStubProvider()
    if path_for_idx is None and source_for_idx is None:
        support_units = _make_support_units_from_texts(provider, support_texts)
    else:
        support_units = _units_with_metadata(
            provider,
            support_texts,
            path_for_idx=path_for_idx,
            source_for_idx=source_for_idx,
        )
    support_batches = partition_support_units(
        support_units, batch_size=max(1, len(support_units))
    )
    response_chunks = _build_response_chunks(
        response_text,
        provider=provider,
        chunk_token_budget=64,
        encode_fn=encode_texts,
    )
    return score_groundedness_response_chunked(
        response_chunks=response_chunks,
        support_batches=support_batches,
        response_text=response_text,
        evidence_limit=8,
        primary_metric="reverse_context",
        coverage_threshold=0.5,
    )


def test_score_groundedness_returns_file_attribution_bundle() -> None:
    result = _score(
        "Berlin is the capital of Germany.",
        [
            "Berlin is the capital of Germany.",
            "Quantum chromodynamics describes the strong nuclear force.",
            "Penguins are flightless aquatic birds native to Antarctica.",
        ],
    )
    fa = result.get("file_attribution")
    assert isinstance(fa, FileAttributionResult)
    assert fa.n_files == 3
    # Plain chunk-ids without metadata fall back to support_id grouping.
    paths = {rec.path for rec in fa.per_file}
    assert paths == {"unit-0", "unit-1", "unit-2"}
    # The two off-topic units must land in dead_weight_files with the
    # NEVER_WON_ARGMAX reason code; the grounded one must not.
    by_path = {rec.path: rec for rec in fa.per_file}
    assert by_path["unit-1"].dead_weight is True
    assert by_path["unit-2"].dead_weight is True
    assert by_path["unit-0"].dead_weight is False
    assert fa.dead_weight_ratio == pytest.approx(2 / 3)
    # Reason code histogram is populated.
    assert fa.reason_code_histogram.get("never_won_argmax", 0) >= 2
    # dead_weight_ratio + count are echoed on the scores payload.
    assert result["scores"]["dead_weight_ratio"] == pytest.approx(2 / 3)
    assert result["scores"]["dead_weight_file_count"] == 2


def test_file_attribution_groups_units_by_metadata_path() -> None:
    """Two chunks sharing ``metadata.path`` should roll up to one file."""

    result = _score(
        "Berlin is the capital of Germany.",
        [
            "Berlin is the capital of Germany.",
            "The capital of Germany is Berlin.",
            "Penguins are flightless aquatic birds native to Antarctica.",
        ],
        path_for_idx=lambda idx: "/docs/germany.md" if idx < 2 else "/docs/penguins.md",
    )
    fa = result["file_attribution"]
    paths = {rec.path for rec in fa.per_file}
    assert paths == {"/docs/germany.md", "/docs/penguins.md"}
    by_path = {rec.path: rec for rec in fa.per_file}
    assert by_path["/docs/germany.md"].n_units == 2
    assert by_path["/docs/penguins.md"].n_units == 1
    assert by_path["/docs/penguins.md"].dead_weight is True


def test_service_level_rag_response_surfaces_file_attribution() -> None:
    provider = _OrthoStubProvider()
    service = GroundednessService(encoder_factory=lambda _model_name=None: provider)
    # Pass three pre-built support units so the RAG scorer sees the
    # three files distinctly; ``raw_context`` would segment into one
    # window when the sentences are short.
    response = service.groundedness(
        GroundednessRequest(
            scoring_mode=ScoringMode.RAG,
            support_units=[
                {
                    "support_id": f"unit-{idx}",
                    "text": text,
                    "metadata": {"path": f"/docs/unit-{idx}.md"},
                }
                for idx, text in enumerate(
                    [
                        "Berlin is the capital of Germany.",
                        "Quantum chromodynamics describes the strong nuclear force.",
                        "Penguins are flightless aquatic birds native to Antarctica.",
                    ]
                )
            ],
            response_text="Berlin is the capital of Germany.",
        )
    )

    assert response.scoring_mode == ScoringMode.RAG
    assert response.file_attribution is not None
    assert isinstance(response.file_attribution, FileAttributionDiagnostics)
    assert response.file_attribution.n_files >= 2
    assert response.file_attribution.dead_weight_ratio > 0.0
    # Reason-code histogram must ship on every response that has attribution.
    assert isinstance(response.file_attribution.reason_code_histogram, dict)
    assert response.file_attribution.reason_code_histogram
    # Scores payload echoes the same numbers so downstream clients
    # that only look at ``scores`` still see the upsell signal.
    assert response.scores.dead_weight_ratio is not None
    assert response.scores.dead_weight_file_count is not None
    assert response.scores.dead_weight_ratio > 0.0
    # Default heatmap_format is "data" — the heatmap payload must be populated.
    assert response.heatmap is not None
    assert response.heatmap_html is None
