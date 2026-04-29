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


def test_service_raw_context_with_file_headers_splits_into_per_file_units() -> None:
    """Regression: multi-file ``raw_context`` must populate ``n_files``.

    Before A0a the raw_context path ran a flat sentence segmenter that
    never saw the ``# file: ...`` headers, so a bundle of three files
    collapsed into a single-path bucket (``n_files == 1``) and
    ``dead_weight_files`` was always empty regardless of how many
    off-topic files were shipped.

    This test pins the fix by passing a 3-file bundle (one grounded,
    two off-topic) and asserting the per-file rollup shows three
    distinct paths and flags the two off-topic files as dead weight.
    """

    provider = _OrthoStubProvider()
    service = GroundednessService(encoder_factory=lambda _model_name=None: provider)

    raw_context = (
        "# file: src/auth.py\n"
        "Berlin is the capital of Germany.\n"
        "\n"
        "# file: src/db.py\n"
        "Quantum chromodynamics describes the strong nuclear force.\n"
        "\n"
        "# file: src/api.py\n"
        "Penguins are flightless aquatic birds native to Antarctica.\n"
    )
    response = service.groundedness(
        GroundednessRequest(
            scoring_mode=ScoringMode.RAG,
            raw_context=raw_context,
            response_text="Berlin is the capital of Germany.",
        )
    )

    assert response.file_attribution is not None
    paths = {rec.path for rec in response.file_attribution.per_file}
    assert {"src/auth.py", "src/db.py", "src/api.py"}.issubset(paths), (
        f"expected all three file-headed paths in per_file bucket, "
        f"got {paths}"
    )
    assert response.file_attribution.n_files == 3

    # The grounded file must NOT be flagged.
    dead = set(response.file_attribution.dead_weight_files)
    assert "src/auth.py" not in dead

    # Each off-topic file must surface at least one dead-weight signal
    # in its per-file reason-code set. Exact dead_weight flag depends
    # on argmax-tie dynamics in the ortho-stub provider, but reason
    # codes (low_cosine, never_won_argmax, dominated_by_single_file)
    # are stable under ties and are the auditable output customers
    # see in production.
    by_path = {rec.path: rec for rec in response.file_attribution.per_file}
    offtopic_codes = {
        path: {
            rc.value if hasattr(rc, "value") else str(rc)
            for rc in by_path[path].reason_codes
        }
        for path in ("src/db.py", "src/api.py")
    }
    for path, codes in offtopic_codes.items():
        assert codes, f"expected dead-weight reason codes for {path}, got none"
        # Any of these three codes flags the file as off-topic.
        assert codes & {
            "never_won_argmax",
            "all_tokens_below_0_40",
            "dominated_by_single_file",
        }, f"{path} reason codes {codes} contain no dead-weight signal"

    # The grounded file must have high owner share and max evidence.
    auth = by_path["src/auth.py"]
    assert auth.owner_share > 0.5
    assert auth.max_evidence > 0.9


def test_service_raw_context_without_headers_falls_back_to_single_bucket() -> None:
    """Plain prose raw_context (no headers) must keep the legacy shape.

    If the splitter fires on prose where no file-header marker is
    present, we'd regress the enterprise RAG story (Veracier,
    HaluEval, RAGTruth) where the context is a single document and
    ``n_files == 1`` is the correct shape.
    """

    provider = _OrthoStubProvider()
    service = GroundednessService(encoder_factory=lambda _model_name=None: provider)

    response = service.groundedness(
        GroundednessRequest(
            scoring_mode=ScoringMode.RAG,
            raw_context=(
                "Berlin is the capital of Germany and one of its major "
                "cultural and political centres. The city has a "
                "population of roughly 3.7 million."
            ),
            response_text="Berlin is the capital of Germany.",
        )
    )

    assert response.file_attribution is not None
    # No headers -> every segment gets the synthetic ``raw-{idx}``
    # support_id as the grouping key; n_files equals the number of
    # segments the splitter produced, and none of those paths have a
    # slash-style file extension.
    for rec in response.file_attribution.per_file:
        assert rec.path.startswith("raw-"), (
            f"prose context without headers should NOT produce a "
            f"file-path grouping key, got {rec.path!r}"
        )
