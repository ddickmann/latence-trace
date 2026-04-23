from __future__ import annotations

import os
import re

import pytest
import torch

from latence_trace.api.models import (
    GroundednessRequest,
    GroundednessUsageState,
)
from latence_trace.api.service import GroundednessService
from latence_trace.core.groundedness import (
    UsageClassifierThresholds,
    _build_response_chunks,
    _compute_redundancy_matrix,
    _usage_redundancy_similarity,
    encode_texts,
    partition_support_units,
    score_groundedness_response_chunked,
)
from tests.test_context_coverage import (
    _OrthoStubProvider,
    _make_support_units_from_texts,
)


@pytest.fixture(autouse=True)
def _hermetic_voyager_env():
    """Strip ``VOYAGER_GROUNDEDNESS_*`` env vars before every test.

    These tests drive ``GroundednessService`` with in-process stub
    encoders and no NLI backend. If an earlier test in the session (e.g.
    ``test_agent_friendly_api`` or ``test_kernel_warmup``) imported
    ``server.main`` and triggered ``apply_profile("balanced")``, the
    process inherits ``VOYAGER_GROUNDEDNESS_NLI_ENABLED=1`` and a full
    fusion / threshold overlay. That silently reconfigures the
    downstream service and flips tri-state usage labels. Clearing the
    namespace at the per-test boundary keeps every assertion in this
    file deterministic regardless of collection order.
    """

    snapshot = {
        key: value
        for key, value in os.environ.items()
        if key.startswith("VOYAGER_GROUNDEDNESS_")
        or key.startswith("LATENCE_TRACE_")
    }
    for key in list(os.environ.keys()):
        if key.startswith("VOYAGER_GROUNDEDNESS_") or key.startswith("LATENCE_TRACE_"):
            del os.environ[key]
    try:
        yield
    finally:
        for key in list(os.environ.keys()):
            if (
                key.startswith("VOYAGER_GROUNDEDNESS_")
                or key.startswith("LATENCE_TRACE_")
            ) and key not in snapshot:
                del os.environ[key]
        for key, value in snapshot.items():
            if os.environ.get(key) != value:
                os.environ[key] = value


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", (text or "").lower())).strip()


class _MappingNLIProvider:
    def __init__(self, entailed_pairs: set[tuple[str, str]]) -> None:
        self._entailed_pairs = {
            (_normalize_text(premise), _normalize_text(hypothesis))
            for premise, hypothesis in entailed_pairs
        }

    def entail(self, premises, hypotheses):
        out = []
        for premise, hypothesis in zip(premises, hypotheses):
            key = (_normalize_text(premise), _normalize_text(hypothesis))
            if key in self._entailed_pairs:
                out.append((0.94, 0.05, 0.01))
            else:
                out.append((0.08, 0.62, 0.30))
        return out


class _MappingReranker:
    def __init__(self, relevant_pairs: set[tuple[str, str]]) -> None:
        self._relevant_pairs = {
            (_normalize_text(claim), _normalize_text(premise))
            for claim, premise in relevant_pairs
        }

    def score(self, claim: str, candidate_premises):
        claim_key = _normalize_text(claim)
        return [
            1.0 if (claim_key, _normalize_text(candidate)) in self._relevant_pairs else 0.0
            for candidate in candidate_premises
        ]


def _score_lower_level(
    response_text: str,
    support_texts: list[str],
    *,
    provider: _OrthoStubProvider | None = None,
    response_chunk_tokens: int = 64,
    coverage_threshold: float = 0.5,
    nli_provider=None,
    nli_reranker=None,
):
    provider = provider or _OrthoStubProvider()
    support_units = _make_support_units_from_texts(provider, support_texts)
    support_batches = partition_support_units(support_units, batch_size=max(1, len(support_units)))
    response_chunks = _build_response_chunks(
        response_text,
        provider=provider,
        chunk_token_budget=response_chunk_tokens,
        encode_fn=encode_texts,
    )
    result = score_groundedness_response_chunked(
        response_chunks=response_chunks,
        support_batches=support_batches,
        response_text=response_text,
        evidence_limit=8,
        primary_metric="reverse_context",
        coverage_threshold=coverage_threshold,
        nli_provider=nli_provider,
        nli_reranker=nli_reranker,
        nli_top_k_premises=1 if nli_provider is not None else None,
        nli_concat_premises=False if nli_provider is not None else None,
        nli_use_atomic_claims=False if nli_provider is not None else None,
    )
    return result


def _build_service(provider: _OrthoStubProvider) -> GroundednessService:
    return GroundednessService(encoder_factory=lambda _model_name=None: provider)


def test_unused_context_emits_tristate_fields_and_counts() -> None:
    result = _score_lower_level(
        "Berlin is the capital of Germany.",
        [
            "Berlin is the capital of Germany.",
            "Quantum chromodynamics describes the strong nuclear force.",
            "Penguins are flightless aquatic birds native to Antarctica.",
        ],
    )

    by_id = {unit["support_id"]: unit for unit in result["support_units"]}
    assert by_id["unit-0"]["usage_state"] == "used"
    assert by_id["unit-1"]["usage_state"] == "unused"
    assert by_id["unit-2"]["usage_state"] == "unused"
    for unit in result["support_units"]:
        assert unit["usage_confidence"] is not None
        assert unit["unused_confidence"] is not None

    scores = result["scores"]
    assert scores["support_units_usage_used"] == 1
    assert scores["support_units_unused"] == 2
    assert scores["support_units_uncertain"] == 0
    assert scores["context_usage_ratio"] == pytest.approx(1 / 3)
    assert scores["context_unused_ratio"] == pytest.approx(2 / 3)
    assert scores["context_uncertain_ratio"] == pytest.approx(0.0)
    assert scores["support_units_used"] == 1


def test_near_duplicate_sibling_becomes_uncertain_not_unused() -> None:
    result = _score_lower_level(
        "Berlin is the capital of Germany.",
        [
            "Berlin is the capital of Germany.",
            "The capital city of Germany is Berlin.",
            "Penguins are flightless aquatic birds native to Antarctica.",
        ],
    )

    by_id = {unit["support_id"]: unit for unit in result["support_units"]}
    assert by_id["unit-0"]["usage_state"] == "used"
    assert by_id["unit-1"]["usage_state"] == "uncertain"
    assert by_id["unit-2"]["usage_state"] == "unused"

    scores = result["scores"]
    assert scores["support_units_usage_used"] == 1
    assert scores["support_units_unused"] == 1
    assert scores["support_units_uncertain"] == 1


def test_usage_state_is_invariant_to_response_chunking() -> None:
    provider = _OrthoStubProvider()
    support_texts = [
        "Berlin is the capital of Germany.",
        "The capital city of Germany is Berlin.",
        "Penguins are flightless aquatic birds native to Antarctica.",
    ]
    response_text = (
        "Berlin is the capital of Germany. "
        "Berlin is the capital of Germany."
    )

    one = _score_lower_level(
        response_text,
        support_texts,
        provider=provider,
        response_chunk_tokens=128,
    )
    many = _score_lower_level(
        response_text,
        support_texts,
        provider=provider,
        response_chunk_tokens=4,
    )

    one_by_id = {unit["support_id"]: unit for unit in one["support_units"]}
    many_by_id = {unit["support_id"]: unit for unit in many["support_units"]}
    assert one_by_id.keys() == many_by_id.keys()
    for support_id in one_by_id:
        assert one_by_id[support_id]["usage_state"] == many_by_id[support_id]["usage_state"]
        assert one_by_id[support_id]["usage_confidence"] == pytest.approx(
            many_by_id[support_id]["usage_confidence"],
            abs=1e-5,
        )
        assert one_by_id[support_id]["unused_confidence"] == pytest.approx(
            many_by_id[support_id]["unused_confidence"],
            abs=1e-5,
        )

    assert one["scores"]["support_units_usage_used"] == many["scores"]["support_units_usage_used"]
    assert one["scores"]["support_units_unused"] == many["scores"]["support_units_unused"]
    assert one["scores"]["support_units_uncertain"] == many["scores"]["support_units_uncertain"]


def test_nli_can_rescue_low_overlap_unit_and_keeps_support_mapping() -> None:
    support_text = "Sales rose versus the prior year."
    response_text = "Revenue climbed compared with earlier periods."
    distractor_text = "Lanterns glow above the quiet harbor at dusk."
    nli_provider = _MappingNLIProvider({(support_text, response_text)})
    nli_reranker = _MappingReranker({(response_text, support_text)})
    result = _score_lower_level(
        response_text,
        [support_text, distractor_text],
        nli_provider=nli_provider,
        nli_reranker=nli_reranker,
    )

    assert result["support_units"][0]["usage_state"] == "used"
    assert result["support_units"][0]["used"] is False
    assert result["support_units"][1]["usage_state"] == "unused"
    assert result["nli_diagnostics"] is not None
    assert result["nli_diagnostics"]["claims"]
    assert result["nli_diagnostics"]["claims"][0]["support_unit_indices"] == [0]
    assert result["nli_diagnostics"]["claims"][0]["support_ids"] == ["unit-0"]


def test_service_marks_mixed_raw_context_window_uncertain() -> None:
    provider = _OrthoStubProvider()
    service = _build_service(provider)
    request = GroundednessRequest(
        raw_context=(
            "Berlin is the capital of Germany. "
            "The museum district includes winding pedestrian streets, weekend markets, "
            "seasonal river cruises, restored observatories, harbor viewpoints, and "
            "many unrelated travel details that the response never mentions."
        ),
        response_text="Berlin is the capital of Germany.",
        raw_context_chunk_tokens=256,
    )

    result = service.groundedness(request)

    assert len(result.support_units) == 1
    assert result.support_units[0].usage_state == GroundednessUsageState.UNCERTAIN
    assert result.scores.support_units_uncertain == 1


def test_multi_support_batch_emits_tristate_fields() -> None:
    """Tri-state classifier must fire on the multi-support-batch path too.

    Partitions 70 synthetic support units across two batches (batch_size=64)
    and asserts the per-unit and aggregate unused-context fields are present
    and consistent with the single-batch contract.
    """

    provider = _OrthoStubProvider()

    target_text = "Berlin is the capital of Germany."
    filler_texts = [
        f"Filler sentence number {idx} about unrelated trivia." for idx in range(69)
    ]
    support_texts = [target_text] + filler_texts

    support_units = _make_support_units_from_texts(provider, support_texts)
    support_batches = partition_support_units(support_units, batch_size=64)
    assert len(support_batches) >= 2, "test requires at least two support batches"

    response_chunks = _build_response_chunks(
        target_text,
        provider=provider,
        chunk_token_budget=128,
        encode_fn=encode_texts,
    )

    result = score_groundedness_response_chunked(
        response_chunks=response_chunks,
        support_batches=support_batches,
        response_text=target_text,
        evidence_limit=8,
        primary_metric="reverse_context",
        coverage_threshold=0.5,
    )

    scores = result["scores"]
    for key in (
        "support_units_usage_used",
        "support_units_unused",
        "support_units_uncertain",
        "context_usage_ratio",
        "context_unused_ratio",
        "context_uncertain_ratio",
    ):
        assert key in scores, f"missing aggregate field: {key}"

    total_units = len(result["support_units"])
    assert total_units == len(support_texts)
    for unit in result["support_units"]:
        assert "usage_state" in unit
        assert unit["usage_state"] in {"used", "unused", "uncertain"}
        assert unit.get("usage_confidence") is not None
        assert unit.get("unused_confidence") is not None

    counts_sum = (
        int(scores["support_units_usage_used"])
        + int(scores["support_units_unused"])
        + int(scores["support_units_uncertain"])
    )
    assert counts_sum == total_units

    by_support_id = {unit["support_id"]: unit for unit in result["support_units"]}
    target_unit = by_support_id["unit-0"]
    assert target_unit["usage_state"] == "used"

    unused_siblings = [
        unit["usage_state"] for unit in result["support_units"] if unit["support_id"] != "unit-0"
    ]
    assert unused_siblings.count("unused") >= 1


def test_vectorized_redundancy_matches_scalar() -> None:
    """`_compute_redundancy_matrix` must agree with the scalar per-pair helper."""

    thresholds = UsageClassifierThresholds()

    dim = 4
    centroid_a = torch.nn.functional.normalize(
        torch.tensor([1.0, 0.0, 0.0, 0.0]), dim=-1
    )
    centroid_b = torch.nn.functional.normalize(
        torch.tensor([0.98, 0.19, 0.0, 0.0]), dim=-1
    )
    centroid_c = torch.nn.functional.normalize(
        torch.tensor([0.0, 1.0, 0.0, 0.0]), dim=-1
    )
    centroid_d = torch.nn.functional.normalize(
        torch.tensor([0.0, 0.0, 1.0, 0.0]), dim=-1
    )
    assert centroid_a.shape[-1] == dim

    signatures = [
        "chunk:alpha",
        "chunk:alpha",
        "chunk:beta",
        "chunk:gamma",
        "chunk:delta",
    ]
    token_sets: list[set[str]] = [
        {"berlin", "capital", "germany"},
        {"berlin", "capital", "germany", "country"},
        {"berlin", "capital", "germany", "city"},
        {"berlin", "city"},
        set(),
    ]
    centroids = [centroid_a, centroid_a, centroid_b, centroid_c, centroid_d]

    matrix = _compute_redundancy_matrix(
        signatures=signatures,
        token_sets=token_sets,
        centroids=centroids,
        thresholds=thresholds,
    )

    u = len(signatures)
    assert matrix.shape == (u, u)
    for i in range(u):
        for j in range(u):
            expected = _usage_redundancy_similarity(
                signature_a=signatures[i],
                signature_b=signatures[j],
                tokens_a=token_sets[i],
                tokens_b=token_sets[j],
                centroid_a=centroids[i],
                centroid_b=centroids[j],
                thresholds=thresholds,
            )
            assert float(matrix[i, j].item()) == pytest.approx(expected, abs=1e-5), (
                f"pair ({i},{j}) vectorised={float(matrix[i, j].item())} scalar={expected}"
            )

    assert float(matrix[0, 1].item()) == pytest.approx(1.0)
    assert float(matrix[0, 4].item()) == pytest.approx(0.0)
    assert float(matrix[4, 4].item()) == pytest.approx(1.0)
