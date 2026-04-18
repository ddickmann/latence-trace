"""K1 + K2 tests: attribution_mode refusal and structured support_units lane.

K1 (refuse-to-score for empty premise) and K2 (structured ``support_units[]``
premise lane) are the two narrow API additions the standalone latence-trace
service ships in v1. These tests exercise the surfaces:

K1:
- ``closed_book`` + zero premises returns risk_band="unknown" /
  reason="no_premise_supplied" instead of inventing a score.
- ``open_domain`` returns risk_band="unsupported" /
  reason="open_domain_pending_v1_next" as a forward-compat schema gate
  for the K5 retrieval-callback lane that ships post-v1.

K2:
- A request that supplies a list of structured ``support_units`` (each with
  text + optional source_id/speaker/timestamp/metadata) ends up scored end-
  to-end and every response support unit echoes the matching attribution
  fields verbatim so callers can show "this answer was grounded in turn 4
  by Dr. X".
- Validator rejects multiple premise lanes simultaneously.
- Backward compatibility: a ``raw_context`` request still works and its
  attribution fields stay ``None``.
"""

from __future__ import annotations

import re
from typing import List

import numpy as np

from latence_trace.api.models import (
    AttributionMode,
    GroundednessRequest,
    GroundednessSupportUnitInput,
)
from latence_trace.api.service import GroundednessService

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


class _DeterministicEncoder:
    """Minimal in-process encoder used to drive the service end-to-end.

    Avoids loading any HF model so the test suite stays CPU-cheap. The
    encode path returns a deterministic per-token vector keyed off the
    lower-cased token + a fixed seed, so identical tokens collapse to
    identical vectors and MaxSim picks them up as exact matches.
    """

    model_name = "deterministic-test-encoder"
    dim = 24
    _rng_seed = 7

    def _vec(self, token: str) -> np.ndarray:
        h = abs(hash((self._rng_seed, token.lower()))) % (2**31 - 1)
        rng = np.random.default_rng(h)
        v = rng.standard_normal(self.dim).astype(np.float32)
        norm = float(np.linalg.norm(v))
        return v / norm if norm > 1e-9 else v

    def tokenize(self, text):
        return [t for t in _TOKEN_RE.findall(text) if t.strip()]

    def encode(self, texts, **_kwargs):
        if isinstance(texts, str):
            texts = [texts]
        out: List[np.ndarray] = []
        for text in texts:
            tokens = self.tokenize(text)
            if not tokens:
                out.append(np.zeros((1, self.dim), dtype=np.float32))
                continue
            out.append(np.stack([self._vec(t) for t in tokens], axis=0))
        return out


def _build_service() -> GroundednessService:
    encoder = _DeterministicEncoder()
    return GroundednessService(encoder_factory=lambda _name: encoder)


# ---------------------------------------------------------------------------
# K1 - attribution_mode + refuse-to-score
# ---------------------------------------------------------------------------


def test_closed_book_with_no_premise_refuses_to_score() -> None:
    service = _build_service()
    request = GroundednessRequest(
        response_text="alpha supports claim",
        attribution_mode=AttributionMode.CLOSED_BOOK,
    )

    result = service.groundedness(request)

    assert result.scores.risk_band == "unknown"
    assert result.reason == "no_premise_supplied"
    assert result.attribution_mode == AttributionMode.CLOSED_BOOK
    assert result.mode == "closed_book"
    assert result.support_units == []
    assert result.response_tokens == []
    assert result.warnings  # exactly one explanatory warning


def test_open_domain_returns_unsupported_until_v1_next() -> None:
    service = _build_service()
    request = GroundednessRequest(
        raw_context="alpha supports claim",
        response_text="alpha supports claim",
        attribution_mode=AttributionMode.OPEN_DOMAIN,
    )

    result = service.groundedness(request)

    assert result.scores.risk_band == "unsupported"
    assert result.reason == "open_domain_pending_v1_next"
    assert result.attribution_mode == AttributionMode.OPEN_DOMAIN
    assert result.mode == "open_domain"
    assert result.support_units == []
    assert result.warnings


def test_default_attribution_mode_is_closed_book() -> None:
    request = GroundednessRequest(
        raw_context="alpha supports claim",
        response_text="alpha supports claim",
    )
    assert request.attribution_mode == AttributionMode.CLOSED_BOOK


def test_attribution_mode_round_trips_on_normal_request() -> None:
    service = _build_service()
    request = GroundednessRequest(
        raw_context="alpha supports claim",
        response_text="alpha supports claim",
        attribution_mode=AttributionMode.CLOSED_BOOK,
    )

    result = service.groundedness(request)
    assert result.attribution_mode == AttributionMode.CLOSED_BOOK
    assert result.reason is None
    assert result.scores.risk_band in {"green", "amber", "red"}


# ---------------------------------------------------------------------------
# K2 - structured support_units[] lane
# ---------------------------------------------------------------------------


def test_support_units_lane_scores_end_to_end_and_echoes_attribution() -> None:
    service = _build_service()
    request = GroundednessRequest(
        response_text="alpha supports claim",
        support_units=[
            GroundednessSupportUnitInput(
                text="alpha supports claim",
                source_id="turn-4",
                speaker="Dr. X",
                timestamp="2026-04-17T10:00:00Z",
                metadata={"channel": "voice"},
            ),
            GroundednessSupportUnitInput(
                text="beta unrelated note",
                source_id="turn-5",
                speaker="Dr. Y",
            ),
        ],
    )

    result = service.groundedness(request)

    assert result.mode == "support_units"
    assert result.attribution_mode == AttributionMode.CLOSED_BOOK
    assert result.reason is None
    assert len(result.support_units) == 2

    by_source = {unit.source_id: unit for unit in result.support_units}
    assert by_source["turn-4"].speaker == "Dr. X"
    assert by_source["turn-4"].timestamp == "2026-04-17T10:00:00Z"
    assert by_source["turn-4"].metadata == {"channel": "voice"}
    assert by_source["turn-4"].source_mode == "support_units"
    assert by_source["turn-5"].speaker == "Dr. Y"
    assert by_source["turn-5"].timestamp is None
    assert by_source["turn-5"].metadata is None

    # The matching unit (alpha) should outscore the unrelated unit (beta).
    assert by_source["turn-4"].score > by_source["turn-5"].score


def test_support_units_without_source_id_get_synthetic_ids() -> None:
    service = _build_service()
    request = GroundednessRequest(
        response_text="alpha supports claim",
        support_units=[
            GroundednessSupportUnitInput(text="alpha supports claim"),
            GroundednessSupportUnitInput(text="gamma is irrelevant"),
        ],
    )

    result = service.groundedness(request)
    assert result.mode == "support_units"
    ids = [unit.support_id for unit in result.support_units]
    assert ids == ["support-0", "support-1"]
    for unit in result.support_units:
        assert unit.source_id is None
        assert unit.speaker is None


def test_validator_rejects_multiple_premise_lanes() -> None:
    import pytest

    with pytest.raises(ValueError, match="exactly one of"):
        GroundednessRequest(
            response_text="alpha",
            raw_context="alpha supports claim",
            support_units=[GroundednessSupportUnitInput(text="alpha supports claim")],
        )


def test_validator_rejects_chunk_ids_plus_support_units() -> None:
    import pytest

    with pytest.raises(ValueError, match="exactly one of"):
        GroundednessRequest(
            response_text="alpha",
            chunk_ids=["doc-1"],
            support_units=[GroundednessSupportUnitInput(text="alpha")],
        )


def test_raw_context_payload_keeps_attribution_fields_none() -> None:
    """Backward compatibility: legacy raw_context callers see no schema drift."""

    service = _build_service()
    request = GroundednessRequest(
        raw_context="alpha supports claim",
        response_text="alpha supports claim",
    )

    result = service.groundedness(request)
    assert result.mode == "raw_context"
    assert result.support_units
    for unit in result.support_units:
        assert unit.source_id is None
        assert unit.speaker is None
        assert unit.timestamp is None
        assert unit.metadata is None
        assert unit.source_mode == "raw_context"
