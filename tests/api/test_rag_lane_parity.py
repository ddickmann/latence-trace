"""RAG-lane parity regression guard.

This suite replays a curated, deterministic fixture of RAG requests
against :class:`~latence_trace.api.service.GroundednessService` and
diffs the output against a committed golden snapshot
(:mod:`tests/api/fixtures/rag_lane_parity_golden.json`).

Why we care
-----------
The code-lane quality-boost sprint introduces a second scoring lane
(``scoring_mode == "code"``) alongside the existing RAG lane. Customers
running production traffic through the RAG lane must see **bitwise
identical** scores and ``top_evidence`` before and after the sprint. A
drift of even 1 ulp on ``reverse_context`` would invalidate downstream
quantile monitors, alert thresholds, and any calibration layer stacked
on top of the service.

How the guard works
-------------------
- The fixture is seeded with a :class:`_DeterministicEncoder` so the
  embeddings are reproducible regardless of where the test runs.
- ``VOYAGER_GROUNDEDNESS_NLI_ENABLED`` is forced to ``"0"`` — the NLI
  path talks to an external vLLM server and is out of scope for a
  byte-parity check.
- The golden snapshot contains exact score fields and the
  ``top_evidence`` ordering. If the RAG pipeline changes any of those,
  the diff surfaces the mismatch in the CI log.
- If you intentionally change the RAG scoring output (e.g. calibration
  bump), set ``LATENCE_TRACE_REGENERATE_PARITY=1`` and rerun the test
  once; it will rewrite the fixture. Commit the diff with a clear
  message so reviewers can audit the baseline move.

This test is deliberately tiny and synchronous. It runs in under a
second on CPU so CI can gate every merge on RAG parity without paying
GPU cost.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable, Sequence

import pytest
import torch

from latence_trace.api.models import (
    GroundednessRequest,
    GroundednessSupportUnitInput,
    ScoringMode,
)
from latence_trace.api.service import GroundednessService


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "rag_lane_parity_golden.json"


class _DeterministicEncoder:
    """Seeded encoder that maps each input string to a stable tensor.

    The seed is derived from a FNV-1a hash of the sentence so the same
    input always produces the same embedding, regardless of Python's
    randomised hash seed.
    """

    model_name = "rag-parity-encoder"
    model_name_or_path = "rag-parity-encoder"

    def __init__(self, dim: int = 32) -> None:
        self._dim = dim

    @staticmethod
    def _stable_seed(text: str) -> int:
        # FNV-1a 32-bit — deterministic across processes/platforms.
        h = 0x811C9DC5
        for byte in text.encode("utf-8"):
            h ^= byte
            h = (h * 0x01000193) & 0xFFFFFFFF
        return h

    def encode(
        self,
        sentences: Sequence[str],
        *,
        is_query: bool = False,  # noqa: ARG002 - signature parity
        prompt_name: str | None = None,  # noqa: ARG002
        **_: object,
    ) -> list[torch.Tensor]:
        out: list[torch.Tensor] = []
        for sentence in sentences:
            seed = self._stable_seed(sentence or "x")
            generator = torch.Generator().manual_seed(seed)
            tokens = max(2, min(8, len((sentence or "x").split())))
            out.append(torch.randn(tokens, self._dim, generator=generator))
        return out


def _requests() -> list[GroundednessRequest]:
    """A small but varied set of RAG cases covering the three lanes.

    1. ``raw_context`` + single-sentence response (classic demo shape).
    2. ``raw_context`` with a drifty response (literal guard should
       dampen the score).
    3. ``support_units`` lane — caller pre-splits context.
    """

    return [
        GroundednessRequest(
            query_text="Where was Heinrich born?",
            raw_context="Heinrich was born in 1851 in Augsburg.",
            response_text="Heinrich was born in Augsburg in 1851.",
            scoring_mode=ScoringMode.RAG,
        ),
        GroundednessRequest(
            query_text="Where was Heinrich born?",
            raw_context="Heinrich was born in 1851 in Augsburg.",
            response_text="Heinrich was born in Paris in 1851.",
            scoring_mode=ScoringMode.RAG,
        ),
        GroundednessRequest(
            query_text="What is the mascot?",
            support_units=[
                GroundednessSupportUnitInput(
                    text="The team's mascot is a penguin named Pip.",
                    source_id="unit-0",
                ),
                GroundednessSupportUnitInput(
                    text="Penguins are flightless aquatic birds.",
                    source_id="unit-1",
                ),
            ],
            response_text="The mascot is a penguin named Pip.",
            scoring_mode=ScoringMode.RAG,
        ),
    ]


def _snapshot(payload: dict) -> dict:
    """Project the raw response dict down to the fields we pin.

    Keeping the snapshot slim avoids false positives when an ancillary
    field changes (e.g. wall-clock ``time_ms``) while still catching
    any drift in scores or evidence ordering.
    """

    scores = payload.get("scores") or {}
    evidence = payload.get("top_evidence") or []
    return {
        "scores": {
            key: scores.get(key)
            for key in (
                "primary_name",
                "primary_score",
                "reverse_context",
                "literal_guarded",
                "support_units_total",
                "support_units_used",
                "context_coverage_ratio",
                "risk_band",
            )
        },
        "top_evidence": [
            {
                "index": ev.get("index"),
                "score": ev.get("score"),
                "text": ev.get("text"),
            }
            for ev in evidence
        ],
    }


def _run_requests(requests: Iterable[GroundednessRequest]) -> list[dict]:
    os.environ["VOYAGER_GROUNDEDNESS_NLI_ENABLED"] = "0"
    torch.manual_seed(0)
    service = GroundednessService(
        encoder_factory=lambda _name: _DeterministicEncoder()
    )
    out: list[dict] = []
    for req in requests:
        response = service.groundedness(req)
        payload = response.model_dump(mode="json")
        out.append(_snapshot(payload))
    return out


def _regenerate_fixture() -> None:
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    snapshot = _run_requests(_requests())
    FIXTURE_PATH.write_text(json.dumps(snapshot, indent=2, sort_keys=True))


def test_rag_lane_parity_against_pre_sprint_snapshot() -> None:
    """Assert RAG scores + evidence match the committed golden snapshot."""

    if os.environ.get("LATENCE_TRACE_REGENERATE_PARITY") == "1":
        _regenerate_fixture()
        pytest.skip(
            "Regenerated RAG parity fixture — commit the diff and rerun "
            "without LATENCE_TRACE_REGENERATE_PARITY to re-activate the guard."
        )

    if not FIXTURE_PATH.exists():
        _regenerate_fixture()
        pytest.fail(
            "Golden RAG parity fixture did not exist; it has been generated "
            f"at {FIXTURE_PATH}. Review and commit it."
        )

    expected = json.loads(FIXTURE_PATH.read_text())
    actual = _run_requests(_requests())

    assert actual == expected, (
        "RAG lane output drifted from the pre-sprint baseline. If the "
        "change is intentional, rerun with "
        "LATENCE_TRACE_REGENERATE_PARITY=1 and commit the diff."
    )
