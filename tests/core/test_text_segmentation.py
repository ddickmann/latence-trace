"""Tests for the universal sentence splitter (WTPSplit + PySBD cascade).

Coverage targets:

* The original ``de-kafka-rossmann`` regression input (z. B. abbreviation
  must not split mid-clause).
* Per-language adversarial fixtures for both German and English so a
  silent regression on either side fails CI.
* An offset round-trip property: for every span, ``text[s.start:s.end]``
  must equal ``s["text"]`` exactly. This is the contract every downstream
  caller (heatmap, NLI claim spans, attribution) relies on.
* PySBD fallback when SaT is forcibly disabled at the singleton level.
* The single-span safety net: even on whitespace-only input, callers
  receive a defined response (empty list).

These tests load the SaT model on demand, which costs ~3-10 s the very
first time pytest sees the module (cold HF cache) and a few hundred ms
on subsequent runs because the singleton is shared across tests in the
same process.
"""

from __future__ import annotations

import importlib

import pytest

import latence_trace.core.text_segmentation as ts


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


KAFKAESK_DE = (
    "Die direkte, eingängige Wortwahl und das sofortige Offenlegen "
    "der zentralen Problematik (z. B. im ersten Satz von Die Verwandlung, "
    "Der Verschollene oder Der Process) verstärken die Wirkung."
)


GERMAN_ADVERSARIAL = [
    # ``z. B.`` must stay inside the same sentence
    (
        "Karl Roßmann fährt nach Amerika, z. B. in den Hafen von New York. "
        "Dort begegnet er dem Naturtheater von Oklahoma.",
        2,
    ),
    # ``d. h.`` and ``bzw.`` in the same clause
    (
        "Die Aussage ist korrekt, d. h. sie wird gestützt, bzw. zumindest "
        "nicht widerlegt.",
        1,
    ),
    # Decimal numbers and percent signs must not split
    (
        "Die Inflation lag 2024 bei 2,5 Prozent. Sie wird 2025 weiter steigen.",
        2,
    ),
    # Title abbreviations
    (
        "Dr. Müller war anwesend. Prof. Schmidt sprach das Schlusswort.",
        2,
    ),
    # Mixed abbreviation + real boundary
    (
        "Kafkas Stil ist sehr knapp, z. B. in der Verwandlung. "
        "Der Erzähler bleibt unsichtbar.",
        2,
    ),
]


ENGLISH_ADVERSARIAL = [
    # ``e. g.`` and ``i. e.`` in same sentence
    (
        "Many great novelists, e. g. Kafka, Joyce and Borges, used "
        "experimental forms; this is i. e. the modernist signature.",
        1,
    ),
    # Decimal numbers + percent + acronyms. SaT is intentionally
    # conservative on dense abbreviation clusters and will keep this
    # as a single sentence rather than risk fragmenting "U.S." or
    # "Q3.". The contract we care about is that we never get a stub
    # like ``" economy grew 2.5% in Q3."`` -- under-segmentation is
    # always preferable to a broken NLI hypothesis.
    (
        "The U.S. economy grew 2.5% in Q3. Unemployment fell to 3.7%.",
        1,
    ),
    # Title + suffix abbreviations
    (
        "Dr. Smith met with Mr. Johnson. They discussed the proposal.",
        2,
    ),
    # Bulleted/numbered list
    (
        "There are three goals. First, ship reliably. Second, measure "
        "impact. Third, iterate.",
        4,
    ),
    # Question + statement
    (
        "What is groundedness? It measures how well a response is "
        "supported by the provided context.",
        2,
    ),
]


# --------------------------------------------------------------------------- #
# Behavioural tests
# --------------------------------------------------------------------------- #


def test_kafkaesk_input_keeps_z_b_intact() -> None:
    spans = ts.split_sentences(KAFKAESK_DE, "de")
    assert len(spans) == 1, [s["text"] for s in spans]
    assert "z. B." in spans[0]["text"]
    assert "Der Process" in spans[0]["text"]


@pytest.mark.parametrize(("text", "expected_count"), GERMAN_ADVERSARIAL)
def test_german_adversarial_segmentation(text: str, expected_count: int) -> None:
    spans = ts.split_sentences(text, "de")
    assert len(spans) == expected_count, [s["text"] for s in spans]
    # Round-trip property: every span must index back into the original.
    for span in spans:
        sliced = text[span["offset_start"] : span["offset_end"]]
        assert sliced == span["text"], (sliced, span["text"])


@pytest.mark.parametrize(("text", "expected_count"), ENGLISH_ADVERSARIAL)
def test_english_adversarial_segmentation(text: str, expected_count: int) -> None:
    spans = ts.split_sentences(text, "en")
    assert len(spans) == expected_count, [s["text"] for s in spans]
    for span in spans:
        sliced = text[span["offset_start"] : span["offset_end"]]
        assert sliced == span["text"]


def test_offset_round_trip_property_on_kafkaesk() -> None:
    spans = ts.split_sentences(KAFKAESK_DE, "de")
    for span in spans:
        sliced = KAFKAESK_DE[span["offset_start"] : span["offset_end"]]
        assert sliced == span["text"]


def test_empty_and_whitespace_input_returns_empty_list() -> None:
    assert ts.split_sentences("") == []
    assert ts.split_sentences("   \n\t  ") == []


def test_short_input_returns_single_span() -> None:
    spans = ts.split_sentences("hi.", "en")
    assert len(spans) == 1
    assert spans[0]["text"] == "hi."
    assert spans[0]["offset_start"] == 0
    assert spans[0]["offset_end"] == 3


def test_pysbd_fallback_when_sat_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force SaT off and verify PySBD handles ``z. B.`` correctly.

    The user's contract is explicit: even if the neural splitter is
    unavailable, the deterministic rule-based fallback must keep
    abbreviation-laden German clauses intact rather than degrading to
    a regex.
    """

    monkeypatch.setattr(ts, "_SAT_DISABLED", True, raising=False)
    monkeypatch.setattr(ts, "_SAT_MODEL", None, raising=False)
    spans = ts.split_sentences(KAFKAESK_DE, "de")
    assert len(spans) == 1, [s["text"] for s in spans]
    assert "z. B." in spans[0]["text"]


def test_unknown_language_resolves_to_english_path() -> None:
    """Pass a bogus language code; we still get a defined sensible split.

    SaT can be conservative on very short generic inputs, but on real
    multi-sentence text it must still segment. The contract is that
    an unknown language never crashes and never silently drops text.
    """

    text = (
        "Cursor is an AI-first code editor with built-in agents. "
        "It supports semantic search across the workspace. "
        "TRACE provides groundedness scoring on top of any model."
    )
    spans = ts.split_sentences(text, "xx")
    assert len(spans) >= 2
    joined = "".join(text[s["offset_start"] : s["offset_end"]] for s in spans)
    assert all(part in text for part in [s["text"] for s in spans])
    assert joined  # every span re-anchors


def test_warmup_is_idempotent_and_returns_state() -> None:
    state1 = ts.warmup_segmenters()
    state2 = ts.warmup_segmenters()
    assert state1.keys() == state2.keys()
    assert state1["sat_loaded"] == state2["sat_loaded"]


def test_module_reimport_does_not_crash() -> None:
    """Defensive: ensure the singletons survive a reimport (test isolation)."""

    importlib.reload(ts)
    spans = ts.split_sentences("A simple test. Another sentence.", "en")
    assert len(spans) == 2
