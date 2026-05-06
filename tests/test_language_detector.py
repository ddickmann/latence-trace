"""Tests for ``latence_trace.core.language_detector``.

The detector is binary by design (de or en), defensive on bad input, and
fast enough to run on every grounding request without breaking the
latency budget. These tests pin all four properties.
"""

from __future__ import annotations

import time

import pytest

from latence_trace.core.language_detector import is_german, resolve_language


# --------------------------------------------------------------------------- #
# Positive / negative classification
# --------------------------------------------------------------------------- #


_GERMAN_PROBE = (
    "Karl Roßmann ist der Held des Romanfragments Der Verschollene von Franz "
    "Kafka, das von Brod unter dem Titel Amerika veröffentlicht wurde."
)

_ENGLISH_PROBE = (
    "Karl Rossmann is the protagonist of the novel fragment The Missing One "
    "by Franz Kafka, which was published by Brod under the title America."
)


def test_is_german_positive_on_kafka_passage() -> None:
    assert is_german(_GERMAN_PROBE) is True


def test_is_german_negative_on_english_passage() -> None:
    assert is_german(_ENGLISH_PROBE) is False


def test_is_german_negative_on_french_passage() -> None:
    # We support exactly de + en. Everything else must resolve to False so
    # callers fall back to the English path rather than misrouting.
    french = (
        "Karl Rossmann est le héros du roman fragmentaire L'Amérique, écrit "
        "par Franz Kafka et publié à titre posthume par Max Brod."
    )
    assert is_german(french) is False


# --------------------------------------------------------------------------- #
# Defensive paths
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("probe", ["", "   ", "\n\t  \n"])
def test_is_german_empty_or_whitespace_returns_false(probe: str) -> None:
    assert is_german(probe) is False


@pytest.mark.parametrize("probe", ["OK", "Ja", "Nein", "Hi", "x" * 5])
def test_is_german_too_short_returns_false(probe: str) -> None:
    # Below the 30-char floor langdetect mis-classifies short strings
    # confidently. We refuse to commit to True.
    assert is_german(probe) is False


# --------------------------------------------------------------------------- #
# Determinism (seeded DetectorFactory)
# --------------------------------------------------------------------------- #


def test_is_german_is_deterministic_over_100_iterations() -> None:
    results = {is_german(_GERMAN_PROBE) for _ in range(100)}
    assert results == {True}, f"Detector flapped across iterations: {results}"

    english_results = {is_german(_ENGLISH_PROBE) for _ in range(100)}
    assert english_results == {False}, f"English detection flapped: {english_results}"


# --------------------------------------------------------------------------- #
# Latency
# --------------------------------------------------------------------------- #


def test_is_german_latency_is_acceptable_on_200_char_probe() -> None:
    # Empirically a 200-char probe runs ~3 ms on commodity CPUs. We allow
    # generous headroom (10 ms / call mean) so CI noise doesn't cause
    # spurious failures, but assert hard so a regression to the
    # multi-second category would be caught immediately.
    probe = (_GERMAN_PROBE * 4)[:200]
    n = 1000
    started = time.perf_counter()
    for _ in range(n):
        is_german(probe)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    per_call_ms = elapsed_ms / n
    assert per_call_ms < 10.0, (
        f"is_german latency regression: {per_call_ms:.3f} ms/call > 10 ms"
    )


# --------------------------------------------------------------------------- #
# resolve_language wrapper
# --------------------------------------------------------------------------- #


def test_resolve_language_explicit_de_short_circuits() -> None:
    lang, source = resolve_language(
        explicit="de",
        response_text=_ENGLISH_PROBE,  # would auto-detect en
        query_text=None,
        raw_context=None,
    )
    assert (lang, source) == ("de", "request")


def test_resolve_language_explicit_en_short_circuits() -> None:
    lang, source = resolve_language(
        explicit="en",
        response_text=_GERMAN_PROBE,  # would auto-detect de
        query_text=None,
        raw_context=None,
    )
    assert (lang, source) == ("en", "request")


def test_resolve_language_auto_detects_de_from_response() -> None:
    lang, source = resolve_language(
        explicit=None,
        response_text=_GERMAN_PROBE,
        query_text=None,
        raw_context=None,
    )
    assert (lang, source) == ("de", "auto")


def test_resolve_language_auto_detects_en_from_response() -> None:
    lang, source = resolve_language(
        explicit=None,
        response_text=_ENGLISH_PROBE,
        query_text=None,
        raw_context=None,
    )
    assert (lang, source) == ("en", "auto")


def test_resolve_language_falls_back_to_query_then_raw_context() -> None:
    lang, source = resolve_language(
        explicit=None,
        response_text=None,
        query_text=_GERMAN_PROBE,
        raw_context=None,
    )
    assert (lang, source) == ("de", "auto")

    lang, source = resolve_language(
        explicit=None,
        response_text=None,
        query_text=None,
        raw_context=_GERMAN_PROBE,
    )
    assert (lang, source) == ("de", "auto")


def test_resolve_language_no_usable_probe_returns_fallback_en() -> None:
    lang, source = resolve_language(
        explicit=None,
        response_text=None,
        query_text=None,
        raw_context=None,
    )
    assert (lang, source) == ("en", "fallback_en")

    lang, source = resolve_language(
        explicit=None,
        response_text="",
        query_text="   ",
        raw_context="\n",
    )
    assert (lang, source) == ("en", "fallback_en")


def test_resolve_language_invalid_explicit_falls_through_to_auto() -> None:
    # Schema-level validation rejects unknown languages, but the helper
    # itself must not crash; it should treat anything other than "de"/"en"
    # as if no explicit value were given.
    lang, source = resolve_language(
        explicit="auto",
        response_text=_GERMAN_PROBE,
        query_text=None,
        raw_context=None,
    )
    assert lang == "de"
    assert source == "auto"
