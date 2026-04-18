"""German-language smoke tests for the multilingual Latence Trace pipeline.

These tests cover the new bilingual surface in core modules without
loading any heavy ML model. They verify that:

- the German function-word stopword union recognizes German fillers,
- the literal extractor lifts dates/numbers/measurements/percentages in
  German formatting and that the German formats normalize to the same
  canonical decimal as their English equivalents,
- atomic-claim splitting honors German conjunctions ("und", "aber"),
- the encoder default + dtype env knobs resolve to the multilingual
  SauerkrautLM ColBERT model loaded in bf16,
- the NLI default switches to mDeBERTa multilingual when the feature
  flag is enabled,
- a German request scored end-to-end against a deterministic in-process
  encoder ranks the supported response above the unsupported one for
  date/entity/number/negation strata.
"""

from __future__ import annotations

import os
import re
from typing import List

import numpy as np
import pytest
import torch

from latence_trace.api.service import (
    DEFAULT_GROUNDEDNESS_MODEL,
    DEFAULT_GROUNDEDNESS_TORCH_DTYPE,
    _resolve_torch_dtype,
)
from latence_trace.core.claims import (
    _COORD_PATTERN,
    _looks_german,
    decompose_sentence_into_atoms,
)
from latence_trace.core.groundedness import (
    SupportUnitInput,
    _STOPWORDS,
    _canonicalize_decimal,
    default_null_bank_texts,
    diff_literals,
    extract_literals,
    score_groundedness,
)
from latence_trace.core import nli as nli_module


# ---------------------------------------------------------------------------
# Shared deterministic test encoder
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


class _DeterministicLemmaEncoder:
    """Lemmatized hashed bag-of-vectors encoder for German + English."""

    def __init__(self, dim: int = 32) -> None:
        self.dim = dim
        self._aliases = {
            # German lemmas / common surface forms collapse to one bucket so
            # paraphrase-style support gets recognized.
            "hauptstadt": "capital_city",
            "haupt": "capital_city",
            "kapitale": "capital_city",
            "deutschlands": "germany",
            "deutschland": "germany",
            "berlin": "berlin",
            "frankfurt": "frankfurt",
            "muenchen": "munich",
            "münchen": "munich",
            "geboren": "born_event",
            "geburt": "born_event",
            "born": "born_event",
            "1980": "year_1980",
            "1981": "year_1981",
            "20.07.1981": "date_19810720",
            "1.234,56": "number_1234p56",
            "1234.56": "number_1234p56",
            "eur": "currency_eur",
            "€": "currency_eur",
            "kilometer": "unit_km",
            "km": "unit_km",
            "ist": "copula_is",
            "is": "copula_is",
            "nicht": "negation_not",
            "not": "negation_not",
            "wurde": "passive_was",
            "was": "passive_was",
        }
        self._rng_seed = 17

    def _vec(self, key: str) -> np.ndarray:
        h = abs(hash((self._rng_seed, key))) % (2**31 - 1)
        rng = np.random.default_rng(h)
        v = rng.standard_normal(self.dim).astype(np.float32)
        norm = float(np.linalg.norm(v))
        return v / norm if norm > 1e-9 else v

    def _lemma(self, token: str) -> str:
        t = token.lower().strip()
        if not t:
            return ""
        return self._aliases.get(t, t)

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        out: List[np.ndarray] = []
        for text in texts:
            tokens = [t for t in _TOKEN_RE.findall(text) if t.strip()]
            if not tokens:
                out.append(np.zeros((1, self.dim), dtype=np.float32))
                continue
            vecs = np.stack([self._vec(self._lemma(tok)) for tok in tokens], axis=0)
            out.append(vecs)
        return out

    def tokenize(self, text):
        return [t for t in _TOKEN_RE.findall(text) if t.strip()]


def _support_unit(text: str, encoder: _DeterministicLemmaEncoder) -> SupportUnitInput:
    embeddings = encoder.encode(text)[0]
    tokens = encoder.tokenize(text)
    return SupportUnitInput(
        support_id="u1",
        text=text,
        embeddings=torch.from_numpy(embeddings),
        tokens=tokens,
        chunk_id="c1",
    )


def _score(
    support_text: str,
    response_text: str,
    encoder: _DeterministicLemmaEncoder,
):
    unit = _support_unit(support_text, encoder)
    response_embeddings = torch.from_numpy(encoder.encode(response_text)[0])
    response_tokens = encoder.tokenize(response_text)
    return score_groundedness(
        support_units=[unit],
        response_embeddings=response_embeddings,
        response_tokens=response_tokens,
        response_text=response_text,
    )


# ---------------------------------------------------------------------------
# 1. Static config: encoder default, dtype, NLI default, null bank
# ---------------------------------------------------------------------------


def test_default_encoder_is_multilingual_sauerkraut_modern_colbert() -> None:
    assert DEFAULT_GROUNDEDNESS_MODEL == (
        "VAGOsolutions/SauerkrautLM-Multi-Reason-ModernColBERT"
    )


def test_default_torch_dtype_is_bfloat16_and_resolver_maps_correctly() -> None:
    assert DEFAULT_GROUNDEDNESS_TORCH_DTYPE == "bfloat16"
    assert _resolve_torch_dtype("bfloat16") is torch.bfloat16
    assert _resolve_torch_dtype("bf16") is torch.bfloat16
    assert _resolve_torch_dtype("fp16") is torch.float16
    assert _resolve_torch_dtype("float32") is torch.float32
    assert _resolve_torch_dtype("default") is None
    assert _resolve_torch_dtype(None) is None


def test_default_nli_model_is_multilingual_mdeberta(monkeypatch) -> None:
    monkeypatch.setenv("VOYAGER_GROUNDEDNESS_NLI_ENABLED", "1")
    monkeypatch.delenv("VOYAGER_GROUNDEDNESS_NLI_MODEL", raising=False)

    captured = {}

    class _StubProvider:
        def __init__(self, model_id, *, max_length):
            captured["model_id"] = model_id
            captured["max_length"] = max_length

    monkeypatch.setattr(nli_module, "HuggingFaceNLIProvider", _StubProvider)
    provider = nli_module.resolve_default_provider()
    assert provider is not None
    assert captured["model_id"] == (
        "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
    )


def test_default_null_bank_is_bilingual() -> None:
    bank = default_null_bank_texts()
    has_umlaut = sum(
        1 for sentence in bank if any(ch in sentence for ch in "äöüß")
    )
    assert len(bank) >= 24
    assert has_umlaut >= 8


# ---------------------------------------------------------------------------
# 2. Stopwords + token weighting
# ---------------------------------------------------------------------------


def test_german_function_words_are_in_stopword_set() -> None:
    for word in (
        "der",
        "die",
        "das",
        "und",
        "ist",
        "mit",
        "nicht",
        "von",
        "zu",
        "auf",
    ):
        assert word in _STOPWORDS, word


def test_english_function_words_remain_in_stopword_set() -> None:
    for word in ("the", "and", "is", "of"):
        assert word in _STOPWORDS, word


# ---------------------------------------------------------------------------
# 3. Literal extraction + cross-language normalization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected_kind, expected_value",
    [
        ("am 20.07.1981 erschienen", "date", "20.07.1981"),
        ("am 3. Oktober 1990", "date", "3. Oktober 1990"),
        ("am 20 July 1981", "date", "20 July 1981"),
        ("Preis 1.234,56 EUR", "currency", "1.234,56 EUR"),
        ("Steuer 20,5%", "percent", "20,5%"),
        ("Strecke 42,5 km", "measurement", "42,5 km"),
        ("Strecke 26.2 miles", "measurement", "26.2 miles"),
    ],
)
def test_literal_extractor_handles_german_and_english_formats(
    text: str, expected_kind: str, expected_value: str
) -> None:
    literals = extract_literals(text)
    found = [(literal["kind"], literal["value"]) for literal in literals]
    assert (expected_kind, expected_value) in found


@pytest.mark.parametrize(
    "value_a, value_b",
    [
        ("1.234,56", "1234.56"),
        ("1.234.567", "1234567"),
        ("42,5", "42.5"),
    ],
)
def test_canonical_decimal_collides_german_and_english(
    value_a: str, value_b: str
) -> None:
    assert _canonicalize_decimal(value_a) == _canonicalize_decimal(value_b)


def test_diff_literals_matches_german_date_and_number_against_german_support() -> None:
    response = "Teardrops erschien am 20.07.1981 und kostete 1.234,56 EUR."
    support_text = (
        "Die Single Teardrops wurde am 20.07.1981 ver\u00f6ffentlicht. "
        "Der Listenpreis betrug 1.234,56 EUR."
    )
    encoder = _DeterministicLemmaEncoder()
    unit = _support_unit(support_text, encoder)
    response_literals, mismatches, matches = diff_literals(response, [unit])
    assert response_literals, "extractor failed to surface any literal"
    assert not mismatches, f"unexpected mismatches: {mismatches}"
    matched_kinds = {match["kind"] for match in matches}
    assert "date" in matched_kinds
    assert "currency" in matched_kinds


def test_diff_literals_flags_german_date_disagreement() -> None:
    response = "Teardrops erschien am 20.07.1982."
    support_text = "Die Single Teardrops wurde am 20.07.1981 ver\u00f6ffentlicht."
    encoder = _DeterministicLemmaEncoder()
    unit = _support_unit(support_text, encoder)
    _, mismatches, matches = diff_literals(response, [unit])
    mismatch_kinds = {item["kind"] for item in mismatches}
    assert "date" in mismatch_kinds, mismatches


# ---------------------------------------------------------------------------
# 4. Conjunction split + language detection
# ---------------------------------------------------------------------------


def test_coord_pattern_matches_german_und_aber_sondern() -> None:
    for sentence in (
        "Goethe schrieb Faust und Schiller verfasste Wallenstein.",
        "Sie wollte gehen, aber er blieb noch eine Stunde.",
        "Es regnete nicht, sondern es schneite leicht.",
    ):
        assert _COORD_PATTERN.search(sentence) is not None, sentence


def test_looks_german_distinguishes_languages() -> None:
    assert _looks_german("Paris ist die Hauptstadt von Frankreich.") is True
    assert _looks_german("Paris is the capital of France.") is False
    # Single umlaut is a strong enough signal on its own.
    assert _looks_german("Wir machen Urlaub in Mu\u0308nchen.") is True


def test_decompose_sentence_splits_german_compound_sentence() -> None:
    sentence = "Goethe schrieb Faust und Schiller verfasste Wallenstein."
    atoms = decompose_sentence_into_atoms(sentence, parent_index=0, parent_start=0)
    assert len(atoms) >= 2, f"expected >=2 atoms, got: {[a.text for a in atoms]}"


# ---------------------------------------------------------------------------
# 5. End-to-end: German support vs response, deterministic encoder
# ---------------------------------------------------------------------------


def _calibrated(scored) -> float:
    return float(
        scored["scores"].get("reverse_context_calibrated")
        or scored["scores"]["reverse_context"]
    )


def test_german_supported_response_outscores_unsupported_paraphrase() -> None:
    encoder = _DeterministicLemmaEncoder()
    support = "Berlin ist die Hauptstadt Deutschlands seit 1990."
    supported_response = "Die Hauptstadt Deutschlands ist Berlin."
    unsupported_response = "Die Hauptstadt Deutschlands ist Mu\u0308nchen."

    supported = _calibrated(_score(support, supported_response, encoder))
    unsupported = _calibrated(_score(support, unsupported_response, encoder))
    assert supported > unsupported, (supported, unsupported)


def test_german_negation_drops_score_versus_affirmation() -> None:
    encoder = _DeterministicLemmaEncoder()
    support = "Goethe wurde in Frankfurt geboren."
    affirmation = "Goethe wurde in Frankfurt geboren."
    negation = "Goethe wurde nicht in Frankfurt geboren."

    affirmed = _calibrated(_score(support, affirmation, encoder))
    negated = _calibrated(_score(support, negation, encoder))
    assert affirmed >= negated, (affirmed, negated)


def test_german_date_swap_is_caught_by_literal_guardrails() -> None:
    encoder = _DeterministicLemmaEncoder()
    support = (
        "Die Single Teardrops wurde am 20.07.1981 in den USA ver\u00f6ffentlicht."
    )
    correct = "Teardrops erschien am 20.07.1981 in den USA."
    wrong = "Teardrops erschien am 20.07.1982 in den USA."
    correct_scored = _score(support, correct, encoder)
    wrong_scored = _score(support, wrong, encoder)

    assert wrong_scored["scores"]["literal_mismatch_count"] >= 1
    assert correct_scored["scores"]["literal_mismatch_count"] == 0
    assert (
        wrong_scored["scores"]["literal_guarded"]
        < correct_scored["scores"]["literal_guarded"]
    )


def test_german_number_swap_is_caught_by_literal_guardrails() -> None:
    encoder = _DeterministicLemmaEncoder()
    support = "Der Bestellpreis betrug 1.234,56 EUR pro Einheit."
    correct = "Der Preis lag bei 1.234,56 EUR."
    wrong = "Der Preis lag bei 1.234,57 EUR."
    correct_scored = _score(support, correct, encoder)
    wrong_scored = _score(support, wrong, encoder)

    assert wrong_scored["scores"]["literal_mismatch_count"] >= 1
    assert correct_scored["scores"]["literal_mismatch_count"] == 0


def test_english_minimal_pair_still_works_after_multilingual_changes() -> None:
    """Regression guard: the English path must not have regressed."""

    encoder = _DeterministicLemmaEncoder()
    support = "Berlin is the capital of Germany since 1990."
    supported = "The capital of Germany is Berlin."
    unsupported = "The capital of Germany is Munich."

    a = _calibrated(_score(support, supported, encoder))
    b = _calibrated(_score(support, unsupported, encoder))
    assert a > b, (a, b)


# ---------------------------------------------------------------------------
# 6. Audit-driven regressions: claim splitter + lexical rerank stopwords
# ---------------------------------------------------------------------------


def test_claim_splitter_refines_long_german_sentences_on_aber_jedoch() -> None:
    """``split_claims`` must split German conjunctions on long sentences.

    Pre-fix the conjunction regex only listed English (``but``/``however``/
    ``whereas``); long German prose sailed through as one giant claim and
    blew up NLI premise selection. The refined splitter now mirrors the
    atomic-claim coordinator regex.
    """

    long_de = (
        "Goethe wurde 1749 in Frankfurt am Main geboren und studierte sp\u00e4ter "
        "in Leipzig sowie in Stra\u00dfburg, aber er kehrte regelm\u00e4\u00dfig "
        "in seine Heimatstadt zur\u00fcck und reiste mehrfach nach Italien, "
        "jedoch siedelte er schlie\u00dflich nach Weimar \u00fcber, wo er "
        "vierzig Jahre lang im Dienst des herzoglichen Hofes blieb"
    )
    claims = nli_module.split_claims(long_de)
    # Without German conjunctions in the splitter, the whole 299-char run
    # collapsed to a single claim. The fix yields >=3 sub-claims, with
    # the German conjunctions ``aber``/``jedoch`` consumed as delimiters
    # (so the resulting sub-claims start with the post-conjunction tail).
    assert len(claims) >= 3, [c.text for c in claims]
    starts = [c.text.split()[0].lower() for c in claims]
    # The sub-claim immediately after ``aber`` starts with ``er``;
    # the sub-claim after ``jedoch`` starts with ``siedelte``.
    assert "er" in starts, starts
    assert "siedelte" in starts, starts
    # And no surviving sub-claim begins with the German conjunction itself.
    for token in ("aber", "jedoch", "w\u00e4hrend", "sondern", "doch"):
        assert token not in starts, (token, starts)


def test_lexical_overlap_filters_german_function_words() -> None:
    """``_content_set`` must skip German function words for premise rerank."""

    content_set = nli_module._content_set
    bag = content_set("der Hund und die Katze sind in dem Garten")
    # All shown words are German function words and must NOT survive the
    # filter; only ``hund``, ``katze``, and ``garten`` remain.
    assert bag == {"hund", "katze", "garten"}, bag


def test_lexical_overlap_still_drops_english_function_words() -> None:
    content_set = nli_module._content_set
    bag = content_set("the cat and the dog were in the garden")
    assert bag == {"cat", "dog", "garden"}, bag


def test_de_months_alternation_has_no_duplicates() -> None:
    """Audit fix: ``_DE_MONTHS`` was emitting duplicate ``Mai`` and ``Apr``.

    A duplicate alternation never breaks correctness but wastes regex
    bytecode and matches confidence diagnostics. This guard keeps the
    pattern minimal so future edits do not silently re-introduce the dup.
    """

    from latence_trace.core.groundedness import _DE_MONTHS

    parts = _DE_MONTHS.strip("()?:").rstrip(")").lstrip("(?:").split("|")
    assert len(parts) == len(set(parts)), [p for p in parts if parts.count(p) > 1]


def test_german_stopword_set_has_no_duplicates() -> None:
    from latence_trace.core.groundedness import _STOPWORDS

    # ``_STOPWORDS`` is a set, so duplicates collapse silently. The audit
    # found ``"ihr"`` listed twice in the source; we keep this guard so
    # future edits are forced to be tidy. The list is regenerated from the
    # source via ``ast`` to detect duplicates explicitly.
    import ast
    import inspect
    from latence_trace.core import groundedness as ground_mod

    source = inspect.getsource(ground_mod)
    tree = ast.parse(source)
    set_node = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "_STOPWORDS"
    )
    literal_words = [el.value for el in set_node.value.elts if isinstance(el, ast.Constant)]
    duplicates = [w for w in literal_words if literal_words.count(w) > 1]
    assert not duplicates, sorted(set(duplicates))
    # And the runtime set still carries the German closed-class words.
    for must_have in ("der", "die", "das", "und", "ist", "nicht"):
        assert must_have in _STOPWORDS
