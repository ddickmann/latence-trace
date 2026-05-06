"""Typed-claim extractor for the structured-evidence lane.

For prose-formatted tables (e.g. ``"Apple Inc. revenue by reportable
segment, Q3 FY2024, in millions: Americas $37,678; Europe $21,883;
Greater China $14,728; ..."``), surface-level NLI and dense
similarity over-trigger on lexical / entity overlap and let cell-level
factual mismatches slip through. The structured-evidence lane fixes
that by turning each numeric / directional sentence in the response
into a *typed claim* which is then matched against typed source cells
under an AND-gate (entity, value, unit, sign).

This module owns the response-side extraction:

- Sentence segmentation (locale-agnostic; punctuation rules cover EN
  and DE).
- Per-sentence numeric / direction-verb scan with regex + light token
  bookkeeping.
- Anchor extraction: the noun-phrase the claim is about, derived from
  the tokens preceding the verb / the value in the same sentence.

We deliberately avoid heavy NLP dependencies. spaCy is used only as a
fallback POS tagger when ``spacy.blank("en"|"de")`` is importable; if
not, we fall back to a regex+heuristic anchor that is good enough for
the financial / legal prose this lane is built for.

Typed claims feed :mod:`latence_trace.core.structured_match` which
applies the AND-gate per-claim then aggregates with ``min`` so a
single broken cell collapses the structured score to zero.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

from latence_trace.core.structured import (
    _PERIOD_MARKER_RE,
    parse_number,
    _detect_locale,
    _normalize_key,
)


Locale = Literal["en", "de"]
Relation = Literal["EQ", "DIRECTION", "DELTA_REL", "RANGE", "ABOUT"]
Sign = Literal["POS", "NEG", "NEUTRAL"]


@dataclass(frozen=True)
class TypedClaim:
    """A single typed claim extracted from a response sentence.

    ``anchor`` is the noun-phrase the claim is about (used for cell
    alignment), ``relation`` is the kind of statement, ``value`` is the
    magnitude-normalised numeric value, ``unit`` is the canonical unit
    (``"M"``, ``"B"``, ``"PCT"``, ``"BPS"``, ``"COUNT"``, ``"NONE"``,
    ...), ``currency`` is the currency code when present, ``sign``
    classifies any direction verb the sentence carries, and
    ``approximate`` flags ``"approximately"`` / ``"about"`` / ``"etwa"``
    so the matcher widens the value tolerance for that claim.
    """

    anchor: str
    anchor_norm: str
    relation: Relation
    value: Optional[float]
    value2: Optional[float]
    unit: str
    currency: str
    sign: Sign
    locale: Locale
    raw_sentence: str
    raw_value_token: str
    approximate: bool


# ----------------------------------------------------------------------
# Direction / approximation lexicons (EN + DE)
# ----------------------------------------------------------------------


_PERIOD_PREFIX_RE = re.compile(
    r"(?:^|[^A-Za-z0-9])(?:Q[1-4]|FY|H[12]|CY|GJ|KW|W\d+)\s*$",
    re.IGNORECASE,
)


def _is_inside_period_marker(sentence: str, span: Tuple[int, int]) -> bool:
    """Reject numeric spans that are part of a fiscal-period marker.

    ``"Apple Q3 FY24 ..."`` should not produce two stray ``EQ`` claims
    for ``3`` and ``24``: those numerals are *period* tokens, not
    factual values. We check the immediate prefix (last 6 chars) for a
    ``Q``/``FY``/``H``/``CY``/``GJ``/``KW``/``W<n>`` token, anchored
    at a word boundary so common verbs that *end* in one of those
    letters (``"gre**w**"``, ``"jul**y**"``, ``"happ**y**"``) cannot
    masquerade as period prefixes and silently drop the legitimate
    numeric claim that follows them.

    We also drop bare 4-digit years (``"Q3 2024"`` -> ``2024``) when
    they sit on the boundary of a period marker.
    """

    start = span[0]
    if start <= 0:
        return False
    prefix = sentence[max(0, start - 6): start]
    return bool(_PERIOD_PREFIX_RE.search(prefix))


_DIR_POS_EN = re.compile(
    r"\b(grew|increased|rose|expanded|gained|jumped|climbed|added|raised|"
    r"surged|extended|widened|advanced|appreciated|risen|grown|expand)\b",
    re.IGNORECASE,
)
_DIR_NEG_EN = re.compile(
    r"\b(declined|fell|dropped|sank|decreased|contracted|lost|shrunk|shrank|"
    r"narrowed|lowered|cut|tumbled|reduced|slid|slipped|edged\s+down|dipped)\b",
    re.IGNORECASE,
)
_DIR_POS_DE = re.compile(
    r"\b(stieg|stiegen|wuchs|wuchsen|nahm\s+zu|nahmen\s+zu|kletterte|kletterten|"
    r"legte\s+zu|legten\s+zu|erhöhte|erhöhten|erhöht|wächst|wachsen|stärkte|"
    r"steigerte|steigerten|verbesserte|verbesserten|sprang|sprangen|"
    r"zugelegt|gestiegen|erhob|gewann)\b",
    re.IGNORECASE,
)
_DIR_NEG_DE = re.compile(
    r"\b(sank|sanken|fiel|fielen|nahm\s+ab|nahmen\s+ab|verringerte|verringerten|"
    r"schrumpfte|schrumpften|verlor|verloren|senkte|senkten|gesenkt|reduziert|"
    r"reduzierte|reduzierten|sinkt|fallen|gesunken|gefallen|verringert|"
    r"rückläufig|verschlechterte|verschlechterten|büßte|büßten|verlorengingen)\b",
    re.IGNORECASE,
)
_APPROX_EN = re.compile(
    r"\b(approximately|about|around|roughly|nearly|some|circa|ca\.?|near)\b|~",
    re.IGNORECASE,
)
_APPROX_DE = re.compile(
    r"\b(etwa|ungefähr|ca\.?|rund|circa|annähernd|nahezu|knapp|gut)\b",
    re.IGNORECASE,
)


# ----------------------------------------------------------------------
# Sentence segmentation (locale-tolerant)
# ----------------------------------------------------------------------


def _split_sentences(text: str) -> List[str]:
    """Sentence split for typed-claim extraction.

    Delegates to the universal splitter
    (``latence_trace.core.text_segmentation.split_sentences``) so we
    pick up the same WTPSplit + PySBD cascade used by groundedness and
    NLI claim extraction. The previous heuristic
    ``(?<=[\.\!\?])\\s+(?=[A-ZÄÖÜ])`` failed on lower-case sentence
    starts (German lists, captioned figures, ``2.5%`` decimal numbers)
    and on every abbreviation it had not been hand-coded for.
    """

    if not text or not text.strip():
        return []
    raw = text.strip()
    from latence_trace.core.text_segmentation import split_sentences

    spans = split_sentences(raw)
    out: List[str] = [span["text"] for span in spans if span.get("text")]
    return out or [raw]


# ----------------------------------------------------------------------
# Numeric span scan (response side)
# ----------------------------------------------------------------------


# Same family of numeric tokens as the source extractor in
# ``structured.py``. Captures one numeric span (with optional currency
# prefix and unit suffix) at a time so we can walk a sentence and pull
# out one or more typed claims.
_RESPONSE_NUM_RE = re.compile(
    r"""
    (?P<currency>\$|\u20AC|EUR|USD|GBP|CHF|JPY)?\s*
    (?P<num>-?\d{1,3}(?:[\.,\u00A0\s]\d{3})*(?:[\.,]\d+)?|\d+(?:[\.,]\d+)?)
    \s*
    (?P<suffix>%|percent|Prozent|bps|Basispunkte|Bp\.?|million|millions|billion|billions|trillion|Mio\.?|Mrd\.?|Tsd\.?|GWh|TWh|MWh|kWh|Punkte|points|Stück|Einheiten|EUR|USD|GBP|CHF|JPY)?
    """,
    re.VERBOSE,
)


# ``DELTA_FROM_TO`` -- captures "from A to B" / "von A auf B" so we can
# pull both endpoints into the typed claim and feed sign / direction
# matching with a paired source cell.
_DELTA_RE_EN = re.compile(
    r"""
    \bfrom\s+
    (?P<from_currency>\$|\u20AC|EUR|USD)?\s*
    (?P<from_num>-?\d{1,3}(?:[\.,\u00A0\s]\d{3})*(?:[\.,]\d+)?|\d+(?:[\.,]\d+)?)
    \s*(?P<from_suffix>%|percent|Prozent|million|billion|Mio\.?|Mrd\.?|GWh|TWh|MWh|Punkte|points|EUR|USD)?
    .{0,40}?
    \bto\s+
    (?P<to_currency>\$|\u20AC|EUR|USD)?\s*
    (?P<to_num>-?\d{1,3}(?:[\.,\u00A0\s]\d{3})*(?:[\.,]\d+)?|\d+(?:[\.,]\d+)?)
    \s*(?P<to_suffix>%|percent|Prozent|million|billion|Mio\.?|Mrd\.?|GWh|TWh|MWh|Punkte|points|EUR|USD)?
    """,
    re.VERBOSE | re.IGNORECASE,
)

_DELTA_RE_DE = re.compile(
    r"""
    \bvon\s+
    (?P<from_currency>\$|\u20AC|EUR|USD)?\s*
    (?P<from_num>-?\d{1,3}(?:[\.,\u00A0\s]\d{3})*(?:[\.,]\d+)?|\d+(?:[\.,]\d+)?)
    \s*(?P<from_suffix>%|Prozent|Mio\.?|Mrd\.?|Tsd\.?|GWh|TWh|MWh|Punkte|EUR|USD|Basispunkte)?
    .{0,40}?
    \b(?:auf|zu|nach)\s+
    (?P<to_currency>\$|\u20AC|EUR|USD)?\s*
    (?P<to_num>-?\d{1,3}(?:[\.,\u00A0\s]\d{3})*(?:[\.,]\d+)?|\d+(?:[\.,]\d+)?)
    \s*(?P<to_suffix>%|Prozent|Mio\.?|Mrd\.?|Tsd\.?|GWh|TWh|MWh|Punkte|EUR|USD|Basispunkte)?
    """,
    re.VERBOSE | re.IGNORECASE,
)


# ----------------------------------------------------------------------
# Anchor extraction
# ----------------------------------------------------------------------


_VERB_TOKENS = frozenset({
    "is", "was", "were", "are", "be", "been", "being",
    "had", "has", "have", "reported", "lay", "lies", "stood", "stand",
    "totalled", "totaled", "amount", "amounts", "amounted",
    "grew", "increased", "rose", "fell", "declined", "dropped", "sank",
    "expanded", "contracted", "narrowed", "widened",
    # DE
    "ist", "war", "waren", "lag", "lagen", "betrug", "betrugen",
    "stieg", "sank", "fiel", "wuchs", "kletterte", "verlor",
    "erreichte", "erreichten", "erzielte", "erzielten", "lieferte", "lieferten",
    "produzierte", "produzierten", "beförderte", "beförderten",
    "nahm", "nahmen", "verringerte", "schrumpfte", "verbesserte",
})


_CURRENCY_TOKENS = frozenset({"USD", "EUR", "GBP", "CHF", "JPY"})


# Pre-compiled at module load (was previously re-compiled per call /
# per claim inside ``_anchor_for_value`` and ``extract_typed_claims``).
_ANCHOR_TOKEN_RE = re.compile(r"[\w\u00C0-\u017F]+(?:[\-/][\w\u00C0-\u017F]+)*")
_PERIOD_TOKEN_RE = re.compile(
    r"^(?:q[1-4]|fy\d{0,4}|h[12]|cy\d{2,4}|gj\d{2,4}|w\d+|\d{4})$",
    re.IGNORECASE,
)
_BARE_NUMBER_ANCHOR_RE = re.compile(r"\d+(\.\d+)?")
_BARE_PERIOD_ANCHOR_RE = re.compile(
    r"(?:q[1-4]|fy\d{0,4}|h[12]|cy\d{2,4}|gj\d{2,4}|w\d+|\d{1,4})",
    re.IGNORECASE,
)


_FILLER_TOKENS = {
    # EN articles / adverbs / quantifiers we strip from the anchor
    "the", "a", "an", "of", "for", "in", "on", "at", "by", "with", "and",
    "to", "from", "into", "as", "such", "approximately", "about", "around",
    "roughly", "nearly", "or", "than", "less", "more", "most", "least",
    "year", "over", "yoy", "fiscal",
    # DE
    "der", "die", "das", "dem", "den", "des", "ein", "eine", "einer", "eines",
    "einen", "etwa", "ungefähr", "rund", "ca", "gegen", "gegenüber",
    "im", "in", "auf", "an", "und", "oder", "vom", "zum", "zur", "bei",
    "von", "zu", "für",
}


def _anchor_for_value(
    sentence: str,
    value_span: Tuple[int, int],
    *,
    locale: Locale,
    prev_value_end: int = 0,
) -> str:
    """Pull a noun-phrase anchor from the tokens left of the value span.

    Strategy:

    - Tokenize the sentence prefix (everything before the value span).
    - Drop verbs, currency tokens, filler determiners / prepositions.
    - Drop fiscal-period tokens (``Q3``, ``FY24``, ``2024``, ``H1``)
      because they would otherwise drown out the actual subject (cell
      anchors are typically 1-3 entity tokens like ``"Cloud"`` or
      ``"Greater China"``).
    - Return the remaining content tokens joined by a space.

    This deliberately does *not* cut at the first preceding verb. In
    German (subject-verb-value: ``"Der SAP Cloud-Umsatz lag in Q2
    2024 bei 4.151 Mio. EUR"``) the subject is *before* the verb, and
    cutting at ``"lag"`` would drop the only token (``"Cloud-Umsatz"``)
    that overlaps with the cell anchor. The matcher already handles
    long anchors via subset / Jaccard overlap, so we rely on the
    structured matcher to find the right cell from the candidate
    tokens.
    """

    # Take the slice from the END of the previous numeric claim (so we
    # don't bleed earlier numbers / units into the anchor of a later
    # claim in the same sentence). Falls back to the sentence start
    # when no previous numeric span exists.
    start_at = max(0, int(prev_value_end))
    prefix = sentence[start_at: value_span[0]].strip()
    if not prefix:
        return sentence.strip()

    tokens = _ANCHOR_TOKEN_RE.findall(prefix)
    if not tokens:
        return prefix

    cleaned: List[str] = []
    for tok in tokens:
        low = tok.lower()
        if low in _VERB_TOKENS:
            continue
        if low in _FILLER_TOKENS:
            continue
        if tok.upper() in _CURRENCY_TOKENS:
            continue
        # Reject fiscal-period tokens so an anchor like ``"q3 fy24
        # greater china revenue"`` becomes ``"greater china revenue"``.
        if _PERIOD_TOKEN_RE.match(tok):
            continue
        cleaned.append(tok)

    # If filtering stripped everything, fall back to the last 8 raw
    # prefix tokens (matches the legacy behaviour).
    if not cleaned:
        cleaned = tokens[-8:]

    # Cap the anchor at the trailing 10 content tokens so very long
    # German subject-clauses do not blow up the Jaccard denominator.
    if len(cleaned) > 10:
        cleaned = cleaned[-10:]

    return " ".join(cleaned).strip()


def _classify_sign(sentence: str, locale: Locale) -> Sign:
    if locale == "de":
        if _DIR_POS_DE.search(sentence):
            return "POS"
        if _DIR_NEG_DE.search(sentence):
            return "NEG"
    if _DIR_POS_EN.search(sentence):
        return "POS"
    if _DIR_NEG_EN.search(sentence):
        return "NEG"
    return "NEUTRAL"


def _is_approximate(sentence: str, locale: Locale) -> bool:
    if locale == "de" and _APPROX_DE.search(sentence):
        return True
    return bool(_APPROX_EN.search(sentence))


# ----------------------------------------------------------------------
# Public extractor
# ----------------------------------------------------------------------


def extract_typed_claims(
    response_text: str,
    *,
    locale: str = "auto",
) -> List[TypedClaim]:
    """Extract typed claims from a response.

    Each numeric span in each sentence becomes one or more typed
    claims:

    - The numeric span itself yields an ``EQ`` (equality) claim with
      the parsed magnitude-normalised value, canonical unit, and
      currency.
    - When the same sentence carries a direction verb
      (``"grew"``/``"declined"``/``"stieg"``/``"sank"``), the same
      claim is tagged with ``sign=POS|NEG``.
    - When the sentence matches the ``"from A to B"`` /
      ``"von A auf B"`` pattern, a ``DELTA_REL`` claim is emitted with
      both endpoints so the matcher can verify the implied direction
      against a paired source cell.

    Anchor selection is regex+heuristic, with no mandatory NLP
    dependency. Empty / unanchorable spans are silently dropped so the
    typed lane stays conservative on free-form prose.
    """

    if not response_text or not response_text.strip():
        return []

    if locale == "auto":
        locale_resolved: Locale = "de" if _detect_locale(response_text) == "de" else "en"
    elif locale == "de":
        locale_resolved = "de"
    else:
        locale_resolved = "en"

    sentences = _split_sentences(response_text)
    claims: List[TypedClaim] = []

    for sentence in sentences:
        sign = _classify_sign(sentence, locale_resolved)
        approximate = _is_approximate(sentence, locale_resolved)
        # Mask fiscal-period spans (``"Q3 FY24"``, ``"Q2 2024"``, bare
        # years) before the numeric scan so the period digits cannot be
        # mis-extracted as ``EQ`` claims (``"Q2 2024"`` would otherwise
        # leak a stray value of ``4`` from the trailing ``2024``).
        period_spans: List[Tuple[int, int]] = [
            (m.start(), m.end()) for m in _PERIOD_MARKER_RE.finditer(sentence)
        ]

        # ``DELTA`` candidates first: catch ``"from A to B"`` /
        # ``"von A auf B"`` so the value endpoints stay paired and we
        # do not also emit two stray EQ claims for the same span.
        delta_re = _DELTA_RE_DE if locale_resolved == "de" else _DELTA_RE_EN
        consumed: List[Tuple[int, int]] = []
        for match in delta_re.finditer(sentence):
            from_token = " ".join(
                tok
                for tok in [
                    match.group("from_currency") or "",
                    match.group("from_num") or "",
                    match.group("from_suffix") or "",
                ]
                if tok
            )
            to_token = " ".join(
                tok
                for tok in [
                    match.group("to_currency") or "",
                    match.group("to_num") or "",
                    match.group("to_suffix") or "",
                ]
                if tok
            )
            from_parsed = parse_number(from_token, locale=locale_resolved)
            to_parsed = parse_number(to_token, locale=locale_resolved)
            if from_parsed is None or to_parsed is None:
                continue
            anchor = _anchor_for_value(
                sentence, (match.start(), match.start()), locale=locale_resolved
            )
            if not anchor:
                continue
            anchor_norm = _normalize_key(anchor)
            if not anchor_norm:
                continue

            # The headline cell carries the *to* value because that is
            # what the sentence asserts ("grew to B" / "stieg auf B").
            implied_sign: Sign
            if to_parsed[0] > from_parsed[0]:
                implied_sign = "POS"
            elif to_parsed[0] < from_parsed[0]:
                implied_sign = "NEG"
            else:
                implied_sign = "NEUTRAL"
            # If the sentence carries an explicit direction verb the
            # claim sign is whatever the verb says (so we can detect
            # contradiction between verb and arithmetic). Otherwise we
            # use the implied sign.
            claim_sign: Sign = sign if sign != "NEUTRAL" else implied_sign

            claims.append(
                TypedClaim(
                    anchor=anchor,
                    anchor_norm=anchor_norm,
                    relation="DELTA_REL",
                    value=float(to_parsed[0]),
                    value2=float(from_parsed[0]),
                    unit=to_parsed[2] or from_parsed[2] or "NONE",
                    currency=to_parsed[1] or from_parsed[1] or "",
                    sign=claim_sign,
                    locale=locale_resolved,
                    raw_sentence=sentence,
                    raw_value_token=to_token,
                    approximate=approximate,
                )
            )
            consumed.append((match.start(), match.end()))

        # Plain numeric spans -- skip any that fall inside a delta
        # match we already consumed.
        prev_value_end = 0
        for match in _RESPONSE_NUM_RE.finditer(sentence):
            if any(c0 <= match.start() < c1 for c0, c1 in consumed):
                continue
            # Reject spans that are actually a fiscal-period token
            # (``Q3``, ``FY24``, ``H1 2024``). Those are not factual
            # value claims; aligning them against random cells would
            # collapse the AND-gate to zero.
            num_start = match.start("num") if match.group("num") else match.start()
            if _is_inside_period_marker(sentence, (num_start, match.end())):
                continue
            # Reject spans that overlap a period marker by any amount
            # (``"Q2 2024"`` -> a stray ``"4"`` after ``"202"`` is still
            # part of the period). The mask catches these even when the
            # immediate prefix check above misses them.
            if any(p0 <= num_start < p1 or p0 < match.end() <= p1
                   for p0, p1 in period_spans):
                continue
            num_token = match.group("num") or ""
            currency_token = match.group("currency") or ""
            suffix_token = match.group("suffix") or ""
            composite = " ".join(
                tok for tok in [currency_token, num_token, suffix_token] if tok
            )
            parsed = parse_number(composite, locale=locale_resolved)
            if parsed is None:
                continue
            value, currency, unit = parsed
            anchor = _anchor_for_value(
                sentence,
                (match.start(), match.end()),
                locale=locale_resolved,
                prev_value_end=prev_value_end,
            )
            prev_value_end = match.end()
            if not anchor:
                continue
            anchor_norm = _normalize_key(anchor)
            if not anchor_norm or len(anchor_norm) < 2:
                continue
            # Drop pure-number anchors (``"5"``, ``"100"``) that bleed
            # in when a sentence stitches numbers without a clear
            # subject (``"5 percent of the..."``).
            if _BARE_NUMBER_ANCHOR_RE.fullmatch(anchor_norm):
                continue
            # Drop numeric spans that follow a bare period marker
            # without any other anchor token: the only "subject" left
            # is the period itself, which cannot align to a cell.
            anchor_tokens = anchor_norm.split()
            if anchor_tokens and all(
                _BARE_PERIOD_ANCHOR_RE.fullmatch(t) for t in anchor_tokens
            ):
                continue

            claims.append(
                TypedClaim(
                    anchor=anchor,
                    anchor_norm=anchor_norm,
                    relation="EQ",
                    value=float(value),
                    value2=None,
                    unit=unit,
                    currency=currency,
                    sign=sign,
                    locale=locale_resolved,
                    raw_sentence=sentence,
                    raw_value_token=composite,
                    approximate=approximate,
                )
            )

    return claims


__all__ = [
    "TypedClaim",
    "extract_typed_claims",
]
