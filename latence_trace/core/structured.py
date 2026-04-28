"""Structured-source verification for groundedness Real Hardening Phase I.

When the support material is actually structured (JSON payload, markdown
pipe-table, or key-value blocks), fuzzy embedding similarity cannot be
trusted on its own: swapping ``"price": 42`` for ``"price": 420`` or
transposing the row/column label of a table cell keeps most of the
surface text intact but breaks the underlying fact.

This module offers a strict, rule-based adapter that:

1. Detects a structured source either from the caller-supplied
   ``content_type`` hint or by inspecting the support text.
2. Extracts ``(subject, predicate, object)`` triples from the support.
3. Extracts candidate triples from the response using patterns that are
   robust to ordinary natural-language responses (``"X is Y"``,
   ``"X: Y"``, ``"X = Y"``, ``"X = 'Y'"``) and dependency-based subject,
   verb, object if spaCy is available.
4. Compares them with normalized string matching plus numeric tolerance
   and simple alias resolution.

The module never raises during normal operation; on any failure it
returns ``None`` or an empty diagnostic so the caller can keep the
embedding-only headline and mark the channel as inactive.

The design is intentionally conservative: we prefer false-negatives
(``structured_source_detected=False``) over false positives. This channel
only fires when we are confident the support is structured and at least
one reliable (subject, predicate, object) triple was extracted on both
sides.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Public constants and tunables
# ----------------------------------------------------------------------

#: Absolute tolerance applied to numeric comparisons.
_DEFAULT_ABS_TOL = 1e-6

#: Relative tolerance applied to numeric comparisons once the magnitude is
#: larger than ~1.
_DEFAULT_REL_TOL = 1e-3

#: Penalty applied to ``structured_source_guarded`` for each detected
#: mismatch. With the default, three mismatches already drop the score
#: below the conservative 0.5 amber band.
_DEFAULT_PENALTY_PER_MISMATCH = 0.2

#: Floor applied to the guarded score so a single catastrophic mismatch
#: does not produce a ``NaN``-looking zero that downstream consumers
#: mis-interpret as "score unavailable".
_MIN_GUARDED = 0.0

#: Known caller-supplied content types.
_CONTENT_TYPE_JSON = {"application/json", "application/json+schema", "json"}
_CONTENT_TYPE_MD = {"text/markdown", "markdown", "md", "markdown/table"}

#: Soft cap on how many triples we ever return on a single side. Keeps
#: the diagnostics payload small and avoids pathological quadratic
#: blow-ups on giant JSON blobs.
_MAX_TRIPLES_PER_SIDE = 128


# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Triple:
    """A simple (subject, predicate, object) record.

    The fields are the *normalized* textual values used for matching.
    ``raw_object`` and ``numeric_value`` are preserved so mismatches can
    surface the original value for debugging.
    """

    subject: str
    predicate: str
    object: str
    raw_object: str
    numeric_value: Optional[float] = None
    source_hint: Optional[str] = None  # for example ``"json://$.items[0].price"``


@dataclass
class TripleMatch:
    """One comparison result between a source triple and a response triple."""

    subject: str
    predicate: str
    object: str
    matched: bool
    mismatch_kind: Optional[str] = None  # "missing", "value_mismatch", "numeric_mismatch"


@dataclass
class StructuredVerification:
    """The public, serializable output of the structured-source adapter."""

    source_format: Optional[str]
    detected: bool
    source_triple_count: int
    response_triple_count: int
    matches: List[TripleMatch] = field(default_factory=list)
    mismatches: List[TripleMatch] = field(default_factory=list)
    guarded_score: Optional[float] = None  # None => channel inactive
    # New (Typed Structured Evidence Lane). When the prose-table /
    # numeric-key-value detector fires, ``typed_score`` carries the
    # AND-gate cell-match score and ``typed_claim_count`` reports how
    # many response claims were aligned. Both stay ``None`` for the
    # legacy JSON / markdown-pipe-table path so the new lane never
    # silently mutates older diagnostics.
    typed_score: Optional[float] = None
    typed_claim_count: int = 0
    typed_claim_aligned: int = 0
    typed_claim_failures: List[Dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class TypedCell:
    """A single typed source-side cell extracted from a prose / KV / table support.

    Carries both the matchable token signature (``anchor_norm``) and
    the parsed numeric / unit / period payload that downstream typed-
    claim matching uses to enforce exact-or-tolerance value match,
    unit consistency, and period-aware sign comparison.
    """

    anchor: str
    anchor_norm: str
    value: float
    unit: str          # canonical unit: "M", "B", "PCT", "BPS", "COUNT", "GWH", "TWH", "MWH", "POINTS", "NONE"
    currency: str      # "USD", "EUR", or ""
    sign: str          # source sign relative to comparison cell ("POS" / "NEG" / "NEUTRAL")
    period: Optional[str] = None     # e.g. "Q3 FY24", "2023", "H1 2024"
    paired_value: Optional[float] = None   # comparison value if the cell carries one (for sign derivation)
    raw_phrase: str = ""


# ----------------------------------------------------------------------
# Normalization helpers
# ----------------------------------------------------------------------


_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[\u2018\u2019\u201C\u201D\"'`()\[\]{}]")


def _nfkc(text: str) -> str:
    return unicodedata.normalize("NFKC", text or "")


def _normalize_value(text: str) -> str:
    """NFKC + lowercase + squeeze whitespace + strip quote punctuation.

    We intentionally keep hyphens, slashes, colons, and digits: those
    carry fact-bearing information (dates, ratios, identifiers). We only
    strip cosmetic quoting and collapse repeated whitespace.
    """

    cleaned = _nfkc(text).strip()
    cleaned = _PUNCT_RE.sub("", cleaned)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned)
    return cleaned.lower()


def _normalize_key(text: str) -> str:
    """Predicate/subject normalization. Same rules plus underscore folding."""

    cleaned = _normalize_value(text)
    cleaned = cleaned.replace("_", " ").replace("-", " ")
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
    return cleaned


_NUMERIC_RE = re.compile(
    r"""
    ^\s*
    (?P<sign>[-+]?)
    (?P<value>\d{1,3}(?:[,\s]\d{3})+(?:\.\d+)? | \d+(?:\.\d+)? | \.\d+)
    \s*$
    """,
    re.VERBOSE,
)


def _parse_numeric(text: str) -> Optional[float]:
    """Parse a pure numeric token like ``"42"``, ``"1,234.5"``, or ``"-3.14"``.

    Returns ``None`` if the string carries any non-numeric glyphs
    (currency, units, percent signs). Those are kept for textual
    comparison instead, which lets us fall back to string equality when a
    response writes ``"42%"`` against a source ``"42"`` without silently
    declaring them equal.
    """

    if text is None:
        return None
    candidate = text.strip()
    if not candidate:
        return None
    if _NUMERIC_RE.match(candidate) is None:
        return None
    cleaned = candidate.replace(",", "").replace(" ", "")
    try:
        return float(cleaned)
    except (TypeError, ValueError):
        return None


def _values_match(
    source_value: str,
    source_numeric: Optional[float],
    response_value: str,
    response_numeric: Optional[float],
    *,
    abs_tol: float = _DEFAULT_ABS_TOL,
    rel_tol: float = _DEFAULT_REL_TOL,
) -> Tuple[bool, str]:
    """Return (matched, mismatch_kind) for two normalized object values.

    Numeric comparison takes priority when both sides parse to a finite
    float. Otherwise we fall back to normalized-string equality, then to
    substring containment so ``"on 20 July 1981"`` still matches
    ``"20 July 1981"`` extracted from a JSON date field.
    """

    if source_numeric is not None and response_numeric is not None:
        if math.isclose(
            float(source_numeric),
            float(response_numeric),
            abs_tol=abs_tol,
            rel_tol=rel_tol,
        ):
            return True, ""
        return False, "numeric_mismatch"

    if source_value == response_value:
        return True, ""
    if source_value and (source_value in response_value or response_value in source_value):
        return True, ""
    return False, "value_mismatch"


# ----------------------------------------------------------------------
# Detection
# ----------------------------------------------------------------------


def _looks_like_json(text: str) -> bool:
    if not text:
        return False
    stripped = text.strip()
    if not stripped:
        return False
    if stripped[0] not in "{[":
        return False
    # Parse a small prefix to avoid misclassifying text that merely starts
    # with a brace. ``json.loads`` is strict and cheap for small payloads.
    try:
        json.loads(stripped)
    except (ValueError, TypeError):
        return False
    return True


_MD_TABLE_SEP_RE = re.compile(
    r"^\s*\|?(\s*:?-+:?\s*\|)+(\s*:?-+:?\s*\|?)?\s*$"
)


def _looks_like_markdown_table(text: str) -> bool:
    if not text:
        return False
    rows = [row for row in text.splitlines() if row.strip()]
    if len(rows) < 2:
        return False
    if _MD_TABLE_SEP_RE.match(rows[1]) is None:
        return False
    return "|" in rows[0]


def _normalize_content_type_hint(hint: Optional[str]) -> Optional[str]:
    if not hint:
        return None
    lowered = hint.strip().lower()
    if not lowered:
        return None
    if any(token in lowered for token in _CONTENT_TYPE_JSON):
        return "json"
    if any(token in lowered for token in _CONTENT_TYPE_MD):
        return "markdown_table"
    return None


_PROSE_SENTENCE_END_RE = re.compile(r"[\.!?][\"'\u201D\u2019)\]]?\s*$")
_PROSE_WORD_RE = re.compile(r"[A-Za-z\u00C0-\u017F]{3,}")


def _support_is_prose(text: str) -> bool:
    """Return True when the support text looks like ordinary prose.

    A chunk is classified as prose when a majority of its non-empty lines
    either end in sentence punctuation or contain enough natural-language
    words to be plausibly a sentence. This is a *prose-safety guard* used
    by :func:`detect_source_format` to suppress the typed lane on
    multilingual enterprise RAG chunks that happen to hold 3+
    ``Label number`` patterns but are otherwise narrative. Table-shaped
    sources (JSON, markdown pipe tables, dense KV lists, short numeric
    facts) fail this check and continue to drive the typed lane.
    """

    if not text:
        return False
    raw_lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in raw_lines if line]
    if not lines:
        return False
    total = len(lines)
    sentence_like = 0
    for line in lines:
        if "|" in line and line.count("|") >= 2:
            continue
        if line.count(":") >= 1 and len(line) <= 80 and len(_PROSE_WORD_RE.findall(line)) <= 3:
            # Short ``Label: value`` lines look tabular, not prose.
            continue
        ends_like_sentence = bool(_PROSE_SENTENCE_END_RE.search(line))
        word_count = len(_PROSE_WORD_RE.findall(line))
        if ends_like_sentence or word_count >= 5:
            sentence_like += 1
    return sentence_like * 2 >= total  # >= 50% of non-empty lines are sentence-shaped


def detect_source_format(
    support_text: str,
    *,
    content_type: Optional[str] = None,
    response_text: Optional[str] = None,
) -> Optional[str]:
    """Return one of ``"json"``, ``"markdown_table"``, ``"prose_table"`` or ``None``.

    Caller hints win when they match the content; otherwise we
    auto-detect. We intentionally never classify a raw free-form blob as
    structured: false positives here directly poison the fusion.

    Prose-safety rule: when the caller did not supply a structured
    ``content_type`` hint *and* the support text reads like ordinary
    prose (:func:`_support_is_prose`), we return ``None`` so the typed
    lane stays silent. This protects enterprise RAG chunks (multilingual
    legal / cyber / financial narratives) that previously tripped
    ``prose_table`` or ``numeric_fact`` on 3 incidental ``Label number``
    hits and collapsed the fused headline via the AND-gate.

    The ``"prose_table"`` lane still covers real inline segment tables
    and dense numeric key-value lists (``"Cloud: 4.151 Mio. EUR;
    Software-Lizenzen: 64; ..."``); the stricter density rules inside
    :func:`_looks_like_prose_table` and :func:`_looks_like_numeric_kv`
    prevent that lane from firing on plain prose.
    """

    hint = _normalize_content_type_hint(content_type)
    if hint == "json" and _looks_like_json(support_text):
        return "json"
    if hint == "markdown_table" and _looks_like_markdown_table(support_text):
        return "markdown_table"
    # Strong structured formats always win regardless of prose guard.
    if _looks_like_json(support_text):
        return "json"
    if _looks_like_markdown_table(support_text):
        return "markdown_table"
    # ``_looks_like_prose_table`` is already strict (5+ hits + 30%
    # density), so trust it for true segment tables even when the
    # surrounding sentence ends in a period.
    if _looks_like_prose_table(support_text):
        return "prose_table"

    prose_safe = hint is None and _support_is_prose(support_text)
    if not prose_safe and _looks_like_numeric_kv(support_text):
        return "prose_table"
    if _looks_like_numeric_fact(support_text):
        # Numeric-fact lane is meant for short single-fact statements
        # (a rate decision, a single metric). The support-side detector is
        # already strict (< 220 chars, typed unit, <= 3 numbers, no
        # semicolon). We only refuse to fire when the *response* is long
        # and narrative, because then a user-written paragraph cannot be
        # aligned cell-by-cell against a short structured statement.
        if response_text is None or len((response_text or "").strip()) <= 400:
            return "numeric_fact"
    return None


# Pre-compiled at module load (used inside ``_looks_like_numeric_fact``
# which is on the request hot path).
_NUMERIC_FACT_NUM_RE = re.compile(
    r"\b-?\d{1,3}(?:[\.,\u00A0\s]\d{3})*(?:[\.,]\d+)?\b"
)
_NUMERIC_FACT_TYPED_UNIT_RE = re.compile(
    r"(?:%|percent|Prozent|bps|Basispunkte|Bp\.?|"
    r"\$|\u20AC|EUR|USD|GBP|CHF|JPY|"
    r"million|billion|trillion|Mio\.?|Mrd\.?|Tsd\.?|"
    r"GWh|TWh|MWh|kWh|Punkte|points)",
    re.IGNORECASE,
)


def _looks_like_numeric_fact(text: str) -> bool:
    """Detect a single-fact numeric statement.

    Targets short central-bank / regulator / weather / power-grid
    statements that read like one row of a table embedded in a
    headline (``"EZB-Zinsentscheidung 18.09.2024. Senkung des
    Einlagesatzes um 25 Basispunkte auf 3,50 %."``). The detector is
    deliberately strict so it does not fire on long prose paragraphs
    that happen to mention a number:

    - Total length below 220 characters.
    - At most 3 numeric values in the whole text.
    - At least one of the numeric values carries an explicit
      *typed* unit (``BPS``, ``%``, ``EUR``, ``USD``, ``M``, ``B``,
      ``GWh``...). Bare integers without a unit don't count.
    - No semicolons (those would qualify the text as a real table /
      prose table and the earlier detectors would have caught it).
    """

    if not text:
        return False
    stripped = text.strip()
    if len(stripped) > 220 or ";" in stripped:
        return False
    # Cap the iteration: we only need to know there are at least 1 and
    # at most 3 numeric matches. Walk the iterator and short-circuit so
    # very-numeric strings don't pay an O(n) scan cost just to be
    # rejected.
    nums = 0
    for _ in _NUMERIC_FACT_NUM_RE.finditer(stripped):
        nums += 1
        if nums > 3:
            return False
    if nums == 0:
        return False
    # Multiple typed-unit occurrences (``42%``, ``38%``, ``21%``) look
    # like a listed-metrics paragraph, not a single numeric fact.
    unit_hits = list(_NUMERIC_FACT_TYPED_UNIT_RE.finditer(stripped))
    if not unit_hits:
        return False
    if len(unit_hits) >= 3:
        return False
    return True


# ----------------------------------------------------------------------
# Prose-table / numeric-KV detection (Typed Structured Evidence Lane)
# ----------------------------------------------------------------------


# Anchor labels followed by a number with optional currency / unit. Picks
# up "Cloud: 4.151", "Greater China $14,728", "FY2025: $5.6 billion",
# "Audi 832", "Einlagesatz: 3,50 %", "Total revenue $30,040", etc.
# We require the label to start with a letter (or German umlaut / SS),
# allow internal spaces / slashes / hyphens / parentheses / digits, and
# accept either a colon, an equals sign, or a single whitespace before
# the number. The number itself may carry an optional currency prefix
# ($, €, EUR, USD), thousands separator (",", "."), decimal separator
# (",", ".") and an optional unit suffix (%, bps, M, B, Mio., Mrd.,
# million, billion, thousand, Mrd EUR, ...).
_PROSE_KV_HIT_RE = re.compile(
    r"""
    (?P<label>[A-Za-z\u00C0-\u017F][A-Za-z0-9\u00C0-\u017F\-\/\&\(\) ]{1,80}?)
    \s*[:=]?\s*
    (?P<currency>\$|\u20AC|EUR|USD|GBP|CHF|JPY)?\s*
    (?P<num>-?\d{1,3}(?:[\.,\u00A0\s]\d{3})*(?:[\.,]\d+)?|\d+(?:[\.,]\d+)?)
    \s*
    (?P<suffix>%|percent|Prozent|bps|Basispunkte|million|millions|billion|billions|trillion|Mio\.?|Mrd\.?|Tsd\.?|Bp\.?|GWh|TWh|MWh|kWh|Punkte|points|Stück|Einheiten|EUR|USD|million\.?)?
    """,
    re.VERBOSE,
)


# Stricter "label : number" rule used by the *detector* so we don't
# flag pure narrative paragraphs containing a stray number. We require
# the colon / equals separator, and we count distinct hits.
_PROSE_KV_DETECT_RE = re.compile(
    r"""
    (?:^|[\n;,\.])
    \s*
    (?P<label>[A-Za-z\u00C0-\u017F][A-Za-z0-9\u00C0-\u017F\-\/\&\(\) ]{1,80}?)
    \s*[:=]\s*
    (?P<currency>\$|\u20AC|EUR|USD|GBP|CHF|JPY)?\s*
    (?P<num>-?\d{1,3}(?:[\.,\u00A0\s]\d{3})*(?:[\.,]\d+)?|\d+(?:[\.,]\d+)?)
    """,
    re.VERBOSE,
)


# Pattern for "Label <number><unit>" pairs separated by ; or , (no
# explicit colon required), used by prose-table detection. Triggers on
# segment-table prose like:
#   "Americas $37,678; Europe $21,883; Greater China $14,728"
_PROSE_TABLE_DETECT_RE = re.compile(
    r"""
    (?P<label>[A-Z][A-Za-z\u00C0-\u017F0-9\-\/\& ]{1,60}?)
    \s+
    (?P<currency>\$|\u20AC|EUR|USD)?\s*
    (?P<num>-?\d{1,3}(?:[\.,\u00A0\s]\d{3})*(?:[\.,]\d+)?|\d+(?:[\.,]\d+)?)
    (?:\s*(?P<suffix>%|percent|Prozent|bps|Basispunkte|million|billion|Mio\.?|Mrd\.?|Tsd\.?|GWh|TWh|MWh|kWh|Punkte|points))?
    \s*[;,]
    """,
    re.VERBOSE,
)


# Period markers inside a prose-table announcement (used by the typed
# extractor to attach a period / fiscal label to each cell).
_PERIOD_MARKER_RE = re.compile(
    r"""
    (
      Q[1-4]\s*(?:FY?)?\s*\d{2,4}        # Q3 FY24, Q3 FY2024, Q2 2024, Q1 24
      | (?:H[12]|FY|GJ|CY)\s*\d{2,4}     # H1 2024, FY2025, GJ 2023, CY24
      | \d{4}                             # bare year (kept short to avoid false positives)
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)


_PROSE_TABLE_LABEL_VERBS = frozenset({
    # English finite verbs / participles that betray a sentence-shaped
    # "label" (i.e. not a real table cell). True tables use compact
    # noun-phrase labels ("Data Center", "Greater China", "Cloud",
    # "Q3 FY24") with no verb in the label segment.
    "is", "are", "was", "were", "be", "been", "being", "has", "have", "had",
    "rose", "fell", "grew", "declined", "increased", "decreased", "dropped",
    "edged", "climbed", "jumped", "surged", "sank", "expanded", "contracted",
    "narrowed", "widened", "totalled", "totaled", "delivered", "reported",
    "amounted", "stood", "reached", "lay", "lies", "exceeded", "posted",
    "revised", "achieved", "lowered", "raised", "lifted", "trimmed",
    "noted", "stated", "guided", "expects", "expect", "expected",
    "saw", "seen", "registered", "recorded", "represented", "marked",
    "came", "ended", "made", "met", "missed", "reaffirmed",
    # DE
    "ist", "war", "waren", "lag", "lagen", "betrug", "betrugen",
    "stieg", "sank", "fiel", "wuchs", "kletterte", "verlor",
    "erreichte", "erreichten", "erzielte", "erzielten", "lieferte", "lieferten",
    "nahm", "nahmen", "verringerte", "schrumpfte", "verbesserte",
    "stiegen", "fielen", "wuchsen", "verringerten",
})

# Pre-compiled at module load (was previously re-compiled on every call
# inside ``_label_is_table_cell`` — that is in the request hot path
# because the detector runs once per support unit and the inner regex
# fires once per qualifying label hit).
_TABLE_CELL_TOKEN_RE = re.compile(r"[\w\u00C0-\u017F]+")
_TABLE_CELL_PERIOD_RE = re.compile(
    r"^(?:q[1-4]|fy\d{0,4}|h[12]|cy\d{2,4}|gj\d{2,4}|w\d+|kw\d+|\d{4})$",
    re.IGNORECASE,
)


def _label_is_table_cell(label: str) -> bool:
    """True iff *label* looks like a real table-cell label.

    Heuristics (intentionally strict so prose with embedded numbers
    cannot masquerade as a table):

    - Label has at most 5 tokens (table cells are concise; sentence
      fragments are not).
    - Label contains zero verbs / participles from
      :data:`_PROSE_TABLE_LABEL_VERBS`.
    - Label is not just a fiscal-period marker (``"Q3 FY24"``); those
      are real cells but they don't carry an entity, so they should
      not push the detector across the threshold on their own.
    """

    tokens = _TABLE_CELL_TOKEN_RE.findall(label)
    if not tokens:
        return False
    if len(tokens) > 5:
        return False
    for tok in tokens:
        if tok.lower() in _PROSE_TABLE_LABEL_VERBS:
            return False
    if all(_TABLE_CELL_PERIOD_RE.match(t) for t in tokens):
        return False
    return True


def _looks_like_prose_table(text: str) -> bool:
    """Detect an inline prose-formatted table.

    A true prose table is dense *and* shaped like a table:

    - At least 5 ``label NUMBER[unit]`` pairs separated by semicolons
      or commas (raised from 3 to avoid tripping on enterprise RAG
      prose that happens to include three incidental metrics).
    - At least 5 of those pairs have compact noun-phrase labels (no
      verbs, <= 5 tokens).
    - Label-density: the number of qualifying pairs is at least 30%
      of the non-empty line count so 3 hits inside a long narrative
      paragraph cannot masquerade as a table.

    This is what separates ``"Americas $37,678; Europe $21,883; ..."``
    (table) from ``"...rose by 206,000 in June, ..."`` (prose with
    embedded numbers).
    """

    if not text or len(text) < 20:
        return False
    hits = list(_PROSE_TABLE_DETECT_RE.finditer(text))
    if len(hits) < 5:
        return False
    table_shaped = sum(1 for h in hits if _label_is_table_cell(h.group("label") or ""))
    if table_shaped < 5:
        return False
    lines = [line for line in text.splitlines() if line.strip()]
    if lines and table_shaped * 10 < len(lines) * 3:
        return False
    return True


def _looks_like_numeric_kv(text: str) -> bool:
    """Detect a numeric key-value list (``"key: value"`` pattern).

    Requires real KV density: at least 3 hits *and* at least 60% of the
    non-empty lines match the KV shape. This keeps the typed lane
    firing for genuine enumerated metric blocks while silencing it on
    prose paragraphs that happen to contain a handful of ``Label: N``
    bullets.
    """

    if not text or len(text) < 12:
        return False
    hits = list(_PROSE_KV_DETECT_RE.finditer(text))
    if len(hits) < 3:
        return False
    non_empty_lines = [line for line in text.splitlines() if line.strip()]
    if non_empty_lines:
        kv_line_count = sum(
            1
            for line in non_empty_lines
            if _PROSE_KV_DETECT_RE.search("\n" + line)
        )
        if kv_line_count * 10 < len(non_empty_lines) * 6:
            return False
    return True


# Currency prefix mapping
_CURRENCY_NORM = {
    "$": "USD",
    "USD": "USD",
    "\u20AC": "EUR",
    "EUR": "EUR",
    "GBP": "GBP",
    "CHF": "CHF",
    "JPY": "JPY",
}


# Multiplier mapping for canonical units.
# We always normalise *into* canonical magnitude units so a claim of
# "$30,040 million" and a source of "$30.04 billion" both become
# 30,040,000,000 in the comparison.
_UNIT_MULTIPLIER = {
    "M": 1_000_000.0,
    "MIO": 1_000_000.0,
    "B": 1_000_000_000.0,
    "MRD": 1_000_000_000.0,
    "T": 1_000_000_000_000.0,
    "TSD": 1_000.0,
    "K": 1_000.0,
}


def _canonicalize_unit_suffix(suffix: Optional[str]) -> Tuple[str, float]:
    """Map a raw unit suffix to (canonical_unit, magnitude_multiplier).

    Returns ``("NONE", 1.0)`` when no suffix is present. The canonical
    unit is what the matcher checks for hard equality; the magnitude
    multiplier is folded into the value so cross-magnitude equality
    (``"30,040 million"`` vs ``"30.04 billion"``) is still detected
    when the rest of the claim is consistent.
    """

    if not suffix:
        return "NONE", 1.0
    raw = suffix.strip().rstrip(".").upper()
    if raw in {"%", "PERCENT", "PROZENT"}:
        return "PCT", 1.0
    if raw in {"BPS", "BASISPUNKTE", "BP"}:
        return "BPS", 1.0
    if raw in {"MILLION", "MILLIONS", "MIO", "MM"}:
        return "M", 1_000_000.0
    if raw in {"BILLION", "BILLIONS", "MRD"}:
        return "B", 1_000_000_000.0
    if raw in {"TRILLION"}:
        return "T", 1_000_000_000_000.0
    if raw in {"TSD", "THOUSAND"}:
        return "TSD", 1_000.0
    if raw in {"GWH"}:
        return "GWH", 1_000_000.0   # treat MWh / GWh / TWh as commensurate energy
    if raw in {"MWH"}:
        return "MWH", 1_000.0
    if raw in {"TWH"}:
        return "TWH", 1_000_000_000.0
    if raw in {"KWH"}:
        return "KWH", 1.0
    if raw in {"PUNKTE", "POINTS"}:
        return "POINTS", 1.0
    if raw in {"STÜCK", "EINHEITEN"}:
        return "COUNT", 1.0
    return raw, 1.0


def _detect_locale(text: str) -> str:
    """Cheap locale heuristic: presence of DE-specific tokens flips to ``"de"``.

    The prose-table extractor uses this to disambiguate the meaning of
    ``"."`` / ``","`` inside a numeric token: in ``"1.234,56 EUR"`` the
    dot is the thousands separator and the comma is the decimal
    separator (DE), whereas in ``"1,234.56"`` the comma is thousands
    and the dot is decimal (EN).
    """

    if not text:
        return "en"
    de_tokens = (
        "Mio.", "Mrd.", "EUR", "€", "Tsd.", "Mio ", "Mrd ", "Mrd. EUR",
        "Prozent", "Basispunkte", "Stück", "Einheiten",
    )
    if any(tok in text for tok in de_tokens):
        return "de"
    de_words = re.search(r"\b(stieg|sank|fiel|wuchs|nahm zu|nahm ab|gegenüber|Konzern|Geschäft)\b", text)
    if de_words is not None:
        return "de"
    return "en"


def parse_number(token: str, locale: str = "auto") -> Optional[Tuple[float, str, str]]:
    """Parse a localized numeric literal into (value, currency, unit).

    Returns ``None`` when the token is not a number. Examples:
    - ``"$14,728"`` (en) -> ``(14728.0, "USD", "NONE")``
    - ``"14.728,00 EUR"`` (de) -> ``(14728.0, "EUR", "NONE")``
    - ``"1.5 billion"`` -> ``(1_500_000_000.0, "", "B")``
    - ``"1,5 Mrd. EUR"`` (de) -> ``(1_500_000_000.0, "EUR", "B")``
    - ``"3,50 %"`` (de) -> ``(3.5, "", "PCT")``
    - ``"108,000"`` (en) -> ``(108000.0, "", "NONE")``

    ``value`` is always magnitude-normalised: a ``"million"`` suffix
    is folded into ``value`` so cross-magnitude comparisons stay
    arithmetically correct. The returned ``unit`` is the canonical
    label (``"M"``, ``"B"``, ``"PCT"``, ``"BPS"``, ``"COUNT"``, ...) that
    the matcher hard-compares.
    """

    if token is None:
        return None
    text = str(token).strip()
    if not text:
        return None
    if locale == "auto":
        locale = _detect_locale(text)

    # Pull the optional leading currency
    currency_match = re.match(r"^\s*(\$|\u20AC|EUR|USD|GBP|CHF|JPY)\s*", text)
    currency = ""
    if currency_match:
        currency = _CURRENCY_NORM.get(currency_match.group(1).upper(), "")
        text = text[currency_match.end():]

    # Pull the optional trailing unit / currency suffix
    suffix_match = re.search(
        r"\s*(?P<suffix>%|percent|Prozent|bps|Basispunkte|Bp\.?|million|millions|billion|billions|trillion|Mio\.?|Mrd\.?|Tsd\.?|GWh|TWh|MWh|kWh|Punkte|points|Stück|Einheiten|EUR|USD|GBP|CHF|JPY)\s*$",
        text,
        re.IGNORECASE,
    )
    unit_token = ""
    if suffix_match:
        suffix_text = suffix_match.group("suffix")
        # Some suffixes are actually trailing currency labels (EUR /
        # USD) - lift them into the currency field instead of the unit
        # so a downstream comparison of ``"4.151 Mio. EUR"`` vs
        # ``"4,151 million USD"`` does not over-trigger on currency.
        upper = suffix_text.upper().strip(".")
        if upper in _CURRENCY_NORM:
            currency = currency or _CURRENCY_NORM[upper]
            unit_token = ""
        else:
            unit_token = suffix_text
        text = text[: suffix_match.start()]

    text = text.strip()
    if not text:
        return None

    # Locale-aware numeric parse
    cleaned = text
    if locale == "de":
        # 1.234,56 -> 1234.56  (drop dots used as thousands sep, keep
        # comma as decimal sep and convert)
        if "," in cleaned:
            cleaned = cleaned.replace(".", "").replace(" ", "").replace("\u00A0", "")
            cleaned = cleaned.replace(",", ".")
        else:
            # 14.728 -> ambiguous; if it has a single trailing 3-digit
            # group treat as thousands sep, otherwise as decimal.
            if re.fullmatch(r"-?\d{1,3}(?:\.\d{3})+", cleaned):
                cleaned = cleaned.replace(".", "")
            else:
                # bare integer or 1.5 (DE doesn't use bare dot decimals
                # for thousands of size <=3, so keep as-is).
                pass
    else:
        # 1,234.56 -> 1234.56  (commas thousands; period decimal)
        if "." in cleaned and "," in cleaned:
            cleaned = cleaned.replace(",", "")
        elif "," in cleaned:
            # Either "1,234" (thousands) or "1,5" — disambiguate by
            # checking the comma group sizes. EN never uses a single
            # comma followed by exactly 3 digits as a decimal.
            if re.fullmatch(r"-?\d{1,3}(?:,\d{3})+", cleaned):
                cleaned = cleaned.replace(",", "")
            else:
                # Treat as decimal (rare in EN financial prose, but seen
                # in mixed-locale PDFs).
                cleaned = cleaned.replace(",", ".")
        cleaned = cleaned.replace(" ", "").replace("\u00A0", "")

    try:
        value = float(cleaned)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None

    canonical_unit, multiplier = _canonicalize_unit_suffix(unit_token)
    value = value * multiplier
    return float(value), currency, canonical_unit


def _split_prose_table_period(text: str) -> Optional[str]:
    """Pull the leading period marker from a prose-table announcement.

    For ``"Apple Inc. revenue by reportable segment, Q3 FY2024, in
    millions: ..."`` we want ``"Q3 FY2024"`` so each cell can carry a
    period and so a downstream typed claim about ``"Q3 FY24"`` aligns
    to the right cell.
    """

    if not text:
        return None
    match = _PERIOD_MARKER_RE.search(text)
    if match is None:
        return None
    return match.group(1).strip()


def _table_announcement_unit(text: str) -> Tuple[str, float, str]:
    """Pull the table-level magnitude / currency announcement from a header.

    ``"Apple Inc. revenue by reportable segment, Q3 FY2024, in millions:
    Americas $37,678; ..."`` declares ``"in millions"`` once and then
    omits the unit on each cell. We capture the announcement so each
    cell inherits the right magnitude multiplier and currency.
    """

    if not text:
        return "NONE", 1.0, ""

    # Currency announcement (very simple heuristic; the most common
    # finance idioms are ``"in EUR"`` / ``"USD"`` / ``"in millions of
    # dollars"``). We default to USD when we see ``"$"`` in the text and
    # to EUR when we see ``"EUR"`` / ``"€"``.
    currency = ""
    if "EUR" in text or "\u20AC" in text:
        currency = "EUR"
    elif "$" in text or "USD" in text:
        currency = "USD"

    # Magnitude announcement.
    lower = text.lower()
    if re.search(r"\bin\s+billion(s)?\b|\bin\s+mrd\.?\b", lower):
        return "B", 1_000_000_000.0, currency
    if re.search(r"\bin\s+million(s)?\b|\bin\s+mio\.?\b", lower):
        return "M", 1_000_000.0, currency
    if re.search(r"\bin\s+thousand(s)?\b|\bin\s+tsd\.?\b", lower):
        return "TSD", 1_000.0, currency
    if re.search(r"\bin\s+(twh|gwh|mwh|kwh)\b", lower):
        return re.search(r"\bin\s+(twh|gwh|mwh|kwh)\b", lower).group(1).upper(), 1.0, currency
    if re.search(r"\bin\s+stück\b|\bin\s+einheiten\b|\bin\s+stk\.?\b", lower):
        return "COUNT", 1.0, currency
    return "NONE", 1.0, currency


# Pre-compiled label-shape and paired-value regexes used inside
# ``extract_typed_cells``. Hoisting them out of the per-cell loop keeps
# the inner work to ``Pattern.search`` lookups instead of re-compiling
# the same patterns dozens of times per request.
_CELL_NUMERIC_LABEL_RE = re.compile(r"\d+([\.,]\d+)?")
_CELL_LABEL_PERIOD_PREFIX_RE = re.compile(
    r"^(?:Q[1-4]|FY|H[12]|GJ|CY|W\d+)\b",
    re.IGNORECASE,
)
_CELL_PAIRED_VALUE_RE = re.compile(
    r"(?:vs|von|from|\(|gegen|prior)[^\d\-+€$]*?"
    r"(?P<num>-?\d{1,3}(?:[\.,\u00A0\s]\d{3})*(?:[\.,]\d+)?|\d+(?:[\.,]\d+)?)",
    re.IGNORECASE,
)
_CELL_LABEL_WS_RE = re.compile(r"\s+")
# Explicit Mio / Mrd / billion mentions inside a paired-value context
# (so we know whether the source already declared a magnitude on the
# paired number and we should NOT inherit the table-level multiplier).
_CELL_PAIRED_HAS_OWN_MAG_RE = re.compile(
    r"\b(?:million|billion|trillion|Mio\.?|Mrd\.?|Tsd\.?|GWh|TWh|MWh|kWh)\b",
    re.IGNORECASE,
)


def extract_typed_cells(
    support_text: str,
    *,
    locale: str = "auto",
) -> List[TypedCell]:
    """Pull typed cells from a prose-formatted table or numeric KV list.

    Each cell carries the parsed magnitude-normalised value, canonical
    unit, currency, optional period, and (when present in the source)
    a paired comparison value used by the matcher to derive a sign.
    """

    if not support_text:
        return []
    if locale == "auto":
        locale = _detect_locale(support_text)

    period = _split_prose_table_period(support_text)
    announced_unit, announced_mult, announced_currency = _table_announcement_unit(
        support_text
    )

    # Mask period-marker spans (``Q3 FY2024``, ``FY24``, ``H1 2024``, ``2023``)
    # before we scan for cells: they otherwise leak into ``_PROSE_KV_HIT_RE``
    # as ``"label NUMBER"`` matches (``"Q3 FY"`` + ``"202"``) and pollute the
    # cell list with bogus entities. Replace with spaces of the same length
    # so character offsets stay stable for downstream tooling.
    masked_text = _PERIOD_MARKER_RE.sub(lambda m: " " * len(m.group(0)), support_text)

    cells: List[TypedCell] = []
    seen: set = set()

    # We scan with the relaxed PROSE_KV_HIT_RE so we pick up both the
    # ``"label: number"`` form (KV list) and the ``"label number"``
    # form that follows a magnitude announcement.
    for match in _PROSE_KV_HIT_RE.finditer(masked_text):
        label = (match.group("label") or "").strip(" \t.,;:-")
        currency_token = (match.group("currency") or "").strip()
        num_token = (match.group("num") or "").strip()
        suffix_token = (match.group("suffix") or "").strip()
        if not label or not num_token:
            continue

        # Reject label tokens that are themselves numeric or look like
        # period markers — they are not entities.
        if _PERIOD_MARKER_RE.fullmatch(label):
            continue
        if _CELL_NUMERIC_LABEL_RE.fullmatch(label):
            continue
        # Drop labels that are now whitespace-padded fragments left over
        # from period masking (a no-op trim on the original text but
        # required because the masked text can leave ``"  FY24  "`` style
        # gaps which then re-glue to the next number).
        label_norm = _CELL_LABEL_WS_RE.sub(" ", label).strip()
        if not label_norm or len(label_norm) < 2:
            continue
        # Reject labels that begin with a known period prefix (``Q``,
        # ``FY``, ``H1``, ``H2``, ``GJ``) followed by a stray number;
        # the masking should have caught these but the regex sometimes
        # reassembles them from neighbouring text.
        if _CELL_LABEL_PERIOD_PREFIX_RE.match(label_norm):
            continue

        # Re-build a single token "currency num suffix" and parse it.
        composite = " ".join(
            tok for tok in [currency_token, num_token, suffix_token] if tok
        )
        parsed = parse_number(composite, locale=locale)
        if parsed is None:
            continue
        value, currency, unit = parsed

        # Inherit the table-level announcement when the cell did not
        # declare its own unit and the announcement is non-trivial.
        if unit == "NONE" and announced_unit != "NONE":
            value *= announced_mult
            unit = announced_unit
        if not currency and announced_currency:
            currency = announced_currency

        anchor = label.strip()
        anchor_norm = _normalize_key(anchor)
        if not anchor_norm:
            continue

        # Look ahead in the same window to see if a "(period: paired)"
        # context is provided, e.g. "Data Center $26,272 (Q2 FY24
        # $10,323)". When present, use it to derive a sign relative to
        # the chronological prior period.
        paired_value: Optional[float] = None
        local_sign = "NEUTRAL"
        # Scan a 80-char window after the match for a paired numeric
        # in parentheses or after "vs" / "von" / "from". We mask period
        # markers in the tail so a stray ``"Q3"`` cannot be picked up
        # as the paired value.
        tail_orig = support_text[match.end(): match.end() + 120]
        tail = _PERIOD_MARKER_RE.sub(lambda m: " " * len(m.group(0)), tail_orig)
        paired_match = _CELL_PAIRED_VALUE_RE.search(tail)
        if paired_match:
            paired_parsed = parse_number(paired_match.group("num"), locale=locale)
            if paired_parsed is not None:
                pv = paired_parsed[0]
                # Inherit the table-level magnitude when the paired
                # value omitted it; otherwise we'd compare $10,323M to
                # $26,272 unit-less.
                if announced_unit != "NONE" and not _CELL_PAIRED_HAS_OWN_MAG_RE.search(
                    paired_match.group(0)
                ):
                    pv *= announced_mult
                paired_value = pv
                if value > pv * 1.0001:
                    local_sign = "POS"
                elif value < pv * 0.9999:
                    local_sign = "NEG"
                else:
                    local_sign = "NEUTRAL"

        cell_key = (anchor_norm, round(value, 6), unit, currency)
        if cell_key in seen:
            continue
        seen.add(cell_key)
        cells.append(
            TypedCell(
                anchor=anchor,
                anchor_norm=anchor_norm,
                value=float(value),
                unit=unit,
                currency=currency,
                sign=local_sign,
                period=period,
                paired_value=paired_value,
                raw_phrase=match.group(0),
            )
        )
        if len(cells) >= _MAX_TRIPLES_PER_SIDE:
            break
    return cells


def extract_triples_from_prose_table(
    support_text: str,
    *,
    locale: str = "auto",
) -> List[Triple]:
    """Compatibility wrapper: emit Triples for the legacy matcher.

    Lets the existing structured-source pipeline continue to operate on
    prose-formatted tables; the typed AND-gate matcher uses
    :func:`extract_typed_cells` for the strict comparison.
    """

    triples: List[Triple] = []
    for cell in extract_typed_cells(support_text, locale=locale):
        # Encode magnitude-normalised value as a string so the existing
        # ``_parse_numeric`` round-trips it cleanly.
        value_token = (
            f"{int(cell.value)}"
            if cell.value.is_integer()
            else f"{cell.value:.6f}".rstrip("0").rstrip(".")
        )
        triples.append(
            Triple(
                subject=cell.anchor_norm,
                predicate="",
                object=value_token,
                raw_object=value_token,
                numeric_value=float(cell.value),
                source_hint="prose_table",
            )
        )
        if len(triples) >= _MAX_TRIPLES_PER_SIDE:
            break
    return triples


# ----------------------------------------------------------------------
# Source-side triple extraction
# ----------------------------------------------------------------------


_IDENTITY_KEYS = ("name", "title", "id", "label", "subject", "item", "entity")


def _object_identity(obj: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """Return ``(identity_value, identity_key)`` for an object, or ``(None, None)``.

    We probe a small allow-list of key names that callers tend to use
    for entity identity (``name``, ``title``, ``id``, ``label``,
    ``subject``, ``item``). If none of them are scalar we fall back to
    the structural path so matching stays deterministic.
    """

    for key in _IDENTITY_KEYS:
        if key not in obj:
            continue
        raw = obj[key]
        if isinstance(raw, (str, int, float, bool)):
            return _stringify_scalar(raw), key
    return None, None


def _iter_json_triples(
    value: Any,
    *,
    path: str = "$",
    parent_key: Optional[str] = None,
    inherited_subject: Optional[str] = None,
) -> Iterable[Triple]:
    """Walk a JSON value and yield ``(subject, predicate, object)`` triples.

    The subject is preferentially an identity-bearing field inside the
    current object (for example ``"Widget"`` from ``{"name": "Widget",
    "price": 42}``); otherwise the dotted JSON path is used. The
    predicate is the literal field name; the object is the stringified
    leaf value. Nested objects recurse into; lists are indexed. Handling
    an identity key this way makes the triples line up with the way
    humans phrase responses (``"The price of Widget is 42"``).
    """

    if isinstance(value, dict):
        local_identity, identity_key = _object_identity(value)
        subject_hint = local_identity if local_identity is not None else inherited_subject
        for key, nested in value.items():
            child_path = f"{path}.{key}"
            if isinstance(nested, (dict, list)):
                yield from _iter_json_triples(
                    nested,
                    path=child_path,
                    parent_key=str(key),
                    inherited_subject=subject_hint,
                )
                continue
            # Skip emitting the identity key itself - it becomes the
            # subject for its sibling fields and would otherwise degenerate
            # into a tautological ``subject == object`` triple that leaks
            # into downstream matching.
            if identity_key is not None and key == identity_key:
                continue
            object_str = _stringify_scalar(nested)
            numeric = _parse_numeric(object_str)
            subject_raw = subject_hint if subject_hint is not None else path
            yield Triple(
                subject=_normalize_key(subject_raw),
                predicate=_normalize_key(str(key)),
                object=_normalize_value(object_str),
                raw_object=object_str,
                numeric_value=numeric,
                source_hint=f"json://{child_path}",
            )
    elif isinstance(value, list):
        for idx, nested in enumerate(value):
            child_path = f"{path}[{idx}]"
            if isinstance(nested, (dict, list)):
                yield from _iter_json_triples(
                    nested,
                    path=child_path,
                    parent_key=parent_key,
                    inherited_subject=inherited_subject,
                )
                continue
            object_str = _stringify_scalar(nested)
            numeric = _parse_numeric(object_str)
            predicate_raw = parent_key if parent_key else f"[{idx}]"
            subject_raw = inherited_subject if inherited_subject is not None else path
            yield Triple(
                subject=_normalize_key(subject_raw),
                predicate=_normalize_key(predicate_raw),
                object=_normalize_value(object_str),
                raw_object=object_str,
                numeric_value=numeric,
                source_hint=f"json://{child_path}",
            )


def _stringify_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        # Keep floats bit-exact so we don't lose precision on round-trip.
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)
    return str(value)


def extract_triples_from_json(support_text: str) -> List[Triple]:
    """Parse JSON support text into a bounded list of triples."""

    try:
        parsed = json.loads(support_text)
    except (ValueError, TypeError):
        return []
    triples: List[Triple] = []
    for triple in _iter_json_triples(parsed):
        triples.append(triple)
        if len(triples) >= _MAX_TRIPLES_PER_SIDE:
            break
    return triples


def extract_triples_from_markdown_table(support_text: str) -> List[Triple]:
    """Parse a markdown pipe-table into (row_header, column_header, cell) triples."""

    rows = [row.strip() for row in (support_text or "").splitlines() if row.strip()]
    if len(rows) < 2 or _MD_TABLE_SEP_RE.match(rows[1]) is None:
        return []

    def _split_cells(row: str) -> List[str]:
        cells = [cell.strip() for cell in row.strip().strip("|").split("|")]
        return cells

    header_cells = _split_cells(rows[0])
    if not header_cells or any(not cell for cell in header_cells):
        return []

    # Column 0 is treated as the row header ("subject") when it carries
    # values. The first header label is typically the table title or
    # axis label, so we keep it as the predicate prefix for extra
    # columns and as the subject label for the row header itself.
    row_subject_label = header_cells[0]
    column_labels = header_cells

    triples: List[Triple] = []
    for row in rows[2:]:
        cells = _split_cells(row)
        if len(cells) < 2:
            continue
        row_subject_value = cells[0]
        if not row_subject_value:
            continue
        # Emit one triple per data cell in this row.
        for col_idx in range(1, min(len(cells), len(column_labels))):
            cell = cells[col_idx]
            if not cell:
                continue
            numeric = _parse_numeric(cell)
            triples.append(
                Triple(
                    subject=_normalize_key(row_subject_value),
                    predicate=_normalize_key(column_labels[col_idx]),
                    object=_normalize_value(cell),
                    raw_object=cell,
                    numeric_value=numeric,
                    source_hint=f"md://{row_subject_label}/{column_labels[col_idx]}",
                )
            )
            if len(triples) >= _MAX_TRIPLES_PER_SIDE:
                return triples
    return triples


def extract_source_triples(
    support_text: str,
    *,
    source_format: str,
) -> List[Triple]:
    """Dispatch to the right source-side extractor."""

    if source_format == "json":
        return extract_triples_from_json(support_text)
    if source_format == "markdown_table":
        return extract_triples_from_markdown_table(support_text)
    if source_format == "prose_table":
        return extract_triples_from_prose_table(support_text)
    return []


# ----------------------------------------------------------------------
# Response-side triple extraction (rule-based)
# ----------------------------------------------------------------------


_KV_COLON_RE = re.compile(
    r"""
    (?P<subject>[\w][\w\- ]{1,64}?)
    \s*[:=]\s*
    (?P<object>"[^"\n]+" | '[^'\n]+' | [^,\.;\n]{1,120})
    """,
    re.VERBOSE,
)

_IS_RE = re.compile(
    r"""
    \b(?P<subject>[A-Z][\w\- ]{1,64}?)
    \s+
    (?:is|are|was|were|equals|equal\s+to)
    \s+
    (?P<object>[^,\.;\n]{1,120})
    """,
    re.VERBOSE,
)

_THE_X_OF_Y_IS_Z = re.compile(
    r"""
    \bthe\s+
    (?P<predicate>[\w\- ]{1,40}?)
    \s+of\s+
    (?P<subject>[\w\- ]{1,40}?)
    \s+(?:is|was|equals)\s+
    (?P<object>[^,\.;\n]{1,120})
    """,
    re.VERBOSE | re.IGNORECASE,
)


def _strip_quotes(value: str) -> str:
    cleaned = value.strip()
    if len(cleaned) >= 2 and cleaned[0] in "\"'" and cleaned[-1] in "\"'":
        cleaned = cleaned[1:-1]
    return cleaned.strip()


def extract_response_triples(response_text: str) -> List[Triple]:
    """Rule-based (subject, predicate, object) triple extraction from free text.

    Three pattern families cover the overwhelming majority of structured
    paraphrases we actually see in RAG outputs:

    1. ``"key: value"`` and ``"key = value"`` (machine-style echoes).
    2. ``"X is Y"`` / ``"X was Y"`` / ``"X equals Y"``.
    3. ``"the <predicate> of <subject> is <object>"``.

    The predicate is omitted (empty string) for cases 1 and 2 because
    the caller-side matcher falls back to predicate-agnostic matching
    when either side has an empty predicate. This avoids spurious
    mismatches when the response paraphrases ``"the price of X is 42"``
    as ``"X is 42"``.
    """

    triples: List[Triple] = []
    if not response_text:
        return triples

    seen = set()

    def _add(subject: str, predicate: str, object_raw: str) -> None:
        if len(triples) >= _MAX_TRIPLES_PER_SIDE:
            return
        subj_norm = _normalize_key(subject)
        pred_norm = _normalize_key(predicate) if predicate else ""
        obj_norm = _normalize_value(_strip_quotes(object_raw))
        if not subj_norm or not obj_norm:
            return
        key = (subj_norm, pred_norm, obj_norm)
        if key in seen:
            return
        seen.add(key)
        numeric = _parse_numeric(_strip_quotes(object_raw))
        triples.append(
            Triple(
                subject=subj_norm,
                predicate=pred_norm,
                object=obj_norm,
                raw_object=_strip_quotes(object_raw),
                numeric_value=numeric,
                source_hint="response",
            )
        )

    for match in _THE_X_OF_Y_IS_Z.finditer(response_text):
        _add(match.group("subject"), match.group("predicate"), match.group("object"))

    for match in _KV_COLON_RE.finditer(response_text):
        _add(match.group("subject"), "", match.group("object"))

    for match in _IS_RE.finditer(response_text):
        _add(match.group("subject"), "", match.group("object"))

    return triples


# ----------------------------------------------------------------------
# Matching
# ----------------------------------------------------------------------


def _candidates_for_source(
    source: Triple,
    response_triples: Sequence[Triple],
) -> List[Triple]:
    """Return response triples whose subject and (optional) predicate match.

    Matching is done on the normalized fields. Predicate matching is
    only required when both sides declare one; if either side has an
    empty predicate we fall back to subject-only matching so natural-
    language paraphrases still align.
    """

    subject_hits = [r for r in response_triples if r.subject == source.subject]
    if not subject_hits:
        # Allow partial subject containment so ``"items 0"`` source maps
        # to ``"the item"`` response without failing.
        subject_hits = [
            r
            for r in response_triples
            if source.subject and (source.subject in r.subject or r.subject in source.subject)
        ]
    if not subject_hits:
        return []

    if not source.predicate:
        return subject_hits
    predicate_hits = [r for r in subject_hits if not r.predicate or r.predicate == source.predicate]
    if predicate_hits:
        return predicate_hits
    # Fall back to subject-only when the predicate did not overlap.
    return subject_hits


def match_triples(
    source_triples: Sequence[Triple],
    response_triples: Sequence[Triple],
) -> Tuple[List[TripleMatch], List[TripleMatch]]:
    """Compare source and response triples, returning (matches, mismatches).

    Only source triples that have at least one response candidate are
    considered; unmentioned source triples do not count as mismatches.
    This is the "at-most-what-you-said" rule that keeps the channel
    focused on hallucinations rather than omissions.
    """

    matches: List[TripleMatch] = []
    mismatches: List[TripleMatch] = []
    for source in source_triples:
        candidates = _candidates_for_source(source, response_triples)
        if not candidates:
            continue
        best = None
        best_kind = "value_mismatch"
        for candidate in candidates:
            matched, kind = _values_match(
                source.object,
                source.numeric_value,
                candidate.object,
                candidate.numeric_value,
            )
            if matched:
                best = candidate
                best_kind = ""
                break
            if best is None or (best_kind == "value_mismatch" and kind == "numeric_mismatch"):
                best = candidate
                best_kind = kind or "value_mismatch"

        if best is None:
            continue
        if not best_kind:
            matches.append(
                TripleMatch(
                    subject=source.subject,
                    predicate=source.predicate,
                    object=best.raw_object,
                    matched=True,
                )
            )
        else:
            mismatches.append(
                TripleMatch(
                    subject=source.subject,
                    predicate=source.predicate,
                    object=best.raw_object,
                    matched=False,
                    mismatch_kind=best_kind,
                )
            )
    return matches, mismatches


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------


def verify_structured_source(
    *,
    support_text: str,
    response_text: str,
    content_type: Optional[str] = None,
    penalty_per_mismatch: float = _DEFAULT_PENALTY_PER_MISMATCH,
) -> StructuredVerification:
    """Run detection + extraction + matching end to end.

    Returns a :class:`StructuredVerification` whose ``guarded_score`` is
    ``None`` when no structured source could be detected. A ``None``
    score is a hard signal to the fusion layer that this channel must
    be skipped for this request, not treated as zero.
    """

    if not support_text or not response_text:
        return StructuredVerification(
            source_format=None,
            detected=False,
            source_triple_count=0,
            response_triple_count=0,
            guarded_score=None,
        )

    source_format = detect_source_format(support_text, content_type=content_type)
    if source_format is None:
        return StructuredVerification(
            source_format=None,
            detected=False,
            source_triple_count=0,
            response_triple_count=0,
            guarded_score=None,
        )

    # Always try the typed AND-gate lane for prose-table sources,
    # independent of the legacy triple matcher. The typed lane uses
    # ``extract_typed_cells`` directly on the raw support text so it
    # still fires when ``extract_source_triples`` returns nothing
    # (e.g. when the legacy regex misses a German-format cell).
    typed_score, typed_aligned, typed_total, typed_failures = _maybe_run_typed_lane(
        support_text=support_text,
        response_text=response_text,
        source_format=source_format,
    )

    source_triples = extract_source_triples(support_text, source_format=source_format)
    if not source_triples:
        return StructuredVerification(
            source_format=source_format,
            detected=True,
            source_triple_count=0,
            response_triple_count=0,
            guarded_score=None,
            typed_score=typed_score,
            typed_claim_count=typed_total,
            typed_claim_aligned=typed_aligned,
            typed_claim_failures=typed_failures,
        )

    response_triples = extract_response_triples(response_text)
    if not response_triples:
        return StructuredVerification(
            source_format=source_format,
            detected=True,
            source_triple_count=len(source_triples),
            response_triple_count=0,
            guarded_score=None,
            typed_score=typed_score,
            typed_claim_count=typed_total,
            typed_claim_aligned=typed_aligned,
            typed_claim_failures=typed_failures,
        )

    matches, mismatches = match_triples(source_triples, response_triples)

    # The guarded score is only meaningful if we actually compared
    # something: no overlap means we stay neutral and leave the channel
    # inactive.
    if not matches and not mismatches:
        return StructuredVerification(
            source_format=source_format,
            detected=True,
            source_triple_count=len(source_triples),
            response_triple_count=len(response_triples),
            guarded_score=None,
            typed_score=typed_score,
            typed_claim_count=typed_total,
            typed_claim_aligned=typed_aligned,
            typed_claim_failures=typed_failures,
        )

    penalty = max(0.0, float(penalty_per_mismatch)) * float(len(mismatches))
    guarded = max(_MIN_GUARDED, 1.0 - penalty)
    return StructuredVerification(
        source_format=source_format,
        detected=True,
        source_triple_count=len(source_triples),
        response_triple_count=len(response_triples),
        matches=matches,
        mismatches=mismatches,
        guarded_score=float(guarded),
        typed_score=typed_score,
        typed_claim_count=typed_total,
        typed_claim_aligned=typed_aligned,
        typed_claim_failures=typed_failures,
    )


def _maybe_run_typed_lane(
    *,
    support_text: str,
    response_text: str,
    source_format: Optional[str],
) -> Tuple[Optional[float], int, int, List[Dict[str, Any]]]:
    """Run the typed AND-gate cell matcher and surface its score.

    Returns ``(score, aligned, total, failures)``. ``score`` is ``None``
    when the lane could not align a single response claim to a typed
    source cell (the lane stays silent on non-numeric prose so it does
    not pollute the headline for legal / FActScore-style content).

    The lane fires for ``"prose_table"`` (multi-cell segment tables /
    KV lists) and ``"numeric_fact"`` (short single-row numeric facts
    like central-bank rate decisions) sources because those are the
    formats where cell-level extraction is reliable. JSON and
    markdown-pipe-table sources already use the strict triple matcher
    above and do not need a parallel run.
    """

    if source_format not in ("prose_table", "numeric_fact"):
        return None, 0, 0, []

    try:
        cells = extract_typed_cells(support_text)
    except Exception:  # pragma: no cover - defensive guard for the hot path
        return None, 0, 0, []
    if not cells:
        return None, 0, 0, []

    try:
        # Local import to avoid a cycle: typed_claims imports from this
        # module, and structured_match imports from typed_claims.
        from latence_trace.core.typed_claims import extract_typed_claims
        from latence_trace.core.structured_match import (
            structured_score_from_claims,
            matched_to_dicts,
        )

        claims = extract_typed_claims(response_text)
        if not claims:
            return None, 0, 0, []
        result = structured_score_from_claims(claims, cells)
    except Exception:  # pragma: no cover - defensive guard for the hot path
        return None, 0, 0, []

    if result.score is None:
        return None, 0, len(claims), []
    failures = [
        entry for entry in matched_to_dicts(result.per_claim) if entry.get("per_claim_score", 1.0) < 1.0
    ]
    return float(result.score), int(result.aligned), int(len(claims)), failures


def is_structured_enabled() -> bool:
    """Feature flag for the structured-source fusion channel.

    When the flag is off we still compute the triples (so diagnostics
    stay honest) but the fusion layer is told to keep its weight at
    zero. Defaults to ``True`` because the channel only fires when a
    structured source is actually detected and at least one triple
    overlaps.
    """

    raw = os.environ.get("VOYAGER_GROUNDEDNESS_STRUCTURED_ENABLED", "1").strip().lower()
    if raw in {"", "0", "false", "no", "off"}:
        return False
    return True


def resolve_structured_mode(mode: Optional[str] = None) -> str:
    """Normalize a ``structured_verification`` mode string.

    The canonical values are ``"auto"``, ``"on"`` and ``"off"``. Callers
    can override via the ``VOYAGER_GROUNDEDNESS_STRUCTURED_MODE`` env
    var; unknown values collapse to ``"auto"``. ``"off"`` forces the
    structured lane to stay silent even when the detector would fire;
    ``"on"`` bypasses the prose-safety guard and runs the typed lane on
    the raw support (useful for true tables passed as plain text).
    """

    candidate = (mode or os.environ.get("VOYAGER_GROUNDEDNESS_STRUCTURED_MODE", "")).strip().lower()
    if candidate in {"off", "0", "false", "no", "disable", "disabled"}:
        return "off"
    if candidate in {"on", "force", "forced", "strict"}:
        return "on"
    return "auto"


def is_structured_gate_enabled() -> bool:
    """Feature flag for the AND-gate fusion (typed structured evidence lane).

    When ``VOYAGER_GROUNDEDNESS_STRUCTURED_GATE`` is enabled and the
    typed lane returns a non-``None`` score, the fusion switches from
    a weighted convex combination to ``min(narrative, structured)`` so a
    single broken cell collapses the headline. Defaults to ON because
    the typed lane is conservative (it only fires when at least one
    response claim could be aligned to a typed cell) and the lane is
    silenced for pure-prose contexts.
    """

    raw = os.environ.get(
        "VOYAGER_GROUNDEDNESS_STRUCTURED_GATE", "1"
    ).strip().lower()
    if raw in {"", "0", "false", "no", "off"}:
        return False
    return True


def default_penalty_per_mismatch() -> float:
    raw = os.environ.get("VOYAGER_GROUNDEDNESS_STRUCTURED_PENALTY", "").strip()
    try:
        value = float(raw) if raw else _DEFAULT_PENALTY_PER_MISMATCH
    except (TypeError, ValueError):
        value = _DEFAULT_PENALTY_PER_MISMATCH
    return max(0.0, min(1.0, value))


def verification_to_dict(result: StructuredVerification) -> Dict[str, Any]:
    """Serialize a :class:`StructuredVerification` for the response payload."""

    def _match_to_dict(match: TripleMatch) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "subject": match.subject,
            "predicate": match.predicate,
            "object": match.object,
            "matched": bool(match.matched),
        }
        if match.mismatch_kind:
            payload["mismatch_kind"] = match.mismatch_kind
        return payload

    return {
        "source_format": result.source_format,
        "source_triple_count": int(result.source_triple_count),
        "response_triple_count": int(result.response_triple_count),
        "matches": [_match_to_dict(m) for m in result.matches],
        "mismatches": [_match_to_dict(m) for m in result.mismatches],
        "typed_score": (
            float(result.typed_score) if result.typed_score is not None else None
        ),
        "typed_claim_count": int(result.typed_claim_count),
        "typed_claim_aligned": int(result.typed_claim_aligned),
        "typed_claim_failures": list(result.typed_claim_failures),
    }


__all__ = [
    "Triple",
    "TripleMatch",
    "TypedCell",
    "StructuredVerification",
    "detect_source_format",
    "extract_triples_from_json",
    "extract_triples_from_markdown_table",
    "extract_triples_from_prose_table",
    "extract_typed_cells",
    "extract_source_triples",
    "extract_response_triples",
    "match_triples",
    "parse_number",
    "verify_structured_source",
    "is_structured_enabled",
    "is_structured_gate_enabled",
    "resolve_structured_mode",
    "default_penalty_per_mismatch",
    "verification_to_dict",
]
