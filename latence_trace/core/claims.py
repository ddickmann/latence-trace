"""Atomic-fact decomposition for the Groundedness NLI lane (Phase F3).

This module provides a deterministic, rule-based atomic-claim splitter that
breaks a sentence into single-predicate atomic propositions. The splitter
prefers spaCy dependency parsing when an installed pipeline is available
(``en_core_web_sm`` for English, ``de_core_news_sm`` for German) and falls
back to a multilingual regex-based approximation otherwise.

Design principles:

- **Deterministic.** Two calls on the same input always produce the same
  atomic sequence, so harness reports stay reproducible.
- **Conservative.** When in doubt, emit the original sentence as a single
  atom - atomic decomposition that drops content is worse than no
  decomposition at all.
- **Offset-preserving.** Each atom keeps its character offsets in the
  source text so per-token NLI projection remains correct.
- **No trained weights.** Pure dependency / regex rules. The optional
  spaCy pipeline is the off-the-shelf English / German small checkpoint.
- **Multilingual.** Coordinator regex includes English (and, but, while,
  whereas) and German (und, aber, sondern, w\u00e4hrend, jedoch, doch)
  coordinations. spaCy auto-detects English vs German via a fast
  function-word heuristic and loads the matching pipeline; both are
  cached after first load.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Public types
# ----------------------------------------------------------------------


@dataclass
class AtomicClaim:
    """A single-predicate atomic proposition extracted from a sentence."""

    parent_index: int
    atom_index: int
    text: str
    char_start: int
    char_end: int


# ----------------------------------------------------------------------
# Regex-based splitter (always available)
# ----------------------------------------------------------------------

# Coordinating conjunctions to split on at the clause level. These are
# intentionally conservative: only conjunctions clearly separating two
# independent assertions are honored. We keep "or" and short clauses
# attached to avoid over-splitting numeric ranges or alternatives.
# Coordinations honored by the regex splitter. Mixes English (and, but,
# while, whereas) and German (und, aber, sondern, w\u00e4hrend, jedoch, doch)
# function words so the same regex works for monolingual EN, monolingual DE,
# and code-switched mixed responses without language detection.
_COORD_PATTERN = re.compile(
    r"\s*(?:"
    r",\s+(?:and|but|while|whereas|und|aber|sondern|w\u00e4hrend|jedoch|doch)\s+"
    r"|;\s+"
    r"|\s+(?:and|but|while|whereas|und|aber|sondern|w\u00e4hrend|jedoch|doch)\s+"
    r")",
    re.IGNORECASE,
)

# Relative-clause introducers we will optionally split on when the parent
# clause is otherwise complete. We avoid restrictive "that"-clauses to keep
# noun-phrase modifiers attached. German relative pronouns (der/die/das)
# are intentionally NOT split on - they double as articles and would
# over-split prose.
_RELATIVE_PATTERN = re.compile(
    r",\s+(?:which|who|where|when|welche[srnm]?|wobei|woher|wohin)\s+",
    re.IGNORECASE,
)

# Parenthetical insertions ", X, " that look appositive - heuristic split.
_APPOSITIVE_PATTERN = re.compile(
    r",\s+(?:also\s+known\s+as|aka|auch\s+bekannt\s+als|i\.e\.,?|e\.g\.,?|"
    r"d\.h\.,?|z\.b\.,?)\s+",
    re.IGNORECASE,
)

_SUBJECT_HINT_RE = re.compile(
    r"^(?:and|but|while|whereas|which|who|where|when|"
    r"und|aber|sondern|w\u00e4hrend|jedoch|doch|welche[srnm]?|wobei|woher|wohin)\s+",
    re.IGNORECASE,
)
_MIN_ATOM_LEN = 8


def _looks_independent(fragment: str) -> bool:
    """Heuristic: does ``fragment`` look like a standalone proposition?"""

    fragment = fragment.strip()
    if len(fragment) < _MIN_ATOM_LEN:
        return False
    # If the fragment starts with a conjunction or relative pronoun, it
    # likely needs the parent clause's subject to be standalone.
    if _SUBJECT_HINT_RE.match(fragment):
        return False
    return True


def _regex_split(sentence: str, parent_start: int) -> List[Tuple[int, int, str]]:
    """Split a sentence on coordinations / relative clauses / appositives.

    Returns a list of ``(absolute_start, absolute_end, text)`` tuples whose
    union covers the input sentence (after trimming whitespace).
    """

    if not sentence or not sentence.strip():
        return []

    pieces: List[Tuple[int, int, str]] = []
    text = sentence

    def _slice_at(positions: Sequence[int]) -> List[Tuple[int, str]]:
        if not positions:
            return [(0, text)]
        ordered = sorted(set(positions))
        out: List[Tuple[int, str]] = []
        last = 0
        for pos in ordered:
            chunk = text[last:pos]
            if chunk.strip():
                out.append((last, chunk))
            last = pos
        tail = text[last:]
        if tail.strip():
            out.append((last, tail))
        return out

    coord_starts = [match.end() for match in _COORD_PATTERN.finditer(text)]
    rel_starts = [match.end() for match in _RELATIVE_PATTERN.finditer(text)]
    appo_starts = [match.end() for match in _APPOSITIVE_PATTERN.finditer(text)]
    candidate_positions = coord_starts + rel_starts + appo_starts

    fragments = _slice_at(candidate_positions)

    # Drop the leading conjunction/relative tokens from each fragment.
    cleaned: List[Tuple[int, str]] = []
    for offset, frag in fragments:
        clean = _SUBJECT_HINT_RE.sub("", frag).strip()
        if not clean:
            continue
        new_start = offset + (frag.find(clean) if clean in frag else 0)
        cleaned.append((new_start, clean))

    if len(cleaned) <= 1:
        # Nothing meaningful to split on — return the whole sentence.
        s = sentence.strip()
        offset = sentence.find(s)
        return [(parent_start + max(0, offset), parent_start + max(0, offset) + len(s), s)]

    # Each cleaned fragment must look independent; otherwise stitch back to
    # the previous one. This keeps relative clauses without explicit
    # subjects attached.
    merged: List[Tuple[int, str]] = []
    for offset, frag in cleaned:
        if merged and not _looks_independent(frag):
            prev_offset, prev_text = merged[-1]
            merged[-1] = (prev_offset, (prev_text + " " + frag).strip())
            continue
        merged.append((offset, frag))

    if len(merged) <= 1:
        s = sentence.strip()
        offset = sentence.find(s)
        return [(parent_start + max(0, offset), parent_start + max(0, offset) + len(s), s)]

    pieces = []
    for offset, frag in merged:
        start = parent_start + offset
        end = start + len(frag)
        pieces.append((start, end, frag))

    return pieces


# ----------------------------------------------------------------------
# spaCy-based splitter (when available)
# ----------------------------------------------------------------------


class _SpacySplitter:
    """Lazy wrapper around an off-the-shelf spaCy English pipeline.

    We only use the dependency parser to identify clause boundaries; no
    weights are trained at runtime. The pipeline is loaded once per
    process and re-used.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._lock = threading.Lock()
        self._nlp = None
        self._unavailable = False

    def _ensure_loaded(self) -> bool:
        if self._unavailable:
            return False
        if self._nlp is not None:
            return True
        with self._lock:
            if self._unavailable:
                return False
            if self._nlp is not None:
                return True
            try:
                import spacy  # noqa: WPS433

                self._nlp = spacy.load(self.model_name, disable=["ner", "lemmatizer"])
                return True
            except Exception as exc:  # broad: spaCy missing, model missing, etc.
                logger.info(
                    "atomic_claims_spacy_unavailable",
                    extra={"model": self.model_name, "error": str(exc)},
                )
                self._unavailable = True
                return False

    def split(self, sentence: str, parent_start: int) -> List[Tuple[int, int, str]]:
        if not self._ensure_loaded():
            return _regex_split(sentence, parent_start)

        doc = self._nlp(sentence)
        # Walk root verbs / clausal heads. Each head plus its dependent
        # subtree spans an atomic clause. We enumerate clausal heads by
        # checking ``conj``, ``ccomp``, ``advcl``, ``relcl`` deps.
        clause_heads = []
        for token in doc:
            if token.dep_ in {"ROOT"}:
                clause_heads.append(token)
            elif token.dep_ in {"conj", "ccomp", "advcl", "relcl"} and token.pos_ in {"VERB", "AUX"}:
                clause_heads.append(token)

        if len(clause_heads) <= 1:
            return _regex_split(sentence, parent_start)

        clause_spans: List[Tuple[int, int]] = []
        for head in clause_heads:
            indices = [t.i for t in head.subtree]
            if not indices:
                continue
            min_i = min(indices)
            max_i = max(indices)
            start_char = doc[min_i].idx
            end_char = doc[max_i].idx + len(doc[max_i].text)
            clause_spans.append((start_char, end_char))

        if not clause_spans:
            return _regex_split(sentence, parent_start)

        clause_spans.sort()

        # Merge overlapping spans (e.g. nested ccomp inside ROOT).
        merged: List[Tuple[int, int]] = []
        for start, end in clause_spans:
            if merged and start <= merged[-1][1]:
                prev_start, prev_end = merged[-1]
                merged[-1] = (prev_start, max(prev_end, end))
                continue
            merged.append((start, end))

        # spaCy gives us nested clauses; we want the outer-most independent
        # atoms. Drop any span fully contained in another.
        outer: List[Tuple[int, int]] = []
        for start, end in merged:
            contained = False
            for o_start, o_end in outer:
                if o_start <= start and end <= o_end:
                    contained = True
                    break
            if not contained:
                outer.append((start, end))

        if len(outer) <= 1:
            return _regex_split(sentence, parent_start)

        atoms: List[Tuple[int, int, str]] = []
        for start, end in outer:
            text = sentence[start:end].strip()
            if not text or len(text) < _MIN_ATOM_LEN:
                continue
            adjusted_start = parent_start + start
            adjusted_end = adjusted_start + len(text)
            atoms.append((adjusted_start, adjusted_end, text))

        if not atoms:
            return _regex_split(sentence, parent_start)
        return atoms


# Language detection for the splitter is intentionally cheap: we count how
# often a small set of high-frequency German function words appears in the
# sentence and route to the German pipeline when the count clears a low
# threshold. Misroutes degrade to the regex fallback - they never crash.
_DE_FUNCTION_WORDS = (
    " der ",
    " die ",
    " das ",
    " den ",
    " dem ",
    " des ",
    " ein ",
    " eine ",
    " und ",
    " ist ",
    " sind ",
    " war ",
    " wurde ",
    " wurden ",
    " mit ",
    " von ",
    " auf ",
    " nicht ",
    " auch ",
    " sich ",
    " als ",
    " im ",
    " am ",
    " zu ",
    " zur ",
    " zum ",
    " f\u00fcr ",
    " \u00fcber ",
)


_DE_UMLAUT_CHARS = frozenset(("\u00e4", "\u00f6", "\u00fc", "\u00df"))


def _looks_german(text: str) -> bool:
    if not text:
        return False
    # NFC-normalize so combining-diaeresis sequences (``M\u0075\u0308nchen``)
    # collapse to the precomposed umlaut (``M\u00fcnchen``) before we count.
    nfc = unicodedata.normalize("NFC", text)
    # Strong signal first: any precomposed umlaut/eszett wins immediately
    # so we do not pay the function-word scan on obviously German strings.
    if any(ch in _DE_UMLAUT_CHARS for ch in nfc):
        return True
    haystack = " " + nfc.lower() + " "
    hits = 0
    for token in _DE_FUNCTION_WORDS:
        if token in haystack:
            hits += 1
            if hits >= 2:
                return True
    return False


_SPACY_SPLITTERS: Dict[str, _SpacySplitter] = {}
_SPACY_SPLITTER_LOCK = threading.Lock()


def _spacy_model_for(lang: str) -> str:
    if lang == "de":
        return os.environ.get("VOYAGER_GROUNDEDNESS_NLI_SPACY_MODEL_DE", "de_core_news_sm")
    return os.environ.get("VOYAGER_GROUNDEDNESS_NLI_SPACY_MODEL", "en_core_web_sm")


def _get_spacy_splitter(language: Optional[str] = None) -> Optional[_SpacySplitter]:
    """Return a cached spaCy splitter for ``language`` ("en" or "de").

    The English pipeline is the default - callers that pass ``None`` get the
    English splitter. ``_get_spacy_splitter("de")`` returns the German
    pipeline (``de_core_news_sm`` by default; override via
    ``VOYAGER_GROUNDEDNESS_NLI_SPACY_MODEL_DE``). Both pipelines are cached
    after first load so the second call is O(1).
    """

    lang = (language or "en").lower()
    if lang not in {"en", "de"}:
        lang = "en"
    cached = _SPACY_SPLITTERS.get(lang)
    if cached is not None:
        return cached
    # Two concurrent first-time accesses must not both create a splitter and
    # race the cache write; guard the slow path with a lock.
    with _SPACY_SPLITTER_LOCK:
        cached = _SPACY_SPLITTERS.get(lang)
        if cached is not None:
            return cached
        splitter = _SpacySplitter(_spacy_model_for(lang))
        _SPACY_SPLITTERS[lang] = splitter
        return splitter


# ----------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------


def is_atomic_enabled() -> bool:
    """Return True when atomic-claim decomposition is enabled via env flag."""

    raw = os.environ.get("VOYAGER_GROUNDEDNESS_NLI_ATOMIC_CLAIMS", "")
    return raw.lower() in {"1", "true", "yes"}


def decompose_sentence_into_atoms(
    sentence: str,
    parent_index: int,
    parent_start: int,
    *,
    use_spacy: bool = True,
    max_atoms_per_sentence: int = 6,
) -> List[AtomicClaim]:
    """Decompose a single sentence into a small set of atomic claims.

    Returns a list with at least one entry; if no decomposition is found,
    the original sentence is returned as a single atom. Character offsets
    are absolute (i.e. relative to the original response text).
    """

    if not sentence or not sentence.strip():
        return []

    pieces: List[Tuple[int, int, str]] = []
    if use_spacy:
        language = "de" if _looks_german(sentence) else "en"
        splitter = _get_spacy_splitter(language)
        if splitter is not None:
            pieces = splitter.split(sentence, parent_start)
        # If the language-specific pipeline failed to load (e.g.
        # de_core_news_sm not installed), retry with English which is the
        # most likely globally-available pipeline.
        if not pieces and language == "de":
            splitter = _get_spacy_splitter("en")
            if splitter is not None:
                pieces = splitter.split(sentence, parent_start)
    if not pieces:
        pieces = _regex_split(sentence, parent_start)

    if not pieces:
        clean = sentence.strip()
        offset = sentence.find(clean)
        pieces = [
            (
                parent_start + max(0, offset),
                parent_start + max(0, offset) + len(clean),
                clean,
            )
        ]

    if max_atoms_per_sentence > 0 and len(pieces) > max_atoms_per_sentence:
        pieces = pieces[:max_atoms_per_sentence]

    return [
        AtomicClaim(
            parent_index=int(parent_index),
            atom_index=int(idx),
            text=text,
            char_start=int(start),
            char_end=int(end),
        )
        for idx, (start, end, text) in enumerate(pieces)
    ]


__all__ = [
    "AtomicClaim",
    "decompose_sentence_into_atoms",
    "is_atomic_enabled",
]
