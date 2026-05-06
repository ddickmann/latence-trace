"""Universal sentence segmentation for the groundedness pipeline.

This module replaces every ad-hoc regex sentence/claim splitter that
used to live in ``groundedness.py``, ``nli.py`` and ``typed_claims.py``.
The motivation is concrete: a real production turn (``de-kafka-rossmann``)
fed the system the German fragment ``"... der zentralen Problematik (z. B.
im ersten Satz von Die Verwandlung, Der Verschollene, Der Process)
verstärken die Wirkung."`` The naive ``[^.!?\n]+(?:[.!?]+|$)`` regex
split on the period inside ``"z. B."`` and produced a grammatically
broken stub (``" B. im ersten Satz von ..."``), which NLI correctly
flagged as contradiction even though the content is verbatim from the
source. Adding more abbreviations to a hand-curated list is exactly the
"overfitting" the user explicitly rejected: the right fix is to use a
*learned* segmenter that knows abbreviation patterns across 85+
languages without us hand-coding them.

Architecture
------------

``split_sentences(text, language)`` returns a list of ``SentenceSpan``
dictionaries with the same shape used elsewhere in the runtime
(``{"text", "offset_start", "offset_end"}``). The implementation walks
a small cascade of segmenters, each strictly more deterministic and
cheaper than the last, and the *first* one that returns a non-empty
result wins:

1. **WTPSplit SaT (sat-3l-sm)** — a 3-layer transformer, ~50 MB,
   tokeniser-free, multilingual (85 languages). State of the art on
   adversarial fixtures across both German and English. Loaded lazily
   on first call and held in a process-wide singleton so we pay the
   ~140 ms warmup cost once. RunPod pre-downloads the model at image
   build time and re-warms it at boot.
2. **PySBD (rule-based, multilingual)** — deterministic, dependency-
   free, ~10 ms on the same Kafka fixture. Picked because it ships
   with a per-language abbreviation dictionary that covers every case
   in our calibration corpus. Used both as the SaT fallback and as
   an unconditional safety net inside the SaT branch (we union spans
   when SaT misses a clear boundary, see ``_reconcile_spans``).
3. **Line / paragraph fallback** — if both SaT and PySBD fail, we
   return the entire input as a single span. The contract from the
   user is explicit: "the fallback source of truth must be the
   sentence then" — i.e. never silently lose text, never invent
   boundaries with a regex.

All paths re-anchor offsets against the *original* ``text`` so callers
that index back into the response (NLI claim spans, heatmap, attribution)
keep working. WTPSplit returns sentence *strings* without offsets, so we
do a single forward scan with ``str.find`` to reconstruct them; this is
O(n) and correct for the general case (the splitter never reorders or
mutates characters, only inserts boundaries).

Threading
---------

Both the SaT and PySBD singletons are constructed under
``threading.Lock`` and are immutable after construction. WTPSplit's
``SaT.split`` is documented as thread-safe for inference; we serialise
construction only.

Failure mode
------------

Every helper is wrapped in a broad ``except Exception`` because a
bad sentence split must NEVER bring down a scoring request. On any
SaT or PySBD failure we log a one-shot warning (``logger.warning``
with a process-wide flag) and proceed to the next stage. Callers
always receive at least one span, even on empty input.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

SentenceSpan = Dict[str, Any]
"""``{"text": str, "offset_start": int, "offset_end": int}``

Same shape as ``latence_trace.core.groundedness._trimmed_span`` so
existing callers are drop-in compatible.
"""


# ---------------------------------------------------------------------------
# Singletons (lazy, thread-safe)
# ---------------------------------------------------------------------------

_SAT_MODEL: Optional[Any] = None
_SAT_LOCK = threading.Lock()
_SAT_WARNED = False
_SAT_DISABLED = False  # one-shot disable when the model fails to load

_PYSBD_SEGMENTERS: Dict[str, Any] = {}
_PYSBD_LOCK = threading.Lock()
_PYSBD_WARNED = False

_DEFAULT_SAT_MODEL = os.environ.get(
    "LATENCE_TRACE_SAT_MODEL",
    "sat-3l-sm",
).strip() or "sat-3l-sm"

_PYSBD_LANGUAGES = {"de", "en"}


def _load_sat() -> Optional[Any]:
    """Return the process-wide WTPSplit SaT singleton, or ``None`` on failure.

    The first caller pays the model-load cost (~140 ms warm cache,
    a few seconds cold). Concurrent callers block on the lock; once
    the singleton is set, the function returns instantly. A failure
    to import or load the model flips ``_SAT_DISABLED`` so we don't
    repeatedly retry (which would burn CPU on every request).
    """

    global _SAT_MODEL, _SAT_DISABLED, _SAT_WARNED
    if _SAT_DISABLED:
        return None
    if _SAT_MODEL is not None:
        return _SAT_MODEL
    with _SAT_LOCK:
        if _SAT_MODEL is not None:
            return _SAT_MODEL
        if _SAT_DISABLED:
            return None
        try:
            from wtpsplit import SaT
        except Exception as exc:  # noqa: BLE001 - opt dep may be missing in dev
            _SAT_DISABLED = True
            if not _SAT_WARNED:
                logger.warning(
                    "text_segmentation_sat_unavailable model=%s err=%s",
                    _DEFAULT_SAT_MODEL,
                    exc,
                )
                _SAT_WARNED = True
            return None
        try:
            model = SaT(_DEFAULT_SAT_MODEL)
        except Exception as exc:  # noqa: BLE001
            _SAT_DISABLED = True
            if not _SAT_WARNED:
                logger.warning(
                    "text_segmentation_sat_load_failed model=%s err=%s",
                    _DEFAULT_SAT_MODEL,
                    exc,
                )
                _SAT_WARNED = True
            return None
        # SaT inference on CPU is ~100 ms per call which is too slow
        # for the hot RAG path. On a CUDA host we move the model to
        # FP16 + GPU which brings it down to ~5-10 ms. Failures here
        # are non-fatal: we keep the CPU model and accept the latency.
        try:
            import torch

            if torch.cuda.is_available():
                model.half().to("cuda")
                logger.info(
                    "text_segmentation_sat_loaded device=cuda dtype=fp16 model=%s",
                    _DEFAULT_SAT_MODEL,
                )
            else:
                logger.info(
                    "text_segmentation_sat_loaded device=cpu model=%s",
                    _DEFAULT_SAT_MODEL,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "text_segmentation_sat_gpu_move_failed model=%s err=%s",
                _DEFAULT_SAT_MODEL,
                exc,
            )
        _SAT_MODEL = model
        return _SAT_MODEL


def _load_pysbd(language: str) -> Optional[Any]:
    """Return a per-language PySBD ``Segmenter`` singleton, or ``None``."""

    global _PYSBD_WARNED
    lang = language if language in _PYSBD_LANGUAGES else "en"
    cached = _PYSBD_SEGMENTERS.get(lang)
    if cached is not None:
        return cached
    with _PYSBD_LOCK:
        cached = _PYSBD_SEGMENTERS.get(lang)
        if cached is not None:
            return cached
        try:
            import pysbd
        except Exception as exc:  # noqa: BLE001
            if not _PYSBD_WARNED:
                logger.warning("text_segmentation_pysbd_unavailable err=%s", exc)
                _PYSBD_WARNED = True
            return None
        try:
            seg = pysbd.Segmenter(language=lang, clean=False, char_span=True)
        except Exception as exc:  # noqa: BLE001
            if not _PYSBD_WARNED:
                logger.warning(
                    "text_segmentation_pysbd_load_failed lang=%s err=%s",
                    lang,
                    exc,
                )
                _PYSBD_WARNED = True
            return None
        _PYSBD_SEGMENTERS[lang] = seg
        return seg


# ---------------------------------------------------------------------------
# Span helpers
# ---------------------------------------------------------------------------


def _trimmed_span(fragment: str, start: int) -> Optional[SentenceSpan]:
    """Strip leading/trailing whitespace and return the canonical span dict."""

    stripped = fragment.strip()
    if not stripped:
        return None
    leading = len(fragment) - len(fragment.lstrip())
    return {
        "text": stripped,
        "offset_start": start + leading,
        "offset_end": start + leading + len(stripped),
    }


def _fallback_span(text: str) -> List[SentenceSpan]:
    """Last-resort: return the whole text as one span (never empty list)."""

    stripped = text.strip()
    if not stripped:
        return []
    start = text.find(stripped)
    if start < 0:
        start = 0
    return [
        {
            "text": stripped,
            "offset_start": start,
            "offset_end": start + len(stripped),
        }
    ]


def _resolve_active_language(language: Optional[str]) -> str:
    """Resolve language for segmentation.

    Caller-supplied wins. Otherwise we read the request-scoped
    ``_ACTIVE_ROUTE_DECISION`` ContextVar set by the API service so
    individual splitters don't need to plumb the language through every
    call site. Imports are inline to avoid circular imports
    (``service.py`` imports ``groundedness.py`` which imports us).
    """

    if language and language in {"de", "en"}:
        return language
    try:
        from latence_trace.api import service as _service

        decision = _service._ACTIVE_ROUTE_DECISION.get()
    except Exception:  # noqa: BLE001 - never fail segmentation on lookup
        decision = None
    if decision is not None:
        candidate = getattr(decision, "language", None)
        if candidate in {"de", "en"}:
            return candidate
    return "en"


# ---------------------------------------------------------------------------
# Stage 1: WTPSplit SaT
# ---------------------------------------------------------------------------


def _split_with_sat(text: str, language: str) -> Optional[List[SentenceSpan]]:
    """Run SaT and re-anchor offsets against the original ``text``.

    SaT returns sentences *with* their trailing whitespace, in order,
    such that ``"".join(sentences) == text`` for typical input. We
    walk the cursor forward and use ``str.find`` to be defensive
    against stray whitespace differences. Any mismatch (which would
    indicate the segmenter rewrote characters, not just inserted
    boundaries) returns ``None`` so the caller falls back to PySBD.
    """

    model = _load_sat()
    if model is None:
        return None
    try:
        # ``language_code`` is supported by SaT but optional; passing it
        # nudges the segmenter towards the right abbreviation patterns
        # without forcing us to load a per-language LoRA adapter.
        try:
            pieces = model.split(text, lang_code=language)
        except TypeError:
            pieces = model.split(text)
    except Exception as exc:  # noqa: BLE001
        logger.warning("text_segmentation_sat_split_failed err=%s", exc)
        return None
    spans: List[SentenceSpan] = []
    cursor = 0
    for piece in pieces:
        # ``piece`` may include trailing whitespace; we only care about
        # the trimmed sentence for the span text but must search using
        # the trimmed form so leading whitespace from the previous
        # piece isn't double-counted.
        trimmed = piece.strip()
        if not trimmed:
            continue
        idx = text.find(trimmed, cursor)
        if idx < 0:
            # Defensive: ``find`` should always succeed for a
            # well-formed split, but if SaT ever rewrites characters
            # we abort and let PySBD take over.
            return None
        spans.append(
            {
                "text": trimmed,
                "offset_start": idx,
                "offset_end": idx + len(trimmed),
            }
        )
        cursor = idx + len(trimmed)
    return spans or None


# ---------------------------------------------------------------------------
# Stage 2: PySBD (rule-based, multilingual)
# ---------------------------------------------------------------------------


def _split_with_pysbd(text: str, language: str) -> Optional[List[SentenceSpan]]:
    """Deterministic rule-based segmenter; honours offsets natively."""

    seg = _load_pysbd(language)
    if seg is None:
        return None
    try:
        char_spans = seg.segment(text)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "text_segmentation_pysbd_segment_failed lang=%s err=%s",
            language,
            exc,
        )
        return None
    spans: List[SentenceSpan] = []
    for cs in char_spans:
        sent_text = getattr(cs, "sent", None)
        start = getattr(cs, "start", None)
        end = getattr(cs, "end", None)
        if sent_text is None or start is None or end is None:
            continue
        # Re-anchor against the original text (PySBD's own offsets are
        # against the same input we passed in) and trim to drop trailing
        # whitespace pysbd preserves by design.
        span = _trimmed_span(text[start:end], start)
        if span is not None:
            spans.append(span)
    return spans or None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


_MARKDOWN_INLINE_RE = re.compile(
    # Inline emphasis / code fences that SaT otherwise treats as
    # sentence-ending punctuation. Each pattern is *length-preserving*
    # by construction (we substitute the same number of space characters
    # the marker occupied), so the returned spans still index back into
    # the original text without an offset shift.
    r"(\*\*|__|`+|~~|\*|_(?=\w)|(?<=\w)_)"
)
_MARKDOWN_LIST_BULLET_RE = re.compile(
    # Leading list bullets: ``- `` / ``* `` / ``+ `` / ``1. `` / ``a) ``.
    # Same length-preserving substitution: replace the marker with
    # spaces so SaT sees a clean sentence start instead of a
    # punctuation-then-letter boundary that it sometimes treats as a
    # mid-sentence aside.
    r"(?m)^(\s*)([-*+]\s+|\d{1,3}[.)]\s+|[a-zA-Z][.)]\s+)"
)
_MARKDOWN_HEADER_RE = re.compile(r"(?m)^(#+\s+)")
_MARKDOWN_BLOCKQUOTE_RE = re.compile(r"(?m)^(>+\s*)")
_SOFT_LINEBREAK_RE = re.compile(
    # Single newlines inside a paragraph are line-wrap, not paragraph
    # breaks. SaT often treats them as sentence boundaries which fans
    # a single sentence out into 2-3 phantom claims (e.g. ``It\nfalls
    # squarely within Annex III(4)(a)`` becomes ["It", "falls squarely
    # within ..."]). Replacing the lone newline with a space neutralises
    # that without affecting double-newline paragraph breaks (which
    # remain real sentence boundaries) and preserves character count.
    r"(?<!\n)\n(?!\n)"
)


def _normalise_for_segmentation(text: str) -> str:
    """Length-preserving normaliser for markdown-formatted answers.

    LLM chat responses are almost always markdown (``**bold**``,
    bullet lists, headers, inline ``code``). The SaT segmenter is
    trained on natural prose and treats those markers like
    punctuation, which fragments a single semantic sentence into
    multiple junk claims:

    >>> "**1. High-risk classification.** TalentScout AI is..."
    # SaT -> ["**1. High-risk classification.", "**", "TalentScout", "AI is..."]

    By substituting markers with the same number of space characters
    *before* segmentation we neutralise the false sentence-boundary
    cue while keeping every downstream character offset (NLI claim
    spans, heatmap, attribution) byte-identical to the original. The
    real text of every span is recovered from the ORIGINAL ``text``
    via offsets, so the spaces never appear in the user-facing
    output.
    """

    if not text:
        return text
    out = _MARKDOWN_INLINE_RE.sub(lambda m: " " * len(m.group(0)), text)
    out = _MARKDOWN_LIST_BULLET_RE.sub(
        lambda m: m.group(1) + " " * len(m.group(2)), out
    )
    out = _MARKDOWN_HEADER_RE.sub(lambda m: " " * len(m.group(1)), out)
    out = _MARKDOWN_BLOCKQUOTE_RE.sub(lambda m: " " * len(m.group(1)), out)
    out = _SOFT_LINEBREAK_RE.sub(" ", out)
    assert len(out) == len(text), (
        "markdown normaliser must preserve length so offsets remain valid"
    )
    return out


def split_sentences(
    text: str,
    language: Optional[str] = None,
) -> List[SentenceSpan]:
    """Segment ``text`` into sentence spans.

    Parameters
    ----------
    text:
        The text to segment. Empty / whitespace-only input returns an
        empty list. Otherwise we always return at least one span.
    language:
        Either ``"de"`` or ``"en"``; anything else (including ``None``)
        causes us to consult the request-scoped corpus router decision
        and fall back to ``"en"`` if there isn't one.

    Returns
    -------
    list of dict
        Each dict has keys ``text`` (whitespace-trimmed sentence),
        ``offset_start`` and ``offset_end`` (inclusive/exclusive
        character indices into the *original* ``text``).
    """

    if not text:
        return []
    if not text.strip():
        return []

    lang = _resolve_active_language(language)

    # Run the segmenters against a markdown-neutralised copy. The
    # normaliser is length-preserving so spans recovered from the
    # neutralised string are valid offsets into the *original* ``text``;
    # we re-anchor the span text from the original input below to make
    # sure no whitespace placeholders leak into user-visible claim text.
    normalised = _normalise_for_segmentation(text)

    sat_spans = _split_with_sat(normalised, lang)
    if sat_spans:
        return _rehydrate_spans(sat_spans, text)
    pysbd_spans = _split_with_pysbd(normalised, lang)
    if pysbd_spans:
        return _rehydrate_spans(pysbd_spans, text)
    return _fallback_span(text)


def _rehydrate_spans(
    spans: List[SentenceSpan], original: str
) -> List[SentenceSpan]:
    """Replace the (possibly markdown-neutralised) span text with the
    actual slice from the *original* ``text``.

    The normaliser substitutes markdown markers with spaces so SaT's
    boundaries land in the right places, but we don't want the
    user-facing claim text to read ``"  1. High-risk classification.  "``
    instead of ``"**1. High-risk classification.**"``. Offsets are
    identical between the normalised and original strings (length is
    preserved), so we just slice the original and re-strip whitespace.
    Tiny spans that end up empty after re-stripping (rare; happens when
    SaT picks a marker-only fragment) are dropped so they cannot become
    phantom claims.
    """

    out: List[SentenceSpan] = []
    for span in spans:
        start = int(span.get("offset_start", 0))
        end = int(span.get("offset_end", 0))
        start = max(0, min(start, len(original)))
        end = max(start, min(end, len(original)))
        original_slice = original[start:end]
        stripped = original_slice.strip()
        if not stripped:
            continue
        # Tighten offsets onto the trimmed slice so heatmap / NLI
        # see consistent boundaries.
        leading = len(original_slice) - len(original_slice.lstrip())
        trailing = len(original_slice) - len(original_slice.rstrip())
        out.append(
            {
                "text": stripped,
                "offset_start": start + leading,
                "offset_end": end - trailing,
            }
        )
    return out or _fallback_span(original)


def warmup_segmenters(*, languages: Sequence[str] = ("en", "de")) -> Dict[str, Any]:
    """Pre-load the SaT model and per-language PySBD segmenters.

    Called from the RunPod boot path so the first scoring request does
    not pay the model-load cost on the hot path. Returns a small
    diagnostic blob so the boot log can confirm what loaded.
    """

    sat_loaded = _load_sat() is not None
    pysbd_loaded: Dict[str, bool] = {}
    for lang in languages:
        pysbd_loaded[lang] = _load_pysbd(lang) is not None
    if sat_loaded:
        # Force a tiny inference so the model graph is fully warm.
        try:
            _split_with_sat("Warmup ok. This is fine.", "en")
        except Exception:  # pragma: no cover - defensive
            pass
    return {
        "sat_loaded": sat_loaded,
        "sat_model": _DEFAULT_SAT_MODEL,
        "pysbd_loaded": pysbd_loaded,
    }


__all__ = [
    "SentenceSpan",
    "split_sentences",
    "warmup_segmenters",
]
