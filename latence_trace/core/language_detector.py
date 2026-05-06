"""Lightweight language detector used by the groundedness API.

We support exactly two languages natively right now: German and English.
The runtime ships per-language calibration bundles for German and uses the
historical English defaults for everything else, so detection is binary —
``is_german`` returns ``True`` for German, ``False`` for any other input.

Design notes:

* Backed by ``langdetect`` (pure-Python n-gram, no model download). The
  ``DetectorFactory.seed`` is pinned at import time so the same probe
  always returns the same answer; otherwise ``langdetect`` randomises
  on each call.
* Probe length matters more than text content. The library is unreliable
  on extremely short strings (``"OK"`` regularly mis-classifies as
  Portuguese). We refuse to commit to ``True`` unless the probe is at
  least ``_MIN_PROBE_CHARS`` long.
* The detector never raises. Empty / whitespace probes, exceptions from
  ``langdetect``, or any other defensive failure return ``False`` so
  callers fall back to the English path. We log the first exception per
  process so a wave of garbage input is visible in observability without
  flooding the log.
* Probe inputs are caller-supplied; pass the *minimum* text needed
  (200 chars is plenty). Don't feed entire RAG corpora through the
  detector — it's wasted CPU and doesn't change the answer.
"""

from __future__ import annotations

import logging
import threading
from typing import Final

logger = logging.getLogger(__name__)

_MIN_PROBE_CHARS: Final[int] = 30
"""Below this length langdetect is too noisy to trust. Empirically
``"OK"`` mis-classifies as ``pt`` with 0.999 confidence; 30 characters
is the conservative floor where de-vs-en becomes reliable on the
benches we care about."""

_PROBE_HARD_CAP: Final[int] = 256
"""Hard upper bound applied internally even if the caller passes more.
The detector's accuracy plateaus around 100-200 characters; longer
probes only burn CPU."""

_LOCK = threading.Lock()
_LOGGED_EXCEPTION = False


def _seed_detector_once() -> None:
    """Seed ``DetectorFactory`` once at module import for determinism.

    ``langdetect`` randomises across calls by default. We pin the seed so
    the same probe always returns the same verdict, which matters for the
    request-replay tests and for the bundle-language observability fields.
    """

    try:
        from langdetect import DetectorFactory

        DetectorFactory.seed = 42
    except Exception:  # noqa: BLE001 - dep may be missing in minimal envs
        # If langdetect isn't installed (development / test envs without
        # the optional dep), is_german() will return False on every call
        # via the import error path. The runtime falls back to English.
        logger.warning("language_detector_seed_skipped_langdetect_unavailable")


_seed_detector_once()


def is_german(text: str) -> bool:
    """Return True iff ``text`` is detected as German with high confidence.

    Public contract:

    - Empty / whitespace-only input: ``False``.
    - Input shorter than ``_MIN_PROBE_CHARS``: ``False`` (defensive; the
      detector returns garbage on tiny strings).
    - Detector exception: ``False`` (logged once per process).
    - Detector resolves to ``de`` with the highest probability: ``True``.
    - Anything else: ``False``.

    The function never raises.
    """

    global _LOGGED_EXCEPTION

    if not text:
        return False
    probe = text[:_PROBE_HARD_CAP].strip()
    if len(probe) < _MIN_PROBE_CHARS:
        return False

    try:
        from langdetect import LangDetectException, detect_langs
    except ImportError:
        return False

    try:
        results = detect_langs(probe)
    except LangDetectException:
        with _LOCK:
            if not _LOGGED_EXCEPTION:
                logger.warning("language_detector_exception")
                _LOGGED_EXCEPTION = True
        return False

    if not results:
        return False

    top = results[0]
    return getattr(top, "lang", None) == "de"


def resolve_language(
    *,
    explicit: str | None,
    response_text: str | None,
    query_text: str | None,
    raw_context: str | None,
) -> tuple[str, str]:
    """Resolve the effective language for a groundedness request.

    Probe order is response → query → raw_context. We keep each probe to
    200 characters so the per-request detector cost stays sub-5 ms even
    on long contexts.

    Returns ``(language, source)`` where:

    - ``language`` is ``"de"`` or ``"en"``.
    - ``source`` is one of ``"request"`` (caller specified ``de``/``en``),
      ``"auto"`` (langdetect resolved), or ``"fallback_en"`` (no usable
      probe text or detector returned False).
    """

    if explicit in {"de", "en"}:
        return explicit, "request"

    probe_chunks = []
    for src in (response_text, query_text, raw_context):
        if src and src.strip():
            probe_chunks.append(src[:200])
            if sum(len(c) for c in probe_chunks) >= _MIN_PROBE_CHARS:
                break
    probe = " ".join(probe_chunks).strip()
    if not probe:
        return "en", "fallback_en"
    if is_german(probe):
        return "de", "auto"
    return "en", "auto"


__all__ = ["is_german", "resolve_language"]
