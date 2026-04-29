"""Extract the 5 student axis labels from a teacher response dict.

The input is the JSON body returned by the dev_app ``/runsync``
endpoint (a flat dict carrying the compact summary plus a ``full``
sub-dict with the typed GroundednessResponse). We walk it with dict
access so the extractor works independently from the Pydantic schema
(which is useful for unit tests and future wire format changes).

The output shape matches what the student's training loop consumes:

* ``turn_score``              — float in [0, 1]
* ``turn_band``               — int in {0, 1, 2}  (0 = green, 1 = amber, 2 = red)
* ``token_support_labels``    — list[int] aligned to response tokens
* ``dead_weight_unit_labels`` — list[int] aligned to support units
* ``coverage_unit_labels``    — list[int] aligned to support units
* ``evidence_units``          — list[str] (sentence-packed support unit texts)

Why ``reverse_context_calibrated``?
-----------------------------------
The raw ``reverse_context`` / ``heatmap_score`` MaxSim values are bounded
in [0, 1] but sit very high (> 0.8) even on completely hallucinated tokens
because ColBERT always finds *some* similar embedding. The calibrated
channel divides out the null-bank mean so unsupported tokens drop to
near-zero. That is the signal the student needs to learn.
"""

from __future__ import annotations

import logging
from typing import Mapping

logger = logging.getLogger("trace.v2.teacher_label.extract")

_BAND_TO_INT = {"green": 0, "amber": 1, "red": 2, "unknown": 1, "unsupported": 2}
_INT_TO_BAND = {0: "green", 1: "amber", 2: "red"}
_DEFAULT_GREEN_THRESHOLD = 0.72
_DEFAULT_RED_THRESHOLD = 0.38
# Per-token thresholds. The teacher's `nli_score` is the propagated
# claim-level entailment (in [-1, 1]): strongly negative means the
# claim was contradicted by the evidence, strongly positive means it
# was entailed. We keep a narrow neutral band around 0 so only clearly
# supported tokens get label=1; the rest get 0 (treated as "not
# reliably grounded" by the student).
_TOKEN_NLI_GROUNDED_THRESHOLD = 0.30
# Fallback when NLI did not fire for a token (e.g. boundary / non-claim
# tokens like [CLS], [SEP]): the calibrated MaxSim channel with a more
# conservative threshold.
_TOKEN_CALIBRATED_THRESHOLD = 0.70
_COVERAGE_THRESHOLD = 0.40


def _dig(d: object, *path: str) -> object:
    """Walk nested dict/list with safe None-passthrough."""

    for step in path:
        if d is None:
            return None
        if isinstance(d, Mapping):
            d = d.get(step)
        else:
            return None
    return d


def _risk_band(resp: dict) -> str:
    band = (resp.get("band") or _dig(resp, "full", "risk_band") or "").strip().lower()
    if band in _BAND_TO_INT:
        return band
    score = _turn_score(resp)
    if score >= _DEFAULT_GREEN_THRESHOLD:
        return "green"
    if score <= _DEFAULT_RED_THRESHOLD:
        return "red"
    return "amber"


def _turn_score(resp: dict) -> float:
    for key in ("score", "nli_aggregate"):
        v = resp.get(key)
        if v is not None:
            return max(0.0, min(1.0, float(v)))
    scores = _dig(resp, "full", "scores") or {}
    if isinstance(scores, Mapping):
        for attr in (
            "groundedness_v2",
            "primary_score",
            "reverse_context_calibrated",
            "reverse_context",
        ):
            v = scores.get(attr)
            if v is not None:
                return max(0.0, min(1.0, float(v)))
    return 0.0


def _response_token_labels(resp: dict) -> tuple[list[str], list[int]]:
    """Per-token binary "grounded" labels.

    Priority:
      1) ``nli_score`` when present (claim-level entailment propagated to
         tokens). This is the cleanest signal we have — on
         ``"The Eiffel Tower is in Berlin."`` every content token carries
         ``nli_score ≈ -0.99`` after the teacher's NLI cascade. We
         threshold at +0.30 so only clearly entailed tokens get label=1.
      2) ``reverse_context_calibrated`` as a fallback for boundary /
         non-claim tokens ([CLS], [SEP], punctuation) that NLI skips.
    """

    tokens = _dig(resp, "full", "response_tokens") or []
    out_tokens: list[str] = []
    out_labels: list[int] = []
    for tok in tokens:
        if not isinstance(tok, Mapping):
            continue
        text = str(tok.get("token") or "")
        nli_score = tok.get("nli_score")
        if nli_score is not None:
            label = 1 if float(nli_score) >= _TOKEN_NLI_GROUNDED_THRESHOLD else 0
        else:
            calibrated = tok.get("reverse_context_calibrated")
            if calibrated is None:
                calibrated = tok.get("reverse_context") or 0.0
            label = 1 if float(calibrated) >= _TOKEN_CALIBRATED_THRESHOLD else 0
        out_tokens.append(text)
        out_labels.append(label)
    return out_tokens, out_labels


def _find_span(haystack: str, needle: str, cursor: int) -> tuple[int, int]:
    """Return (char_start, char_end) for ``needle`` inside ``haystack``.

    Walks forward from ``cursor`` so repeated phrases map to distinct
    locations in source order. Falls back to (cursor, cursor + len) on
    miss so we never crash — the downstream collate clamps spans.
    """

    if not needle:
        return (cursor, cursor)
    idx = haystack.find(needle, cursor)
    if idx < 0:
        # Retry from the start: teacher chunker may have reordered
        # trailing fragments.
        idx = haystack.find(needle)
        if idx < 0:
            return (cursor, cursor + len(needle))
    return (idx, idx + len(needle))


def _unit_labels(
    resp: dict, *, evidence_text: str = ""
) -> tuple[list[dict], list[int], list[int]]:
    """Return (evidence_units, dead_labels, coverage_labels).

    ``evidence_units`` follows the synthetic-lane schema:
    ``{"text": str, "char_start": int, "char_end": int, "tokens": []}``
    so the student's collate can align labels via offsets.
    """

    compact_units = resp.get("support_units") or []
    full_units = _dig(resp, "full", "support_units") or []
    text_by_id: dict[str, str] = {}
    for u in full_units:
        if isinstance(u, Mapping):
            text_by_id[str(u.get("support_id"))] = str(u.get("text") or "")

    units_out: list[dict] = []
    dead: list[int] = []
    coverage: list[int] = []
    cursor = 0
    ordered = compact_units if compact_units else full_units
    for u in ordered:
        if not isinstance(u, Mapping):
            continue
        sid = str(u.get("support_id"))
        text = text_by_id.get(sid) or str(u.get("text") or "")
        state = str(u.get("usage_state") or "").lower()
        used = bool(u.get("used", False))
        cov_score = float(u.get("coverage_score") or 0.0)
        char_start, char_end = _find_span(evidence_text, text, cursor)
        cursor = char_end
        units_out.append(
            {
                "text": text,
                "tokens": [],
                "char_start": char_start,
                "char_end": char_end,
            }
        )
        if state == "unused":
            dead.append(1)
        elif state == "used":
            dead.append(0)
        else:
            dead.append(0 if used else 1)
        coverage.append(1 if cov_score >= _COVERAGE_THRESHOLD else 0)
    return units_out, dead, coverage


def extract_labels(
    resp: dict,
    *,
    gold_band: str | None = None,
    evidence_text: str = "",
) -> dict:
    """Return the 5-axis label dict for the teacher response ``resp``.

    Passing ``evidence_text`` lets the extractor compute char spans on
    each support unit so downstream collate can align per-unit labels
    to tokenizer offsets. When not supplied we emit char_start=0 /
    char_end=len(text) which the collate will treat as whole-evidence.
    """

    teacher_band = _risk_band(resp)
    effective_band = (gold_band or "").strip().lower()
    if effective_band not in _BAND_TO_INT:
        effective_band = teacher_band

    tokens, token_labels = _response_token_labels(resp)
    evidence_units, dead_labels, coverage_labels = _unit_labels(
        resp, evidence_text=evidence_text
    )

    # Emit band as its string name for downstream consistency with the
    # synthetic labeler. ``collate.BAND_TO_IDX`` maps back to the int.
    band_idx = _BAND_TO_INT.get(effective_band, 1)

    return {
        "turn_score": _turn_score(resp),
        "turn_band": _INT_TO_BAND.get(band_idx, "amber"),
        "teacher_band": teacher_band,
        "nli_aggregate": resp.get("nli_aggregate"),
        "response_tokens_teacher": tokens,
        "token_support_labels": token_labels,
        "evidence_units": evidence_units,
        "dead_weight_unit_labels": dead_labels,
        "coverage_unit_labels": coverage_labels,
        "n_support_units": len(evidence_units),
        "n_response_tokens": len(tokens),
    }
