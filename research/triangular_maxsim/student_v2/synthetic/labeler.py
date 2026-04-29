"""Deterministic synthetic labeler.

Given a :class:`~research.triangular_maxsim.student_v2.synthetic.enterprise_generator.SyntheticRow`
(as a dict with the standard fields), produce the full three-axis
label set the TRACE v2 student consumes:

1. ``turn_score`` (continuous in [0, 1]) - hallucination axis regression target.
2. ``turn_band`` (green / amber / red) - hallucination axis classification target.
3. ``token_support_labels`` (binary per response token) - token-support BCE target.
4. ``dead_weight_unit_labels`` (binary per evidence unit) - dead-weight BCE target.
5. ``coverage_unit_labels`` (continuous per evidence unit in [0, 1]) - coverage MSE target.

Plus the necessary auxiliary tensors the forward pass needs:

* ``evidence_units``: list of sentence-chunked units with their text.
* ``unit_assign``: which unit each evidence-token position belongs to.

The labeler is **deterministic**, **no-torch**, and **no-LLM**. It uses:

* Role + modified_slots + dropped_sentences from the synthetic generator.
* Simple sentence chunking by punctuation.
* Case-folded word overlap between response and unit (a cheap proxy for
  the teacher's per-token attribution).

Why this is good enough for Stage 1 (pure distillation):

* The synthetic generator already produced intentional hallucinations
  with known slot-level perturbations, so the *role* carries most of
  the gold-label information; the labeler just makes it explicit.
* Word-overlap is a strong proxy for the teacher's MaxSim attribution
  on short-to-medium passages; it is what the v1 TRACE literal channel
  uses under the hood.
* Stage 2 replaces the coarse synthetic labels with real teacher labels
  on external-bench rows; the labeler is not the final training signal.

Unit tests assert: deterministic output, correct role -> label mapping,
monotonic response-overlap -> coverage, drop-shaped dead-weight.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Mapping, Sequence


# ---------------------------------------------------------------------------
# Tokenization + sentence chunking
# ---------------------------------------------------------------------------


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[\.\!\?])\s+(?=[A-Z0-9])")
_TOKEN_SPLIT_RE = re.compile(r"\b[\w\-\.%]+\b")
_STOPWORDS = frozenset({
    "the", "a", "an", "of", "and", "or", "to", "in", "on", "for",
    "with", "at", "by", "as", "is", "are", "was", "were", "be", "been",
    "has", "have", "had", "will", "would", "this", "that", "these",
    "those", "it", "its", "we", "they", "our", "their", "than", "from",
    "up", "down", "into", "over", "under", "via", "more", "less",
})


def _tokenise(text: str) -> list[str]:
    return _TOKEN_SPLIT_RE.findall((text or "").lower())


def _content_tokens(tokens: Sequence[str]) -> list[str]:
    return [t for t in tokens if t and t not in _STOPWORDS and len(t) > 1]


def _split_into_sentences(text: str) -> list[str]:
    """Split text into sentence-like units. Collapses whitespace."""
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return []
    parts = _SENTENCE_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Unit chunking
# ---------------------------------------------------------------------------


@dataclass
class EvidenceUnit:
    text: str
    tokens: list[str]
    content_tokens: set[str]
    # Character offset range (inclusive start, exclusive end) in the
    # original evidence text. Useful for eval-time attribution.
    char_start: int
    char_end: int


def chunk_evidence_into_units(evidence_text: str) -> list[EvidenceUnit]:
    """Deterministically chunk evidence into sentence-level units.

    Guarantees at least one unit when ``evidence_text`` is non-empty
    (the whole text becomes a single unit).
    """
    evidence_text = evidence_text or ""
    if not evidence_text.strip():
        return []
    sentences = _split_into_sentences(evidence_text)
    if not sentences:
        sentences = [evidence_text]

    units: list[EvidenceUnit] = []
    cursor = 0
    for sent in sentences:
        # Advance cursor to the start of this sentence in the original text.
        idx = evidence_text.find(sent, cursor)
        if idx < 0:
            idx = cursor
        char_start = idx
        char_end = idx + len(sent)
        cursor = char_end
        tokens = _tokenise(sent)
        units.append(
            EvidenceUnit(
                text=sent,
                tokens=tokens,
                content_tokens=set(_content_tokens(tokens)),
                char_start=char_start,
                char_end=char_end,
            )
        )
    return units


# ---------------------------------------------------------------------------
# Label derivation per axis
# ---------------------------------------------------------------------------


# Role -> (turn_score, turn_band). Grounded variants get a high score
# with small role-specific variation; adversarials get a low-to-mid
# score depending on severity (entity_swap is more jarring than
# support_drop in practice).
_ROLE_SCORES: Mapping[str, float] = {
    "grounded": 0.88,
    "entity_swap": 0.18,
    "numeric_flip": 0.22,
    "support_drop": 0.28,
}


def _band_from_score(score: float) -> str:
    if score >= 0.65:
        return "green"
    if score <= 0.35:
        return "red"
    return "amber"


def _role_to_score(role: str, modified_slots: Sequence[str]) -> float:
    """Deterministic turn-score from role + #modified slots.

    More modified slots -> lower score (more aggressive adversarial).
    """
    base = _ROLE_SCORES.get(role, 0.5)
    n_mods = len(modified_slots or ())
    if role == "grounded":
        return base
    # For adversarials, each additional modification shaves 0.03 off the
    # score (floor at 0.05). No role exceeds its base score.
    return max(0.05, base - 0.03 * max(0, n_mods - 1))


def _token_support_from_response(
    response_tokens: Sequence[str],
    evidence_content_tokens: set[str],
    role: str,
    modified_slots: Sequence[str],
    slot_values: Mapping[str, str],
) -> list[int]:
    """Per-response-token binary support label.

    A token is supported iff its lowercased content form appears in the
    evidence's content tokens. For adversarial roles, tokens coming
    from a swapped slot value are additionally unsupported (even if
    they happen to share vocab with evidence, the specific span is
    not supported).
    """
    labels: list[int] = []
    # Build the set of "poisoned" tokens from the modified slots: the
    # substituted value in the adversarial is NOT supported by evidence.
    poisoned: set[str] = set()
    for slot in (modified_slots or ()):
        v = slot_values.get(slot, "")
        for t in _content_tokens(_tokenise(v)):
            poisoned.add(t)

    for raw in response_tokens:
        t = raw.lower()
        if not t or t in _STOPWORDS or len(t) <= 1:
            labels.append(0)
            continue
        if t in poisoned:
            labels.append(0)
            continue
        labels.append(1 if t in evidence_content_tokens else 0)

    # Hard safety: grounded rows should never have zero supported
    # tokens even if the content is very short; clamp to >=1.
    if role == "grounded" and not any(labels) and response_tokens:
        # At least mark the longest content token as supported.
        content_with_idx = [
            (i, t) for i, t in enumerate(response_tokens)
            if t and len(t) > 1 and t.lower() not in _STOPWORDS
        ]
        if content_with_idx:
            longest = max(content_with_idx, key=lambda x: len(x[1]))
            labels[longest[0]] = 1

    return labels


def _unit_overlap_metrics(
    unit: EvidenceUnit,
    response_content: set[str],
) -> tuple[float, float]:
    """Return (unit_coverage, unit_fraction_used).

    * ``unit_coverage``: fraction of unit content-tokens that appear in
      the response. Directly maps to the teacher's coverage per unit.
    * ``unit_fraction_used``: same numerator but /len(unit) tokens; for
      short units this coincides with coverage. Kept as a second
      signal so dead-weight can distinguish "fully unused" from
      "mostly unused".
    """
    if not unit.content_tokens:
        return 0.0, 0.0
    overlap = unit.content_tokens & response_content
    cov = len(overlap) / max(1, len(unit.content_tokens))
    return float(cov), float(cov)  # one metric for now; second slot reserved


def _dead_weight_labels_from_units(
    units: Sequence[EvidenceUnit],
    response_content: set[str],
    role: str,
    dropped_sentences: Sequence[str],
    dead_threshold: float = 0.15,
) -> list[int]:
    """Binary dead-weight per unit.

    A unit is dead iff < ``dead_threshold`` of its content tokens are
    referenced by the response. In practice:

    * grounded: few-to-zero dead units (response covers the passage).
    * entity_swap / numeric_flip: zero dead units (same semantic overlap,
      one wrong fact; the evidence is still broadly referenced).
    * support_drop: the response talks about a fact whose source was
      dropped, so the remaining units tend to be fully referenced but
      coverage is imperfect (signal is on the coverage axis, not
      dead-weight). **Unless** the drop was from a multi-sentence unit,
      in which case one unit may end up less-referenced.

    The ``dropped_sentences`` tuple is informational only for this
    labeler: the dropped sentence is already absent from ``units``.
    """
    labels: list[int] = []
    for unit in units:
        cov, _ = _unit_overlap_metrics(unit, response_content)
        labels.append(1 if cov < dead_threshold else 0)
    return labels


def _coverage_labels_from_units(
    units: Sequence[EvidenceUnit],
    response_content: set[str],
) -> list[float]:
    """Per-unit continuous coverage in [0, 1]."""
    out: list[float] = []
    for unit in units:
        cov, _ = _unit_overlap_metrics(unit, response_content)
        out.append(cov)
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class LabelledRow:
    """Fully-labelled synthetic row, ready for training."""

    pair_id: str
    class_key: str
    split_hint: str
    role: str
    paraphrase_variant: str

    response_text: str
    response_tokens: list[str]
    evidence_text: str
    evidence_units: list[EvidenceUnit]

    # Hallucination axis.
    turn_score: float
    turn_band: str
    gold_band: str  # binary: "green" (grounded) / "red" (ungrounded)

    # Token-support axis (response-side).
    token_support_labels: list[int]

    # Evidence-side axes.
    dead_weight_unit_labels: list[int]
    coverage_unit_labels: list[float]

    # Extras for auditability.
    modified_slots: tuple[str, ...] = ()
    dropped_sentences: tuple[str, ...] = ()
    slot_values: Mapping[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "pair_id": self.pair_id,
            "class_key": self.class_key,
            "split_hint": self.split_hint,
            "role": self.role,
            "paraphrase_variant": self.paraphrase_variant,
            "response_text": self.response_text,
            "response_tokens": list(self.response_tokens),
            "evidence_text": self.evidence_text,
            "evidence_units": [
                {"text": u.text, "tokens": list(u.tokens),
                 "char_start": u.char_start, "char_end": u.char_end}
                for u in self.evidence_units
            ],
            "turn_score": float(self.turn_score),
            "turn_band": self.turn_band,
            "gold_band": self.gold_band,
            "token_support_labels": list(self.token_support_labels),
            "dead_weight_unit_labels": list(self.dead_weight_unit_labels),
            "coverage_unit_labels": [float(x) for x in self.coverage_unit_labels],
            "modified_slots": list(self.modified_slots),
            "dropped_sentences": list(self.dropped_sentences),
            "slot_values": dict(self.slot_values),
        }


def label_synthetic_row(row: Mapping[str, object]) -> LabelledRow:
    """Deterministically label a synthetic enterprise-lane row.

    Args:
        row: a dict produced by ``SyntheticRow.to_dict()`` (or
            equivalent shape).

    Returns:
        A :class:`LabelledRow` carrying all five axis labels.
    """
    role = str(row.get("role") or "grounded")
    response_text = str(row.get("response_text") or "")
    evidence_text = str(row.get("evidence_text") or "")
    modified_slots = tuple(row.get("modified_slots") or ())
    dropped_sentences = tuple(row.get("dropped_sentences") or ())
    slot_values_raw = row.get("slot_values") or {}
    # Defensive copy to a plain dict.
    slot_values: dict[str, str] = {
        str(k): str(v) for k, v in (slot_values_raw.items() if hasattr(slot_values_raw, 'items') else [])
    }

    # --- Tokenise + chunk ---
    response_tokens = _tokenise(response_text)
    response_content = set(_content_tokens(response_tokens))
    units = chunk_evidence_into_units(evidence_text)

    # --- Hallucination axis ---
    turn_score = _role_to_score(role, modified_slots)
    turn_band = _band_from_score(turn_score)
    gold_band = "green" if role == "grounded" else "red"

    # --- Token support ---
    ev_content = set().union(*(u.content_tokens for u in units)) if units else set()
    token_support = _token_support_from_response(
        response_tokens, ev_content, role, modified_slots, slot_values,
    )

    # --- Evidence-side axes ---
    dead = _dead_weight_labels_from_units(
        units, response_content, role, dropped_sentences,
    )
    cov = _coverage_labels_from_units(units, response_content)

    return LabelledRow(
        pair_id=str(row.get("pair_id") or ""),
        class_key=str(row.get("class_key") or ""),
        split_hint=str(row.get("split_hint") or "train"),
        role=role,
        paraphrase_variant=str(row.get("paraphrase_variant") or ""),
        response_text=response_text,
        response_tokens=response_tokens,
        evidence_text=evidence_text,
        evidence_units=units,
        turn_score=turn_score,
        turn_band=turn_band,
        gold_band=gold_band,
        token_support_labels=token_support,
        dead_weight_unit_labels=dead,
        coverage_unit_labels=cov,
        modified_slots=modified_slots,
        dropped_sentences=dropped_sentences,
        slot_values=slot_values,
    )


__all__ = [
    "EvidenceUnit",
    "LabelledRow",
    "chunk_evidence_into_units",
    "label_synthetic_row",
]
