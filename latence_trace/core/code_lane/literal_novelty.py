"""Per-identifier first-match novelty score.

The legacy ``literal_guard`` flag counts any occurrence of a response
identifier anywhere in context as a match — but a phantom import of
``photon_labs`` can still "match" when the context mentions the word
``photon`` in a doc-string. The novelty guard tightens this by looking
at *where* an identifier first matches: if it only matches a low
``reverse_context`` support unit the identifier is suspicious even
though the binary literal-guard flag stays clean.

Public API
----------
``compute_literal_novelty(response_literals, support_units, per_unit_score)``
returns :class:`LiteralNoveltyResult`. The minimum first-match score
(``literal_novelty_min``) is the one we feed into the composite; the
full per-identifier breakdown is preserved for the diagnostics payload.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class LiteralNoveltyResult:
    """Per-turn novelty diagnostics for the literal guard."""

    literal_novelty_min: float
    literal_novelty_mean: float
    literal_novelty_count: int
    missing_literal_count: int
    per_literal: List[Dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "literal_novelty_min": float(self.literal_novelty_min),
            "literal_novelty_mean": float(self.literal_novelty_mean),
            "literal_novelty_count": int(self.literal_novelty_count),
            "missing_literal_count": int(self.missing_literal_count),
            "per_literal": list(self.per_literal),
        }


def compute_literal_novelty(
    *,
    response_literals: Sequence[str],
    support_units: Sequence[Any],
    per_unit_score: Sequence[float],
    missing_penalty: float = 0.0,
) -> LiteralNoveltyResult:
    """Score every response literal against the first support unit it
    appears in.

    Parameters
    ----------
    response_literals:
        Identifiers / literals pulled from the response (the caller
        passes the same set used for ``literal_guard``).
    support_units:
        Sequence whose elements carry a ``tokens`` attribute / dict key
        (the lower-cased token list from the encoder). Supports both
        :class:`~latence_trace.core.code_lane.types.SupportUnitPack` and
        the RAG :class:`~latence_trace.core.groundedness.SupportUnitInput`
        structs.
    per_unit_score:
        One score per unit, same length as ``support_units``. Use
        :attr:`~latence_trace.core.code_lane.gpu_scorer.ScorerOutput.per_unit_max`.
    missing_penalty:
        Score assigned to literals with no matching unit. ``0.0``
        (default) reads as "maximally suspicious"; bump to 0.5 if you
        prefer a softer signal during tuning.
    """
    if not response_literals:
        return LiteralNoveltyResult(
            literal_novelty_min=1.0,
            literal_novelty_mean=1.0,
            literal_novelty_count=0,
            missing_literal_count=0,
        )

    unit_tokens: List[set[str]] = []
    for unit in support_units:
        tokens = getattr(unit, "tokens", None)
        if tokens is None and isinstance(unit, dict):
            tokens = unit.get("tokens")
        if tokens is None:
            unit_tokens.append(set())
            continue
        unit_tokens.append({str(t).lower() for t in tokens if t})

    per_literal: List[Dict[str, Any]] = []
    missing = 0
    first_match_scores: List[float] = []
    for literal in response_literals:
        needle = str(literal).strip().lower()
        if not needle:
            continue
        best_idx: Optional[int] = None
        best_score: Optional[float] = None
        for idx, tokens in enumerate(unit_tokens):
            if needle in tokens:
                score = float(per_unit_score[idx]) if idx < len(per_unit_score) else 0.0
                if best_score is None or score > best_score:
                    best_score = score
                    best_idx = idx
        if best_score is None:
            missing += 1
            per_literal.append(
                {
                    "literal": needle,
                    "first_match_unit_index": None,
                    "first_match_score": float(missing_penalty),
                    "status": "missing",
                }
            )
            first_match_scores.append(float(missing_penalty))
        else:
            per_literal.append(
                {
                    "literal": needle,
                    "first_match_unit_index": int(best_idx or 0),
                    "first_match_score": float(best_score),
                    "status": "matched",
                }
            )
            first_match_scores.append(float(best_score))

    if not first_match_scores:
        return LiteralNoveltyResult(
            literal_novelty_min=1.0,
            literal_novelty_mean=1.0,
            literal_novelty_count=0,
            missing_literal_count=missing,
        )

    return LiteralNoveltyResult(
        literal_novelty_min=float(min(first_match_scores)),
        literal_novelty_mean=float(sum(first_match_scores) / len(first_match_scores)),
        literal_novelty_count=len(first_match_scores),
        missing_literal_count=missing,
        per_literal=per_literal,
    )
