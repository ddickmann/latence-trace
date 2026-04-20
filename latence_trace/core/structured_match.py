"""AND-gate cell matcher for the typed structured-evidence lane.

Given a list of typed source cells (extracted by
:func:`latence_trace.core.structured.extract_typed_cells`) and a list of
typed response claims (extracted by
:func:`latence_trace.core.typed_claims.extract_typed_claims`), align
each claim to its best-matching cell and score the alignment with an
AND-gate:

- ``entity_align``: token-overlap between the claim anchor and the cell
  anchor, snapped to ``{0.0, 0.5, 1.0}`` so weak alignment cannot
  rescue a numeric mismatch.
- ``value_match``: exact for table cells, tolerance-bounded for
  approximate prose claims.
- ``unit_match``: hard ``0/1`` check on canonical unit (``M`` vs
  ``B`` vs ``PCT`` vs ``BPS``).
- ``sign_match``: hard ``0/1`` check on the direction verb against the
  cell's period-over-period sign.

Per-claim score = ``min(entity, value, unit, sign)`` (the user's
explicit AND-gate rule). Aggregate score = ``min`` across all aligned
claims so a single broken cell collapses the structured score.

The matcher returns ``None`` when no claim could be aligned (entity
overlap below threshold for all candidates), so the fusion layer
correctly silences this lane on pure-prose contexts and on responses
that simply paraphrase the question.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from latence_trace.core.structured import TypedCell
from latence_trace.core.typed_claims import TypedClaim

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Public dataclasses
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class MatchedClaim:
    """One typed claim's match outcome against the source cells."""

    claim_anchor: str
    claim_value: Optional[float]
    claim_unit: str
    claim_currency: str
    claim_sign: str
    matched_cell_anchor: Optional[str]
    matched_cell_value: Optional[float]
    matched_cell_unit: str
    matched_cell_currency: str
    matched_cell_sign: str
    entity_align: float
    value_match: float
    unit_match: float
    sign_match: float
    per_claim_score: float
    failure_reason: Optional[str] = None


@dataclass(frozen=True)
class StructuredMatchResult:
    """Aggregated typed-cell match result for the whole response."""

    score: Optional[float]
    aligned: int
    dropped: int
    per_claim: List[MatchedClaim]


# ----------------------------------------------------------------------
# Tunables
# ----------------------------------------------------------------------


#: Minimum entity overlap needed to *consider* a claim aligned. Below
#: this we drop the claim (rather than score it as 0) so unrelated
#: prose claims don't poison the structured score.
_ENTITY_DROP_THRESHOLD = 0.35

#: Entity-alignment levels: snap to {0, 0.5, 1.0} so weak overlap
#: cannot rescue a numeric mismatch.
_ENTITY_LEVELS = (0.0, 0.5, 1.0)

#: Default relative tolerance for approximate-language value matching
#: (``"approximately"`` / ``"about"`` / ``"etwa"``). For exact table
#: cells we use a much tighter tolerance.
_APPROX_REL_TOLERANCE = 0.05

#: Tight relative tolerance applied to all numeric comparisons (covers
#: rounding noise such as ``"$26,272 million"`` vs source value
#: ``26272.0``).
_EXACT_REL_TOLERANCE = 0.005


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


_TOKEN_SPLIT_RE = re.compile(r"[\s\-/_,;:]+")


def _token_set(text: str) -> set:
    """Split into matchable tokens.

    Splits on whitespace and the joiners commonly seen in compound nouns
    (``"Software-Lizenzen"`` -> ``{"software","lizenzen"}``,
    ``"Cloud-Umsatz"`` -> ``{"cloud","umsatz"}``). Token must be at
    least 2 chars and contain at least one alpha character so we don't
    pick up ``"q2"`` or ``"2024"`` as entity tokens.
    """

    if not text:
        return set()
    tokens = _TOKEN_SPLIT_RE.split(text)
    out = set()
    for tok in tokens:
        tok = tok.strip(".,;:()[]")
        if not tok or len(tok) < 2:
            continue
        if not any(ch.isalpha() for ch in tok):
            continue
        out.add(tok.lower())
    return out


def _entity_overlap(claim_anchor: str, cell_anchor: str) -> float:
    """Jaccard-style overlap snapped to ``{0.0, 0.5, 1.0}``.

    Rules:
      - ``1.0`` when claim anchor is a superset of cell anchor (cell
        tokens fully appear in the claim) or vice versa.
      - ``0.5`` when the Jaccard >= 0.4.
      - ``0.0`` otherwise.

    Snapping prevents a fuzzy 0.6 entity overlap from dragging an
    otherwise solid (value=1, unit=1, sign=1) match down by 40 %.
    The matcher is ``min``-aggregated so the snap directly controls
    the AND-gate.
    """

    a = _token_set(claim_anchor)
    b = _token_set(cell_anchor)
    if not a or not b:
        return 0.0
    inter = a & b
    if not inter:
        return 0.0
    if b.issubset(a) or a.issubset(b):
        return 1.0
    jaccard = len(inter) / float(len(a | b))
    if jaccard >= 0.5:
        return 1.0
    if jaccard >= 0.3 and len(inter) >= 1:
        return 0.5
    return 0.0


def _value_match_score(
    claim_value: Optional[float],
    cell_value: Optional[float],
    *,
    approximate: bool,
    cell_paired: Optional[float] = None,
) -> Tuple[float, str]:
    """Return ``(value_match, failure_reason)``.

    Hard 0/1 by default; widened to ``[0, 1]`` only when the claim was
    flagged ``approximate``. Falls back to ``1`` when no claim value is
    asserted (the claim is purely directional).
    """

    if claim_value is None:
        return 1.0, ""
    if cell_value is None:
        return 0.0, "no_cell_value"
    rel_tol = _APPROX_REL_TOLERANCE if approximate else _EXACT_REL_TOLERANCE
    diff = abs(claim_value - cell_value)
    denom = max(1e-6, abs(cell_value))
    if diff / denom <= rel_tol:
        return 1.0, ""
    # Allow the claim to refer to the *paired* (prior-period) value
    # when the source carries one, but only with the same tolerance --
    # otherwise a "from A to B" claim that mis-states A still passes.
    if cell_paired is not None:
        diff_p = abs(claim_value - cell_paired)
        denom_p = max(1e-6, abs(cell_paired))
        if diff_p / denom_p <= rel_tol:
            return 1.0, ""
    if approximate:
        # Linear penalty between exact-tolerance and 2x tolerance, then
        # zero. This is the only soft component in the AND-gate.
        slack = (diff / denom - rel_tol) / max(rel_tol, 1e-6)
        score = max(0.0, 1.0 - slack)
        if score == 0.0:
            return 0.0, "value_mismatch"
        return float(score), "" if score >= 0.5 else "value_drift"
    return 0.0, "value_mismatch"


_UNIT_FAMILIES = {
    "MONEY_OR_COUNT": {"M", "B", "TRN", "TSD", "NONE", "COUNT", "MIO"},
    "PERCENT": {"PCT"},
    "BPS": {"BPS"},
    "ENERGY": {"GWH", "TWH", "MWH", "KWH"},
    "POINTS": {"POINTS"},
}


def _unit_family(unit: str) -> str:
    u = (unit or "NONE").upper()
    for fam, members in _UNIT_FAMILIES.items():
        if u in members:
            return fam
    return "OTHER"


def _unit_match_score(claim_unit: str, cell_unit: str, claim_currency: str, cell_currency: str) -> Tuple[float, str]:
    """Hard 0/1 unit and currency check.

    The unit check only fires when the claim and cell live in
    *different unit families* (e.g. ``PCT`` vs ``M``, ``BPS`` vs
    ``PCT``, ``GWH`` vs ``M``). Within the money-or-count family
    (``M``, ``B``, ``TRN``, ``TSD``, ``COUNT``, ``NONE``) the suffix
    is just a labelling convention -- after magnitude normalisation
    ``"$30.04 billion"`` and ``"$30,040 million"`` carry the same
    value, so a hard unit-mismatch would be a false negative. The
    value-match check (post-normalisation) is what actually catches
    the ``$30,040 million`` vs ``$30,040 billion`` adversarial case.

    Special cases preserved:
      - ``"NONE"`` on the response side matches anything (a typed
        claim that omitted the unit is not a unit-mismatch by itself).
      - ``"NONE"`` on the source side means "no unit declared" -- we
        only fail when the claim asserts a unit that contradicts the
        source's announcement (``"BPS"`` vs unit-less percent).
      - ``"BPS"`` and ``"PCT"`` are different families: a ``"BPS"``
        claim must match a ``"BPS"`` cell.
      - Currency mismatch is a hard 0 only when both sides assert one.
    """

    cu = (claim_unit or "NONE").upper()
    su = (cell_unit or "NONE").upper()
    if cu == "NONE" or su == "NONE":
        unit_ok = 1.0
    else:
        cu_fam = _unit_family(cu)
        su_fam = _unit_family(su)
        unit_ok = 1.0 if cu_fam == su_fam else 0.0

    cc = (claim_currency or "").upper()
    sc = (cell_currency or "").upper()
    if cc and sc and cc != sc:
        return 0.0, "currency_mismatch"
    return unit_ok, ("" if unit_ok > 0 else "unit_mismatch")


def _sign_match_score(claim_sign: str, cell_sign: str) -> Tuple[float, str]:
    """Hard 0/1 direction check.

    Cell sign is ``NEUTRAL`` when no paired comparison value is
    available; in that case any claim sign is permitted (the lane
    cannot judge direction without a prior period to compare against).
    """

    if cell_sign == "NEUTRAL" or claim_sign == "NEUTRAL":
        return 1.0, ""
    if claim_sign == cell_sign:
        return 1.0, ""
    return 0.0, "sign_mismatch"


# ----------------------------------------------------------------------
# Matcher
# ----------------------------------------------------------------------


def _best_cell_for_claim(
    claim: TypedClaim, cells: Sequence[TypedCell]
) -> Tuple[Optional[TypedCell], float]:
    """Pick the cell with the highest entity overlap.

    Ranking (in order):
      1. Entity-overlap score (snapped to ``{0.0, 0.5, 1.0}``).
      2. Same-unit-family preference: a ``PCT`` claim should match
         the ``"growth 122%"`` cell, not the ``"Total revenue $30B"``
         cell. Without this tiebreaker the entity overlap alone wins
         on the longer-token cell and the AND-gate then fails the
         value/unit checks.
      3. Closeness in numeric value (so the ``"Audi"`` claim aligns
         to the ``"Audi"`` cell even when ``"Volkswagen"`` is in the
         same table).
    """

    claim_fam = _unit_family(claim.unit)
    best: Optional[TypedCell] = None
    best_score = 0.0
    best_family_match = -1
    best_value_diff = math.inf
    for cell in cells:
        score = _entity_overlap(claim.anchor_norm, cell.anchor_norm)
        if score < _ENTITY_DROP_THRESHOLD:
            continue
        cell_fam = _unit_family(cell.unit)
        family_match = 1 if cell_fam == claim_fam else 0
        diff = math.inf
        if claim.value is not None and cell.value is not None and cell.value != 0:
            diff = abs(claim.value - cell.value) / max(1e-6, abs(cell.value))
        better = False
        if score > best_score:
            better = True
        elif score == best_score:
            if family_match > best_family_match:
                better = True
            elif family_match == best_family_match and diff < best_value_diff:
                better = True
        if better:
            best_score = score
            best_family_match = family_match
            best_value_diff = diff
            best = cell
    return best, best_score


def structured_score_from_claims(
    claims: Sequence[TypedClaim],
    cells: Sequence[TypedCell],
) -> StructuredMatchResult:
    """Run the AND-gate matcher over typed claims and source cells.

    Returns ``score=None`` when no claim could be aligned, so the
    fusion layer can keep its weighted-sum behaviour for pure-prose
    contexts and only escalate to ``min(narrative, structured)`` when
    the typed lane is actually scoring something.
    """

    matched: List[MatchedClaim] = []
    aligned = 0
    dropped = 0
    if not claims or not cells:
        return StructuredMatchResult(score=None, aligned=0, dropped=len(claims), per_claim=[])

    for claim in claims:
        cell, entity_score = _best_cell_for_claim(claim, cells)
        if cell is None or entity_score <= 0:
            dropped += 1
            continue

        value_score, value_reason = _value_match_score(
            claim.value,
            cell.value,
            approximate=claim.approximate,
            cell_paired=cell.paired_value,
        )
        unit_score, unit_reason = _unit_match_score(
            claim.unit, cell.unit, claim.currency, cell.currency
        )
        sign_score, sign_reason = _sign_match_score(claim.sign, cell.sign)

        per_claim = min(entity_score, value_score, unit_score, sign_score)
        failure_reason = None
        if per_claim < 1.0:
            for reason in (value_reason, unit_reason, sign_reason):
                if reason:
                    failure_reason = reason
                    break
            if failure_reason is None and entity_score < 1.0:
                failure_reason = "weak_entity_align"
        matched.append(
            MatchedClaim(
                claim_anchor=claim.anchor,
                claim_value=claim.value,
                claim_unit=claim.unit,
                claim_currency=claim.currency,
                claim_sign=claim.sign,
                matched_cell_anchor=cell.anchor,
                matched_cell_value=cell.value,
                matched_cell_unit=cell.unit,
                matched_cell_currency=cell.currency,
                matched_cell_sign=cell.sign,
                entity_align=float(entity_score),
                value_match=float(value_score),
                unit_match=float(unit_score),
                sign_match=float(sign_score),
                per_claim_score=float(per_claim),
                failure_reason=failure_reason,
            )
        )
        aligned += 1

    if aligned == 0:
        return StructuredMatchResult(score=None, aligned=0, dropped=dropped, per_claim=[])

    # AND-gate aggregate: ``min`` over all aligned per-claim scores so
    # one broken cell kills support.
    per_claim_scores = [m.per_claim_score for m in matched]
    aggregate = min(per_claim_scores)
    return StructuredMatchResult(
        score=float(aggregate),
        aligned=aligned,
        dropped=dropped,
        per_claim=matched,
    )


def matched_to_dicts(matched: Sequence[MatchedClaim]) -> List[Dict[str, object]]:
    """Serialize per-claim match results for the diagnostic payload."""

    out: List[Dict[str, object]] = []
    for entry in matched:
        out.append(
            {
                "claim_anchor": entry.claim_anchor,
                "claim_value": entry.claim_value,
                "claim_unit": entry.claim_unit,
                "claim_currency": entry.claim_currency,
                "claim_sign": entry.claim_sign,
                "matched_cell_anchor": entry.matched_cell_anchor,
                "matched_cell_value": entry.matched_cell_value,
                "matched_cell_unit": entry.matched_cell_unit,
                "matched_cell_currency": entry.matched_cell_currency,
                "matched_cell_sign": entry.matched_cell_sign,
                "entity_align": entry.entity_align,
                "value_match": entry.value_match,
                "unit_match": entry.unit_match,
                "sign_match": entry.sign_match,
                "per_claim_score": entry.per_claim_score,
                "failure_reason": entry.failure_reason,
            }
        )
    return out


__all__ = [
    "MatchedClaim",
    "StructuredMatchResult",
    "structured_score_from_claims",
    "matched_to_dicts",
]
