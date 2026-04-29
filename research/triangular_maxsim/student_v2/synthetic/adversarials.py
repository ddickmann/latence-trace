"""Adversarial perturbations for the synthetic enterprise lane.

Three deterministic transforms are applied to ``(passage, response,
slot_values)`` triples to produce the three adversarial shapes called
out in the plan:

* :func:`entity_swap`  — replace one named entity slot (company,
  counterparty, brand, auditor, etc.) in the response with a same-type
  distractor drawn from the same sub-domain's bank but guaranteed not
  to equal the source entity. Dominant RAGTruth failure mode.
* :func:`numeric_flip` — perturb one numeric slot in the response by a
  small amount chosen to cross a calibration boundary (e.g. ±1.4 on a
  percentage, ±10-25% on monetary amounts), rather than a trivial
  rounding change. Grounds the numeric-match phi bit.
* :func:`support_drop` — remove one sentence of the passage that
  uniquely supports one claim in the response, so the claim becomes
  truly unsupported even though the response text is unchanged.
  Grounds the evidence-side dead-weight head.

Every transform is driven by an explicit ``random.Random`` so the
same (seed, shape) produces the same output row-by-row. Transforms
return a :class:`AdversarialOutput` that records:

* ``response`` / ``passage`` - potentially perturbed text.
* ``modified_slots`` - the slot name(s) changed in the response.
* ``dropped_sentences`` - the passage sentences removed (for
  support_drop; empty otherwise).
* ``notes`` - free-text description used in eval bundle logs.

Failures to apply the transform (no eligible slot, no safe flip
target, etc.) return ``None`` so the caller can skip and keep the
grounded pair only.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Mapping, Optional

from research.triangular_maxsim.student_v2.synthetic.industry_banks import (
    IndustryBank,
    get_industry,
)


# Slot names treated as entity swaps (must resolve to a string bank
# under ``industry.entities[sub_domain]``). Everything else is either
# a number slot or a shared default and not swappable.
ENTITY_SWAPPABLE_SLOTS: frozenset[str] = frozenset({
    "company", "counterparty", "brand", "competitor", "carrier",
    "auditor", "processor", "sponsor", "principal_investigator",
    "agency", "analyst_firm", "attorney", "executive_title",
    "remediation_owner", "committee", "team", "system", "service",
    "policy_name", "audience", "channel", "product_line",
    "provider_type", "regimen", "rfc_label", "campaign_name",
    "contract_type", "jurisdiction", "case_cite", "doctrine",
    "statute", "framework",
})

# Slot names that must always be treated as numeric flips. These
# correspond to entries in ``industry.numerics`` *and* to common
# shared defaults.
NUMERIC_SLOT_HINTS: frozenset[str] = frozenset({
    "revenue_billions", "growth_pct", "margin_pct", "eps_usd",
    "installed_base_billions", "var_usd_millions", "breach_count",
    "remediation_days", "capital_ratio_pct", "term_months",
    "notice_days", "liability_cap_millions", "termination_cure_days",
    "damages_millions", "citation_year", "match_pct",
    "deductible_usd", "leave_weeks", "stipend_usd",
    "review_interval_months", "retention_months", "dpia_score",
    "subject_request_days", "finding_count", "remediation_sla_days",
    "slo_p99_ms", "throughput_rps", "replication_lag_ms",
    "timeout_seconds", "retry_count", "error_budget_pct",
    "budget_millions", "target_mqls", "ctr_pct", "market_share_pct",
    "nps", "enrollment_count", "followup_months", "completion_pct",
    "follow_up_weeks", "adherence_pct", "dose_mg", "hr_ratio",
    "award_usd_millions", "period_of_performance_months",
    "indirect_rate_pct", "gdp_impact_pct", "beneficiary_thousands",
})


@dataclass
class AdversarialOutput:
    kind: str  # "entity_swap" | "numeric_flip" | "support_drop"
    passage: str
    response: str
    slot_values: Mapping[str, str]
    modified_slots: tuple[str, ...] = ()
    dropped_sentences: tuple[str, ...] = ()
    notes: str = ""
    # The evidence-side label vector we *intend* to teach: False ->
    # well-used, True -> dead-weight. This is a hint; the authoritative
    # labels come from the teacher scoring pass. For synthetic adversarials
    # we use this to sanity-check the teacher output.
    intended_unit_dead_weight: tuple[bool, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9$])")


def _split_sentences(text: str) -> list[str]:
    """Split on sentence boundaries robust to leading numbers / dollar signs.

    The synthetic passages are well-formed; we just need the split to
    be stable so ``support_drop`` can remove one sentence reliably.
    """
    parts = [p for p in _SENT_SPLIT_RE.split(text.strip()) if p]
    return parts


def _bank_for_slot(
    industry: IndustryBank,
    sub_domain: str,
    slot: str,
) -> tuple[str, ...]:
    """Return the bank for an entity slot in the given sub-domain.

    Falls back to an empty tuple if the slot is not defined for this
    sub-domain - the caller will then skip the swap.
    """
    ents = industry.entities.get(sub_domain, {}) or {}
    bank = ents.get(slot)
    if bank is None:
        return ()
    return tuple(bank)


def _numeric_flip_value(
    original: str,
    rng: random.Random,
    *,
    slot_hint: str,
) -> Optional[str]:
    """Return a perturbed numeric string or ``None`` if unflippable.

    The perturbation is calibrated per slot family so hallucinations
    stay plausible (a 24% revenue miss rather than a 0.1% one).
    """
    original = original.strip()
    # Parse as float if possible.
    try:
        n = float(original)
    except ValueError:
        return None
    if n == 0:
        # Flip to a small non-zero value with the same sign.
        return "1"
    abs_n = abs(n)
    is_pct = slot_hint.endswith("_pct") or slot_hint in {
        "margin_pct", "ctr_pct", "capital_ratio_pct",
        "error_budget_pct", "completion_pct", "adherence_pct",
        "market_share_pct", "gdp_impact_pct", "indirect_rate_pct",
        "growth_pct",
    }
    is_year = slot_hint == "citation_year"
    if is_year:
        # Shift by 2-5 years, don't drift past the present/future.
        delta = rng.choice((-5, -3, -2, 2, 3))
        return str(int(n) + delta)
    # Percents: flip by ±(1.5 .. 4.5), keep sign of original.
    if is_pct:
        delta = round(rng.uniform(1.5, 4.5), 1) * rng.choice((-1, 1))
        new_val = n + delta
        if abs(new_val - n) < 0.5:
            new_val = n + rng.choice((2.1, -2.1))
        if "." in original:
            return f"{new_val:.{max(1, len(original.split('.')[-1]))}f}"
        return str(int(round(new_val)))
    # Money/count: flip by ±(12-32%) of the original magnitude, with a
    # minimum absolute delta to avoid rounding-only changes.
    pct = rng.uniform(0.12, 0.32) * rng.choice((-1, 1))
    min_abs = max(1.0, abs_n * 0.10)
    delta = pct * abs_n
    if abs(delta) < min_abs:
        delta = min_abs * (1 if pct >= 0 else -1)
    new_val = n + delta
    # Preserve formatting (int vs float with N dp).
    if "." in original:
        dp = len(original.split(".")[-1])
        return f"{new_val:.{dp}f}"
    return str(int(round(new_val)))


def _choose_slot(
    slot_values: Mapping[str, str],
    *,
    predicate,
    rng: random.Random,
) -> Optional[str]:
    eligible = [s for s in slot_values if predicate(s)]
    if not eligible:
        return None
    return rng.choice(sorted(eligible))  # sorted for determinism given seed


def _replace_numeric(text: str, source: str, replacement: str) -> str:
    """Replace ``source`` with ``replacement`` in ``text`` with digit-safe boundaries.

    A naive ``str.replace`` corrupts embedded digit runs (e.g. flipping
    ``growth_pct="2"`` would turn ``"FY2026"`` into ``"FY4046"``). We
    use negative look-arounds that reject any match where the
    immediately adjacent char is a digit or a decimal point, so the
    flip only fires on intact numeric tokens.
    """
    pattern = r"(?<![\d.])" + re.escape(source) + r"(?![\d.])"
    return re.sub(pattern, replacement, text, count=0)


def _replace_entity(text: str, source: str, replacement: str) -> str:
    """Replace a named-entity token with word-boundary safety.

    Entity bank values are multi-word phrases; standard ``\\b``
    boundaries around the escaped source prevent accidental matches
    on substrings of adjacent tokens.
    """
    pattern = r"\b" + re.escape(source) + r"\b"
    return re.sub(pattern, replacement, text, count=0)


# ---------------------------------------------------------------------------
# Entity swap
# ---------------------------------------------------------------------------


def entity_swap(
    *,
    industry_name: str,
    sub_domain: str,
    passage: str,
    response: str,
    slot_values: Mapping[str, str],
    seed: int,
) -> Optional[AdversarialOutput]:
    """Replace one entity slot in the response with a bank distractor."""
    rng = random.Random(seed ^ 0xE57A)
    industry = get_industry(industry_name)

    slot = _choose_slot(
        slot_values,
        predicate=lambda s: (
            s in ENTITY_SWAPPABLE_SLOTS
            and _bank_for_slot(industry, sub_domain, s)
        ),
        rng=rng,
    )
    if slot is None:
        return None
    bank = _bank_for_slot(industry, sub_domain, slot)
    source = slot_values[slot]
    distractors = [d for d in bank if d != source]
    if not distractors:
        return None
    distractor = rng.choice(distractors)

    # Replace only in the response, leaving the passage intact so the
    # evidence "correctly" identifies a different entity than the
    # claim.
    new_response = _replace_entity(response, source, distractor)
    if new_response == response:
        return None
    return AdversarialOutput(
        kind="entity_swap",
        passage=passage,
        response=new_response,
        slot_values={**slot_values, slot: distractor},
        modified_slots=(slot,),
        notes=f"entity_swap[{slot}]: {source!r} -> {distractor!r}",
    )


# ---------------------------------------------------------------------------
# Numeric flip
# ---------------------------------------------------------------------------


def numeric_flip(
    *,
    industry_name: str,
    sub_domain: str,
    passage: str,
    response: str,
    slot_values: Mapping[str, str],
    seed: int,
) -> Optional[AdversarialOutput]:
    """Perturb one numeric slot in the response by a plausible amount."""
    rng = random.Random(seed ^ 0x17EA)
    # Any slot whose value looks numeric or whose name matches a
    # numeric-slot hint.
    slot = _choose_slot(
        slot_values,
        predicate=lambda s: (
            s in NUMERIC_SLOT_HINTS
            or (
                slot_values[s].replace(".", "").replace("-", "").isdigit()
            )
        ),
        rng=rng,
    )
    if slot is None:
        return None
    source = slot_values[slot]
    flipped = _numeric_flip_value(source, rng, slot_hint=slot)
    if flipped is None or flipped == source:
        return None
    # Replace only in the response, not in the passage. Digit-safe
    # boundaries avoid corrupting years / amounts where the slot's
    # value appears as a substring.
    new_response = _replace_numeric(response, source, flipped)
    if new_response == response:
        return None
    return AdversarialOutput(
        kind="numeric_flip",
        passage=passage,
        response=new_response,
        slot_values={**slot_values, slot: flipped},
        modified_slots=(slot,),
        notes=f"numeric_flip[{slot}]: {source!r} -> {flipped!r}",
    )


# ---------------------------------------------------------------------------
# Support drop
# ---------------------------------------------------------------------------


def support_drop(
    *,
    industry_name: str,
    sub_domain: str,
    passage: str,
    response: str,
    slot_values: Mapping[str, str],
    seed: int,
) -> Optional[AdversarialOutput]:
    """Remove one sentence of the passage that supports the response.

    The response is unchanged; after the drop it contains a claim that
    is no longer supported by any evidence sentence.
    """
    rng = random.Random(seed ^ 0xD40B)
    sentences = _split_sentences(passage)
    if len(sentences) < 2:
        # Need at least two sentences so that after dropping one
        # there's still evidence left.
        return None
    # Pick the sentence that best "covers" the response so that
    # removing it produces the most informative unsupported claim.
    # Heuristic: the sentence with the highest token overlap with the
    # response, deterministically broken by the seed for variety.
    def overlap(s: str) -> int:
        ws_s = set(re.findall(r"[A-Za-z0-9]+", s.lower()))
        ws_r = set(re.findall(r"[A-Za-z0-9]+", response.lower()))
        return len(ws_s & ws_r)

    scored = sorted(
        range(len(sentences)),
        key=lambda i: (-overlap(sentences[i]), rng.random()),
    )
    drop_idx = scored[0]
    dropped = sentences.pop(drop_idx)
    new_passage = " ".join(sentences)
    return AdversarialOutput(
        kind="support_drop",
        passage=new_passage,
        response=response,
        slot_values=dict(slot_values),
        modified_slots=(),
        dropped_sentences=(dropped,),
        notes=f"support_drop: dropped sentence idx={drop_idx}",
    )


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


ADVERSARIAL_KINDS: tuple[str, ...] = ("entity_swap", "numeric_flip", "support_drop")


_DISPATCH = {
    "entity_swap": entity_swap,
    "numeric_flip": numeric_flip,
    "support_drop": support_drop,
}


def apply_adversarial(
    kind: str,
    *,
    industry_name: str,
    sub_domain: str,
    passage: str,
    response: str,
    slot_values: Mapping[str, str],
    seed: int,
) -> Optional[AdversarialOutput]:
    fn = _DISPATCH.get(kind)
    if fn is None:
        raise KeyError(f"unknown adversarial kind: {kind!r}")
    return fn(
        industry_name=industry_name,
        sub_domain=sub_domain,
        passage=passage,
        response=response,
        slot_values=slot_values,
        seed=seed,
    )


__all__ = [
    "ADVERSARIAL_KINDS",
    "AdversarialOutput",
    "apply_adversarial",
    "entity_swap",
    "numeric_flip",
    "support_drop",
]
