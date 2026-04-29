"""Deterministic synthetic-enterprise-lane generator.

One function, :func:`generate_enterprise_rows`, produces the full
enterprise synthetic lane described in the plan:

* 6 training industries + 2 OOD held-out industries.
* 2 sub-domains per industry.
* 3 shapes per sub-domain (short / medium / long).
* 3 adversarials per grounded pair (entity_swap / numeric_flip /
  support_drop).
* 3 surface paraphrases per base row (v0 / v1 / v2).

Everything is driven by SHA-256-derived seeds so the same manifest
entry always produces the same rows. No network, no LLM, no torch.
Rows are emitted as plain dicts matching the :func:`row_schema`
contract (which ``distill_dataset.py`` then augments with teacher-
derived three-axis labels).

The generator's output volume scales with ``passages_per_bucket``:
default is 50, which yields ``50 * 2 * 3 = 300`` grounded rows per
industry (i.e. 1800 training-grounded rows at the default). Each
grounded row fans out into 3 adversarials and 3 paraphrases, so the
default totals ``1800 * 4 * 3 = 21,600`` rows for training + another
``600 * 4 * 3 = 7200`` OOD rows = ~29k, near the 30k plan target. Set
``passages_per_bucket=60`` to hit the 30k floor exactly.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Iterator, Mapping, Sequence

from research.triangular_maxsim.student_v2.synthetic.adversarials import (
    ADVERSARIAL_KINDS,
    apply_adversarial,
)
from research.triangular_maxsim.student_v2.synthetic.industry_banks import (
    IndustryBank,
    get_industry,
    ood_industries,
    training_industries,
)
from research.triangular_maxsim.student_v2.synthetic.paraphrase import (
    PARAPHRASE_VARIANTS,
    paraphrase,
)
from research.triangular_maxsim.student_v2.synthetic.templates import (
    SHAPES,
    Template,
    templates_for,
)


CLASS_KEY = "rag.prose.enterprise"


# ---------------------------------------------------------------------------
# Seed derivation
# ---------------------------------------------------------------------------


def _seed_for(*parts: str | int) -> int:
    payload = "||".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    # Use the first 8 bytes as a 64-bit signed-compatible int.
    return int.from_bytes(digest[:8], "big", signed=False)


# ---------------------------------------------------------------------------
# Slot filling
# ---------------------------------------------------------------------------


@dataclass
class _SlotResolver:
    industry: IndustryBank
    sub_domain: str
    rng: random.Random

    def _entity_bank(self, slot: str) -> Sequence[str] | None:
        return self.industry.entities.get(self.sub_domain, {}).get(slot)

    def _numeric_bank(self, slot: str) -> Sequence[str] | None:
        return self.industry.numerics.get(self.sub_domain, {}).get(slot)

    def _default_bank(self, slot: str) -> Sequence[str] | None:
        return self.industry.slot_defaults.get(slot)

    def resolve(self, slot: str) -> str:
        # Priority: entity -> numeric -> industry default -> unknown.
        for source in (
            self._entity_bank,
            self._numeric_bank,
            self._default_bank,
        ):
            bank = source(slot)
            if bank:
                return self.rng.choice(tuple(bank))
        # Shared default banks (quarters / fiscal years) for convenience.
        from research.triangular_maxsim.student_v2.synthetic.industry_banks import (
            FISCAL_YEARS, QUARTERS, SEVERITY, STATUSES,
        )
        if slot == "quarter":
            return self.rng.choice(QUARTERS)
        if slot == "fy":
            return self.rng.choice(FISCAL_YEARS)
        if slot == "severity":
            return self.rng.choice(SEVERITY)
        if slot == "status":
            return self.rng.choice(STATUSES)
        raise KeyError(
            f"slot {slot!r} not found in industry={self.industry.name} "
            f"sub_domain={self.sub_domain}"
        )


def _fill_template(
    template: Template,
    resolver: _SlotResolver,
) -> tuple[str, str, dict[str, str]]:
    slot_values: dict[str, str] = {}
    for slot in template.required_slots:
        slot_values[slot] = resolver.resolve(slot)
    passage = template.passage.format(**slot_values)
    response = template.response.format(**slot_values)
    return passage, response, slot_values


# ---------------------------------------------------------------------------
# Row schema
# ---------------------------------------------------------------------------


@dataclass
class SyntheticRow:
    pair_id: str
    class_key: str
    split_hint: str  # "train" (training industries) or "ood_eval"
    industry: str
    sub_domain: str
    shape: str
    passage_index: int
    role: str  # "grounded" | "entity_swap" | "numeric_flip" | "support_drop"
    paraphrase_variant: str
    response_text: str
    evidence_text: str
    intended_label: str  # "grounded" | "ungrounded"
    modified_slots: tuple[str, ...] = ()
    dropped_sentences: tuple[str, ...] = ()
    template_id: str = ""
    notes: str = ""
    slot_values: Mapping[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "pair_id": self.pair_id,
            "class_key": self.class_key,
            "split_hint": self.split_hint,
            "industry": self.industry,
            "sub_domain": self.sub_domain,
            "shape": self.shape,
            "passage_index": self.passage_index,
            "role": self.role,
            "paraphrase_variant": self.paraphrase_variant,
            "response_text": self.response_text,
            "evidence_text": self.evidence_text,
            "intended_label": self.intended_label,
            "modified_slots": list(self.modified_slots),
            "dropped_sentences": list(self.dropped_sentences),
            "template_id": self.template_id,
            "notes": self.notes,
            "slot_values": dict(self.slot_values),
            "source": "synthetic_enterprise",
        }


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def _industry_buckets(
    industry: IndustryBank,
) -> Iterator[tuple[str, str, Template, int]]:
    """Yield (sub_domain, shape, template, template_idx) across the industry."""
    for sub_domain in industry.sub_domains:
        for shape in SHAPES:
            tpls = templates_for(industry.name, sub_domain, shape)
            for idx, tpl in enumerate(tpls):
                yield sub_domain, shape, tpl, idx


def generate_enterprise_rows(
    *,
    industries: Sequence[IndustryBank] | None = None,
    passages_per_bucket: int = 50,
    master_seed: int = 42,
    emit_ood: bool = True,
) -> Iterator[SyntheticRow]:
    """Yield all synthetic enterprise rows in deterministic order.

    Args:
        industries: if provided, restrict generation to this list.
            Otherwise emit the full 6-training + 2-OOD set (unless
            ``emit_ood`` is False, in which case only training
            industries are emitted).
        passages_per_bucket: how many passages to generate per
            (industry, sub_domain, shape, template) bucket. Default
            50 -> ~21k training rows + ~7k OOD rows.
        master_seed: top-level seed mixed into every row seed.
        emit_ood: if False, skip held-out industries.
    """
    if industries is None:
        industries = list(training_industries())
        if emit_ood:
            industries = industries + list(ood_industries())

    for industry in industries:
        split_hint = "train" if industry.training else "ood_eval"
        for sub_domain, shape, template, tpl_idx in _industry_buckets(industry):
            for pidx in range(passages_per_bucket):
                yield from _emit_passage_fanout(
                    industry=industry,
                    sub_domain=sub_domain,
                    shape=shape,
                    template=template,
                    template_idx=tpl_idx,
                    passage_index=pidx,
                    split_hint=split_hint,
                    master_seed=master_seed,
                )


def _emit_passage_fanout(
    *,
    industry: IndustryBank,
    sub_domain: str,
    shape: str,
    template: Template,
    template_idx: int,
    passage_index: int,
    split_hint: str,
    master_seed: int,
) -> Iterator[SyntheticRow]:
    """Emit the full fan-out for a single passage: grounded + 3 adv, each x 3 paraphrases."""

    # Base seed ties all variants of one passage together.
    base_seed = _seed_for(
        master_seed, industry.name, sub_domain, shape,
        template_idx, passage_index,
    ) & 0xFFFFFFFF
    resolver = _SlotResolver(
        industry=industry,
        sub_domain=sub_domain,
        rng=random.Random(base_seed),
    )
    passage, grounded_response, slot_values = _fill_template(template, resolver)
    template_id = f"{industry.name}/{sub_domain}/{shape}/{template_idx}"
    pair_id = (
        f"syn_enterprise::{industry.name}::{sub_domain}::{shape}"
        f"::t{template_idx}::p{passage_index}"
    )

    # Grounded variant: 3 paraphrase flavours of the same (passage, response).
    for variant in PARAPHRASE_VARIANTS:
        variant_seed = _seed_for(base_seed, "grounded", variant) & 0xFFFFFFFF
        yield SyntheticRow(
            pair_id=pair_id,
            class_key=CLASS_KEY,
            split_hint=split_hint,
            industry=industry.name,
            sub_domain=sub_domain,
            shape=shape,
            passage_index=passage_index,
            role="grounded",
            paraphrase_variant=variant,
            response_text=paraphrase(variant, grounded_response, seed=variant_seed),
            evidence_text=passage,
            intended_label="grounded",
            template_id=template_id,
            notes="",
            slot_values=slot_values,
        )

    # 3 adversarial variants of the grounded pair, each x 3 paraphrases.
    for kind in ADVERSARIAL_KINDS:
        adv_seed = _seed_for(base_seed, kind) & 0xFFFFFFFF
        adv = apply_adversarial(
            kind,
            industry_name=industry.name,
            sub_domain=sub_domain,
            passage=passage,
            response=grounded_response,
            slot_values=slot_values,
            seed=adv_seed,
        )
        if adv is None:
            # Skip when no eligible slot / sentence available. We
            # deliberately do not fabricate a fallback; the generator
            # logs the drop by not emitting a row.
            continue
        for variant in PARAPHRASE_VARIANTS:
            variant_seed = _seed_for(adv_seed, variant) & 0xFFFFFFFF
            # Only paraphrase the response; the passage stays exactly
            # as produced by the adversarial (e.g. with a dropped
            # sentence).
            final_response = paraphrase(variant, adv.response, seed=variant_seed)
            yield SyntheticRow(
                pair_id=pair_id,
                class_key=CLASS_KEY,
                split_hint=split_hint,
                industry=industry.name,
                sub_domain=sub_domain,
                shape=shape,
                passage_index=passage_index,
                role=kind,
                paraphrase_variant=variant,
                response_text=final_response,
                evidence_text=adv.passage,
                intended_label="ungrounded",
                modified_slots=adv.modified_slots,
                dropped_sentences=adv.dropped_sentences,
                template_id=template_id,
                notes=adv.notes,
                slot_values=adv.slot_values,
            )


__all__ = [
    "CLASS_KEY",
    "SyntheticRow",
    "generate_enterprise_rows",
]
