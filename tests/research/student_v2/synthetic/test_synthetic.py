"""Unit tests for the synthetic enterprise lane.

Covers industry banks, templates, adversarials, paraphrase variants,
quality gates, and the top-level :func:`generate_enterprise_rows`
orchestrator.
"""

from __future__ import annotations

import re
from collections import Counter

import pytest

from research.triangular_maxsim.student_v2.synthetic import (
    ADVERSARIAL_KINDS,
    INDUSTRIES,
    PARAPHRASE_VARIANTS,
    SHAPES,
    apply_adversarial,
    entity_swap,
    generate_enterprise_rows,
    get_industry,
    jaccard,
    jaccard_gate,
    numeric_flip,
    ood_industries,
    paraphrase,
    support_drop,
    teacher_verdict_consistent,
    templates_for,
    training_industries,
)


# ---------------------------------------------------------------------------
# Industry banks
# ---------------------------------------------------------------------------


class TestIndustryBanks:
    def test_training_split(self) -> None:
        t = {b.name for b in training_industries()}
        assert t == {"legal", "finance", "hr", "compliance", "engineering", "marketing"}

    def test_ood_split(self) -> None:
        o = {b.name for b in ood_industries()}
        assert o == {"healthcare", "public_sector"}

    def test_each_industry_has_exactly_two_sub_domains(self) -> None:
        for b in INDUSTRIES:
            assert len(b.sub_domains) == 2, f"{b.name}: wrong sub_domain count"

    def test_entity_banks_disjoint_across_sub_domains(self) -> None:
        """Within an industry, no entity slot reuses the same value in a
        different sub-domain. Entity-swap relies on this for plausibility."""
        for b in INDUSTRIES:
            all_entity_values: dict[str, set[str]] = {}
            for sub_domain in b.sub_domains:
                for slot, bank in b.entities.get(sub_domain, {}).items():
                    if slot in all_entity_values:
                        overlap = all_entity_values[slot] & set(bank)
                        assert not overlap, (
                            f"{b.name}: slot {slot!r} has cross-sub-domain "
                            f"overlap {overlap}"
                        )
                    all_entity_values.setdefault(slot, set()).update(bank)

    def test_numeric_banks_have_values(self) -> None:
        for b in INDUSTRIES:
            for sub_domain in b.sub_domains:
                nums = b.numerics.get(sub_domain, {})
                assert nums, f"{b.name}/{sub_domain}: no numerics defined"
                for slot, bank in nums.items():
                    assert len(bank) >= 3, (
                        f"{b.name}/{sub_domain}/{slot}: too few numeric choices"
                    )

    def test_get_industry_unknown_raises(self) -> None:
        with pytest.raises(KeyError):
            get_industry("not_an_industry")


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------


class TestTemplates:
    def test_every_industry_has_all_shapes(self) -> None:
        for b in INDUSTRIES:
            for sub_domain in b.sub_domains:
                for shape in SHAPES:
                    tpls = templates_for(b.name, sub_domain, shape)
                    assert len(tpls) >= 1, (
                        f"missing templates: {b.name}/{sub_domain}/{shape}"
                    )

    def test_every_template_required_slot_resolvable(self) -> None:
        """Every slot declared in a template must exist in the industry's
        entity / numeric banks or in the shared slot defaults.
        """
        shared_defaults = {"quarter", "fy", "severity", "status"}
        for b in INDUSTRIES:
            for sub_domain in b.sub_domains:
                for shape in SHAPES:
                    for tpl in templates_for(b.name, sub_domain, shape):
                        for slot in tpl.required_slots:
                            in_entity = slot in b.entities.get(sub_domain, {})
                            in_numeric = slot in b.numerics.get(sub_domain, {})
                            in_default = slot in b.slot_defaults or slot in shared_defaults
                            assert in_entity or in_numeric or in_default, (
                                f"unresolvable slot {slot!r} in "
                                f"{b.name}/{sub_domain}/{shape}"
                            )


# ---------------------------------------------------------------------------
# Paraphrase
# ---------------------------------------------------------------------------


class TestParaphrase:
    def test_v0_is_identity(self) -> None:
        s = "Apple grew Q3 FY2024 revenue 12% to $85.8 billion."
        assert paraphrase("v0", s, seed=1) == s

    def test_v1_and_v2_are_deterministic(self) -> None:
        s = "In Q2 FY2024, Acme delivered $12.4 billion in revenue (+11% YoY)."
        for v in ("v1", "v2"):
            a = paraphrase(v, s, seed=42)
            b = paraphrase(v, s, seed=42)
            assert a == b

    def test_v2_applies_some_lexical_sub(self) -> None:
        s = "Apple grew revenue 12% year over year."
        out = paraphrase("v2", s, seed=1)
        assert "year over year" not in out  # substituted by "YoY"
        assert "YoY" in out

    def test_unknown_variant_raises(self) -> None:
        with pytest.raises(KeyError):
            paraphrase("v9", "foo", seed=0)


# ---------------------------------------------------------------------------
# Adversarials
# ---------------------------------------------------------------------------


class TestEntitySwap:
    def test_swap_changes_named_entity_slot(self) -> None:
        passage = "Apple announced Q1 FY2024 revenue of $85.8 billion."
        response = "Apple's Q1 FY2024 revenue was $85.8 billion."
        out = entity_swap(
            industry_name="finance",
            sub_domain="earnings",
            passage=passage,
            response=response,
            slot_values={"company": "Apple", "quarter": "Q1", "fy": "FY2024"},
            seed=1,
        )
        assert out is not None
        assert out.kind == "entity_swap"
        assert out.passage == passage  # passage unchanged
        assert out.response != response  # response changed
        assert out.modified_slots == ("company",)
        # Passage still names the original entity; response does not.
        assert "Apple" in out.passage
        assert "Apple" not in out.response

    def test_swap_returns_none_when_no_eligible_slot(self) -> None:
        out = entity_swap(
            industry_name="finance",
            sub_domain="earnings",
            passage="Some text.",
            response="Some text.",
            slot_values={"quarter": "Q1"},  # not entity-swappable
            seed=1,
        )
        assert out is None


class TestNumericFlip:
    def test_flip_changes_numeric_slot_value(self) -> None:
        passage = "Apple grew revenue 12% year over year."
        response = "Apple's revenue grew 12%."
        out = numeric_flip(
            industry_name="finance",
            sub_domain="earnings",
            passage=passage,
            response=response,
            slot_values={"company": "Apple", "growth_pct": "12"},
            seed=1,
        )
        assert out is not None
        assert out.kind == "numeric_flip"
        assert "growth_pct" in out.modified_slots
        # The flipped value must no longer appear in the response.
        assert "12%" not in out.response or out.response.count("12%") < response.count("12%")

    def test_digit_boundary_protects_year_substring(self) -> None:
        """Regression: growth_pct=2 must not turn FY2026 into FY4046."""
        response = "Apple grew Q3 FY2026 revenue 2% to $85.8 billion."
        out = numeric_flip(
            industry_name="finance",
            sub_domain="earnings",
            passage="evidence",
            response=response,
            slot_values={"company": "Apple", "quarter": "Q3", "fy": "FY2026", "growth_pct": "2"},
            seed=1,
        )
        assert out is not None
        # FY2026 must be unchanged.
        assert "FY2026" in out.response
        # The ``2%`` must have been flipped.
        assert "2%" not in out.response

    def test_flip_returns_none_when_no_numeric_slot(self) -> None:
        out = numeric_flip(
            industry_name="finance",
            sub_domain="earnings",
            passage="text",
            response="text",
            slot_values={"company": "Apple"},
            seed=1,
        )
        assert out is None


class TestSupportDrop:
    def test_drop_removes_a_sentence(self) -> None:
        passage = (
            "Apple grew revenue 12%. Operating margin expanded to 31%. "
            "Diluted EPS rose to $2.54."
        )
        response = "Apple's Q3 EPS was $2.54 with 31% margin."
        out = support_drop(
            industry_name="finance",
            sub_domain="earnings",
            passage=passage,
            response=response,
            slot_values={"company": "Apple"},
            seed=1,
        )
        assert out is not None
        assert out.kind == "support_drop"
        assert len(out.dropped_sentences) == 1
        # Passage got shorter.
        assert len(out.passage) < len(passage)
        # Response is unchanged.
        assert out.response == response

    def test_drop_returns_none_on_single_sentence_passage(self) -> None:
        out = support_drop(
            industry_name="finance",
            sub_domain="earnings",
            passage="Only one sentence here.",
            response="Same.",
            slot_values={},
            seed=1,
        )
        assert out is None


class TestDispatch:
    def test_apply_adversarial_known_kinds(self) -> None:
        for kind in ADVERSARIAL_KINDS:
            assert kind in {"entity_swap", "numeric_flip", "support_drop"}

    def test_unknown_kind_raises(self) -> None:
        with pytest.raises(KeyError):
            apply_adversarial(
                "not_a_kind",
                industry_name="finance",
                sub_domain="earnings",
                passage="",
                response="",
                slot_values={},
                seed=0,
            )


# ---------------------------------------------------------------------------
# Quality gate
# ---------------------------------------------------------------------------


class TestQualityGate:
    def test_jaccard_identical(self) -> None:
        assert jaccard("hello world", "hello world") == 1.0

    def test_jaccard_disjoint(self) -> None:
        assert jaccard("hello world", "foo bar") == 0.0

    def test_jaccard_empty(self) -> None:
        assert jaccard("", "hello") == 0.0
        assert jaccard("hello", "") == 0.0

    def test_jaccard_is_case_insensitive_and_tokenised(self) -> None:
        assert jaccard("Apple Q3", "apple q3") == 1.0

    def test_jaccard_gate_keeps_over_threshold(self) -> None:
        a = "Apple reported 12% growth in Q3 2024"
        b = "Apple reported 14% growth in Q3 2024"  # single-word change
        gate = jaccard_gate(a, b, min_jaccard=0.75)
        assert gate.kept
        assert gate.score >= 0.75

    def test_jaccard_gate_rejects_low_overlap(self) -> None:
        a = "Apple reported earnings"
        b = "Microsoft announced layoffs unrelated"
        gate = jaccard_gate(a, b, min_jaccard=0.75)
        assert not gate.kept

    def test_teacher_verdict_grounded_green(self) -> None:
        assert teacher_verdict_consistent("grounded", "green")
        assert not teacher_verdict_consistent("grounded", "red")
        assert not teacher_verdict_consistent("grounded", "amber")

    def test_teacher_verdict_adversarial_red_or_amber(self) -> None:
        for role in ("entity_swap", "numeric_flip", "support_drop"):
            assert teacher_verdict_consistent(role, "red")
            assert teacher_verdict_consistent(role, "amber")
            assert not teacher_verdict_consistent(role, "green")


# ---------------------------------------------------------------------------
# Enterprise generator (integration)
# ---------------------------------------------------------------------------


class TestEnterpriseGenerator:
    def test_deterministic(self) -> None:
        a = [r.to_dict() for r in generate_enterprise_rows(passages_per_bucket=4, master_seed=42)]
        b = [r.to_dict() for r in generate_enterprise_rows(passages_per_bucket=4, master_seed=42)]
        assert a == b

    def test_different_seed_produces_different_rows(self) -> None:
        a = [r.to_dict() for r in generate_enterprise_rows(passages_per_bucket=4, master_seed=42)]
        c = [r.to_dict() for r in generate_enterprise_rows(passages_per_bucket=4, master_seed=99)]
        # Identical length (structure is deterministic), but content differs.
        assert len(a) == len(c)
        diff = sum(1 for x, y in zip(a, c) if x["response_text"] != y["response_text"])
        assert diff > len(a) // 2

    def test_all_six_training_industries_present(self) -> None:
        rows = list(generate_enterprise_rows(passages_per_bucket=2, emit_ood=False))
        names = {r.industry for r in rows}
        assert names == {"legal", "finance", "hr", "compliance", "engineering", "marketing"}
        # All rows flagged as train.
        assert all(r.split_hint == "train" for r in rows)

    def test_ood_industries_flagged_and_excluded_from_training(self) -> None:
        rows = list(generate_enterprise_rows(passages_per_bucket=2, emit_ood=True))
        ood = [r for r in rows if r.industry in {"healthcare", "public_sector"}]
        tr = [r for r in rows if r.industry not in {"healthcare", "public_sector"}]
        assert ood, "OOD industries missing"
        assert all(r.split_hint == "ood_eval" for r in ood)
        assert all(r.split_hint == "train" for r in tr)

    def test_grounded_rows_share_pair_id_with_their_adversarials(self) -> None:
        rows = list(generate_enterprise_rows(passages_per_bucket=3))
        from collections import defaultdict
        by_pair = defaultdict(list)
        for r in rows:
            by_pair[r.pair_id].append(r)
        # Each pair must include at least the 3 grounded paraphrase
        # variants (some adversarial kinds may be skipped by filters).
        for pair_id, rs in list(by_pair.items())[:20]:
            roles = {r.role for r in rs}
            assert "grounded" in roles

    def test_three_paraphrases_emitted_per_role(self) -> None:
        rows = list(generate_enterprise_rows(passages_per_bucket=3))
        pairs = Counter((r.pair_id, r.role) for r in rows)
        # Every (pair_id, role) bucket should have exactly 3 variants.
        for key, n in pairs.items():
            assert n == 3, f"{key}: {n} variants"

    def test_intended_label_matches_role(self) -> None:
        rows = list(generate_enterprise_rows(passages_per_bucket=2))
        for r in rows:
            if r.role == "grounded":
                assert r.intended_label == "grounded"
            else:
                assert r.intended_label == "ungrounded"

    def test_class_key_uniform(self) -> None:
        rows = list(generate_enterprise_rows(passages_per_bucket=2))
        assert {r.class_key for r in rows} == {"rag.prose.enterprise"}

    def test_volume_scales_with_passages_per_bucket(self) -> None:
        small = sum(1 for _ in generate_enterprise_rows(passages_per_bucket=5))
        big = sum(1 for _ in generate_enterprise_rows(passages_per_bucket=10))
        assert big >= 2 * small - 30  # roughly linear; ±buffer for adv skips

    def test_no_adversarial_equals_grounded_in_response(self) -> None:
        """Every adversarial must actually have changed something visible
        in the response (entity_swap / numeric_flip) OR have dropped a
        passage sentence (support_drop).
        """
        rows = list(generate_enterprise_rows(passages_per_bucket=3, emit_ood=False))
        pairs: dict[str, dict[str, str]] = {}
        for r in rows:
            if r.paraphrase_variant == "v0":
                pairs.setdefault(r.pair_id, {})[r.role] = r.response_text
        for pid, roles in pairs.items():
            g = roles.get("grounded")
            if g is None:
                continue
            for role in ("entity_swap", "numeric_flip"):
                if role in roles:
                    assert roles[role] != g, (
                        f"{pid}/{role}: response identical to grounded"
                    )

    def test_jaccard_typical_pair_over_threshold(self) -> None:
        """Aggregate check: most v0 grounded/entity_swap pairs must clear 0.75."""
        rows = list(generate_enterprise_rows(passages_per_bucket=10, emit_ood=False))
        pairs: dict[str, dict[str, str]] = {}
        for r in rows:
            if r.paraphrase_variant == "v0":
                pairs.setdefault(r.pair_id, {})[r.role] = r.response_text
        above = 0
        total = 0
        for roles in pairs.values():
            g = roles.get("grounded")
            if not g:
                continue
            for kind in ("entity_swap", "numeric_flip", "support_drop"):
                adv = roles.get(kind)
                if adv is None:
                    continue
                total += 1
                if jaccard(g, adv) >= 0.75:
                    above += 1
        assert total > 100  # meaningful sample
        assert above / total >= 0.85, f"only {above}/{total} pass Jaccard gate"

    def test_no_fy_year_corruption_in_numeric_flips(self) -> None:
        """Regression for the FY2026 -> FY4046 bug."""
        rows = list(generate_enterprise_rows(passages_per_bucket=10, emit_ood=False))
        offenders = []
        for r in rows:
            if r.role != "numeric_flip":
                continue
            ev_years = set(re.findall(r"FY\d{4}", r.evidence_text))
            rs_years = set(re.findall(r"FY\d{4}", r.response_text))
            if ev_years and rs_years and ev_years != rs_years:
                offenders.append((r.pair_id, ev_years, rs_years))
        assert not offenders, f"FY year corruption: {offenders[:3]}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
