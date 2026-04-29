"""Unit tests for the deterministic synthetic labeler.

Covers:
- chunking correctness (sentence splitting, offsets, empty text)
- role -> turn_score mapping (grounded high, adversarials low)
- token_support: supported tokens reflect evidence overlap; swapped
  slot values are explicitly unsupported
- dead_weight: distractor / unrelated units get flagged
- coverage: monotonic with response-overlap
- determinism: same input row -> same labels across calls
- integration with generator: labelling a real generator row produces
  a non-empty, well-typed LabelledRow
"""

from __future__ import annotations

from research.triangular_maxsim.student_v2.synthetic import (
    LabelledRow,
    chunk_evidence_into_units,
    generate_enterprise_rows,
    get_industry,
    label_synthetic_row,
)


class TestChunkEvidence:
    def test_empty_text_returns_empty(self) -> None:
        assert chunk_evidence_into_units("") == []
        assert chunk_evidence_into_units("   ") == []

    def test_single_sentence_becomes_one_unit(self) -> None:
        units = chunk_evidence_into_units("Revenue grew 12% in Q3 2026.")
        assert len(units) == 1
        assert "revenue" in units[0].content_tokens

    def test_multiple_sentences_split_by_period(self) -> None:
        txt = "Revenue grew 12% in Q3. Operating margin fell. New product launched."
        units = chunk_evidence_into_units(txt)
        assert len(units) == 3
        assert "revenue" in units[0].content_tokens
        assert "margin" in units[1].content_tokens
        assert "product" in units[2].content_tokens

    def test_char_offsets_monotonically_increase(self) -> None:
        txt = "First sentence here. Second sentence here. Third sentence here."
        units = chunk_evidence_into_units(txt)
        starts = [u.char_start for u in units]
        assert starts == sorted(starts)
        for u in units:
            assert u.char_end > u.char_start

    def test_content_tokens_exclude_stopwords(self) -> None:
        units = chunk_evidence_into_units("The quick brown fox is a fox.")
        assert len(units) == 1
        assert "quick" in units[0].content_tokens
        assert "brown" in units[0].content_tokens
        assert "fox" in units[0].content_tokens
        assert "the" not in units[0].content_tokens
        assert "is" not in units[0].content_tokens


class TestLabelGrounded:
    def _row(self) -> dict:
        return {
            "pair_id": "test-grounded",
            "class_key": "rag.prose.enterprise",
            "split_hint": "train",
            "role": "grounded",
            "paraphrase_variant": "v0",
            "response_text": "Acme revenue grew 12% in Q3 2026.",
            "evidence_text": (
                "Acme Corp reported revenue growth of 12% in Q3 2026. "
                "Operating margin expanded by 180 basis points."
            ),
            "modified_slots": [],
            "dropped_sentences": [],
            "slot_values": {"company": "Acme", "growth_pct": "12", "quarter": "Q3"},
        }

    def test_turn_score_high_for_grounded(self) -> None:
        labelled = label_synthetic_row(self._row())
        assert labelled.turn_score >= 0.8
        assert labelled.turn_band == "green"
        assert labelled.gold_band == "green"

    def test_token_support_mostly_supported_for_grounded(self) -> None:
        labelled = label_synthetic_row(self._row())
        # At least one content token should be flagged supported.
        assert any(labelled.token_support_labels)
        # Of the supported tokens, "acme", "revenue", "q3", "2026", "12"
        # should all be in; stopwords should be 0.
        text_tokens = [t.lower() for t in labelled.response_tokens]
        for keyword in ("acme", "revenue", "12", "q3", "2026"):
            if keyword in text_tokens:
                idx = text_tokens.index(keyword)
                assert labelled.token_support_labels[idx] == 1

    def test_dead_weight_tracks_response_overlap_for_grounded(self) -> None:
        labelled = label_synthetic_row(self._row())
        # The response mirrors the first evidence sentence but does
        # not mention operating margin. That makes the second unit
        # genuinely dead-weight on the coverage side even though
        # the turn is still grounded. This is the *correct* signal
        # for the dead-weight head: "which retrieved unit went
        # unused" is orthogonal to "did the model hallucinate".
        assert labelled.dead_weight_unit_labels[0] == 0
        assert labelled.dead_weight_unit_labels[1] == 1

    def test_coverage_nontrivial_for_grounded(self) -> None:
        labelled = label_synthetic_row(self._row())
        assert all(c >= 0.0 for c in labelled.coverage_unit_labels)
        # The first unit (which the response mirrors) should have
        # higher coverage than the second (margin-only).
        assert labelled.coverage_unit_labels[0] > labelled.coverage_unit_labels[1]


class TestLabelAdversarial:
    def test_entity_swap_is_red_band(self) -> None:
        row = {
            "pair_id": "test-entity-swap",
            "class_key": "rag.prose.enterprise",
            "role": "entity_swap",
            "paraphrase_variant": "v0",
            "response_text": "Globex revenue grew 12% in Q3 2026.",
            "evidence_text": "Acme Corp reported revenue growth of 12% in Q3 2026.",
            "modified_slots": ["company"],
            "slot_values": {"company": "Globex"},
        }
        labelled = label_synthetic_row(row)
        assert labelled.gold_band == "red"
        assert labelled.turn_band == "red"
        assert labelled.turn_score < 0.4

    def test_entity_swap_marks_swapped_token_unsupported(self) -> None:
        row = {
            "pair_id": "test-entity-swap-2",
            "role": "entity_swap",
            "response_text": "Globex revenue grew 12% in Q3.",
            "evidence_text": "Acme Corp reported revenue growth of 12% in Q3.",
            "modified_slots": ["company"],
            "slot_values": {"company": "Globex"},
        }
        labelled = label_synthetic_row(row)
        tokens = [t.lower() for t in labelled.response_tokens]
        # "globex" is the swapped entity -> unsupported.
        idx = tokens.index("globex")
        assert labelled.token_support_labels[idx] == 0

    def test_numeric_flip_marks_flipped_number_unsupported(self) -> None:
        row = {
            "pair_id": "test-numeric-flip",
            "role": "numeric_flip",
            "response_text": "Acme revenue grew 40% in Q3.",
            "evidence_text": "Acme Corp reported revenue growth of 12% in Q3.",
            "modified_slots": ["growth_pct"],
            "slot_values": {"growth_pct": "40"},
        }
        labelled = label_synthetic_row(row)
        tokens = [t.lower() for t in labelled.response_tokens]
        # "40" is the poisoned numeric -> unsupported.
        idx = tokens.index("40")
        assert labelled.token_support_labels[idx] == 0

    def test_support_drop_role_is_red(self) -> None:
        row = {
            "pair_id": "test-support-drop",
            "role": "support_drop",
            "response_text": "Revenue grew 12% and margin expanded 180bps.",
            "evidence_text": "Revenue grew 12% in Q3.",  # margin sentence dropped
            "dropped_sentences": ["Operating margin expanded by 180 basis points."],
            "slot_values": {},
        }
        labelled = label_synthetic_row(row)
        assert labelled.gold_band == "red"
        assert labelled.turn_band == "red"


class TestDeadWeightOnDistractor:
    def test_unit_with_no_response_overlap_flagged_dead(self) -> None:
        row = {
            "pair_id": "dw-1",
            "role": "grounded",
            "response_text": "Revenue grew 12% in Q3 2026.",
            "evidence_text": (
                "Revenue grew 12% in Q3 2026. "  # used
                "The office cafeteria menu changed on Tuesday."  # distractor (zero overlap)
            ),
            "modified_slots": [],
            "slot_values": {},
        }
        labelled = label_synthetic_row(row)
        # Two units expected; the distractor (index 1) is dead-weight.
        assert len(labelled.evidence_units) == 2
        assert labelled.dead_weight_unit_labels[0] == 0
        assert labelled.dead_weight_unit_labels[1] == 1

    def test_coverage_monotonic_with_overlap(self) -> None:
        row = {
            "pair_id": "dw-2",
            "role": "grounded",
            "response_text": "Revenue growth hit 12% in Q3 2026.",
            "evidence_text": (
                "Revenue growth hit 12% in Q3 2026. "          # high overlap
                "Revenue margin was stable year over year. "   # partial overlap
                "Cafeteria menu changed on Tuesday."            # zero overlap
            ),
            "modified_slots": [],
            "slot_values": {},
        }
        labelled = label_synthetic_row(row)
        cov = labelled.coverage_unit_labels
        assert len(cov) == 3
        assert cov[0] > cov[1] >= cov[2]


class TestDeterminism:
    def test_same_input_same_output(self) -> None:
        row = {
            "role": "entity_swap",
            "response_text": "Globex revenue grew 12% in Q3.",
            "evidence_text": "Acme Corp reported revenue growth of 12% in Q3.",
            "modified_slots": ["company"],
            "slot_values": {"company": "Globex"},
        }
        a = label_synthetic_row(row).to_dict()
        b = label_synthetic_row(row).to_dict()
        assert a == b


class TestGeneratorIntegration:
    def test_labeller_handles_generator_row(self) -> None:
        bank = get_industry("finance")
        it = generate_enterprise_rows(
            industries=[bank], passages_per_bucket=1, master_seed=7, emit_ood=False,
        )
        sample = next(it).to_dict()
        labelled = label_synthetic_row(sample)
        assert isinstance(labelled, LabelledRow)
        assert labelled.class_key == sample["class_key"]
        assert labelled.pair_id == sample["pair_id"]
        assert len(labelled.response_tokens) > 0
        assert len(labelled.evidence_units) > 0
        assert len(labelled.token_support_labels) == len(labelled.response_tokens)
        assert len(labelled.dead_weight_unit_labels) == len(labelled.evidence_units)
        assert len(labelled.coverage_unit_labels) == len(labelled.evidence_units)
        if labelled.role == "grounded":
            assert labelled.gold_band == "green"
        else:
            assert labelled.gold_band == "red"

    def test_labeller_over_many_generator_rows_produces_balanced_labels(self) -> None:
        """Across a larger sample, verify that roles produce the
        expected hallmark label distributions.
        """
        bank = get_industry("finance")
        rows = [
            label_synthetic_row(r.to_dict())
            for r in generate_enterprise_rows(
                industries=[bank], passages_per_bucket=3, master_seed=11, emit_ood=False,
            )
        ]
        assert len(rows) > 10
        grounded = [r for r in rows if r.role == "grounded"]
        adversarial = [r for r in rows if r.role != "grounded"]
        assert grounded and adversarial

        # Grounded rows should average a higher turn_score than adversarial.
        g_mean = sum(r.turn_score for r in grounded) / len(grounded)
        a_mean = sum(r.turn_score for r in adversarial) / len(adversarial)
        assert g_mean > a_mean + 0.4  # grounded much higher

        # Grounded rows should have mostly-green bands, adversarial mostly-red.
        assert all(r.gold_band == "green" for r in grounded)
        assert all(r.gold_band == "red" for r in adversarial)
