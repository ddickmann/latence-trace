"""Unit tests for the training collate + label-alignment helpers."""

from __future__ import annotations

import torch

from research.triangular_maxsim.student_v2.synthetic import (
    label_synthetic_row,
)
from research.triangular_maxsim.student_v2.training.collate import (
    CLASS_TO_IDX,
    _phi_vector_from_row,
    _subtoken_to_unit_idx,
    _subtoken_to_word_idx,
    _word_spans,
    build_batch,
    build_example,
)


class _FakeTokenizer:
    """Minimal whitespace tokenizer with HF-compatible interface.

    Each whitespace-delimited word becomes one subtoken. Offsets are
    exact character spans from the original text. Pads with id=0.
    """

    def __init__(self, vocab_size: int = 512) -> None:
        self.vocab_size = vocab_size

    def __call__(
        self,
        text: str,
        *,
        max_length: int,
        truncation: bool = True,
        padding: str = "max_length",
        return_offsets_mapping: bool = False,
        return_tensors: str = "pt",
    ):
        spans, toks = _word_spans(text or "")
        ids: list[int] = []
        offs: list[list[int]] = []
        mask: list[int] = []
        for t, (s, e) in zip(toks, spans):
            if len(ids) >= max_length:
                break
            ids.append(abs(hash(t)) % self.vocab_size)
            offs.append([s, e])
            mask.append(1)
        while len(ids) < max_length:
            ids.append(0)
            offs.append([0, 0])
            mask.append(0)
        result = {
            "input_ids": torch.tensor([ids], dtype=torch.long),
            "attention_mask": torch.tensor([mask], dtype=torch.long),
        }
        if return_offsets_mapping:
            result["offset_mapping"] = torch.tensor([offs], dtype=torch.long)
        return result


class TestWordSpans:
    def test_handles_leading_whitespace(self) -> None:
        spans, toks = _word_spans("  hello world")
        assert toks == ["hello", "world"]
        assert spans == [(2, 7), (8, 13)]

    def test_empty(self) -> None:
        assert _word_spans("") == ([], [])


class TestSubtokenMappers:
    def test_word_idx_maps_correctly(self) -> None:
        spans = [(0, 5), (6, 11)]
        offsets = [(0, 5), (6, 11), (0, 0)]
        assert _subtoken_to_word_idx(offsets, spans) == [0, 1, -1]

    def test_unit_idx_maps_to_enclosing_unit(self) -> None:
        unit_spans = [(0, 20), (21, 60)]
        offsets = [(3, 8), (25, 30), (0, 0)]
        assert _subtoken_to_unit_idx(offsets, unit_spans) == [0, 1, -1]


class TestPhiVector:
    def test_default_prose_type(self) -> None:
        row = {}
        v = _phi_vector_from_row(row)
        assert len(v) == 12
        # prose one-hot (index 11) is 1, everything else 0.
        assert v[11] == 1.0
        assert sum(v) == 1.0

    def test_code_phi_channels(self) -> None:
        row = {
            "phi_channels": {
                "exact_overlap": 0.8, "numeric_overlap": 0.3,
                "identifier_overlap": 0.5, "lemma_overlap": 0.1,
                "same_file": 1.0, "same_symbol": 0.0,
                "source_type": "code",
            }
        }
        v = _phi_vector_from_row(row)
        assert v[:4] == [0.8, 0.3, 0.5, 0.1]
        assert v[4:6] == [1.0, 0.0]
        assert v[6] == 1.0  # code one-hot
        assert v[7:12] == [0.0, 0.0, 0.0, 0.0, 0.0]


class TestBuildExample:
    def _row(self) -> dict:
        return {
            "pair_id": "test-pair",
            "class_key": "rag.prose.enterprise",
            "split_hint": "train",
            "role": "grounded",
            "paraphrase_variant": "v0",
            "response_text": "Revenue grew 12% in Q3 2026.",
            "evidence_text": (
                "Revenue grew 12% in Q3 2026. "
                "Operating margin expanded by 180 basis points."
            ),
            "modified_slots": [],
            "slot_values": {},
        }

    def test_build_example_produces_expected_tensors(self) -> None:
        row = self._row()
        labelled = label_synthetic_row(row).to_dict()
        tok = _FakeTokenizer()
        ex = build_example(
            row=row, labelled=labelled, tokenizer=tok,
            max_resp=32, max_ev=64, num_units=6,
        )
        assert ex["resp_ids"].shape == (32,)
        assert ex["ev_ids"].shape == (64,)
        assert ex["unit_assign"].shape == (64,)
        assert ex["token_support_aligned"].shape == (32,)
        assert ex["phi_vector"].shape == (12,)
        assert ex["class_idx"].item() == CLASS_TO_IDX["rag.prose.enterprise"]
        assert ex["dead_weight_unit_labels"].shape == (6,)
        assert ex["coverage_unit_labels"].shape == (6,)
        assert ex["pair_id"] == "test-pair"

    def test_build_example_units_clamped_to_range(self) -> None:
        row = self._row()
        labelled = label_synthetic_row(row).to_dict()
        tok = _FakeTokenizer()
        ex = build_example(
            row=row, labelled=labelled, tokenizer=tok,
            max_resp=32, max_ev=64, num_units=6,
        )
        ua = ex["unit_assign"]
        valid = ua[ua >= 0]
        assert (valid < 6).all()

    def test_build_batch_stacks_tensors(self) -> None:
        row = self._row()
        labelled = label_synthetic_row(row).to_dict()
        tok = _FakeTokenizer()
        examples = [
            build_example(
                row=row, labelled=labelled, tokenizer=tok,
                max_resp=32, max_ev=64, num_units=6,
            )
            for _ in range(4)
        ]
        batch = build_batch(examples)
        assert batch["resp_ids"].shape == (4, 32)
        assert batch["ev_ids"].shape == (4, 64)
        assert batch["unit_assign"].shape == (4, 64)
        assert batch["phi"].shape == (4, 32, 64, 12)
        assert batch["class_idx"].shape == (4,)
        assert len(batch["pair_ids"]) == 4

    def test_token_support_aligned_nonzero_for_grounded_overlap(self) -> None:
        row = self._row()
        labelled = label_synthetic_row(row).to_dict()
        tok = _FakeTokenizer()
        ex = build_example(
            row=row, labelled=labelled, tokenizer=tok,
            max_resp=32, max_ev=64, num_units=6,
        )
        # At least some response tokens should align to supported labels.
        assert ex["token_support_aligned"].sum().item() > 0
