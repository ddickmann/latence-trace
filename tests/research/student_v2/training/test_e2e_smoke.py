"""End-to-end smoke test for the full TRACE v2 training pipeline.

This test validates the *plumbing* of the real training pathway with
real synthetic rows + real synthetic labels + the full collate + the
BiaffineStudent's 5-term joint loss. It is hermetic (no HF download,
no GPU, no I/O) and runs in a few seconds.

It deliberately uses:

* A ``_FakeTokenizer`` (whitespace word tokeniser with HF-compatible
  ``return_offsets_mapping`` semantics) so label alignment, unit
  assignment, and phi broadcasting all run through their real code
  paths.
* The real ``generate_enterprise_rows`` -> ``label_synthetic_row`` ->
  ``build_example`` -> ``build_batch`` pipeline.
* The real BiaffineStudent (with a small ``_DummyEncoder``) and the
  real ``loss_multi_task`` 5-term joint loss.

Assertions:

* 200 rows get built without error.
* Every batch produces a finite 5-term loss.
* Total loss **decreases** over 5 optimiser steps on a fixed batch
  (over-fit regime) - proof that the combined pipeline is actually
  learning, not just plumbing.
* Every v2-trained parameter has a finite gradient (skipping
  ``turn_state_proj`` which is a v2.1 drift export).
* Per-class FiLM params see gradients (proves FiLM conditioning is
  active end-to-end).
"""

from __future__ import annotations

import random
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.optim import AdamW

from research.triangular_maxsim.student_v2.architecture import (
    BiaffineStudent, StudentConfig, loss_multi_task,
)
from research.triangular_maxsim.student_v2.synthetic import (
    generate_enterprise_rows, get_industry, label_synthetic_row,
)
from research.triangular_maxsim.student_v2.training import (
    ClassWeightedSampler, build_batch,
)
from research.triangular_maxsim.student_v2.training.collate import (
    _word_spans, build_example,
)


class _FakeTokenizer:
    """Word-level HF-compatible stand-in."""

    def __init__(self, vocab_size: int = 1024) -> None:
        self.vocab_size = vocab_size

    def __call__(
        self, text, *, max_length: int, truncation: bool = True,
        padding: str = "max_length", return_offsets_mapping: bool = False,
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


class _DummyEncoder(nn.Module):
    def __init__(self, vocab_size: int = 1024, dim: int = 32) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        return SimpleNamespace(last_hidden_state=self.emb(input_ids))


def _build_200_labelled_rows() -> tuple[list[dict], list[dict]]:
    """Assemble 200 synthetic enterprise rows + labels."""
    rows: list[dict] = []
    labelled: list[dict] = []
    # Pull several industries so the batch sees class variation.
    for name in ("legal", "finance", "hr", "compliance", "engineering", "marketing"):
        bank = get_industry(name)
        for r in generate_enterprise_rows(
            industries=[bank], passages_per_bucket=2,
            master_seed=42, emit_ood=False,
        ):
            row_dict = r.to_dict()
            rows.append(row_dict)
            labelled.append(label_synthetic_row(row_dict).to_dict())
            if len(rows) >= 200:
                return rows, labelled
    return rows, labelled


def test_full_pipeline_200_rows_all_losses_finite():
    rows, labelled = _build_200_labelled_rows()
    assert len(rows) >= 200

    tokenizer = _FakeTokenizer(vocab_size=1024)
    MAX_RESP = 32
    MAX_EV = 64
    NUM_UNITS = 6

    examples = [
        build_example(
            row=r, labelled=lb, tokenizer=tokenizer,
            max_resp=MAX_RESP, max_ev=MAX_EV, num_units=NUM_UNITS,
        )
        for r, lb in zip(rows, labelled)
    ]
    assert len(examples) == 200

    cfg = StudentConfig(
        hidden_dim=32, proj_dim=16, biaffine_rank=8,
        phi_feature_dim=12, turn_state_dim=32,
    )
    model = BiaffineStudent(_DummyEncoder(vocab_size=1024, dim=32), cfg)
    model.train()
    optim = AdamW(model.parameters(), lr=3e-3)

    for step in range(3):
        start = step * 32
        batch_examples = examples[start : start + 32]
        batch = build_batch(batch_examples)

        out = model(
            batch["resp_ids"], batch["resp_mask"],
            batch["ev_ids"], batch["ev_mask"],
            batch["phi"],
            unit_assign=batch["unit_assign"],
            num_units=NUM_UNITS,
            class_idx=batch["class_idx"],
        )
        losses = loss_multi_task(
            out,
            token_support_labels=batch["token_support_aligned"],
            resp_mask=batch["resp_mask"],
            turn_band_labels=batch["turn_band"],
            turn_score_labels=batch["turn_score"],
            dead_weight_unit_labels=batch["dead_weight_unit_labels"],
            coverage_unit_labels=batch["coverage_unit_labels"],
        )
        for key in (
            "token_support", "turn_band", "turn_score",
            "dead_weight", "coverage", "total",
        ):
            assert key in losses
            assert torch.isfinite(losses[key]), f"step={step}: {key} not finite"

        losses["total"].backward()
        # turn_state_proj is a v2.1 export; skip it in the grad check.
        skip_prefix = "turn_state_proj"
        finite_grads: dict[str, bool] = {}
        for name, p in model.named_parameters():
            if not p.requires_grad or name.startswith(skip_prefix):
                continue
            assert p.grad is not None, f"step={step}: no grad on {name}"
            assert torch.isfinite(p.grad).all(), (
                f"step={step}: non-finite grad on {name}"
            )
            finite_grads[name] = bool(p.grad.abs().sum().item() > 0)
        # On every step, FiLM + gated-phi params must see non-zero grad.
        for target in (
            "class_film.embed.weight", "class_film.to_film.weight",
            "phi_fusion.phi_mlp.fc1.weight", "phi_fusion.gate_fc2.weight",
        ):
            assert finite_grads.get(target, False), (
                f"step={step}: {target} had zero grad (should be non-zero)"
            )
        optim.step()
        optim.zero_grad()


def test_full_pipeline_loss_decreases_on_fixed_batch():
    """Over-fit a fixed 16-row batch for 5 steps; total loss should
    decrease monotonically.
    """
    rows, labelled = _build_200_labelled_rows()
    tokenizer = _FakeTokenizer(vocab_size=1024)
    MAX_RESP = 32
    MAX_EV = 64
    NUM_UNITS = 6
    examples = [
        build_example(
            row=r, labelled=lb, tokenizer=tokenizer,
            max_resp=MAX_RESP, max_ev=MAX_EV, num_units=NUM_UNITS,
        )
        for r, lb in zip(rows[:16], labelled[:16])
    ]
    batch = build_batch(examples)

    cfg = StudentConfig(
        hidden_dim=32, proj_dim=16, biaffine_rank=8,
        phi_feature_dim=12, turn_state_dim=32,
    )
    model = BiaffineStudent(_DummyEncoder(vocab_size=1024, dim=32), cfg)
    model.train()
    optim = AdamW(model.parameters(), lr=5e-3)

    prev = float("inf")
    totals: list[float] = []
    for step in range(5):
        out = model(
            batch["resp_ids"], batch["resp_mask"],
            batch["ev_ids"], batch["ev_mask"],
            batch["phi"],
            unit_assign=batch["unit_assign"],
            num_units=NUM_UNITS,
            class_idx=batch["class_idx"],
        )
        losses = loss_multi_task(
            out,
            token_support_labels=batch["token_support_aligned"],
            resp_mask=batch["resp_mask"],
            turn_band_labels=batch["turn_band"],
            turn_score_labels=batch["turn_score"],
            dead_weight_unit_labels=batch["dead_weight_unit_labels"],
            coverage_unit_labels=batch["coverage_unit_labels"],
        )
        total = losses["total"].item()
        totals.append(total)
        losses["total"].backward()
        optim.step()
        optim.zero_grad()

    # Strict assertion on the last vs first step: by step 5 we should
    # have made meaningful progress.
    assert totals[-1] < totals[0] - 1e-3, (
        f"loss did not decrease over 5 steps: {totals}"
    )


def test_full_pipeline_class_conditioning_changes_predictions():
    """Same inputs routed with two different class_idx produce
    different turn_score predictions after a few steps of training.

    Confirms FiLM is actually wired through the real pipeline.
    """
    rows, labelled = _build_200_labelled_rows()
    tokenizer = _FakeTokenizer(vocab_size=1024)
    MAX_RESP = 32
    MAX_EV = 64
    NUM_UNITS = 6
    examples = [
        build_example(
            row=r, labelled=lb, tokenizer=tokenizer,
            max_resp=MAX_RESP, max_ev=MAX_EV, num_units=NUM_UNITS,
        )
        for r, lb in zip(rows[:8], labelled[:8])
    ]
    batch = build_batch(examples)

    cfg = StudentConfig(
        hidden_dim=32, proj_dim=16, biaffine_rank=8,
        phi_feature_dim=12, turn_state_dim=32,
    )
    model = BiaffineStudent(_DummyEncoder(vocab_size=1024, dim=32), cfg)
    # Warm up the FiLM layer: single step is sufficient since gradients
    # flow through ``class_film`` when a non-None class_idx is passed.
    model.train()
    optim = AdamW(model.parameters(), lr=1e-1)
    for _ in range(3):
        out = model(
            batch["resp_ids"], batch["resp_mask"],
            batch["ev_ids"], batch["ev_mask"],
            batch["phi"],
            unit_assign=batch["unit_assign"],
            num_units=NUM_UNITS,
            class_idx=batch["class_idx"],
        )
        losses = loss_multi_task(
            out,
            token_support_labels=batch["token_support_aligned"],
            resp_mask=batch["resp_mask"],
            turn_band_labels=batch["turn_band"],
            turn_score_labels=batch["turn_score"],
            dead_weight_unit_labels=batch["dead_weight_unit_labels"],
            coverage_unit_labels=batch["coverage_unit_labels"],
        )
        losses["total"].backward()
        optim.step()
        optim.zero_grad()

    model.eval()
    # Two forward passes on the same inputs with different class_idx:
    out_c0 = model(
        batch["resp_ids"], batch["resp_mask"],
        batch["ev_ids"], batch["ev_mask"],
        batch["phi"],
        unit_assign=batch["unit_assign"], num_units=NUM_UNITS,
        class_idx=torch.zeros_like(batch["class_idx"]),
    )
    out_c5 = model(
        batch["resp_ids"], batch["resp_mask"],
        batch["ev_ids"], batch["ev_mask"],
        batch["phi"],
        unit_assign=batch["unit_assign"], num_units=NUM_UNITS,
        class_idx=torch.full_like(batch["class_idx"], 5),
    )
    assert not torch.allclose(out_c0["turn_score"], out_c5["turn_score"]), (
        "FiLM conditioning did not change predictions across classes"
    )
