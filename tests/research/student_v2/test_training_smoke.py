"""200-row smoke training on the synthetic toy dataset.

Validates the end-to-end plumbing for Phase B:

  1. ``collect_rows`` produces a non-empty corpus with all three-axis
     label placeholders.
  2. A toy label synthesizer can fill plausible dead-weight + coverage
     labels from the row's ``role`` (grounded -> all-coverage /
     no-dead-weight; adversarial -> partial coverage + one dead-weight
     unit).
  3. The full five-head ``BiaffineStudent`` forward works on these
     rows, the five-term ``loss_multi_task`` produces finite values
     for every term, and gradients flow through every parameter
     without NaN / Inf across three optimiser steps.

The encoder is a hermetic ``_DummyEncoder`` (no transformers download
required). Tokenisation is a character-hash tokeniser - the goal
here is plumbing validation, not model quality.
"""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from torch.optim import AdamW

from research.triangular_maxsim.student_v2.architecture import (
    BiaffineStudent,
    StudentConfig,
    loss_multi_task,
)
from research.triangular_maxsim.student_v2.distill_dataset import (
    MIN_JACCARD_SYNTHETIC,
    collect_rows,
)


VOCAB_SIZE = 512
RESP_MAX = 32
EV_MAX = 64
UNITS_MAX = 6


class _DummyEncoder(nn.Module):
    def __init__(self, vocab: int = VOCAB_SIZE, dim: int = 32) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        x = self.emb(input_ids)
        return SimpleNamespace(last_hidden_state=x)


def _char_hash_tokenise(text: str, max_len: int) -> tuple[list[int], list[int]]:
    words = (text or "").split()[:max_len]
    ids: list[int] = []
    for w in words:
        h = hashlib.md5(w.encode("utf-8")).digest()
        ids.append(int.from_bytes(h[:2], "big") % VOCAB_SIZE)
    mask = [1] * len(ids)
    while len(ids) < max_len:
        ids.append(0)
        mask.append(0)
    return ids, mask


def _chunk_unit_assign(ev_mask: list[int], num_units: int) -> list[int]:
    valid_positions = [i for i, m in enumerate(ev_mask) if m == 1]
    if not valid_positions:
        return [-1] * len(ev_mask)
    per_unit = max(1, len(valid_positions) // num_units)
    assign = [-1] * len(ev_mask)
    for rank, pos in enumerate(valid_positions):
        u = min(num_units - 1, rank // per_unit)
        assign[pos] = u
    return assign


def _toy_three_axis_labels(row: dict, num_units: int = UNITS_MAX):
    role = row.get("role") or "grounded"
    dead = [0.0] * num_units
    cov = [0.0] * num_units
    if role == "grounded":
        for u in range(num_units):
            cov[u] = 0.9
    elif role == "support_drop":
        for u in range(num_units):
            cov[u] = 0.7
        dead[-1] = 1.0
    else:
        for u in range(num_units):
            cov[u] = 0.6
        dead[0] = 1.0
    return dead, cov


def _band_to_idx(band):
    return {"green": 0, "amber": 1, "red": 2}.get((band or "").lower(), 1)


def _build_toy_batch(rows):
    B = len(rows)
    resp_ids = torch.zeros(B, RESP_MAX, dtype=torch.long)
    resp_mask = torch.zeros(B, RESP_MAX, dtype=torch.long)
    ev_ids = torch.zeros(B, EV_MAX, dtype=torch.long)
    ev_mask = torch.zeros(B, EV_MAX, dtype=torch.long)
    unit_assign = torch.full((B, EV_MAX), -1, dtype=torch.long)
    band = torch.zeros(B, dtype=torch.long)
    score = torch.zeros(B, dtype=torch.float32)
    token_support = torch.zeros(B, RESP_MAX, dtype=torch.float32)
    dead = torch.zeros(B, UNITS_MAX, dtype=torch.float32)
    cov = torch.zeros(B, UNITS_MAX, dtype=torch.float32)

    for i, row in enumerate(rows):
        r_ids, r_m = _char_hash_tokenise(row.get("response_text", ""), RESP_MAX)
        e_ids, e_m = _char_hash_tokenise(row.get("evidence_text", ""), EV_MAX)
        resp_ids[i] = torch.tensor(r_ids)
        resp_mask[i] = torch.tensor(r_m)
        ev_ids[i] = torch.tensor(e_ids)
        ev_mask[i] = torch.tensor(e_m)
        ua = _chunk_unit_assign(e_m, UNITS_MAX)
        unit_assign[i] = torch.tensor(ua)
        band[i] = _band_to_idx(row.get("turn_band") or row.get("gold_band"))
        score[i] = float(row.get("turn_score") or 0.5)
        d, c = _toy_three_axis_labels(row)
        dead[i] = torch.tensor(d)
        cov[i] = torch.tensor(c)

    phi = torch.zeros(B, RESP_MAX, EV_MAX, 12, dtype=torch.float32)
    return {
        "resp_ids": resp_ids, "resp_mask": resp_mask,
        "ev_ids": ev_ids, "ev_mask": ev_mask,
        "phi": phi, "unit_assign": unit_assign,
        "turn_band": band, "turn_score": score,
        "token_support_labels": token_support,
        "dead_weight_unit_labels": dead,
        "coverage_unit_labels": cov,
    }


def test_smoke_training_200_rows_all_five_losses_finite():
    rows, stats = collect_rows(
        audit_log=None,
        passages_per_bucket=3,
        master_seed=42,
        emit_ood=False,
        min_jaccard=MIN_JACCARD_SYNTHETIC,
        audit_log_cap=10,
    )
    assert stats.kept > 0
    labelled = [r for r in rows if r.get("gold_band")]
    assert len(labelled) >= 50
    toy = labelled[:200]
    if len(toy) < 200:
        toy = (toy * (200 // max(1, len(toy)) + 1))[:200]

    cfg = StudentConfig(
        hidden_dim=32, proj_dim=16, biaffine_rank=8,
        phi_feature_dim=12, turn_state_dim=32,
    )
    encoder = _DummyEncoder(dim=32)
    model = BiaffineStudent(encoder, cfg)
    model.train()
    optim = AdamW(model.parameters(), lr=1e-3)

    for step in range(3):
        start = step * 32
        batch_rows = toy[start : start + 32]
        batch = _build_toy_batch(batch_rows)

        out = model(
            batch["resp_ids"], batch["resp_mask"],
            batch["ev_ids"], batch["ev_mask"],
            batch["phi"],
            unit_assign=batch["unit_assign"],
            num_units=UNITS_MAX,
        )
        losses = loss_multi_task(
            out,
            token_support_labels=batch["token_support_labels"],
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
            assert key in losses, f"step={step}: missing loss term {key!r}"
            assert torch.isfinite(losses[key]), (
                f"step={step}: loss[{key}]={float(losses[key])} not finite"
            )

        losses["total"].backward()
        # Every v2-trained parameter must have a finite grad. The
        # ``turn_state_proj`` layer is deliberately a v2.1 drift
        # handoff export and is **not** reached by any v2 loss term,
        # so we skip it here; v2.1's session head will train it.
        v21_handoff_prefix = "turn_state_proj"
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith(v21_handoff_prefix):
                continue
            assert p.grad is not None, f"step={step}: no grad on {name}"
            assert torch.isfinite(p.grad).all(), (
                f"step={step}: non-finite grad on {name}"
            )
        optim.step()
        optim.zero_grad()


def test_smoke_loss_decreases_over_three_steps():
    """Sanity: total loss should trend downward across 3 optimiser
    steps on the toy corpus. A downward trend (not strictly
    monotonic) is enough of a signal that the 5-term loss is
    meaningfully driving optimisation; measuring actual convergence
    is the job of Phase C (full training), not this smoke.
    """
    rows, stats = collect_rows(
        audit_log=None,
        passages_per_bucket=3,
        master_seed=42,
        emit_ood=False,
        min_jaccard=MIN_JACCARD_SYNTHETIC,
        audit_log_cap=0,
    )
    labelled = [r for r in rows if r.get("gold_band")]
    if len(labelled) < 64:
        pytest.skip("not enough labelled rows")

    cfg = StudentConfig(
        hidden_dim=32, proj_dim=16, biaffine_rank=8,
        phi_feature_dim=12, turn_state_dim=32,
    )
    encoder = _DummyEncoder(dim=32)
    model = BiaffineStudent(encoder, cfg)
    optim = AdamW(model.parameters(), lr=3e-3)

    # Train on the same batch 3 times (over-fitting regime). Loss
    # should decrease monotonically because we hold the data fixed.
    chunk = labelled[:32]
    batch = _build_toy_batch(chunk)
    prev = float("inf")
    model.train()
    for step in range(3):
        out = model(
            batch["resp_ids"], batch["resp_mask"], batch["ev_ids"],
            batch["ev_mask"], batch["phi"],
            unit_assign=batch["unit_assign"], num_units=UNITS_MAX,
        )
        losses = loss_multi_task(
            out,
            token_support_labels=batch["token_support_labels"],
            resp_mask=batch["resp_mask"],
            turn_band_labels=batch["turn_band"],
            turn_score_labels=batch["turn_score"],
            dead_weight_unit_labels=batch["dead_weight_unit_labels"],
            coverage_unit_labels=batch["coverage_unit_labels"],
        )
        total = losses["total"].item()
        assert total < prev + 1e-3, (
            f"step={step}: total loss did not decrease: prev={prev:.4f} now={total:.4f}"
        )
        prev = total
        losses["total"].backward()
        optim.step()
        optim.zero_grad()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
