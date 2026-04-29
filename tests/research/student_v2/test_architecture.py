"""Unit tests for TRACE v2 biaffine student architecture.

Covers:
  - shape / grad for all five heads
  - phi mask behaviour (masked evidence tokens never win the max)
  - soft_topk_logsumexp converges to hard-max as alpha -> inf
  - top_k_mean_max returns the mean of the top-k values
  - compute_unit_features aggregates correctly via scatter_reduce
  - turn_state_vector has the right shape and is differentiable
  - loss_multi_task runs end-to-end with and without optional heads
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from research.triangular_maxsim.student_v2.architecture import (
    BiaffineStudent,
    CoverageHead,
    DeadWeightHead,
    LowRankBiaffine,
    PhiFeatureMLP,
    StudentConfig,
    compute_unit_features,
    loss_multi_task,
    soft_topk_logsumexp,
    top_k_mean_max,
)


# ---------------------------------------------------------------------------
# Fixtures / dummy encoder
# ---------------------------------------------------------------------------


class _DummyEncoder(nn.Module):
    """Minimal encoder stub: embedding lookup + positional bump.

    Keeps tests hermetic (no HF download) and deterministic.
    """

    def __init__(self, vocab: int = 128, dim: int = 32) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        x = self.emb(input_ids)
        return SimpleNamespace(last_hidden_state=x)


def _mk_student(
    *,
    hidden_dim: int = 32,
    proj_dim: int = 16,
    biaffine_rank: int = 8,
    phi_feature_dim: int = 12,
    turn_state_dim: int = 32,
) -> BiaffineStudent:
    cfg = StudentConfig(
        hidden_dim=hidden_dim,
        proj_dim=proj_dim,
        biaffine_rank=biaffine_rank,
        phi_feature_dim=phi_feature_dim,
        turn_state_dim=turn_state_dim,
    )
    encoder = _DummyEncoder(dim=hidden_dim)
    return BiaffineStudent(encoder, cfg)


def _mk_inputs(
    *,
    batch: int = 2,
    tr: int = 6,
    te: int = 8,
    phi_feature_dim: int = 12,
    units_per_batch: list[list[int]] | None = None,
):
    """Build a batch of toy inputs for the student.

    If ``units_per_batch`` is provided it is an int list-of-lists
    assigning each of the ``te`` evidence positions to a unit index.
    """
    torch.manual_seed(42)
    resp_ids = torch.randint(0, 64, (batch, tr))
    ev_ids = torch.randint(0, 64, (batch, te))
    resp_mask = torch.ones(batch, tr, dtype=torch.bool)
    ev_mask = torch.ones(batch, te, dtype=torch.bool)
    phi = torch.randn(batch, tr, te, phi_feature_dim) * 0.1

    unit_assign = None
    num_units = None
    if units_per_batch is not None:
        unit_assign = torch.tensor(units_per_batch, dtype=torch.long)
        num_units = int(unit_assign.max().item()) + 1
    return {
        "resp_ids": resp_ids,
        "resp_mask": resp_mask,
        "ev_ids": ev_ids,
        "ev_mask": ev_mask,
        "phi_features": phi,
        "unit_assign": unit_assign,
        "num_units": num_units,
    }


# ---------------------------------------------------------------------------
# Pooling primitives
# ---------------------------------------------------------------------------


class TestSoftTopKLSE:
    def test_converges_to_hard_max_as_alpha_grows(self) -> None:
        torch.manual_seed(0)
        scores = torch.randn(4, 10, 32)
        hard = scores.max(dim=-1).values
        for alpha in (50.0, 200.0):
            soft = soft_topk_logsumexp(scores, k=4, alpha=alpha)
            err = (soft - hard).abs().max().item()
            assert err < 0.25, f"alpha={alpha}: max err {err:.4f}"

    def test_is_differentiable(self) -> None:
        scores = torch.randn(2, 5, 7, requires_grad=True)
        soft = soft_topk_logsumexp(scores, k=3, alpha=8.0)
        soft.sum().backward()
        assert scores.grad is not None
        assert torch.isfinite(scores.grad).all()

    def test_handles_k_larger_than_last_dim(self) -> None:
        scores = torch.randn(2, 3, 4)
        # k=8 > last dim (4) -> should fall through to "all elements"
        result = soft_topk_logsumexp(scores, k=8, alpha=4.0)
        assert result.shape == (2, 3)


class TestTopKMeanMax:
    def test_k1_equals_hard_max(self) -> None:
        torch.manual_seed(0)
        scores = torch.randn(3, 4, 12)
        assert torch.allclose(top_k_mean_max(scores, k=1), scores.max(dim=-1).values)

    def test_k2_is_mean_of_top_2(self) -> None:
        scores = torch.tensor([[[1.0, 5.0, 3.0, 2.0]]])
        # Top-2 = {5, 3}, mean = 4
        assert torch.allclose(top_k_mean_max(scores, k=2), torch.tensor([[4.0]]))

    def test_k_larger_than_last_dim_clips(self) -> None:
        scores = torch.tensor([[[1.0, 5.0, 3.0, 2.0]]])
        # k=8 clips to last-dim=4 -> mean of all = 2.75
        assert torch.allclose(
            top_k_mean_max(scores, k=8),
            torch.tensor([[2.75]]),
        )


# ---------------------------------------------------------------------------
# Per-unit aggregation
# ---------------------------------------------------------------------------


class TestComputeUnitFeatures:
    def test_basic_max_mean_min(self) -> None:
        # Batch 1, 6 evidence tokens, 3 units:
        #   unit 0 -> positions [0, 1]
        #   unit 1 -> positions [2, 3]
        #   unit 2 -> positions [4, 5]
        signal = torch.tensor([[1.0, 3.0, 5.0, 5.0, 0.1, 0.9]])
        unit_assign = torch.tensor([[0, 0, 1, 1, 2, 2]])
        feats, mask = compute_unit_features(signal, unit_assign, num_units=3)
        # (1, 3, 3): [max, mean, min] per unit.
        assert feats.shape == (1, 3, 3)
        assert torch.allclose(feats[0, 0], torch.tensor([3.0, 2.0, 1.0]))
        assert torch.allclose(feats[0, 1], torch.tensor([5.0, 5.0, 5.0]))
        assert torch.allclose(feats[0, 2], torch.tensor([0.9, 0.5, 0.1]))
        assert mask.all()

    def test_padded_unit_marks_mask_false(self) -> None:
        # Only two real units present; pass num_units=4 -> units 2, 3
        # have no tokens.
        signal = torch.tensor([[0.5, 0.7, 0.9, 0.3]])
        unit_assign = torch.tensor([[0, 0, 1, 1]])
        feats, mask = compute_unit_features(signal, unit_assign, num_units=4)
        assert mask.shape == (1, 4)
        assert mask[0, 0] and mask[0, 1]
        assert not mask[0, 2] and not mask[0, 3]
        # Padded units return zeros, not +/-1e4.
        assert torch.allclose(feats[0, 2], torch.zeros(3))
        assert torch.allclose(feats[0, 3], torch.zeros(3))

    def test_ev_mask_excludes_positions(self) -> None:
        signal = torch.tensor([[1.0, 100.0, 3.0, 4.0]])
        unit_assign = torch.tensor([[0, 0, 1, 1]])
        # Mask out position 1 (the 100.0 outlier).
        ev_mask = torch.tensor([[True, False, True, True]])
        feats, _ = compute_unit_features(signal, unit_assign, 2, ev_mask=ev_mask)
        assert torch.allclose(feats[0, 0], torch.tensor([1.0, 1.0, 1.0]))

    def test_negative_unit_index_is_dropped(self) -> None:
        signal = torch.tensor([[1.0, 2.0, 3.0]])
        # -1 is "no unit" -> should be excluded.
        unit_assign = torch.tensor([[0, -1, 1]])
        feats, mask = compute_unit_features(signal, unit_assign, num_units=2)
        assert mask.tolist() == [[True, True]]
        assert torch.allclose(feats[0, 0], torch.tensor([1.0, 1.0, 1.0]))
        assert torch.allclose(feats[0, 1], torch.tensor([3.0, 3.0, 3.0]))


# ---------------------------------------------------------------------------
# Heads (shape + grad)
# ---------------------------------------------------------------------------


class TestHeadShapes:
    def test_dead_weight_head_shape_and_grad(self) -> None:
        head = DeadWeightHead(input_dim=3, hidden=32)
        feats = torch.randn(4, 6, 3, requires_grad=True)
        out = head(feats)
        assert out.shape == (4, 6)
        out.sum().backward()
        assert feats.grad is not None and torch.isfinite(feats.grad).all()

    def test_coverage_head_in_unit_interval(self) -> None:
        head = CoverageHead(input_dim=3, hidden=32)
        feats = torch.randn(4, 6, 3)
        out = head(feats)
        assert out.shape == (4, 6)
        assert ((out >= 0) & (out <= 1)).all()


class TestBiaffineScorer:
    def test_pair_logit_shape(self) -> None:
        scorer = LowRankBiaffine(dim=16, rank=8)
        h_r = torch.randn(2, 5, 16)
        h_e = torch.randn(2, 7, 16)
        out = scorer(h_r, h_e)
        assert out.shape == (2, 5, 7)


class TestPhiFeatureMLP:
    def test_forward_returns_scalar_per_pair(self) -> None:
        phi = PhiFeatureMLP(feat_dim=12, hidden=16)
        x = torch.randn(3, 5, 7, 12)
        out = phi(x)
        assert out.shape == (3, 5, 7)


# ---------------------------------------------------------------------------
# Full student forward pass
# ---------------------------------------------------------------------------


class TestStudentForward:
    def test_legacy_call_without_unit_assign(self) -> None:
        model = _mk_student()
        inputs = _mk_inputs()
        inputs.pop("unit_assign")
        inputs.pop("num_units")
        out = model(**inputs)
        # Core response-side outputs are always populated.
        assert out["pair_logits"].shape == (2, 6, 8)
        assert out["token_g"].shape == (2, 6)
        assert out["turn_score"].shape == (2,)
        assert out["turn_band_logits"].shape == (2, 3)
        assert out["token_support_logits"].shape == (2, 6)
        # Evidence-side heads deliberately off.
        assert out["dead_weight_unit_logits"] is None
        assert out["coverage_unit_scores"] is None
        assert out["unit_mask"] is None
        # Turn-state vector always emitted (v2.1 handoff).
        assert out["turn_state_vector"].shape == (2, 32)

    def test_with_unit_assign_activates_evidence_heads(self) -> None:
        model = _mk_student()
        # Two batch items, 8 evidence tokens, 3 units each.
        units = [
            [0, 0, 0, 1, 1, 2, 2, 2],
            [0, 1, 1, 1, 2, 2, 2, 2],
        ]
        inputs = _mk_inputs(units_per_batch=units)
        out = model(**inputs)
        assert out["dead_weight_unit_logits"].shape == (2, 3)
        assert out["coverage_unit_scores"].shape == (2, 3)
        assert out["unit_mask"].shape == (2, 3)
        assert out["unit_mask"].all()
        # Coverage is bounded in [0, 1].
        assert ((out["coverage_unit_scores"] >= 0) & (out["coverage_unit_scores"] <= 1)).all()

    def test_hard_pool_modes(self) -> None:
        for mode in ("top2_mean_max", "hard_max"):
            cfg = StudentConfig(hidden_dim=32, proj_dim=16, phi_feature_dim=12)
            cfg.infer_pool = mode
            model = BiaffineStudent(_DummyEncoder(dim=32), cfg)
            inputs = _mk_inputs()
            inputs.pop("unit_assign")
            inputs.pop("num_units")
            out = model(**inputs, hard_pool=True)
            assert out["token_g"].shape == (2, 6)

    def test_padded_evidence_cannot_win_max(self) -> None:
        model = _mk_student()
        inputs = _mk_inputs()
        # Mask out every evidence position except the first two.
        inputs["ev_mask"] = torch.tensor(
            [[True, True, False, False, False, False, False, False]] * 2,
            dtype=torch.bool,
        )
        inputs.pop("unit_assign")
        inputs.pop("num_units")
        out = model(**inputs)
        # Padded evidence gets -1e4 bias; pair_logits for those
        # positions should be very negative.
        pair = out["pair_logits"]
        for b in range(2):
            for e in (2, 3, 4, 5, 6, 7):
                assert pair[b, :, e].max().item() < -1000.0

    def test_gradient_flows_through_all_heads(self) -> None:
        model = _mk_student()
        units = [
            [0, 0, 1, 1, 2, 2, 2, 2],
            [0, 0, 0, 1, 1, 1, 2, 2],
        ]
        inputs = _mk_inputs(units_per_batch=units)
        out = model(**inputs)
        # Sum every output head into one scalar.
        loss = (
            out["turn_score"].sum()
            + out["turn_band_logits"].sum()
            + out["token_support_logits"].sum()
            + out["dead_weight_unit_logits"].sum()
            + out["coverage_unit_scores"].sum()
            + out["turn_state_vector"].sum()
        )
        loss.backward()
        # Every trainable parameter should have a finite, non-None grad.
        no_grad: list[str] = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is None:
                    no_grad.append(name)
                else:
                    assert torch.isfinite(param.grad).all(), f"non-finite grad at {name}"
        assert not no_grad, f"no grad on: {no_grad}"


# ---------------------------------------------------------------------------
# Joint loss
# ---------------------------------------------------------------------------


class TestLossMultiTask:
    def _build(self, *, with_units: bool = False):
        model = _mk_student()
        units = None
        if with_units:
            units = [[0, 0, 1, 1, 2, 2, 2, 2], [0, 0, 0, 1, 1, 1, 2, 2]]
        inputs = _mk_inputs(units_per_batch=units)
        out = model(**inputs)
        return out, inputs

    def test_basic_three_term_loss_no_evidence_heads(self) -> None:
        out, inputs = self._build(with_units=False)
        B, Tr = inputs["resp_ids"].shape
        labels = {
            "token_support_labels": torch.randint(0, 2, (B, Tr)),
            "resp_mask": inputs["resp_mask"],
            "turn_band_labels": torch.randint(0, 3, (B,)),
            "turn_score_labels": torch.rand(B),
        }
        losses = loss_multi_task(out, **labels)
        for key in ("token_support", "turn_band", "turn_score", "total"):
            assert key in losses
            assert torch.isfinite(losses[key])
        # No dead-weight / coverage terms when labels are absent.
        assert "dead_weight" not in losses
        assert "coverage" not in losses

    def test_five_term_loss_with_evidence_heads(self) -> None:
        out, inputs = self._build(with_units=True)
        B, Tr = inputs["resp_ids"].shape
        U = 3
        dead_labels = torch.randint(0, 2, (B, U)).float()
        cov_labels = torch.rand(B, U)
        losses = loss_multi_task(
            out,
            token_support_labels=torch.randint(0, 2, (B, Tr)),
            resp_mask=inputs["resp_mask"],
            turn_band_labels=torch.randint(0, 3, (B,)),
            turn_score_labels=torch.rand(B),
            dead_weight_unit_labels=dead_labels,
            coverage_unit_labels=cov_labels,
        )
        for key in (
            "token_support",
            "turn_band",
            "turn_score",
            "dead_weight",
            "coverage",
            "total",
        ):
            assert key in losses
            assert torch.isfinite(losses[key])

    def test_pair_ranking_term(self) -> None:
        # Build a 4-sample batch: pair (0, 1) grounded vs hallucinated.
        model = _mk_student()
        inputs = _mk_inputs(batch=4, tr=5, te=6)
        inputs.pop("unit_assign")
        inputs.pop("num_units")
        out = model(**inputs)
        B, Tr = inputs["resp_ids"].shape
        pair_indices = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
        losses = loss_multi_task(
            out,
            token_support_labels=torch.randint(0, 2, (B, Tr)),
            resp_mask=inputs["resp_mask"],
            turn_band_labels=torch.randint(0, 3, (B,)),
            turn_score_labels=torch.rand(B),
            pair_indices=pair_indices,
        )
        assert "pair_ranking" in losses
        assert torch.isfinite(losses["pair_ranking"])

    def test_backward_does_not_explode(self) -> None:
        out, inputs = self._build(with_units=True)
        B, Tr = inputs["resp_ids"].shape
        U = 3
        losses = loss_multi_task(
            out,
            token_support_labels=torch.randint(0, 2, (B, Tr)),
            resp_mask=inputs["resp_mask"],
            turn_band_labels=torch.randint(0, 3, (B,)),
            turn_score_labels=torch.rand(B),
            dead_weight_unit_labels=torch.randint(0, 2, (B, U)).float(),
            coverage_unit_labels=torch.rand(B, U),
        )
        losses["total"].backward()


# ---------------------------------------------------------------------------
# Turn-state vector (v2.1 drift handoff)
# ---------------------------------------------------------------------------


class TestTurnStateVector:
    def test_shape_matches_config(self) -> None:
        for dim in (16, 32, 64):
            cfg = StudentConfig(hidden_dim=32, proj_dim=16, phi_feature_dim=12, turn_state_dim=dim)
            model = BiaffineStudent(_DummyEncoder(dim=32), cfg)
            inputs = _mk_inputs()
            inputs.pop("unit_assign")
            inputs.pop("num_units")
            out = model(**inputs)
            assert out["turn_state_vector"].shape == (2, dim)

    def test_differentiable(self) -> None:
        model = _mk_student()
        inputs = _mk_inputs()
        inputs.pop("unit_assign")
        inputs.pop("num_units")
        out = model(**inputs)
        out["turn_state_vector"].sum().backward()
        assert model.turn_state_proj.weight.grad is not None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
