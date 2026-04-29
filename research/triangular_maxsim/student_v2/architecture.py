"""TRACE v2 biaffine student architecture.

Implements the learned local-support head described in
``v2_biaffine_student_c9f07f3d.plan.md``. Compact enough to fit on a
single A10 and export to ONNX + TensorRT for the same latency envelope
as the v1 MaxSim reader.

Five task heads, all fed from a single shared biaffine pair-logit
matrix ``S[R, E]``:

* Response-side (hallucination axis):
  - ``token_support_logits`` — per-response-token BCE target.
  - ``turn_score``           — per-turn regression target in [0, 1].
  - ``turn_band_logits``     — per-turn 3-way CE (green / amber / red).
* Evidence-side (dead-weight + coverage axes):
  - ``dead_weight_unit_logits`` — per-evidence-unit BCE target.
  - ``coverage_unit_scores``    — per-evidence-unit regression in [0, 1].

All five heads share the biaffine scorer's pair-logit matrix; the two
new evidence-side heads pool it *over the response axis* (per-evidence
token max / mean / min) and then aggregate per-unit with
``scatter_reduce`` ops. Zero additional encoder forward passes.

In addition to the five heads, the forward exposes
``turn_state_vector`` — a 32-d pooled response representation reserved
as the v2.1 drift-head handoff. v2 does not consume it; v2.1's session
head reads a sequence of these at rollup time without retraining the
student.

* ``phi_features`` is an external cheap-feature channel (12 bits by
  default: exact / lemma / identifier / numeric match, same-file,
  same-symbol, and a 6-way source-type one-hot for
  ``{code, test, docstring, prose, table, log}`` pairs).
* Training pooling is ``soft_topk_logsumexp`` (differentiable top-k
  LSE). Inference pooling can be ``hard_max`` (legacy) or
  ``top2_mean_max`` (the new default; robust on long evidence).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class StudentConfig:
    # Encoder geometry (matches ModernBERT-small / MiniLM-L6 class).
    hidden_dim: int = 384
    proj_dim: int = 96

    # Biaffine scorer.
    biaffine_rank: int = 32

    # Cheap phi feature channel: 12 bits.
    #   4 lexical: exact / lemma / identifier / numeric match
    #   2 structural: same-file / same-symbol
    #   6 source-type one-hot pair: code / test / docstring / prose /
    #     table / log (asymmetric response-vs-evidence role)
    phi_feature_dim: int = 12

    # Pooling (response side).
    topk: int = 4
    softmax_alpha: float = 8.0
    # Inference pooling strategy: "hard_max" matches v1 TRACE; the new
    # default "top2_mean_max" is the average of the top-2 scores per
    # response token, which is more robust on long evidence (single
    # noisy argmax tokens can otherwise drag the turn score).
    infer_pool: str = "top2_mean_max"

    # Heads.
    num_bands: int = 3  # {green, amber, red}

    # Dead-weight + coverage heads read a 3-feature per-unit summary
    # ``[max, mean, min]`` over the per-evidence-token signal. Small
    # hidden dim keeps them cheap (~few hundred params each).
    unit_feat_hidden: int = 32

    # v2.1 drift handoff: pooled response representation exposed on
    # every forward. Consumed by the v2.1 session head at rollup time.
    # Zero training cost for v2.
    turn_state_dim: int = 32


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------


class LowRankBiaffine(nn.Module):
    """Low-rank biaffine scorer.

    ``s_ij = h_i^T (U V^T) h_j + u.h_i + v.h_j + b``.

    Splitting ``W = U V^T`` keeps the parameter count at ``2 * d * r``
    instead of ``d^2``. ``r`` defaults to 32 which is empirically
    enough on verifier tasks and lets us export the matrix as a pair
    of ``Linear`` layers for ONNX.
    """

    def __init__(self, dim: int, rank: int) -> None:
        super().__init__()
        self.u_proj = nn.Linear(dim, rank, bias=False)
        self.v_proj = nn.Linear(dim, rank, bias=False)
        self.u_bias = nn.Linear(dim, 1, bias=False)
        self.v_bias = nn.Linear(dim, 1, bias=False)
        self.scalar_bias = nn.Parameter(torch.zeros(1))

    def forward(self, h_resp: torch.Tensor, h_ev: torch.Tensor) -> torch.Tensor:
        u = self.u_proj(h_resp)                         # (B, Tr, R)
        v = self.v_proj(h_ev)                           # (B, Te, R)
        pair = torch.einsum("brd,bed->bre", u, v)       # (B, Tr, Te)
        pair = pair + self.u_bias(h_resp).expand(-1, -1, h_ev.size(1))
        pair = pair + self.v_bias(h_ev).squeeze(-1).unsqueeze(1).expand_as(pair)
        return pair + self.scalar_bias


class PhiFeatureMLP(nn.Module):
    """Cheap pairwise feature channel — added to the biaffine logit."""

    def __init__(self, feat_dim: int, hidden: int = 32) -> None:
        super().__init__()
        self.fc1 = nn.Linear(feat_dim, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        # phi: (B, Tr, Te, F)
        x = F.gelu(self.fc1(phi))
        return self.fc2(x).squeeze(-1)


class DeadWeightHead(nn.Module):
    """Per-evidence-unit dead-weight classifier.

    Reads the 3-feature per-unit summary ``[max, mean, min]`` of the
    per-evidence-token argmax signal and emits a BCE logit. Trained
    against the teacher's ``file_attribution.per_unit.dead_weight``
    gold labels (binary: True iff the unit never won a response-token
    argmax or had all cosines below 0.40 or was dominated by a
    single-file peer).
    """

    def __init__(self, input_dim: int = 3, hidden: int = 32) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, unit_features: torch.Tensor) -> torch.Tensor:
        # unit_features: (B, U, 3)
        h = F.gelu(self.fc1(unit_features))
        return self.fc2(h).squeeze(-1)  # (B, U)


class CoverageHead(nn.Module):
    """Per-evidence-unit coverage score regressor.

    Reads the same 3-feature per-unit summary and emits a value in
    ``[0, 1]`` via sigmoid. Trained against the teacher's
    ``compute_unit_coverage.per_unit_max`` continuous labels.
    """

    def __init__(self, input_dim: int = 3, hidden: int = 32) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, unit_features: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.fc1(unit_features))
        return torch.sigmoid(self.fc2(h).squeeze(-1))  # (B, U)


# ---------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------


def soft_topk_logsumexp(
    scores: torch.Tensor, *, k: int, alpha: float
) -> torch.Tensor:
    """Differentiable approximation to ``max_j s_ij``.

    Train-time pooling. Returns ``(1/alpha) * logsumexp(alpha * top_k
    scores)`` which converges to hard-max as ``alpha -> inf`` and to
    the mean of the top-k scores as ``alpha -> 0``.
    """
    if k < scores.size(-1):
        top, _ = scores.topk(k, dim=-1)
    else:
        top = scores
    return (1.0 / alpha) * torch.logsumexp(alpha * top, dim=-1)


def top_k_mean_max(scores: torch.Tensor, *, k: int = 2) -> torch.Tensor:
    """Inference-time pooling: mean of the top-k scores per row.

    More robust than pure hard-max on long evidence because a single
    noisy argmax token cannot by itself drag the turn score up or
    down. Equivalent to hard-max when ``k == 1``.
    """
    k = min(max(k, 1), scores.size(-1))
    top, _ = scores.topk(k, dim=-1)
    return top.mean(dim=-1)


# ---------------------------------------------------------------------------
# Per-evidence-unit aggregation
# ---------------------------------------------------------------------------


def compute_unit_features(
    per_evidence_signal: torch.Tensor,
    unit_assign: torch.Tensor,
    num_units: int,
    *,
    ev_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Roll per-evidence-token signal into per-unit ``[max, mean, min]``.

    Args:
        per_evidence_signal: ``(B, Te)`` — typically the per-evidence-
            token argmax over response tokens of the pair-logit matrix.
        unit_assign: ``(B, Te)`` int tensor mapping each evidence
            position to a unit index in ``[0, num_units)``. Positions
            outside that range (or masked by ``ev_mask``) are dropped.
        num_units: maximum number of units in the batch.
        ev_mask: optional ``(B, Te)`` boolean mask (True = valid
            token). If provided, masked positions are excluded from
            the aggregation.

    Returns:
        ``(unit_features, unit_mask)`` where ``unit_features`` is
        ``(B, U, 3)`` with columns ``[max, mean, min]`` and
        ``unit_mask`` is a ``(B, U)`` bool marking units that had at
        least one valid token.
    """
    batch_size, _ = per_evidence_signal.shape
    U = int(num_units)
    device = per_evidence_signal.device
    dtype = per_evidence_signal.dtype

    in_range = (unit_assign >= 0) & (unit_assign < U)
    if ev_mask is not None:
        in_range = in_range & ev_mask.bool()
    idx = unit_assign.clamp(min=0, max=U - 1)

    # Max: initialise -inf-ish, scatter amax over valid positions.
    src_max = torch.where(
        in_range, per_evidence_signal, torch.full_like(per_evidence_signal, -1e4)
    )
    unit_max = torch.full((batch_size, U), -1e4, device=device, dtype=dtype)
    unit_max.scatter_reduce_(
        dim=1, index=idx, src=src_max, reduce="amax", include_self=True
    )

    # Sum + count -> mean.
    src_sum = torch.where(
        in_range, per_evidence_signal, torch.zeros_like(per_evidence_signal)
    )
    unit_sum = torch.zeros((batch_size, U), device=device, dtype=dtype)
    unit_sum.scatter_add_(dim=1, index=idx, src=src_sum)
    ones = in_range.to(dtype)
    unit_count = torch.zeros((batch_size, U), device=device, dtype=dtype)
    unit_count.scatter_add_(dim=1, index=idx, src=ones)
    unit_mask = unit_count > 0
    unit_mean = unit_sum / unit_count.clamp(min=1.0)

    # Min: initialise +inf-ish, scatter amin over valid positions.
    src_min = torch.where(
        in_range, per_evidence_signal, torch.full_like(per_evidence_signal, 1e4)
    )
    unit_min = torch.full((batch_size, U), 1e4, device=device, dtype=dtype)
    unit_min.scatter_reduce_(
        dim=1, index=idx, src=src_min, reduce="amin", include_self=True
    )

    # Units that saw no tokens: zero out so downstream MLP doesn't
    # receive ±1e4. The unit_mask must be checked by the loss.
    zeros = torch.zeros_like(unit_max)
    unit_max = torch.where(unit_mask, unit_max, zeros)
    unit_min = torch.where(unit_mask, unit_min, zeros)

    return torch.stack([unit_max, unit_mean, unit_min], dim=-1), unit_mask


# ---------------------------------------------------------------------------
# Full student
# ---------------------------------------------------------------------------


class BiaffineStudent(nn.Module):
    """Full student: encoder -> biaffine -> pooling -> 5 heads."""

    def __init__(
        self,
        encoder: nn.Module,
        cfg: Optional[StudentConfig] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg or StudentConfig()
        self.encoder = encoder
        self.proj = nn.Linear(self.cfg.hidden_dim, self.cfg.proj_dim)
        self.scorer = LowRankBiaffine(self.cfg.proj_dim, self.cfg.biaffine_rank)
        self.phi = PhiFeatureMLP(self.cfg.phi_feature_dim)

        # Response-side heads.
        self.token_support = nn.Linear(self.cfg.proj_dim, 1)
        self.turn_score_head = nn.Sequential(
            nn.Linear(3, 16), nn.GELU(), nn.Linear(16, 1), nn.Sigmoid(),
        )
        self.turn_band_head = nn.Linear(3, self.cfg.num_bands)

        # Evidence-side heads (new in v2).
        self.dead_weight_head = DeadWeightHead(
            input_dim=3, hidden=self.cfg.unit_feat_hidden
        )
        self.coverage_head = CoverageHead(
            input_dim=3, hidden=self.cfg.unit_feat_hidden
        )

        # v2.1 drift handoff: 32-d pooled response representation.
        # Exported on every forward; not consumed by any v2 head.
        self.turn_state_proj = nn.Linear(self.cfg.proj_dim, self.cfg.turn_state_dim)

    # .. encode .....................................................

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        return self.proj(hidden)

    # .. forward ....................................................

    def forward(
        self,
        resp_ids: torch.Tensor,
        resp_mask: torch.Tensor,
        ev_ids: torch.Tensor,
        ev_mask: torch.Tensor,
        phi_features: torch.Tensor,
        *,
        hard_pool: bool = False,
        unit_assign: Optional[torch.Tensor] = None,
        num_units: Optional[int] = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            resp_ids, resp_mask: response tokens + attention mask.
            ev_ids, ev_mask: evidence tokens + attention mask.
            phi_features: ``(B, Tr, Te, F)`` cheap-feature channel.
            hard_pool: if ``True``, use inference pooling
                (``hard_max`` or ``top2_mean_max`` depending on
                ``cfg.infer_pool``) instead of the differentiable
                ``soft_topk_logsumexp``.
            unit_assign: optional ``(B, Te)`` int tensor mapping each
                evidence token to a unit index. When provided (with
                ``num_units``), the dead-weight + coverage heads fire;
                when absent, they return ``None`` (legacy call path).
            num_units: maximum number of units in the batch.

        Returns:
            Dict with keys:
              - ``pair_logits``           (B, Tr, Te)
              - ``token_g``               (B, Tr)
              - ``turn_score``            (B,)
              - ``turn_band_logits``      (B, num_bands)
              - ``token_support_logits``  (B, Tr)
              - ``per_evidence_max``      (B, Te)
              - ``dead_weight_unit_logits`` (B, U) or None
              - ``coverage_unit_scores``    (B, U) or None
              - ``unit_mask``              (B, U) or None
              - ``turn_state_vector``     (B, turn_state_dim)  [v2.1 drift handoff]
        """
        h_r = self.encode(resp_ids, resp_mask)  # (B, Tr, D)
        h_e = self.encode(ev_ids, ev_mask)      # (B, Te, D)

        pair_logits = self.scorer(h_r, h_e) + self.phi(phi_features)

        # Mask padded evidence positions with a large negative so they
        # never win the max / log-sum-exp.
        ev_neg_bias = (~ev_mask.bool()).to(pair_logits.dtype) * -1e4
        pair_logits = pair_logits + ev_neg_bias.unsqueeze(1)

        # Response-side pooling over evidence axis.
        if hard_pool:
            if self.cfg.infer_pool == "top2_mean_max":
                token_g = top_k_mean_max(pair_logits, k=2)
            else:
                # "hard_max" fallback (parity with v1 MaxSim).
                token_g = pair_logits.max(dim=-1).values
        else:
            token_g = soft_topk_logsumexp(
                pair_logits, k=self.cfg.topk, alpha=self.cfg.softmax_alpha,
            )

        # Turn-level pooling over response axis (mean / max / min).
        resp_mask_f = resp_mask.to(token_g.dtype)
        valid = resp_mask_f.sum(dim=1).clamp(min=1.0)
        token_g_masked = token_g * resp_mask_f
        turn_mean = token_g_masked.sum(dim=1) / valid
        turn_max = token_g_masked.masked_fill(
            ~resp_mask.bool(), -1e4
        ).max(dim=1).values
        turn_min = token_g_masked.masked_fill(
            ~resp_mask.bool(), 1e4
        ).min(dim=1).values
        turn_feat = torch.stack([turn_mean, turn_max, turn_min], dim=-1)
        turn_score = self.turn_score_head(turn_feat).squeeze(-1)
        turn_band_logits = self.turn_band_head(turn_feat)
        token_support_logits = self.token_support(h_r).squeeze(-1)

        # Evidence-side pooling: per-evidence-token max over response
        # tokens. Masked response positions are filled with -1e4 so
        # they cannot dominate the max.
        resp_neg_bias = (~resp_mask.bool()).to(pair_logits.dtype).unsqueeze(-1) * -1e4
        per_evidence_max = (pair_logits + resp_neg_bias).max(dim=1).values  # (B, Te)

        # Per-unit aggregation + two evidence-side heads.
        dead_weight_unit_logits: Optional[torch.Tensor] = None
        coverage_unit_scores: Optional[torch.Tensor] = None
        unit_mask: Optional[torch.Tensor] = None
        if unit_assign is not None and num_units is not None and num_units > 0:
            unit_features, unit_mask = compute_unit_features(
                per_evidence_max,
                unit_assign,
                num_units,
                ev_mask=ev_mask,
            )
            dead_weight_unit_logits = self.dead_weight_head(unit_features)
            coverage_unit_scores = self.coverage_head(unit_features)
            # Zero out padded units so downstream aggregation over the
            # unit axis does not leak padded values.
            mask_f = unit_mask.to(dead_weight_unit_logits.dtype)
            dead_weight_unit_logits = dead_weight_unit_logits * mask_f
            coverage_unit_scores = coverage_unit_scores * mask_f

        # Turn-state vector (v2.1 drift handoff). Mean-pool the
        # response-side projected embeddings and project to 32-d.
        resp_mask_full = resp_mask_f.unsqueeze(-1)  # (B, Tr, 1)
        resp_valid = resp_mask_f.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled_resp = (h_r * resp_mask_full).sum(dim=1) / resp_valid  # (B, D)
        turn_state_vector = self.turn_state_proj(pooled_resp)  # (B, turn_state_dim)

        return {
            "pair_logits": pair_logits,
            "token_g": token_g,
            "turn_score": turn_score,
            "turn_band_logits": turn_band_logits,
            "token_support_logits": token_support_logits,
            "per_evidence_max": per_evidence_max,
            "dead_weight_unit_logits": dead_weight_unit_logits,
            "coverage_unit_scores": coverage_unit_scores,
            "unit_mask": unit_mask,
            "turn_state_vector": turn_state_vector,
        }


# ---------------------------------------------------------------------------
# 5-term joint loss
# ---------------------------------------------------------------------------


def loss_multi_task(
    out: dict[str, torch.Tensor],
    *,
    token_support_labels: torch.Tensor,
    resp_mask: torch.Tensor,
    turn_band_labels: torch.Tensor,
    turn_score_labels: torch.Tensor,
    dead_weight_unit_labels: Optional[torch.Tensor] = None,
    coverage_unit_labels: Optional[torch.Tensor] = None,
    pair_indices: Optional[torch.Tensor] = None,
    pair_margin: float = 0.2,
    # Per-term weights. Defaults match the Stage-2 recipe in the plan.
    lambda_support: float = 1.0,
    lambda_band: float = 1.0,
    lambda_score: float = 0.5,
    lambda_pair: float = 0.5,
    lambda_dead: float = 0.7,
    lambda_cov: float = 0.5,
) -> dict[str, torch.Tensor]:
    """5-term joint loss: support + band + score + pair + dead-weight + coverage.

    All terms are masked by label availability:
    - Dead-weight / coverage are optional; missing labels skip the term.
    - Pair ranking is optional; no pair_indices -> no term.
    - Dead-weight / coverage loss is masked by ``out["unit_mask"]`` so
      padded units don't contribute.
    """
    losses: dict[str, torch.Tensor] = {}

    # Token support BCE (response-side, masked by resp_mask).
    bce = F.binary_cross_entropy_with_logits(
        out["token_support_logits"],
        token_support_labels.to(out["token_support_logits"].dtype),
        reduction="none",
    )
    rmask = resp_mask.to(bce.dtype)
    losses["token_support"] = (bce * rmask).sum() / rmask.sum().clamp(min=1.0)

    # Turn-level heads.
    losses["turn_band"] = F.cross_entropy(out["turn_band_logits"], turn_band_labels)
    losses["turn_score"] = F.mse_loss(out["turn_score"], turn_score_labels)

    # Pair ranking (optional).
    if pair_indices is not None and pair_indices.numel() > 0:
        grounded = out["turn_score"][pair_indices[:, 0]]
        hallucinated = out["turn_score"][pair_indices[:, 1]]
        losses["pair_ranking"] = F.relu(
            pair_margin - (grounded - hallucinated)
        ).mean()

    # Evidence-side heads (new).
    unit_mask = out.get("unit_mask")
    dead_logits = out.get("dead_weight_unit_logits")
    if (
        dead_weight_unit_labels is not None
        and dead_logits is not None
    ):
        um = (
            unit_mask.to(dead_logits.dtype)
            if unit_mask is not None
            else torch.ones_like(dead_logits)
        )
        bce_dead = F.binary_cross_entropy_with_logits(
            dead_logits,
            dead_weight_unit_labels.to(dead_logits.dtype),
            reduction="none",
        )
        losses["dead_weight"] = (bce_dead * um).sum() / um.sum().clamp(min=1.0)

    cov_scores = out.get("coverage_unit_scores")
    if (
        coverage_unit_labels is not None
        and cov_scores is not None
    ):
        um = (
            unit_mask.to(cov_scores.dtype)
            if unit_mask is not None
            else torch.ones_like(cov_scores)
        )
        mse_cov = (cov_scores - coverage_unit_labels.to(cov_scores.dtype)) ** 2
        losses["coverage"] = (mse_cov * um).sum() / um.sum().clamp(min=1.0)

    # Weighted sum.
    total = (
        lambda_support * losses["token_support"]
        + lambda_band * losses["turn_band"]
        + lambda_score * losses["turn_score"]
    )
    if "pair_ranking" in losses:
        total = total + lambda_pair * losses["pair_ranking"]
    if "dead_weight" in losses:
        total = total + lambda_dead * losses["dead_weight"]
    if "coverage" in losses:
        total = total + lambda_cov * losses["coverage"]
    losses["total"] = total
    return losses


__all__ = [
    "BiaffineStudent",
    "CoverageHead",
    "DeadWeightHead",
    "LowRankBiaffine",
    "PhiFeatureMLP",
    "StudentConfig",
    "compute_unit_features",
    "loss_multi_task",
    "soft_topk_logsumexp",
    "top_k_mean_max",
]
