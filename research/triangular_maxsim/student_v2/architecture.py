"""TRACE v2 biaffine student architecture.

Implements the learned local-support head described in
``README.md``.  Compact enough to fit on a single A10 and export to
ONNX + TensorRT for the same latency envelope as the v1 MaxSim
reader.

Design follows the plan agreed with the product owner:

* tiny encoder (ModernBERT-base distilled or 6-layer MiniLM) emitting
  64-128d token embeddings;
* low-rank biaffine scorer (U @ V^T form to keep params < 32k);
* soft Top-k log-sum-exp pooling at train time, hard max at inference;
* three thin heads:
  - token support BCE
  - turn groundedness regression
  - turn band 3-way classification
* an external ``phi_features`` channel lets callers inject cheap
  matching bits (exact / numeric / identifier / source-type) without
  baking them into the encoder.

This file only contains the module definitions + a ``forward``
reference implementation.  Training loop + distillation dataset live
in ``train.py`` / ``distill_dataset.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class StudentConfig:
    hidden_dim: int = 384         # ModernBERT/MiniLM output size
    proj_dim: int = 96            # token projection used by the biaffine head
    biaffine_rank: int = 32       # low-rank factor for W = U @ V^T
    phi_feature_dim: int = 8      # number of cheap matching bits fed per pair
    topk: int = 4                 # soft Top-k pooling window at train time
    softmax_alpha: float = 8.0    # sharpness of log-sum-exp
    num_bands: int = 3            # {green, amber, red}


class LowRankBiaffine(nn.Module):
    """Low-rank biaffine scorer ``s_ij = h_i^T (U V^T) h_j + u.h_i + v.h_j + b``.

    Split into U and V keeps the parameter count at ``2 * d * r`` instead
    of ``d^2``.  ``r`` defaults to 32 which is empirically enough on
    verifier tasks and lets us export the matrix as a pair of Linear
    layers for ONNX.
    """

    def __init__(self, dim: int, rank: int) -> None:
        super().__init__()
        self.u_proj = nn.Linear(dim, rank, bias=False)
        self.v_proj = nn.Linear(dim, rank, bias=False)
        self.u_bias = nn.Linear(dim, 1, bias=False)
        self.v_bias = nn.Linear(dim, 1, bias=False)
        self.scalar_bias = nn.Parameter(torch.zeros(1))

    def forward(self, h_resp: torch.Tensor, h_ev: torch.Tensor) -> torch.Tensor:
        # h_resp: (B, Tr, D), h_ev: (B, Te, D)
        u = self.u_proj(h_resp)                # (B, Tr, R)
        v = self.v_proj(h_ev)                  # (B, Te, R)
        pair = torch.einsum("brd,bed->bre", u, v)  # (B, Tr, Te)
        pair = pair + self.u_bias(h_resp).expand(-1, -1, h_ev.size(1))
        pair = pair + self.v_bias(h_ev).squeeze(-1).unsqueeze(1).expand_as(pair)
        return pair + self.scalar_bias


class PhiFeatureMLP(nn.Module):
    """Cheap pairwise feature channel - concatenated to the biaffine logit."""

    def __init__(self, feat_dim: int, hidden: int = 32) -> None:
        super().__init__()
        self.fc1 = nn.Linear(feat_dim, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        # phi: (B, Tr, Te, F)
        x = F.gelu(self.fc1(phi))
        return self.fc2(x).squeeze(-1)


def soft_topk_logsumexp(
    scores: torch.Tensor, *, k: int, alpha: float
) -> torch.Tensor:
    """Differentiable approximation to ``max_j s_ij``.

    Train-time pooling.  At inference we simply call ``scores.max(dim=-1)``.
    ``alpha`` controls sharpness - higher means closer to a hard max.
    """

    if k < scores.size(-1):
        top, _ = scores.topk(k, dim=-1)
    else:
        top = scores
    return (1.0 / alpha) * torch.logsumexp(alpha * top, dim=-1)


class BiaffineStudent(nn.Module):
    """Full student: encoder -> biaffine -> pooling -> heads."""

    def __init__(self, encoder: nn.Module, cfg: Optional[StudentConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or StudentConfig()
        self.encoder = encoder
        self.proj = nn.Linear(self.cfg.hidden_dim, self.cfg.proj_dim)
        self.scorer = LowRankBiaffine(self.cfg.proj_dim, self.cfg.biaffine_rank)
        self.phi = PhiFeatureMLP(self.cfg.phi_feature_dim)
        # Heads.
        self.token_support = nn.Linear(self.cfg.proj_dim, 1)
        self.turn_score_head = nn.Sequential(
            nn.Linear(3, 16), nn.GELU(), nn.Linear(16, 1), nn.Sigmoid(),
        )
        self.turn_band_head = nn.Linear(3, self.cfg.num_bands)

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        return self.proj(hidden)

    def forward(
        self,
        resp_ids: torch.Tensor,
        resp_mask: torch.Tensor,
        ev_ids: torch.Tensor,
        ev_mask: torch.Tensor,
        phi_features: torch.Tensor,
        *,
        hard_pool: bool = False,
    ) -> dict[str, torch.Tensor]:
        h_r = self.encode(resp_ids, resp_mask)
        h_e = self.encode(ev_ids, ev_mask)

        pair_logits = self.scorer(h_r, h_e) + self.phi(phi_features)

        # Mask padded evidence positions with a large negative so they
        # never win the max / log-sum-exp.
        ev_bias = (~ev_mask.bool()).float() * -1e4
        pair_logits = pair_logits + ev_bias.unsqueeze(1)

        if hard_pool:
            token_g = pair_logits.max(dim=-1).values
        else:
            token_g = soft_topk_logsumexp(
                pair_logits, k=self.cfg.topk, alpha=self.cfg.softmax_alpha,
            )

        resp_mask_f = resp_mask.float()
        valid = resp_mask_f.sum(dim=1).clamp(min=1.0)
        token_g_masked = token_g * resp_mask_f
        turn_mean = token_g_masked.sum(dim=1) / valid
        turn_max = token_g_masked.masked_fill(~resp_mask.bool(), -1e4).max(dim=1).values
        turn_min = token_g_masked.masked_fill(~resp_mask.bool(), 1e4).min(dim=1).values

        turn_feat = torch.stack([turn_mean, turn_max, turn_min], dim=-1)
        turn_score = self.turn_score_head(turn_feat).squeeze(-1)
        turn_band_logits = self.turn_band_head(turn_feat)
        token_support_logits = self.token_support(h_r).squeeze(-1)

        return {
            "pair_logits": pair_logits,
            "token_g": token_g,
            "turn_score": turn_score,
            "turn_band_logits": turn_band_logits,
            "token_support_logits": token_support_logits,
        }


def loss_multi_task(
    out: dict[str, torch.Tensor],
    *,
    token_support_labels: torch.Tensor,
    resp_mask: torch.Tensor,
    turn_band_labels: torch.Tensor,
    turn_score_labels: torch.Tensor,
    pair_indices: Optional[torch.Tensor] = None,
    pair_margin: float = 0.2,
) -> dict[str, torch.Tensor]:
    """Multi-task loss: token BCE + band CE + score MSE + pair ranking."""

    losses: dict[str, torch.Tensor] = {}

    bce = F.binary_cross_entropy_with_logits(
        out["token_support_logits"],
        token_support_labels.float(),
        reduction="none",
    )
    mask = resp_mask.float()
    losses["token_support"] = (bce * mask).sum() / mask.sum().clamp(min=1.0)

    losses["turn_band"] = F.cross_entropy(out["turn_band_logits"], turn_band_labels)
    losses["turn_score"] = F.mse_loss(out["turn_score"], turn_score_labels)

    if pair_indices is not None and pair_indices.numel() > 0:
        # pair_indices: (P, 2) - (grounded_idx, hallucinated_idx)
        grounded = out["turn_score"][pair_indices[:, 0]]
        hallucinated = out["turn_score"][pair_indices[:, 1]]
        losses["pair_ranking"] = F.relu(pair_margin - (grounded - hallucinated)).mean()

    losses["total"] = (
        1.0 * losses["token_support"]
        + 1.0 * losses["turn_band"]
        + 0.5 * losses["turn_score"]
        + (0.5 * losses["pair_ranking"] if "pair_ranking" in losses else 0.0)
    )
    return losses


__all__ = [
    "BiaffineStudent",
    "LowRankBiaffine",
    "PhiFeatureMLP",
    "StudentConfig",
    "loss_multi_task",
    "soft_topk_logsumexp",
]
