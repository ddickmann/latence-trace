"""GPU-resident scorer slim for the phantom-guard + dead-weight tracer.

The production ``score_groundedness_response_chunked`` runs a big analysis
pipeline (NLI, semantic entropy, structured-evidence detection, null-bank
calibration, debug dense matrices, ...). For the phantom-guard request
lane we only need:

  - ``reverse_context``    – weighted mean of per-token max-cos against context
  - ``per_token_p10``      – 10th percentile of per-token max-cos
  - ``literal_guard``      – fraction of response literals matched in context
  - per-support-unit ``max_score`` and ``usage_state``

All matrix ops stay on ``device`` (default: CUDA) from input → output.
CPU hops happen exactly once at the end when we materialise the result
dict. The literal guard runs on tokenised strings; we accept it as a
small CPU side-channel.

Public API
----------

``GPUScorer(device="cuda")`` → callable that takes::

    out = scorer(
        response_tokens=aligned_tokens,
        response_embeddings=response_T_dim,
        support_units=[(tokens, emb_U_dim), ...],
        literal_tokens=[...],          # optional, from extract_literals()
    )

and returns::

    {
        "reverse_context": float,
        "per_token_p10": float,
        "per_token_p5": float,
        "per_token_min": float,
        "literal_guard": float,
        "per_unit_max": [...],         # len == n_support_units
        "usage_states": [...],         # "used" | "uncertain" | "unused"
        "latency_ms": float,
    }

Design notes
------------
- We never build a full ``(T, U, d)`` tensor — for each unit we compute
  ``R @ C_u.T`` (shape ``(T, Lu)``) and take ``max`` along ``Lu``, then
  accumulate per-token and per-unit maxima. That keeps peak memory
  bounded by the *largest* unit instead of the full context.
- ``R`` and ``C_u`` are both assumed L2-normalised (the GTE ColBERT
  tokens already are); cosine collapses to a plain matmul.
- The scorer is reentrant and threadsafe per-device; a singleton in the
  API layer is fine.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch


@dataclass
class UsageThresholds:
    """Cutoffs for the ternary ``used | uncertain | unused`` label."""

    used_min: float = 0.55
    uncertain_min: float = 0.40


class GPUScorer:
    """Callable scorer keeping everything on the configured device."""

    def __init__(
        self,
        device: str = "cuda",
        *,
        usage: Optional[UsageThresholds] = None,
        default_dtype: torch.dtype = torch.float32,
    ) -> None:
        self.device = torch.device(device)
        self.usage = usage or UsageThresholds()
        self.default_dtype = default_dtype

    # --- utilities -------------------------------------------------------

    def _prep(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.device != self.device or tensor.dtype != self.default_dtype:
            tensor = tensor.to(self.device, dtype=self.default_dtype, non_blocking=True)
        return tensor

    def _token_weights(self, tokens: Sequence[str]) -> torch.Tensor:
        """Uniform weights by default; downstream users may override by
        passing their own. This slim scorer keeps weights uniform which
        matches the simplified phantom-guard composite."""
        return torch.ones(len(tokens), dtype=self.default_dtype, device=self.device)

    # --- main call -------------------------------------------------------

    def __call__(
        self,
        *,
        response_tokens: Sequence[str],
        response_embeddings: torch.Tensor,
        support_units: Sequence[Tuple[Sequence[str], torch.Tensor]],
        literal_tokens: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        if response_embeddings.shape[0] == 0:
            raise ValueError("Response has zero tokens")
        if not support_units:
            raise ValueError("Support units list is empty")

        start = time.perf_counter()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            start = time.perf_counter()

        R = self._prep(response_embeddings)          # (T, d)
        T = R.shape[0]

        per_token_max = torch.full(
            (T,), -1.0, dtype=self.default_dtype, device=self.device
        )
        # per_token_owner[t] = index of the unit that currently owns token t's
        # global argmax. We use -1 to denote "no owner yet".
        per_token_owner = torch.full(
            (T,), -1, dtype=torch.int64, device=self.device
        )
        per_unit_max_scores: List[float] = []

        literal_set: Optional[set] = None
        if literal_tokens is not None:
            literal_set = {str(t).lower() for t in literal_tokens if t}

        support_text_flat: List[str] = []

        for unit_idx, (tokens, emb) in enumerate(support_units):
            C = self._prep(emb)                      # (L_u, d)
            # Cosine simplification (embeddings are L2-normalised already).
            sim = R @ C.T                            # (T, L_u)
            per_token_unit_max = sim.max(dim=1).values  # (T,)
            # Track owner: where this unit's score strictly exceeds the
            # current best, re-assign ownership to it.
            wins = per_token_unit_max > per_token_max
            per_token_owner = torch.where(
                wins,
                torch.full_like(per_token_owner, unit_idx),
                per_token_owner,
            )
            per_token_max = torch.maximum(per_token_max, per_token_unit_max)
            unit_max = float(per_token_unit_max.max().item())
            per_unit_max_scores.append(unit_max)
            if literal_set is not None:
                support_text_flat.extend(str(t).lower() for t in tokens)

        weights = self._token_weights(response_tokens)
        weight_sum = weights.sum().clamp(min=torch.finfo(self.default_dtype).eps)
        reverse_context = float(((per_token_max * weights).sum() / weight_sum).item())

        sorted_scores = torch.sort(per_token_max).values
        n = sorted_scores.shape[0]

        def _quantile(q: float) -> float:
            if n == 0:
                return 0.0
            idx = max(0, min(n - 1, int(q * (n - 1))))
            return float(sorted_scores[idx].item())

        per_token_p5 = _quantile(0.05)
        per_token_p10 = _quantile(0.10)
        per_token_min = float(sorted_scores[0].item()) if n else 0.0

        if literal_set is not None:
            support_set = set(support_text_flat)
            if literal_set:
                matched = sum(1 for t in literal_set if t in support_set)
                literal_guard = matched / len(literal_set)
            else:
                literal_guard = 1.0
        else:
            literal_guard = 1.0

        usage_states: List[str] = []
        for score in per_unit_max_scores:
            if score >= self.usage.used_min:
                usage_states.append("used")
            elif score >= self.usage.uncertain_min:
                usage_states.append("uncertain")
            else:
                usage_states.append("unused")

        # Per-unit ownership count: number of response tokens whose global
        # argmax lands inside this unit (regardless of absolute cosine). A
        # unit with zero owned tokens is a dead-weight candidate even when
        # its own per_unit_max saturates near 1.0.
        owner_counts = torch.zeros(
            len(support_units), dtype=torch.int64, device=self.device
        )
        if T > 0:
            valid = per_token_owner >= 0
            if valid.any():
                owner_counts.scatter_add_(
                    0,
                    per_token_owner[valid],
                    torch.ones(int(valid.sum().item()), dtype=torch.int64, device=self.device),
                )
        per_unit_owner_counts: List[int] = [int(x) for x in owner_counts.tolist()]

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        latency_ms = (time.perf_counter() - start) * 1000.0

        return {
            "reverse_context": reverse_context,
            "per_token_p10": per_token_p10,
            "per_token_p5": per_token_p5,
            "per_token_min": per_token_min,
            "literal_guard": float(literal_guard),
            "per_unit_max": per_unit_max_scores,
            "per_unit_owner_count": per_unit_owner_counts,
            "usage_states": usage_states,
            "n_tokens": int(T),
            "n_support_units": int(len(support_units)),
            "latency_ms": float(latency_ms),
        }


def self_test() -> Dict[str, Any]:
    """Smoke test: score a 32-token response against 3 support units on GPU
    (or CPU fallback) and report the latency + metric values."""
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dim = 128
    R = torch.randn(32, dim).float()
    R = R / R.norm(dim=1, keepdim=True)
    C1 = torch.randn(40, dim).float()
    C1 = C1 / C1.norm(dim=1, keepdim=True)
    C2 = torch.randn(25, dim).float()
    C2 = C2 / C2.norm(dim=1, keepdim=True)
    C3 = torch.randn(18, dim).float()
    C3 = C3 / C3.norm(dim=1, keepdim=True)
    scorer = GPUScorer(device=device)
    out = scorer(
        response_tokens=[f"tok{i}" for i in range(32)],
        response_embeddings=R,
        support_units=[
            ([f"c1_{i}" for i in range(40)], C1),
            ([f"c2_{i}" for i in range(25)], C2),
            ([f"c3_{i}" for i in range(18)], C3),
        ],
        literal_tokens=["tok0", "tok1"],
    )
    return out


if __name__ == "__main__":
    import json

    out = self_test()
    print(json.dumps({k: v for k, v in out.items() if k != "per_unit_max"}, indent=2))
    print("per_unit_max:", out["per_unit_max"])
