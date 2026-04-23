"""GPU-resident MaxSim scorer for the code lane.

Promoted from ``research/triangular_maxsim/coding/experiments/gpu_scorer.py``
and extended with query-aware ownership tracking used by the
file-attribution module.

Contract
--------
The scorer never builds a full ``(T, U, d)`` tensor. For every support
unit it computes ``R @ C_u.T`` (shape ``(T, L_u)``), reduces to per-token
maxima, and accumulates ownership against a rolling best so peak memory
is bounded by the *largest* unit rather than the full context.

All inputs are assumed L2-normalised by the upstream ColBERT encoder so
cosine similarity collapses to a plain matmul.

The scorer is reentrant and thread-safe per-device — a single instance
serves concurrent requests. All tensors flow through a dedicated
CUDA stream (when available) so multiple in-flight requests overlap
instead of serialising on the default stream.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UsageThresholds:
    """Cutoffs for the ternary ``used | uncertain | unused`` label.

    The defaults match the RAG lane's tri-state classifier so a support
    unit carrying a cosine of 0.55 reads the same way in both lanes.
    """

    used_min: float = 0.55
    uncertain_min: float = 0.40


@dataclass
class ScorerOutput:
    """Typed bundle returned by :class:`GPUScorer`.

    Every aggregate is materialised to a plain ``float`` / ``int`` /
    ``list`` *once* at the end of the forward pass. Callers stay on the
    CPU side for all subsequent work.
    """

    reverse_context: float
    per_token_p10: float
    per_token_p5: float
    per_token_min: float
    literal_guard: float
    per_unit_max: List[float]
    per_unit_owner_count: List[int]
    per_unit_query_owner_count: List[int]
    # ``per_unit_max_cos`` is kept as a stable-name alias of
    # ``per_unit_max`` (the scorer operates on L2-normalised ColBERT
    # vectors so MaxSim *is* a cosine). Downstream IDE dashboards pin
    # this field; we keep the alias to avoid forcing a client version
    # bump alongside the server rollout.
    per_unit_max_cos: List[float]
    usage_states: List[str]
    n_tokens: int
    n_support_units: int
    latency_ms: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "reverse_context": self.reverse_context,
            "per_token_p10": self.per_token_p10,
            "per_token_p5": self.per_token_p5,
            "per_token_min": self.per_token_min,
            "literal_guard": self.literal_guard,
            "per_unit_max": list(self.per_unit_max),
            "per_unit_owner_count": list(self.per_unit_owner_count),
            "per_unit_query_owner_count": list(self.per_unit_query_owner_count),
            "per_unit_max_cos": list(self.per_unit_max_cos),
            "usage_states": list(self.usage_states),
            "n_tokens": int(self.n_tokens),
            "n_support_units": int(self.n_support_units),
            "latency_ms": float(self.latency_ms),
        }


def _resolve_device(device: str | torch.device) -> torch.device:
    dev = torch.device(device) if isinstance(device, str) else device
    if dev.type == "cuda" and not torch.cuda.is_available():
        logger.warning("gpu_scorer_cuda_unavailable_falling_back", extra={"requested": str(dev)})
        return torch.device("cpu")
    return dev


class GPUScorer:
    """Callable scorer keeping every tensor on the configured device.

    Parameters
    ----------
    device:
        Either ``"cuda"``, ``"cpu"`` or a :class:`torch.device`. CUDA
        falls back to CPU gracefully when unavailable.
    usage:
        Custom tri-state cutoffs; defaults to the RAG lane settings.
    default_dtype:
        Internal compute dtype. ``float32`` keeps MaxSim numerically
        stable on the top-of-the-distribution where ``float16`` can
        saturate.
    use_dedicated_stream:
        On CUDA, run every forward pass on a per-instance stream so two
        concurrent requests overlap instead of sharing the default
        stream. Set ``False`` in benchmark harnesses to collapse
        ordering.
    """

    __slots__ = (
        "device",
        "usage",
        "default_dtype",
        "_stream",
        "_stream_lock",
        "_use_dedicated_stream",
    )

    def __init__(
        self,
        device: str | torch.device = "cuda",
        *,
        usage: Optional[UsageThresholds] = None,
        default_dtype: torch.dtype = torch.float32,
        use_dedicated_stream: bool = True,
    ) -> None:
        self.device = _resolve_device(device)
        self.usage = usage or UsageThresholds()
        self.default_dtype = default_dtype
        self._use_dedicated_stream = bool(use_dedicated_stream)
        self._stream: Optional[torch.cuda.Stream] = None
        self._stream_lock = threading.Lock()
        if self.device.type == "cuda" and self._use_dedicated_stream:
            try:
                self._stream = torch.cuda.Stream(device=self.device)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("gpu_scorer_stream_init_failed", extra={"error": str(exc)})
                self._stream = None

    # ------------------------------------------------------------------
    # utilities
    # ------------------------------------------------------------------

    def _prep(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move ``tensor`` onto the configured device/dtype lazily.

        Uses non-blocking transfers when the source tensor is pinned;
        otherwise falls back to the default blocking transfer.
        """
        if tensor.device == self.device and tensor.dtype == self.default_dtype:
            return tensor
        non_blocking = bool(getattr(tensor, "is_pinned", lambda: False)()) and tensor.device.type == "cpu"
        return tensor.to(self.device, dtype=self.default_dtype, non_blocking=non_blocking)

    @contextmanager
    def _stream_ctx(self):
        if self._stream is None:
            yield
            return
        with self._stream_lock:
            with torch.cuda.stream(self._stream):
                yield
            self._stream.synchronize()

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def __call__(
        self,
        *,
        response_tokens: Sequence[str],
        response_embeddings: torch.Tensor,
        support_units: Sequence[Tuple[Sequence[str], torch.Tensor]],
        literal_tokens: Optional[Sequence[str]] = None,
        query_embeddings: Optional[torch.Tensor] = None,
    ) -> ScorerOutput:
        if response_embeddings.shape[0] == 0:
            raise ValueError("Response has zero tokens")
        if not support_units:
            raise ValueError("Support units list is empty")

        with torch.inference_mode(), self._stream_ctx():
            return self._forward(
                response_tokens=response_tokens,
                response_embeddings=response_embeddings,
                support_units=support_units,
                literal_tokens=literal_tokens,
                query_embeddings=query_embeddings,
            )

    def _forward(
        self,
        *,
        response_tokens: Sequence[str],
        response_embeddings: torch.Tensor,
        support_units: Sequence[Tuple[Sequence[str], torch.Tensor]],
        literal_tokens: Optional[Sequence[str]],
        query_embeddings: Optional[torch.Tensor],
    ) -> ScorerOutput:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()

        R = self._prep(response_embeddings)
        T = R.shape[0]
        Q = self._prep(query_embeddings) if query_embeddings is not None else None

        per_token_max = torch.full(
            (T,), -1.0, dtype=self.default_dtype, device=self.device
        )
        per_token_owner = torch.full(
            (T,), -1, dtype=torch.int64, device=self.device
        )
        n_units = len(support_units)
        per_unit_max_scores: List[float] = []

        literal_set: Optional[set[str]] = None
        if literal_tokens is not None:
            literal_set = {str(t).lower() for t in literal_tokens if t}
        support_text_flat: List[str] = []

        # Query-side ownership (only computed when a query is supplied).
        if Q is not None:
            Qt = Q.shape[0]
            per_qtoken_max = torch.full(
                (Qt,), -1.0, dtype=self.default_dtype, device=self.device
            )
            per_qtoken_owner = torch.full(
                (Qt,), -1, dtype=torch.int64, device=self.device
            )
        else:
            Qt = 0
            per_qtoken_max = None
            per_qtoken_owner = None

        for unit_idx, (tokens, emb) in enumerate(support_units):
            C = self._prep(emb)
            sim = R @ C.T
            per_token_unit_max = sim.max(dim=1).values
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
            if Q is not None and per_qtoken_max is not None and per_qtoken_owner is not None:
                qsim = Q @ C.T
                per_q_unit_max = qsim.max(dim=1).values
                q_wins = per_q_unit_max > per_qtoken_max
                per_qtoken_owner = torch.where(
                    q_wins,
                    torch.full_like(per_qtoken_owner, unit_idx),
                    per_qtoken_owner,
                )
                per_qtoken_max = torch.maximum(per_qtoken_max, per_q_unit_max)

        # Weighted groundedness (uniform weights for the code lane — the
        # richer RAG-lane weighting lives in ``core.groundedness``).
        weights = torch.ones(T, dtype=self.default_dtype, device=self.device)
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

        # Literal guard (CPU side-channel — tiny cost vs. MaxSim kernel).
        if literal_set is not None:
            support_set = set(support_text_flat)
            if literal_set:
                matched = sum(1 for t in literal_set if t in support_set)
                literal_guard = matched / len(literal_set)
            else:
                literal_guard = 1.0
        else:
            literal_guard = 1.0

        # Per-unit usage state.
        usage_states: List[str] = []
        for score in per_unit_max_scores:
            if score >= self.usage.used_min:
                usage_states.append("used")
            elif score >= self.usage.uncertain_min:
                usage_states.append("uncertain")
            else:
                usage_states.append("unused")

        owner_counts = torch.zeros(
            n_units, dtype=torch.int64, device=self.device
        )
        if T > 0:
            valid = per_token_owner >= 0
            if bool(valid.any().item()):
                scatter_idx = per_token_owner[valid]
                owner_counts.scatter_add_(
                    0,
                    scatter_idx,
                    torch.ones(
                        int(valid.sum().item()),
                        dtype=torch.int64,
                        device=self.device,
                    ),
                )
        per_unit_owner_counts = [int(x) for x in owner_counts.tolist()]

        if Q is not None and per_qtoken_owner is not None:
            q_owner_counts = torch.zeros(
                n_units, dtype=torch.int64, device=self.device
            )
            if Qt > 0:
                q_valid = per_qtoken_owner >= 0
                if bool(q_valid.any().item()):
                    q_scatter = per_qtoken_owner[q_valid]
                    q_owner_counts.scatter_add_(
                        0,
                        q_scatter,
                        torch.ones(
                            int(q_valid.sum().item()),
                            dtype=torch.int64,
                            device=self.device,
                        ),
                    )
            per_unit_query_owner_counts = [int(x) for x in q_owner_counts.tolist()]
        else:
            per_unit_query_owner_counts = [0] * n_units

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        latency_ms = (time.perf_counter() - start) * 1000.0

        return ScorerOutput(
            reverse_context=float(reverse_context),
            per_token_p10=float(per_token_p10),
            per_token_p5=float(per_token_p5),
            per_token_min=float(per_token_min),
            literal_guard=float(literal_guard),
            per_unit_max=per_unit_max_scores,
            per_unit_owner_count=per_unit_owner_counts,
            per_unit_query_owner_count=per_unit_query_owner_counts,
            per_unit_max_cos=list(per_unit_max_scores),
            usage_states=usage_states,
            n_tokens=int(T),
            n_support_units=int(n_units),
            latency_ms=float(latency_ms),
        )


# ----------------------------------------------------------------------
# Module-level singleton for the production handler
# ----------------------------------------------------------------------

_DEFAULT_SINGLETON: Optional[GPUScorer] = None
_SINGLETON_LOCK = threading.Lock()


def get_default_scorer() -> GPUScorer:
    """Return a process-wide :class:`GPUScorer`, creating it lazily.

    Device is picked once from the ``LATENCE_TRACE_CODE_LANE_DEVICE`` or
    ``LATENCE_TRACE_SERVICE_DEVICE`` env var (CUDA when available) and
    the usage thresholds default to the RAG-lane values. Use this from
    the FastAPI / RunPod handler so concurrent requests share one
    scorer, one CUDA stream, and one stable pin.
    """
    global _DEFAULT_SINGLETON
    if _DEFAULT_SINGLETON is not None:
        return _DEFAULT_SINGLETON
    with _SINGLETON_LOCK:
        if _DEFAULT_SINGLETON is None:
            device = (
                os.environ.get("LATENCE_TRACE_CODE_LANE_DEVICE")
                or os.environ.get("LATENCE_TRACE_SERVICE_DEVICE")
                or ("cuda" if torch.cuda.is_available() else "cpu")
            )
            _DEFAULT_SINGLETON = GPUScorer(device=device)
    return _DEFAULT_SINGLETON


def reset_default_scorer() -> None:
    """Drop the cached singleton; used by tests and shutdown hooks."""
    global _DEFAULT_SINGLETON
    with _SINGLETON_LOCK:
        _DEFAULT_SINGLETON = None
