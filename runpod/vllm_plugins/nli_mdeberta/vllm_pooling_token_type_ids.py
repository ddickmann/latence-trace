"""
Patch vLLM 0.19 pooling token-type construction for DeBERTa NLI.

vLLM's V1 ``GPUModelRunner._init_model_kwargs`` builds pooled
``token_type_ids`` from ``PoolingParams.extra_kwargs["compressed_token_type_ids"]``.
In 0.19.0 it mixes CPU ``torch.arange`` tensors with CUDA ``seq_lens`` entries,
which kills the engine under batched NLI traffic.

This patch keeps the existing compressed token-type behavior, but constructs the
intermediate tensors on ``runner.device`` so pooled DeBERTa sequence-pair
requests remain stable.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)

_PATCH_ATTR = "_latence_trace_nli_token_type_ids_patched"


def _as_int(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.item())
    return int(value)


def _make_patched_init_model_kwargs(orig_init_model_kwargs):
    def _init_model_kwargs(self):
        if not getattr(self, "is_pooling_model", False):
            return orig_init_model_kwargs(self)

        num_reqs = self.input_batch.num_reqs
        pooling_params = self.input_batch.get_pooling_params()

        token_type_id_requests: dict[int, int] = {}
        for i, param in enumerate(pooling_params):
            extra_kwargs = getattr(param, "extra_kwargs", None)
            if not extra_kwargs:
                continue
            token_types = extra_kwargs.get("compressed_token_type_ids")
            if token_types is None:
                continue
            token_type_id_requests[i] = _as_int(token_types)

        if not token_type_id_requests:
            return orig_init_model_kwargs(self)

        seq_lens = self.seq_lens[:num_reqs]
        token_type_ids = []
        for i in range(num_reqs):
            seq_len = _as_int(seq_lens[i])
            pos = token_type_id_requests.get(i, seq_len)
            ids = (torch.arange(seq_len, device=self.device) >= pos).int()
            token_type_ids.append(ids)

        return {
            "token_type_ids": torch.concat(token_type_ids).to(device=self.device),
        }

    return _init_model_kwargs


def apply_pooling_token_type_ids_patch() -> bool:
    """Patch ``GPUModelRunner._init_model_kwargs`` once per process."""
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except ImportError:
        logger.warning("vLLM GPUModelRunner not found; token_type_ids patch skipped")
        return False

    if getattr(GPUModelRunner, _PATCH_ATTR, False):
        return True

    GPUModelRunner._init_model_kwargs = _make_patched_init_model_kwargs(
        GPUModelRunner._init_model_kwargs
    )
    setattr(GPUModelRunner, _PATCH_ATTR, True)
    logger.info("Patched GPUModelRunner._init_model_kwargs for pooling token_type_ids")
    return True
