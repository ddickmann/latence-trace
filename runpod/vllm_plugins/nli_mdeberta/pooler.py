"""Pooler for the latence-trace NLI plugin."""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
from transformers.activations import ACT2FN

from vllm_factory.pooling.protocol import PoolerContext, split_hidden_states

_ACT_FN = {name.lower(): fn for name, fn in ACT2FN.items()}


class NLIClassifyPooler(nn.Module):
    """ContextPooler + 3-way classifier for entailment / neutral / contradiction.

    Weights are owned by the *outer* model so that ``load_weights`` can map raw
    HuggingFace ``pooler.dense.*`` and ``classifier.*`` tensors directly onto
    them. We accept the two ``nn.Linear`` layers by reference at construction
    time and only run the forward.
    """

    def __init__(
        self,
        pooler_dense: nn.Linear,
        classifier: nn.Linear,
        *,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.pooler_dense = pooler_dense
        self.classifier = classifier
        act = _ACT_FN.get(activation.lower())
        if act is None:
            raise ValueError(f"Unsupported pooler_hidden_act={activation!r}")
        if isinstance(act, type):
            act = act()
        self._activation = act

    def get_tasks(self) -> set[str]:
        return {"plugin", "classify", "embed"}

    def forward(
        self,
        hidden_states: torch.Tensor,
        ctx: PoolerContext,
    ) -> List[Optional[torch.Tensor]]:
        per_seq = split_hidden_states(hidden_states, ctx.seq_lengths)
        if not per_seq:
            return []

        cls = torch.stack([h[0] for h in per_seq], dim=0)
        pooled = self.pooler_dense(cls)
        pooled = self._activation(pooled)
        logits = self.classifier(pooled)
        return [logits[i] for i in range(logits.shape[0])]
