"""Pooler for the MiniCheck-Flan-T5 BYOP plugin.

Mirrors the contract of
:class:`latence_trace_nli_plugin.pooler.NLIClassifyPooler` so the
:class:`vllm_factory.pooling.vllm_adapter.VllmPoolerAdapter` glue stays
identical for both plugins. The model writes the per-row
``(yes_logit, no_logit)`` pair into position 0 of each sequence's slice
of the packed ``(total_tokens, 2)`` tensor; this pooler simply slices
``h[0]`` and lets the IO processor finish the wire mapping
(``entail = softmax([yes, no])[0]``, ``contradict = 1 - entail``,
``neutral = 0``).
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from vllm_factory.pooling.protocol import PoolerContext, split_hidden_states


class MiniCheckPooler(nn.Module):
    """Slice the position-0 Yes/No logits and softmax them per sequence.

    Why softmax inside the pooler instead of inside the IO processor?
    Two reasons:

    1. We keep the IO processor wire-shape compatible with the existing
       ``nli_mdeberta`` plugin — both return a 3-vector
       ``(entail, neutral, contradict)`` per pair after the pooler runs.
       Doing the softmax here means the IO processor's
       :meth:`factory_post_process` is the same byte-for-byte function
       in both plugins, modulo the contradict/neutral mapping.
    2. Softmax over only two logits is numerically stable in fp32 and
       gives us the calibrated grounding probability the calibration
       sweep expects (Tang et al. 2024 §4.2 reports macro F1 against
       this exact two-class softmax). Softmaxing over the full Flan-T5
       vocabulary would give wildly different numbers and break the
       existing threshold bundles.
    """

    def get_tasks(self) -> set[str]:
        # ``plugin`` is the canonical task name FactoryIOProcessor merges
        # into PoolingParams; ``classify`` and ``embed`` keep us
        # compatible with operators that still send legacy task strings.
        return {"plugin", "classify", "embed"}

    def forward(
        self,
        hidden_states: torch.Tensor,
        ctx: PoolerContext,
    ) -> List[Optional[torch.Tensor]]:
        per_seq = split_hidden_states(hidden_states, ctx.seq_lengths)
        if not per_seq:
            return []

        # The model packs (yes_logit, no_logit) into position 0 of every
        # sequence. We never look at the rest of the slice (it's filled
        # with zeros by :meth:`MiniCheckT5Model.forward`).
        yes_no = torch.stack([h[0].to(torch.float32) for h in per_seq], dim=0)
        probs = torch.softmax(yes_no, dim=-1)
        return [probs[i] for i in range(probs.shape[0])]
