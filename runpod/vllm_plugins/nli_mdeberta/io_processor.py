"""IO processor for the latence-trace NLI plugin.

Entry-point group: ``vllm.io_processor_plugins``
Entry-point name:  ``nli_mdeberta``

Wire format (online ``POST /pooling`` and offline ``llm.encode(...)``):

Single pair::

    {"task": "plugin",
     "data": {"premise": "Apple beat estimates.",
              "hypothesis": "Apple missed estimates."}}

Batched pairs (preferred — fuses one forward over the whole batch)::

    {"task": "plugin",
     "data": {"premise":   ["p1", "p2", "p3"],
              "hypothesis": ["h1", "h2", "h3"]}}

Response::

    {"data": [
        {"entail": 0.84, "neutral": 0.10, "contradict": 0.06,
         "label": "entailment"},
        ...
    ]}

The pair order is fixed at ``(entail, neutral, contradict)`` regardless of
the model's underlying ``id2label``; the IO processor resolves the index map
once at startup so the server is the single source of truth and the client
side stays trivial.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from vllm.config import VllmConfig

from vllm_factory.io.base import FactoryIOProcessor, PoolingRequestOutput, PromptType, TokensPrompt


def _resolve_label_indices(id2label: Dict[Any, Any]) -> Tuple[int, int, int]:
    """Return ``(entail_idx, neutral_idx, contradict_idx)``.

    Falls back to the conventional ``(0, 1, 2)`` ordering if any label is
    missing — this matches :class:`HuggingFaceNLIProvider._resolve_label_indices`
    in ``latence_trace/core/nli.py`` so vLLM-backed and in-process providers
    return identical triples.
    """

    if not id2label:
        return (0, 1, 2)
    normalized = {int(idx): str(label).lower() for idx, label in id2label.items()}
    entail = neutral = contradict = -1
    for idx, label in normalized.items():
        if "contradiction" in label or "contradict" in label:
            contradict = idx
            continue
        if "neutral" in label:
            neutral = idx
            continue
        if "entailment" in label or "entail" in label:
            entail = idx
    if entail < 0 or neutral < 0 or contradict < 0:
        return (0, 1, 2)
    return entail, neutral, contradict


def _request_output_index(output: PoolingRequestOutput) -> int | None:
    request_id = getattr(output, "request_id", None)
    if not isinstance(request_id, str):
        return None
    _, sep, suffix = request_id.rpartition("-")
    if not sep:
        return None
    try:
        return int(suffix)
    except ValueError:
        return None


@dataclass
class NLIInput:
    """Validated NLI request after :meth:`factory_parse`."""

    premises: List[str]
    hypotheses: List[str]


class NLIDebertaV2IOProcessor(FactoryIOProcessor):
    """IOProcessor for the latence-trace NLI plugin.

    Pipeline::

        IOProcessorRequest(data={premise, hypothesis})
          -> factory_parse        -> NLIInput
          -> factory_pre_process  -> Sequence[TokensPrompt]
          -> merge_pooling_params -> PoolingParams(task="plugin")
          -> engine.encode        -> Sequence[PoolingRequestOutput]
          -> factory_post_process -> {"data": [{entail,neutral,contradict,label}, ...]}
    """

    pooling_task = "plugin"

    def __init__(self, vllm_config: VllmConfig, *args: Any, **kwargs: Any) -> None:
        super().__init__(vllm_config, *args, **kwargs)
        model_id = vllm_config.model_config.model
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
            trust_remote_code=True,
        )

        cfg = vllm_config.model_config.hf_config
        max_model_len = getattr(vllm_config.model_config, "max_model_len", None)
        max_pos = getattr(cfg, "encoder_max_position_embeddings", None) or getattr(
            cfg,
            "max_position_embeddings",
            512,
        )
        self._max_length = int(min(item for item in (max_model_len, max_pos, 512) if item))

        id2label = getattr(cfg, "id2label", None) or {}
        self._entail_idx, self._neutral_idx, self._contradict_idx = _resolve_label_indices(id2label)
        self._labels = ("entailment", "neutral", "contradiction")

    def factory_parse(self, data: Any) -> NLIInput:
        if hasattr(data, "data"):
            data = data.data
        elif isinstance(data, dict) and "data" in data:
            data = data["data"]

        if not isinstance(data, dict):
            raise ValueError(f"Expected dict with 'premise' and 'hypothesis' keys, got {type(data)}")

        prem = data.get("premise") or data.get("premises") or data.get("context")
        hyp = data.get("hypothesis") or data.get("hypotheses") or data.get("claim")
        if prem is None or hyp is None:
            raise ValueError("Request data must contain 'premise' and 'hypothesis' (or aliases)")

        if isinstance(prem, str):
            prem = [prem]
        if isinstance(hyp, str):
            hyp = [hyp]

        if len(prem) != len(hyp):
            raise ValueError(f"premise ({len(prem)}) and hypothesis ({len(hyp)}) must align")
        if not prem:
            raise ValueError("Empty NLI batch")

        return NLIInput(premises=list(prem), hypotheses=list(hyp))

    def factory_pre_process(
        self,
        parsed_input: NLIInput,
        request_id: str | None,
    ) -> PromptType | Sequence[PromptType]:
        encoded = self._tokenizer(
            parsed_input.premises,
            parsed_input.hypotheses,
            add_special_tokens=True,
            truncation=True,
            max_length=self._max_length,
            padding=False,
            return_tensors=None,
        )

        token_type_rows = encoded.get("token_type_ids")
        split_positions: list[int] = []
        if token_type_rows is not None:
            for row in token_type_rows:
                split_positions.append(
                    next((idx for idx, value in enumerate(row) if int(value) == 1), len(row))
                )

        prompts = []
        for ids in encoded["input_ids"]:
            prompts.append(TokensPrompt(prompt_token_ids=list(ids)))

        extra_kwargs = None
        if split_positions and all(pos == split_positions[0] for pos in split_positions):
            extra_kwargs = {"compressed_token_type_ids": int(split_positions[0])}

        self._stash(
            extra_kwargs=extra_kwargs,
            request_id=request_id,
            meta={
                "n": len(prompts),
                "entail_idx": self._entail_idx,
                "neutral_idx": self._neutral_idx,
                "contradict_idx": self._contradict_idx,
            },
        )

        if len(prompts) == 1:
            return prompts[0]
        return prompts

    def factory_post_process(
        self,
        model_output: Sequence[PoolingRequestOutput],
        request_meta: Any,
    ) -> Dict[str, List[Dict[str, Any]]]:
        if not model_output:
            return {"data": []}

        meta = request_meta or {
            "entail_idx": self._entail_idx,
            "neutral_idx": self._neutral_idx,
            "contradict_idx": self._contradict_idx,
        }
        e_idx = int(meta.get("entail_idx", self._entail_idx))
        n_idx = int(meta.get("neutral_idx", self._neutral_idx))
        c_idx = int(meta.get("contradict_idx", self._contradict_idx))

        rows = []
        indexed_outputs = [(_request_output_index(output), output) for output in model_output]
        if indexed_outputs and all(idx is not None for idx, _ in indexed_outputs):
            ordered_outputs = [output for _, output in sorted(indexed_outputs, key=lambda item: item[0])]
        else:
            ordered_outputs = list(model_output)

        for output in ordered_outputs:
            raw = output.outputs.data
            if raw is None:
                rows.append(
                    {
                        "entail": 0.0,
                        "neutral": 1.0,
                        "contradict": 0.0,
                        "label": "neutral",
                    }
                )
                continue

            logits = raw if isinstance(raw, torch.Tensor) else torch.as_tensor(raw)
            if logits.dim() == 0 or logits.numel() < 3:
                rows.append(
                    {
                        "entail": 0.0,
                        "neutral": 1.0,
                        "contradict": 0.0,
                        "label": "neutral",
                    }
                )
                continue

            logits = logits.float().reshape(-1)[:3]
            probs = F.softmax(logits, dim=-1).tolist()
            entail = float(probs[e_idx])
            neutral = float(probs[n_idx])
            contradict = float(probs[c_idx])
            ordered = (entail, neutral, contradict)
            label = self._labels[max(range(3), key=lambda i: ordered[i])]
            rows.append(
                {
                    "entail": entail,
                    "neutral": neutral,
                    "contradict": contradict,
                    "label": label,
                }
            )

        return {"data": rows}


def get_processor_cls() -> str:
    return "latence_trace_nli_plugin.io_processor.NLIDebertaV2IOProcessor"
