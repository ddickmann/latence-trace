"""IO processor for the MiniCheck-Flan-T5 BYOP plugin.

Entry-point group: ``vllm.io_processor_plugins``
Entry-point name:  ``minicheck_t5_io``

Wire format (online ``POST /pooling`` and offline ``llm.encode(...)``):

Single pair::

    {"task": "plugin",
     "data": {"premise": "Apple beat estimates.",
              "hypothesis": "Apple missed estimates."}}

Batched pairs::

    {"task": "plugin",
     "data": {"premise":   ["p1", "p2", "p3"],
              "hypothesis": ["h1", "h2", "h3"]}}

Response (identical shape to ``nli_mdeberta`` so the
:mod:`latence_trace.providers.nli_pooling` HTTP client doesn't have to
branch)::

    {"data": [
        {"entail": 0.84, "neutral": 0.0, "contradict": 0.16,
         "label": "entailment"},
        ...
    ]}

MiniCheck is binary (supported / not-supported) so the ``neutral`` slot
is hard-zero on the wire. This matches what the in-process
:class:`latence_trace.providers.nli_transformers.MiniCheckNLIProvider`
returns, so the calibration bundles trained on the in-process backend
transfer directly to the vLLM-served one.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from transformers import AutoTokenizer
from vllm.config import VllmConfig

from vllm_factory.io.base import (
    FactoryIOProcessor,
    PoolingRequestOutput,
    PromptType,
    TokensPrompt,
)

logger = logging.getLogger(__name__)


# MiniCheck context budget. Mirrors the in-process provider so the wire
# behaviour matches bit-for-bit: claim is capped first, premise consumes
# whatever budget is left up to ``max_input_tokens``.
_MINICHECK_INPUT_BUDGET = 1024
_MINICHECK_MAX_CLAIM_TOKENS = 384


@dataclass
class MiniCheckInput:
    """Validated NLI request after :meth:`factory_parse`."""

    premises: List[str]
    hypotheses: List[str]


def _request_output_index(output: PoolingRequestOutput) -> int | None:
    """Return the per-request integer suffix vLLM stamps onto request_id.

    Same trick the ``nli_mdeberta`` plugin uses to defend against vLLM's
    pooling runner returning batched outputs out of order; we sort by
    suffix before reading hopen so the post-process result lines up with
    the IO request order.
    """
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


class MiniCheckT5IOProcessor(FactoryIOProcessor):
    """IOProcessor for the MiniCheck-Flan-T5 BYOP plugin.

    Pipeline::

        IOProcessorRequest(data={premise, hypothesis})
          -> factory_parse        -> MiniCheckInput
          -> factory_pre_process  -> Sequence[TokensPrompt]
          -> merge_pooling_params -> PoolingParams(task="plugin")
          -> engine.encode        -> Sequence[PoolingRequestOutput]
          -> factory_post_process -> {"data": [{entail,neutral,contradict,label}, ...]}

    The pre-process step is where we apply the MiniCheck prompt template
    (``predict: <premise>\\nclaim: <claim>``) and stash the resolved
    Yes/No token IDs onto the live config object so the model's
    :meth:`forward` can slice the LM head without re-tokenising.
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
        # ``MiniCheck`` was trained at 2048 tokens; we default to 1024 to
        # match the in-process provider (Phase 0 proof showed no quality
        # delta at 1024 vs 2048 on LLM-AggreFact). Operators can lift
        # the cap by setting ``--max-model-len 2048`` on the vLLM CLI.
        budget_candidates = [item for item in (max_model_len, _MINICHECK_INPUT_BUDGET) if item]
        self._max_length = int(min(budget_candidates))
        self._max_claim_tokens = int(_MINICHECK_MAX_CLAIM_TOKENS)

        # Resolve Yes/No vocabulary token IDs once at startup. T5 ``Yes``
        # /``No`` tokenise to a single sub-token each in the Flan-T5
        # SentencePiece vocab.
        yes_ids = self._tokenizer("Yes", add_special_tokens=False).input_ids
        no_ids = self._tokenizer("No", add_special_tokens=False).input_ids
        if not yes_ids or not no_ids:
            raise RuntimeError(
                f"Tokenizer for {model_id!r} could not resolve Yes/No tokens"
            )
        self._yes_token_id = int(yes_ids[0])
        self._no_token_id = int(no_ids[0])

        # Stash on the live HF config so MiniCheckT5Model.forward can
        # slice the LM head without re-tokenising. We avoid clobbering
        # explicit values that may already be on the checkpoint config.
        if getattr(cfg, "yes_token_id", None) is None:
            try:
                cfg.yes_token_id = self._yes_token_id
            except Exception:  # pragma: no cover - immutable config
                logger.debug(
                    "[MiniCheckT5IO] Could not stash yes_token_id on hf_config"
                )
        if getattr(cfg, "no_token_id", None) is None:
            try:
                cfg.no_token_id = self._no_token_id
            except Exception:  # pragma: no cover - immutable config
                logger.debug(
                    "[MiniCheckT5IO] Could not stash no_token_id on hf_config"
                )

    # ── factory_* hooks ──────────────────────────────────────────────

    def factory_parse(self, data: Any) -> MiniCheckInput:
        if hasattr(data, "data"):
            data = data.data
        elif isinstance(data, dict) and "data" in data:
            data = data["data"]

        if not isinstance(data, dict):
            raise ValueError(
                f"Expected dict with 'premise' and 'hypothesis' keys, got {type(data)}"
            )

        prem = data.get("premise") or data.get("premises") or data.get("context")
        hyp = data.get("hypothesis") or data.get("hypotheses") or data.get("claim")
        if prem is None or hyp is None:
            raise ValueError(
                "Request data must contain 'premise' and 'hypothesis' (or aliases)"
            )

        if isinstance(prem, str):
            prem = [prem]
        if isinstance(hyp, str):
            hyp = [hyp]
        if len(prem) != len(hyp):
            raise ValueError(
                f"premise ({len(prem)}) and hypothesis ({len(hyp)}) must align"
            )
        if not prem:
            raise ValueError("Empty NLI batch")

        return MiniCheckInput(premises=list(prem), hypotheses=list(hyp))

    def _format_one(self, premise: str, claim: str) -> str:
        """MiniCheck prompt template — claim-first, premise-second cap.

        Mirrors :meth:`MiniCheckNLIProvider._format_one` so HTTP-served
        and in-process backends produce bit-identical formatted strings
        for the same ``(premise, claim)`` pair. Capping the claim avoids
        the pathological case where someone passes a long ``raw_context``
        as the "claim" and starves the premise budget.
        """

        claim_ids = self._tokenizer.encode(claim, add_special_tokens=False)
        if len(claim_ids) > self._max_claim_tokens:
            claim = self._tokenizer.decode(
                claim_ids[: self._max_claim_tokens],
                skip_special_tokens=True,
            )
        return f"predict: {premise}\nclaim: {claim}"

    def factory_pre_process(
        self,
        parsed_input: MiniCheckInput,
        request_id: str | None,
    ) -> PromptType | Sequence[PromptType]:
        formatted = [
            self._format_one(p or "", h or "")
            for p, h in zip(parsed_input.premises, parsed_input.hypotheses)
        ]
        encoded = self._tokenizer(
            formatted,
            add_special_tokens=True,
            truncation=True,
            max_length=self._max_length,
            padding=False,
            return_tensors=None,
        )
        prompts = [TokensPrompt(prompt_token_ids=list(ids)) for ids in encoded["input_ids"]]

        self._stash(
            extra_kwargs=None,
            request_id=request_id,
            meta={"n": len(prompts)},
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

        # Restore IO order — vLLM's pooling runner can finish prompts
        # out of order and rely on the request_id suffix for sequencing.
        indexed_outputs = [(_request_output_index(o), o) for o in model_output]
        if indexed_outputs and all(idx is not None for idx, _ in indexed_outputs):
            ordered_outputs = [o for _, o in sorted(indexed_outputs, key=lambda item: item[0])]
        else:
            ordered_outputs = list(model_output)

        rows: List[Dict[str, Any]] = []
        for output in ordered_outputs:
            raw = output.outputs.data
            if raw is None:
                rows.append(
                    {
                        "entail": 0.0,
                        "neutral": 0.0,
                        "contradict": 1.0,
                        "label": "contradiction",
                    }
                )
                continue

            probs = raw if isinstance(raw, torch.Tensor) else torch.as_tensor(raw)
            if probs.numel() < 2:
                rows.append(
                    {
                        "entail": 0.0,
                        "neutral": 0.0,
                        "contradict": 1.0,
                        "label": "contradiction",
                    }
                )
                continue

            probs = probs.float().reshape(-1)[:2]
            entail = float(probs[0])
            contradict = float(max(0.0, 1.0 - entail))
            label = "entailment" if entail >= 0.5 else "contradiction"
            rows.append(
                {
                    "entail": entail,
                    "neutral": 0.0,
                    "contradict": contradict,
                    "label": label,
                }
            )

        return {"data": rows}


def get_processor_cls() -> str:
    """Entry-point hook for vLLM to import the IO processor class."""
    return "minicheck_t5.io_processor.MiniCheckT5IOProcessor"
