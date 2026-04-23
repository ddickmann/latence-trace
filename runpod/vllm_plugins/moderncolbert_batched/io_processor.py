"""IO processor for batched ModernColBERT / LateOn pooling requests."""

from __future__ import annotations

import base64
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer
from vllm.config import VllmConfig

from vllm_factory.io.base import FactoryIOProcessor, PoolingRequestOutput, PromptType, TokensPrompt


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
class ModernColBERTBatchedInput:
    texts: list[str]
    is_query: list[bool]
    batched: bool


class ModernColBERTBatchedIOProcessor(FactoryIOProcessor):
    """Batched superset of upstream ``moderncolbert_io``."""

    pooling_task = "token_embed"
    query_prefix_id = 50368
    document_prefix_id = 50369

    def __init__(self, vllm_config: VllmConfig, *args: Any, **kwargs: Any) -> None:
        super().__init__(vllm_config, *args, **kwargs)
        model_id = vllm_config.model_config.model
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
            trust_remote_code=True,
        )
        try:
            cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        except Exception:
            cfg = vllm_config.model_config.hf_config
        self._query_max_length = int(
            getattr(cfg, "query_length", getattr(cfg, "query_maxlen", 256))
        )
        self._document_max_length = int(
            getattr(
                cfg,
                "document_length",
                getattr(cfg, "document_maxlen", getattr(cfg, "max_position_embeddings", 8192)),
            )
        )

    def _normalize_is_query(self, value: Any, n: int) -> list[bool]:
        if isinstance(value, bool):
            return [value] * n
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            flags = [bool(item) for item in value]
            if len(flags) != n:
                raise ValueError(f"'is_query' batch length ({len(flags)}) must match text batch ({n})")
            return flags
        raise ValueError("'is_query' must be a boolean or a list of booleans")

    def factory_parse(self, data: Any) -> ModernColBERTBatchedInput:
        if hasattr(data, "data"):
            data = data.data
        elif isinstance(data, dict) and "data" in data:
            data = data["data"]

        if not isinstance(data, dict):
            raise ValueError("Expected request data dict")

        text = data.get("text")
        if isinstance(text, str):
            texts = [text]
            batched = False
        elif isinstance(text, Sequence) and not isinstance(text, (bytes, str)):
            texts = [str(item) for item in text]
            if not texts:
                raise ValueError("Empty ModernColBERT batch")
            batched = True
        else:
            raise ValueError("'text' must be a string or a list of strings")

        flags = self._normalize_is_query(data.get("is_query", False), len(texts))
        if len(set(flags)) > 1:
            raise ValueError(
                "Mixed query/document batches are not supported; send homogeneous batches "
                "so ModernColBERT query/document semantics stay identical to the reference path."
            )
        return ModernColBERTBatchedInput(texts=texts, is_query=flags, batched=batched)

    def _prepare_prompt(self, text: str, *, is_query: bool) -> tuple[list[int], list[int]]:
        max_length = self._query_max_length if is_query else self._document_max_length
        encoded = self._tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=max(1, int(max_length) - 1),
            padding=False,
            return_tensors=None,
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        if input_ids and isinstance(input_ids[0], list):
            input_ids = input_ids[0]
        if attention_mask and isinstance(attention_mask[0], list):
            attention_mask = attention_mask[0]
        input_ids = list(input_ids)
        attention_mask = list(attention_mask)
        if not input_ids:
            return [], []
        prefix_id = self.query_prefix_id if is_query else self.document_prefix_id
        return (
            [int(input_ids[0]), int(prefix_id), *[int(item) for item in input_ids[1:]]],
            [int(attention_mask[0]), 1, *[int(item) for item in attention_mask[1:]]],
        )

    def factory_pre_process(
        self,
        parsed_input: ModernColBERTBatchedInput,
        request_id: str | None,
    ) -> PromptType | Sequence[PromptType]:
        prepared = [
            self._prepare_prompt(text, is_query=is_query)
            for text, is_query in zip(parsed_input.texts, parsed_input.is_query)
        ]
        prompts = [TokensPrompt(prompt_token_ids=input_ids) for input_ids, _ in prepared]
        seq_lengths = [len(input_ids) for input_ids, _ in prepared]
        input_ids_payload = [input_ids for input_ids, _ in prepared]
        attention_masks = [mask for _, mask in prepared]
        extra_kwargs: dict[str, Any] = {
            "is_query": parsed_input.is_query[0],
            "sequence_length": seq_lengths[0] if len(seq_lengths) == 1 else seq_lengths,
            "attention_mask": attention_masks[0] if len(attention_masks) == 1 else attention_masks,
            "input_ids": input_ids_payload[0] if len(input_ids_payload) == 1 else input_ids_payload,
        }
        self._stash(
            extra_kwargs=extra_kwargs,
            request_id=request_id,
            meta={
                "batched": parsed_input.batched,
                "n": len(prompts),
                "is_query": parsed_input.is_query[0],
            },
        )
        if len(prompts) == 1:
            return prompts[0]
        return prompts

    @staticmethod
    def _encode_output(raw: Any) -> str:
        tensor = raw if isinstance(raw, torch.Tensor) else torch.as_tensor(raw)
        array = tensor.detach().float().cpu().numpy().reshape(-1)
        return base64.b64encode(array.astype(np.float32, copy=False).tobytes()).decode("ascii")

    def factory_post_process(
        self,
        model_output: Sequence[PoolingRequestOutput],
        request_meta: Any,
    ) -> dict[str, Any]:
        if not model_output:
            return {"data": []}

        indexed_outputs = [(_request_output_index(output), output) for output in model_output]
        if indexed_outputs and all(idx is not None for idx, _ in indexed_outputs):
            ordered_outputs = [output for _, output in sorted(indexed_outputs, key=lambda item: item[0])]
        else:
            ordered_outputs = list(model_output)

        rows = [self._encode_output(output.outputs.data) for output in ordered_outputs]
        meta = request_meta or {}
        if not meta.get("batched", False) and len(rows) == 1:
            return {"data": rows[0]}
        return {"data": rows}


def get_processor_cls() -> str:
    return "moderncolbert_batched.io_processor.ModernColBERTBatchedIOProcessor"
