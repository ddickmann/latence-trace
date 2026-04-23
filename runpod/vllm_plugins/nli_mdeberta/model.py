"""NLI mDeBERTa-v3 — DeBERTa v2/v3 encoder + 3-way sequence-classification head.

Backbone: ``models.deberta_v2.DebertaV2EncoderModel`` (Flash-DeBERTa Triton kernel
+ vLLM parallel layers, supplied by ``vllm-factory``).
Head: HuggingFace ``DebertaV2ForSequenceClassification`` layout — ContextPooler
(dense + activation) followed by ``Linear(num_labels=3)``.

Why pull the encoder via ``importlib.util`` first, with a clean fallback?
``vllm-factory`` ships ``models/`` as a top-level package on disk but it is not
guaranteed to be importable at every install (different setuptools layouts,
editable installs, namespace packages). The same trick used by the official
``deberta_gliner`` plugin (``Path(...).parents[2] / "models" / "deberta_v2"``)
will not work from this vendored location because we live outside the
``vllm-factory`` repo. We therefore look up the ``models`` directory relative
to ``vllm_factory.__file__`` at runtime and only fall back to a regular import
if that fails.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn
from transformers import DebertaV2Config
from vllm.config import VllmConfig

try:
    from vllm.model_executor.models.interfaces import SupportsLoRA
except Exception:  # pragma: no cover - compatibility shim
    class SupportsLoRA:  # type: ignore[no-redef]
        pass

from vllm_factory.pooling.vllm_adapter import VllmPoolerAdapter

from .config import NLIDebertaV2Config
from .pooler import NLIClassifyPooler

logger = logging.getLogger(__name__)


def _import_deberta_v2_encoder():
    try:
        import vllm_factory

        candidate = (
            Path(vllm_factory.__file__).resolve().parent.parent
            / "models"
            / "deberta_v2"
            / "deberta_v2_encoder.py"
        )
        if candidate.exists():
            spec = importlib.util.spec_from_file_location(
                "latence_trace_nli_deberta_v2_encoder",
                str(candidate),
            )
            if spec is not None and spec.loader is not None:
                module = importlib.util.module_from_spec(spec)
                sys.modules.setdefault("latence_trace_nli_deberta_v2_encoder", module)
                spec.loader.exec_module(module)
                return module
    except Exception:
        pass

    from models.deberta_v2 import deberta_v2_encoder as module

    return module


_encoder_mod = _import_deberta_v2_encoder()
DebertaV2EncoderModel = _encoder_mod.DebertaV2EncoderModel
_ENCODER_PACKED_MODULES_MAPPING = getattr(_encoder_mod, "PACKED_MODULES_MAPPING", {})
_ENCODER_EMBEDDING_MODULES = getattr(_encoder_mod, "EMBEDDING_MODULES", {})


def _coerce_nli_config(hf_config: object) -> NLIDebertaV2Config:
    """Normalize either the custom plugin config or a stock HF DeBERTa config.

    When the plugin is selected through the architecture alias
    ``DebertaV2ForSequenceClassification``, vLLM hands us the checkpoint's native
    ``transformers.DebertaV2Config``. The BYOP model still needs the expanded
    ``encoder_*`` fields, so derive them from the stock config on the fly.
    """

    if isinstance(hf_config, NLIDebertaV2Config):
        return hf_config
    if hasattr(hf_config, "encoder_hidden_size"):
        return hf_config  # type: ignore[return-value]

    hidden_size = int(getattr(hf_config, "hidden_size", 768))
    num_attention_heads = int(getattr(hf_config, "num_attention_heads", 1))
    pooler_hidden_size = getattr(hf_config, "pooler_hidden_size", None)
    pooler_hidden_act = getattr(hf_config, "pooler_hidden_act", None) or "gelu"
    pooler_dropout = getattr(hf_config, "classifier_dropout", None)
    if pooler_dropout is None:
        pooler_dropout = getattr(hf_config, "hidden_dropout_prob", 0.0)

    return NLIDebertaV2Config(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        vocab_size=int(getattr(hf_config, "vocab_size", 251000)),
        encoder_hidden_size=hidden_size,
        encoder_num_hidden_layers=int(getattr(hf_config, "num_hidden_layers", 12)),
        encoder_num_attention_heads=num_attention_heads,
        encoder_intermediate_size=int(
            getattr(hf_config, "intermediate_size", hidden_size * 4)
        ),
        encoder_hidden_act=getattr(hf_config, "hidden_act", "gelu"),
        encoder_hidden_dropout_prob=float(getattr(hf_config, "hidden_dropout_prob", 0.0)),
        encoder_attention_probs_dropout_prob=float(
            getattr(hf_config, "attention_probs_dropout_prob", 0.0)
        ),
        encoder_max_position_embeddings=int(
            getattr(hf_config, "max_position_embeddings", 512)
        ),
        encoder_type_vocab_size=int(getattr(hf_config, "type_vocab_size", 0)),
        encoder_layer_norm_eps=float(getattr(hf_config, "layer_norm_eps", 1e-7)),
        encoder_relative_attention=bool(getattr(hf_config, "relative_attention", True)),
        encoder_max_relative_positions=int(
            getattr(hf_config, "max_relative_positions", -1)
        ),
        encoder_position_buckets=int(getattr(hf_config, "position_buckets", 256)),
        encoder_pos_att_type=getattr(hf_config, "pos_att_type", None),
        encoder_share_att_key=bool(getattr(hf_config, "share_att_key", True)),
        encoder_norm_rel_ebd=getattr(hf_config, "norm_rel_ebd", "layer_norm"),
        encoder_position_biased_input=bool(
            getattr(hf_config, "position_biased_input", False)
        ),
        encoder_pad_token_id=int(getattr(hf_config, "pad_token_id", 0)),
        num_labels=int(getattr(hf_config, "num_labels", 3)),
        pooler_hidden_size=pooler_hidden_size,
        pooler_hidden_act=pooler_hidden_act,
        pooler_dropout=float(pooler_dropout),
        id2label=getattr(hf_config, "id2label", None),
        label2id=getattr(hf_config, "label2id", None),
    )


def _split_packed_batch(
    input_ids: torch.Tensor,
    positions: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
    *,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, list[int]]:
    """Convert packed 1D vLLM pooling inputs into a padded 2D batch.

    vLLM's pooling runner may concatenate multiple prompt rows into one flat
    ``input_ids`` tensor and supply ``positions`` with per-row resets
    (``0, 1, 2, ..., 0, 1, 2, ...``). DeBERTa must see independent rows, so we
    reconstruct a padded batch before calling the encoder and later flatten the
    valid tokens back to the factory pooler contract.
    """

    if input_ids.dim() != 1 or positions is None or positions.dim() != 1:
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if positions is not None and positions.dim() == 1:
            positions = positions.unsqueeze(0)
        if attention_mask is not None and attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)
        seq_len = int(input_ids.shape[1]) if input_ids.dim() == 2 else int(input_ids.shape[0])
        return input_ids, positions, attention_mask, [seq_len]

    reset_points = torch.nonzero(positions == 0, as_tuple=False).flatten().tolist()
    if not reset_points or reset_points[0] != 0:
        reset_points.insert(0, 0)
    if len(reset_points) == 1:
        return (
            input_ids.unsqueeze(0),
            positions.unsqueeze(0),
            attention_mask.unsqueeze(0) if attention_mask is not None else None,
            [int(input_ids.shape[0])],
        )

    bounds = reset_points + [int(input_ids.shape[0])]
    seq_lengths = [end - start for start, end in zip(bounds, bounds[1:])]
    batch_size = len(seq_lengths)
    max_len = max(seq_lengths)
    device = input_ids.device

    batched_ids = torch.full(
        (batch_size, max_len),
        pad_token_id,
        dtype=input_ids.dtype,
        device=device,
    )
    batched_positions = torch.zeros(
        (batch_size, max_len),
        dtype=positions.dtype,
        device=positions.device,
    )
    batched_attention = torch.zeros(
        (batch_size, max_len),
        dtype=torch.long,
        device=device,
    )

    for row_idx, (start, end) in enumerate(zip(bounds, bounds[1:])):
        seq_ids = input_ids[start:end]
        seq_pos = positions[start:end]
        length = end - start
        batched_ids[row_idx, :length] = seq_ids
        batched_positions[row_idx, :length] = seq_pos
        if attention_mask is not None:
            batched_attention[row_idx, :length] = attention_mask[start:end].to(dtype=torch.long)
        else:
            batched_attention[row_idx, :length] = 1

    return batched_ids, batched_positions, batched_attention, seq_lengths


class NLIDebertaV2Model(nn.Module, SupportsLoRA):
    """Custom vLLM-optimized DeBERTa v2/v3 encoder + NLI classification head.

    Architecture::

        input_ids -> DebertaV2EncoderModel (Flash-DeBERTa Triton + vLLM parallel layers)
                  -> hidden_states (total_tokens, hidden)
                  -> NLIClassifyPooler ([CLS] -> dense -> act -> classifier)
                  -> (num_labels=3,) raw logits per sequence
    """

    is_pooling_model = True
    packed_modules_mapping = {
        f"model.{k}": [f"model.{n}" for n in v]
        for k, v in _ENCODER_PACKED_MODULES_MAPPING.items()
    }
    embedding_modules = {
        f"model.{k}": v for k, v in _ENCODER_EMBEDDING_MODULES.items()
    }

    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        cfg = _coerce_nli_config(vllm_config.model_config.hf_config)
        self.config = cfg
        self.vllm_config = vllm_config

        self.encoder_hidden_size = cfg.encoder_hidden_size
        self.pooler_hidden_size = cfg.pooler_hidden_size
        self.num_labels = cfg.num_labels

        encoder_cfg = DebertaV2Config(
            vocab_size=cfg.vocab_size,
            hidden_size=cfg.encoder_hidden_size,
            num_hidden_layers=cfg.encoder_num_hidden_layers,
            num_attention_heads=cfg.encoder_num_attention_heads,
            intermediate_size=cfg.encoder_intermediate_size,
            hidden_act=cfg.encoder_hidden_act,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            max_position_embeddings=cfg.encoder_max_position_embeddings,
            type_vocab_size=cfg.encoder_type_vocab_size,
            layer_norm_eps=cfg.encoder_layer_norm_eps,
            relative_attention=cfg.encoder_relative_attention,
            max_relative_positions=cfg.encoder_max_relative_positions,
            position_buckets=cfg.encoder_position_buckets,
            pos_att_type=cfg.encoder_pos_att_type,
            share_att_key=cfg.encoder_share_att_key,
            norm_rel_ebd=cfg.encoder_norm_rel_ebd,
            position_biased_input=cfg.encoder_position_biased_input,
            pad_token_id=cfg.encoder_pad_token_id,
        )

        self.model = DebertaV2EncoderModel(config=encoder_cfg)
        self.pooler_dense = nn.Linear(self.encoder_hidden_size, self.pooler_hidden_size)
        self.classifier = nn.Linear(self.pooler_hidden_size, self.num_labels)
        self._business_pooler = NLIClassifyPooler(
            self.pooler_dense,
            self.classifier,
            activation=cfg.pooler_hidden_act,
        )
        self.pooler = VllmPoolerAdapter(self._business_pooler, requires_token_ids=False)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Required by vLLM's pooling runner for embedding lookup."""
        return self.model.embeddings.word_embeddings(input_ids)

    def sample(self, logits: torch.Tensor, sampling_metadata):
        """Pooling-only model — return an empty sampler output."""
        try:
            from vllm.sequence import SamplerOutput

            return SamplerOutput(outputs=[])
        except ImportError:
            return None

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        positions: Optional[torch.Tensor] = None,
        intermediate_tensors=None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Run the encoder; return 2D ``(total_tokens, hidden)`` for the pooler."""

        attention_mask = kwargs.pop("attention_mask", None)
        if input_ids is not None:
            input_ids, positions, attention_mask, seq_lengths = _split_packed_batch(
                input_ids,
                positions,
                attention_mask if isinstance(attention_mask, torch.Tensor) else None,
                pad_token_id=int(getattr(self.config, "encoder_pad_token_id", 0)),
            )
        else:
            seq_lengths = []

        with torch.no_grad():
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=positions,
                inputs_embeds=inputs_embeds,
                **kwargs,
            )

        hs = (
            output.last_hidden_state
            if hasattr(output, "last_hidden_state")
            else output
            if isinstance(output, torch.Tensor)
            else output[0]
        )

        if hs.dim() == 3:
            if not seq_lengths:
                hs = hs.view(-1, hs.shape[-1])
            else:
                hs = torch.cat(
                    [hs[row_idx, :length] for row_idx, length in enumerate(seq_lengths)],
                    dim=0,
                )

        return hs

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Map HF ``DebertaV2ForSequenceClassification`` weights onto the
        backbone + pooler + classifier.

        Prefix mapping:

        * ``deberta.*``   -> backbone (passed to encoder ``load_weights``)
        * ``pooler.dense.*`` -> ``self.pooler_dense.{weight,bias}``
        * ``classifier.*``   -> ``self.classifier.{weight,bias}``

        The encoder strips its own ``deberta.`` prefix and remaps
        ``attention.self.*`` -> ``attention.self_attn.*`` internally.
        """

        backbone_weights = []
        head_state = {}

        for hf_name, tensor in weights:
            if hf_name.startswith("deberta."):
                backbone_weights.append((hf_name, tensor))
                continue
            if hf_name.startswith("pooler.dense."):
                stripped = hf_name[len("pooler.dense.") :]
                head_state[f"pooler_dense.{stripped}"] = tensor
                continue
            if hf_name.startswith("classifier."):
                stripped = hf_name[len("classifier.") :]
                head_state[f"classifier.{stripped}"] = tensor

        self.model.load_weights(backbone_weights)
        logger.info("[NLIDebertaV2] Loaded backbone: %d weight tensors", len(backbone_weights))

        if head_state:
            missing, unexpected = self.load_state_dict(head_state, strict=False)
            device = next(self.model.parameters()).device
            dtype = self.vllm_config.model_config.dtype
            self.pooler_dense.to(device=device, dtype=dtype)
            self.classifier.to(device=device, dtype=dtype)
            logger.info(
                "[NLIDebertaV2] Loaded head: %d tensors (missing=%d unexpected=%d)",
                len(head_state),
                len(missing),
                len(unexpected),
            )
        else:
            logger.warning("[NLIDebertaV2] No classification-head weights found in checkpoint!")

        return set(name for name, _ in self.named_parameters())
