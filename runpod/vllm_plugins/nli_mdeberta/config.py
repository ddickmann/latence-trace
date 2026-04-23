"""NLI mDeBERTa-v3 Configuration — encoder-only pooling model.

Uses ``PretrainedConfig`` with ``num_hidden_layers=0`` so vLLM skips KV-cache
allocation. The actual DeBERTa v2/v3 encoder config is reconstructed from the
``encoder_*`` attributes inside ``model.py``.

The classification head matches the upstream HuggingFace
``DebertaV2ForSequenceClassification`` layout:

    [CLS] -> ContextPooler.dense(hidden -> pooler_hidden_size)
          -> pooler_hidden_act (default "gelu")
          -> classifier(pooler_hidden_size -> num_labels=3)

so that loading raw HuggingFace weights via ``load_weights`` is exact.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from transformers import PretrainedConfig


class NLIDebertaV2Config(PretrainedConfig):
    """Config for the latence-trace NLI plugin (DeBERTa-v2/v3 backbone)."""

    model_type = "nli_mdeberta"

    def __init__(
        self,
        num_hidden_layers: int = 0,
        num_attention_heads: int = 1,
        hidden_size: int = 768,
        vocab_size: int = 251000,
        encoder_hidden_size: int = 768,
        encoder_num_hidden_layers: int = 12,
        encoder_num_attention_heads: int = 12,
        encoder_intermediate_size: int = 3072,
        encoder_hidden_act: str = "gelu",
        encoder_hidden_dropout_prob: float = 0.0,
        encoder_attention_probs_dropout_prob: float = 0.0,
        encoder_max_position_embeddings: int = 512,
        encoder_type_vocab_size: int = 0,
        encoder_layer_norm_eps: float = 1e-7,
        encoder_relative_attention: bool = True,
        encoder_max_relative_positions: int = -1,
        encoder_position_buckets: int = 256,
        encoder_pos_att_type: Optional[List[str]] = None,
        encoder_share_att_key: bool = True,
        encoder_norm_rel_ebd: str = "layer_norm",
        encoder_position_biased_input: bool = False,
        encoder_pad_token_id: int = 0,
        num_labels: int = 3,
        pooler_hidden_size: Optional[int] = None,
        pooler_hidden_act: str = "gelu",
        pooler_dropout: float = 0.0,
        id2label: Optional[Dict[str, str]] = None,
        label2id: Optional[Dict[str, int]] = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.num_hidden_layers = 0
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size

        self.vocab_size = vocab_size
        self.encoder_hidden_size = encoder_hidden_size
        self.encoder_num_hidden_layers = encoder_num_hidden_layers
        self.encoder_num_attention_heads = encoder_num_attention_heads
        self.encoder_intermediate_size = encoder_intermediate_size
        self.encoder_hidden_act = encoder_hidden_act
        self.encoder_hidden_dropout_prob = encoder_hidden_dropout_prob
        self.encoder_attention_probs_dropout_prob = encoder_attention_probs_dropout_prob
        self.encoder_max_position_embeddings = encoder_max_position_embeddings
        self.encoder_type_vocab_size = encoder_type_vocab_size
        self.encoder_layer_norm_eps = encoder_layer_norm_eps
        self.encoder_relative_attention = encoder_relative_attention
        self.encoder_max_relative_positions = encoder_max_relative_positions
        self.encoder_position_buckets = encoder_position_buckets
        self.encoder_pos_att_type = encoder_pos_att_type or ["p2c", "c2p"]
        self.encoder_share_att_key = encoder_share_att_key
        self.encoder_norm_rel_ebd = encoder_norm_rel_ebd
        self.encoder_position_biased_input = encoder_position_biased_input
        self.encoder_pad_token_id = encoder_pad_token_id

        self.num_labels = int(num_labels)
        self.pooler_hidden_size = int(pooler_hidden_size or encoder_hidden_size)
        self.pooler_hidden_act = pooler_hidden_act
        self.pooler_dropout = float(pooler_dropout)

        self.id2label = id2label or {
            "0": "entailment",
            "1": "neutral",
            "2": "contradiction",
        }
        self.label2id = label2id or {
            "entailment": 0,
            "neutral": 1,
            "contradiction": 2,
        }
