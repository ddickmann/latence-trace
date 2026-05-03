"""
Qwen3 Token Classification Model for vLLM 0.14

This module implements Qwen3ForTokenClassification for LLMLingua2 compression.
Returns per-token keep/remove logits for text compression.

Key Features:
- Uses vLLM's native Qwen3Model (NOT Qwen2Model - Qwen3 has attention_bias=False)
- is_pooling_model = True for pooling task
- Classification head outputs (seq_len, 2) logits per sequence
"""

from collections.abc import Iterable

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.model_executor.layers.pooler.tokwise import pooler_for_token_classify

# IMPORTANT: Use Qwen3Model, NOT Qwen2Model!
# Qwen2Model hardcodes bias=True for QKV projections.
# Qwen3Model reads attention_bias from config (False for this model),
# avoiding uninitialized qkv_proj.bias weights.
from vllm.model_executor.models.qwen3 import Qwen3Model
from vllm.model_executor.models.utils import AutoWeightsLoader, maybe_prefix
from vllm.sequence import IntermediateTensors


class Qwen3ForTokenClassification(nn.Module):
    """Qwen3 model with token classification head for compression.

    This model:
    1. Encodes input text using Qwen3 (via Qwen2Model in vLLM)
    2. Applies classification head to each token
    3. Returns (seq_len, 2) logits for keep/remove decision

    Used for LLMLingua2 text compression.
    """

    # Tell vLLM this is a pooling model
    is_pooling_model = True

    def __init__(
        self,
        *,  # vLLM 0.14 requires keyword-only args
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()

        config = vllm_config.model_config.hf_config

        # Get classification parameters from config
        self.num_labels = getattr(config, 'num_labels', 2)
        hidden_size = config.hidden_size

        # Get dtype from vllm config - use model dtype, fallback to float16
        # head_dtype can be None, so we need a fallback
        self.head_dtype = vllm_config.model_config.head_dtype
        if self.head_dtype is None:
            # Match the model's dtype
            self.head_dtype = vllm_config.model_config.dtype
        if self.head_dtype is None:
            self.head_dtype = torch.float16

        self.config = config

        # Qwen3 encoder backbone (must use Qwen3Model to respect attention_bias=False)
        self.model = Qwen3Model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
        )

        # Set up pooler for token classification
        # The pooler handles the classification head internally
        pooler_config = vllm_config.model_config.pooler_config
        assert pooler_config is not None, "pooler_config is required for token classification"

        # Create classification head for weight loading
        # vLLM's pooler will use this for token classification
        # IMPORTANT: Use same dtype as backbone model
        self.classifier = nn.Linear(
            hidden_size,
            self.num_labels,
            dtype=self.head_dtype,
        )

        # Create pooler with the classification head
        # Pass our classifier to the pooler so it uses our weights
        def classifier_fn(hidden_states: torch.Tensor) -> torch.Tensor:
            # Ensure dtype matches
            if self.classifier.weight.dtype != hidden_states.dtype:
                # #region agent log - DEBUG: Dtype conversion happening
                print(f"[CLASSIFIER_DTYPE_CONV] Converting classifier from {self.classifier.weight.dtype} to {hidden_states.dtype}")
                # #endregion
                self.classifier = self.classifier.to(hidden_states.dtype)

            # #region agent log - DEBUG: Check hidden_states for NaN before classification
            _hs_nan = torch.isnan(hidden_states).sum().item()
            _hs_inf = torch.isinf(hidden_states).sum().item()
            if _hs_nan > 0 or _hs_inf > 0:
                print(f"[CLASSIFIER_INPUT_NAN] hidden_states has NaN={_hs_nan}, Inf={_hs_inf}, shape={hidden_states.shape}")
            # #endregion

            logits = self.classifier(hidden_states)

            # #region agent log - DEBUG: Check logits for NaN after classification
            _logits_nan = torch.isnan(logits).sum().item()
            _logits_inf = torch.isinf(logits).sum().item()
            if _logits_nan > 0 or _logits_inf > 0:
                print(f"[CLASSIFIER_OUTPUT_NAN] logits has NaN={_logits_nan}, Inf={_logits_inf}, shape={logits.shape}")
                print(f"[CLASSIFIER_OUTPUT_NAN] classifier.weight has_nan={torch.isnan(self.classifier.weight).any().item()}")
                print(f"[CLASSIFIER_OUTPUT_NAN] first few hidden_states: {hidden_states[:3, :5].tolist()}")
            # #endregion

            return logits

        self.pooler = pooler_for_token_classify(
            pooler_config,
            classifier=classifier_fn,
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get token embeddings from input_ids (required by vLLM)."""
        return self.model.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **_kwargs,
    ) -> torch.Tensor:
        """Forward pass through encoder.

        vLLM pooling models receive:
        - input_ids: (total_tokens,) 1D tensor with concatenated sequences
        - positions: (total_tokens,) 1D

        Returns:
            Hidden states tensor (total_tokens, hidden_size)
            The pooler will apply classification to get logits
        """
        # Pass through Qwen2 backbone
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            inputs_embeds=inputs_embeds,
            intermediate_tensors=intermediate_tensors,
        )

        # Qwen2Model returns 2D tensor (seq_len, hidden) for pooling models
        # Ensure it's 2D
        if hidden_states.dim() == 3:
            hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        # #region agent log - DEBUG: Check hidden states for NaN before pooler
        _nan_count = torch.isnan(hidden_states).sum().item()
        _inf_count = torch.isinf(hidden_states).sum().item()
        if _nan_count > 0 or _inf_count > 0:
            print(f"[MODEL_HIDDEN_NAN] NaN={_nan_count}, Inf={_inf_count}, shape={hidden_states.shape}, dtype={hidden_states.dtype}")
        # #endregion

        # IMPORTANT: Return hidden_states, NOT logits!
        # The vLLM pooler (pooler_for_token_classify) will apply classification.
        # Returning logits here causes the pooler to classify already-classified outputs -> NaN
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights from checkpoint.

        Handles mapping:
        - score.weight -> classifier.weight
        - score.bias -> classifier.bias
        """
        # Remap "score" to "classifier" for the classification head
        def remap_weights():
            for name, tensor in weights:
                if name.startswith("score."):
                    # Map score -> classifier
                    new_name = name.replace("score.", "classifier.")
                    # #region agent log - DEBUG: Log classifier weight loading
                    print(f"[WEIGHT_LOAD] {name} -> {new_name}: shape={tensor.shape}, dtype={tensor.dtype}, has_nan={torch.isnan(tensor).any().item()}")
                    # #endregion
                    yield new_name, tensor
                else:
                    yield name, tensor

        loader = AutoWeightsLoader(self)
        loaded_params = loader.load_weights(remap_weights())

        # #region agent log - DEBUG: Verify classifier weights after loading
        if hasattr(self, 'classifier') and self.classifier is not None:
            w = self.classifier.weight
            b = self.classifier.bias if self.classifier.bias is not None else None
            w_nan = torch.isnan(w).any().item()
            b_nan = torch.isnan(b).any().item() if b is not None else False
            print(f"[CLASSIFIER_WEIGHTS] weight: shape={w.shape}, dtype={w.dtype}, has_nan={w_nan}, min={w.min().item():.6f}, max={w.max().item():.6f}")
            if b is not None:
                print(f"[CLASSIFIER_WEIGHTS] bias: shape={b.shape}, dtype={b.dtype}, has_nan={b_nan}, min={b.min().item():.6f}, max={b.max().item():.6f}")
        # #endregion

        return loaded_params


# Register models
_MODEL_REGISTRY = {
    "Qwen3ForTokenClassification": Qwen3ForTokenClassification,
}


def get_model(vllm_config: VllmConfig, **kwargs):
    """Get the appropriate model based on config."""
    architectures = vllm_config.model_config.hf_config.architectures
    if architectures:
        model_class = architectures[0]
        if model_class in _MODEL_REGISTRY:
            return _MODEL_REGISTRY[model_class](vllm_config=vllm_config, **kwargs)

    # Default to token classification model
    return Qwen3ForTokenClassification(vllm_config=vllm_config, **kwargs)
