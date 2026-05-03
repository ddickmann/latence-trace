"""
Qwen3 Token Classification Configuration

Extends Qwen3Config to add token classification parameters.
Used for LLMLingua2 compression model.
"""

from transformers import Qwen2Config


class Qwen3TokenClassificationConfig(Qwen2Config):
    """Configuration for Qwen3ForTokenClassification.

    This config extends Qwen2Config (Qwen3 uses same config class)
    and adds token classification parameters:
    - num_labels: Number of classification labels (2 for keep/remove)
    - classifier_dropout: Dropout for classification head

    The model_type is "qwen3_token_classification" for vLLM registration.
    """
    model_type = "qwen3_token_classification"

    def __init__(
        self,
        num_labels: int = 2,
        classifier_dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_labels = num_labels
        self.classifier_dropout = classifier_dropout

