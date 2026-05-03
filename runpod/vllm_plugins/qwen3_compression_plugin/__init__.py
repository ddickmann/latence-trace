"""
Qwen3 Compression Plugin for vLLM 0.14

Token classification model for LLMLingua2 text compression.
Returns per-token keep/remove logits for intelligent compression.

Key Features:
- Uses vLLM native Qwen2Model (Qwen3 compatible)
- Classification head returns (seq_len, 2) logits per sequence
- Works with Rust text_processing module for fast compression

Usage with vLLM:
    from vllm import LLM

    # Plugin auto-registers via entry point
    llm = LLM(
        model="path/to/compression-model",
        task="classify",  # vLLM 0.14 uses "classify" for token classification
        trust_remote_code=True,
    )
"""

__version__ = "1.1.0"


def register_model():
    """vLLM plugin entry point - called when plugin is loaded.

    Called automatically on import and via vLLM plugin entry point.
    All imports are inside this function to prevent module-level
    crashes from making the entry point attribute unreachable.
    """
    try:
        from transformers import AutoConfig
        from vllm import ModelRegistry

        from .config import Qwen3TokenClassificationConfig
        from .model import Qwen3ForTokenClassification

        # Register with HuggingFace AutoConfig
        try:
            AutoConfig.register("qwen3_token_classification", Qwen3TokenClassificationConfig)
            print("[qwen3_compression_plugin] Registered Qwen3TokenClassificationConfig")
        except ValueError:
            pass  # Already registered

        # Register with vLLM ModelRegistry
        try:
            ModelRegistry.register_model("Qwen3ForTokenClassification", Qwen3ForTokenClassification)
            print("[qwen3_compression_plugin] Registered Qwen3ForTokenClassification")
        except (KeyError, ValueError):
            pass  # Already registered

    except Exception as e:
        print(f"[qwen3_compression_plugin] ERROR: Failed to register model: {e}")
        import traceback
        traceback.print_exc()


# Alias for backward compatibility
register_qwen3_compression_models = register_model


# Auto-register on import (critical for vLLM subprocess loading)
register_model()

__all__ = [
    "register_model",
    "register_qwen3_compression_models",
]
