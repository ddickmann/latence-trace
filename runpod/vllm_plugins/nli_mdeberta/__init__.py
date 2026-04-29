"""latence-trace NLI BYOP plugin."""

from forge.registration import register_plugin

from .config import NLIDebertaV2Config
from .model import NLIDebertaV2Model
from .vllm_pooling_token_type_ids import apply_pooling_token_type_ids_patch


def register() -> None:
    register_plugin(
        "nli_mdeberta",
        NLIDebertaV2Config,
        "NLIDebertaV2Model",
        NLIDebertaV2Model,
        aliases=["DebertaV2ForSequenceClassification"],
    )
    # IMPORTANT: this monkey patch must also run inside the vLLM V1
    # ``EngineCore`` subprocess, where the actual ``GPUModelRunner``
    # executes the forward pass. vLLM calls ``load_general_plugins()``
    # from both ``v1/engine/core.py`` and ``v1/worker/worker_base.py``
    # before the model is built, so applying the patch in ``register()``
    # (bound to the ``vllm.general_plugins`` entry point) is the only
    # way to prevent the ``cuda:0 vs cpu`` device-mismatch crash under
    # batched DeBERTa NLI traffic. Applying it in the IO processor
    # ``__init__`` is not enough - that only runs in the API server
    # process.
    apply_pooling_token_type_ids_patch()


register()

__all__ = ["NLIDebertaV2Config", "NLIDebertaV2Model", "register"]
