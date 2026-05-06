"""latence-trace MiniCheck-Flan-T5 BYOP plugin.

Registers the :class:`MiniCheckT5Model` + :class:`MiniCheckT5Config`
pair with vLLM's ModelRegistry under both the canonical plugin name and
the upstream HuggingFace architecture aliases. This is the entry point
vLLM calls when it executes the ``vllm.general_plugins`` group at
worker startup — see :mod:`runpod.handler` for the boot wiring.

The module-level :func:`register` invocation runs once per Python
process. It is idempotent, matching the convention :mod:`forge.registration`
expects across the rest of the plugin ecosystem.
"""

from forge.registration import register_plugin

from .config import MiniCheckT5Config
from .model import MiniCheckT5Model
from .vllm_encoder_only_patch import apply_encoder_only_patch


def register() -> None:
    register_plugin(
        "minicheck_t5",
        MiniCheckT5Config,
        "MiniCheckT5Model",
        MiniCheckT5Model,
        # Upstream Flan-T5 / MiniCheck checkpoints declare these
        # architecture strings in ``config.json``. Aliasing means
        # operators can point ``vllm serve`` at the public HF
        # checkpoint without monkey-patching its config.
        aliases=[
            "T5ForConditionalGeneration",
            "MT5ForConditionalGeneration",
        ],
    )
    # IMPORTANT: this monkey-patch must run inside BOTH the API
    # server process AND the vLLM V1 EngineCore subprocess. vLLM
    # calls ``load_general_plugins()`` from each, which in turn
    # imports this plugin and runs ``register()``, so triggering the
    # patch here is sufficient. Same convention as
    # ``nli_mdeberta.apply_pooling_token_type_ids_patch``.
    #
    # Why we need it: HF AutoConfig loads the upstream MiniCheck
    # ``config.json`` as a regular ``T5Config`` with
    # ``is_encoder_decoder=True``. vLLM's V1 scheduler then asserts
    # that encoder-decoder configs implement the multimodal
    # interface — which we don't, because the decoder + LM head +
    # softmax all run inside :meth:`MiniCheckT5Model.forward` and
    # vLLM only sees a single pooled (yes, no) logit pair per row.
    apply_encoder_only_patch()


register()


__all__ = ["MiniCheckT5Config", "MiniCheckT5Model", "register"]
