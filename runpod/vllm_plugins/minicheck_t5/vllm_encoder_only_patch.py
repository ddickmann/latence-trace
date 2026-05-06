"""Force vLLM 0.19 to treat MiniCheck T5 as encoder-only.

The upstream ``lytang/MiniCheck-Flan-T5-Large`` checkpoint ships with a
stock T5 ``config.json`` (``model_type="t5"``,
``is_encoder_decoder=true``, ``architectures=["T5ForConditionalGeneration"]``).
HuggingFace's ``AutoConfig.from_pretrained`` therefore loads a regular
:class:`~transformers.T5Config`, not our :class:`MiniCheckT5Config` —
vLLM dispatches our model class via the ``architectures`` alias but
the *config* it reads is upstream HF.

That config has ``is_encoder_decoder=True``, which trips
``vllm.v1.core.sched.scheduler.Scheduler.__init__``::

    AssertionError: Encoder-decoder models are expected to implement
    the multimodal interface with at most one modality.

Our model ISN'T encoder-decoder from vLLM's scheduling perspective —
:meth:`MiniCheckT5Model.forward` runs the encoder, the single-step
decoder, the LM head, and the softmax all internally and emits a
single (yes_logit, no_logit) pair per row through the pooling runner.
So we need to make :func:`vllm.transformers_utils.config.is_encoder_decoder`
return ``False`` for any HF config whose ``architectures`` lists one of
our registered class names.

Same plumbing convention as
:mod:`runpod.vllm_plugins.nli_mdeberta.vllm_pooling_token_type_ids` —
applied from the plugin's ``register()`` so the patch lands in BOTH
the API server process and the EngineCore subprocess. (vLLM calls
``load_general_plugins()`` from both, so registering once in
``register()`` is sufficient.)
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

logger = logging.getLogger(__name__)

_PATCH_ATTR = "_latence_trace_minicheck_encoder_only_patched"

# Architecture names whose HF config we should treat as encoder-only.
# Kept as a module-level set so it stays consistent with the aliases
# registered in ``minicheck_t5.__init__.register``.
_MINICHECK_ARCHITECTURES = frozenset(
    [
        "MiniCheckT5Model",
        "T5ForConditionalGeneration",
        "MT5ForConditionalGeneration",
    ]
)


def _config_arch_names(config: Any) -> Iterable[str]:
    arch = getattr(config, "architectures", None) or []
    try:
        return [str(name) for name in arch]
    except TypeError:
        return []


def _config_is_minicheck(config: Any) -> bool:
    """Return True if this HF config is the upstream MiniCheck checkpoint.

    Architecture alone is too coarse — every T5 / MT5 / Flan-T5
    checkpoint declares ``architectures=["T5ForConditionalGeneration"]``,
    so matching on architecture only would silently turn stock
    ``google/flan-t5-base`` into an encoder-only model if the plugin
    were loaded. Instead we require:

    * ``model_type == "minicheck_t5"`` (set when AutoConfig loaded our
      :class:`MiniCheckT5Config` — happens for local fine-tunes that
      ship our config_class), OR
    * ``architectures`` lists ``MiniCheckT5Model`` (our plugin's own
      class name — never present on upstream T5 checkpoints), OR
    * Both an architecture in :data:`_MINICHECK_ARCHITECTURES` AND
      the ``_name_or_path`` containing ``"minicheck"`` (the upstream
      ``lytang/MiniCheck-Flan-T5-*`` family).

    This keeps the patch a strict no-op for any T5 checkpoint that
    isn't MiniCheck, even if the plugin is unintentionally loaded
    into a multi-model worker.
    """

    if str(getattr(config, "model_type", "")) == "minicheck_t5":
        return True

    arch_names = list(_config_arch_names(config))
    if "MiniCheckT5Model" in arch_names:
        return True

    if arch_names and any(name in _MINICHECK_ARCHITECTURES for name in arch_names):
        name = str(getattr(config, "_name_or_path", "")).lower()
        if "minicheck" in name:
            return True

    return False


def _make_patched_is_encoder_decoder(orig_is_encoder_decoder):
    def _is_encoder_decoder(config: Any) -> bool:
        if _config_is_minicheck(config):
            return False
        return orig_is_encoder_decoder(config)

    _is_encoder_decoder.__wrapped__ = orig_is_encoder_decoder  # type: ignore[attr-defined]
    return _is_encoder_decoder


def apply_encoder_only_patch() -> bool:
    """Patch ``vllm.transformers_utils.config.is_encoder_decoder`` (idempotent).

    Returns ``True`` if we just installed the patch, ``False`` if it
    was already in place. Safe to call multiple times — the second
    call is a no-op so importing the plugin a second time (e.g. from
    the EngineCore subprocess) doesn't double-patch.
    """

    try:
        import vllm.transformers_utils.config as _vllm_cfg
        import vllm.config.model as _vllm_model_cfg
    except ImportError:  # pragma: no cover - vLLM is required
        logger.warning("vllm not importable; skipping is_encoder_decoder patch")
        return False

    if getattr(_vllm_cfg, _PATCH_ATTR, False):
        return False

    orig_module_fn = _vllm_cfg.is_encoder_decoder
    patched = _make_patched_is_encoder_decoder(orig_module_fn)
    _vllm_cfg.is_encoder_decoder = patched

    # vllm.config.model imports the symbol at module load time
    # (``from vllm.transformers_utils.config import is_encoder_decoder``
    # in some 0.19.x patch revs), so re-binding the module attribute
    # alone isn't always enough — also rebind the imported alias.
    if hasattr(_vllm_model_cfg, "is_encoder_decoder"):
        _vllm_model_cfg.is_encoder_decoder = patched

    setattr(_vllm_cfg, _PATCH_ATTR, True)
    logger.info(
        "patched vllm.transformers_utils.config.is_encoder_decoder for "
        "MiniCheckT5 (architectures=%s)",
        sorted(_MINICHECK_ARCHITECTURES),
    )
    return True


__all__ = ["apply_encoder_only_patch"]
