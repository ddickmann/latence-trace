"""HuggingFace + vLLM config for the MiniCheck-Flan-T5 BYOP plugin.

Two design choices worth calling out:

1. ``num_layers``/``num_hidden_layers`` is stored under custom
   ``encoder_*`` / ``decoder_*`` fields rather than the base
   :class:`~transformers.MT5Config` slots. The base slots are clamped to
   ``0`` so vLLM's pooling runner does *not* try to allocate KV cache for
   the encoder-decoder stack — our :class:`MiniCheckT5Model` owns the
   layers itself (encoder via ``vllm-factory``'s Triton-optimised
   :class:`~models.mt5.MT5Encoder`, decoder via a single-step
   pure-PyTorch stack with cross-attention). This is the same trick used
   by the ``deberta_gliner`` plugin and by the existing
   :class:`~latence_trace_nli_plugin.NLIDebertaV2Config`.

2. The MiniCheck inference recipe (Tang et al. 2024) only ever runs *one*
   decoder step from ``decoder_start_token_id`` and reads the logits over
   the ``Yes`` / ``No`` vocab tokens. We resolve those token IDs at
   tokenizer load time inside :class:`MiniCheckT5IOProcessor` and stash
   them on the config so :meth:`MiniCheckT5Model.forward` can slice the
   LM-head output without re-tokenising on every batch.
"""

from __future__ import annotations

from typing import Any, Optional

from transformers import MT5Config


class MiniCheckT5Config(MT5Config):
    """Plugin config for ``lytang/MiniCheck-Flan-T5-Large``.

    The model is a Flan-T5-Large (770M params, 24 enc + 24 dec layers,
    ``d_model=1024``, ``d_ff=2816``, ``num_heads=16``, ``d_kv=64``)
    fine-tuned on the LLM-AggreFact training mixture for grounded
    fact-checking. Vocabulary, tokenizer and feed-forward variant
    (``gated-gelu``) come straight from the upstream Flan-T5 v1.1
    family.

    Defaults below match the published checkpoint so a stock
    ``from_pretrained("lytang/MiniCheck-Flan-T5-Large")`` round-trips
    cleanly through this config when used as a vLLM ``hf_config``.
    """

    model_type = "minicheck_t5"

    def __init__(
        self,
        # ── Backbone (Flan-T5-Large defaults) ───────────────────────────
        vocab_size: int = 32128,
        d_model: int = 1024,
        d_kv: int = 64,
        d_ff: int = 2816,
        encoder_num_layers: int = 24,
        decoder_num_layers: int = 24,
        num_heads: int = 16,
        relative_attention_num_buckets: int = 32,
        relative_attention_max_distance: int = 128,
        dropout_rate: float = 0.0,
        layer_norm_epsilon: float = 1e-6,
        initializer_factor: float = 1.0,
        feed_forward_proj: str = "gated-gelu",
        # ── Decoding tokens ─────────────────────────────────────────────
        # Flan-T5 uses ``<pad>`` (id=0) as the decoder start token (T5
        # convention). MiniCheck inherits this — Tang et al. 2024 §3.
        decoder_start_token_id: int = 0,
        eos_token_id: int = 1,
        pad_token_id: int = 0,
        # ── MiniCheck-specific (resolved at tokenizer load) ────────────
        # We leave these unset on the class default and let the IO
        # processor populate them from the tokenizer at startup. They are
        # stashed on the live config object so the model's ``forward``
        # can slice the LM head without re-tokenising every batch.
        yes_token_id: Optional[int] = None,
        no_token_id: Optional[int] = None,
        # ── tie/untie (Flan-T5 v1.1 keeps lm_head untied) ──────────────
        tie_word_embeddings: bool = False,
        # Force ``is_encoder_decoder=False`` from vLLM's perspective.
        # Our model owns the decoder + LM head + softmax internally
        # inside :meth:`MiniCheckT5Model.forward` and emits a single
        # pooled logit pair per row, so to vLLM's pooling runner this
        # behaves as an encoder-only classifier. Leaving this ``True``
        # makes vLLM 0.19.x's pooling path raise ``Encoder-decoder
        # models are expected to implement the multimodal interface
        # with at most one modality`` — which we are decidedly not.
        is_encoder_decoder: bool = False,
        use_cache: bool = False,
        **kwargs: Any,
    ) -> None:
        # Hide layer counts from vLLM's KV-cache allocator. Our model
        # builds the encoder + decoder layers internally, so vLLM should
        # not try to track them as managed Attention modules.
        kwargs.pop("num_layers", None)
        kwargs.pop("num_decoder_layers", None)
        kwargs.pop("num_hidden_layers", None)
        # Caller may have spelled this in kwargs; we drop it so our
        # explicit False below wins. (HF ``MT5Config.from_pretrained``
        # always pulls ``is_encoder_decoder=True`` from the upstream
        # ``config.json`` — this scrub is what neutralises that.)
        kwargs.pop("is_encoder_decoder", None)

        super().__init__(
            vocab_size=vocab_size,
            d_model=d_model,
            d_kv=d_kv,
            d_ff=d_ff,
            num_layers=0,
            num_decoder_layers=0,
            num_heads=num_heads,
            relative_attention_num_buckets=relative_attention_num_buckets,
            relative_attention_max_distance=relative_attention_max_distance,
            dropout_rate=dropout_rate,
            layer_norm_epsilon=layer_norm_epsilon,
            initializer_factor=initializer_factor,
            feed_forward_proj=feed_forward_proj,
            is_encoder_decoder=is_encoder_decoder,
            use_cache=use_cache,
            tie_word_embeddings=tie_word_embeddings,
            decoder_start_token_id=decoder_start_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            **kwargs,
        )

        # ``MT5Config.__init__`` will sometimes restore ``True`` from
        # the parent class default after ``__init__`` runs. Pin it
        # post-super so the live config object is the one vLLM reads.
        self.is_encoder_decoder = False

        # vLLM 0.19.x peeks at ``num_hidden_layers`` from the live config
        # object (not just from the constructor kwargs) when building the
        # KV-cache plan. Force it to zero here too so the trick from the
        # super().__init__ above survives any HF post-init normalisation.
        self.num_hidden_layers = 0
        self.num_layers = 0
        self.num_decoder_layers = 0

        # Real layer counts the model actually instantiates.
        self.encoder_num_layers = int(encoder_num_layers)
        self.decoder_num_layers = int(decoder_num_layers)

        # MiniCheck label vocabulary (resolved by IO processor at startup).
        self.yes_token_id = yes_token_id
        self.no_token_id = no_token_id

        # Convenience flags mirroring vllm-factory MT5 config.
        self.is_gated_act = feed_forward_proj.startswith("gated")
        self.dense_act_fn = "gelu_new" if "gelu" in feed_forward_proj else "relu"

        # AutoModel mapping is intentionally left empty — the plugin
        # registers itself with vLLM via :func:`forge.register_plugin`.
        self.auto_map = {}


__all__ = ["MiniCheckT5Config"]
