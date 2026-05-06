"""MiniCheck-Flan-T5 model for vLLM — encoder-decoder grounded NLI.

The published MiniCheck inference recipe (Tang et al. 2024) is dead
simple: encode the formatted ``predict: <premise>\\nclaim: <claim>``
string, run **one** decoder step from the model's
``decoder_start_token_id``, then read the LM-head logits at position 0
over the ``Yes`` and ``No`` vocabulary entries. The softmax of those two
logits is the grounding probability in ``[0, 1]``.

This file ports that recipe to a vLLM pooling-style model so we can
serve MiniCheck through the same ``--task pooling --io-processor-plugin``
plumbing as the existing :mod:`latence_trace_nli_plugin` (mDeBERTa). The
two backends therefore share an identical wire shape and the
``latence_trace`` registry can swap them with a single env var.

Architecture
------------

    input_ids  ─────────────►  MT5Encoder (Triton)         ─►  enc_h
                                                                │
                              T5SingleStepDecoder              ◄┘
                                  │
                             LM head (untied)
                                  │
                       extract Yes / No logits
                                  │
            pack into (total_input_tokens, 2) at position 0

The encoder is the heavy lift (premises run up to ~1k tokens) and is
served by ``vllm-factory``'s :class:`~models.mt5.MT5Encoder`, which
already wires in the ``flash_attention_rpb`` and
``fused_gelu_mul_dropout`` Triton kernels we want. The decoder runs one
step on one token per sequence and is therefore *intentionally*
pure-PyTorch + ``F.scaled_dot_product_attention``: at B=64 it is
sub-millisecond, optimising it would be a wasted weekend.

Why pack the result into ``(total_input_tokens, 2)`` instead of returning
``(B, 2)`` directly? vLLM's pooling runner concatenates all sequences
into a flat ``(total_tokens, hidden)`` tensor and slices it with
``seq_lengths`` from :class:`PoolerContext` before handing the per-row
hidden states to the pooler. We adopt the same convention as the
``nli_mdeberta`` plugin so :class:`MiniCheckPooler` can mirror
:class:`NLIClassifyPooler`'s ``h[0]`` slice without any custom routing —
the Yes/No logits ride at position 0 of each row, the rest are zero
fillers, and the whole tensor stays in vLLM's expected layout.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import MT5Config
from vllm.config import VllmConfig

from vllm_factory.pooling.vllm_adapter import VllmPoolerAdapter

from .config import MiniCheckT5Config
from .pooler import MiniCheckPooler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# vllm-factory MT5Encoder import
# ---------------------------------------------------------------------------
#
# We follow the same trick the existing ``nli_mdeberta`` plugin uses to
# locate the encoder file: try the installed ``vllm_factory`` package
# first, then fall back to the regular ``models.mt5`` import path that
# works inside the vllm-factory monorepo. Both branches end up with the
# same :class:`MT5Encoder` class, so callers get the Triton kernels
# regardless of which install layout the host image happens to use.

def _import_mt5_encoder():
    try:
        import vllm_factory  # noqa: WPS433

        candidate = (
            Path(vllm_factory.__file__).resolve().parent.parent
            / "models"
            / "mt5"
            / "mt5_encoder.py"
        )
        if candidate.exists():
            spec = importlib.util.spec_from_file_location(
                "latence_trace_minicheck_t5_encoder",
                str(candidate),
            )
            if spec is not None and spec.loader is not None:
                module = importlib.util.module_from_spec(spec)
                sys.modules.setdefault(
                    "latence_trace_minicheck_t5_encoder", module
                )
                spec.loader.exec_module(module)
                return module
    except Exception:  # pragma: no cover — vllm_factory layout differs
        pass

    from models.mt5 import mt5_encoder as module  # noqa: WPS433

    return module


_encoder_mod = _import_mt5_encoder()
MT5Encoder = _encoder_mod.MT5Encoder
T5SelfAttention = _encoder_mod.T5SelfAttention
T5FF = _encoder_mod.T5FF
T5RelativePositionBias = _encoder_mod.T5RelativePositionBias

# vllm-factory uses MT5RMSNorm under the hood; expose it through the
# same alias the encoder file uses so the layer-norm match is bit-exact.
try:
    from vllm.model_executor.layers.layernorm import RMSNorm as MT5RMSNorm
except Exception:  # pragma: no cover - import shouldn't fail in vLLM env
    from torch.nn import LayerNorm as MT5RMSNorm  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Single-step T5 decoder
# ---------------------------------------------------------------------------
#
# vllm-factory only exposes ``MT5Encoder`` — the encoder-decoder stack is
# documented as future work in the file header. For MiniCheck we only
# need ONE decoder step on ONE token per sequence (the
# ``decoder_start_token_id``), so we hand-roll a minimal decoder block
# in pure PyTorch. Self-attention on a single token is trivial; the
# meaningful compute is the cross-attention against the encoder hidden
# states. We use ``F.scaled_dot_product_attention`` (memory-efficient
# kernel) so even a B=128 batch finishes well under a millisecond.


class T5DecoderCrossAttention(nn.Module):
    """T5 cross-attention: queries from decoder, keys/values from encoder.

    No relative position bias on cross-attention (T5 convention).
    Keeps the same projection layout as the encoder self-attention so
    HuggingFace ``EncDecAttention.{q,k,v,o}`` weights map cleanly via
    :func:`MiniCheckT5Model._map_hf_decoder_key`.
    """

    def __init__(self, d_model: int, n_heads: int, d_kv: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_kv
        self.inner_dim = n_heads * d_kv
        self.q_proj = nn.Linear(d_model, self.inner_dim, bias=False)
        self.k_proj = nn.Linear(d_model, self.inner_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.inner_dim, bias=False)
        self.out_proj = nn.Linear(self.inner_dim, d_model, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,           # (B, Q, d_model)
        encoder_hidden_states: torch.Tensor,   # (B, K, d_model)
        encoder_attention_mask: Optional[torch.Tensor] = None,  # (B, K)
    ) -> torch.Tensor:
        b, q_len, _ = hidden_states.shape
        k_len = encoder_hidden_states.shape[1]

        q = self.q_proj(hidden_states).view(b, q_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(encoder_hidden_states).view(b, k_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(encoder_hidden_states).view(b, k_len, self.n_heads, self.head_dim).transpose(1, 2)

        attn_mask = None
        if encoder_attention_mask is not None:
            # (B, K) → (B, 1, 1, K); 0 → -inf, 1 → 0.0 additive mask.
            mask = encoder_attention_mask.to(dtype=q.dtype)
            attn_mask = (1.0 - mask)[:, None, None, :] * torch.finfo(q.dtype).min

        # T5 deliberately uses scale=1.0 (the projections are scaled at
        # init time instead). Mirror the encoder's behaviour.
        ctx = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False,
            scale=1.0,
        )
        ctx = ctx.transpose(1, 2).contiguous().view(b, q_len, self.inner_dim)
        return self.out_proj(ctx)


class T5DecoderLayer(nn.Module):
    """One T5 decoder block: self-attn → cross-attn → FFN, all pre-norm.

    Self-attention reuses the encoder's :class:`T5SelfAttention` with
    ``is_decoder=True``. Only the *first* decoder layer carries the
    relative-position-bias table — every subsequent layer reads its
    sibling's bias the same way the encoder stack does (``has_rpb`` flag
    on layer 0).
    """

    def __init__(
        self,
        cfg: MT5Config,
        cache_config,
        quant_config,
        prefix: str,
        *,
        has_rpb: bool,
    ) -> None:
        super().__init__()
        self.ln1 = MT5RMSNorm(cfg.d_model, eps=cfg.layer_norm_epsilon)
        self.self_attn = T5SelfAttention(
            d_model=cfg.d_model,
            n_heads=cfg.num_heads,
            d_kv=cfg.d_kv,
            dropout=cfg.dropout_rate,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.SelfAttention",
            use_rpb=has_rpb,
            rpb_num_buckets=cfg.relative_attention_num_buckets,
            rpb_max_distance=cfg.relative_attention_max_distance,
            is_decoder=True,
        )
        self.ln2 = MT5RMSNorm(cfg.d_model, eps=cfg.layer_norm_epsilon)
        self.cross_attn = T5DecoderCrossAttention(
            d_model=cfg.d_model,
            n_heads=cfg.num_heads,
            d_kv=cfg.d_kv,
        )
        self.ln3 = MT5RMSNorm(cfg.d_model, eps=cfg.layer_norm_epsilon)
        self.ff = T5FF(
            cfg.d_model,
            cfg.d_ff,
            cfg.dense_act_fn,
            dropout=cfg.dropout_rate,
            quant_config=quant_config,
            prefix=f"{prefix}.FF",
            gated=cfg.is_gated_act,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,           # (B, 1, d_model)
        encoder_hidden_states: torch.Tensor,   # (B, K, d_model)
        encoder_attention_mask: Optional[torch.Tensor],
        rpb_table: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # Self-attention on a single decoder token. The T5 self-attention
        # block expects (B, L, d_model) and returns the same; with L=1
        # the math collapses to ``v`` weighted by the RPB[0,0] offset
        # which is fine — T5 uses bucket 0 for the diagonal anyway.
        h = hidden_states + self.self_attn(
            self.ln1(hidden_states),
            attn_metadata=None,
            rpb_table=rpb_table,
            attention_mask=None,
        )
        h = h + self.cross_attn(
            self.ln2(h),
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )
        h = h + self.ff(self.ln3(h))
        return h


class T5SingleStepDecoder(nn.Module):
    """Stack of :class:`T5DecoderLayer` for ONE decoder step.

    No KV cache. No autoregression. The runtime cost is a single forward
    over ``num_decoder_layers`` (~24 for Flan-T5-Large) with sequence
    length 1 on the query side. At B=64 this is sub-millisecond on an
    A5000; we have not bothered to Triton-fuse it.
    """

    def __init__(
        self,
        cfg: MT5Config,
        cache_config,
        quant_config,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList(
            [
                T5DecoderLayer(
                    cfg,
                    cache_config,
                    quant_config,
                    prefix=f"decoder.block.{i}",
                    has_rpb=(i == 0),
                )
                for i in range(num_layers)
            ]
        )
        self.final_ln = MT5RMSNorm(cfg.d_model, eps=cfg.layer_norm_epsilon)

    def forward(
        self,
        decoder_input_ids: torch.Tensor,         # (B, 1)
        encoder_hidden_states: torch.Tensor,     # (B, K, d_model)
        encoder_attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        x = self.embed_tokens(decoder_input_ids)

        # Pull RPB table from layer 0 once, broadcast to every layer
        # downstream — same approach as :class:`MT5Encoder`.
        rpb_table_weight: Optional[torch.Tensor] = None
        first_attn = self.layers[0].self_attn
        if first_attn.rpb is not None:
            rpb_table_weight = first_attn.rpb.relative_attention_bias.weight.to(
                x.device
            ).contiguous()

        for layer in self.layers:
            x = layer(
                x,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                rpb_table=rpb_table_weight,
            )
        x = self.final_ln(x)
        return x


# ---------------------------------------------------------------------------
# vLLM-side packing helper
# ---------------------------------------------------------------------------


def _split_packed_batch(
    input_ids: torch.Tensor,
    positions: Optional[torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    *,
    pad_token_id: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, List[int]]:
    """Reconstruct a padded ``(B, T)`` batch from vLLM's flat 1D inputs.

    Same algorithm as
    :func:`latence_trace_nli_plugin.model._split_packed_batch` so the
    behaviour matches the mDeBERTa plugin — vLLM's pooling runner can
    pack multiple sequences into one flat ``input_ids`` tensor with
    per-row ``positions`` resets, and the encoder needs them split out.
    """

    if input_ids.dim() != 1 or positions is None or positions.dim() != 1:
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if positions is not None and positions.dim() == 1:
            positions = positions.unsqueeze(0)
        if attention_mask is not None and attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)
        seq_len = (
            int(input_ids.shape[1]) if input_ids.dim() == 2 else int(input_ids.shape[0])
        )
        if attention_mask is None:
            attention_mask = torch.ones(
                (input_ids.shape[0], seq_len),
                dtype=torch.long,
                device=input_ids.device,
            )
        return input_ids, positions, attention_mask, [seq_len]

    reset_points = torch.nonzero(positions == 0, as_tuple=False).flatten().tolist()
    if not reset_points or reset_points[0] != 0:
        reset_points.insert(0, 0)
    if len(reset_points) == 1:
        single_mask = (
            attention_mask.unsqueeze(0)
            if attention_mask is not None
            else torch.ones((1, input_ids.shape[0]), dtype=torch.long, device=input_ids.device)
        )
        return (
            input_ids.unsqueeze(0),
            positions.unsqueeze(0),
            single_mask,
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
    batched_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)

    for row_idx, (start, end) in enumerate(zip(bounds, bounds[1:])):
        length = end - start
        batched_ids[row_idx, :length] = input_ids[start:end]
        batched_positions[row_idx, :length] = positions[start:end]
        if attention_mask is not None:
            batched_mask[row_idx, :length] = attention_mask[start:end].to(
                dtype=torch.long
            )
        else:
            batched_mask[row_idx, :length] = 1

    return batched_ids, batched_positions, batched_mask, seq_lengths


# ---------------------------------------------------------------------------
# Top-level vLLM model
# ---------------------------------------------------------------------------


class MiniCheckT5Model(nn.Module):
    """vLLM pooling-style wrapper around MiniCheck-Flan-T5-Large.

    Public surface (the bits vLLM actually touches):

    * ``is_pooling_model = True`` — vLLM treats us as an encoder model
      and skips KV-cache allocation.
    * :meth:`forward` returns a ``(total_tokens, 2)`` tensor with the
      ``(yes_logit, no_logit)`` packed at position 0 of each row. The
      attached :class:`MiniCheckPooler` then slices ``h[0]`` and
      softmaxes per row to produce the grounding probability.
    * :meth:`load_weights` accepts the upstream HuggingFace
      ``T5ForConditionalGeneration`` checkpoint format and dispatches it
      to the encoder, decoder and LM head.

    The model deliberately does NOT expose a ``sample`` method backed by
    a SamplingMetadata — it is pooling-only.
    """

    is_pooling_model = True

    def __init__(self, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        cfg = _coerce_minicheck_config(vllm_config.model_config.hf_config)
        self.config = cfg
        self.vllm_config = vllm_config

        self.d_model = int(cfg.d_model)
        self.vocab_size = int(cfg.vocab_size)
        self.encoder_num_layers = int(cfg.encoder_num_layers)
        self.decoder_num_layers = int(cfg.decoder_num_layers)

        cache_config = getattr(vllm_config, "cache_config", None)
        quant_config = getattr(vllm_config, "quant_config", None)

        # ── Encoder (Triton-accelerated) ─────────────────────────────
        # vllm-factory's ``MT5Encoder`` wires in ``flash_attention_rpb``
        # and ``fused_gelu_mul_dropout`` for us. We feed it a config
        # whose ``num_layers`` reflects the *real* depth so it builds
        # the full 24-layer stack.
        encoder_cfg = _build_backbone_cfg(cfg, num_layers=self.encoder_num_layers)
        self.encoder = MT5Encoder(
            cfg=encoder_cfg,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix="encoder",
        )

        # ── Decoder (single-step, pure PyTorch) ──────────────────────
        decoder_cfg = _build_backbone_cfg(cfg, num_layers=self.decoder_num_layers)
        self.decoder = T5SingleStepDecoder(
            cfg=decoder_cfg,
            cache_config=cache_config,
            quant_config=quant_config,
            num_layers=self.decoder_num_layers,
        )

        # ── LM head ──────────────────────────────────────────────────
        # Flan-T5 v1.1 keeps the LM head untied from the input embedding
        # (``tie_word_embeddings=False``). MiniCheck inherits that
        # convention. Loading is handled in :meth:`load_weights` via the
        # ``lm_head.weight`` checkpoint key.
        self.lm_head = nn.Linear(self.d_model, self.vocab_size, bias=False)

        # MiniCheck label tokens — populated either from the checkpoint
        # config (``yes_token_id`` / ``no_token_id``) or by the IO
        # processor at startup. Defaults match the Flan-T5 SentencePiece
        # IDs for ``Yes`` / ``No`` (``2163`` and ``465``) from the
        # public ``lytang/MiniCheck-Flan-T5-Large`` checkpoint.
        self.yes_token_id = int(getattr(cfg, "yes_token_id", None) or 2163)
        self.no_token_id = int(getattr(cfg, "no_token_id", None) or 465)

        # ── Pooler ───────────────────────────────────────────────────
        self._business_pooler = MiniCheckPooler()
        self.pooler = VllmPoolerAdapter(
            self._business_pooler, requires_token_ids=False
        )

    # ── vLLM hooks ───────────────────────────────────────────────────

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Required by vLLM's pooling runner for embedding lookup."""
        return self.encoder.embed_tokens(input_ids)

    def sample(self, logits: torch.Tensor, sampling_metadata):
        """Pooling-only model — return an empty sampler output."""
        try:
            from vllm.sequence import SamplerOutput  # noqa: WPS433

            return SamplerOutput(outputs=[])
        except ImportError:  # pragma: no cover - vLLM API drift
            return None

    # ── Forward pass ─────────────────────────────────────────────────

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        positions: Optional[torch.Tensor] = None,
        intermediate_tensors=None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Encoder → single-step decoder → LM head → pack Yes/No logits.

        Returns a tensor of shape ``(total_input_tokens, 2)`` so that
        vLLM's pooling runner can split it back into per-sequence chunks
        via :class:`PoolerContext.seq_lengths` and hand each chunk to
        :class:`MiniCheckPooler`. We deliberately fill row-position 0
        with ``(yes_logit, no_logit)`` and zero the rest — the pooler
        only ever inspects ``h[0]``.
        """

        attention_mask = kwargs.pop("attention_mask", None)
        pad_id = int(getattr(self.config, "pad_token_id", 0) or 0)
        if input_ids is None:
            raise ValueError("MiniCheckT5Model requires input_ids")

        batched_ids, batched_positions, batched_mask, seq_lengths = _split_packed_batch(
            input_ids,
            positions,
            attention_mask if isinstance(attention_mask, torch.Tensor) else None,
            pad_token_id=pad_id,
        )

        # 1. Encoder (Triton).
        with torch.no_grad():
            enc_h = self.encoder(
                input_ids=batched_ids,
                attention_mask=batched_mask,
            )
        # MT5Encoder returns (B, T, d_model) when called with a 2D batch.
        if enc_h.dim() == 2:
            enc_h = enc_h.unsqueeze(0)

        batch_size = enc_h.shape[0]
        device = enc_h.device

        # 2. One decoder step from ``decoder_start_token_id``.
        decoder_start = int(getattr(self.config, "decoder_start_token_id", 0) or 0)
        decoder_input_ids = torch.full(
            (batch_size, 1),
            decoder_start,
            dtype=torch.long,
            device=device,
        )
        with torch.no_grad():
            dec_h = self.decoder(
                decoder_input_ids=decoder_input_ids,
                encoder_hidden_states=enc_h,
                encoder_attention_mask=batched_mask,
            )

        # 3. LM head over the single decoder step.
        # Flan-T5 v1.1 multiplies the decoder output by ``d_model**-0.5``
        # before the LM head when ``tie_word_embeddings=True``. Untied
        # heads (which MiniCheck inherits) skip the rescale — see
        # ``T5ForConditionalGeneration.forward`` in transformers.
        logits = self.lm_head(dec_h.squeeze(1))  # (B, vocab)

        yes_logits = logits[:, self.yes_token_id]
        no_logits = logits[:, self.no_token_id]
        # Stack per-row Yes/No into the position-0 slot of each sequence.
        per_row = torch.stack([yes_logits, no_logits], dim=-1)  # (B, 2)

        total_tokens = sum(seq_lengths)
        out = torch.zeros((total_tokens, 2), dtype=per_row.dtype, device=device)
        offset = 0
        for row_idx, length in enumerate(seq_lengths):
            out[offset] = per_row[row_idx]
            offset += length

        return out

    # ── Weight loader ────────────────────────────────────────────────

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        """Map HuggingFace ``T5ForConditionalGeneration`` weights onto us.

        The encoder gets its own remap (HF T5 → vllm-factory MT5Encoder
        keys, identical to what ``mt5_gliner`` does). The decoder uses a
        cleaner local mapping. The LM head and any shared embedding land
        in :class:`nn.Linear`/:class:`nn.Embedding` slots directly via
        ``state_dict``.
        """

        encoder_pairs: list[Tuple[str, torch.Tensor]] = []
        decoder_state: dict[str, torch.Tensor] = {}
        head_state: dict[str, torch.Tensor] = {}

        for hf_name, tensor in weights:
            if hf_name == "shared.weight":
                # Shared embedding: feed both the encoder's
                # ``embed_tokens`` and the decoder's ``embed_tokens``.
                # The encoder's loader will pick this up via its own
                # ``shared.weight`` -> ``embed_tokens.weight`` rule.
                encoder_pairs.append((hf_name, tensor))
                decoder_state["embed_tokens.weight"] = tensor.clone()
                # Untied LM-head fallback: when a checkpoint omits
                # ``lm_head.weight`` we tie to the shared embedding so
                # the model still produces sensible logits. MiniCheck
                # ships an explicit untied head, but defensive coding
                # never hurt anyone.
                if "weight" not in head_state:
                    head_state["weight"] = tensor.clone()
                continue

            if hf_name.startswith("encoder."):
                encoder_pairs.append((hf_name, tensor))
                continue

            if hf_name.startswith("decoder."):
                mapped = _map_hf_decoder_key(hf_name)
                if mapped is not None:
                    decoder_state[mapped] = tensor
                continue

            if hf_name == "lm_head.weight":
                head_state["weight"] = tensor
                continue

        # Delegate to the MT5Encoder loader (it knows about the
        # vllm-factory key remap already).
        self._load_encoder_weights(encoder_pairs)

        # Decoder + LM head land via state_dict so pure-PyTorch nn.Linear
        # / nn.Embedding pick the tensors up natively.
        if decoder_state:
            missing, unexpected = self.decoder.load_state_dict(decoder_state, strict=False)
            logger.info(
                "[MiniCheckT5] Loaded decoder: %d tensors (missing=%d unexpected=%d)",
                len(decoder_state),
                len(missing),
                len(unexpected),
            )
            if unexpected:
                logger.debug("[MiniCheckT5] Unexpected decoder keys: %s", unexpected[:8])
        if head_state:
            self.lm_head.load_state_dict(head_state, strict=False)
            logger.info("[MiniCheckT5] Loaded lm_head")

        # Push everything to the configured device + dtype so the first
        # forward doesn't have to re-shuffle. The encoder owns its own
        # device placement via vLLM's parallel layers; the decoder and
        # LM head are pure-torch, so we pin them here.
        device = next(self.encoder.parameters()).device
        dtype = self.vllm_config.model_config.dtype
        self.decoder.to(device=device, dtype=dtype)
        self.lm_head.to(device=device, dtype=dtype)

        return set(name for name, _ in self.named_parameters())

    # ── Internal helpers ─────────────────────────────────────────────

    def _load_encoder_weights(
        self, encoder_pairs: list[Tuple[str, torch.Tensor]]
    ) -> None:
        """Apply the vllm-factory MT5 HF→internal key map and load."""

        from vllm.model_executor.model_loader.weight_utils import (  # noqa: WPS433
            default_weight_loader,
        )

        params = dict(self.encoder.named_parameters())
        for hf_name, tensor in encoder_pairs:
            mapped = _map_hf_encoder_key(hf_name)
            if mapped is None:
                continue
            param = params.get(mapped)
            if param is None:
                logger.debug(
                    "[MiniCheckT5] Encoder param missing for HF key %s -> %s",
                    hf_name,
                    mapped,
                )
                continue
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, tensor)


# ---------------------------------------------------------------------------
# HF → internal key remap (encoder + decoder)
# ---------------------------------------------------------------------------

import re  # placed late so the noqa scope above stays narrow

_ENC_ATTN_PROJ_MAP = {"q": "q_proj", "k": "k_proj", "v": "v_proj", "o": "out_proj"}
_ENC_BLOCK_ATTN_RE = re.compile(r"^encoder\.block\.(\d+)\.layer\.0\.SelfAttention\.(.+)$")
_ENC_BLOCK_ATTN_NORM_RE = re.compile(r"^encoder\.block\.(\d+)\.layer\.0\.layer_norm\.(.+)$")
_ENC_BLOCK_FF_RE = re.compile(r"^encoder\.block\.(\d+)\.layer\.1\.DenseReluDense\.(.+)$")
_ENC_BLOCK_FF_NORM_RE = re.compile(r"^encoder\.block\.(\d+)\.layer\.1\.layer_norm\.(.+)$")

_DEC_ATTN_PROJ_MAP = {"q": "q_proj", "k": "k_proj", "v": "v_proj", "o": "out_proj"}
_DEC_BLOCK_SELFATTN_RE = re.compile(
    r"^decoder\.block\.(\d+)\.layer\.0\.SelfAttention\.(.+)$"
)
_DEC_BLOCK_SELFATTN_NORM_RE = re.compile(
    r"^decoder\.block\.(\d+)\.layer\.0\.layer_norm\.(.+)$"
)
_DEC_BLOCK_CROSSATTN_RE = re.compile(
    r"^decoder\.block\.(\d+)\.layer\.1\.EncDecAttention\.(.+)$"
)
_DEC_BLOCK_CROSSATTN_NORM_RE = re.compile(
    r"^decoder\.block\.(\d+)\.layer\.1\.layer_norm\.(.+)$"
)
_DEC_BLOCK_FF_RE = re.compile(r"^decoder\.block\.(\d+)\.layer\.2\.DenseReluDense\.(.+)$")
_DEC_BLOCK_FF_NORM_RE = re.compile(r"^decoder\.block\.(\d+)\.layer\.2\.layer_norm\.(.+)$")


def _map_hf_encoder_key(hf_key: str) -> str | None:
    """Map HF T5 ``encoder.*`` keys to the vllm-factory MT5Encoder layout."""

    if hf_key in ("shared.weight", "encoder.embed_tokens.weight"):
        return "embed_tokens.weight"
    if hf_key.startswith("encoder.final_layer_norm."):
        return f"final_ln.{hf_key[len('encoder.final_layer_norm.'):]}"

    m = _ENC_BLOCK_ATTN_RE.match(hf_key)
    if m:
        idx, remainder = m.group(1), m.group(2)
        parts = remainder.split(".", 1)
        proj = parts[0]
        suffix = parts[1] if len(parts) > 1 else ""
        if proj == "relative_attention_bias":
            return (
                f"layers.{idx}.self_attn.rpb.relative_attention_bias.{suffix}"
                if suffix
                else None
            )
        if proj in _ENC_ATTN_PROJ_MAP:
            mapped = _ENC_ATTN_PROJ_MAP[proj]
            return (
                f"layers.{idx}.self_attn.{mapped}.{suffix}"
                if suffix
                else f"layers.{idx}.self_attn.{mapped}"
            )
        return None

    m = _ENC_BLOCK_ATTN_NORM_RE.match(hf_key)
    if m:
        return f"layers.{m.group(1)}.ln1.{m.group(2)}"

    m = _ENC_BLOCK_FF_RE.match(hf_key)
    if m:
        return f"layers.{m.group(1)}.ff.{m.group(2)}"

    m = _ENC_BLOCK_FF_NORM_RE.match(hf_key)
    if m:
        return f"layers.{m.group(1)}.ln2.{m.group(2)}"

    return None


def _map_hf_decoder_key(hf_key: str) -> str | None:
    """Map HF T5 ``decoder.*`` keys to our :class:`T5SingleStepDecoder`."""

    if hf_key == "decoder.embed_tokens.weight":
        return "embed_tokens.weight"
    if hf_key.startswith("decoder.final_layer_norm."):
        return f"final_ln.{hf_key[len('decoder.final_layer_norm.'):]}"

    m = _DEC_BLOCK_SELFATTN_RE.match(hf_key)
    if m:
        idx, remainder = m.group(1), m.group(2)
        parts = remainder.split(".", 1)
        proj = parts[0]
        suffix = parts[1] if len(parts) > 1 else ""
        if proj == "relative_attention_bias":
            return (
                f"layers.{idx}.self_attn.rpb.relative_attention_bias.{suffix}"
                if suffix
                else None
            )
        if proj in _DEC_ATTN_PROJ_MAP:
            mapped = _DEC_ATTN_PROJ_MAP[proj]
            return (
                f"layers.{idx}.self_attn.{mapped}.{suffix}"
                if suffix
                else f"layers.{idx}.self_attn.{mapped}"
            )
        return None

    m = _DEC_BLOCK_SELFATTN_NORM_RE.match(hf_key)
    if m:
        return f"layers.{m.group(1)}.ln1.{m.group(2)}"

    m = _DEC_BLOCK_CROSSATTN_RE.match(hf_key)
    if m:
        idx, remainder = m.group(1), m.group(2)
        parts = remainder.split(".", 1)
        proj = parts[0]
        suffix = parts[1] if len(parts) > 1 else ""
        if proj in _DEC_ATTN_PROJ_MAP:
            mapped = _DEC_ATTN_PROJ_MAP[proj]
            return (
                f"layers.{idx}.cross_attn.{mapped}.{suffix}"
                if suffix
                else f"layers.{idx}.cross_attn.{mapped}"
            )
        return None

    m = _DEC_BLOCK_CROSSATTN_NORM_RE.match(hf_key)
    if m:
        return f"layers.{m.group(1)}.ln2.{m.group(2)}"

    m = _DEC_BLOCK_FF_RE.match(hf_key)
    if m:
        return f"layers.{m.group(1)}.ff.{m.group(2)}"

    m = _DEC_BLOCK_FF_NORM_RE.match(hf_key)
    if m:
        return f"layers.{m.group(1)}.ln3.{m.group(2)}"

    return None


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _coerce_minicheck_config(hf_config: object) -> MiniCheckT5Config:
    """Normalise either the plugin config or a stock HF MT5/T5 config.

    When the plugin is selected via the architecture alias (the
    upstream ``T5ForConditionalGeneration`` / ``MT5ForConditionalGeneration``
    string) vLLM hands us the checkpoint's native config — we synthesise
    the plugin fields from it so the rest of the model never branches on
    the source type.
    """

    if isinstance(hf_config, MiniCheckT5Config):
        return hf_config

    return MiniCheckT5Config(
        vocab_size=int(getattr(hf_config, "vocab_size", 32128)),
        d_model=int(getattr(hf_config, "d_model", 1024)),
        d_kv=int(getattr(hf_config, "d_kv", 64)),
        d_ff=int(getattr(hf_config, "d_ff", 2816)),
        encoder_num_layers=int(
            getattr(hf_config, "num_layers", None)
            or getattr(hf_config, "num_hidden_layers", 24)
        ),
        decoder_num_layers=int(
            getattr(hf_config, "num_decoder_layers", None)
            or getattr(hf_config, "num_layers", None)
            or 24
        ),
        num_heads=int(getattr(hf_config, "num_heads", 16)),
        relative_attention_num_buckets=int(
            getattr(hf_config, "relative_attention_num_buckets", 32)
        ),
        relative_attention_max_distance=int(
            getattr(hf_config, "relative_attention_max_distance", 128)
        ),
        dropout_rate=float(getattr(hf_config, "dropout_rate", 0.0)),
        layer_norm_epsilon=float(getattr(hf_config, "layer_norm_epsilon", 1e-6)),
        feed_forward_proj=str(
            getattr(hf_config, "feed_forward_proj", "gated-gelu")
        ),
        decoder_start_token_id=int(
            getattr(hf_config, "decoder_start_token_id", 0) or 0
        ),
        eos_token_id=int(getattr(hf_config, "eos_token_id", 1) or 1),
        pad_token_id=int(getattr(hf_config, "pad_token_id", 0) or 0),
        tie_word_embeddings=bool(getattr(hf_config, "tie_word_embeddings", False)),
        yes_token_id=getattr(hf_config, "yes_token_id", None),
        no_token_id=getattr(hf_config, "no_token_id", None),
    )


def _build_backbone_cfg(cfg: MiniCheckT5Config, *, num_layers: int) -> MT5Config:
    """Build a plain :class:`MT5Config` for the encoder/decoder stacks.

    ``MT5Encoder`` reads ``cfg.num_layers`` to know how many blocks to
    instantiate. Our plugin-level config keeps that at zero (so vLLM's
    cache planner stays happy), so we synthesise a minimal MT5Config
    here with the *real* layer count for the backbone constructor.
    """

    return MT5Config(
        vocab_size=cfg.vocab_size,
        d_model=cfg.d_model,
        d_kv=cfg.d_kv,
        d_ff=cfg.d_ff,
        num_layers=num_layers,
        num_decoder_layers=num_layers,
        num_heads=cfg.num_heads,
        relative_attention_num_buckets=cfg.relative_attention_num_buckets,
        relative_attention_max_distance=cfg.relative_attention_max_distance,
        dropout_rate=cfg.dropout_rate,
        layer_norm_epsilon=cfg.layer_norm_epsilon,
        feed_forward_proj=cfg.feed_forward_proj,
        is_encoder_decoder=True,
        use_cache=False,
        tie_word_embeddings=cfg.tie_word_embeddings,
        decoder_start_token_id=cfg.decoder_start_token_id,
        eos_token_id=cfg.eos_token_id,
        pad_token_id=cfg.pad_token_id,
    )


__all__ = ["MiniCheckT5Model"]
