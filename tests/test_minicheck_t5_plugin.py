"""Unit tests for the ``minicheck_t5`` BYOP plugin (no GPU, no vLLM).

We can't exercise the full ``MiniCheckT5Model.forward`` here without
booting a vLLM server on a GPU. What we *can* test \u2014 and what catches
the highest-value regression risks \u2014 is:

* the HuggingFace \u2192 internal weight-key remap (one wrong regex anywhere
  silently breaks 24 layers of weight loading);
* the IO processor's pre/post-process logic (request parsing, prompt
  template, label mapping, out-of-order request handling);
* the :class:`MiniCheckPooler` softmax shape (one-element-per-sequence
  output, fp32 stability);
* the plugin config sandbagging :attr:`num_hidden_layers` to 0 so vLLM
  doesn't try to allocate KV cache for our hand-rolled stack.

Together with the existing ``tests/test_runpod_handler.py`` coverage of
``_build_servers`` + ``_minicheck_plugin_available`` (extended in this
file), these protect the two hottest "didn't realise it broke" paths.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator
from unittest import mock

import pytest


# Make the plugin importable without ``pip install`` (the Dockerfile
# does the install in production; locally we just want to run pytest
# against the source tree). The plugin folder is a top-level package
# named ``minicheck_t5`` \u2014 same convention as ``moderncolbert_batched``.
_PLUGIN_PARENT = (
    Path(__file__).resolve().parents[1] / "runpod" / "vllm_plugins"
)
if str(_PLUGIN_PARENT) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_PARENT))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_config_sandbags_num_hidden_layers_to_zero() -> None:
    """vLLM peeks at ``num_hidden_layers`` for KV-cache planning.

    Our model owns the encoder + decoder layers itself, so we must
    return zero from every layer-count slot the base ``MT5Config``
    exposes. Otherwise vLLM tries to allocate KV blocks for 24 + 24
    phantom layers and the pooling runner crashes.
    """

    from minicheck_t5.config import MiniCheckT5Config

    cfg = MiniCheckT5Config()
    assert cfg.num_hidden_layers == 0
    assert cfg.num_layers == 0
    assert cfg.num_decoder_layers == 0
    # The real layer counts must be intact under the plugin-specific
    # field names so :class:`MiniCheckT5Model.__init__` can build the
    # right number of blocks.
    assert cfg.encoder_num_layers == 24
    assert cfg.decoder_num_layers == 24
    assert cfg.model_type == "minicheck_t5"
    assert cfg.is_gated_act is True  # Flan-T5 uses gated-gelu.


def test_config_constructor_strips_caller_supplied_layer_counts() -> None:
    """A caller passing ``num_hidden_layers=24`` must NOT leak through.

    HuggingFace's ``AutoConfig.from_pretrained`` will pass the
    checkpoint's full set of fields through ``__init__``. We strip the
    layer-count kwargs at the top of the constructor so the sandbag
    from the previous test holds even when someone sets them
    explicitly.
    """

    from minicheck_t5.config import MiniCheckT5Config

    cfg = MiniCheckT5Config(
        num_layers=24,
        num_decoder_layers=24,
        num_hidden_layers=24,
        encoder_num_layers=12,
        decoder_num_layers=12,
    )
    assert cfg.num_hidden_layers == 0
    assert cfg.num_layers == 0
    assert cfg.num_decoder_layers == 0
    # Plugin-specific kwargs are honoured.
    assert cfg.encoder_num_layers == 12
    assert cfg.decoder_num_layers == 12


# ---------------------------------------------------------------------------
# HF \u2192 internal key remap
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def keymaps() -> tuple:
    """Return ``(_map_hf_encoder_key, _map_hf_decoder_key)`` lazily.

    Importing :mod:`minicheck_t5.model` pulls in vLLM and
    ``vllm-factory``. To keep the fast unit tests independent of that
    heavy stack, we import the module inside the fixture and skip the
    whole keymap test class if the import fails (e.g. on a CI image
    without vLLM installed).
    """

    pytest.importorskip("vllm")
    pytest.importorskip("vllm_factory")
    model = importlib.import_module("minicheck_t5.model")
    return model._map_hf_encoder_key, model._map_hf_decoder_key


class TestEncoderKeyMap:
    """Cover every HF ``encoder.*`` key shape the plugin expects."""

    def test_shared_embedding(self, keymaps) -> None:
        enc, _ = keymaps
        assert enc("shared.weight") == "embed_tokens.weight"

    def test_encoder_embed_tokens(self, keymaps) -> None:
        enc, _ = keymaps
        assert enc("encoder.embed_tokens.weight") == "embed_tokens.weight"

    def test_encoder_final_layer_norm(self, keymaps) -> None:
        enc, _ = keymaps
        assert enc("encoder.final_layer_norm.weight") == "final_ln.weight"

    @pytest.mark.parametrize(
        "hf_key,internal_key",
        [
            (
                "encoder.block.0.layer.0.SelfAttention.q.weight",
                "layers.0.self_attn.q_proj.weight",
            ),
            (
                "encoder.block.7.layer.0.SelfAttention.k.weight",
                "layers.7.self_attn.k_proj.weight",
            ),
            (
                "encoder.block.23.layer.0.SelfAttention.v.weight",
                "layers.23.self_attn.v_proj.weight",
            ),
            (
                "encoder.block.4.layer.0.SelfAttention.o.weight",
                "layers.4.self_attn.out_proj.weight",
            ),
        ],
    )
    def test_encoder_attention_projections(
        self, keymaps, hf_key: str, internal_key: str
    ) -> None:
        enc, _ = keymaps
        assert enc(hf_key) == internal_key

    def test_encoder_relative_attention_bias(self, keymaps) -> None:
        enc, _ = keymaps
        # Only layer 0 in vllm-factory MT5Encoder owns the RPB table.
        # The remap still translates the per-layer key; the encoder's
        # layer constructor decides whether to actually load it.
        assert (
            enc("encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight")
            == "layers.0.self_attn.rpb.relative_attention_bias.weight"
        )

    def test_encoder_layer_norms(self, keymaps) -> None:
        enc, _ = keymaps
        assert enc("encoder.block.3.layer.0.layer_norm.weight") == "layers.3.ln1.weight"
        assert enc("encoder.block.3.layer.1.layer_norm.weight") == "layers.3.ln2.weight"

    @pytest.mark.parametrize(
        "hf_key,internal_key",
        [
            (
                "encoder.block.5.layer.1.DenseReluDense.wi_0.weight",
                "layers.5.ff.wi_0.weight",
            ),
            (
                "encoder.block.5.layer.1.DenseReluDense.wi_1.weight",
                "layers.5.ff.wi_1.weight",
            ),
            (
                "encoder.block.5.layer.1.DenseReluDense.wo.weight",
                "layers.5.ff.wo.weight",
            ),
        ],
    )
    def test_encoder_feed_forward(
        self, keymaps, hf_key: str, internal_key: str
    ) -> None:
        enc, _ = keymaps
        assert enc(hf_key) == internal_key

    def test_decoder_key_returns_none(self, keymaps) -> None:
        """Decoder keys must NOT match the encoder map \u2014 they live on
        :class:`T5SingleStepDecoder` instead and are loaded via a
        separate state_dict path."""

        enc, _ = keymaps
        assert (
            enc("decoder.block.0.layer.0.SelfAttention.q.weight") is None
        )
        assert enc("lm_head.weight") is None


class TestDecoderKeyMap:
    """Cover every HF ``decoder.*`` key shape the plugin expects."""

    def test_decoder_embed_tokens(self, keymaps) -> None:
        _, dec = keymaps
        assert dec("decoder.embed_tokens.weight") == "embed_tokens.weight"

    def test_decoder_final_layer_norm(self, keymaps) -> None:
        _, dec = keymaps
        assert dec("decoder.final_layer_norm.weight") == "final_ln.weight"

    @pytest.mark.parametrize(
        "hf_key,internal_key",
        [
            (
                "decoder.block.0.layer.0.SelfAttention.q.weight",
                "layers.0.self_attn.q_proj.weight",
            ),
            (
                "decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight",
                "layers.0.self_attn.rpb.relative_attention_bias.weight",
            ),
            (
                "decoder.block.5.layer.0.SelfAttention.o.weight",
                "layers.5.self_attn.out_proj.weight",
            ),
        ],
    )
    def test_decoder_self_attention(self, keymaps, hf_key, internal_key) -> None:
        _, dec = keymaps
        assert dec(hf_key) == internal_key

    @pytest.mark.parametrize(
        "hf_key,internal_key",
        [
            (
                "decoder.block.0.layer.1.EncDecAttention.q.weight",
                "layers.0.cross_attn.q_proj.weight",
            ),
            (
                "decoder.block.7.layer.1.EncDecAttention.k.weight",
                "layers.7.cross_attn.k_proj.weight",
            ),
            (
                "decoder.block.23.layer.1.EncDecAttention.v.weight",
                "layers.23.cross_attn.v_proj.weight",
            ),
            (
                "decoder.block.4.layer.1.EncDecAttention.o.weight",
                "layers.4.cross_attn.out_proj.weight",
            ),
        ],
    )
    def test_decoder_cross_attention(self, keymaps, hf_key, internal_key) -> None:
        _, dec = keymaps
        assert dec(hf_key) == internal_key

    def test_decoder_layer_norms(self, keymaps) -> None:
        _, dec = keymaps
        # In HF T5 decoder layer ordering: layer.0 = self-attn,
        # layer.1 = cross-attn, layer.2 = FF. We map their layer_norms
        # to ln1, ln2 and ln3 respectively.
        assert dec("decoder.block.3.layer.0.layer_norm.weight") == "layers.3.ln1.weight"
        assert dec("decoder.block.3.layer.1.layer_norm.weight") == "layers.3.ln2.weight"
        assert dec("decoder.block.3.layer.2.layer_norm.weight") == "layers.3.ln3.weight"

    def test_decoder_feed_forward(self, keymaps) -> None:
        _, dec = keymaps
        assert (
            dec("decoder.block.6.layer.2.DenseReluDense.wi_0.weight")
            == "layers.6.ff.wi_0.weight"
        )
        assert (
            dec("decoder.block.6.layer.2.DenseReluDense.wo.weight")
            == "layers.6.ff.wo.weight"
        )


# ---------------------------------------------------------------------------
# IO processor (pure-python pieces \u2014 no vLLM/torch GPU needed)
# ---------------------------------------------------------------------------


class _FakeBatchEncoding(dict):
    """Mimic HuggingFace's ``BatchEncoding`` (dict + attribute access).

    The real :class:`transformers.BatchEncoding` is a ``UserDict`` so
    both ``encoded["input_ids"]`` and ``encoded.input_ids`` work. We
    only need that subset for the IO processor tests.
    """

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc


class _FakeTokenizer:
    """Minimal tokenizer stub for the IO processor unit tests.

    Behaviour mirrors a SentencePiece-backed ``AutoTokenizer`` enough to
    exercise the ``factory_pre_process`` path: it emits one integer per
    whitespace-separated token, supports ``add_special_tokens=False``
    for Yes/No resolution, and supports ``decode`` for the claim-cap
    path. The Yes / No vocab IDs are fixed so the test can assert on
    them.
    """

    def __init__(self) -> None:
        self.vocab = {"Yes": 7, "No": 11}
        self._next_id = 100

    def __call__(
        self,
        texts,
        add_special_tokens: bool = True,
        truncation: bool = True,
        max_length: int = 1024,
        padding=False,
        return_tensors=None,
    ):
        # Match the real HF tokenizer's branching: a single string input
        # yields a flat ``input_ids`` list; a list of strings yields a
        # list-of-lists. The IO processor relies on this distinction
        # (single-string ``"Yes"`` → ``[7]``, batched formatted prompts
        # → ``[[...], [...]]``).
        if isinstance(texts, str):
            ids = self._encode(texts)
            if truncation and len(ids) > max_length:
                ids = ids[:max_length]
            return _FakeBatchEncoding({"input_ids": ids})

        rows = []
        for text in texts:
            ids = self._encode(text)
            if truncation and len(ids) > max_length:
                ids = ids[:max_length]
            rows.append(ids)
        return _FakeBatchEncoding({"input_ids": rows})

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return self._encode(text)

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return " ".join(f"<{i}>" for i in ids)

    def _encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for tok in text.split():
            if tok in self.vocab:
                ids.append(self.vocab[tok])
            else:
                ids.append(self._next_id)
                self._next_id += 1
        return ids


@pytest.fixture
def io_processor() -> Iterator:
    """Build the IO processor with a fake tokenizer, no vLLM init."""

    pytest.importorskip("vllm")
    pytest.importorskip("vllm_factory")
    io_module = importlib.import_module("minicheck_t5.io_processor")

    fake_cfg = SimpleNamespace(
        model="lytang/MiniCheck-Flan-T5-Large",
        max_model_len=1024,
        hf_config=SimpleNamespace(
            yes_token_id=None,
            no_token_id=None,
            pad_token_id=0,
        ),
    )
    fake_vllm_config = SimpleNamespace(model_config=fake_cfg)

    with mock.patch.object(
        io_module.AutoTokenizer, "from_pretrained", return_value=_FakeTokenizer()
    ):
        with mock.patch.object(
            io_module.FactoryIOProcessor,
            "__init__",
            lambda self, *a, **kw: None,
        ):
            proc = io_module.MiniCheckT5IOProcessor(fake_vllm_config)
            # FactoryIOProcessor.__init__ was no-op'd above, so wire the
            # bare-minimum stash plumbing the post-process step expects.
            proc._extra_lock = None  # type: ignore[attr-defined]

            def _stash(*args, **kwargs):  # noqa: ARG001
                return None

            proc._stash = _stash  # type: ignore[attr-defined]
            yield proc


def test_io_resolves_yes_no_token_ids(io_processor) -> None:
    """Yes / No token IDs are resolved from the tokenizer at startup."""

    assert io_processor._yes_token_id == 7
    assert io_processor._no_token_id == 11


def test_io_factory_parse_accepts_aliases(io_processor) -> None:
    """``premise`` / ``premises`` / ``context`` should all parse."""

    parsed = io_processor.factory_parse(
        {"data": {"context": "Premise A.", "claim": "Hypothesis A."}}
    )
    assert parsed.premises == ["Premise A."]
    assert parsed.hypotheses == ["Hypothesis A."]


def test_io_factory_parse_accepts_batched_lists(io_processor) -> None:
    parsed = io_processor.factory_parse(
        {
            "premise": ["P1", "P2"],
            "hypothesis": ["H1", "H2"],
        }
    )
    assert parsed.premises == ["P1", "P2"]
    assert parsed.hypotheses == ["H1", "H2"]


def test_io_factory_parse_rejects_misaligned_batch(io_processor) -> None:
    with pytest.raises(ValueError, match="must align"):
        io_processor.factory_parse({"premise": ["a", "b"], "hypothesis": ["c"]})


def test_io_factory_parse_rejects_empty_batch(io_processor) -> None:
    """Empty lists are falsy so they fall through the ``or`` chain in
    ``factory_parse`` and trigger the "must contain" guard. This
    matches :class:`NLIDebertaV2IOProcessor` exactly so an operator
    sending the wrong shape gets a consistent error from either backend.
    """

    with pytest.raises(ValueError, match="must contain 'premise' and 'hypothesis'"):
        io_processor.factory_parse({"premise": [], "hypothesis": []})


def test_io_format_one_template(io_processor) -> None:
    """The MiniCheck prompt template is ``predict: {premise}\\nclaim: {claim}``.

    Bit-identical to :meth:`MiniCheckNLIProvider._format_one` so the
    HTTP-served and in-process backends agree on the formatted string
    for the same ``(premise, claim)`` pair.
    """

    formatted = io_processor._format_one("Apple beat estimates.", "Apple beat estimates.")
    assert formatted == "predict: Apple beat estimates.\nclaim: Apple beat estimates."


def test_io_format_one_caps_long_claim(io_processor) -> None:
    """Long claims must be truncated to ``max_claim_tokens`` first."""

    io_processor._max_claim_tokens = 4
    long_claim = "tok " * 20
    formatted = io_processor._format_one("short premise", long_claim.strip())
    # Our fake tokenizer emits the claim as ``<id> <id> <id> <id>`` after
    # decode; just check we shrunk it to <= 4 tokens.
    claim_section = formatted.split("claim: ", 1)[1]
    assert len(claim_section.split()) <= 4


def test_io_factory_post_process_applies_softmax_logic(io_processor) -> None:
    """Post-process turns pooler probs into the wire-shape triple.

    The pooler returns a 2-vector ``(p_yes, p_no)`` that already sums
    to ~1 (it softmaxed). The IO processor maps this onto the same
    ``(entail, neutral, contradict)`` triple shape ``nli_mdeberta``
    returns, but with ``neutral=0`` and ``contradict=1-entail``.
    """

    import torch

    fake_outputs = [
        SimpleNamespace(
            request_id="abc-0",
            outputs=SimpleNamespace(data=torch.tensor([0.84, 0.16])),
        ),
        SimpleNamespace(
            request_id="abc-1",
            outputs=SimpleNamespace(data=torch.tensor([0.10, 0.90])),
        ),
    ]
    out = io_processor.factory_post_process(fake_outputs, request_meta=None)
    assert out["data"][0]["entail"] == pytest.approx(0.84, rel=1e-3)
    assert out["data"][0]["contradict"] == pytest.approx(0.16, rel=1e-3)
    assert out["data"][0]["neutral"] == 0.0
    assert out["data"][0]["label"] == "entailment"
    assert out["data"][1]["entail"] == pytest.approx(0.10, rel=1e-3)
    assert out["data"][1]["contradict"] == pytest.approx(0.90, rel=1e-3)
    assert out["data"][1]["label"] == "contradiction"


def test_io_factory_post_process_restores_request_order(io_processor) -> None:
    """vLLM can return outputs out-of-order; the suffix must rescue us."""

    import torch

    out_of_order = [
        SimpleNamespace(
            request_id="abc-1",
            outputs=SimpleNamespace(data=torch.tensor([0.1, 0.9])),
        ),
        SimpleNamespace(
            request_id="abc-0",
            outputs=SimpleNamespace(data=torch.tensor([0.9, 0.1])),
        ),
    ]
    out = io_processor.factory_post_process(out_of_order, request_meta=None)
    assert out["data"][0]["entail"] == pytest.approx(0.9, rel=1e-3)
    assert out["data"][1]["entail"] == pytest.approx(0.1, rel=1e-3)


def test_io_factory_post_process_handles_empty_output(io_processor) -> None:
    out = io_processor.factory_post_process([], request_meta=None)
    assert out == {"data": []}


def test_io_factory_post_process_handles_missing_data(io_processor) -> None:
    """If the pooler returned ``None`` we must not crash; emit the
    most-conservative answer (full contradict)."""

    fake_outputs = [
        SimpleNamespace(
            request_id="abc-0",
            outputs=SimpleNamespace(data=None),
        ),
    ]
    out = io_processor.factory_post_process(fake_outputs, request_meta=None)
    assert out["data"][0]["entail"] == 0.0
    assert out["data"][0]["contradict"] == 1.0
    assert out["data"][0]["label"] == "contradiction"


# ---------------------------------------------------------------------------
# Pooler shape contract
# ---------------------------------------------------------------------------


def test_pooler_emits_one_softmaxed_pair_per_sequence() -> None:
    """:class:`MiniCheckPooler.forward` must return one ``(2,)`` tensor
    per row, with the per-row entries summing to ~1 (softmax)."""

    pytest.importorskip("vllm_factory")
    import torch
    from minicheck_t5.pooler import MiniCheckPooler
    from vllm_factory.pooling.protocol import PoolerContext

    # Two sequences, lengths (5, 3) \u2014 the model packs (yes, no) into
    # position 0 of each slice. We simulate that by setting position 0
    # to the per-row logits and the rest to zero.
    seq_lengths = [5, 3]
    hidden = torch.zeros(sum(seq_lengths), 2)
    hidden[0] = torch.tensor([1.5, 0.5])     # row 0 strongly entail
    hidden[5] = torch.tensor([-2.0, 2.0])    # row 1 strongly contradict

    pooler = MiniCheckPooler()
    ctx = PoolerContext(seq_lengths=seq_lengths)
    out = pooler.forward(hidden, ctx)
    assert len(out) == 2
    for tensor in out:
        assert tensor.shape == (2,)
        assert pytest.approx(float(tensor.sum()), abs=1e-4) == 1.0
    # Row 0 prefers Yes; row 1 prefers No.
    assert out[0][0] > out[0][1]
    assert out[1][1] > out[1][0]


def test_pooler_supports_legacy_task_names() -> None:
    """``plugin`` is the canonical task name; ``classify`` and ``embed``
    should also work so operators with legacy clients keep functioning."""

    pytest.importorskip("vllm_factory")
    from minicheck_t5.pooler import MiniCheckPooler

    tasks = MiniCheckPooler().get_tasks()
    assert {"plugin", "classify", "embed"} <= tasks
