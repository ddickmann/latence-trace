from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
PATCH_PATH = (
    ROOT
    / "runpod"
    / "vllm_plugins"
    / "nli_mdeberta"
    / "vllm_pooling_token_type_ids.py"
)
SPEC = importlib.util.spec_from_file_location("nli_vllm_patch", PATCH_PATH)
assert SPEC is not None and SPEC.loader is not None
patch_module = importlib.util.module_from_spec(SPEC)
sys.modules.setdefault(SPEC.name, patch_module)
SPEC.loader.exec_module(patch_module)


class _Param:
    def __init__(self, extra_kwargs):
        self.extra_kwargs = extra_kwargs


class _InputBatch:
    def __init__(self, params):
        self.num_reqs = len(params)
        self._params = params

    def get_pooling_params(self):
        return self._params


class _Runner:
    def __init__(self, params, seq_lens):
        self.is_pooling_model = True
        self.device = torch.device("cpu")
        self.input_batch = _InputBatch(params)
        self.seq_lens = seq_lens


def test_patched_init_model_kwargs_builds_expected_token_type_ids() -> None:
    def _orig(_self):
        return {"unchanged": True}

    patched = patch_module._make_patched_init_model_kwargs(_orig)
    runner = _Runner(
        [_Param({"compressed_token_type_ids": 2}), _Param(None)],
        [torch.tensor(5), torch.tensor(4)],
    )

    model_kwargs = patched(runner)

    assert "unchanged" not in model_kwargs
    assert model_kwargs["token_type_ids"].tolist() == [0, 0, 1, 1, 1, 0, 0, 0, 0]


def test_patched_init_model_kwargs_delegates_without_token_type_hints() -> None:
    sentinel = {"unchanged": True}

    def _orig(_self):
        return sentinel

    patched = patch_module._make_patched_init_model_kwargs(_orig)
    runner = _Runner([_Param(None)], [torch.tensor(3)])

    assert patched(runner) is sentinel
