#!/usr/bin/env bash
set -euo pipefail

cd /workspace/latence-trace

export PYTHONPATH="/workspace/latence-trace:${PYTHONPATH:-}"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN="${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

python - <<'PY'
from importlib.metadata import entry_points

import torch
import vllm
import vllm._C  # fail fast at runtime if the extension and torch ABI disagree
import vllm_factory  # noqa: F401
import moderncolbert_batched  # noqa: F401
import latence_trace_nli_plugin  # noqa: F401
import latence_trace  # noqa: F401

io_plugins = {ep.name for ep in entry_points(group="vllm.io_processor_plugins")}
required = {"moderncolbert_batched_io", "nli_mdeberta"}
missing = required - io_plugins
if missing:
    raise SystemExit(f"missing vllm io_processor_plugins entry points: {sorted(missing)}")

print(f"runtime smoke ok: torch={torch.__version__} cuda={torch.version.cuda} vllm={vllm.__version__}")
PY

exec python -u runpod/handler.py
