#!/usr/bin/env bash
set -euo pipefail

cd /workspace/latence-trace

export PYTHONPATH="/workspace/latence-trace:${PYTHONPATH:-}"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN="${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

python - <<'PY'
from importlib.metadata import entry_points
from pathlib import Path

import torch
import vllm
import vllm._C  # fail fast at runtime if the extension and torch ABI disagree
import vllm_factory  # noqa: F401
import moderncolbert_batched  # noqa: F401
import latence_trace_nli_plugin  # noqa: F401
import qwen3_compression_plugin  # noqa: F401
import text_processing  # noqa: F401
import latence_trace  # noqa: F401

compression_assets = Path("/workspace/latence-trace/runpod/compression_model")
required_assets = {
    "tokenizer.json",
    "text_processing-0.1.0-cp38-abi3-manylinux_2_34_x86_64.whl",
}
missing_assets = sorted(name for name in required_assets if not (compression_assets / name).exists())
if missing_assets:
    raise SystemExit(f"missing compression runtime assets: {missing_assets}")

io_plugins = {ep.name for ep in entry_points(group="vllm.io_processor_plugins")}
required = {"moderncolbert_batched_io", "nli_mdeberta", "deberta_gliner_io"}
missing = required - io_plugins
if missing:
    raise SystemExit(f"missing vllm io_processor_plugins entry points: {sorted(missing)}")

general_plugins = {ep.name for ep in entry_points(group="vllm.general_plugins")}
missing_general = {"qwen3_compression"} - general_plugins
if missing_general:
    raise SystemExit(f"missing vllm general_plugins entry points: {sorted(missing_general)}")

print(f"runtime smoke ok: torch={torch.__version__} cuda={torch.version.cuda} vllm={vllm.__version__}")
PY

exec python -m uvicorn runpod.api_server:app --host 0.0.0.0 --port 8000 --workers 1
