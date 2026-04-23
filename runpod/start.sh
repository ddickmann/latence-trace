#!/usr/bin/env bash
set -euo pipefail

cd /workspace/latence-trace

export PYTHONPATH="/workspace/latence-trace:${PYTHONPATH:-}"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN="${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

exec python -u runpod/handler.py
