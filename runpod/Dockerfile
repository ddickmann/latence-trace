# syntax=docker/dockerfile:1.4

FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

ARG VLLM_FACTORY_REF=main

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    TOKENIZERS_PARALLELISM=false \
    VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

WORKDIR /workspace/latence-trace

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE /workspace/latence-trace/
COPY latence_trace /workspace/latence-trace/latence_trace
COPY runpod /workspace/latence-trace/runpod

RUN python -m pip install --upgrade pip setuptools wheel

# Runtime helpers that latence-trace's pyproject does not pull in.
RUN python -m pip install "runpod>=1.7.0" "hf_transfer>=0.1.4"

# Overlay a wheel install so pip does not try to uninstall Debian's cryptography.
RUN python -m pip install --ignore-installed "cryptography>=42.0"

# Install latence-trace and its full dependency closure FIRST. This pulls
# pylate -> sentence-transformers -> transformers / torch / accelerate /
# fast-plaid, etc. Doing this before vLLM means a later "pip install vllm"
# is the final word on torch / torchvision / torchaudio versions.
RUN python -m pip install /workspace/latence-trace

# spaCy models are needed by the atomic-claims splitter at runtime.
RUN python -m spacy download en_core_web_sm \
    && python -m spacy download de_core_news_sm

# Install vLLM LAST so its strict pin (torch==2.10.0, torchvision==0.25.0,
# torchaudio==2.10.0) overrides anything the latence-trace dependency closure
# resolved. This guarantees vllm/_C.abi3.so is ABI-compatible with the torch
# wheel actually present in /usr/local/lib/python3.12/dist-packages and
# avoids the classic
#   ImportError: vllm/_C.abi3.so: undefined symbol c10_cuda_check_implementation
# that appears when an older torch wheel shadows vLLM's compiled bindings.
RUN python -m pip install "vllm==0.19.0"

# vllm-factory and the BYOP plugins must be installed AFTER vllm so the
# importlib.metadata entry-point lookup (which vllm runs at server start)
# can resolve them against the installed vllm distribution. We install with
# --no-deps because every dep is already pinned by the steps above.
RUN git clone --depth 1 --branch "${VLLM_FACTORY_REF}" https://github.com/ddickmann/vllm-factory.git /opt/vllm-factory \
    && python -m pip install --no-deps /opt/vllm-factory \
    && rm -rf /opt/vllm-factory

RUN python -m pip install --no-deps /workspace/latence-trace/runpod/vllm_plugins/moderncolbert_batched
RUN python -m pip install --no-deps /workspace/latence-trace/runpod/vllm_plugins/nli_mdeberta

# Build-time smoke tests: validate Python imports and vLLM plugin
# registration. Docker builds do not have the NVIDIA driver mounted, so
# `vllm._C` can only be imported when `libcuda.so.1` is available.
RUN python - <<'PY'
from ctypes import CDLL
from importlib.metadata import entry_points

import torch
import vllm
import vllm_factory  # noqa: F401
import moderncolbert_batched  # noqa: F401
import latence_trace_nli_plugin  # noqa: F401
import latence_trace  # noqa: F401

try:
    CDLL("libcuda.so.1")
except OSError:
    print("build smoke note: skipping vllm._C import because libcuda.so.1 is unavailable")
else:
    import vllm._C  # noqa: F401

io_plugins = {ep.name for ep in entry_points(group="vllm.io_processor_plugins")}
required = {"moderncolbert_batched_io", "nli_mdeberta"}
missing = required - io_plugins
if missing:
    raise SystemExit(f"missing vllm io_processor_plugins entry points: {sorted(missing)}")

print(f"build smoke ok: torch={torch.__version__} cuda={torch.version.cuda} vllm={vllm.__version__}")
PY

RUN chmod +x /workspace/latence-trace/runpod/start.sh

CMD ["bash", "/workspace/latence-trace/runpod/start.sh"]
