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
RUN python -m pip install "vllm==0.19.0" "runpod>=1.7.0" "hf_transfer>=0.1.4"

RUN git clone --depth 1 --branch "${VLLM_FACTORY_REF}" https://github.com/ddickmann/vllm-factory.git /opt/vllm-factory \
    && python -m pip install --no-deps /opt/vllm-factory \
    && rm -rf /opt/vllm-factory

RUN python -m pip install --no-deps /workspace/latence-trace/runpod/vllm_plugins/moderncolbert_batched
RUN python -m pip install --no-deps /workspace/latence-trace/runpod/vllm_plugins/nli_mdeberta
RUN python -m pip install /workspace/latence-trace
RUN python -m spacy download en_core_web_sm \
    && python -m spacy download de_core_news_sm

RUN chmod +x /workspace/latence-trace/runpod/start.sh

CMD ["bash", "/workspace/latence-trace/runpod/start.sh"]
