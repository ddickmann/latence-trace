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
    git \
    && rm -rf /var/lib/apt/lists/*

COPY . /workspace/latence-trace

RUN python -m pip install --upgrade pip setuptools wheel
RUN python -m pip install "vllm==0.19.0" "runpod>=1.7.0" "hf_transfer>=0.1.4"

RUN git clone https://github.com/ddickmann/vllm-factory.git /opt/vllm-factory \
    && cd /opt/vllm-factory \
    && git checkout "${VLLM_FACTORY_REF}" \
    && python -m pip install --no-deps .

RUN python -m pip install --no-deps /workspace/latence-trace/runpod/vllm_plugins/moderncolbert_batched
RUN python -m pip install --no-deps /workspace/latence-trace/runpod/vllm_plugins/nli_mdeberta
RUN python -m pip install /workspace/latence-trace
RUN python -m spacy download en_core_web_sm \
    && python -m spacy download de_core_news_sm

RUN chmod +x /workspace/latence-trace/runpod/start.sh

CMD ["bash", "/workspace/latence-trace/runpod/start.sh"]
