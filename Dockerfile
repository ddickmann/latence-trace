## Multi-stage build for the latence-trace FastAPI app.
##
## The app container does NOT load any LLM model in-process by default -
## the vLLM-Factory sidecar in docker-compose.yaml owns the ColBERT
## encoder and exposes it via HTTP at the BYOP pooler endpoint. The app
## container only carries the scoring math, the NLI/reranker stack
## (HF transformers), and the FastAPI surface, so it stays small.
##
## Build:
##   docker build -t ghcr.io/ddickmann/latence-trace:dev .
##
## Run standalone (in-process pylate fallback):
##   docker run --rm -p 8090:8090 ghcr.io/ddickmann/latence-trace:dev
##
## Run with vLLM-Factory sidecar (recommended, see docker-compose.yaml):
##   docker compose up

# ----- builder ---------------------------------------------------------------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY latence_trace ./latence_trace
COPY server ./server

RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip wheel \
    && /opt/venv/bin/pip install .

# ----- runtime ---------------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    LATENCE_TRACE_HOST=0.0.0.0 \
    LATENCE_TRACE_PORT=8090 \
    LATENCE_TRACE_PROFILE=balanced

# Default to the vLLM-Factory sidecar host that docker-compose.yaml
# defines. Operators that run the container standalone can unset
# VOYAGER_GROUNDEDNESS_VLLM_ENDPOINT to fall back to the in-process
# pylate provider.
ENV VOYAGER_GROUNDEDNESS_VLLM_ENDPOINT=http://vllm-factory:8000 \
    VOYAGER_GROUNDEDNESS_VLLM_MODEL=VAGOsolutions/SauerkrautLM-Multi-Reason-ModernColBERT

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash app

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=app:app pyproject.toml README.md LICENSE ./
COPY --chown=app:app latence_trace ./latence_trace
COPY --chown=app:app server ./server

USER app

EXPOSE 8090

# Liveness vs readiness:
#   /healthz returns 200 immediately as long as the process is alive
#   /readyz  returns 200 only after Triton kernel warmup completes
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl --fail --silent http://localhost:${LATENCE_TRACE_PORT}/readyz || exit 1

CMD ["python", "-m", "server.main"]
