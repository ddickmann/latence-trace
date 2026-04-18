# syntax=docker/dockerfile:1.7
##
## latence-trace -- Groundedness Tracker (commercial sidecar).
##
## Multi-stage, distroless-friendly image hardened for Fortune-500 ops:
##   - non-root user (uid/gid 65532) and read-only root filesystem support
##   - no shells, no package managers and no compilers in the runtime layer
##   - OCI image annotations + SPDX license identifier baked in
##   - reproducible BuildKit cache mounts so CI builds are deterministic
##   - HEALTHCHECK gated on /readyz so orchestrators never see warming
##     workers as healthy traffic targets
##
## The runtime container intentionally contains *only* the Python wheel
## tree and the certificate bundle. The vLLM-Factory sidecar in
## docker-compose.yaml owns the ColBERT encoder; the NLI / reranker
## stack still loads in-process via HF transformers (the runtime user
## needs HOME=/home/app for the transformers cache).
##
## Build:
##   docker buildx build --platform linux/amd64 \
##     -t ghcr.io/latence-ai/latence-trace:1.0.0 \
##     --build-arg VCS_REF=$(git rev-parse --short HEAD) \
##     --build-arg BUILD_DATE=$(date -u +%FT%TZ) .
##
## Generate an SPDX SBOM:
##   syft packages docker:ghcr.io/latence-ai/latence-trace:1.0.0 -o spdx-json > sbom.spdx.json
##

ARG PYTHON_VERSION=3.11
ARG VCS_REF=local
ARG BUILD_DATE=1970-01-01T00:00:00Z
ARG IMAGE_VERSION=1.0.0

# ----- builder ---------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build

# build-essential / git are scoped to this stage only; the runtime
# stage has no compiler or package manager footprint.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY latence_trace ./latence_trace
COPY server ./server

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip wheel \
    && /opt/venv/bin/pip install .

# ----- runtime ---------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS runtime

ARG VCS_REF
ARG BUILD_DATE
ARG IMAGE_VERSION

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    HOME=/home/app \
    LATENCE_TRACE_HOST=0.0.0.0 \
    LATENCE_TRACE_PORT=8090 \
    LATENCE_TRACE_PROFILE=balanced \
    LATENCE_TRACE_LOG_FORMAT=json \
    LATENCE_TRACE_LICENSE_REQUIRE=true \
    HF_HOME=/home/app/.cache/huggingface \
    TORCH_HOME=/home/app/.cache/torch \
    XDG_CACHE_HOME=/home/app/.cache \
    PIP_NO_CACHE_DIR=1

# Default to the vLLM-Factory sidecar host that docker-compose.yaml
# defines. Operators that run the container standalone unset
# VOYAGER_GROUNDEDNESS_VLLM_ENDPOINT to fall back to the in-process
# pylate provider.
ENV VOYAGER_GROUNDEDNESS_VLLM_ENDPOINT=http://vllm-factory:8000 \
    VOYAGER_GROUNDEDNESS_VLLM_MODEL=VAGOsolutions/SauerkrautLM-Multi-Reason-ModernColBERT

# Runtime base: only ca-certificates and tini for proper signal
# handling. No curl, no shells inside critical paths -- the readiness
# check uses Python so we stay shell-less.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        tini \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/* \
    && groupadd --system --gid 65532 app \
    && useradd --system --uid 65532 --gid 65532 \
        --home-dir /home/app --create-home --shell /sbin/nologin app

COPY --from=builder --chown=65532:65532 /opt/venv /opt/venv

WORKDIR /app
COPY --chown=65532:65532 pyproject.toml README.md LICENSE ./
COPY --chown=65532:65532 latence_trace ./latence_trace
COPY --chown=65532:65532 server ./server

# Pre-create the writable cache directory so a read-only root fs
# deployment can mount /home/app/.cache as an emptyDir or PVC.
RUN mkdir -p /home/app/.cache /etc/latence-trace \
    && chown -R 65532:65532 /home/app /etc/latence-trace

USER 65532:65532

EXPOSE 8090

# Liveness vs readiness:
#   /healthz returns 200 immediately as long as the process is alive.
#   /readyz  returns 200 only after Triton kernel warmup completes
#     AND the license is valid (when enforcement is enabled).
# Implemented in pure Python so the container has no curl dependency.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import os, sys, urllib.request as u; \
sys.exit(0 if u.urlopen(f'http://127.0.0.1:{os.environ.get(\"LATENCE_TRACE_PORT\", \"8090\")}/readyz', timeout=4).status == 200 else 1)"]

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-m", "server.main"]

# ----- OCI / SPDX annotations ------------------------------------------------
LABEL org.opencontainers.image.title="latence-trace" \
      org.opencontainers.image.description="Calibrated, auditable groundedness scoring for RAG and evidence-bearing LLM outputs." \
      org.opencontainers.image.vendor="latence.ai" \
      org.opencontainers.image.url="https://latence.ai/trace" \
      org.opencontainers.image.documentation="https://latence.ai/trace/docs" \
      org.opencontainers.image.source="https://github.com/latence-ai/latence-trace" \
      org.opencontainers.image.version="${IMAGE_VERSION}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.licenses="LicenseRef-latence-ai-Commercial" \
      org.opencontainers.image.authors="latence.ai <support@latence.ai>" \
      org.opencontainers.image.ref.name="latence-trace" \
      org.opencontainers.image.base.name="docker.io/library/python:3.11-slim"
