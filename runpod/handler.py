"""RunPod serverless entrypoint for latence-trace.

The worker exposes two scoring lanes on one endpoint:

- ``scoring_mode="rag"`` (default) — the long-standing enterprise RAG
  groundedness pipeline. Unchanged semantics, unchanged response shape.
- ``scoring_mode="code"`` — a code-aware lane built on top of the same
  ColBERT MaxSim backbone with AST drift, literal novelty, an
  ambiguity-gated NLI cascade, semantic entropy, a calibrated composite
  phantom score, and per-file / per-unit attribution. Designed for IDE
  plugins (Cursor, Claude Code, OpenAI Codex, OpenCode, aider) that
  want a real-time dashboard tracing the quality of their agent.

Concurrency model
-----------------
Each lane gets its own inflight semaphore so a burst of code-lane calls
never starves in-flight RAG calls (and vice versa). The total worker
concurrency is still bounded by ``max_concurrency``; the per-lane
budget defaults to ``ceil(max_concurrency / 2)`` each so one lane can
burst through the shared ceiling when the other is idle.

Warmup
------
Two warmup requests run at boot: one RAG request through the full
pipeline (primes ColBERT + NLI vLLM + null bank) and one code-lane
request (primes ``GPUScorer`` streams + pre-loads tree-sitter grammars
+ exercises the composite model). Boot is a no-op once the singletons
are primed.

Observability
-------------
One structured log line per request (``groundedness_turn``) captures
``{lane, session_id, cascade_fired, phantom_verdict, nli_ms, ast_ms,
composite_ms, total_ms}``. No PII is logged — only hashed session
labels.
"""

from __future__ import annotations

import asyncio
import atexit
import logging
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from pydantic import ValidationError as PydanticValidationError

from latence_trace import __version__
from latence_trace.api.compliance_models import (
    ComplianceRedactionRequest,
    ComplianceRedactionResponse,
)
from latence_trace.api.compliance_service import (
    ComplianceRedactionService,
    ComplianceServiceError,
)
from latence_trace.api.compression_models import CompressionRequest, CompressionResponse
from latence_trace.api.compression_service import CompressionService
from latence_trace.api.models import (
    GroundednessRequest,
    GroundednessResponse,
    RollupRequest,
    RollupResponse,
    ScoringMode,
)
from latence_trace.api.service import (
    GroundednessService,
    ServiceError,
    ValidationError,
    apply_profile,
)
from latence_trace.core.groundedness import warm_context_trust_runtime
from latence_trace.core.text_segmentation import warmup_segmenters
from latence_trace.kernels.warmup import warm_all, warm_code_lane
from latence_trace.memory.models import MemoryUpdateRequest
from latence_trace.memory.service import update_memory
from latence_trace.memory.signature import extract_exact_critical_terms
from latence_trace.observability.metrics import (
    BUDGET_EXCEEDED_COUNT,
    CASCADE_FIRE_COUNT,
    LANE_REQUEST_COUNT,
    PHANTOM_VERDICT_COUNT,
)
from latence_trace.sessions.models import (
    TraceSessionCreateRequest,
    TraceSessionEventRequest,
    TraceSessionRepairRequest,
    TraceSessionRollupRequest,
    TraceSessionScoreRequest,
)
from latence_trace.sessions.service import TraceSessionService
from server import ManagedVllmServer

try:  # pragma: no cover - optional in local dev
    import runpod
except Exception:  # pragma: no cover - local smoke tests import this module
    runpod = None

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=os.environ.get("LATENCE_TRACE_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _detect_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


_STARTUP_WARMUP_REQUEST_COUNT = 4
_DEFAULT_COMPRESSION_MODEL = "latence/compression-v0.1"
_RUNPOD_DIR = Path(__file__).resolve().parent
_VENDORED_COMPRESSION_MODEL_DIR = _RUNPOD_DIR / "compression_model"
SUPPORTED_RUNPOD_ACTIONS = frozenset(
    {
        "score",
        "",
        "rollup",
        "redact",
        "compliance_redaction",
        "compress",
        "compression",
        "memory.update",
        "memory_update",
        "session.create",
        "session.get",
        "session.event",
        "session.score",
        "session.context",
        "session.source",
        "session.repair",
        "session.rollup",
        "session.close",
    }
)


def _env_bool(name: str) -> bool | None:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return None
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


def _env_bool_with_default(name: str, default: bool) -> bool:
    """Like :func:`_env_bool` but returns ``default`` when the env var
    is unset or blank, instead of ``None``.

    Used for the SOTA NLI / reranker enabled flags which now default to
    ``True`` — the previous ``bool(_env_bool(...))`` pattern silently
    coerced ``None`` to ``False`` and made it impossible to express
    "default on, can be turned off via env=0".
    """

    raw = _env_bool(name)
    if raw is None:
        return default
    return raw


# ---------------------------------------------------------------------------
# GPU budget — single knob for the whole 7-model production topology.
# ---------------------------------------------------------------------------
# Production co-hosts GPU-resident models on one 24 GiB worker.
#
# **Guardian topology** (``LATENCE_TRACE_GUARDIAN_ENABLED=1``):
#   1. guardian            (Granite Guardian 4.1-8b W8A16)  vLLM server :8005  0.50
#   2. colbert             (lightonai/LateOn)               vLLM server :8001  0.13
#   3. compliance_gliner   (knowledgator/gliner-pii-v1.0)   vLLM server :8003  0.11
#   4. compression         (LLMLingua-2 Qwen3)              vLLM server :8004  0.11
#   5. prompt_guard        (meta-llama/Llama-Prompt-Guard-2-86M)
#                          in-process PyTorch                                   ~0
#   + Triton MaxSim kernel (in-process)                                        ~0
#   → 0.85 total vLLM utilisation → ~20.9 GiB, ~3 GiB headroom.
#
# **Legacy topology** (Guardian disabled):
#   1. colbert            (lightonai/LateOn)               vLLM server :8001
#   2. compliance_gliner  (knowledgator/gliner-pii-v1.0)   vLLM server :8003
#   3. compression        (LLMLingua-2 Qwen3)              vLLM server :8004
#   4. nli_en             (lytang/MiniCheck-Flan-T5-Large) vLLM server :8005
#   5. nli_multi          (MoritzLaurer/bge-m3-zs-v2.0)    vLLM server :8006
#   6. reranker           (BAAI/bge-reranker-v2-m3)        vLLM server :8007
#   7. prompt_guard       (meta-llama/Llama-Prompt-Guard-2-86M)
#   + Triton MaxSim kernel (in-process)
#   → 6 × 0.145 = 0.87 vLLM utilisation → ~20.9 GiB.
#
# Set ``LATENCE_TRACE_VLLM_GPU_MEM_DEFAULT=0.135`` (or lower) if vLLM
# crashes with OOM at boot — that knob shifts every per-server default
# in lock-step. Per-server overrides (e.g. ``LATENCE_TRACE_COLBERT_GPU_MEM``)
# still trump the shared default.
_DEFAULT_VLLM_GPU_MEM = _env_float("LATENCE_TRACE_VLLM_GPU_MEM_DEFAULT", 0.145)


def _local_compression_model_available(model_dir: Path) -> bool:
    if not model_dir.is_dir():
        return False
    if not (model_dir / "config.json").exists():
        return False
    weight_markers = (
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    )
    return any((model_dir / marker).exists() for marker in weight_markers)


def _resolve_compression_model() -> tuple[str, bool]:
    """Resolve the managed compression vLLM model.

    Production should not silently degrade to sentence fallback. The managed
    server is enabled by default and uses the SuperPod LLMLingua2 checkpoint
    unless operators explicitly point to a complete local model, configure an
    external compression endpoint, or opt out for local/dev.
    """
    explicit_model = (os.environ.get("LATENCE_TRACE_COMPRESSION_MODEL") or "").strip()
    explicit_endpoint = (os.environ.get("LATENCE_TRACE_COMPRESSION_ENDPOINT") or "").strip()
    enable_override = _env_bool("LATENCE_TRACE_ENABLE_COMPRESSION_SERVER")

    if explicit_endpoint:
        return explicit_model, False

    if enable_override is False:
        return explicit_model, False

    if explicit_model:
        return explicit_model, True

    if _local_compression_model_available(_VENDORED_COMPRESSION_MODEL_DIR):
        return str(_VENDORED_COMPRESSION_MODEL_DIR), True

    logger.warning(
        "No complete vendored compression model found at %s. Starting managed "
        "compression vLLM with the public production default %s; use "
        "LATENCE_TRACE_COMPRESSION_MODEL to override it.",
        _VENDORED_COMPRESSION_MODEL_DIR,
        _DEFAULT_COMPRESSION_MODEL,
    )
    return _DEFAULT_COMPRESSION_MODEL, True


@dataclass(frozen=True)
class WorkerConfig:
    profile: str
    version: str
    request_timeout_s: int
    code_request_timeout_s: float
    rollup_request_timeout_s: float
    max_concurrency: int
    collection_label: str
    service_device: str
    docs_url: str
    managed_vllm_enabled: bool
    colbert_model: str
    colbert_port: int
    colbert_gpu_mem: float
    colbert_max_model_len: int
    colbert_max_num_seqs: int
    colbert_max_batched_tokens: int
    nli_model: str
    nli_port: int
    nli_gpu_mem: float
    nli_max_model_len: int
    nli_max_num_seqs: int
    nli_max_batched_tokens: int
    compliance_model: str
    compliance_port: int
    compliance_gpu_mem: float
    compliance_max_model_len: int
    compliance_max_num_seqs: int
    compliance_max_batched_tokens: int
    compliance_threshold: float
    compliance_dataset_path: str
    compliance_request_timeout_s: float
    # PII / context_trust GLiNER server is on by default for production
    # parity. Operators (or memory-constrained calibration runs) can
    # flip ``LATENCE_TRACE_ENABLE_COMPLIANCE_GLINER_SERVER=0`` to skip
    # spawning it; the request-level ``context_trust_enabled`` flag
    # already gates whether the lane is invoked, so disabling the
    # server is safe as long as no caller asks for the scan.
    compliance_gliner_server_enabled: bool = True
    compression_model: str = ""
    compression_server_enabled: bool = False
    compression_port: int = 8004
    compression_gpu_mem: float = _DEFAULT_VLLM_GPU_MEM
    compression_max_model_len: int = 8192
    compression_max_num_seqs: int = 64
    compression_max_batched_tokens: int = 8192
    compression_dtype: str = "bfloat16"
    compression_enforce_eager: bool = True
    compression_trust_remote_code: bool = False
    compression_request_timeout_s: float = 30.0
    compression_default_chunk_size: int = 4096
    compression_default_compression_rate: float = 0.4
    compression_force_preserve_digit: bool = True
    compression_fallback_mode: bool = True
    # ----------------------------------------------------------------
    # SOTA dual-NLI + vLLM-served reranker — DEFAULT ON.
    #
    # Topology now boots six vLLM servers in lock-step at
    # ``_DEFAULT_VLLM_GPU_MEM=0.145`` each:
    #   colbert + compliance_gliner + compression + nli_en + nli_multi
    #   + reranker = 6 × 0.145 = 0.87 total utilisation, ~13% headroom.
    # The 7th model — Llama-Prompt-Guard-2-86M — boots in-process via
    # ``_warm_prompt_guard_at_boot`` (provider default = ``prompt_guard``)
    # and lives in the same CUDA context as the Triton MaxSim kernels.
    #
    # ``max_num_seqs`` is held at 64 across every vLLM server. The
    # generative-style compression server (Qwen3-0.6B with 8 K context)
    # would otherwise burn its KV-cache budget at ``max_num_seqs=128``
    # and crash the server with ``exited with code 1`` (KV cache cannot
    # fit a single max-length sequence). 64 keeps every server inside
    # 0.145 of a 24 GiB GPU while still saturating real production
    # batch sizes — the previous 128 cap was never reached under any
    # measured workload.
    #
    # Legacy single-NLI ``nli`` (mDeBERTa) server is dropped from the
    # topology — replaced by the dual SOTA path (MiniCheck for English,
    # bge-m3-zeroshot-v2.0 for German + multilingual). The in-process
    # ``HuggingFaceNLIProvider`` fallback in
    # :mod:`latence_trace.api.service` still loads mDeBERTa lazily as a
    # defensive safety net for the language=None lane, but the GPU no
    # longer carries it as a vLLM process.
    #
    # ``nli_en`` requires the BYOP MiniCheck plugin
    # (``runpod/vllm_plugins/minicheck_t5/``); when the plugin is
    # missing the server is skipped and per-language EN traffic falls
    # back to the in-process MiniCheckNLIProvider in
    # :mod:`latence_trace.providers.nli_transformers`.
    # ``nli_multi`` and ``reranker`` boot vLLM-native via
    # ``--convert classify``.
    # ----------------------------------------------------------------
    nli_en_enabled: bool = True
    nli_en_model: str = "lytang/MiniCheck-Flan-T5-Large"
    nli_en_port: int = 8005
    nli_en_gpu_mem: float = _DEFAULT_VLLM_GPU_MEM
    nli_en_max_model_len: int = 1024
    nli_en_max_num_seqs: int = 64
    nli_en_max_batched_tokens: int = 8192
    nli_multi_enabled: bool = True
    nli_multi_model: str = "MoritzLaurer/bge-m3-zeroshot-v2.0"
    nli_multi_port: int = 8006
    nli_multi_gpu_mem: float = _DEFAULT_VLLM_GPU_MEM
    nli_multi_max_model_len: int = 512
    nli_multi_max_num_seqs: int = 64
    nli_multi_max_batched_tokens: int = 8192
    reranker_enabled: bool = True
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_port: int = 8007
    reranker_gpu_mem: float = _DEFAULT_VLLM_GPU_MEM
    reranker_max_model_len: int = 512
    reranker_max_num_seqs: int = 64
    reranker_max_batched_tokens: int = 8192
    # ----------------------------------------------------------------
    # Granite Guardian — single multilingual groundedness LLM that
    # replaces nli_en + nli_multi + reranker when enabled.  Frees two
    # vLLM server slots and their GPU memory.
    #
    # W8A16 GPTQ quantization cuts the model from ~16 GB (BF16) to
    # ~8.5 GB, allowing 0.50 GPU utilization.  The remaining budget
    # is redistributed to the other servers:
    #   guardian  0.50 + colbert 0.13 + gliner 0.11 + compression 0.11
    #   = 0.85 total → fits comfortably on a 24 GB GPU.
    # ----------------------------------------------------------------
    guardian_enabled: bool = True
    guardian_model: str = "latence/granite-4.1-guardian-W8A16"
    guardian_port: int = 8005
    guardian_gpu_mem: float = 0.50
    guardian_max_model_len: int = 1024
    guardian_max_num_seqs: int = 64


def create_config() -> WorkerConfig:
    profile = (
        (
            os.environ.get("LATENCE_TRACE_PROFILE")
            or os.environ.get("LATENCE_TRACE_ACTIVE_PROFILE")
            or "quality"
        )
        .strip()
        .lower()
    )
    colbert_model = os.environ.get("LATENCE_TRACE_COLBERT_MODEL", "lightonai/LateOn")
    nli_model = os.environ.get(
        "LATENCE_TRACE_NLI_MODEL",
        os.environ.get(
            "VOYAGER_GROUNDEDNESS_NLI_MODEL",
            "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
        ),
    )
    compliance_model = os.environ.get(
        "LATENCE_TRACE_COMPLIANCE_GLINER_MODEL",
        "knowledgator/gliner-pii-large-v1.0",
    )
    compression_model, compression_server_enabled = _resolve_compression_model()
    return WorkerConfig(
        profile=profile,
        version=os.environ.get("LATENCE_TRACE_RUNPOD_VERSION", __version__),
        request_timeout_s=_env_int("LATENCE_TRACE_RUNPOD_REQUEST_TIMEOUT", 120),
        # Per-lane timeout: code lane ships a tight 150ms p95 SLO; a 2s
        # ceiling protects the tail while still catching hung requests.
        code_request_timeout_s=_env_float("LATENCE_TRACE_CODE_REQUEST_TIMEOUT_S", 2.0),
        # CPU-only stateless aggregation; sub-ms typical, 250ms
        # ceiling is plenty for 1000+ turn sessions.
        rollup_request_timeout_s=_env_float("LATENCE_TRACE_ROLLUP_REQUEST_TIMEOUT_S", 0.25),
        # RunPod prod cap. Sized for the 1/8 GPU pod (24 GiB GPU + 24 GiB
        # CPU) carrying the full 7-model topology: empirically 32 in-flight
        # /runsync calls leaves enough CPU memory for transient PyTorch /
        # tokenizer / Triton MaxSim allocations under load. Bumping past 32
        # has crashed the worker in calibration runs, so 32 is the tested
        # ceiling and the production default. Operators can override via
        # ``LATENCE_TRACE_MAX_CONCURRENCY`` for larger GPU pods.
        max_concurrency=_env_int("LATENCE_TRACE_MAX_CONCURRENCY", 32),
        collection_label=os.environ.get("LATENCE_TRACE_COLLECTION_LABEL", "latence-trace"),
        service_device=os.environ.get("LATENCE_TRACE_SERVICE_DEVICE", _detect_device()),
        docs_url=os.environ.get("LATENCE_TRACE_DOCS_URL", ""),
        managed_vllm_enabled=os.environ.get("LATENCE_TRACE_START_MANAGED_VLLM", "1").lower()
        not in {"0", "false", "no", "off"},
        colbert_model=colbert_model,
        colbert_port=_env_int("LATENCE_TRACE_COLBERT_PORT", 8001),
        colbert_gpu_mem=_env_float("LATENCE_TRACE_COLBERT_GPU_MEM", _DEFAULT_VLLM_GPU_MEM),
        colbert_max_model_len=_env_int("LATENCE_TRACE_COLBERT_MAX_MODEL_LEN", 8192),
        colbert_max_num_seqs=_env_int("LATENCE_TRACE_COLBERT_MAX_NUM_SEQS", 64),
        colbert_max_batched_tokens=_env_int("LATENCE_TRACE_COLBERT_MAX_BATCHED_TOKENS", 8192),
        nli_model=nli_model,
        nli_port=_env_int("LATENCE_TRACE_NLI_PORT", 8002),
        # Legacy mDeBERTa NLI server is no longer booted (replaced by
        # the dual SOTA path). Field kept on WorkerConfig for env-var
        # backwards compatibility (e.g. consumers reading via
        # diagnostics) but ignored by ``_build_servers``.
        nli_gpu_mem=_env_float("LATENCE_TRACE_NLI_GPU_MEM", _DEFAULT_VLLM_GPU_MEM),
        nli_max_model_len=_env_int("LATENCE_TRACE_NLI_MAX_MODEL_LEN", 512),
        nli_max_num_seqs=_env_int("LATENCE_TRACE_NLI_MAX_NUM_SEQS", 64),
        nli_max_batched_tokens=_env_int("LATENCE_TRACE_NLI_MAX_BATCHED_TOKENS", 8192),
        compliance_model=compliance_model,
        compliance_port=_env_int("LATENCE_TRACE_COMPLIANCE_GLINER_PORT", 8003),
        compliance_gpu_mem=_env_float("LATENCE_TRACE_COMPLIANCE_GLINER_GPU_MEM", _DEFAULT_VLLM_GPU_MEM),
        compliance_max_model_len=_env_int("LATENCE_TRACE_COMPLIANCE_MAX_MODEL_LEN", 768),
        compliance_max_num_seqs=_env_int("LATENCE_TRACE_COMPLIANCE_MAX_NUM_SEQS", 64),
        compliance_max_batched_tokens=_env_int("LATENCE_TRACE_COMPLIANCE_MAX_BATCHED_TOKENS", 8192),
        compliance_threshold=_env_float("LATENCE_TRACE_COMPLIANCE_THRESHOLD", 0.5),
        compliance_dataset_path=os.environ.get(
            "LATENCE_TRACE_COMPLIANCE_DATASET_PATH",
            "doubledsbv/pii-replacement-dataset",
        ),
        compliance_request_timeout_s=_env_float("LATENCE_TRACE_COMPLIANCE_REQUEST_TIMEOUT_S", 30.0),
        compliance_gliner_server_enabled=_env_bool_with_default(
            "LATENCE_TRACE_ENABLE_COMPLIANCE_GLINER_SERVER", True
        ),
        compression_model=compression_model,
        compression_server_enabled=compression_server_enabled,
        compression_port=_env_int("LATENCE_TRACE_COMPRESSION_PORT", 8004),
        compression_gpu_mem=_env_float("LATENCE_TRACE_COMPRESSION_GPU_MEM", _DEFAULT_VLLM_GPU_MEM),
        compression_max_model_len=_env_int("LATENCE_TRACE_COMPRESSION_MAX_MODEL_LEN", 8192),
        compression_max_num_seqs=_env_int("LATENCE_TRACE_COMPRESSION_MAX_NUM_SEQS", 64),
        compression_max_batched_tokens=_env_int(
            "LATENCE_TRACE_COMPRESSION_MAX_BATCHED_TOKENS", 8192
        ),
        compression_dtype=os.environ.get("LATENCE_TRACE_COMPRESSION_DTYPE", "auto"),
        compression_enforce_eager=os.environ.get(
            "LATENCE_TRACE_COMPRESSION_ENFORCE_EAGER", "1"
        ).lower()
        not in {"0", "false", "no"},
        compression_trust_remote_code=os.environ.get(
            "LATENCE_TRACE_COMPRESSION_TRUST_REMOTE_CODE", "0"
        ).lower()
        not in {"0", "false", "no"},
        compression_request_timeout_s=_env_float(
            "LATENCE_TRACE_COMPRESSION_REQUEST_TIMEOUT_S", 30.0
        ),
        compression_default_chunk_size=_env_int("LATENCE_TRACE_COMPRESSION_DEFAULT_CHUNK_SIZE", 4096),
        compression_default_compression_rate=_env_float(
            "LATENCE_TRACE_COMPRESSION_DEFAULT_COMPRESSION_RATE", 0.4
        ),
        compression_force_preserve_digit=os.environ.get(
            "LATENCE_TRACE_COMPRESSION_FORCE_PRESERVE_DIGIT", "1"
        ).lower()
        not in {"0", "false", "no"},
        compression_fallback_mode=os.environ.get("LATENCE_TRACE_COMPRESSION_FALLBACK_MODE", "1").lower()
        not in {"0", "false", "no"},
        # SOTA dual-NLI + reranker stack — DEFAULT ON. Set the matching
        # ``LATENCE_TRACE_*_ENABLED=0`` env var to disable any of the
        # three on a deployment-by-deployment basis (e.g. to free GPU
        # for one of the legacy lanes during incident response).
        nli_en_enabled=_env_bool_with_default("LATENCE_TRACE_NLI_EN_ENABLED", True),
        nli_en_model=os.environ.get(
            "LATENCE_TRACE_NLI_EN_MODEL", "lytang/MiniCheck-Flan-T5-Large"
        ),
        nli_en_port=_env_int("LATENCE_TRACE_NLI_EN_PORT", 8005),
        nli_en_gpu_mem=_env_float("LATENCE_TRACE_NLI_EN_GPU_MEM", _DEFAULT_VLLM_GPU_MEM),
        nli_en_max_model_len=_env_int("LATENCE_TRACE_NLI_EN_MAX_MODEL_LEN", 1024),
        nli_en_max_num_seqs=_env_int("LATENCE_TRACE_NLI_EN_MAX_NUM_SEQS", 64),
        nli_en_max_batched_tokens=_env_int(
            "LATENCE_TRACE_NLI_EN_MAX_BATCHED_TOKENS", 8192
        ),
        nli_multi_enabled=_env_bool_with_default("LATENCE_TRACE_NLI_MULTI_ENABLED", True),
        nli_multi_model=os.environ.get(
            "LATENCE_TRACE_NLI_MULTI_MODEL",
            "MoritzLaurer/bge-m3-zeroshot-v2.0",
        ),
        nli_multi_port=_env_int("LATENCE_TRACE_NLI_MULTI_PORT", 8006),
        nli_multi_gpu_mem=_env_float("LATENCE_TRACE_NLI_MULTI_GPU_MEM", _DEFAULT_VLLM_GPU_MEM),
        nli_multi_max_model_len=_env_int(
            "LATENCE_TRACE_NLI_MULTI_MAX_MODEL_LEN", 512
        ),
        nli_multi_max_num_seqs=_env_int(
            "LATENCE_TRACE_NLI_MULTI_MAX_NUM_SEQS", 64
        ),
        nli_multi_max_batched_tokens=_env_int(
            "LATENCE_TRACE_NLI_MULTI_MAX_BATCHED_TOKENS", 8192
        ),
        reranker_enabled=_env_bool_with_default("LATENCE_TRACE_RERANKER_ENABLED", True),
        reranker_model=os.environ.get(
            "LATENCE_TRACE_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"
        ),
        reranker_port=_env_int("LATENCE_TRACE_RERANKER_PORT", 8007),
        reranker_gpu_mem=_env_float("LATENCE_TRACE_RERANKER_GPU_MEM", _DEFAULT_VLLM_GPU_MEM),
        reranker_max_model_len=_env_int(
            "LATENCE_TRACE_RERANKER_MAX_MODEL_LEN", 512
        ),
        reranker_max_num_seqs=_env_int(
            "LATENCE_TRACE_RERANKER_MAX_NUM_SEQS", 64
        ),
        reranker_max_batched_tokens=_env_int(
            "LATENCE_TRACE_RERANKER_MAX_BATCHED_TOKENS", 8192
        ),
        guardian_enabled=_env_bool_with_default("LATENCE_TRACE_GUARDIAN_ENABLED", True),
        guardian_model=os.environ.get(
            "LATENCE_TRACE_GUARDIAN_MODEL", "latence/granite-4.1-guardian-W8A16"
        ),
        guardian_port=_env_int("LATENCE_TRACE_GUARDIAN_PORT", 8005),
        guardian_gpu_mem=_env_float("LATENCE_TRACE_GUARDIAN_GPU_MEM", 0.67),
        guardian_max_model_len=_env_int("LATENCE_TRACE_GUARDIAN_MAX_MODEL_LEN", 1024),
        guardian_max_num_seqs=_env_int("LATENCE_TRACE_GUARDIAN_MAX_NUM_SEQS", 64),
    )


_initialized = False
_config: WorkerConfig | None = None
_servers: dict[str, ManagedVllmServer] = {}
_service: GroundednessService | None = None
_compliance_service: ComplianceRedactionService | None = None
_compression_service: CompressionService | None = None
_session_service: TraceSessionService | None = None
_initialize_lock = threading.Lock()
_request_executor: ThreadPoolExecutor | None = None
_request_executor_lock = threading.Lock()
_lane_semaphores: dict[ScoringMode, asyncio.Semaphore] = {}
_lane_semaphores_loop: asyncio.AbstractEventLoop | None = None
_compliance_semaphore: asyncio.Semaphore | None = None
_compliance_semaphore_loop: asyncio.AbstractEventLoop | None = None


def _get_request_executor(config: WorkerConfig) -> ThreadPoolExecutor:
    global _request_executor
    if _request_executor is not None:
        return _request_executor
    with _request_executor_lock:
        if _request_executor is None:
            _request_executor = ThreadPoolExecutor(
                max_workers=config.max_concurrency,
                thread_name_prefix="latence-trace-runpod",
            )
        return _request_executor


def _lane_budget(config: WorkerConfig, lane: ScoringMode) -> int:
    """Per-lane inflight budget.

    Defaults to ``ceil(max_concurrency / 2)`` each so one lane can
    burst through the shared ceiling when the other is idle. Callers
    can pin the budget explicitly via
    ``LATENCE_TRACE_RAG_CONCURRENCY`` /
    ``LATENCE_TRACE_CODE_CONCURRENCY``.
    """
    env_key = (
        "LATENCE_TRACE_RAG_CONCURRENCY"
        if lane == ScoringMode.RAG
        else "LATENCE_TRACE_CODE_CONCURRENCY"
    )
    override = _env_int(env_key, 0)
    if override > 0:
        return override
    return max(1, int(math.ceil(config.max_concurrency / 2)))


def _get_lane_semaphore(config: WorkerConfig, lane: ScoringMode) -> asyncio.Semaphore:
    """Return (and lazily build) the per-lane semaphore for the active loop.

    The semaphore is pinned to the asyncio loop that owns it; if the
    loop flips (RunPod cold start, test reuse), we rebuild both
    semaphores against the new loop.
    """
    global _lane_semaphores, _lane_semaphores_loop
    loop = asyncio.get_running_loop()
    if _lane_semaphores_loop is not loop:
        _lane_semaphores = {
            ScoringMode.RAG: asyncio.Semaphore(_lane_budget(config, ScoringMode.RAG)),
            ScoringMode.CODE: asyncio.Semaphore(_lane_budget(config, ScoringMode.CODE)),
        }
        _lane_semaphores_loop = loop
    return _lane_semaphores[lane]


def _get_compliance_semaphore(config: WorkerConfig) -> asyncio.Semaphore:
    global _compliance_semaphore, _compliance_semaphore_loop
    loop = asyncio.get_running_loop()
    if _compliance_semaphore_loop is not loop or _compliance_semaphore is None:
        _compliance_semaphore = asyncio.Semaphore(max(1, config.max_concurrency))
        _compliance_semaphore_loop = loop
    return _compliance_semaphore


def _make_startup_sentence(prefix: str, index: int) -> str:
    return (
        f"{prefix} record {index} states the reference sample remained internally "
        f"consistent on day {index} with calibration value {100 + index}."
    )


_CODE_LANE_WARMUP_RESPONSE = (
    "```python\n"
    "from httpx import AsyncClient\n\n"
    "async def fetch_user(id: int):\n"
    "    async with AsyncClient() as client:\n"
    "        return await client.get(f'/users/{id}')\n"
    "```"
)

_CODE_LANE_WARMUP_CONTEXT = (
    "# fetch_user.py\n"
    "async def fetch_user(id: int):\n"
    "    async with httpx.AsyncClient() as client:\n"
    "        response = await client.get(f'/users/{id}')\n"
    "        return response.json()\n"
)


def _build_startup_warmup_requests() -> list[GroundednessRequest]:
    query_text = "Which recorded days remained internally consistent in the reference sample?"
    shapes = (
        ("alpha", 8, 4),
        ("beta", 24, 8),
        ("gamma", 48, 16),
    )
    requests: list[GroundednessRequest] = []
    for prefix, context_count, response_count in shapes:
        context_sentences = [
            _make_startup_sentence(prefix, idx) for idx in range(1, context_count + 1)
        ]
        response_sentences = [
            _make_startup_sentence(prefix, idx) for idx in range(1, response_count + 1)
        ]
        requests.append(
            GroundednessRequest(
                query_text=query_text,
                raw_context=" ".join(context_sentences),
                response_text=" ".join(response_sentences),
                include_triangular_diagnostics=True,
                evidence_limit=4,
            )
        )
    # One code-lane warmup turn primes GPUScorer streams, pre-loads the
    # tree-sitter grammars, and touches the composite model.
    requests.append(
        GroundednessRequest(
            scoring_mode=ScoringMode.CODE,
            session_id="warmup-code-lane",
            response_language_hint="python",
            query_text="Add retry logic to fetch_user",
            raw_context=_CODE_LANE_WARMUP_CONTEXT,
            response_text=_CODE_LANE_WARMUP_RESPONSE,
            emit_chunk_ownership=False,
            evidence_limit=4,
        )
    )
    return requests


def _ensure_kernel_warmup(profile: str) -> None:
    result = warm_all(profile)
    if not result.ok:
        raise RuntimeError(
            f"Triton kernel warmup failed for profile '{result.profile}' "
            f"on {result.device}: {result.error or 'unknown error'}"
        )
    # Code lane warmup is now a hard gate: tree-sitter grammars are a
    # first-class production dependency and a missing grammar would
    # silently degrade the AST phantom / drift signals to a regex
    # fallback. We refuse to serve traffic in that state.
    code_result = warm_code_lane()
    if not code_result.ok:
        raise RuntimeError(
            f"Code-lane warmup failed on {code_result.device}: "
            f"{code_result.error} (details={code_result.details})"
        )
    # Pre-load the WTPSplit SaT model + per-language PySBD segmenters
    # so the first scoring request does not pay the model-load cost
    # on the hot path. This is best-effort: a SaT load failure is
    # logged but does not block traffic, since the segmenter cascade
    # gracefully falls through to PySBD and a single-span safety net.
    try:
        seg_state = warmup_segmenters(languages=("en", "de"))
        logger.info(
            "text_segmentation_boot_warmup sat=%s pysbd=%s model=%s",
            seg_state.get("sat_loaded"),
            seg_state.get("pysbd_loaded"),
            seg_state.get("sat_model"),
        )
    except Exception:  # pragma: no cover - segmentation warmup is non-fatal
        logger.exception("text_segmentation_boot_warmup_failed")


def _prompt_guard_startup_enabled() -> bool:
    """Return True when Llama-Prompt-Guard-2 should be loaded at boot.

    Production default is ``LATENCE_TRACE_CONTEXT_TRUST_PROVIDER=prompt_guard``
    so the 86M Llama-Guard checkpoint is resident on the GPU when the worker
    becomes ready. The per-request ``context_trust_enabled`` flag still gates
    whether the lane is invoked, so opting out of the scan stays free
    (~0 ms) — but opting in never pays the multi-second cold-start penalty.
    """
    if os.environ.get("LATENCE_TRACE_CONTEXT_TRUST_ENABLED", "1").strip().lower() in {
        "0",
        "false",
        "no",
        "off",
    }:
        return False
    provider = os.environ.get(
        "LATENCE_TRACE_CONTEXT_TRUST_PROVIDER", "prompt_guard"
    ).strip().lower()
    return provider in {"prompt_guard", "llama_prompt_guard", "llama_prompt_guard_2"}


def _warm_prompt_guard_at_boot() -> dict[str, Any]:
    """Warm the Llama-Prompt-Guard-2 runtime, defensively.

    The 7th model (``meta-llama/Llama-Prompt-Guard-2-86M``) is a Meta gated
    repo on HuggingFace. Operators without a HF token bound to a Meta-Llama-
    accepted account cannot fetch it. The lane is opt-in (only invoked when
    a request sets ``context_trust_enabled=true``), so failing to warm it
    must NEVER crash the worker — that would take the entire 6-server
    topology offline because of one optional model.

    On failure we:
      1. Log a clear warning telling the operator how to enable the lane
         (set ``HF_TOKEN`` / ``HUGGING_FACE_HUB_TOKEN`` and accept the
         Meta-Llama license at the HuggingFace model page).
      2. Force ``LATENCE_TRACE_CONTEXT_TRUST_PROVIDER=heuristic`` so any
         future ``context_trust_enabled=true`` request degrades to the
         CPU-only heuristic provider instead of repeatedly trying to load
         the gated model and timing out.
      3. Return a skipped-result envelope so ``initialize()`` continues.
    """
    os.environ.setdefault("LATENCE_TRACE_CONTEXT_TRUST_PROMPT_GUARD_COMPILE", "1")
    os.environ.setdefault(
        "LATENCE_TRACE_CONTEXT_TRUST_PROMPT_GUARD_COMPILE_MODE",
        "reduce-overhead",
    )
    try:
        result = warm_context_trust_runtime()
    except Exception as exc:  # noqa: BLE001 — boot must not propagate
        os.environ["LATENCE_TRACE_CONTEXT_TRUST_PROVIDER"] = "heuristic"
        model_id = os.environ.get(
            "LATENCE_TRACE_CONTEXT_TRUST_PROMPT_GUARD_MODEL",
            "meta-llama/Llama-Prompt-Guard-2-86M",
        )
        logger.warning(
            "prompt_guard_boot_warmup_failed: model=%s falling back to "
            "provider=heuristic. Set HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) "
            "to a token with Meta-Llama license access at "
            "https://huggingface.co/%s to enable the Llama-Guard lane. "
            "error=%s",
            model_id,
            model_id,
            exc,
        )
        return {
            "status": "skipped",
            "reason": "prompt_guard_load_failed",
            "model_id": model_id,
            "fallback_provider": "heuristic",
            "error": str(exc),
        }
    logger.info(
        "prompt_guard_boot_warmup_complete: model=%s device=%s compiled=%s mode=%s elapsed_ms=%.2f states=%s",
        result.get("model_id"),
        result.get("device"),
        result.get("compiled"),
        result.get("compile_mode"),
        float(result.get("elapsed_ms") or 0.0),
        result.get("states"),
    )
    return result


def _prime_service_runtime(service: GroundednessService) -> None:
    requests = _build_startup_warmup_requests()
    for idx, request in enumerate(requests, start=1):
        response = service.groundedness(request)
        logger.info(
            "startup warmup request %s/%s complete: primary=%s score=%.4f latency_ms=%.2f",
            idx,
            len(requests),
            response.scores.primary_name,
            response.scores.primary_score,
            response.time_ms,
        )


def _prepare_compliance_model_for_vllm(config: WorkerConfig) -> WorkerConfig:
    """Resolve GLiNER repos with only gliner_config.json into vLLM-ready dirs."""
    if os.environ.get("LATENCE_TRACE_COMPLIANCE_PREPARE_MODEL", "1").lower() in {
        "0",
        "false",
        "no",
        "off",
    }:
        return config

    model_ref = config.compliance_model
    if os.path.isdir(model_ref) and os.path.exists(os.path.join(model_ref, "config.json")):
        return config
    if "/" not in model_ref:
        return config

    output_dir = os.environ.get("LATENCE_TRACE_COMPLIANCE_PREPARED_MODEL_DIR")
    if not output_dir:
        slug = model_ref.replace("/", "--").replace(".", "-")
        output_dir = f"/tmp/{slug}-vllm"

    try:
        from forge.model_prep import prepare_gliner_model

        prepared_model = prepare_gliner_model(
            model_ref,
            plugin="deberta_gliner",
            output_dir=output_dir,
            force=os.environ.get("LATENCE_TRACE_COMPLIANCE_FORCE_PREPARE", "0").lower()
            in {"1", "true", "yes", "on"},
        )
    except Exception:
        logger.exception("Failed to prepare compliance GLiNER model %s for vLLM", model_ref)
        raise

    if prepared_model == model_ref:
        return config
    logger.info("Prepared compliance GLiNER model for vLLM: %s -> %s", model_ref, prepared_model)
    return replace(config, compliance_model=prepared_model)


def _minicheck_plugin_available() -> bool:
    """Return True when the BYOP MiniCheck T5 plugin is importable.

    The plugin lives at ``runpod/vllm_plugins/minicheck_t5/`` and is the
    encoder-decoder + custom pooler that lets vLLM serve
    ``lytang/MiniCheck-Flan-T5-Large`` with the same /pooling wire shape
    as ``nli_mdeberta``. Until that module exists ``LATENCE_TRACE_NLI_EN_ENABLED=1``
    cannot boot a server; we log a warning and the in-process
    transformers fallback in
    :class:`latence_trace.providers.nli_transformers.MiniCheckNLIProvider`
    handles the requests instead. This keeps the env-var contract
    forward-compatible: nothing changes when the plugin lands except a
    server appears.
    """

    import importlib.util

    return importlib.util.find_spec("minicheck_t5") is not None


def _build_servers(config: WorkerConfig) -> dict[str, ManagedVllmServer]:
    if not config.managed_vllm_enabled:
        return {}

    servers: dict[str, ManagedVllmServer] = {}

    # ----------------------------------------------------------------
    # Granite Guardian — single multilingual groundedness LLM.
    # Booted FIRST because it uses CUDA graphs (enforce_eager=False)
    # and chunked prefill, which need early GPU memory reservation.
    # When enabled it supersedes nli_en + nli_multi + reranker,
    # freeing two vLLM slots.  The W8A16 GPTQ model fits at 0.50
    # utilization alongside colbert (0.13), gliner (0.11), and
    # compression (0.11) = 0.85 total on 24 GB.
    # ----------------------------------------------------------------
    if config.guardian_enabled:
        servers["guardian"] = ManagedVllmServer(
            name="guardian",
            model=config.guardian_model,
            port=config.guardian_port,
            runner="generate",
            gpu_memory_utilization=config.guardian_gpu_mem,
            max_model_len=config.guardian_max_model_len,
            max_num_seqs=config.guardian_max_num_seqs,
            dtype="auto",
            enforce_eager=False,
            enable_chunked_prefill=True,
        )

    # When Guardian is active it takes 0.50; shrink the co-hosted
    # servers so the total stays ≤ 0.91 on a 24 GB GPU.
    # Per-server env vars (LATENCE_TRACE_COLBERT_GPU_MEM, etc.) always
    # win; these defaults only apply when the env var is unset.
    _guardian_defaults = {
        "colbert": 0.12,
        "gliner": 0.14,
        "compression": 0.15,
    }

    def _gpu_mem(config_val: float, guardian_key: str) -> float:
        if config.guardian_enabled:
            return config_val if config_val != _DEFAULT_VLLM_GPU_MEM else _guardian_defaults[guardian_key]
        return config_val

    servers["colbert"] = ManagedVllmServer(
        name="colbert",
        model=config.colbert_model,
        port=config.colbert_port,
        io_processor_plugin="moderncolbert_batched_io",
        gpu_memory_utilization=_gpu_mem(config.colbert_gpu_mem, "colbert"),
        max_model_len=config.colbert_max_model_len,
        max_num_seqs=config.colbert_max_num_seqs,
        max_num_batched_tokens=config.colbert_max_batched_tokens,
        plugins=["moderncolbert", "moderncolbert_batched_io"],
        enforce_eager=True,
    )
    if config.compliance_gliner_server_enabled:
        servers["compliance_gliner"] = ManagedVllmServer(
            name="compliance_gliner",
            model=config.compliance_model,
            port=config.compliance_port,
            io_processor_plugin="deberta_gliner_io",
            gpu_memory_utilization=_gpu_mem(config.compliance_gpu_mem, "gliner"),
            max_model_len=config.compliance_max_model_len,
            max_num_seqs=config.compliance_max_num_seqs,
            max_num_batched_tokens=config.compliance_max_batched_tokens,
            plugins=["deberta_gliner", "deberta_gliner_io"],
            enforce_eager=True,
        )
    if config.compression_server_enabled:
        servers["compression"] = ManagedVllmServer(
            name="compression",
            model=config.compression_model,
            port=config.compression_port,
            gpu_memory_utilization=_gpu_mem(config.compression_gpu_mem, "compression"),
            max_model_len=config.compression_max_model_len,
            max_num_seqs=config.compression_max_num_seqs,
            max_num_batched_tokens=config.compression_max_batched_tokens,
            dtype=config.compression_dtype,
            trust_remote_code=config.compression_trust_remote_code,
            plugins=["qwen3_compression"],
            enforce_eager=config.compression_enforce_eager,
        )

    # When Guardian is enabled, skip legacy NLI/reranker servers.
    if config.guardian_enabled:
        return servers

    # ----------------------------------------------------------------
    # Legacy SOTA dual-NLI + reranker servers. Skipped when Guardian
    # is enabled above.
    # ----------------------------------------------------------------
    if config.nli_en_enabled:
        if _minicheck_plugin_available():
            servers["nli_en"] = ManagedVllmServer(
                name="nli_en",
                model=config.nli_en_model,
                port=config.nli_en_port,
                io_processor_plugin="minicheck_t5_io",
                gpu_memory_utilization=config.nli_en_gpu_mem,
                max_model_len=config.nli_en_max_model_len,
                max_num_seqs=config.nli_en_max_num_seqs,
                max_num_batched_tokens=config.nli_en_max_batched_tokens,
                plugins=["minicheck_t5", "minicheck_t5_io"],
                enforce_eager=True,
            )
        else:
            logger.warning(
                "nli_en_server_skipped_plugin_missing",
                extra={
                    "model": config.nli_en_model,
                    "hint": (
                        "Set LATENCE_TRACE_NLI_EN_ENABLED=0 or land the "
                        "minicheck_t5 BYOP plugin under runpod/vllm_plugins/. "
                        "Until then per-request English NLI uses the "
                        "in-process MiniCheckNLIProvider fallback."
                    ),
                },
            )
    if config.nli_multi_enabled:
        servers["nli_multi"] = ManagedVllmServer(
            name="nli_multi",
            model=config.nli_multi_model,
            port=config.nli_multi_port,
            gpu_memory_utilization=config.nli_multi_gpu_mem,
            max_model_len=config.nli_multi_max_model_len,
            max_num_seqs=config.nli_multi_max_num_seqs,
            max_num_batched_tokens=config.nli_multi_max_batched_tokens,
            enforce_eager=True,
            extra_args=["--convert", "classify"],
        )
    if config.reranker_enabled:
        servers["reranker"] = ManagedVllmServer(
            name="reranker",
            model=config.reranker_model,
            port=config.reranker_port,
            gpu_memory_utilization=config.reranker_gpu_mem,
            max_model_len=config.reranker_max_model_len,
            max_num_seqs=config.reranker_max_num_seqs,
            max_num_batched_tokens=config.reranker_max_batched_tokens,
            enforce_eager=True,
            extra_args=["--convert", "classify"],
        )
    return servers


def initialize() -> None:
    global _initialized, _config, _servers, _service, _compliance_service, _compression_service
    global _session_service
    if _initialized:
        return

    with _initialize_lock:
        if _initialized:
            return

        config = create_config()
        if config.managed_vllm_enabled and config.compliance_gliner_server_enabled:
            config = _prepare_compliance_model_for_vllm(config)
        _config = config

        os.environ.setdefault("LATENCE_TRACE_PROFILE", config.profile)
        apply_profile(config.profile)

        servers = _build_servers(config)

        try:
            prompt_guard_boot_enabled = _prompt_guard_startup_enabled()
            # Boot vLLM servers SEQUENTIALLY to avoid GPU memory
            # fragmentation. Parallel boot via ThreadPoolExecutor used to
            # have all 6 vLLM workers race for HBM at the same instant —
            # vLLM 0.19's KV-cache allocator can then carve up the
            # remaining budget at non-aligned offsets and silently leave
            # the last server with a fragmented free pool that can't fit
            # one max-length sequence (observed as "exited with code 1"
            # from the Qwen3-0.6B compression server). Sequential boot
            # gives each server a clean, contiguous slice of HBM.
            #
            # The pod is always-on, so the ~2x cold-start cost (boot once
            # on rebuild, then run forever) is irrelevant compared to the
            # reliability win. Per-server enforce_eager=True is already
            # set on every vLLM entry above to disable CUDA graphs (which
            # also fragment HBM during warmup capture).
            #
            # Llama-Prompt-Guard-2 stays parallel-with-the-pool because it
            # touches a separate CUDA stream and doesn't compete for the
            # vLLM KV-cache pool. We launch it first so it can warm up
            # alongside the sequential server starts.
            prompt_guard_thread = None
            prompt_guard_result_box: dict[str, Any] = {}
            if prompt_guard_boot_enabled:
                def _prompt_guard_worker() -> None:
                    try:
                        prompt_guard_result_box["result"] = _warm_prompt_guard_at_boot()
                    except Exception as exc:  # noqa: BLE001 — defensive, mirrors warm helper
                        prompt_guard_result_box["error"] = exc
                prompt_guard_thread = threading.Thread(
                    target=_prompt_guard_worker,
                    name="prompt-guard-boot-warmup",
                    daemon=True,
                )
                prompt_guard_thread.start()

            for name, server in servers.items():
                logger.info("vllm_boot_sequential_start: server=%s", name)
                server.start()
                logger.info("vllm_boot_sequential_ready: server=%s url=%s", name, server.base_url)

            if prompt_guard_thread is not None:
                prompt_guard_thread.join()
                # _warm_prompt_guard_at_boot is itself defensive (catches
                # GatedRepoError etc. and falls back to heuristic), so any
                # exception that escapes here is a hard programmer bug —
                # surface it.
                if "error" in prompt_guard_result_box:
                    raise prompt_guard_result_box["error"]

            colbert_server = servers.get("colbert")
            if colbert_server is not None:
                os.environ["VOYAGER_GROUNDEDNESS_VLLM_ENDPOINT"] = colbert_server.base_url
            os.environ["VOYAGER_GROUNDEDNESS_VLLM_MODEL"] = config.colbert_model
            os.environ["VOYAGER_GROUNDEDNESS_VLLM_MAX_CONCURRENCY"] = str(config.max_concurrency)
            # Legacy ``nli`` (mDeBERTa) vLLM server is no longer booted.
            # We still publish the model id under the legacy env names
            # so the in-process ``HuggingFaceNLIProvider`` defensive
            # fallback (used only when language=None and no vLLM
            # endpoint is set) and any diagnostic readers see the
            # historical default. ``LATENCE_TRACE_NLI_VLLM_ENDPOINT``
            # is intentionally left unset so the legacy lane skips the
            # vLLM-client constructor and goes straight to the in-
            # process provider.
            os.environ["LATENCE_TRACE_NLI_VLLM_MODEL"] = config.nli_model
            os.environ["LATENCE_TRACE_NLI_VLLM_MAX_CONCURRENCY"] = str(config.max_concurrency)
            os.environ["VOYAGER_GROUNDEDNESS_NLI_MODEL"] = config.nli_model
            # Per-language SOTA NLI servers (registry consumes these
            # via LATENCE_TRACE_NLI_<LANG>_ENDPOINT). Whether or not
            # the operator opted in, we publish the *MODEL* env var so
            # the in-process transformers fallback uses the same model
            # name surfaced in profile_diagnostics.
            os.environ["LATENCE_TRACE_NLI_EN_TRANSFORMERS_MODEL"] = config.nli_en_model
            os.environ["LATENCE_TRACE_NLI_MULTI_TRANSFORMERS_MODEL"] = (
                config.nli_multi_model
            )
            guardian_server = servers.get("guardian")
            if guardian_server is not None:
                os.environ["LATENCE_TRACE_GUARDIAN_ENDPOINT"] = guardian_server.base_url
                os.environ["LATENCE_TRACE_GUARDIAN_MODEL"] = config.guardian_model
            nli_en_server = servers.get("nli_en")
            if nli_en_server is not None:
                os.environ["LATENCE_TRACE_NLI_EN_ENDPOINT"] = nli_en_server.base_url
                os.environ["LATENCE_TRACE_NLI_EN_MODEL"] = config.nli_en_model
                os.environ["LATENCE_TRACE_NLI_EN_PROTOCOL"] = "pooling"
            nli_multi_server = servers.get("nli_multi")
            if nli_multi_server is not None:
                os.environ["LATENCE_TRACE_NLI_MULTI_ENDPOINT"] = (
                    nli_multi_server.base_url
                )
                os.environ["LATENCE_TRACE_NLI_MULTI_MODEL"] = config.nli_multi_model
                os.environ["LATENCE_TRACE_NLI_MULTI_PROTOCOL"] = "classify"
            reranker_server = servers.get("reranker")
            if reranker_server is not None:
                os.environ["LATENCE_TRACE_RERANKER_ENDPOINT"] = (
                    reranker_server.base_url
                )
                os.environ["LATENCE_TRACE_RERANKER_MODEL"] = config.reranker_model
            compliance_server = servers.get("compliance_gliner")
            if compliance_server is not None:
                os.environ["LATENCE_TRACE_COMPLIANCE_GLINER_ENDPOINT"] = compliance_server.base_url
            os.environ["LATENCE_TRACE_COMPLIANCE_GLINER_MODEL"] = config.compliance_model
            os.environ["LATENCE_TRACE_COMPLIANCE_MAX_CONCURRENCY"] = str(config.max_concurrency)
            os.environ["LATENCE_TRACE_COMPLIANCE_MAX_MODEL_LEN"] = str(
                config.compliance_max_model_len
            )
            os.environ["LATENCE_TRACE_COMPLIANCE_DATASET_PATH"] = config.compliance_dataset_path
            compression_server = servers.get("compression")
            if compression_server is not None:
                os.environ["LATENCE_TRACE_COMPRESSION_ENDPOINT"] = compression_server.base_url
                os.environ["LATENCE_TRACE_COMPRESSION_MODEL"] = config.compression_model
                os.environ["LATENCE_TRACE_COMPRESSION_CONCURRENCY"] = str(config.max_concurrency)
                os.environ["LATENCE_TRACE_COMPRESSION_DEFAULT_CHUNK_SIZE"] = str(
                    config.compression_default_chunk_size
                )
                os.environ["LATENCE_TRACE_COMPRESSION_DEFAULT_COMPRESSION_RATE"] = str(
                    config.compression_default_compression_rate
                )
                os.environ["LATENCE_TRACE_COMPRESSION_FORCE_PRESERVE_DIGIT"] = str(
                    int(config.compression_force_preserve_digit)
                )
                os.environ["LATENCE_TRACE_COMPRESSION_FALLBACK_MODE"] = str(
                    int(config.compression_fallback_mode)
                )
            os.environ.setdefault("VOYAGER_GROUNDEDNESS_NLI_ENABLED", "1")
            os.environ.setdefault(
                "VOYAGER_GROUNDEDNESS_NLI_PREMISE_RERANKER_MODEL",
                "BAAI/bge-reranker-v2-m3",
            )

            service = GroundednessService(
                device=config.service_device,
                collection_label=config.collection_label,
            )
            compliance_service = ComplianceRedactionService.from_env()
            compression_service = CompressionService.from_env()
            service._compression_service = compression_service
            if config.managed_vllm_enabled:
                _ensure_kernel_warmup(config.profile)
                _prime_service_runtime(service)
            else:
                logger.info("Skipping managed vLLM startup and warmup; using external/stateless dev mode")
            _get_request_executor(config)

            session_service = TraceSessionService(groundedness_service=service)

            _servers = servers
            _service = service
            _compliance_service = compliance_service
            _compression_service = compression_service
            _session_service = session_service
            _initialized = True
            logger.info(
                (
                    "latence-trace RunPod worker ready: profile=%s colbert=%s "
                    "guardian=%s nli_en=%s nli_multi=%s reranker=%s compliance_gliner=%s "
                    "compression=%s prompt_guard=%s triton_maxsim=%s"
                ),
                config.profile,
                colbert_server.base_url if colbert_server is not None else "external",
                guardian_server.base_url if guardian_server is not None else "off",
                nli_en_server.base_url if nli_en_server is not None else "fallback",
                nli_multi_server.base_url if nli_multi_server is not None else "fallback",
                reranker_server.base_url if reranker_server is not None else "fallback",
                compliance_server.base_url if compliance_server is not None else "external",
                compression_server.base_url if compression_server is not None else "fallback",
                "in_process" if _prompt_guard_startup_enabled() else "off",
                "in_process",
            )
        except Exception:
            for server in servers.values():
                server.stop()
            raise


def shutdown() -> None:
    global _initialized, _servers, _service, _compliance_service, _compression_service
    global _session_service
    global _request_executor
    global _lane_semaphores, _lane_semaphores_loop
    global _compliance_semaphore, _compliance_semaphore_loop
    for server in _servers.values():
        try:
            server.stop()
        except Exception:
            logger.exception("failed to stop server %s", server.name)
    _servers = {}
    _service = None
    if _compliance_service is not None:
        try:
            _compliance_service.provider.close()
        except Exception:
            logger.exception("failed to close compliance provider")
    _compliance_service = None
    _compression_service = None
    _session_service = None
    _initialized = False
    with _request_executor_lock:
        if _request_executor is not None:
            _request_executor.shutdown(wait=False, cancel_futures=True)
            _request_executor = None
    _lane_semaphores = {}
    _lane_semaphores_loop = None
    _compliance_semaphore = None
    _compliance_semaphore_loop = None


def _code_lane_health() -> dict[str, Any]:
    """Snapshot code-lane readiness for the /healthz probe.

    Reports the current tree-sitter parser backend, which grammars
    loaded, and whether the code-lane warmup pass succeeded. Allows
    operators and alerting to instantly flag a deploy that slipped
    into the regex fallback path.
    """

    payload: dict[str, Any] = {
        "parser_backend": None,
        "loaded_grammars": [],
        "required_grammars": [],
        "warmup_ok": None,
        "warmup_error": None,
    }
    try:
        from latence_trace.core.code_lane.ast_grounding import (
            SUPPORTED_LANGUAGES,
            AstSymbolExtractor,
        )
        from latence_trace.kernels.warmup import warmup_state

        payload["required_grammars"] = sorted(SUPPORTED_LANGUAGES)
        payload["parser_backend"] = AstSymbolExtractor(enabled=True).parser_backend
        payload["loaded_grammars"] = AstSymbolExtractor.available_languages()
        code_result = warmup_state().get("code_lane")
        if code_result is not None:
            payload["warmup_ok"] = bool(code_result.ok)
            payload["warmup_error"] = code_result.error
    except Exception as exc:  # pragma: no cover - defensive
        payload["warmup_error"] = str(exc)
    return payload


def _runtime_model_config(config: WorkerConfig | None) -> dict[str, Any]:
    if config is None:
        return {}
    return {
        "colbert": {
            "model": config.colbert_model,
            "port": config.colbert_port,
            "runner": "pooling",
            "io_processor_plugin": "moderncolbert_batched_io",
            "plugins": ["moderncolbert", "moderncolbert_batched_io"],
            "gpu_memory_utilization": config.colbert_gpu_mem,
            "max_model_len": config.colbert_max_model_len,
            "max_num_seqs": config.colbert_max_num_seqs,
            "max_num_batched_tokens": config.colbert_max_batched_tokens,
            "dtype": "bfloat16",
            "quantization": None,
            "enforce_eager": True,
            "trust_remote_code": True,
            "enable_prefix_caching": False,
            "enable_chunked_prefill": False,
        },
        # SOTA dual-NLI + reranker stack — these blocks replace the
        # legacy ``nli`` (mDeBERTa) entry. The diagnostics surface
        # mirrors the live topology booted by ``_build_servers``.
        "nli_en": {
            "model": config.nli_en_model,
            "port": config.nli_en_port,
            "enabled": config.nli_en_enabled,
            "runner": "pooling",
            "io_processor_plugin": "minicheck_t5_io",
            "plugins": ["minicheck_t5", "minicheck_t5_io"],
            "gpu_memory_utilization": config.nli_en_gpu_mem,
            "max_model_len": config.nli_en_max_model_len,
            "max_num_seqs": config.nli_en_max_num_seqs,
            "max_num_batched_tokens": config.nli_en_max_batched_tokens,
            "dtype": "bfloat16",
            "quantization": None,
            "enforce_eager": True,
            "trust_remote_code": True,
            "enable_prefix_caching": False,
            "enable_chunked_prefill": False,
        },
        "nli_multi": {
            "model": config.nli_multi_model,
            "port": config.nli_multi_port,
            "enabled": config.nli_multi_enabled,
            "runner": "classify",
            "extra_args": ["--convert", "classify"],
            "gpu_memory_utilization": config.nli_multi_gpu_mem,
            "max_model_len": config.nli_multi_max_model_len,
            "max_num_seqs": config.nli_multi_max_num_seqs,
            "max_num_batched_tokens": config.nli_multi_max_batched_tokens,
            "dtype": "bfloat16",
            "quantization": None,
            "enforce_eager": True,
            "trust_remote_code": False,
            "enable_prefix_caching": False,
            "enable_chunked_prefill": False,
        },
        "reranker": {
            "model": config.reranker_model,
            "port": config.reranker_port,
            "enabled": config.reranker_enabled,
            "runner": "classify",
            "extra_args": ["--convert", "classify"],
            "gpu_memory_utilization": config.reranker_gpu_mem,
            "max_model_len": config.reranker_max_model_len,
            "max_num_seqs": config.reranker_max_num_seqs,
            "max_num_batched_tokens": config.reranker_max_batched_tokens,
            "dtype": "bfloat16",
            "quantization": None,
            "enforce_eager": True,
            "trust_remote_code": False,
            "enable_prefix_caching": False,
            "enable_chunked_prefill": False,
        },
        "guardian": {
            "model": config.guardian_model,
            "port": config.guardian_port,
            "enabled": config.guardian_enabled,
            "runner": "generate",
            "gpu_memory_utilization": config.guardian_gpu_mem,
            "max_model_len": config.guardian_max_model_len,
            "max_num_seqs": config.guardian_max_num_seqs,
            "dtype": "auto",
            "enforce_eager": False,
            "enable_chunked_prefill": True,
        },
        "compliance_gliner": {
            "model": config.compliance_model,
            "port": config.compliance_port,
            "runner": "pooling",
            "io_processor_plugin": "deberta_gliner_io",
            "plugins": ["deberta_gliner", "deberta_gliner_io"],
            "gpu_memory_utilization": config.compliance_gpu_mem,
            "max_model_len": config.compliance_max_model_len,
            "max_num_seqs": config.compliance_max_num_seqs,
            "max_num_batched_tokens": config.compliance_max_batched_tokens,
            "dtype": "bfloat16",
            "quantization": None,
            "enforce_eager": True,
            "trust_remote_code": True,
            "enable_prefix_caching": False,
            "enable_chunked_prefill": False,
        },
        "compression": {
            "model": config.compression_model,
            "enabled": config.compression_server_enabled,
            "port": config.compression_port,
            "runner": "pooling",
            "task": "token_classify",
            "plugins": ["qwen3_compression"],
            "gpu_memory_utilization": config.compression_gpu_mem,
            "max_model_len": config.compression_max_model_len,
            "max_num_seqs": config.compression_max_num_seqs,
            "max_num_batched_tokens": config.compression_max_batched_tokens,
            "dtype": config.compression_dtype,
            "quantization": None,
            "enforce_eager": config.compression_enforce_eager,
            "trust_remote_code": config.compression_trust_remote_code,
            "enable_prefix_caching": False,
            "enable_chunked_prefill": False,
            "default_chunk_size": config.compression_default_chunk_size,
            "default_compression_rate": config.compression_default_compression_rate,
            "force_preserve_digit": config.compression_force_preserve_digit,
            "fallback_mode": config.compression_fallback_mode,
        },
        "context_trust": _context_trust_runtime_config(),
    }


def _context_trust_runtime_config() -> dict[str, Any]:
    provider = os.environ.get(
        "LATENCE_TRACE_CONTEXT_TRUST_PROVIDER", "prompt_guard"
    ).strip().lower()
    enabled = os.environ.get("LATENCE_TRACE_CONTEXT_TRUST_ENABLED", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    compile_enabled = os.environ.get(
        "LATENCE_TRACE_CONTEXT_TRUST_PROMPT_GUARD_COMPILE",
        "1",
    ).strip().lower() not in {"0", "false", "no", "off"}
    return {
        "enabled": enabled,
        "request_default_enabled": True,
        "request_field": "context_trust_enabled",
        "request_aliases": ["guard_check_enabled"],
        "provider": provider if enabled else "off",
        "model": os.environ.get(
            "LATENCE_TRACE_CONTEXT_TRUST_PROMPT_GUARD_MODEL",
            "meta-llama/Llama-Prompt-Guard-2-86M",
        ),
        "compile_enabled": compile_enabled,
        "compile_mode": os.environ.get(
            "LATENCE_TRACE_CONTEXT_TRUST_PROMPT_GUARD_COMPILE_MODE",
            "reduce-overhead",
        ),
    }


def _health_payload() -> dict[str, Any]:
    return {
        "success": True,
        "version": _config.version if _config else __version__,
        "profile": _config.profile if _config else None,
        "service_device": _config.service_device if _config else None,
        "max_concurrency": _config.max_concurrency if _config else None,
        "startup_warmup_requests": _STARTUP_WARMUP_REQUEST_COUNT,
        "servers": {name: server.health() for name, server in _servers.items()},
        "model_runtime_config": _runtime_model_config(_config),
        "context_trust": _context_trust_runtime_config(),
        "code_lane": _code_lane_health(),
    }


def _build_request(input_data: dict[str, Any]) -> tuple[GroundednessRequest, bool]:
    raw_context = input_data.get("raw_context")
    if raw_context is None:
        raw_context = input_data.get("context")
    query_text = input_data.get("query_text")
    if query_text is None:
        query_text = input_data.get("query")
    response_text = input_data.get("response_text")
    if response_text is None:
        response_text = input_data.get("response")

    payload: dict[str, Any] = {
        "query_text": query_text,
        "raw_context": raw_context,
        "response_text": response_text,
    }

    passthrough_keys = (
        "chunk_ids",
        "support_units",
        "attribution_mode",
        "segmentation_mode",
        "raw_context_chunk_tokens",
        "response_chunk_tokens",
        "coverage_threshold",
        "primary_metric",
        "evidence_limit",
        "debug_dense_matrices",
        "include_triangular_diagnostics",
        "context_trust_enabled",
        "guard_check_enabled",
        "model",
        "query_prompt_name",
        "document_prompt_name",
        "verification_samples",
        "profile",
        "content_type",
        "structured_verification",
        "risk_band_stratum",
        # Code-lane fields (ignored by the RAG path).
        "scoring_mode",
        "session_id",
        "response_language_hint",
        "emit_chunk_ownership",
        "session_state",
        "heatmap_format",
        "auto_decide",
        "runtime_head_features",
        "trajectory_features",
        "memory_state",
        "memory_policy",
        "enable_memory_shadow",
        "apply_memory_context",
        # Corpus router: tenant-declared override for the per-class
        # calibration bundle. Ignored when absent (classifier infers).
        "corpus_type",
        # Per-request language + NLI premise-selection overrides. The
        # service.py layer resolves ``language`` against langdetect when
        # absent / "auto" and applies German balanced defaults
        # (top_k=2, concat=False, max-aggregate). The runpod handler is
        # a transport shim and does not interpret these; they flow
        # straight into ``GroundednessRequest``.
        "language",
        # Per-request NLI knobs. ``nli_max_claims`` defaults to 0
        # (unlimited) on the runtime so a long enterprise RAG answer
        # is analysed end-to-end and the heatmap covers every
        # sentence; the latency budget is the real safety net for
        # very long answers. Pass an explicit positive integer here
        # to force a hard cap (cost-sensitive batch jobs).
        "nli_max_claims",
        "nli_top_k_premises",
        "nli_premise_concat",
        "nli_premise_aggregate",
    )
    for key in passthrough_keys:
        if key in input_data:
            payload[key] = input_data[key]

    verbose = bool(input_data.get("verbose", False))
    return GroundednessRequest.model_validate(payload), verbose


def _compact_response(response: GroundednessResponse, *, verbose: bool) -> dict[str, Any]:
    scores = response.scores
    guardian_active = scores.guardian_aggregate is not None
    primary_metric = (
        "groundedness_v2" if scores.groundedness_v2 is not None else scores.primary_name
    )
    score = scores.groundedness_v2 if scores.groundedness_v2 is not None else scores.primary_score

    if guardian_active:
        band = "green" if scores.grounded else "red"
        score_channels = {
            "guardian_aggregate": scores.guardian_aggregate,
            "groundedness_v2": scores.groundedness_v2,
        }
    else:
        band = scores.risk_band
        score_channels = {
            "primary": scores.primary_score,
            "reverse_context": scores.reverse_context,
            "reverse_context_calibrated": scores.reverse_context_calibrated,
            "literal_guarded": scores.literal_guarded,
            "nli_aggregate": scores.nli_aggregate,
            "guardian_aggregate": scores.guardian_aggregate,
            "semantic_entropy_aggregate": scores.semantic_entropy_aggregate,
            "structured_source": scores.structured_source,
            "structured_source_guarded": scores.structured_source_guarded,
            "groundedness_v2": scores.groundedness_v2,
            "consensus_hardened": scores.consensus_hardened,
        }
    if scores.context_trust_score is not None:
        score_channels["context_trust"] = scores.context_trust_score
    if scores.context_trust_max_risk is not None:
        score_channels["context_trust_max_risk"] = scores.context_trust_max_risk
    result: dict[str, Any] = {
        "success": True,
        "score": float(score),
        "primary_metric": primary_metric,
        "band": band,
        "grounded": scores.grounded,
        "groundedness_v2": scores.groundedness_v2,
        "guardian_aggregate": scores.guardian_aggregate,
        "score_channels": score_channels,
    }
    if guardian_active and scores.guardian_segments is not None:
        result["guardian_segments"] = [
            {"text": seg.text, "score": seg.score, "grounded": seg.grounded}
            for seg in scores.guardian_segments
        ]
    if not guardian_active:
        result.update({
            "reverse_context_calibrated": scores.reverse_context_calibrated,
            "literal_guarded": scores.literal_guarded,
            "structured_score": scores.structured_source,
            "structured_source_guarded": scores.structured_source_guarded,
            "structured_source_detected": scores.structured_source_detected,
            "nli_aggregate": scores.nli_aggregate,
            "semantic_entropy_aggregate": scores.semantic_entropy_aggregate,
            "semantic_entropy_raw": scores.semantic_entropy_raw,
            "semantic_entropy_sample_count": scores.semantic_entropy_sample_count,
        })
    result.update({
        "context_trust_score": scores.context_trust_score,
        "context_trust_suspicious_count": scores.context_trust_suspicious_count,
        "context_trust_blocked_count": scores.context_trust_blocked_count,
        "context_trust_max_risk": scores.context_trust_max_risk,
        "context_coverage_ratio": scores.context_coverage_ratio,
        "context_coverage_threshold": scores.context_coverage_threshold,
        "context_unused_ratio": scores.context_unused_ratio,
        "context_uncertain_ratio": scores.context_uncertain_ratio,
        "context_usage_ratio": scores.context_usage_ratio,
        "support_units_usage": {
            "used": scores.support_units_usage_used,
            "unused": scores.support_units_unused,
            "uncertain": scores.support_units_uncertain,
        },
        "support_units": [
            {
                "support_id": unit.support_id,
                "usage_state": unit.usage_state.value,
                "usage_confidence": unit.usage_confidence,
                "unused_confidence": unit.unused_confidence,
                "coverage_score": unit.coverage_score,
                "used": unit.used,
                "context_trust_state": unit.context_trust_state,
                "context_trust_score": unit.context_trust_score,
                "context_trust_labels": [
                    label.model_dump(mode="json") for label in unit.context_trust_labels
                ],
                "context_trust_spans": [
                    span.model_dump(mode="json") for span in unit.context_trust_spans
                ],
            }
            for unit in response.support_units
        ],
        "latency_ms": float(response.time_ms),
        "version": _config.version if _config else __version__,
        "scoring_mode": response.scoring_mode.value if response.scoring_mode else None,
        "profile": response.profile.value if response.profile else None,
        "effective_profile": (
            response.effective_profile.value if response.effective_profile else None
        ),
        "session_id": response.session_id,
    })
    if response.profile_diagnostics:
        result["profile_diagnostics"] = dict(response.profile_diagnostics)
    if response.context_trust_diagnostics is not None:
        result["context_trust_diagnostics"] = response.context_trust_diagnostics.model_dump(mode="json")
    if response.scoring_mode == ScoringMode.CODE:
        result["code_lane"] = {
            "composite_score": scores.composite_phantom_score,
            "phantom_probability": scores.composite_phantom_probability,
            "phantom_verdict": scores.composite_phantom_verdict,
            "literal_novelty_min": scores.literal_novelty_min,
            "literal_novelty_missing_count": scores.literal_novelty_missing_count,
            "ast_phantom_symbol_count": scores.ast_phantom_symbol_count,
            "ast_literal_drift_count": scores.ast_literal_drift_count,
            "ast_phantom_verdict": scores.ast_phantom_verdict,
            "nli_contradiction_prob_max": scores.nli_contradiction_prob_max,
            "nli_cascade_triggered": scores.nli_cascade_triggered,
            "dead_weight_ratio": scores.dead_weight_ratio,
            "dead_weight_file_count": scores.dead_weight_file_count,
        }
        if response.code_lane_diagnostics is not None:
            diag = response.code_lane_diagnostics
            result["code_lane"]["diagnostics"] = {
                "config": diag.config,
                "total_latency_ms": diag.total_latency_ms,
                "component_latency_ms": diag.component_latency_ms,
                "file_attribution": (
                    diag.file_attribution.model_dump() if diag.file_attribution else None
                ),
                "ast": diag.ast.model_dump() if diag.ast else None,
                "literal_novelty": (
                    diag.literal_novelty.model_dump() if diag.literal_novelty else None
                ),
                "nli_cascade": (diag.nli_cascade.model_dump() if diag.nli_cascade else None),
                "composite": diag.composite.model_dump() if diag.composite else None,
            }
        # Opt-in caller-portable session blob + derived signals. The
        # client is expected to round-trip ``next_session_state`` verbatim
        # on the next turn; ``session_signals`` is advisory.
        if response.next_session_state is not None:
            result["next_session_state"] = response.next_session_state.model_dump(mode="json")
        if response.session_signals is not None:
            result["session_signals"] = response.session_signals.model_dump(mode="json")
    # Lane-neutral file attribution — populated for both RAG and code
    # lanes so downstream dashboards have one canonical field to read
    # from regardless of ``scoring_mode``.
    if response.file_attribution is not None:
        result["file_attribution"] = response.file_attribution.model_dump(mode="json")
    if response.heatmap is not None:
        result["heatmap"] = response.heatmap.model_dump(mode="json")
    if response.heatmap_html is not None:
        result["heatmap_html"] = response.heatmap_html
    if response.reason:
        result["reason"] = response.reason
    if response.warnings:
        result["warnings"] = list(response.warnings)
    if response.amber_escalation is not None:
        result["amber_escalation"] = response.amber_escalation.model_dump(mode="json")
    if response.corpus_route is not None:
        result["corpus_route"] = response.corpus_route.model_dump(mode="json")
    if response.runtime_decision is not None:
        result["runtime_decision"] = response.runtime_decision.model_dump(mode="json")
    if response.runtime_head_features is not None:
        result["runtime_head_features"] = dict(response.runtime_head_features)
    if response.runtime_feature_source is not None:
        result["runtime_feature_source"] = response.runtime_feature_source
    if response.runtime_feature_missing_groups:
        result["runtime_feature_missing_groups"] = list(response.runtime_feature_missing_groups)
    if verbose:
        result["full"] = response.model_dump(mode="json")
    return result


def _score_response(
    response: GroundednessResponse,
    *,
    verbose: bool,
    response_format: str,
) -> dict[str, Any]:
    """Return the requested RunPod scoring envelope.

    ``compact`` is the legacy dashboard/benchmark shape. ``canonical`` mirrors
    the other RunPod product actions by putting the Pydantic response dump under
    ``result`` so SDKs and gateways can share one typed contract.
    """

    if response_format == "canonical":
        return {
            "success": True,
            "action": "score",
            "result": response.model_dump(mode="json"),
            "version": _config.version if _config else __version__,
        }
    return _compact_response(response, verbose=verbose)


def _response_format(input_data: dict[str, Any]) -> str:
    raw = str(input_data.get("response_format") or "").strip().lower()
    if raw in {"canonical", "full", "model"}:
        return "canonical"
    return "compact"


def _resolve_request_id(input_data: dict[str, Any]) -> str:
    """Derive a stable request id from the RunPod input envelope.

    Source priority: ``auth.request_id`` → ``input.request_id`` →
    monotonic auto-id.  Every response must carry this so callers
    can correlate scoring results, error payloads, and audit rows.
    """
    auth = input_data.get("auth") if isinstance(input_data, dict) else None
    request_id = auth.get("request_id") if isinstance(auth, dict) else None
    request_id = request_id or input_data.get("request_id")
    if not request_id:
        request_id = f"auto-{int(time.monotonic_ns())}"
    return str(request_id)


def _emit_audit_record(
    *,
    input_data: dict[str, Any],
    response: dict[str, Any],
    request: GroundednessRequest,
) -> None:
    """Emit an append-only audit-log record for this score (Plan B6).

    The writer is a no-op unless ``LATENCE_TRACE_AUDIT_LOG_DIR`` is set,
    which keeps the default RunPod / self-hosted behaviour unchanged.
    When enabled, records are enqueued on a background thread so the
    scoring hot path pays only ~80us per call (see
    ``tests/middleware/test_audit_log.py::test_write_is_fast_on_hot_path``).
    """

    try:
        from latence_trace.middleware import audit_log as _audit_log_mod
    except Exception:  # pragma: no cover - import failure must not fail a score
        return
    tenant_id = None
    request_id = None
    auth = input_data.get("auth") if isinstance(input_data, dict) else None
    if isinstance(auth, dict):
        tenant_id = auth.get("tenant_id")
        request_id = auth.get("request_id")
    tenant_id = tenant_id or input_data.get("tenant_id")
    request_id = request_id or input_data.get("request_id") or getattr(request, "request_id", None)
    if not request_id:
        # Fall back to a monotonic-unique request id so every row is
        # distinct even if the caller did not supply one.
        request_id = f"auto-{int(time.monotonic_ns())}"
    data_class = input_data.get("data_class") or "standard"
    _audit_log_mod.write_audit_record(
        request_id=str(request_id),
        tenant_id=str(tenant_id) if tenant_id else None,
        payload={
            "query": getattr(request, "query", None),
            "response": getattr(request, "response", None),
            "raw_context": getattr(request, "raw_context", None),
            "scoring_mode": getattr(request.scoring_mode, "value", None)
            if getattr(request, "scoring_mode", None) is not None
            else None,
        },
        response=response,
        data_class=str(data_class),
    )


def _log_turn_event(
    *,
    request: GroundednessRequest,
    response: GroundednessResponse,
    duration_ms: float,
) -> None:
    """Emit one structured per-turn log line (no PII, PII-safe labels only).

    The payload is the minimal set of fields an IDE-plugin dashboard
    needs to plot lane latency, cascade fire rate, and phantom verdicts
    over time. ``session_id`` is echoed verbatim so plugins can bucket
    turn events client-side; callers must supply a hashed token rather
    than a raw user identifier (documented on the request model).
    """
    _ = request
    diag = response.code_lane_diagnostics
    scores = response.scores
    component = dict((diag.component_latency_ms if diag else {}) or {})
    lane = response.scoring_mode.value if response.scoring_mode else "rag"
    cascade_fired = bool(scores.nli_cascade_triggered)
    phantom_verdict = scores.composite_phantom_verdict
    ast_parser_backend: str | None = None
    ast_language: str | None = None
    if diag and diag.ast is not None:
        ast_parser_backend = getattr(diag.ast, "parser_backend", None)
        ast_language = getattr(diag.ast, "language", None)
    # Caller-carried session signals (opt-in, derived from the pure
    # transform in ``latence_trace.core.code_lane.session``). We log
    # the *signals*, not the opaque state blob, so the log volume stays
    # bounded and no per-file paths leak into the metric store.
    signals = response.session_signals
    logger.info(
        "groundedness_turn",
        extra={
            "lane": lane,
            "session_id": response.session_id,
            "cascade_fired": cascade_fired,
            "phantom_verdict": phantom_verdict,
            "phantom_probability": scores.composite_phantom_probability,
            "dead_weight_ratio": scores.dead_weight_ratio,
            "ast_parser_backend": ast_parser_backend,
            "ast_language": ast_language,
            "nli_ms": component.get("nli_ms"),
            "ast_ms": component.get("ast_ms"),
            "scorer_ms": component.get("scorer_ms"),
            "composite_ms": component.get("composite_ms"),
            "file_attribution_ms": component.get("file_attribution_ms"),
            "literal_novelty_ms": component.get("literal_novelty_ms"),
            "total_ms": float(duration_ms),
            "risk_band": scores.risk_band,
            "session_total_turns": (signals.total_turns if signals else None),
            "session_drift_z": (signals.drift_z_score if signals else None),
            "session_ema_groundedness": (signals.ema_groundedness if signals else None),
            "session_cascade_density": (signals.cascade_density if signals else None),
            "session_phantom_rate": (signals.phantom_rate if signals else None),
            "session_red_streak": (signals.red_streak if signals else None),
            "session_recommendation": (signals.recommendation if signals else None),
        },
    )
    # Loud WARNING when a code-lane request landed on the regex
    # fallback for a language tree-sitter was supposed to cover. This
    # must never fire in production — if it does, grammars are not
    # installed and the AST drift signal has degraded in quality.
    if lane == "code" and ast_parser_backend is not None and ast_parser_backend != "tree_sitter":
        logger.warning(
            "ast_regex_fallback_on_supported_lang",
            extra={
                "lane": lane,
                "ast_parser_backend": ast_parser_backend,
                "ast_language": ast_language,
                "session_id": response.session_id,
            },
        )
    # Prometheus counters — keep cardinality bounded; labels are
    # enum-like so no PII leaks into the metric store.
    try:
        LANE_REQUEST_COUNT.labels(lane=lane).inc()
        if cascade_fired:
            CASCADE_FIRE_COUNT.labels(lane=lane).inc()
        if phantom_verdict is not None:
            PHANTOM_VERDICT_COUNT.labels(verdict="true" if phantom_verdict else "false").inc()
    except Exception:  # pragma: no cover - metrics must never fail a turn
        logger.exception("groundedness_turn_metrics_failed")


def _service_error_payload(
    message: str,
    *,
    error_code: str,
    hint: str | None = None,
    status_code: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "success": False,
        "error": message,
        "error_code": error_code,
    }
    if hint:
        payload["hint"] = hint
    if status_code is not None:
        payload["status_code"] = status_code
    if _config and _config.docs_url:
        payload["docs_url"] = _config.docs_url
    return payload


async def _handle_rollup(input_data: dict[str, Any]) -> dict[str, Any]:
    """Dispatch for ``action == "rollup"``.

    Stateless, CPU-only, sub-ms typical. No semaphore or threadpool
    needed — the aggregation runs inline on the asyncio loop.
    """
    initialize()
    service = _service
    config = _config
    if service is None or config is None:
        return _service_error_payload(
            "RunPod worker was not initialized",
            error_code="service_error",
            status_code=500,
        )
    try:
        payload = dict(input_data)
        payload.pop("action", None)
        rollup_request = RollupRequest.model_validate(payload)
    except PydanticValidationError as exc:
        return _service_error_payload(
            str(exc),
            error_code="validation_error",
            hint="Pass a list of per-turn records under 'turns'.",
            status_code=400,
        )
    try:
        loop = asyncio.get_running_loop()
        response: RollupResponse = await asyncio.wait_for(
            loop.run_in_executor(None, service.rollup, rollup_request),
            timeout=config.rollup_request_timeout_s,
        )
    except asyncio.TimeoutError:
        return _service_error_payload(
            f"Rollup exceeded {config.rollup_request_timeout_s:.2f}s execution timeout",
            error_code="job_timeout",
            hint="Split the session into smaller windows or raise LATENCE_TRACE_ROLLUP_REQUEST_TIMEOUT_S.",
            status_code=504,
        )
    except ServiceError as exc:
        return _service_error_payload(
            str(exc),
            error_code=exc.error_code,
            hint=getattr(exc, "hint", None),
            status_code=getattr(exc, "status_code", None),
        )
    except Exception as exc:  # pragma: no cover - runtime safeguard
        logger.exception("rollup_failed")
        return _service_error_payload(
            str(exc),
            error_code="service_error",
            status_code=500,
        )
    return {
        "success": True,
        "action": "rollup",
        "rollup": response.model_dump(mode="json"),
        "version": config.version if config else __version__,
    }


def _build_compliance_request(input_data: dict[str, Any]) -> ComplianceRedactionRequest:
    payload = dict(input_data)
    payload.pop("action", None)
    payload.pop("endpoint_id", None)
    config = payload.pop("config", None)
    if isinstance(config, dict):
        for key, value in config.items():
            payload.setdefault(key, value)
    if "mode" not in payload and "label_mode" in payload:
        payload["mode"] = payload.pop("label_mode")
    return ComplianceRedactionRequest.model_validate(payload)


async def _handle_compliance_redaction(input_data: dict[str, Any]) -> dict[str, Any]:
    initialize()
    service = _compliance_service
    config = _config
    if service is None or config is None:
        return _service_error_payload(
            "RunPod worker was not initialized",
            error_code="service_error",
            status_code=500,
        )
    try:
        request = _build_compliance_request(input_data)
    except PydanticValidationError as exc:
        return _service_error_payload(
            str(exc),
            error_code="validation_error",
            hint=ComplianceServiceError.hint,
            status_code=400,
        )
    except Exception as exc:
        return _service_error_payload(
            str(exc),
            error_code="validation_error",
            hint=ComplianceServiceError.hint,
            status_code=400,
        )

    executor = _get_request_executor(config)
    semaphore = _get_compliance_semaphore(config)
    started = time.perf_counter()
    try:
        async with semaphore:
            loop = asyncio.get_running_loop()
            response: ComplianceRedactionResponse = await asyncio.wait_for(
                loop.run_in_executor(executor, service.redact, request),
                timeout=config.compliance_request_timeout_s,
            )
    except asyncio.TimeoutError:
        return _service_error_payload(
            f"Compliance redaction exceeded {config.compliance_request_timeout_s:.2f}s execution timeout",
            error_code="job_timeout",
            hint="Retry with a smaller input or raise LATENCE_TRACE_COMPLIANCE_REQUEST_TIMEOUT_S.",
            status_code=504,
        )
    except ComplianceServiceError as exc:
        return _service_error_payload(
            str(exc),
            error_code=exc.error_code,
            hint=getattr(exc, "hint", None),
            status_code=getattr(exc, "status_code", None),
        )
    except Exception as exc:
        logger.exception("compliance_redaction_failed")
        return _service_error_payload(
            str(exc),
            error_code="service_error",
            hint="Inspect compliance GLiNER endpoint health and worker logs.",
            status_code=500,
        )

    duration_ms = (time.perf_counter() - started) * 1000.0
    logger.info(
        "compliance_redaction_turn",
        extra={
            "duration_ms": round(duration_ms, 2),
            "chunks_processed": response.chunks_processed,
            "entity_count": response.entity_count,
            "label_mode": response.label_mode,
            "selected_categories": response.selected_categories,
            "redacted": request.redact,
            "redaction_mode": request.redaction_mode if request.redact else "none",
        },
    )
    return {
        "success": True,
        "action": "redact",
        "result": response.model_dump(mode="json"),
        "version": config.version,
    }


async def _handle_compression(input_data: dict[str, Any]) -> dict[str, Any]:
    initialize()
    service = _compression_service
    config = _config
    if service is None or config is None:
        return _service_error_payload(
            "Compression service is not initialized",
            error_code="service_unavailable",
            hint="Retry after worker startup completes.",
            status_code=503,
        )
    try:
        payload = dict(input_data)
        payload.pop("action", None)
        payload.pop("endpoint_id", None)
        request = CompressionRequest.model_validate(payload)
    except PydanticValidationError as exc:
        return _service_error_payload(
            str(exc),
            error_code="validation_error",
            hint="Send action='compress' with text or messages.",
            status_code=400,
        )
    try:
        response: CompressionResponse = await asyncio.wait_for(
            service.compress(request),
            timeout=config.compression_request_timeout_s,
        )
    except asyncio.TimeoutError:
        return _service_error_payload(
            f"Compression exceeded {config.compression_request_timeout_s:.2f}s execution timeout",
            error_code="job_timeout",
            hint="Retry with a smaller payload or raise LATENCE_TRACE_COMPRESSION_REQUEST_TIMEOUT_S.",
            status_code=504,
        )
    except Exception as exc:  # pragma: no cover - runtime safeguard
        logger.exception("compression_failed")
        return _service_error_payload(
            str(exc),
            error_code="service_error",
            status_code=500,
        )
    return {
        "success": True,
        "action": "compress",
        "result": response.model_dump(mode="json"),
        "version": config.version,
    }


async def _handle_memory_update(input_data: dict[str, Any]) -> dict[str, Any]:
    try:
        payload = dict(input_data)
        payload.pop("action", None)
        payload.pop("endpoint_id", None)
        payload = await _maybe_compress_memory_payload(payload)
        request = MemoryUpdateRequest.model_validate(payload)
        response = update_memory(request)
    except PydanticValidationError as exc:
        return _service_error_payload(
            str(exc),
            error_code="validation_error",
            hint="Send action='memory.update' with turn_text and optional prior_memory_state.",
            status_code=400,
        )
    except Exception as exc:  # pragma: no cover - runtime safeguard
        logger.exception("memory_update_failed")
        return _service_error_payload(
            str(exc),
            error_code="service_error",
            status_code=500,
        )
    return {
        "success": True,
        "action": "memory.update",
        "result": response.model_dump(mode="json"),
        "version": _config.version if _config else __version__,
    }


async def _handle_session_action(input_data: dict[str, Any], action: str) -> dict[str, Any]:
    initialize()
    service = _session_service
    config = _config
    if service is None or config is None:
        return _service_error_payload(
            "TRACE session service unavailable",
            error_code="session_unavailable",
            status_code=503,
        )
    payload = dict(input_data)
    payload.pop("action", None)
    payload.pop("endpoint_id", None)
    session_id = str(payload.pop("session_id", "") or "").strip()
    try:
        if action == "session.create":
            if session_id:
                payload["session_id"] = session_id
            response = service.create(TraceSessionCreateRequest.model_validate(payload))
        elif action == "session.get":
            response = service.get(session_id)
        elif action == "session.event":
            response = service.append_event(
                session_id,
                TraceSessionEventRequest.model_validate(payload),
            )
        elif action == "session.score":
            loop = asyncio.get_running_loop()
            request = TraceSessionScoreRequest.model_validate(payload)
            response = await asyncio.wait_for(
                loop.run_in_executor(
                    _get_request_executor(config),
                    service.score,
                    session_id,
                    request,
                ),
                timeout=config.request_timeout_s,
            )
        elif action == "session.context":
            response = service.context(session_id)
        elif action == "session.source":
            response = service.source(
                session_id,
                str(payload.get("source_id") or ""),
                include_raw=bool(payload.get("include_raw", False)),
            )
        elif action == "session.repair":
            response = service.repair(
                session_id,
                TraceSessionRepairRequest.model_validate(payload),
            )
        elif action == "session.rollup":
            response = service.rollup(
                session_id,
                TraceSessionRollupRequest.model_validate(payload),
            )
        elif action == "session.close":
            response = service.close(session_id)
        else:
            return _service_error_payload(
                f"Unknown session action: {action}",
                error_code="invalid_action",
                status_code=400,
            )
    except PydanticValidationError as exc:
        return _service_error_payload(
            str(exc),
            error_code="validation_error",
            hint="Send a valid TRACE session payload for the selected session action.",
            status_code=400,
        )
    except asyncio.TimeoutError:
        return _service_error_payload(
            f"TRACE session action exceeded {config.request_timeout_s:.2f}s execution timeout",
            error_code="job_timeout",
            status_code=504,
        )
    except ServiceError as exc:
        return _service_error_payload(
            str(exc),
            error_code=getattr(exc, "error_code", "service_error"),
            hint=getattr(exc, "hint", None),
            status_code=getattr(exc, "status_code", None),
        )
    except Exception as exc:  # pragma: no cover - runtime safeguard
        logger.exception("session_action_failed")
        return _service_error_payload(str(exc), error_code="service_error")
    return {
        "success": True,
        "action": action,
        "result": response.model_dump(mode="json"),
        "version": config.version,
    }


async def _maybe_compress_memory_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not payload.get("enable_ingress_compression"):
        return payload
    raw_context = str(payload.get("raw_context") or "")
    min_tokens = int(payload.get("ingress_compression_min_tokens") or 512)
    if not raw_context.strip() or len(raw_context.split()) < min_tokens:
        return payload
    service = _compression_service
    config = _config
    if service is None or config is None:
        return payload
    domain = str(payload.get("memory_domain") or "rag")
    force_tokens = extract_exact_critical_terms(
        "\n".join(
            item
            for item in [
                str(payload.get("query_text") or ""),
                str(payload.get("response_text") or ""),
                raw_context,
            ]
            if item
        ),
        domain=domain,
    )[:160]
    compression_rate = payload.get("ingress_compression_rate")
    if compression_rate is None:
        compression_rate = _memory_ingress_compression_rate(domain)
    request = CompressionRequest(
        text=raw_context,
        compression_rate=float(compression_rate),
        chunk_size=config.compression_default_chunk_size,
        force_tokens=force_tokens,
        preserve_exact=force_tokens,
        force_preserve_digit=config.compression_force_preserve_digit,
        fallback_mode=config.compression_fallback_mode,
        apply_toon=bool(payload.get("ingress_apply_toon", domain != "code")),
    )
    compressed = await service.compress(request)
    if not compressed.compressed_text.strip():
        return payload
    sidecar = _memory_ingress_sidecar(domain, force_tokens)
    payload["raw_context"] = "\n".join(part for part in [sidecar, compressed.compressed_text] if part)
    payload.setdefault("trace_signals", {})
    if isinstance(payload["trace_signals"], dict):
        payload["trace_signals"]["ingress_compression"] = compressed.model_dump(mode="json")
    return payload


def _memory_ingress_compression_rate(domain: str) -> float:
    return {
        "rag": 0.40,
        "search": 0.40,
        "grounding": 0.40,
        "tool": 0.40,
        "code": 0.40,
        "chat": 0.40,
    }.get(domain, 0.40)


def _memory_ingress_sidecar(domain: str, terms: list[str], *, max_terms: int = 40) -> str:
    chunks = []
    for start in range(0, len(terms), max_terms):
        chunks.append(f"{domain}_exact_index " + " ".join(terms[start : start + max_terms]))
    return "\n".join(chunks)


async def handler(job: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(job, dict):
        return _service_error_payload(
            f"Invalid event type: {type(job).__name__}",
            error_code="invalid_event",
            hint="Send a JSON object with an 'input' field.",
            status_code=400,
        )

    initialize()

    input_data = job.get("input")
    if not isinstance(input_data, dict):
        return _service_error_payload(
            "Missing or invalid 'input' field",
            error_code="invalid_input",
            hint="Wrap the request payload under input={query, context, response}.",
            status_code=400,
        )

    if input_data.get("endpoint_id") == "_health" or bool(input_data.get("health")):
        return _health_payload()

    # Optional action dispatch — defaults to "score". Compliance redaction
    # is intentionally a separate runtime branch so it cannot perturb
    # groundedness latency or response shape.
    action = str(input_data.get("action") or "score").strip().lower()
    endpoint_id = str(input_data.get("endpoint_id") or "").strip().lower()
    if action == "rollup":
        return await _handle_rollup(input_data)
    if action in {"redact", "compliance_redaction"} or endpoint_id in {
        "compliance_redaction",
        "redaction",
    }:
        return await _handle_compliance_redaction(input_data)
    if action in {"compress", "compression"} or endpoint_id == "compression":
        return await _handle_compression(input_data)
    if action in {"memory.update", "memory_update"} or endpoint_id == "memory":
        return await _handle_memory_update(input_data)
    if action in {
        "session.create",
        "session.get",
        "session.event",
        "session.score",
        "session.context",
        "session.source",
        "session.repair",
        "session.rollup",
        "session.close",
    } or endpoint_id == "trace_session":
        return await _handle_session_action(input_data, action)
    if action not in {"score", ""}:
        return _service_error_payload(
            f"Unknown action: {action}",
            error_code="invalid_action",
            hint="Set action to 'score' (default), 'rollup', 'redact', 'compress', 'memory.update', or a session.* action.",
            status_code=400,
        )

    try:
        request, verbose = _build_request(input_data)
    except PydanticValidationError as exc:
        return _service_error_payload(
            str(exc),
            error_code="validation_error",
            hint=ValidationError.hint,
            status_code=400,
        )
    except Exception as exc:
        return _service_error_payload(
            str(exc),
            error_code="validation_error",
            hint=ValidationError.hint,
            status_code=400,
        )

    # Pre-compute the lane so exception handlers below can refer to it
    # even if initialisation failed mid-flight.
    lane: ScoringMode = request.scoring_mode or ScoringMode.RAG

    try:
        config = _config
        service = _service
        if config is None or service is None:
            raise RuntimeError("RunPod worker was not initialized")
        executor = _get_request_executor(config)
        semaphore = _get_lane_semaphore(config, lane)
        started = time.perf_counter()
        if semaphore.locked():
            # The lane is at capacity — surface as a metric so the
            # operator dashboard can alert on sustained backpressure.
            try:
                BUDGET_EXCEEDED_COUNT.labels(lane=lane.value).inc()
            except Exception:  # pragma: no cover
                logger.exception("budget_exceeded_metric_failed")
        # Code-lane calls get a much tighter ceiling (150ms p95 SLO);
        # RAG stays on the long-running timeout so document-heavy
        # requests still fit.
        effective_timeout = (
            config.code_request_timeout_s if lane == ScoringMode.CODE else config.request_timeout_s
        )
        async with semaphore:
            loop = asyncio.get_running_loop()
            response = await asyncio.wait_for(
                loop.run_in_executor(executor, service.groundedness, request),
                timeout=effective_timeout,
            )
            duration_ms = (time.perf_counter() - started) * 1000.0
            try:
                _log_turn_event(request=request, response=response, duration_ms=duration_ms)
            except Exception:  # pragma: no cover - logging must never fail a turn
                logger.exception("groundedness_turn_log_failed")
            score_response = _score_response(
                response,
                verbose=verbose,
                response_format=_response_format(input_data),
            )
            score_response["request_id"] = _resolve_request_id(input_data)
            try:
                _emit_audit_record(input_data=input_data, response=score_response, request=request)
            except Exception:  # pragma: no cover - audit log must never fail a turn
                logger.exception("audit_log_emit_failed")
            return score_response
    except asyncio.TimeoutError:
        if lane == ScoringMode.CODE:
            timeout_value = _config.code_request_timeout_s if _config else 2.0
            hint = "Retry with a smaller request or increase LATENCE_TRACE_CODE_REQUEST_TIMEOUT_S."
        else:
            timeout_value = _config.request_timeout_s if _config else 120
            hint = "Retry with a smaller request or increase LATENCE_TRACE_RUNPOD_REQUEST_TIMEOUT."
        err = _service_error_payload(
            f"Job exceeded {timeout_value}s execution timeout",
            error_code="job_timeout",
            hint=hint,
            status_code=504,
        )
        err["request_id"] = _resolve_request_id(input_data)
        return err
    except ServiceError as exc:
        err = _service_error_payload(
            str(exc),
            error_code=exc.error_code,
            hint=getattr(exc, "hint", None),
            status_code=getattr(exc, "status_code", None),
        )
        err["request_id"] = _resolve_request_id(input_data)
        return err
    except Exception as exc:  # pragma: no cover - runtime safeguard
        logger.exception("RunPod request failed")
        err = _service_error_payload(
            str(exc),
            error_code="service_error",
            hint="Inspect worker logs; both vLLM lanes and the reranker are process-wide singletons.",
            status_code=500,
        )
        err["request_id"] = _resolve_request_id(input_data)
        return err


__all__ = [
    "SUPPORTED_RUNPOD_ACTIONS",
    "create_config",
    "handler",
    "initialize",
    "shutdown",
]


if __name__ == "__main__":  # pragma: no cover - exercised in container
    if runpod is None:
        raise RuntimeError("runpod package is required to start the serverless worker")
    initialize()
    atexit.register(shutdown)
    runpod.serverless.start(
        {
            "handler": handler,
            "concurrency_modifier": lambda _current: _config.max_concurrency if _config else 64,
        }
    )
