from __future__ import annotations

import asyncio
import importlib.util
import os
import sys
import threading
import time
from contextlib import suppress
from pathlib import Path

import pytest

from latence_trace.api.compliance_models import (
    ComplianceEntity,
    ComplianceRedactionResponse,
    ComplianceUsage,
)
from latence_trace.api.models import (
    AttributionMode,
    CollectionKind,
    CorpusRouteDiagnostics,
    GroundednessEligibility,
    GroundednessResponse,
    GroundednessScores,
    GroundednessSupportUnit,
    GroundednessUsageState,
    RuntimeDecisionRecord,
    TraceRuntimeProfile,
)

_RUNPOD_DIR = Path(__file__).resolve().parents[1] / "runpod"
# Temporarily expose ``runpod/`` on sys.path so ``handler.py``'s
# ``from server import ManagedVllmServer`` resolves to ``runpod/server.py``.
# We must tear this down immediately after loading because ``runpod/server.py``
# shadows the workspace-root ``server/`` *package* in ``sys.modules``; leaving
# it in place breaks later tests that need ``server.main`` (the real FastAPI
# entrypoint) for profile / agent-help coverage.
_RUNPOD_PATH_INSERTED = False
if str(_RUNPOD_DIR) not in sys.path:
    sys.path.insert(0, str(_RUNPOD_DIR))
    _RUNPOD_PATH_INSERTED = True

_REAL_SERVER_MODULE = sys.modules.pop("server", None)
_RUNPOD_SERVER_BACKUP_KEYS = {
    name for name in list(sys.modules.keys()) if name == "server" or name.startswith("server.")
}

_SPEC = importlib.util.spec_from_file_location(
    "latence_trace_runpod_handler",
    _RUNPOD_DIR / "handler.py",
)
assert _SPEC is not None and _SPEC.loader is not None
runpod_handler = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = runpod_handler
_SPEC.loader.exec_module(runpod_handler)

# Handler holds a direct reference to ``ManagedVllmServer``; drop the
# ``server`` module alias that pointed at ``runpod/server.py`` so the
# workspace-root ``server/`` package can be imported fresh by downstream
# tests. Also pull the ``runpod/`` entry back off ``sys.path`` so a new
# ``import server`` resolves to the package, not the shadow module.
_runpod_server_module = sys.modules.pop("server", None)
if _runpod_server_module is not None:
    sys.modules["latence_trace_runpod_vllm_server"] = _runpod_server_module
for _name in list(sys.modules.keys()):
    if (
        _name == "server" or _name.startswith("server.")
    ) and _name not in _RUNPOD_SERVER_BACKUP_KEYS:
        del sys.modules[_name]
if _REAL_SERVER_MODULE is not None:
    sys.modules["server"] = _REAL_SERVER_MODULE
if _RUNPOD_PATH_INSERTED:
    with suppress(ValueError):
        sys.path.remove(str(_RUNPOD_DIR))


class _SlowService:
    def __init__(self, sleep_ms: int = 80) -> None:
        self._sleep_ms = sleep_ms
        self._lock = threading.Lock()
        self.inflight = 0
        self.peak_inflight = 0
        self.calls = 0

    def groundedness(self, request) -> GroundednessResponse:
        with self._lock:
            self.calls += 1
            self.inflight += 1
            self.peak_inflight = max(self.peak_inflight, self.inflight)
        try:
            time.sleep(self._sleep_ms / 1000.0)
            return GroundednessResponse(
                collection="latence-trace",
                mode="raw_context",
                model="test-model",
                scores=GroundednessScores(
                    primary_name="reverse_context",
                    primary_score=0.9,
                    reverse_context=0.9,
                    risk_band="low",
                ),
                response_tokens=[],
                support_units=[],
                top_evidence=[],
                eligibility=GroundednessEligibility(
                    collection_kind=CollectionKind.LATE_INTERACTION,
                    vector_source="encoded_raw_context",
                    storage_compression=None,
                    quantization_mode=None,
                    dequantized=True,
                    user_facing_supported=True,
                    warnings=[],
                ),
                time_ms=float(self._sleep_ms),
                scoring_mode=getattr(request, "scoring_mode", None),
                profile=getattr(request, "profile", None),
                effective_profile=(
                    TraceRuntimeProfile.QUALITY
                    if getattr(request, "profile", None) == TraceRuntimeProfile.QUALITY
                    else TraceRuntimeProfile.STANDARD
                ),
                profile_diagnostics={
                    "effective_profile": (
                        "quality"
                        if getattr(request, "profile", None) == TraceRuntimeProfile.QUALITY
                        else "standard"
                    )
                },
                session_id=getattr(request, "session_id", None),
                attribution_mode=AttributionMode.CLOSED_BOOK,
            )
        finally:
            with self._lock:
                self.inflight -= 1

    def rollup(self, request) -> runpod_handler.RollupResponse:
        return runpod_handler.RollupResponse(
            turns=len(request.turns),
            session_id=request.session_id,
            risk_band_trail=[turn.risk_band for turn in request.turns if turn.risk_band],
        )


class _ComplianceService:
    def __init__(self, sleep_ms: int = 40) -> None:
        self.sleep_ms = sleep_ms
        self.calls = 0
        self.inflight = 0
        self.peak_inflight = 0
        self._lock = threading.Lock()

    def redact(self, request) -> ComplianceRedactionResponse:
        with self._lock:
            self.calls += 1
            self.inflight += 1
            self.peak_inflight = max(self.peak_inflight, self.inflight)
        try:
            time.sleep(self.sleep_ms / 1000.0)
            return ComplianceRedactionResponse(
                original_text=request.text,
                entities=[
                    ComplianceEntity(
                        start=0,
                        end=4,
                        text=request.text[:4],
                        label="person",
                        score=0.9,
                    )
                ],
                entity_count=1,
                unique_labels=["person"],
                redacted_text=None,
                chunks_processed=1,
                labels_used=["person"],
                label_mode=request.mode,
                selected_categories=request.categories,
                processing_time_ms=float(self.sleep_ms),
                timings_ms={"total_ms": float(self.sleep_ms)},
                usage=ComplianceUsage(
                    chunks_processed=1,
                    labels_used=1,
                    mode=request.mode,
                    categories=request.categories,
                ),
            )
        finally:
            with self._lock:
                self.inflight -= 1


def _config(
    *,
    profile: str = "quality",
    max_concurrency: int = 2,
    managed_vllm_enabled: bool = True,
    compression_model: str = "",
    compression_server_enabled: bool = False,
    compliance_gliner_server_enabled: bool = True,
):
    return runpod_handler.WorkerConfig(
        profile=profile,
        version="test",
        request_timeout_s=5,
        code_request_timeout_s=2.0,
        rollup_request_timeout_s=0.25,
        max_concurrency=max_concurrency,
        collection_label="latence-trace",
        service_device="cpu",
        docs_url="",
        managed_vllm_enabled=managed_vllm_enabled,
        colbert_model="lightonai/LateOn",
        colbert_port=18001,
        colbert_gpu_mem=0.2,
        colbert_max_model_len=8192,
        colbert_max_num_seqs=128,
        colbert_max_batched_tokens=8192,
        nli_model="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
        nli_port=18002,
        nli_gpu_mem=0.2,
        nli_max_model_len=512,
        nli_max_num_seqs=128,
        nli_max_batched_tokens=8192,
        compliance_model="knowledgator/gliner-pii-large-v1.0",
        compliance_port=18003,
        compliance_gpu_mem=0.2,
        compliance_max_model_len=768,
        compliance_max_num_seqs=128,
        compliance_max_batched_tokens=8192,
        compliance_threshold=0.5,
        compliance_dataset_path="doubledsbv/pii-replacement-dataset",
        compliance_request_timeout_s=30.0,
        compression_model=compression_model,
        compression_server_enabled=compression_server_enabled,
        compliance_gliner_server_enabled=compliance_gliner_server_enabled,
    )


async def _burst(size: int) -> list[dict]:
    payload = {
        "input": {
            "query_text": "Where was Heinrich born?",
            "raw_context": "Heinrich was born in 1851 in Augsburg.",
            "response_text": "Heinrich was born in Augsburg in 1851.",
        }
    }
    return await asyncio.gather(*(runpod_handler.handler(payload) for _ in range(size)))


def test_create_config_uses_fixed_runpod_max_concurrency(monkeypatch) -> None:
    """Production default is 32 (sized for the 1/8 GPU pod carrying the full
    7-model topology). Foreign caps via ``MAX_CONCURRENCY`` are ignored — the
    handler reads ``LATENCE_TRACE_MAX_CONCURRENCY`` only.
    """
    monkeypatch.setenv("MAX_CONCURRENCY", "7")
    monkeypatch.setenv("LATENCE_TRACE_PROFILE", "quality")
    monkeypatch.delenv("LATENCE_TRACE_MAX_CONCURRENCY", raising=False)
    assert runpod_handler.create_config().max_concurrency == 32

    monkeypatch.setenv("LATENCE_TRACE_MAX_CONCURRENCY", "48")
    assert runpod_handler.create_config().max_concurrency == 48


def test_create_config_pins_vllm_runtime_defaults(monkeypatch) -> None:
    monkeypatch.setenv("LATENCE_TRACE_PROFILE", "quality")
    monkeypatch.delenv("LATENCE_TRACE_COLBERT_MAX_NUM_SEQS", raising=False)
    monkeypatch.delenv("LATENCE_TRACE_COLBERT_MAX_BATCHED_TOKENS", raising=False)
    monkeypatch.delenv("LATENCE_TRACE_NLI_MAX_NUM_SEQS", raising=False)
    monkeypatch.delenv("LATENCE_TRACE_NLI_MAX_BATCHED_TOKENS", raising=False)
    monkeypatch.delenv("LATENCE_TRACE_COMPRESSION_ENDPOINT", raising=False)
    monkeypatch.delenv("LATENCE_TRACE_COMPRESSION_MODEL", raising=False)
    monkeypatch.delenv("LATENCE_TRACE_ENABLE_COMPRESSION_SERVER", raising=False)

    config = runpod_handler.create_config()

    assert config.colbert_max_num_seqs == 128
    assert config.colbert_max_batched_tokens == 8192
    assert config.nli_max_num_seqs == 128
    assert config.nli_max_batched_tokens == 8192
    assert config.compliance_model == "knowledgator/gliner-pii-large-v1.0"
    assert config.compliance_max_model_len == 768
    assert config.compliance_max_num_seqs == 128
    assert config.compliance_max_batched_tokens == 8192
    # Phase 1 / SOTA topology: per-server GPU memory now defaults to
    # ``LATENCE_TRACE_VLLM_GPU_MEM_DEFAULT`` (0.145) so 6 vLLM servers
    # co-host on one 24 GiB GPU at 0.87 total utilisation, leaving the
    # remaining ~3.1 GiB to the 7th model (in-process Llama-Prompt-
    # Guard-2-86M) plus the Triton MaxSim kernels. Per-server env
    # overrides still trump the shared default.
    assert config.colbert_gpu_mem == pytest.approx(0.145)
    assert config.nli_gpu_mem == pytest.approx(0.145)
    assert config.compliance_gpu_mem == pytest.approx(0.145)
    assert config.nli_en_gpu_mem == pytest.approx(0.145)
    assert config.nli_multi_gpu_mem == pytest.approx(0.145)
    assert config.reranker_gpu_mem == pytest.approx(0.145)
    assert config.compression_model == "latence/compression-v0.1"
    assert config.compression_server_enabled is True
    assert config.compression_gpu_mem == pytest.approx(0.145)
    assert config.compression_max_model_len == 8192
    assert config.compression_max_batched_tokens == 8192
    assert config.compression_dtype == "auto"
    assert config.compression_enforce_eager is True


def test_create_config_uses_public_default_compression_model(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("LATENCE_TRACE_COMPRESSION_MODEL", raising=False)
    monkeypatch.delenv("LATENCE_TRACE_COMPRESSION_ENDPOINT", raising=False)
    monkeypatch.delenv("LATENCE_TRACE_ENABLE_COMPRESSION_SERVER", raising=False)
    monkeypatch.setattr(runpod_handler, "_VENDORED_COMPRESSION_MODEL_DIR", tmp_path / "missing-model")

    config = runpod_handler.create_config()

    assert config.compression_model == "latence/compression-v0.1"
    assert config.compression_server_enabled is True


def test_create_config_can_explicitly_disable_managed_compression(monkeypatch) -> None:
    monkeypatch.delenv("LATENCE_TRACE_COMPRESSION_MODEL", raising=False)
    monkeypatch.delenv("LATENCE_TRACE_COMPRESSION_ENDPOINT", raising=False)
    monkeypatch.setenv("LATENCE_TRACE_ENABLE_COMPRESSION_SERVER", "0")

    config = runpod_handler.create_config()

    assert config.compression_model == ""
    assert config.compression_server_enabled is False


def test_create_config_uses_external_compression_endpoint_without_managed_server(monkeypatch) -> None:
    monkeypatch.setenv("LATENCE_TRACE_COMPRESSION_ENDPOINT", "http://compression.internal:8004")
    monkeypatch.setenv("LATENCE_TRACE_COMPRESSION_MODEL", "external-compression")
    monkeypatch.delenv("LATENCE_TRACE_ENABLE_COMPRESSION_SERVER", raising=False)

    config = runpod_handler.create_config()

    assert config.compression_model == "external-compression"
    assert config.compression_server_enabled is False


def test_create_config_preserves_explicit_compression_model_override(monkeypatch) -> None:
    monkeypatch.setenv("LATENCE_TRACE_COMPRESSION_MODEL", "local-or-hf/compression-model")
    monkeypatch.delenv("LATENCE_TRACE_ENABLE_COMPRESSION_SERVER", raising=False)

    config = runpod_handler.create_config()

    assert config.compression_model == "local-or-hf/compression-model"
    assert config.compression_server_enabled is True


def test_create_config_uses_complete_vendored_compression_model(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("LATENCE_TRACE_COMPRESSION_MODEL", raising=False)
    monkeypatch.delenv("LATENCE_TRACE_ENABLE_COMPRESSION_SERVER", raising=False)
    model_dir = tmp_path / "compression_model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"stub")
    monkeypatch.setattr(runpod_handler, "_VENDORED_COMPRESSION_MODEL_DIR", model_dir)

    config = runpod_handler.create_config()

    assert config.compression_model == str(model_dir)
    assert config.compression_server_enabled is True


def test_create_config_can_disable_managed_vllm_for_local_wrapper(monkeypatch) -> None:
    monkeypatch.setenv("LATENCE_TRACE_START_MANAGED_VLLM", "0")

    config = runpod_handler.create_config()

    assert config.managed_vllm_enabled is False


def test_build_servers_pin_requested_vllm_settings() -> None:
    servers = runpod_handler._build_servers(_config(max_concurrency=64))

    assert servers["colbert"].max_num_seqs == 128
    assert servers["colbert"].max_num_batched_tokens == 8192
    assert servers["colbert"].gpu_memory_utilization == 0.2
    assert servers["colbert"].enforce_eager is True
    assert servers["colbert"].quantization is None
    # Legacy mDeBERTa nli server is permanently dropped. The SOTA trio
    # below (nli_en/nli_multi/reranker) replaces it and is asserted in
    # ``test_build_servers_default_topology_drops_legacy_nli``.
    assert "nli" not in servers
    assert servers["compliance_gliner"].model == "knowledgator/gliner-pii-large-v1.0"
    assert servers["compliance_gliner"].io_processor_plugin == "deberta_gliner_io"
    assert servers["compliance_gliner"].plugins == ["deberta_gliner", "deberta_gliner_io"]
    assert servers["compliance_gliner"].max_model_len == 768
    assert servers["compliance_gliner"].gpu_memory_utilization == 0.2
    assert servers["compliance_gliner"].enforce_eager is True
    assert servers["compliance_gliner"].quantization is None


def test_build_servers_only_adds_compression_when_enabled() -> None:
    disabled = runpod_handler._build_servers(_config())
    enabled = runpod_handler._build_servers(
        _config(compression_model="/models/compression", compression_server_enabled=True)
    )

    assert "compression" not in disabled
    assert enabled["compression"].model == "/models/compression"
    assert enabled["compression"].plugins == ["qwen3_compression"]
    assert enabled["compression"].dtype == "auto"
    assert enabled["compression"].quantization is None
    assert enabled["compression"].enforce_eager is True
    assert enabled["compression"].trust_remote_code is False
    # Default GPU utilisation tracks ``LATENCE_TRACE_VLLM_GPU_MEM_DEFAULT``
    # (0.145) — see Phase 1 topology change in handler.py.
    assert enabled["compression"].gpu_memory_utilization == pytest.approx(0.145)
    assert enabled["compression"].max_model_len == 8192
    assert enabled["compression"].max_num_batched_tokens == 8192


def test_build_servers_can_use_external_vllm_mode() -> None:
    assert runpod_handler._build_servers(_config(managed_vllm_enabled=False)) == {}


def test_build_servers_can_disable_compliance_gliner(monkeypatch) -> None:
    """Memory-constrained calibration runs flip the GLiNER PII server off
    via ``LATENCE_TRACE_ENABLE_COMPLIANCE_GLINER_SERVER=0``. The flag must
    actually skip the server entry while leaving the rest of the topology
    intact.
    """
    full = runpod_handler._build_servers(_config())
    minimal = runpod_handler._build_servers(_config(compliance_gliner_server_enabled=False))

    assert "compliance_gliner" in full
    assert "compliance_gliner" not in minimal
    for name in ("colbert", "nli_en", "nli_multi", "reranker"):
        assert name in minimal, f"{name} must remain in the minimal topology"


def test_prompt_guard_default_provider_loads_at_boot(monkeypatch) -> None:
    """Llama-Prompt-Guard-2-86M is the 7th GPU model in the production
    topology and must boot in-process by default so the first
    ``context_trust_enabled=true`` request never pays the cold-start
    penalty.
    """
    for var in (
        "LATENCE_TRACE_CONTEXT_TRUST_ENABLED",
        "LATENCE_TRACE_CONTEXT_TRUST_PROVIDER",
    ):
        monkeypatch.delenv(var, raising=False)

    assert runpod_handler._prompt_guard_startup_enabled() is True

    runtime = runpod_handler._context_trust_runtime_config()
    assert runtime["enabled"] is True
    assert runtime["provider"] == "prompt_guard"

    monkeypatch.setenv("LATENCE_TRACE_CONTEXT_TRUST_PROVIDER", "heuristic")
    assert runpod_handler._prompt_guard_startup_enabled() is False

    monkeypatch.setenv("LATENCE_TRACE_CONTEXT_TRUST_PROVIDER", "prompt_guard")
    monkeypatch.setenv("LATENCE_TRACE_CONTEXT_TRUST_ENABLED", "0")
    assert runpod_handler._prompt_guard_startup_enabled() is False


def test_warm_prompt_guard_at_boot_is_defensive_against_gated_repo(monkeypatch) -> None:
    """Llama-Prompt-Guard-2-86M is a Meta gated repo. Operators without a
    HF token bound to a Meta-Llama-accepted account hit a 401
    ``GatedRepoError`` on first download. This must NEVER crash the worker
    boot — the lane is opt-in and the rest of the topology has to stay up.

    The defensive path must:
      1. Swallow the exception so ``initialize()`` keeps booting.
      2. Force the per-request provider env var to ``heuristic`` so the
         per-request lookup degrades cleanly instead of repeatedly
         retrying the same gated download.
      3. Return a structured ``status=skipped`` envelope so health
         endpoints can report the fallback to operators.
    """
    monkeypatch.delenv("LATENCE_TRACE_CONTEXT_TRUST_PROVIDER", raising=False)

    def _explode() -> dict:
        raise RuntimeError(
            "LATENCE_TRACE_CONTEXT_TRUST_PROVIDER=prompt_guard could not load "
            "'meta-llama/Llama-Prompt-Guard-2-86M'. The model may require "
            "Hugging Face gated access and Meta Llama license acceptance"
        )

    monkeypatch.setattr(runpod_handler, "warm_context_trust_runtime", _explode)

    result = runpod_handler._warm_prompt_guard_at_boot()

    assert result["status"] == "skipped"
    assert result["reason"] == "prompt_guard_load_failed"
    assert result["fallback_provider"] == "heuristic"
    assert "Llama-Prompt-Guard-2-86M" in result["model_id"]
    assert os.environ["LATENCE_TRACE_CONTEXT_TRUST_PROVIDER"] == "heuristic"


def test_health_payload_surfaces_selected_model_runtime_config(monkeypatch) -> None:
    config = _config(compression_model="/models/compression", compression_server_enabled=True)
    runpod_handler._config = config
    runpod_handler._servers = {}

    try:
        payload = runpod_handler._health_payload()
    finally:
        runpod_handler._config = None

    compression = payload["model_runtime_config"]["compression"]
    colbert = payload["model_runtime_config"]["colbert"]
    # Legacy ``nli`` block is gone — ``model_runtime_config`` now
    # mirrors the live SOTA trio (``nli_en`` + ``nli_multi`` +
    # ``reranker``). Asserting the absence of the legacy key locks the
    # diagnostics surface to the new contract.
    assert "nli" not in payload["model_runtime_config"]
    nli_en = payload["model_runtime_config"]["nli_en"]
    nli_multi = payload["model_runtime_config"]["nli_multi"]
    reranker = payload["model_runtime_config"]["reranker"]
    compliance = payload["model_runtime_config"]["compliance_gliner"]

    assert colbert["model"] == "lightonai/LateOn"
    assert colbert["io_processor_plugin"] == "moderncolbert_batched_io"
    # ``_config`` factory pins the legacy lane budgets (colbert,
    # compliance) to 0.2 explicitly. The SOTA trio + compression rely
    # on dataclass defaults, which now track ``_DEFAULT_VLLM_GPU_MEM``
    # (0.145) — the Phase 1 shared knob.
    assert colbert["gpu_memory_utilization"] == 0.2
    assert colbert["quantization"] is None
    assert colbert["enforce_eager"] is True
    assert nli_en["model"] == "lytang/MiniCheck-Flan-T5-Large"
    assert nli_en["io_processor_plugin"] == "minicheck_t5_io"
    assert nli_en["enabled"] is True
    assert nli_en["gpu_memory_utilization"] == pytest.approx(0.145)
    assert nli_multi["model"] == "MoritzLaurer/bge-m3-zeroshot-v2.0"
    assert nli_multi["enabled"] is True
    assert nli_multi["extra_args"] == ["--convert", "classify"]
    assert nli_multi["gpu_memory_utilization"] == pytest.approx(0.145)
    assert reranker["model"] == "BAAI/bge-reranker-v2-m3"
    assert reranker["enabled"] is True
    assert reranker["extra_args"] == ["--convert", "classify"]
    assert reranker["gpu_memory_utilization"] == pytest.approx(0.145)
    assert compliance["gpu_memory_utilization"] == 0.2
    assert compliance["quantization"] is None
    assert compliance["enforce_eager"] is True
    assert compression["model"] == "/models/compression"
    assert compression["task"] == "token_classify"
    assert compression["default_compression_rate"] == 0.4
    assert compression["default_chunk_size"] == 4096
    assert compression["force_preserve_digit"] is True
    assert compression["max_model_len"] == 8192
    assert compression["max_num_batched_tokens"] == 8192
    assert compression["gpu_memory_utilization"] == pytest.approx(0.145)
    assert compression["quantization"] is None
    assert compression["enforce_eager"] is True


def test_dev_app_exposes_runpod_wrapper_routes() -> None:
    spec = importlib.util.spec_from_file_location(
        "latence_trace_runpod_dev_app_test",
        _RUNPOD_DIR / "dev_app.py",
    )
    assert spec is not None and spec.loader is not None
    previous_handler = sys.modules.get("handler")
    sys.modules["handler"] = runpod_handler
    try:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        if previous_handler is not None:
            sys.modules["handler"] = previous_handler
        else:
            sys.modules.pop("handler", None)

    routes = {(route.path, frozenset(getattr(route, "methods", []))) for route in module.app.routes}

    assert any(path == "/healthz" and "GET" in methods for path, methods in routes)
    assert any(path == "/run" and "POST" in methods for path, methods in routes)
    assert any(path == "/runsync" and "POST" in methods for path, methods in routes)


def test_initialize_exports_handler_concurrency_to_internal_vllm_clients(monkeypatch) -> None:
    class _FakeServer:
        def __init__(self, name: str, base_url: str) -> None:
            self.name = name
            self.base_url = base_url

        def start(self) -> None:
            return None

        def stop(self) -> None:
            return None

        def health(self) -> dict[str, str]:
            return {"status": "healthy", "name": self.name}

    class _FakeGroundednessService:
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs

        def groundedness(self, request) -> GroundednessResponse:
            return _SlowService(sleep_ms=0).groundedness(request)

    monkeypatch.setattr(runpod_handler, "apply_profile", lambda _profile: None)
    monkeypatch.setattr(
        runpod_handler,
        "_build_servers",
        lambda _config: {
            "colbert": _FakeServer("colbert", "http://127.0.0.1:18001"),
            "nli": _FakeServer("nli", "http://127.0.0.1:18002"),
            "compliance_gliner": _FakeServer("compliance_gliner", "http://127.0.0.1:18003"),
        },
    )
    monkeypatch.setattr(runpod_handler, "GroundednessService", _FakeGroundednessService)
    monkeypatch.setattr(
        runpod_handler.ComplianceRedactionService,
        "from_env",
        classmethod(lambda cls: object()),
    )
    monkeypatch.setattr(runpod_handler, "_ensure_kernel_warmup", lambda _profile: None)
    monkeypatch.setattr(runpod_handler, "_prime_service_runtime", lambda _service: None)
    monkeypatch.setattr(
        runpod_handler,
        "_prepare_compliance_model_for_vllm",
        lambda config: config,
    )

    runpod_handler.shutdown()
    runpod_handler.initialize()

    try:
        assert os.environ["VOYAGER_GROUNDEDNESS_VLLM_MAX_CONCURRENCY"] == "32"
        assert os.environ["LATENCE_TRACE_NLI_VLLM_MAX_CONCURRENCY"] == "32"
        assert os.environ["LATENCE_TRACE_COMPLIANCE_GLINER_ENDPOINT"] == "http://127.0.0.1:18003"
        assert os.environ["LATENCE_TRACE_COMPLIANCE_MAX_CONCURRENCY"] == "32"
    finally:
        runpod_handler.shutdown()


def test_build_startup_warmup_requests_exercises_three_triangular_probes() -> None:
    requests = runpod_handler._build_startup_warmup_requests()
    # 3 RAG triangular warmups + 1 code-lane warmup = 4.
    assert len(requests) == 4
    rag_requests = [r for r in requests if r.scoring_mode.value == "rag"]
    code_requests = [r for r in requests if r.scoring_mode.value == "code"]
    assert len(rag_requests) == 3
    assert len(code_requests) == 1
    assert all(request.include_triangular_diagnostics for request in rag_requests)
    assert all(request.query_text for request in requests)
    assert all(request.raw_context for request in requests)
    assert all(request.response_text for request in requests)
    code_req = code_requests[0]
    assert code_req.response_language_hint == "python"
    assert code_req.session_id == "warmup-code-lane"


def test_runpod_request_builder_preserves_structured_verification() -> None:
    request, verbose = runpod_handler._build_request(
        {
            "query_text": "What amount was approved?",
            "raw_context": "customer_1001 amount_usd=1200 status=approved",
            "response_text": "customer_1001 was approved for 1200 USD.",
            "content_type": "application/json+schema",
            "structured_verification": "on",
            "verbose": True,
        }
    )

    assert verbose is True
    assert request.content_type == "application/json+schema"
    assert request.structured_verification == "on"


def test_prime_service_runtime_issues_all_startup_requests() -> None:
    service = _SlowService(sleep_ms=0)
    runpod_handler._prime_service_runtime(service)
    assert service.calls == 4


def test_runpod_handler_bounds_inflight_requests(monkeypatch) -> None:
    service = _SlowService(sleep_ms=80)
    config = _config(max_concurrency=2)

    monkeypatch.setattr(runpod_handler, "initialize", lambda: None)
    runpod_handler._initialized = True
    runpod_handler._config = config
    runpod_handler._service = service
    runpod_handler._servers = {}
    runpod_handler._request_executor = None
    runpod_handler._lane_semaphores = {}
    runpod_handler._lane_semaphores_loop = None

    results = asyncio.run(_burst(6))

    try:
        assert all(item["success"] for item in results)
        assert service.peak_inflight <= config.max_concurrency
    finally:
        runpod_handler.shutdown()


def test_runpod_handler_dispatches_compliance_redaction(monkeypatch) -> None:
    compliance = _ComplianceService(sleep_ms=1)
    config = _config(max_concurrency=2)

    monkeypatch.setattr(runpod_handler, "initialize", lambda: None)
    runpod_handler._initialized = True
    runpod_handler._config = config
    runpod_handler._service = _SlowService(sleep_ms=1)
    runpod_handler._compliance_service = compliance
    runpod_handler._servers = {}
    runpod_handler._request_executor = None
    runpod_handler._compliance_semaphore = None
    runpod_handler._compliance_semaphore_loop = None

    try:
        result = asyncio.run(
            runpod_handler.handler(
                {
                    "input": {
                        "action": "redact",
                        "text": "Jane Doe",
                        "labels": ["person"],
                    }
                }
            )
        )
    finally:
        runpod_handler.shutdown()

    assert result["success"] is True
    assert result["action"] == "redact"
    assert result["result"]["entity_count"] == 1
    assert compliance.calls == 1


def test_runpod_handler_dispatches_compression_action(monkeypatch) -> None:
    class _CompressionService:
        def __init__(self) -> None:
            self.calls = 0

        async def compress(self, request):
            self.calls += 1
            return runpod_handler.CompressionResponse(
                compressed_text=request.text,
                original_tokens=3,
                compressed_tokens=3,
                compression_ratio=1.0,
                provider="fallback",
            )

    compression = _CompressionService()
    config = _config(max_concurrency=2)

    monkeypatch.setattr(runpod_handler, "initialize", lambda: None)
    runpod_handler._initialized = True
    runpod_handler._config = config
    runpod_handler._compression_service = compression

    try:
        result = asyncio.run(
            runpod_handler.handler(
                {
                    "input": {
                        "action": "compression",
                        "text": "alpha beta gamma",
                    }
                }
            )
        )
    finally:
        runpod_handler.shutdown()

    assert result["success"] is True
    assert result["action"] == "compress"
    assert result["result"]["compressed_text"] == "alpha beta gamma"
    assert compression.calls == 1


def test_runpod_handler_preserves_session_id_on_create(monkeypatch) -> None:
    config = _config(max_concurrency=2)
    session_service = runpod_handler.TraceSessionService(groundedness_service=_SlowService(sleep_ms=0))

    monkeypatch.setattr(runpod_handler, "initialize", lambda: None)
    runpod_handler._initialized = True
    runpod_handler._config = config
    runpod_handler._session_service = session_service

    try:
        created = asyncio.run(
            runpod_handler.handler(
                {
                    "input": {
                        "action": "session.create",
                        "session_id": "explicit-runpod-session",
                        "kind": "general",
                    }
                }
            )
        )
        event = asyncio.run(
            runpod_handler.handler(
                {
                    "input": {
                        "action": "session.event",
                        "session_id": "explicit-runpod-session",
                        "event": {
                            "event_type": "observation",
                            "content": "Order INV-42 amount 1200 USD remains approved.",
                        },
                        "memory_domain": "rag",
                    }
                }
            )
        )
        fetched = asyncio.run(
            runpod_handler.handler(
                {
                    "input": {
                        "action": "session.get",
                        "session_id": "explicit-runpod-session",
                    }
                }
            )
        )
    finally:
        runpod_handler.shutdown()

    assert created["success"] is True
    assert created["result"]["session"]["session_id"] == "explicit-runpod-session"
    assert event["success"] is True
    assert event["result"]["session"]["session_id"] == "explicit-runpod-session"
    assert fetched["success"] is True
    assert fetched["result"]["session"]["session_id"] == "explicit-runpod-session"


def test_runpod_handler_dispatches_rollup(monkeypatch) -> None:
    config = _config(max_concurrency=2)

    monkeypatch.setattr(runpod_handler, "initialize", lambda: None)
    runpod_handler._initialized = True
    runpod_handler._config = config
    runpod_handler._service = _SlowService(sleep_ms=0)

    try:
        result = asyncio.run(
            runpod_handler.handler(
                {
                    "input": {
                        "action": "rollup",
                        "session_id": "sess-rollup",
                        "turns": [
                            {
                                "risk_band": "green",
                                "scores": {"groundedness_v2": 0.9},
                            },
                            {
                                "risk_band": "amber",
                                "scores": {"groundedness_v2": 0.55},
                            },
                        ],
                    }
                }
            )
        )
    finally:
        runpod_handler.shutdown()

    assert result["success"] is True
    assert result["action"] == "rollup"
    assert result["rollup"]["turns"] == 2
    assert result["rollup"]["session_id"] == "sess-rollup"
    assert result["rollup"]["risk_band_trail"] == ["green", "amber"]


def test_runpod_handler_dispatches_memory_update(monkeypatch) -> None:
    config = _config(max_concurrency=2)

    monkeypatch.setattr(runpod_handler, "initialize", lambda: None)
    runpod_handler._initialized = True
    runpod_handler._config = config
    runpod_handler._compression_service = None

    try:
        result = asyncio.run(
            runpod_handler.handler(
                {
                    "input": {
                        "action": "memory.update",
                        "turn_text": "Keep invoice INV-42 and refund approval constraints.",
                        "memory_domain": "rag",
                    }
                }
            )
        )
    finally:
        runpod_handler.shutdown()

    assert result["success"] is True
    assert result["action"] == "memory.update"
    assert result["result"]["next_memory_state"]["turn_index"] == 1
    assert "hot_context" in result["result"]


def test_runpod_compliance_redaction_uses_own_concurrency_budget(monkeypatch) -> None:
    compliance = _ComplianceService(sleep_ms=80)
    config = _config(max_concurrency=4)

    monkeypatch.setattr(runpod_handler, "initialize", lambda: None)
    runpod_handler._initialized = True
    runpod_handler._config = config
    runpod_handler._service = _SlowService(sleep_ms=1)
    runpod_handler._compliance_service = compliance
    runpod_handler._servers = {}
    runpod_handler._request_executor = None
    runpod_handler._compliance_semaphore = None
    runpod_handler._compliance_semaphore_loop = None

    async def _burst() -> list[dict]:
        payload = {"input": {"action": "redact", "text": "Jane Doe", "labels": ["person"]}}
        return await asyncio.gather(*(runpod_handler.handler(payload) for _ in range(8)))

    started = time.perf_counter()
    try:
        results = asyncio.run(_burst())
    finally:
        runpod_handler.shutdown()
    elapsed = time.perf_counter() - started

    assert all(item["success"] for item in results), results
    assert compliance.peak_inflight > 1
    assert compliance.peak_inflight <= config.max_concurrency
    assert elapsed < 0.5


def test_compact_response_surfaces_unused_context_contract() -> None:
    """Regression guard: the RunPod serverless envelope must expose the
    precision-first tri-state unused-context signals without requiring
    callers to opt into ``verbose=true``. If this slips, agents hitting
    the serverless endpoint silently lose access to a SOTA feature.
    """

    runpod_handler._config = runpod_handler.WorkerConfig(
        profile="quality",
        version="test",
        request_timeout_s=5,
        code_request_timeout_s=2.0,
        rollup_request_timeout_s=0.25,
        max_concurrency=4,
        collection_label="latence-trace",
        service_device="cpu",
        docs_url="",
        managed_vllm_enabled=True,
        colbert_model="lightonai/LateOn",
        colbert_port=18001,
        colbert_gpu_mem=0.2,
        colbert_max_model_len=8192,
        colbert_max_num_seqs=128,
        colbert_max_batched_tokens=8192,
        nli_model="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
        nli_port=18002,
        nli_gpu_mem=0.2,
        nli_max_model_len=512,
        nli_max_num_seqs=128,
        nli_max_batched_tokens=8192,
        compliance_model="knowledgator/gliner-pii-large-v1.0",
        compliance_port=18003,
        compliance_gpu_mem=0.2,
        compliance_max_model_len=768,
        compliance_max_num_seqs=128,
        compliance_max_batched_tokens=8192,
        compliance_threshold=0.5,
        compliance_dataset_path="doubledsbv/pii-replacement-dataset",
        compliance_request_timeout_s=30.0,
    )

    response = GroundednessResponse(
        collection="latence-trace",
        mode="raw_context",
        model="test-model",
        scores=GroundednessScores(
            primary_name="reverse_context",
            primary_score=0.82,
            reverse_context=0.82,
            reverse_context_calibrated=0.84,
            literal_guarded=0.81,
            nli_aggregate=0.79,
            groundedness_v2=0.83,
            semantic_entropy_aggregate=0.77,
            semantic_entropy_raw=0.23,
            semantic_entropy_sample_count=4,
            structured_source=0.86,
            structured_source_guarded=0.85,
            structured_source_detected=True,
            consensus_hardened=0.8,
            risk_band="amber",
            context_coverage_ratio=0.5,
            context_coverage_threshold=0.5,
            context_usage_ratio=0.5,
            context_unused_ratio=0.25,
            context_uncertain_ratio=0.25,
            support_units_usage_used=2,
            support_units_unused=1,
            support_units_uncertain=1,
        ),
        response_tokens=[],
        support_units=[
            GroundednessSupportUnit(
                index=0,
                support_id="unit-0",
                source_mode="raw_context",
                text="Berlin is the capital of Germany.",
                token_count=6,
                tokens=["Berlin", "is", "the", "capital", "of", "Germany"],
                token_scores=[0.9, 0.0, 0.0, 0.9, 0.0, 0.9],
                score=0.9,
                matched_response_tokens=3,
                coverage_score=0.9,
                used=True,
                usage_state=GroundednessUsageState.USED,
                usage_confidence=0.91,
                unused_confidence=0.03,
            ),
            GroundednessSupportUnit(
                index=1,
                support_id="unit-1",
                source_mode="raw_context",
                text="Penguins are flightless aquatic birds.",
                token_count=5,
                tokens=["Penguins", "are", "flightless", "aquatic", "birds"],
                token_scores=[0.0, 0.0, 0.0, 0.0, 0.0],
                score=0.05,
                matched_response_tokens=0,
                coverage_score=0.05,
                used=False,
                usage_state=GroundednessUsageState.UNUSED,
                usage_confidence=0.88,
                unused_confidence=0.88,
            ),
        ],
        top_evidence=[],
        eligibility=GroundednessEligibility(
            collection_kind=CollectionKind.LATE_INTERACTION,
            vector_source="encoded_raw_context",
            storage_compression=None,
            quantization_mode=None,
            dequantized=True,
            user_facing_supported=True,
            warnings=[],
        ),
        time_ms=42.0,
        attribution_mode=AttributionMode.CLOSED_BOOK,
        corpus_route=CorpusRouteDiagnostics(
            corpus_type="rag.structured",
            source="explicit",
            fusion_weights_applied={
                "calibrated": 0.2,
                "literal": 0.2,
                "nli": 0.2,
                "semantic_entropy": 0.2,
                "structured": 0.2,
            },
            thresholds_applied={"green": 0.83, "amber": 0.61},
            scoring_mode_applied="rag",
        ),
        runtime_decision=RuntimeDecisionRecord(
            policy_version="runtime_decision",
            head_id="cell_schema_verifier",
            head_version="root_cause_solution_v1",
            head_enabled=True,
            head_score=0.91,
            class_key="rag.structured",
            score=0.91,
            score_channel="runtime_head",
            band="green",
            action="allow",
            allow_disabled=False,
            block_disabled=False,
            allow_threshold=0.82,
            block_threshold=0.41,
        ),
        runtime_head_features={"cell_provenance_match": 1.0},
    )

    compact = runpod_handler._compact_response(response, verbose=False)

    assert compact["success"] is True
    assert compact["primary_metric"] == "groundedness_v2"
    assert compact["score"] == 0.83
    assert compact["groundedness_v2"] == 0.83
    assert compact["reverse_context_calibrated"] == 0.84
    assert compact["literal_guarded"] == 0.81
    assert compact["nli_aggregate"] == 0.79
    assert compact["semantic_entropy_aggregate"] == 0.77
    assert compact["semantic_entropy_raw"] == 0.23
    assert compact["semantic_entropy_sample_count"] == 4
    assert compact["structured_score"] == 0.86
    assert compact["structured_source_guarded"] == 0.85
    assert compact["structured_source_detected"] is True
    assert compact["score_channels"] == {
        "primary": 0.82,
        "reverse_context": 0.82,
        "reverse_context_calibrated": 0.84,
        "literal_guarded": 0.81,
        "nli_aggregate": 0.79,
        "semantic_entropy_aggregate": 0.77,
        "structured_source": 0.86,
        "structured_source_guarded": 0.85,
        "groundedness_v2": 0.83,
        "consensus_hardened": 0.8,
    }
    assert compact["context_coverage_ratio"] == 0.5
    assert compact["context_coverage_threshold"] == 0.5
    assert compact["context_unused_ratio"] == 0.25
    assert compact["context_uncertain_ratio"] == 0.25
    assert compact["context_usage_ratio"] == 0.5
    assert compact["corpus_route"]["corpus_type"] == "rag.structured"
    assert compact["corpus_route"]["thresholds_applied"] == {"green": 0.83, "amber": 0.61}
    assert compact["corpus_route"]["fusion_weights_applied"]["nli"] == 0.2
    assert compact["runtime_decision"]["action"] == "allow"
    assert compact["runtime_decision"]["head_enabled"] is True
    assert compact["runtime_decision"]["allow_threshold"] == 0.82
    assert compact["runtime_decision"]["block_threshold"] == 0.41
    assert compact["runtime_head_features"] == {"cell_provenance_match": 1.0}
    assert compact["support_units_usage"] == {
        "used": 2,
        "unused": 1,
        "uncertain": 1,
    }

    assert "full" not in compact, "verbose=False must stay compact"
    assert len(compact["support_units"]) == 2
    unit0, unit1 = compact["support_units"]
    assert unit0["support_id"] == "unit-0"
    assert unit0["usage_state"] == "used"
    assert unit0["usage_confidence"] == 0.91
    assert unit0["unused_confidence"] == 0.03
    assert unit0["coverage_score"] == 0.9
    assert unit0["used"] is True
    assert unit1["support_id"] == "unit-1"
    assert unit1["usage_state"] == "unused"
    assert unit1["unused_confidence"] == 0.88

    verbose = runpod_handler._compact_response(response, verbose=True)
    assert "full" in verbose
    assert verbose["full"]["scores"]["support_units_unused"] == 1
    assert verbose["full"]["support_units"][0]["usage_state"] == "used"


def test_score_response_can_return_canonical_model_dump() -> None:
    response = _SlowService(sleep_ms=0).groundedness(
        type(
            "Req",
            (),
            {
                "scoring_mode": runpod_handler.ScoringMode.RAG,
                "profile": None,
                "session_id": "sess-1",
            },
        )()
    )

    canonical = runpod_handler._score_response(
        response,
        verbose=False,
        response_format="canonical",
    )

    assert canonical["success"] is True
    assert canonical["action"] == "score"
    assert canonical["result"]["session_id"] == "sess-1"
    assert canonical["result"]["scores"]["primary_name"] == "reverse_context"


def test_runpod_handler_passes_profile_and_compacts_effective_profile(monkeypatch) -> None:
    service = _SlowService(sleep_ms=1)
    config = _config(max_concurrency=2)

    monkeypatch.setattr(runpod_handler, "initialize", lambda: None)
    runpod_handler._initialized = True
    runpod_handler._config = config
    runpod_handler._service = service
    runpod_handler._servers = {}
    runpod_handler._request_executor = None
    runpod_handler._lane_semaphores = {}
    runpod_handler._lane_semaphores_loop = None

    try:
        result = asyncio.run(
            runpod_handler.handler(
                {
                    "input": {
                        "query_text": "Where was Heinrich born?",
                        "raw_context": "Heinrich was born in Augsburg.",
                        "response_text": "Heinrich was born in Augsburg.",
                        "profile": "quality",
                    }
                }
            )
        )
    finally:
        runpod_handler.shutdown()

    assert result["success"] is True
    assert result["profile"] == "quality"
    assert result["effective_profile"] == "quality"
    assert result["profile_diagnostics"]["effective_profile"] == "quality"


async def _mixed_lane_burst(total: int) -> list[dict]:
    """Fire ``total`` requests alternating between the RAG and code lanes."""
    base_payload = {
        "query_text": "Where was Heinrich born?",
        "raw_context": "# Heinrich.py\nHeinrich was born in 1851 in Augsburg.",
        "response_text": "Heinrich was born in Augsburg in 1851.",
    }
    payloads: list[dict] = []
    for idx in range(total):
        lane_payload = dict(base_payload)
        if idx % 2 == 0:
            lane_payload["scoring_mode"] = "code"
            lane_payload["session_id"] = f"sess-{idx}"
            lane_payload["response_language_hint"] = "python"
        else:
            lane_payload["scoring_mode"] = "rag"
            lane_payload["session_id"] = f"sess-{idx}"
        payloads.append({"input": lane_payload})
    return await asyncio.gather(*(runpod_handler.handler(payload) for payload in payloads))


def test_runpod_handler_stress_32_mixed_lanes(monkeypatch) -> None:
    """Per-lane semaphores must let 32 simultaneous requests (16 RAG + 16
    code) complete under the timeout without starving either lane.
    """
    service = _SlowService(sleep_ms=25)
    config = _config(max_concurrency=16)

    monkeypatch.setattr(runpod_handler, "initialize", lambda: None)
    runpod_handler._initialized = True
    runpod_handler._config = config
    runpod_handler._service = service
    runpod_handler._servers = {}
    runpod_handler._request_executor = None
    runpod_handler._lane_semaphores = {}
    runpod_handler._lane_semaphores_loop = None

    try:
        results = asyncio.run(_mixed_lane_burst(32))
        assert all(item.get("success") for item in results), results
        # Every request emitted a ``scoring_mode`` in the compact envelope.
        lanes = {item.get("scoring_mode") for item in results}
        assert lanes.issubset({"rag", "code"})
        # At least one call of each lane must have landed.
        assert "rag" in lanes and "code" in lanes
        # Peak inflight never exceeds the total worker ceiling.
        assert service.peak_inflight <= config.max_concurrency
        # Each lane's budget is ceil(16/2) = 8, so the absolute ceiling
        # a single lane can reach is 8 — guards against the regression
        # where a shared semaphore lets one lane starve the other.
        assert (
            service.peak_inflight
            <= max(
                runpod_handler._lane_budget(config, runpod_handler.ScoringMode.RAG),
                runpod_handler._lane_budget(config, runpod_handler.ScoringMode.CODE),
            )
            * 2
        )
    finally:
        runpod_handler.shutdown()
