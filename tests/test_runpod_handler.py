from __future__ import annotations

import asyncio
import importlib.util
import os
import sys
import threading
import time
from pathlib import Path

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
    name
    for name in list(sys.modules.keys())
    if name == "server" or name.startswith("server.")
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
    if (_name == "server" or _name.startswith("server.")) and _name not in _RUNPOD_SERVER_BACKUP_KEYS:
        del sys.modules[_name]
if _REAL_SERVER_MODULE is not None:
    sys.modules["server"] = _REAL_SERVER_MODULE
if _RUNPOD_PATH_INSERTED:
    try:
        sys.path.remove(str(_RUNPOD_DIR))
    except ValueError:
        pass


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


def _config(*, profile: str = "quality", max_concurrency: int = 2):
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
        colbert_model="lightonai/LateOn",
        colbert_port=18001,
        colbert_gpu_mem=0.34,
        colbert_max_model_len=8192,
        colbert_max_num_seqs=128,
        colbert_max_batched_tokens=8192,
        nli_model="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
        nli_port=18002,
        nli_gpu_mem=0.24,
        nli_max_model_len=512,
        nli_max_num_seqs=128,
        nli_max_batched_tokens=8192,
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
    monkeypatch.setenv("MAX_CONCURRENCY", "7")
    monkeypatch.setenv("LATENCE_TRACE_PROFILE", "quality")
    assert runpod_handler.create_config().max_concurrency == 64


def test_create_config_pins_vllm_runtime_defaults(monkeypatch) -> None:
    monkeypatch.setenv("LATENCE_TRACE_PROFILE", "quality")
    monkeypatch.delenv("LATENCE_TRACE_COLBERT_MAX_NUM_SEQS", raising=False)
    monkeypatch.delenv("LATENCE_TRACE_COLBERT_MAX_BATCHED_TOKENS", raising=False)
    monkeypatch.delenv("LATENCE_TRACE_NLI_MAX_NUM_SEQS", raising=False)
    monkeypatch.delenv("LATENCE_TRACE_NLI_MAX_BATCHED_TOKENS", raising=False)

    config = runpod_handler.create_config()

    assert config.colbert_max_num_seqs == 128
    assert config.colbert_max_batched_tokens == 8192
    assert config.nli_max_num_seqs == 128
    assert config.nli_max_batched_tokens == 8192


def test_build_servers_pin_requested_vllm_settings() -> None:
    servers = runpod_handler._build_servers(_config(max_concurrency=64))

    assert servers["colbert"].max_num_seqs == 128
    assert servers["colbert"].max_num_batched_tokens == 8192
    assert servers["colbert"].enforce_eager is False
    assert servers["nli"].max_num_seqs == 128
    assert servers["nli"].max_num_batched_tokens == 8192
    assert servers["nli"].enforce_eager is False


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
        },
    )
    monkeypatch.setattr(runpod_handler, "GroundednessService", _FakeGroundednessService)
    monkeypatch.setattr(runpod_handler, "_ensure_kernel_warmup", lambda _profile: None)
    monkeypatch.setattr(runpod_handler, "_prime_service_runtime", lambda _service: None)

    runpod_handler.shutdown()
    runpod_handler.initialize()

    try:
        assert os.environ["VOYAGER_GROUNDEDNESS_VLLM_MAX_CONCURRENCY"] == "64"
        assert os.environ["LATENCE_TRACE_NLI_VLLM_MAX_CONCURRENCY"] == "64"
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
        colbert_model="lightonai/LateOn",
        colbert_port=18001,
        colbert_gpu_mem=0.34,
        colbert_max_model_len=8192,
        colbert_max_num_seqs=128,
        colbert_max_batched_tokens=8192,
        nli_model="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
        nli_port=18002,
        nli_gpu_mem=0.24,
        nli_max_model_len=512,
        nli_max_num_seqs=128,
        nli_max_batched_tokens=8192,
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
    return await asyncio.gather(
        *(runpod_handler.handler(payload) for payload in payloads)
    )


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
        assert service.peak_inflight <= max(
            runpod_handler._lane_budget(config, runpod_handler.ScoringMode.RAG),
            runpod_handler._lane_budget(config, runpod_handler.ScoringMode.CODE),
        ) * 2
    finally:
        runpod_handler.shutdown()
