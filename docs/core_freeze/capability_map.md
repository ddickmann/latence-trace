# TRACE Core Freeze Capability Map

This document is the discovery baseline for freezing `latence-trace` as a serious product API. It records existing capabilities that must be preserved before changing endpoint names, SDK ergonomics, RunPod behavior, or deployment topology.

`api_surface_manifest.json` is the executable inventory for this map. If a capability is missing there, it is not considered packaged for the product API.

## Product Paths

| Product path | Current API surface | State model | Canonical implementation | Evidence |
|---|---|---|---|---|
| `privacy.redact` | `POST /v1/compliance/redact`, RunPod `action=redact` | Stateless compute | `latence_trace/api/compliance_*`, `latence_trace/compliance/*` | `tests/test_compliance_redaction.py`, `scripts/canary_compliance_runtime.py` |
| `grounding.rag` | `POST /groundedness` with default or `scoring_mode=rag`, RunPod `action=score` | Stateless compute | `latence_trace/api/service.py`, `latence_trace/core/groundedness.py`, `latence_trace/core/nli.py` | `tests/test_groundedness_service.py`, `scripts/bench_runpod_360.py`, `research/triangular_maxsim/*` |
| `grounding.code` | `POST /groundedness` with `scoring_mode=code`, RunPod `action=score` | Caller-carried optional code session state | `latence_trace/core/code_lane/*`, `latence_trace/api/service.py` | `tests/test_ast_grounding.py`, `tests/test_code_lane_temporal_drift.py`, `scripts/bench_coding.py`, `scripts/bench_transcripts_v2.py` |
| `compression.text` | `POST /v1/compression`, RunPod `action=compress` | Stateless compute | `latence_trace/api/compression_*`, `latence_trace/providers/compression.py` | `tests/test_compression_service.py`, `tests/test_compression_superpod_contract.py` |
| `compression.messages` | `POST /v1/compression` with message action/payload | Stateless compute | `latence_trace/api/compression_models.py`, `latence_trace/api/compression_service.py` | `tests/test_compression_service.py` |
| `memory.step` | `POST /v1/memory/update`, RunPod `action=memory.update` | Caller-carried `prior_memory_state -> next_memory_state` | `latence_trace/memory/*` | `tests/test_memory_core.py`, `scripts/run_infinimem_real_bootstrap.py` |
| `rollup` | `POST /groundedness/rollup`, RunPod `action=rollup` | Stateless aggregation | `latence_trace/api/routes.py`, `latence_trace/api/models.py` | `tests/test_runpod_handler.py`, `scripts/bench_runpod_360.py` |
| `session.*` | `/v1/trace/sessions/*`, RunPod `action=session.*` | Server-stateful by default | `latence_trace/sessions/*` | `tests/test_trace_sessions.py`, `scripts/run_stateful_trace_session_proof.py` |

## Non-Canonical Surfaces

| Surface | Status | Reason |
|---|---|---|
| `/compliance/*` | Legacy public alias | Backwards-compatible mount. Canonical path is `/v1/compliance/*`. |
| `/v1/amber/*` | Internal optional review queue | Not part of the stateless compute product API. |
| `/mcp/*` | Optional integration surface | Mounted only when `LATENCE_TRACE_ENABLE_MCP_HTTP` is enabled. |
| `/metrics` | Ops surface | Prometheus scrape endpoint, intentionally excluded from OpenAPI. |

## Grounding Is A System, Not A Score

### RAG Lane

The RAG lane combines:

- Triangular and reverse MaxSim kernels in `latence_trace/kernels/triton_triangular_maxsim.py`.
- Evidence lanes: `chunk_ids`, `raw_context`, and structured `support_units`.
- NLI claim verification and score fusion in `latence_trace/core/nli.py`.
- Structured source verification, context coverage, support-unit attribution, and unused-context tri-state classification in `latence_trace/core/groundedness.py`.
- Runtime profile and policy orchestration in `latence_trace/api/service.py`.
- Heatmap-ready token and source diagnostics.

The product API must preserve route/bundle/profile metadata, thresholds, risk band, support-unit attribution, unused-context signals, NLI diagnostics, structured-evidence signals, heatmap data, and runtime decisions.

### Code Lane

The code lane combines:

- Shared encoder/MaxSim scoring.
- Tree-sitter AST grounding with regex fallback only as a development fallback.
- Literal novelty and missing-literal checks.
- Composite/logistic phantom scoring.
- Ambiguity-gated NLI cascade.
- File attribution and dead-weight context diagnostics.
- Caller-portable code session state in `latence_trace/core/code_lane/session.py`.

The product API must preserve AST diagnostics, literal novelty, phantom probability/verdict, composite score, NLI cascade diagnostics, file attribution, dead-weight ratio, code session-state round trip, and runtime decisions.

## Routing, Calibration, And Heads

| Capability | Files | Product implication |
|---|---|---|
| Corpus router rules | `latence_trace/core/corpus_router/rules.py` | Deterministic high-precision routing before classifier fallback. |
| Pretrained router classifier | `latence_trace/core/corpus_router/classifier.py`, `latence_trace/data/corpus_classifier.joblib` | Product responses need route/class diagnostics so operators know which bundle fired. |
| Router feature schema | `latence_trace/core/corpus_router/features.py` | Feature names are part of artifact compatibility. |
| Calibration bundles | `latence_trace/core/corpus_router/bundles.py`, `latence_trace/data/calibration.*.json` | Fusion weights and thresholds are per corpus class and must not be flattened away. |
| Tenant thresholds | `latence_trace/core/thresholds.py` | Product deployments can override green/amber bands by tenant/profile. |
| Runtime head registry | `latence_trace/core/runtime_heads.py`, `latence_trace/data/runtime_head_registry.*` | Per-head trained models and rule heads must be tracked as contract metadata. |
| Runtime decision policy | `latence_trace/core/runtime_decision.py`, `latence_trace/data/runtime_policy.*` | The product decision is allow/repair/block, not just a score. |
| Runtime feature synthesis | `latence_trace/core/runtime_feature_synthesizer.py` | SDKs may omit head features; server can synthesize from payload/route context. |

## Memory And Repair

InfiniMem is already a caller-carried state system through `MemoryState`. The stateless product path is:

1. SDK/plugin sends turn data plus prior memory state.
2. TRACE computes span extraction, survival, deduplication, adaptive budgets, and diagnostics.
3. TRACE returns next memory state and hot context.
4. SDK/plugin stores that state and sends it again later.

Server-held sessions remain valuable for demos, local proofs, and future tenant backends, but `InMemorySessionStore` must not be required for production RunPod continuity.

## Compliance

The compliance runtime already includes:

- Canonical GDPR labels.
- Model alias mapping, such as `person -> name`.
- Address component calibration, such as `postal_code -> zip code`.
- Canonical labels returned after model aliases.
- Redaction output and privacy-safe response options.
- Sanity checks.
- Custom regex override.
- Chunk concurrency.
- Route offloading so health checks are not blocked.
- Optional native GLiNER/vLLM parity.

## Freeze Classification

| Area | Freeze stance |
|---|---|
| Stateless privacy, grounding, compression, memory step, rollup | Product-ready after parity and benchmark reproduction. |
| Code grounding | Product-ready only when tree-sitter is active and code-lane benchmark gates pass. |
| Server-held sessions | Demo/internal unless durable store or SDK-managed state is selected. |
| External guard models | Shadow/amber only until tenant-specific calibration promotes them. |
| Tenant backend and portal | Post-core-freeze work. |
