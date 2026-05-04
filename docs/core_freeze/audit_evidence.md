# TRACE Core Freeze Audit Evidence

Date: 2026-05-04

## Scope Audited

This audit covered the freeze artifacts and implementation touched during the TRACE core hardening pass:

- product-path docs in `docs/core_freeze/`
- benchmark manifest in `scripts/trace_core_freeze_gate.py`
- Python SDK namespaces and SDK-managed `TraceSession`
- FastAPI discovery via `/agent-help`
- RunPod action parity for score, redact, compress, memory update, rollup, and sessions
- session route error envelopes
- unit and regression tests for SDK and RunPod behavior

## Fixes Applied During Audit

- Added `/agent-help` discovery metadata for `trace_session_source_post`.
- Documented that `client.session(...)` is the stateless SDK facade, while server-held REST sessions remain lower-level HTTP/RunPod capabilities.
- Added RunPod request examples for canonical score and memory update envelopes.
- Added `api_surface_manifest.json` as the machine-readable product API source of truth.
- Added `scripts/trace_core_contract_check.py` and `tests/test_trace_core_contract_check.py` so FastAPI, `/agent-help`, RunPod actions, SDK namespaces, examples, and gates must match before benchmark evidence is accepted.
- Added missing golden examples for compression messages, RunPod redact/compress/rollup, server sessions, and SDK-managed session facade.
- Added concrete integration event mapping from product intent to HTTP path, RunPod action, and SDK path.
- Expanded the gate manifest for memory hardening, compression, groundedness service, sessions/repair, runtime decisions, runtime-head readiness, unused-context precision, and response-chunking.
- Split compliance canaries into RunPod and gateway live gates with explicit required environment variables.
- Added timeout recording to the gate runner so hung benchmarks produce evidence instead of blocking the audit.
- Added regression coverage for RunPod `rollup`, RunPod `memory.update`, SDK `compression.messages`, SDK-managed session RAG/code calls, and `agent_help()`.
- Fixed a corpus-router `featurize` latency budget failure on 100 KB extreme payloads by sampling large regex inputs while preserving the full context length feature.
- Fixed full-SDK typing/lint issues in changed SDK modules and optional integration adapters.

## Local Gate Evidence

`reports/core_freeze/full_local_gate_report.json` passed after the audit fixes:

- `pii_local_release`: 37 passed, 2 skipped native GLiNER parity tests
- `memory_local_core`: 19 passed
- `memory_hardening_local`: 16 passed
- `compression_local_core`: 6 passed
- `grounding_local_core`: 65 passed
- `grounding_service_local`: 120 passed, 10 skipped Voyager-index integration tests
- `code_local_core`: 26 passed
- `sessions_repair_local`: 23 passed
- `sdk_local_core`: 16 passed
- `observability_local_core`: 5 passed
- `trace_v2_readiness_local`: 38 passed plus artifact/policy readiness checks

Total local freeze evidence: 371 passing checks, 12 explicitly skipped integration/parity checks.

## Benchmark Script Evidence

`reports/core_freeze/local_benchmark_script_report.json` records two manual benchmark scripts as timed out at 120 seconds each:

- `scripts/bench_unused_context_precision.py`
- `scripts/bench_response_chunking.py`

These scripts are now represented in the freeze manifest, but their current runtime behavior prevents them from being used as automated proof in this environment. They are marked `manual` rather than `local` so the core local release gate remains deterministic. They need a dedicated benchmark-harness fix or smaller CI mode before the freeze can claim reproducible numeric benchmark values from them.

## Live And Gated Evidence

`reports/core_freeze/live_gated_gate_report.json` records live/gated checks as skipped because required services or credentials are not present:

- native GLiNER/vLLM parity: missing `LATENCE_TRACE_COMPLIANCE_GLINER_ENDPOINT`
- RunPod 360: missing `RUNPOD_ENDPOINT_ID`
- SDK/gateway 360: missing `LATENCE_API_KEY`
- RunPod compliance canary: missing `RUNPOD_ENDPOINT_ID`
- gateway compliance canary: missing `LATENCE_TRACE_API_KEY` and `LATENCE_TRACE_COMPLIANCE_GATEWAY_URL`

## Code Quality Evidence

Commands completed successfully after fixes:

- `python -m ruff check clients/python/latence_trace_client clients/python/tests/test_client.py latence_trace/api/routes.py latence_trace/sessions/routes.py runpod/handler.py scripts/trace_core_freeze_gate.py tests/test_runpod_handler.py`
- `PYTHONPATH=/workspace/latence-trace/clients/python:/workspace/latence-trace python -m mypy clients/python/latence_trace_client`
- `PYTHONPATH=/workspace/latence-trace/clients/python:/workspace/latence-trace python -m pytest clients/python/tests/test_client.py tests/test_runpod_handler.py`
- `PYTHONPATH=/workspace/latence-trace/clients/python:/workspace/latence-trace python scripts/trace_core_contract_check.py`
- `PYTHONPATH=/workspace/latence-trace/clients/python:/workspace/latence-trace python -m compileall -q clients/python/latence_trace_client scripts/trace_core_freeze_gate.py`

## Conclusion

The local product-path surface is now materially better covered and packaged: privacy, grounding RAG, grounding code, compression, caller-carried memory, sessions/repair, rollup, SDK namespaces, RunPod parity, discovery metadata, runtime policy, and observability all have reproducible local evidence.

The freeze is not yet allowed to claim full reproduction of prior numeric benchmark values. Two local benchmark scripts time out, and live/GPU/service-backed gates need credentials or running endpoints. Those are the remaining blockers before declaring the core API and benchmark story fully frozen.
