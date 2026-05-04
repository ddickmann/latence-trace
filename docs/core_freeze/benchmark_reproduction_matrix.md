# TRACE Benchmark Reproduction Matrix

The core freeze must reproduce existing proof work before adding new benchmarks. This matrix records the current gates and whether they are local, gated, or live.

## PII And Compliance

| Path | What it proves | Kind |
|---|---|---|
| `tests/test_compliance_redaction.py` | GDPR label resolution, model aliases, address calibration, canonical labels, redaction output, sanity checks, custom regex, chunk concurrency, route offloading, mounted schema/redact routes, optional native GLiNER parity. | Local plus gated parity |
| `tests/test_runpod_handler.py` | RunPod compliance dispatch and branch isolation. | Local |
| `scripts/canary_compliance_runtime.py` | Live RunPod/gateway fixture, no raw text returned, `[EMAIL]` masking, label detection, entity count, latency budget. | Live |

Main local command:

```bash
PYTHONPATH=/workspace/latence-trace/clients/python:/workspace/latence-trace \
python -m pytest tests/test_compliance_redaction.py tests/test_runpod_handler.py
```

Native GLiNER/vLLM parity:

```bash
LATENCE_TRACE_COMPLIANCE_PARITY=1 \
LATENCE_TRACE_COMPLIANCE_GLINER_ENDPOINT=http://127.0.0.1:8003 \
python -m pytest tests/test_compliance_redaction.py -k 'native_gliner_parity or parallel_requests'
```

Live canaries:

```bash
RUNPOD_API_KEY=... \
RUNPOD_ENDPOINT_ID=... \
python scripts/canary_compliance_runtime.py
```

```bash
LATENCE_TRACE_API_KEY=... \
LATENCE_TRACE_COMPLIANCE_GATEWAY_URL=https://api.latence.ai/v1/compliance/redact \
python scripts/canary_compliance_runtime.py
```

## InfiniMem And TRACE Memory

| Path | What it proves | Kind |
|---|---|---|
| `tests/test_memory_core.py` | Core memory update, critical spans, deduplication, ranker signal, policy ratios. | Local |
| `tests/test_memory_api_models.py` | Public memory model compatibility. | Local |
| `tests/test_memory_ood_hardening.py` | Out-of-distribution hardening. | Local |
| `tests/test_memory_serious_v1_pipeline.py` | Serious v1 memory pipeline behavior. | Local |
| `tests/test_memory_shadow_integration.py` | Memory shadow integration behavior. | Local |
| `tests/test_memory_domain_horizons.py` | Domain-specific memory horizons. | Local |
| `tests/test_memory_eval_harness.py` | Offline evaluation harness on synthetic trajectories. | Local |
| `tests/test_mem0_trace_adapter.py` | Mem0-shaped adapter behavior and repair ids. | Local |
| `scripts/run_infinimem_real_bootstrap.py` | Real-data bootstrap over LongMemEval, MTRAG, RAGTruth, GaRAGe, SWE-agent, SWE-bench, Tau-bench. | Gated data/network |
| `scripts/run_memory_adaptive_transcript_audit.py` | Cursor transcript audit and long-trace adaptive budget stress. | Local/gated data |
| `scripts/run_memory_llm_ab_equivalence.py` | Full-context vs TRACE-compressed-context LLM equivalence. | Gated LLM |
| `scripts/run_mem0_trace_benchmarks.py` | LOCOMO, LongMemEval, and BEAM via TRACE-backed Mem0 adapter. | Gated checkout/data |
| `scripts/run_stateful_trace_session_proof.py` | Stateful session and hot-context proof. | Local |
| `scripts/run_history_vault_repair_proof.py` | Immutable source-vault and spot-repair proof. | Local |
| `scripts/eval_trace_memory.py` | Offline TRACE Memory evaluation. | Local |
| `scripts/replay_trace_memory.py` | Replayed TRACE signals over memory trajectories. | Local |
| `scripts/score_trace_memory.py` | TRACE Memory scorecards. | Local |
| `scripts/train_trace_memory_baseline.py` | Baseline memory ranker training/eval. | Local/gated data |
| `scripts/prepare_trace_memory_datasets.py` | Canonical TRACE Memory JSONL dataset normalization. | Local/gated data |
| `scripts/promote_trace_memory.py` | Promotion/readiness report from scorecards. | Local |

## General Grounding

| Path | What it proves | Kind |
|---|---|---|
| `research/triangular_maxsim/groundedness_external_eval.py` | External groundedness harness over minimal pairs, RAGTruth, HaluEval, FActScore. | Gated data |
| `research/triangular_maxsim/groundedness_external_benchmarks.py` | Dataset adapters for RAGTruth, HaluEval, FActScore. | Gated data |
| `research/triangular_maxsim/groundedness_minimal_pairs.py` | Deterministic internal minimal-pair fixtures. | Local |
| `scripts/bench_external.py` | Live TRACE benchmark for HaluEval QA/summarization and RAGTruth. | Live/gated data |
| `scripts/diagnose_halueval.py` | Paired correct vs hallucinated HaluEval diagnostics. | Gated data |
| `scripts/diagnose_domain_pairs.py` | Finance/legal/table adversarial domain pairs. | Local/gated data |
| `scripts/bench_veracier_rag_validation.py` | Veracier Industries RAG validation proof. | Local/gated data |
| `scripts/bench_context_coverage.py` | Context coverage observability on real data. | Gated data |
| `scripts/bench_unused_context_precision.py` | Held-out precision gate for unused-context tri-state classification. | Local |
| `scripts/bench_response_chunking.py` | CPU response-chunking parity and scaling microbenchmark. | Local |
| `scripts/bench_routed_replay.py` | End-to-end replay for corpus routing and per-class calibration. | Local live service |
| `tests/test_groundedness_service.py` | Main groundedness service behavior. | Local/gated voyager |
| `tests/test_unused_context.py` | Unused-context tri-state fields and invariants. | Local |
| `tests/core/test_fusion_hardening.py` | Fusion math and missing-channel substitution. | Local |
| `tests/core/test_per_tenant_thresholds.py` | Tenant threshold files and hot reload. | Local |
| `tests/core/corpus_router/*` | Rules, feature schema, classifier, bundles. | Local |

## Code Grounding

| Path | What it proves | Kind |
|---|---|---|
| `tests/test_ast_grounding.py` | Tree-sitter AST grounding and no silent production fallback. | Local |
| `tests/test_code_lane_temporal_drift.py` | Temporal drift, literal novelty, session signal behavior. | Local |
| `tests/test_composite_artifact.py` | Composite/logistic artifact ordering. | Local |
| `tests/test_session_state.py` | Caller-portable code session-state blob. | Local |
| `tests/research/test_coding_trajectory_head.py` | Per-head coding trajectory artifact behavior. | Local |
| `scripts/bench_coding.py` | CRUXEval/HumanEval+ style adversarial coding benchmark. | Live/local service |
| `scripts/bench_transcripts_v2.py` | Mined Cursor coding-agent transcript benchmark. | Local/gated data |
| `tests/test_runpod_live_endpoint.py` | Live RunPod code-lane smoke. | Live |

## RunPod, SDK, Gateway, Latency

| Path | What it proves | Kind |
|---|---|---|
| `tests/test_runpod_handler.py` | Handler config, branches, health payload, compact/canonical contracts. | Local |
| `scripts/trace_core_contract_check.py`, `tests/test_trace_core_contract_check.py` | Product-surface contract parity before benchmarks are accepted as freeze evidence. | Local |
| `tests/test_compression_service.py`, `tests/test_compression_superpod_contract.py` | Compression fallback, preservation, and SuperPod-compatible contract. | Local |
| `tests/test_trace_sessions.py`, `tests/core/test_runtime_decision.py`, `tests/test_customer_breaker_smoke.py` | Sessions, repair packets, runtime decisions, and customer breaker cases. | Local |
| `scripts/verify_trace_v2_readiness.py` | Runtime head registry, policy checksums, and focused readiness tests. | Local |
| `scripts/bench_runpod_live.py` | Small live RunPod benchmark over handcrafted groundedness cases. | Live |
| `scripts/bench_runpod_360.py` | RAG, unused context, code, session state, concurrency, attribution, heatmaps, rollup. | Live |
| `scripts/bench_runpod_360_via_sdk.py` | 360-degree suite routed through Python SDK/gateway. | Live |
| `clients/python/tests/test_client.py` | Python SDK sync/async, retries, auth, typed parsing. | Local |
| `clients/typescript/src/index.test.ts` | TypeScript client score and retry behavior. | Local |
| `scripts/prove_integrations.py` | Raw HTTP, Python client, LangChain/LangGraph/CrewAI/etc., MCP. | Local live service |
| `scripts/bench_latency.py` | Isolated and sustained p50/p95/p99 latency. | Local/live |
| `scripts/bench_response_chunking.py` | Response chunking parity and scaling. | Local/live |
| `tests/test_observability.py` | Request IDs, metrics, lane counters, observability contract. | Local |

## Release Gate

`scripts/verify_trace_v2_readiness.py` should become the local top-level freeze command after its focused test list is reconciled with this matrix. Live and gated scripts should produce a report that records:

- command
- environment gates
- git SHA
- runtime topology
- pass/fail/skipped
- latency and quality thresholds
- artifact output path

Skipped live/gated items are acceptable only when the reason is explicit, for example missing RunPod credentials, missing GLiNER endpoint, missing Hugging Face datasets, or no local dev service.

The executable manifest in `scripts/trace_core_freeze_gate.py` is the current local reproduction runner for this matrix. It deliberately keeps live/gated scripts in the manifest but marks them skipped when credentials or endpoints are absent.
