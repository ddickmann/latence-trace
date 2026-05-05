# SDK Runtime Freeze Audit

## Plan Reflection

- Mechanical feature discovery gate: implemented `scripts/trace_feature_inventory.py`, emitted `docs/core_freeze/trace_feature_inventory.json`, and made `--check` fail on route, RunPod action, manifest, SDK, and example drift.
- Plan file update: the attached final plan already contains the Phase 4 SDK/runtime freeze scope. Per instruction, this audit did not edit `/root/.cursor/plans/sdk_runtime_freeze_063e4089.plan.md`.
- SDK guard surface: `context_trust_enabled` is explicit on product namespace and session grounding methods, defaults to enabled, and can be disabled per request.
- SDK product API: sync and async clients expose privacy, grounding, compression, memory, rollup, and session surfaces with typed stable responses and raw/extra passthrough.
- Stateful SDK features: `TraceSession` and `AsyncTraceSession` now keep session id, memory state, local events, idempotency keys, metadata, and storage-backed snapshots in the SDK.
- Public repo/package: SDK package metadata is `latence-trace`, base dependencies are thin, examples and release checklist are present, and generated/cache paths are ignored.
- Generic API server: `runpod/api_server.py` exposes native FastAPI product paths plus `/run` and `/runsync` compatibility while reusing the RunPod runtime lifecycle.
- Observability 360: `docs/core_freeze/endpoint_observability_360.md` documents endpoint evidence fields and residual observability risks.
- Implementation audit: this pass fixed an inventory-script default-argument bug, cleaned SDK license metadata, hardened clean-wheel checks against repo-root metadata shadowing, and fixed SDK parsing of native `/groundedness` responses.
- Proof gates: inventory, contract, unit, lint, package, clean-wheel, Prompt Guard compile, local RunPod-shaped 360, generic API smoke, and cleanup gates passed.

## Findings Fixed

- Fixed `trace_feature_inventory.py` so functions with no positional defaults do not trigger a `zip(strict=True)` mismatch.
- Fixed SDK package metadata from deprecated `license = { text = "MIT" }` to `license = "MIT"` so future Setuptools builds stay clean.
- Fixed the clean-wheel release checklist to run from `/tmp`, avoiding accidental metadata shadowing from the compute-engine repo.
- Fixed SDK native FastAPI parsing: direct `/groundedness` responses can carry `risk_band` under `scores.risk_band`, and the SDK now maps that into product-level `green` / `amber` / `red` bands.
- Added a regression test for native groundedness response parsing.

## Gates Run

- `python scripts/trace_feature_inventory.py --write docs/core_freeze/trace_feature_inventory.json`
- `python scripts/trace_feature_inventory.py --check`
- `python scripts/trace_core_contract_check.py`
- `PYTHONPATH=/workspace/latence-trace python -m pytest tests/test_context_trust.py tests/test_runpod_handler.py tests/test_generic_api_server.py -q`
- `cd /workspace/latence-trace-python && python -m pytest && python -m ruff check .`
- `python -m ruff check scripts/trace_feature_inventory.py scripts/trace_core_contract_check.py runpod/api_server.py tests/test_generic_api_server.py`
- `python -m build`
- `python -m twine check dist/*`
- Clean-wheel install in `/tmp/latence-trace-sdk-smoke`, with dependency audit proving no base `torch`, `transformers`, `triton`, `fastapi`, or `vllm` dependency.
- Local managed-vLLM `runpod/dev_app.py` proof with ColBERT, NLI, GLiNER, and compression servers healthy.
- `scripts/bench_runpod_360_local.py --concurrency 1 --burst-size 4 --dump docs/core_freeze/local_sdk_runtime_freeze_360.json`
- Local managed-vLLM `runpod/api_server.py` smoke over `/groundedness`, `/v1/compliance/redact`, `/v1/compression`, `/v1/memory/update`, `/groundedness/rollup`, and `/runsync`.
- Prompt Guard provider warmup with `LATENCE_TRACE_CONTEXT_TRUST_PROVIDER=prompt_guard`, CUDA device, `torch.compile` enabled, and `compiled=true`.
- Runtime cleanup: no dev app, generic API server, vLLM, or GPU processes remained after proof.

## Evidence Highlights

- Focused tests: `67 passed`.
- Local 360 benchmark: overall `PASS`, written to `docs/core_freeze/local_sdk_runtime_freeze_360.json`.
- Prompt Guard compile: provider `prompt_guard`, model `meta-llama/Llama-Prompt-Guard-2-86M`, device `cuda`, compile mode `reduce-overhead`, compiled `true`.
- Generic API smoke: all managed servers healthy; guard enabled and disabled diagnostics surfaced correctly; redaction, compression, memory, rollup, and `/runsync` compatibility all returned valid payloads.

## Residual Risks

- `env_settings_to_review` intentionally remains a non-blocking inventory section because many `LATENCE_TRACE_*` flags are deployment tuning controls rather than public SDK contract fields.
- The SDK now lives in `latence-trace-python`, publishes the existing `latence` PyPI package, and imports as `latence`; runtime gates consume it as a sibling checkout instead of in-repo source.

## Verdict

Ready for SDK/runtime freeze. The implementation is verified against the plan with passing mechanical discovery, contract, unit, lint, packaging, clean-wheel, managed-runtime, Prompt Guard compile, RunPod-shaped, and generic API proof gates.
