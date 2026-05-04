# TRACE Core Freeze

This folder contains the implementation artifacts for freezing `latence-trace` before portal/cloud-hosting work.

- `capability_map.md`: existing features, artifacts, lanes, routing, calibration, heads, memory, and compliance capabilities.
- `benchmark_reproduction_matrix.md`: existing proof scripts and release gates to reproduce.
- `api_contract.md`: product-path contract across FastAPI, RunPod, SDK, and discovery.
- `api_surface_manifest.json`: machine-readable source of truth for product paths, non-canonical surfaces, SDK paths, RunPod actions, examples, and gates.
- `integration_event_schema.md`: standard event/check schema for adapters and plugins.
- `tenant_backend_blueprint.md`: post-freeze tenant backend and portal split.
- `audit_evidence.md`: audit fixes, local proof, live/gated skips, and remaining blockers.
- `examples/`: golden request examples for the v1 product paths.

The core rule: reproduce and preserve the proven runtime capabilities first, then make the product API coherent around them.

Run `python scripts/trace_core_contract_check.py` before benchmark work. If it fails, capability mapping or product API packaging is not frozen yet.
