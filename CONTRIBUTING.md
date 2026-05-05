# Contributing

Latence TRACE has two clear surfaces:

- The compute runtime is Docker/deployment-first and may depend on model, GPU,
  FastAPI, vLLM, and benchmark tooling.
- The public Python SDK package, `latence-trace`, must stay thin. Base
  dependencies are limited to `httpx` and `pydantic`; framework integrations
  belong behind optional extras.

Before opening a PR that changes the public API or SDK surface, run:

```bash
python scripts/trace_feature_inventory.py --write docs/core_freeze/trace_feature_inventory.json
python scripts/trace_feature_inventory.py --check
python scripts/trace_core_contract_check.py
PYTHONPATH=clients/python python -m pytest clients/python/tests/test_client.py -q
```

Do not commit model caches, dataset dumps, local benchmark artifacts, tokens,
or customer data. Keep integrations as adapters only; scoring, redaction,
compression, memory, and rollup logic must live behind the TRACE runtime API.
