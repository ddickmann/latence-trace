# SDK Release Checklist

Package name: `latence-trace`

Release invariants:

- `pip install latence-trace` installs the thin SDK only.
- Base dependencies stay limited to `httpx` and `pydantic`.
- Runtime/model dependencies such as `torch`, `transformers`, `triton`, FastAPI,
  and vLLM are not dependencies of the SDK package.
- The SDK package exposes `latence_trace_client` and does not conflict with any
  future runtime package.
- Optional framework adapters remain optional extras only.

Required gates before publishing:

```bash
python scripts/trace_feature_inventory.py --check
python scripts/trace_core_contract_check.py
PYTHONPATH=clients/python python -m pytest clients/python/tests/test_client.py -q
cd clients/python && python -m build
cd clients/python && python -m twine check dist/*
```

Clean-wheel smoke gate:

```bash
python -m venv /tmp/latence-trace-sdk-smoke
/tmp/latence-trace-sdk-smoke/bin/pip install clients/python/dist/*.whl
cd /tmp && /tmp/latence-trace-sdk-smoke/bin/python - <<'PY'
from importlib.metadata import distribution
from latence_trace_client import InMemorySessionStorage, LatenceTraceClient

requires = distribution("latence-trace").requires or []
for forbidden in ("torch", "transformers", "triton", "fastapi", "vllm"):
    assert not any(req.lower().startswith(forbidden) for req in requires), requires

client = LatenceTraceClient(base_url="http://localhost:8090")
session = client.session(session_id="smoke", storage=InMemorySessionStorage())
assert session.session_id == "smoke"
print("latence-trace SDK import smoke passed")
PY
```
