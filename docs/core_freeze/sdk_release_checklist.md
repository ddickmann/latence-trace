# SDK Release Checklist

SDK repository: `latence-trace-python`

PyPI package: `latence`

Release invariants:

- `pip install latence` installs the thin TRACE SDK only.
- Base dependencies stay limited to `httpx` and `pydantic`.
- Runtime/model dependencies such as `torch`, `transformers`, `triton`, FastAPI,
  and vLLM are not dependencies of the SDK package.
- The import package is `latence`, exposing `Latence` and `AsyncLatence`.
- Optional framework adapters remain optional extras only.

Required gates before publishing:

```bash
cd /workspace/latence-trace-python
python -m pytest
python -m ruff check .
python scripts/check_contract.py --manifest /workspace/latence-trace/docs/core_freeze/api_surface_manifest.json
python -m build
python -m twine check dist/*
```

Clean-wheel smoke gate:

```bash
python -m venv /tmp/latence-sdk-smoke
/tmp/latence-sdk-smoke/bin/pip install /workspace/latence-trace-python/dist/*.whl
cd /tmp && /tmp/latence-sdk-smoke/bin/python - <<'PY'
from importlib.metadata import distribution
from latence import InMemorySessionStorage, Latence

requires = distribution("latence").requires or []
for forbidden in ("torch", "transformers", "triton", "fastapi", "vllm"):
    assert not any(req.lower().startswith(forbidden) for req in requires), requires

client = Latence(base_url="http://localhost:8090")
session = client.session(session_id="smoke", storage=InMemorySessionStorage())
assert session.session_id == "smoke"
print("latence SDK import smoke passed")
PY
```

Trusted Publisher:

- Publisher: GitHub
- Owner: `latenceainew`
- Repository: `latence-trace-python`
- Workflow filename: `publish.yml`
- Environment name: `pypi`
