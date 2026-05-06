"""Unit tests for the language-aware NLI registry + handler plumbing.

Validates the env-var fall-through matrix the registry consumes:

* No NLI feature flag → no provider returned (registry returns ``None``).
* Feature flag on, no per-language config → legacy single-NLI lane
  (``HuggingFaceNLIProvider`` mDeBERTa).
* Feature flag on, per-language opt-in → matching transformers
  fallback (``MiniCheckNLIProvider`` for ``en``,
  ``BgeM3ZeroShotNLIProvider`` for ``de``).
* Feature flag on, per-language vLLM endpoint set but unhealthy → falls
  through to the matching transformers fallback (graceful degradation).

Also asserts the handler ``_build_servers`` adds the new ``nli_multi``
and ``reranker`` server entries when the env flags flip on, and skips
the MiniCheck server while logging when its BYOP plugin is missing.

Heavy ``transformers`` weight loading is *not* triggered: the providers
returned are class instances; their ``_ensure_loaded`` only runs on
the first ``entail`` / ``score`` call.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


REGISTRY_ENV_VARS = (
    "VOYAGER_GROUNDEDNESS_NLI_ENABLED",
    "LATENCE_TRACE_NLI_DUAL_ENABLED",
    "LATENCE_TRACE_NLI_EN_ENABLED",
    "LATENCE_TRACE_NLI_DE_ENABLED",
    "LATENCE_TRACE_NLI_EN_ENDPOINT",
    "LATENCE_TRACE_NLI_DE_ENDPOINT",
    "LATENCE_TRACE_NLI_EN_MODEL",
    "LATENCE_TRACE_NLI_DE_MODEL",
    "LATENCE_TRACE_NLI_EN_TRANSFORMERS_MODEL",
    "LATENCE_TRACE_NLI_DE_TRANSFORMERS_MODEL",
    "LATENCE_TRACE_NLI_EN_PROTOCOL",
    "LATENCE_TRACE_NLI_DE_PROTOCOL",
    "LATENCE_TRACE_NLI_VLLM_ENDPOINT",
    "LATENCE_TRACE_NLI_VLLM_MODEL",
    "LATENCE_TRACE_RERANKER_ENDPOINT",
    "LATENCE_TRACE_NLI_EN_ENABLED",
    "LATENCE_TRACE_NLI_MULTI_ENABLED",
    "LATENCE_TRACE_RERANKER_ENABLED",
)


@contextmanager
def _scrubbed_registry_env() -> Iterator[None]:
    """Snapshot + clear all NLI env vars; restore on exit."""

    backup = {key: os.environ.get(key) for key in REGISTRY_ENV_VARS}
    for key in REGISTRY_ENV_VARS:
        os.environ.pop(key, None)
    try:
        from latence_trace.providers.nli_registry import reset_cache

        reset_cache()
        yield
    finally:
        for key, value in backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        from latence_trace.providers.nli_registry import reset_cache as reset_cache_after

        reset_cache_after()


def test_registry_returns_none_when_feature_flag_off() -> None:
    from latence_trace.providers.nli_registry import resolve_nli_provider

    with _scrubbed_registry_env():
        assert resolve_nli_provider("en") is None
        assert resolve_nli_provider("de") is None
        assert resolve_nli_provider(None) is None


def test_registry_returns_legacy_when_no_per_language_config() -> None:
    from latence_trace.providers.nli_registry import resolve_nli_provider

    with _scrubbed_registry_env():
        os.environ["VOYAGER_GROUNDEDNESS_NLI_ENABLED"] = "1"
        from latence_trace.providers.nli_registry import reset_cache

        reset_cache()
        provider = resolve_nli_provider(None)
        # Either HuggingFaceNLIProvider (default fallback) or
        # VllmFactoryNLIProvider (when LATENCE_TRACE_NLI_VLLM_ENDPOINT
        # is set in the local env outside this scrub) — but never the
        # SOTA per-language classes.
        assert provider is not None
        assert type(provider).__name__ in {
            "HuggingFaceNLIProvider",
            "VllmFactoryNLIProvider",
        }


def test_registry_routes_en_to_minicheck_when_per_language_opt_in() -> None:
    from latence_trace.providers.nli_registry import resolve_nli_provider

    with _scrubbed_registry_env():
        os.environ["VOYAGER_GROUNDEDNESS_NLI_ENABLED"] = "1"
        os.environ["LATENCE_TRACE_NLI_EN_ENABLED"] = "1"
        from latence_trace.providers.nli_registry import reset_cache

        reset_cache()
        provider = resolve_nli_provider("en")
        assert provider is not None
        assert type(provider).__name__ == "MiniCheckNLIProvider"


def test_registry_routes_de_to_bge_m3_when_per_language_opt_in() -> None:
    from latence_trace.providers.nli_registry import resolve_nli_provider

    with _scrubbed_registry_env():
        os.environ["VOYAGER_GROUNDEDNESS_NLI_ENABLED"] = "1"
        os.environ["LATENCE_TRACE_NLI_DE_ENABLED"] = "1"
        from latence_trace.providers.nli_registry import reset_cache

        reset_cache()
        provider = resolve_nli_provider("de")
        assert provider is not None
        assert type(provider).__name__ == "BgeM3ZeroShotNLIProvider"


def test_registry_falls_through_to_transformers_when_vllm_unhealthy() -> None:
    """An unreachable vLLM endpoint must not silently degrade groundedness."""

    from latence_trace.providers.nli_registry import resolve_nli_provider

    with _scrubbed_registry_env():
        os.environ["VOYAGER_GROUNDEDNESS_NLI_ENABLED"] = "1"
        os.environ["LATENCE_TRACE_NLI_EN_ENABLED"] = "1"
        # Pick a closed port; healthcheck must fail and we should fall
        # through to the in-process MiniCheck fallback.
        os.environ["LATENCE_TRACE_NLI_EN_ENDPOINT"] = "http://127.0.0.1:1"
        from latence_trace.providers.nli_registry import reset_cache

        reset_cache()
        provider = resolve_nli_provider("en")
        assert provider is not None
        assert type(provider).__name__ == "MiniCheckNLIProvider"


def test_registry_caches_per_language() -> None:
    from latence_trace.providers.nli_registry import resolve_nli_provider

    with _scrubbed_registry_env():
        os.environ["VOYAGER_GROUNDEDNESS_NLI_ENABLED"] = "1"
        os.environ["LATENCE_TRACE_NLI_EN_ENABLED"] = "1"
        os.environ["LATENCE_TRACE_NLI_DE_ENABLED"] = "1"
        from latence_trace.providers.nli_registry import reset_cache

        reset_cache()
        en1 = resolve_nli_provider("en")
        en2 = resolve_nli_provider("en")
        de1 = resolve_nli_provider("de")
        de2 = resolve_nli_provider("de")
        assert en1 is en2, "per-language cache must reuse the same provider instance"
        assert de1 is de2, "per-language cache must reuse the same provider instance"
        assert en1 is not de1, "different languages must get different providers"


def test_registry_normalises_unsupported_language_to_legacy_lane() -> None:
    from latence_trace.providers.nli_registry import resolve_nli_provider

    with _scrubbed_registry_env():
        os.environ["VOYAGER_GROUNDEDNESS_NLI_ENABLED"] = "1"
        from latence_trace.providers.nli_registry import reset_cache

        reset_cache()
        # ``fr`` is not in {en, de}; the registry must collapse to the
        # legacy lane instead of crashing.
        provider = resolve_nli_provider("fr")
        assert provider is not None
        assert type(provider).__name__ in {
            "HuggingFaceNLIProvider",
            "VllmFactoryNLIProvider",
        }


# ---------------------------------------------------------------------------
# Handler _build_servers gating
# ---------------------------------------------------------------------------


def _load_handler():
    """Import the runpod handler with the right ``server`` module alias.

    ``handler.py`` does ``from server import ManagedVllmServer`` which
    resolves to ``runpod/server.py`` only when ``runpod/`` is on
    ``sys.path`` *and* no other test has cached the workspace-root
    ``server`` package. We do the same dance the existing
    ``tests/test_runpod_handler.py`` uses: insert ``runpod/`` on the
    path, pop any pre-cached ``server`` module, exec the handler
    fresh, then restore the original ``server`` package so downstream
    tests still see ``server.main`` etc.
    """

    runpod_dir = ROOT / "runpod"
    saved_path = list(sys.path)
    saved_server = sys.modules.pop("server", None)
    saved_server_subs = {
        name: sys.modules.pop(name)
        for name in list(sys.modules.keys())
        if name == "server" or name.startswith("server.")
    }
    sys.path.insert(0, str(runpod_dir))
    try:
        spec = importlib.util.spec_from_file_location(
            "_handler_under_test", runpod_dir / "handler.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        # Restore: pop the runpod ``server`` shim and re-install the
        # workspace-root package so other tests can import ``server.main``.
        sys.modules.pop("server", None)
        for name in list(sys.modules.keys()):
            if name == "server" or name.startswith("server."):
                del sys.modules[name]
        if saved_server is not None:
            sys.modules["server"] = saved_server
        for name, mod in saved_server_subs.items():
            sys.modules[name] = mod
        sys.path[:] = saved_path


@pytest.fixture
def handler_module():
    module = _load_handler()
    yield module
    # Best-effort: remove from cache so other tests can re-import fresh.
    sys.modules.pop("_handler_under_test", None)


def test_build_servers_default_topology_drops_legacy_nli(handler_module) -> None:
    """The default topology MUST drop the legacy mDeBERTa ``nli`` server
    and boot the SOTA trio (``nli_en`` + ``nli_multi`` + ``reranker``)
    without any opt-in env vars. This is the regression gate for
    Phase 1 of the SOTA stack rollout — flipping back to the legacy
    single-NLI topology should require an explicit env override
    (``LATENCE_TRACE_NLI_EN_ENABLED=0`` etc), not the absence of an
    opt-in.
    """

    import unittest.mock as mock

    backup = {
        key: os.environ.get(key)
        for key in (
            "LATENCE_TRACE_START_MANAGED_VLLM",
            "LATENCE_TRACE_NLI_EN_ENABLED",
            "LATENCE_TRACE_NLI_MULTI_ENABLED",
            "LATENCE_TRACE_RERANKER_ENABLED",
            "LATENCE_TRACE_ENABLE_COMPRESSION_SERVER",
        )
    }
    try:
        os.environ["LATENCE_TRACE_START_MANAGED_VLLM"] = "1"
        # Pop any opt-in flags so we exercise the bare defaults.
        os.environ.pop("LATENCE_TRACE_NLI_EN_ENABLED", None)
        os.environ.pop("LATENCE_TRACE_NLI_MULTI_ENABLED", None)
        os.environ.pop("LATENCE_TRACE_RERANKER_ENABLED", None)
        # Keep compression off in the unit test to avoid pulling in the
        # external LLMLingua2 weight check (covered by separate tests).
        os.environ["LATENCE_TRACE_ENABLE_COMPRESSION_SERVER"] = "0"

        with mock.patch.object(
            handler_module, "_minicheck_plugin_available", return_value=True
        ):
            config = handler_module.create_config()
            servers = handler_module._build_servers(config)
        # Legacy mDeBERTa server is permanently dropped from the default
        # topology — this is the Phase 1 contract.
        assert "nli" not in servers
        assert set(servers.keys()) == {
            "colbert",
            "compliance_gliner",
            "nli_en",
            "nli_multi",
            "reranker",
        }
    finally:
        for key, value in backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_build_servers_never_boots_legacy_mdeberta_even_with_opt_in(handler_module) -> None:
    """Belt-and-braces: even when callers set legacy ``LATENCE_TRACE_NLI_*``
    env vars, the legacy ``nli`` server must NOT be booted. The fields
    survive on ``WorkerConfig`` only as deprecated diagnostic carriers.
    """

    import unittest.mock as mock

    backup = {
        key: os.environ.get(key)
        for key in (
            "LATENCE_TRACE_START_MANAGED_VLLM",
            "LATENCE_TRACE_ENABLE_COMPRESSION_SERVER",
            "LATENCE_TRACE_NLI_GPU_MEM",
            "LATENCE_TRACE_NLI_PORT",
        )
    }
    try:
        os.environ["LATENCE_TRACE_START_MANAGED_VLLM"] = "1"
        os.environ["LATENCE_TRACE_ENABLE_COMPRESSION_SERVER"] = "0"
        os.environ["LATENCE_TRACE_NLI_GPU_MEM"] = "0.2"
        os.environ["LATENCE_TRACE_NLI_PORT"] = "8002"

        with mock.patch.object(
            handler_module, "_minicheck_plugin_available", return_value=True
        ):
            config = handler_module.create_config()
            servers = handler_module._build_servers(config)
        assert "nli" not in servers
    finally:
        for key, value in backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_default_vllm_gpu_mem_knob_shifts_all_servers_in_lockstep(handler_module) -> None:
    """``LATENCE_TRACE_VLLM_GPU_MEM_DEFAULT`` is the OOM-fallback knob:
    setting it to ``0.135`` must shift the default per-server budget
    for *every* lane that doesn't carry an explicit override. We test
    by re-importing the handler module with the env var set, then
    asserting the new ``_DEFAULT_VLLM_GPU_MEM`` constant + each
    per-server default propagated."""

    import importlib

    backup = os.environ.get("LATENCE_TRACE_VLLM_GPU_MEM_DEFAULT")
    saved_module = sys.modules.pop("_handler_under_test", None)
    try:
        os.environ["LATENCE_TRACE_VLLM_GPU_MEM_DEFAULT"] = "0.135"
        # Pop any per-server overrides so we exercise the shared default.
        for key in (
            "LATENCE_TRACE_COLBERT_GPU_MEM",
            "LATENCE_TRACE_COMPLIANCE_GLINER_GPU_MEM",
            "LATENCE_TRACE_COMPRESSION_GPU_MEM",
            "LATENCE_TRACE_NLI_EN_GPU_MEM",
            "LATENCE_TRACE_NLI_MULTI_GPU_MEM",
            "LATENCE_TRACE_RERANKER_GPU_MEM",
        ):
            os.environ.pop(key, None)

        module = _load_handler()
        assert module._DEFAULT_VLLM_GPU_MEM == pytest.approx(0.135)
        config = module.create_config()
        # All six co-resident lanes track the shared default.
        assert config.colbert_gpu_mem == pytest.approx(0.135)
        assert config.compliance_gpu_mem == pytest.approx(0.135)
        assert config.compression_gpu_mem == pytest.approx(0.135)
        assert config.nli_en_gpu_mem == pytest.approx(0.135)
        assert config.nli_multi_gpu_mem == pytest.approx(0.135)
        assert config.reranker_gpu_mem == pytest.approx(0.135)
    finally:
        if backup is None:
            os.environ.pop("LATENCE_TRACE_VLLM_GPU_MEM_DEFAULT", None)
        else:
            os.environ["LATENCE_TRACE_VLLM_GPU_MEM_DEFAULT"] = backup
        sys.modules.pop("_handler_under_test", None)
        if saved_module is not None:
            sys.modules["_handler_under_test"] = saved_module


def _opt_in_three_servers_env() -> dict[str, str | None]:
    """Snapshot + set the env vars that opt the three new servers in."""

    keys = (
        "LATENCE_TRACE_START_MANAGED_VLLM",
        "LATENCE_TRACE_NLI_EN_ENABLED",
        "LATENCE_TRACE_NLI_MULTI_ENABLED",
        "LATENCE_TRACE_RERANKER_ENABLED",
        "LATENCE_TRACE_ENABLE_COMPRESSION_SERVER",
    )
    backup = {key: os.environ.get(key) for key in keys}
    os.environ["LATENCE_TRACE_START_MANAGED_VLLM"] = "1"
    os.environ["LATENCE_TRACE_NLI_EN_ENABLED"] = "1"
    os.environ["LATENCE_TRACE_NLI_MULTI_ENABLED"] = "1"
    os.environ["LATENCE_TRACE_RERANKER_ENABLED"] = "1"
    os.environ["LATENCE_TRACE_ENABLE_COMPRESSION_SERVER"] = "0"
    return backup


def _restore_env(backup: dict[str, str | None]) -> None:
    for key, value in backup.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def test_build_servers_skips_nli_en_when_plugin_missing(handler_module) -> None:
    """If ``minicheck_t5`` is not importable we MUST skip ``nli_en`` and
    log the warning, even when the operator opted in via env. The
    in-process :class:`MiniCheckNLIProvider` fallback in
    :mod:`latence_trace.providers.nli_registry` keeps the request path
    working in the meantime.
    """

    import unittest.mock as mock

    backup = _opt_in_three_servers_env()
    try:
        with mock.patch.object(
            handler_module, "_minicheck_plugin_available", return_value=False
        ):
            config = handler_module.create_config()
            servers = handler_module._build_servers(config)
        assert "nli_en" not in servers
        assert "nli_multi" in servers
        assert "reranker" in servers
        # Legacy mDeBERTa never co-exists with the SOTA trio.
        assert "nli" not in servers

        # vLLM 0.19.x renamed ``--task`` to ``--convert``; both
        # nli_multi (XLM-R classify head) and reranker (single-logit
        # cross-encoder, exposed at /v1/score automatically when
        # ``--convert classify`` is set) ride the same flag.
        nli_multi_cmd = servers["nli_multi"]._build_command()
        assert "--convert" in nli_multi_cmd
        assert nli_multi_cmd[nli_multi_cmd.index("--convert") + 1] == "classify"

        reranker_cmd = servers["reranker"]._build_command()
        assert "--convert" in reranker_cmd
        assert reranker_cmd[reranker_cmd.index("--convert") + 1] == "classify"
    finally:
        _restore_env(backup)


def test_build_servers_boots_nli_en_when_plugin_available(handler_module) -> None:
    """When ``minicheck_t5`` IS importable AND the operator opted in,
    the handler MUST boot the MiniCheck server with the right model id,
    pooling runner and BYOP plugin wiring (``minicheck_t5`` general
    plugin + ``minicheck_t5_io`` io processor)."""

    import unittest.mock as mock

    backup = _opt_in_three_servers_env()
    try:
        with mock.patch.object(
            handler_module, "_minicheck_plugin_available", return_value=True
        ):
            config = handler_module.create_config()
            servers = handler_module._build_servers(config)

        assert "nli_en" in servers
        assert "nli_multi" in servers
        assert "reranker" in servers
        # Legacy mDeBERTa stays dropped under all enabled-flag matrices.
        assert "nli" not in servers

        nli_en = servers["nli_en"]
        assert nli_en.model == "lytang/MiniCheck-Flan-T5-Large"
        assert nli_en.io_processor_plugin == "minicheck_t5_io"
        # Both the general plugin (registers the model class) and the IO
        # processor plugin (registers the wire shape) must be loaded.
        assert "minicheck_t5" in nli_en.plugins
        assert "minicheck_t5_io" in nli_en.plugins

        nli_en_cmd = nli_en._build_command()
        assert "--runner" in nli_en_cmd
        assert nli_en_cmd[nli_en_cmd.index("--runner") + 1] == "pooling"
    finally:
        _restore_env(backup)
