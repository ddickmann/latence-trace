"""Regression guards for the code-lane AST drift extractor.

These tests lock in production-grade behaviour of the tree-sitter
pipeline. They catch the two classes of silent quality regression we
most fear:

1. The tree-sitter grammars are not installed (e.g. a Docker image
   shipped without pinning the wheels). The extractor would silently
   fall back to a regex parser that cannot detect invented imports.
2. A keyword leaks into the phantom-symbol set (e.g. ``from``,
   ``import``), inflating ``ast_phantom_symbol_count`` with noise.

If either regression lands, CI must fail *before* any image hits the
RunPod endpoint.
"""

from __future__ import annotations

import pytest

from latence_trace.core.code_lane.ast_grounding import (
    AstSymbolExtractor,
    SUPPORTED_LANGUAGES,
    extract_symbols,
    preload_grammars,
    reset_default_extractor,
)


@pytest.fixture(autouse=True)
def _reset_singleton() -> None:
    reset_default_extractor()
    yield
    reset_default_extractor()


def test_tree_sitter_backend_is_active() -> None:
    """The production singleton MUST expose the tree-sitter backend.

    Any other value (``regex_fallback``, ``disabled``) means the
    shipped image is missing grammar wheels and the code lane has
    silently degraded.
    """

    extractor = AstSymbolExtractor(enabled=True)
    assert extractor.parser_backend == "tree_sitter", (
        f"parser_backend must be 'tree_sitter' in production, got "
        f"{extractor.parser_backend!r}. This means tree-sitter grammars "
        "are not installed — the code lane is degraded."
    )


def test_all_supported_grammars_load() -> None:
    """Every language in ``SUPPORTED_LANGUAGES`` must have a loaded
    grammar; partial coverage is a deploy regression."""

    loaded = set(preload_grammars())
    required = set(SUPPORTED_LANGUAGES)
    missing = required - loaded
    assert not missing, (
        f"tree-sitter grammar coverage gap: missing={sorted(missing)}, "
        f"loaded={sorted(loaded)}"
    )
    assert set(AstSymbolExtractor.available_languages()) == required


def test_phantom_import_detected_on_canonical_case() -> None:
    """The canonical ``from photon_labs import Tracer`` phantom must
    be flagged with a non-empty ``phantom_symbols`` list and
    ``ast_phantom_verdict=True``.

    This is exactly the regression surfaced by the 360° live
    benchmark on 2026-04-23: the endpoint was silently on the regex
    fallback and ``ast_phantom_symbol_count`` came back 0 for every
    invented import.
    """

    response = (
        "```python\n"
        "from photon_labs import Tracer\n"
        "async def fetch_user(id: int):\n"
        "    tracer = Tracer()\n"
        "    return await tracer.trace('fetch', id)\n"
        "```"
    )
    context = (
        "```python\n"
        "import httpx\n"
        "async def fetch_user(id: int):\n"
        "    async with httpx.AsyncClient() as client:\n"
        "        return await client.get(f'/users/{id}')\n"
        "```"
    )

    result = extract_symbols(
        response_text=response,
        context_texts=[context],
        language_hint="python",
    )

    assert result.parser_backend == "tree_sitter"
    assert result.ast_phantom_verdict is True
    assert "photon_labs" in result.phantom_symbols
    # Keywords in import statements must never leak into the phantom
    # list — they would inflate ``ast_phantom_symbol_count`` with noise
    # and spuriously fire the deterministic verdict.
    for keyword in ("from", "import", "as"):
        assert keyword not in result.phantom_symbols, (
            f"keyword {keyword!r} leaked into phantom_symbols: "
            f"{result.phantom_symbols}"
        )


def test_grounded_response_has_no_phantom_verdict() -> None:
    """A response whose imports / calls all appear in context must
    not trip the deterministic AST phantom verdict."""

    response = (
        "```python\n"
        "import httpx\n"
        "async def fetch_user(id: int):\n"
        "    async with httpx.AsyncClient(timeout=1.0) as client:\n"
        "        return await client.get(f'/users/{id}')\n"
        "```"
    )
    context = (
        "```python\n"
        "import httpx\n"
        "async def fetch_user(id: int):\n"
        "    async with httpx.AsyncClient() as client:\n"
        "        return await client.get(f'/users/{id}')\n"
        "```"
    )

    result = extract_symbols(
        response_text=response,
        context_texts=[context],
        language_hint="python",
    )

    assert result.parser_backend == "tree_sitter"
    assert result.ast_phantom_verdict is False
    assert result.ast_phantom_symbol_count == 0


def test_supported_language_missing_grammar_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """If a supported language's grammar is missing at request time,
    ``extract_block`` refuses to silently fall back to regex.

    Simulates the worst-case deploy: grammar wheel dropped from the
    image but ``SUPPORTED_LANGUAGES`` still advertises it. The request
    path must raise instead of emitting a lower-quality signal.
    """

    import latence_trace.core.code_lane.ast_grounding as mod

    extractor = AstSymbolExtractor(enabled=True)
    # Pretend no grammars loaded.
    monkeypatch.setattr(mod, "_BUNDLES", {})

    with pytest.raises(RuntimeError, match="tree-sitter grammar for 'python'"):
        extractor.extract_block("python", "x = 1\n")
