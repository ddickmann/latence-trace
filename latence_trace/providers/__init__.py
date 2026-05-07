"""Provider implementations (NLI, reranker, encoders, compression, GLiNER).

Each module here exposes a thin client for one external runtime
(transformers, vLLM, etc.) and stays import-light: heavy imports
(``transformers``, ``torch``, ``httpx``) live behind lazy methods so
``import latence_trace`` stays fast and side-effect-free.

Sub-modules:

* :mod:`.nli` — :class:`VllmFactoryNLIProvider` (HTTP /pooling client
  for the legacy ``nli_mdeberta`` plugin and the future
  ``minicheck_t5`` BYOP plugin).
* :mod:`.nli_classify` — :class:`VllmClassifyNLIProvider` (HTTP
  /classify client for vLLM-native ``--convert classify`` servers,
  e.g. ``bge-m3-zeroshot-v2.0``).
* :mod:`.nli_transformers` — in-process transformers fallbacks
  (:class:`MiniCheckNLIProvider`, :class:`BgeM3ZeroShotNLIProvider`)
  that the registry falls back to when no vLLM endpoint is configured.
* :mod:`.granite_guardian` — :class:`GraniteGuardianNLIProvider`
  (vLLM-served IBM Granite Guardian 4.1; multilingual groundedness
  via token logprobs, supersedes per-language providers when
  ``LATENCE_TRACE_GUARDIAN_ENDPOINT`` is set).
* :mod:`.nli_registry` — :func:`resolve_nli_provider` (language-aware
  factory over the above).
* :mod:`.reranker` — :class:`VllmRerankerProvider` and
  :func:`resolve_reranker` (HTTP /v1/score client for vLLM-served
  cross-encoders, falls back to the in-process
  :class:`latence_trace.core.nli.CrossEncoderPremiseReranker`).

Shared HTTP transport
---------------------

:func:`get_shared_async_client` returns a process-global
:class:`httpx.AsyncClient` keyed by the active event loop, with
``Limits(max_connections=64, max_keepalive_connections=32)`` and a
120-second timeout. Every vLLM-backed provider (NLI x2, reranker)
uses it so the three endpoints share one connection pool, one set of
keepalive sessions, and one set of HTTP/1.1 sockets — instead of each
provider holding its own pool, three of which sit idle most of the
time. This matters once the hot path fires NLI verify_claims +
NLI semantic_entropy + reranker batched POST concurrently:
shared-pool keepalive amortises TLS / TCP handshake cost across the
fan-out.

The client is rebuilt on event-loop swap (relevant for test suites
that spin up fresh loops per test). Provider classes still hold a
reference to the same instance per loop so ``close()`` /
``aclose()`` lifecycles stay deterministic.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Optional

# Single source of truth for vLLM HTTP timeouts and pool sizing across
# every provider. Per-provider overrides are still honoured (callers
# pass their own ``timeout=...`` to ``client.post``) but the pool
# limits are global so concurrent fan-outs don't fight for sockets.
_SHARED_TIMEOUT = 120.0
_SHARED_MAX_CONNECTIONS = 64
_SHARED_MAX_KEEPALIVE = 32

_shared_client: Any = None
_shared_client_loop: Optional[asyncio.AbstractEventLoop] = None
_shared_client_lock = threading.Lock()


def get_shared_async_client() -> Any:
    """Return the process-global :class:`httpx.AsyncClient` for this loop.

    Lazy-imports ``httpx`` so module import stays free; rebuilds on
    event-loop swap so tests and reconnect paths stay safe.
    """

    global _shared_client, _shared_client_loop

    try:
        import httpx
    except ImportError as exc:  # pragma: no cover - optional install
        raise ImportError(
            "httpx is required for the shared async transport. "
            "Install latence-trace with the [vllm-client] extra."
        ) from exc

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No active loop — provider sync entry points should call
        # ``asyncio.run`` and have a running loop by the time we get
        # here. If a caller still asks without one, return the cached
        # client (or build a no-loop one) so we don't crash hard.
        loop = None

    with _shared_client_lock:
        if _shared_client is not None and _shared_client_loop is loop:
            return _shared_client
        # Loop changed (or first call) → build a fresh client. We let
        # the previous client leak its sockets to GC; the keepalive
        # pool releases them on the next gc cycle. Calling .aclose()
        # here would require an event loop and risks a deadlock if the
        # caller is mid-request, so we keep the swap path light.
        limits = httpx.Limits(
            max_connections=_SHARED_MAX_CONNECTIONS,
            max_keepalive_connections=_SHARED_MAX_KEEPALIVE,
        )
        _shared_client = httpx.AsyncClient(
            timeout=_SHARED_TIMEOUT,
            limits=limits,
        )
        _shared_client_loop = loop
        return _shared_client


async def close_shared_async_client() -> None:
    """Close the shared client. Safe to call from a coroutine."""

    global _shared_client, _shared_client_loop

    with _shared_client_lock:
        client = _shared_client
        _shared_client = None
        _shared_client_loop = None
    if client is not None:
        try:
            await client.aclose()
        except Exception:  # pragma: no cover - best-effort close
            pass


__all__ = [
    "get_shared_async_client",
    "close_shared_async_client",
]
