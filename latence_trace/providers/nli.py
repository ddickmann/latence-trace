"""HTTP-backed NLI provider for a live vLLM pooling endpoint.

The provider exposes both a synchronous ``entail`` and an asynchronous
:meth:`VllmFactoryNLIProvider.entail_async` that shares a pooled
:class:`httpx.AsyncClient`. The async API lets the FastAPI / RunPod
handler await NLI from the event loop without parking a thread, which
matters once the code lane's ambiguity-gated cascade fires on ~30% of
coding-agent traffic.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Sequence


class VllmFactoryNLIProvider:
    """Thin HTTP client for the latence-trace NLI vLLM plugin.

    Parameters
    ----------
    endpoint, model:
        Base URL and model tag for the vLLM pooling plugin.
    timeout, health_timeout:
        Per-request timeouts in seconds.
    max_concurrency:
        Caps the keepalive connection pool size. Matched by the
        vLLM-side ``max_num_seqs`` so bursts never backlog at the
        client.
    """

    def __init__(
        self,
        endpoint: str,
        model: str,
        *,
        timeout: float = 60.0,
        health_timeout: float = 10.0,
        max_concurrency: int = 8,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.timeout = float(timeout)
        self.health_timeout = float(health_timeout)
        self.max_concurrency = max(1, int(max_concurrency))
        self._http_client: Any = None
        self._client_lock = threading.Lock()
        # NOTE: The async client is process-shared (see
        # :func:`latence_trace.providers.get_shared_async_client`).
        # No per-provider async-client state is held here; the sync
        # ``httpx.Client`` above is still per-provider so legacy sync
        # callers don't get blocked on shared-pool capacity.

    def _get_http_client(self):
        if self._http_client is not None:
            return self._http_client
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - optional install shape
            raise ImportError("httpx is required for VllmFactoryNLIProvider.") from exc

        with self._client_lock:
            if self._http_client is not None:
                return self._http_client
            limits = httpx.Limits(
                max_connections=max(self.max_concurrency * 2, 8),
                max_keepalive_connections=max(self.max_concurrency, 4),
            )
            self._http_client = httpx.Client(
                base_url=self.endpoint,
                timeout=self.timeout,
                limits=limits,
            )
        return self._http_client

    def _get_async_client(self):
        """Return the process-shared :class:`httpx.AsyncClient`.

        See :func:`latence_trace.providers.get_shared_async_client` for
        pooling rationale: one client per process means concurrent NLI
        + reranker fan-outs reuse the same TCP keepalive sessions and
        don't fight for sockets in the per-loop pool.
        """

        from latence_trace.providers import get_shared_async_client

        return get_shared_async_client()

    def close(self) -> None:
        with self._client_lock:
            if self._http_client is not None:
                try:
                    self._http_client.close()
                finally:
                    self._http_client = None
        # The async client is process-shared (see
        # :func:`latence_trace.providers.get_shared_async_client`) — we
        # MUST NOT close it from a per-provider lifecycle hook because
        # other providers / requests are still using it.

    async def aclose(self) -> None:
        """Async-aware close; safe to call from an asyncio coroutine.

        Closes only this provider's *sync* httpx.Client. The shared
        async client is owned by the providers package and closed via
        :func:`latence_trace.providers.close_shared_async_client` at
        process shutdown, not per-provider.
        """

        with self._client_lock:
            if self._http_client is not None:
                try:
                    self._http_client.close()
                finally:
                    self._http_client = None

    def healthcheck(self) -> dict[str, Any]:
        client = self._get_http_client()
        response = client.get("/health", timeout=self.health_timeout)
        response.raise_for_status()
        try:
            return response.json()
        except Exception:
            return {"status": response.text}

    @staticmethod
    def _unwrap_data(payload: Any) -> Any:
        current = payload
        while isinstance(current, dict) and "data" in current:
            current = current["data"]
        return current

    @staticmethod
    def _coerce_row(row: Any) -> tuple[float, float, float]:
        if isinstance(row, dict):
            return (
                float(row.get("entail", 0.0)),
                float(row.get("neutral", 1.0)),
                float(row.get("contradict", 0.0)),
            )
        if isinstance(row, (list, tuple)) and len(row) >= 3:
            return (float(row[0]), float(row[1]), float(row[2]))
        raise TypeError(f"Unsupported NLI payload row: {type(row)!r}")

    def entail(
        self,
        premises: Sequence[str],
        hypotheses: Sequence[str],
    ) -> list[tuple[float, float, float]]:
        if len(premises) != len(hypotheses):
            raise ValueError("premises and hypotheses must align")
        if not premises:
            return []

        client = self._get_http_client()
        response = client.post(
            "/pooling",
            json={
                "model": self.model,
                "task": "plugin",
                "data": {
                    "premise": list(premises),
                    "hypothesis": list(hypotheses),
                },
            },
        )
        response.raise_for_status()

        body = response.json()
        rows = self._unwrap_data(body)
        if isinstance(rows, dict):
            rows = [rows]
        if not isinstance(rows, list):
            raise TypeError(f"Unsupported NLI response payload: {type(rows)!r}")
        return [self._coerce_row(row) for row in rows]

    async def entail_async(
        self,
        premises: Sequence[str],
        hypotheses: Sequence[str],
    ) -> list[tuple[float, float, float]]:
        """Async counterpart to :meth:`entail`.

        Shares a pooled :class:`httpx.AsyncClient` so coroutines on the
        same event loop re-use keepalive connections. Safe to await
        from the FastAPI thread pool *and* from the RunPod handler's
        native async path.
        """
        if len(premises) != len(hypotheses):
            raise ValueError("premises and hypotheses must align")
        if not premises:
            return []

        client = self._get_async_client()
        # Shared client carries no base_url; pass the absolute URL.
        response = await client.post(
            f"{self.endpoint}/pooling",
            json={
                "model": self.model,
                "task": "plugin",
                "data": {
                    "premise": list(premises),
                    "hypothesis": list(hypotheses),
                },
            },
            timeout=self.timeout,
        )
        response.raise_for_status()

        body = response.json()
        rows = self._unwrap_data(body)
        if isinstance(rows, dict):
            rows = [rows]
        if not isinstance(rows, list):
            raise TypeError(f"Unsupported NLI response payload: {type(rows)!r}")
        return [self._coerce_row(row) for row in rows]

    async def healthcheck_async(self) -> dict[str, Any]:
        client = self._get_async_client()
        response = await client.get(
            f"{self.endpoint}/health",
            timeout=self.health_timeout,
        )
        response.raise_for_status()
        try:
            return response.json()
        except Exception:
            return {"status": response.text}


__all__ = ["VllmFactoryNLIProvider"]
