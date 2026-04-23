"""HTTP-backed NLI provider for a live vLLM pooling endpoint."""

from __future__ import annotations

import threading
from typing import Any, Sequence


class VllmFactoryNLIProvider:
    """Thin HTTP client for the latence-trace NLI vLLM plugin."""

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

    def close(self) -> None:
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


__all__ = ["VllmFactoryNLIProvider"]
