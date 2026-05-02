"""HTTP provider for vLLM Factory DeBERTa GLiNER PII detection."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Sequence
from typing import Any


class VllmFactoryDebertaGlinerProvider:
    """Thin client for the ``deberta_gliner_io`` vLLM Factory endpoint."""

    def __init__(
        self,
        endpoint: str,
        model: str,
        *,
        timeout: float = 60.0,
        health_timeout: float = 10.0,
        max_concurrency: int = 16,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.timeout = float(timeout)
        self.health_timeout = float(health_timeout)
        self.max_concurrency = max(1, int(max_concurrency))
        self._http_client: Any = None
        self._client_lock = threading.Lock()
        self._async_client: Any = None
        self._async_client_loop: asyncio.AbstractEventLoop | None = None
        self._async_client_lock = threading.Lock()

    def _limits(self):
        import httpx

        return httpx.Limits(
            max_connections=max(self.max_concurrency * 2, 8),
            max_keepalive_connections=max(self.max_concurrency, 4),
        )

    def _get_http_client(self):
        if self._http_client is not None:
            return self._http_client
        import httpx

        with self._client_lock:
            if self._http_client is not None:
                return self._http_client
            self._http_client = httpx.Client(
                base_url=self.endpoint,
                timeout=self.timeout,
                limits=self._limits(),
            )
            return self._http_client

    def _get_async_client(self):
        import httpx

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        with self._async_client_lock:
            if self._async_client is not None and self._async_client_loop is loop:
                return self._async_client
            self._async_client = httpx.AsyncClient(
                base_url=self.endpoint,
                timeout=self.timeout,
                limits=self._limits(),
            )
            self._async_client_loop = loop
            return self._async_client

    def close(self) -> None:
        with self._client_lock:
            if self._http_client is not None:
                try:
                    self._http_client.close()
                finally:
                    self._http_client = None
        with self._async_client_lock:
            async_client = self._async_client
            self._async_client = None
            self._async_client_loop = None
        if async_client is not None:
            try:
                coro = async_client.aclose()
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = None
                if loop is not None and loop.is_running():
                    asyncio.ensure_future(coro, loop=loop)
                else:
                    asyncio.run(coro)
            except Exception:
                pass

    def healthcheck(self) -> dict[str, Any]:
        response = self._get_http_client().get("/health", timeout=self.health_timeout)
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

    def build_payload(
        self,
        *,
        text: str,
        labels: Sequence[str],
        threshold: float,
        flat_ner: bool,
        multi_label: bool,
    ) -> dict[str, Any]:
        return {
            "model": self.model,
            "task": "plugin",
            "data": {
                "text": text,
                "labels": list(labels),
                "threshold": float(threshold),
                "flat_ner": bool(flat_ner),
                "multi_label": bool(multi_label),
            },
        }

    @staticmethod
    def _coerce_entities(raw: Any) -> list[dict[str, Any]]:
        data = VllmFactoryDebertaGlinerProvider._unwrap_data(raw)
        if data is None:
            return []
        if isinstance(data, dict) and "entities" in data:
            data = data["entities"]
        if not isinstance(data, list):
            raise TypeError(f"Unsupported GLiNER response payload: {type(data)!r}")
        return [dict(item) for item in data if isinstance(item, dict)]

    def detect(
        self,
        *,
        text: str,
        labels: Sequence[str],
        threshold: float = 0.5,
        flat_ner: bool = True,
        multi_label: bool = False,
    ) -> list[dict[str, Any]]:
        response = self._get_http_client().post(
            "/pooling",
            json=self.build_payload(
                text=text,
                labels=labels,
                threshold=threshold,
                flat_ner=flat_ner,
                multi_label=multi_label,
            ),
        )
        response.raise_for_status()
        return self._coerce_entities(response.json())

    async def detect_async(
        self,
        *,
        text: str,
        labels: Sequence[str],
        threshold: float = 0.5,
        flat_ner: bool = True,
        multi_label: bool = False,
    ) -> list[dict[str, Any]]:
        response = await self._get_async_client().post(
            "/pooling",
            json=self.build_payload(
                text=text,
                labels=labels,
                threshold=threshold,
                flat_ner=flat_ner,
                multi_label=multi_label,
            ),
        )
        response.raise_for_status()
        return self._coerce_entities(response.json())
