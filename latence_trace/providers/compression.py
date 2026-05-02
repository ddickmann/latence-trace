"""HTTP client for the TRACE/SuperPod LLMLingua2 vLLM service."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx


class VllmCompressionProvider:
    """Async client for the LLMLingua2 token-classification vLLM server."""

    def __init__(
        self,
        *,
        endpoint: str,
        model: str | None = None,
        timeout: float = 30.0,
        max_concurrency: int = 16,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.timeout = float(timeout)
        self._semaphore = asyncio.Semaphore(max(1, int(max_concurrency)))

    async def keep_probabilities(self, text: str) -> list[float] | None:
        if not text.strip():
            return []
        payload: dict[str, Any] = {
            "input": text,
            "task": "token_classify",
        }
        if self.model:
            payload["model"] = self.model
        async with self._semaphore, httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(f"{self.endpoint}/pooling", json=payload)
            response.raise_for_status()
            body = response.json()
        data_items = body.get("data") or []
        if not data_items:
            return None
        raw = data_items[0].get("data") if isinstance(data_items[0], dict) else data_items[0]
        if not isinstance(raw, list):
            return None
        if raw and isinstance(raw[0], list):
            return [float(row[1]) for row in raw if isinstance(row, list) and len(row) >= 2]
        return [float(value) for value in raw]
