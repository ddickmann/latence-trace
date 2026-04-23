"""Encoder providers for the latence-trace Groundedness Tracker.

This module ports the two providers the standalone product needs without
keeping a runtime dependency on the upstream voyager-index package:

- :class:`VllmFactoryModernColBERTProvider` - thin HTTP client for the
  vllm-factory ``/pooling`` endpoint serving a ModernColBERT model. This is
  the production deployment path for high-throughput groundedness scoring.
- :func:`load_pylate_colbert` - small helper that lazily imports pylate and
  returns a local ``models.ColBERT`` instance for offline / dev runs.
"""

from __future__ import annotations

import base64
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

import numpy as np


class VllmFactoryModernColBERTProvider:
    """ModernColBERT-specific HTTP client for vllm-factory ``/pooling``.

    Implements the minimal encode/tokenize surface the groundedness scorer
    expects and stays compatible with the upstream ``moderncolbert_io``
    plugin contract used by latence.ai's vllm-factory deployments.
    """

    query_prefix_id = 50368
    document_prefix_id = 50369

    def __init__(
        self,
        endpoint: str,
        model: str,
        *,
        timeout: float = 60.0,
        health_timeout: float = 10.0,
        batch_size: int = 16,
        max_concurrency: int = 8,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.model_name = model
        self.model_name_or_path = model
        self.timeout = float(timeout)
        self.health_timeout = float(health_timeout)
        self.batch_size = max(1, int(batch_size))
        self.max_concurrency = max(1, int(max_concurrency))
        self._http_client: Any = None
        self._client_lock = threading.Lock()
        self._executor: Optional[ThreadPoolExecutor] = (
            ThreadPoolExecutor(
                max_workers=self.max_concurrency,
                thread_name_prefix="latence-trace-encoder",
            )
            if self.max_concurrency > 1
            else None
        )

        try:
            from transformers import AutoConfig, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional install shape
            raise ImportError(
                "transformers is required for VllmFactoryModernColBERTProvider."
            ) from exc

        self.tokenizer = AutoTokenizer.from_pretrained(
            model,
            use_fast=True,
            trust_remote_code=True,
        )
        config = None
        try:
            config = AutoConfig.from_pretrained(model, trust_remote_code=True)
        except Exception:
            config = None
        self.colbert_dim = int(getattr(config, "colbert_dim", getattr(config, "dim", 128)) if config else 128)
        self.query_maxlen = int(getattr(config, "query_length", getattr(config, "query_maxlen", 256)) if config else 256)
        self.doc_maxlen = int(
            getattr(
                config,
                "document_length",
                getattr(config, "document_maxlen", getattr(config, "max_position_embeddings", 8192)),
            )
            if config
            else 8192
        )

    def _get_http_client(self):
        if self._http_client is not None:
            return self._http_client
        try:
            import httpx
        except ImportError as exc:
            raise ImportError(
                "httpx is required for VllmFactoryModernColBERTProvider."
            ) from exc

        with self._client_lock:
            if self._http_client is not None:
                return self._http_client
            limits = httpx.Limits(
                max_connections=max(self.max_concurrency * 2, 8),
                max_keepalive_connections=max(self.max_concurrency, 4),
            )
            self._http_client = httpx.Client(base_url=self.endpoint, timeout=self.timeout, limits=limits)
        return self._http_client

    def close(self) -> None:
        # Serialize close() so two concurrent shutdowns don't both call
        # close() on a half-released httpx client.
        with self._client_lock:
            if self._http_client is not None:
                try:
                    self._http_client.close()
                finally:
                    self._http_client = None
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None

    def healthcheck(self) -> dict[str, Any]:
        client = self._get_http_client()
        response = client.get("/health", timeout=self.health_timeout)
        response.raise_for_status()
        try:
            return response.json()
        except Exception:
            return {"status": response.text}

    def _max_length(self, *, is_query: bool) -> int:
        return self.query_maxlen if is_query else self.doc_maxlen

    def _token_ids(self, text: str, *, is_query: bool) -> list[int]:
        encoded = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=max(1, self._max_length(is_query=is_query) - 1),
            padding=False,
            return_tensors=None,
        )
        input_ids = encoded["input_ids"]
        if input_ids and isinstance(input_ids[0], list):
            input_ids = input_ids[0]
        input_ids = list(input_ids)
        if not input_ids:
            return []
        prefix_id = self.query_prefix_id if is_query else self.document_prefix_id
        return [int(input_ids[0]), int(prefix_id), *[int(item) for item in input_ids[1:]]]

    def tokenize(self, text: str, *, is_query: bool = False) -> list[str]:
        input_ids = self._token_ids(text, is_query=is_query)
        if not input_ids:
            return []
        try:
            tokens = list(self.tokenizer.convert_ids_to_tokens(input_ids))
        except Exception:
            tokens = [str(item) for item in input_ids]
        normalized: list[str] = []
        for token_id, token in zip(input_ids, tokens):
            if token_id == self.query_prefix_id and (token is None or str(token) == str(token_id)):
                normalized.append("[Q]")
            elif token_id == self.document_prefix_id and (token is None or str(token) == str(token_id)):
                normalized.append("[D]")
            else:
                normalized.append(str(token))
        return normalized

    def encoded_token_count(self, text: str, *, is_query: bool = False) -> int:
        """Return the exact token id sequence length the backend will see.

        Includes special tokens and the ``[D]/[Q]`` prefix so the
        groundedness packer can budget windows that match what the encoder
        actually consumes.
        """

        return len(self._token_ids(text, is_query=is_query))

    def build_payload(self, text: Any, *, is_query: Any) -> dict[str, Any]:
        return {
            "model": self.model,
            "data": {
                "text": text,
                "is_query": is_query,
            },
            "task": "token_embed",
        }

    @staticmethod
    def _unwrap_data(payload: Any) -> Any:
        current = payload
        while isinstance(current, dict) and "data" in current:
            current = current["data"]
        return current

    def _decode_embedding(self, payload: Any) -> np.ndarray:
        data = self._unwrap_data(payload)
        if isinstance(data, dict):
            for key in ("data", "embedding", "embeddings", "output"):
                if key in data:
                    data = data[key]
                    break

        if isinstance(data, str):
            raw = np.frombuffer(base64.b64decode(data.encode("ascii")), dtype=np.float32)
            if raw.size == 0:
                return np.zeros((0, self.colbert_dim), dtype=np.float32)
            if raw.size % self.colbert_dim != 0:
                raise ValueError(
                    f"ModernColBERT response length {raw.size} is not divisible by colbert_dim={self.colbert_dim}"
                )
            return raw.reshape(-1, self.colbert_dim)

        array = np.asarray(data, dtype=np.float32)
        if array.ndim == 1:
            if array.size == 0:
                return np.zeros((0, self.colbert_dim), dtype=np.float32)
            if array.size % self.colbert_dim != 0:
                raise ValueError(
                    "ModernColBERT /pooling response length {size} is not divisible by colbert_dim={dim}; "
                    "the server did not return a multi-vector matrix. Check the model id and IO processor wiring.".format(
                        size=array.size,
                        dim=self.colbert_dim,
                    )
                )
            return array.reshape(-1, self.colbert_dim)
        if array.ndim == 2:
            if array.shape[1] != self.colbert_dim:
                raise ValueError(
                    "ModernColBERT /pooling response inner dim {got} does not match colbert_dim={expected}".format(
                        got=int(array.shape[1]),
                        expected=self.colbert_dim,
                    )
                )
            return array
        raise TypeError("Unsupported ModernColBERT /pooling payload shape")

    def _pool_texts(self, texts: list[str], *, is_query: bool) -> list[np.ndarray]:
        client = self._get_http_client()
        payload = self.build_payload(
            texts[0] if len(texts) == 1 else texts,
            is_query=bool(is_query) if len(texts) == 1 else [bool(is_query)] * len(texts),
        )
        response = client.post("/pooling", json=payload)
        response.raise_for_status()
        body = response.json()
        raw = self._unwrap_data(body)
        if len(texts) == 1:
            return [self._decode_embedding(raw)]
        if not isinstance(raw, list):
            raise TypeError(f"Unsupported batched ModernColBERT payload: {type(raw)!r}")
        return [self._decode_embedding(item) for item in raw]

    def encode(self, inputs: Any, **kwargs: Any) -> list[np.ndarray]:
        texts = [inputs] if isinstance(inputs, str) else list(inputs)
        if not texts:
            return []
        is_query = bool(kwargs.get("is_query", False))
        outputs: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            outputs.extend(self._pool_texts(texts[start : start + self.batch_size], is_query=is_query))
        return outputs


def load_pylate_colbert(
    model_name_or_path: str,
    *,
    device: str = "cpu",
    do_query_expansion: bool = False,
    **extra: Any,
) -> Any:
    """Lazily import pylate and instantiate a local ColBERT encoder.

    Used by the default :class:`GroundednessService` encoder factory when no
    vllm-factory endpoint is configured. Kept as a thin helper so callers
    can swap in their own loader (e.g. with custom precision settings).
    """

    try:
        from pylate import models
    except ImportError as exc:
        raise ImportError(
            "pylate is required for the local ColBERT encoder. "
            "Install pylate or configure VOYAGER_GROUNDEDNESS_VLLM_ENDPOINT."
        ) from exc
    return models.ColBERT(
        model_name_or_path=model_name_or_path,
        device=device,
        do_query_expansion=do_query_expansion,
        **extra,
    )


__all__ = [
    "VllmFactoryModernColBERTProvider",
    "load_pylate_colbert",
]
