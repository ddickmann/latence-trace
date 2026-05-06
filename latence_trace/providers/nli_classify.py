"""HTTP-backed NLI provider for vLLM-native ``--convert classify`` servers.

This is the wire shape that fits ``MoritzLaurer/bge-m3-zeroshot-v2.0``
served by vLLM 0.19.x with ``--runner pooling --convert classify`` (the
0.19 spelling of the legacy ``--task classify``; the latter was removed
in 0.19). vLLM exposes the head under ``/classify`` — the
OpenAI-compatible ``/v1/classify`` alias only landed in 0.20+. We hit
``/classify`` directly so the client matches what production actually
serves:

    POST /classify
    { "model": "...", "input": ["text1", "text2", ...] }

For NLI we feed the concatenated ``"<premise></s></s><hypothesis>"``
form — the same form the in-process
:class:`BgeM3ZeroShotNLIProvider` uses, so the model sees exactly the
input it was fine-tuned on.

The provider implements :class:`latence_trace.core.nli.NLIProvider` so
the registry can swap it in for the in-process transformers fallback by
flipping ``LATENCE_TRACE_NLI_MULTI_ENDPOINT`` once the vLLM-served
multilingual server is online. Until then the registry routes requests
to :class:`BgeM3ZeroShotNLIProvider` and the system stays SOTA without
any vLLM plugin work.

The MiniCheck T5 path is *not* served by this provider — the encoder-
decoder runs through the BYOP plugin
(``runpod/vllm_plugins/minicheck_t5/``, future work) which keeps the
``/pooling`` wire shape compatible with
:class:`latence_trace.providers.nli.VllmFactoryNLIProvider`. So callers
choose:

* ``VllmFactoryNLIProvider`` — for any plugin that already returns
  ``{entail, neutral, contradict}`` triples server-side (current
  ``nli_mdeberta``, future ``minicheck_t5``).
* ``VllmClassifyNLIProvider`` — for vLLM-native ``/classify``
  servers where the client owns the binary→triple label map (bge-m3-zs).
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# bge-m3-zeroshot-v2.0 fine-tunes the first XLM-R label for ``entailment``
# and the second for ``not_entailment``. We default to that ordering and
# refresh from the live model card on the first /classify response when
# the server returns label names.
_DEFAULT_ENTAIL_INDEX = 0
_DEFAULT_NOT_ENTAIL_INDEX = 1


def _xlm_r_pair(premise: str, hypothesis: str) -> str:
    """XLM-R NLI input: ``premise </s></s> hypothesis`` (single string).

    The HuggingFace tokenizer for XLM-R produces this exact pair format
    when called with ``(text, text_pair)``. vLLM's ``/classify``
    expects a flat string per row, so we reproduce the wire form
    ourselves. Keeping the ``</s></s>`` literal is important — XLM-R was
    pre-trained with explicit segment separators and zero-shot models
    fine-tuned on top of it (bge-m3-zeroshot, ynie/anli) all use the
    same format.
    """

    return f"{premise} </s></s> {hypothesis}"


class VllmClassifyNLIProvider:
    """Thin HTTP client for a vLLM-native ``/classify`` NLI endpoint."""

    def __init__(
        self,
        endpoint: str,
        model: str,
        *,
        timeout: float = 60.0,
        health_timeout: float = 10.0,
        max_concurrency: int = 8,
        entail_index: int = _DEFAULT_ENTAIL_INDEX,
        not_entail_index: int = _DEFAULT_NOT_ENTAIL_INDEX,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.timeout = float(timeout)
        self.health_timeout = float(health_timeout)
        self.max_concurrency = max(1, int(max_concurrency))
        self.entail_index = int(entail_index)
        self.not_entail_index = int(not_entail_index)
        self._http_client: Any = None
        self._client_lock = threading.Lock()
        self._async_client: Any = None
        self._async_client_loop: Optional[asyncio.AbstractEventLoop] = None
        self._async_client_lock = threading.Lock()
        # Set on first response when the server tells us the actual
        # label ordering; we trust the wire over the env var because the
        # server is the single source of truth.
        self._labels_resolved = False

    def _get_http_client(self) -> Any:
        if self._http_client is not None:
            return self._http_client
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "httpx is required for VllmClassifyNLIProvider."
            ) from exc

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

    def _get_async_client(self) -> Any:
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "httpx is required for VllmClassifyNLIProvider."
            ) from exc

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:  # pragma: no cover - sync caller
            loop = None

        with self._async_client_lock:
            if (
                self._async_client is not None
                and self._async_client_loop is loop
            ):
                return self._async_client
            limits = httpx.Limits(
                max_connections=max(self.max_concurrency * 2, 8),
                max_keepalive_connections=max(self.max_concurrency, 4),
            )
            self._async_client = httpx.AsyncClient(
                base_url=self.endpoint,
                timeout=self.timeout,
                limits=limits,
            )
            self._async_client_loop = loop
            return self._async_client

    def healthcheck(self) -> dict[str, Any]:
        client = self._get_http_client()
        response = client.get("/health", timeout=self.health_timeout)
        response.raise_for_status()
        try:
            return response.json()
        except Exception:
            return {"status": response.text}

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
            except Exception:  # pragma: no cover
                pass

    def _maybe_refresh_labels(self, payload: Any) -> None:
        """Update entail/not-entail indices from a response if labels are present."""

        if self._labels_resolved or not isinstance(payload, dict):
            return
        data = payload.get("data")
        if not isinstance(data, list) or not data:
            return
        first = data[0]
        if not isinstance(first, dict):
            return
        scores = first.get("scores") or first.get("logits") or first.get("data")
        # Server may emit either ``[{"label": "entailment", "score": 0.7}, ...]``
        # or aligned arrays; we only refresh when explicit labels are seen.
        if not isinstance(scores, list):
            return
        labelled = [s for s in scores if isinstance(s, dict) and "label" in s]
        if not labelled:
            return
        entail_idx = -1
        notentail_idx = -1
        for idx, item in enumerate(labelled):
            label = str(item["label"]).lower().replace("-", "_")
            if "not_entail" in label or label == "contradiction":
                notentail_idx = idx
            elif "entail" in label:
                entail_idx = idx
        if entail_idx >= 0 and notentail_idx >= 0:
            self.entail_index = entail_idx
            self.not_entail_index = notentail_idx
        self._labels_resolved = True

    @staticmethod
    def _row_probs(row: Any) -> List[float]:
        """Coerce one /classify row to a probability list.

        Supports the three shapes vLLM ships across versions:
        * ``[float, float]`` — already-softmaxed probabilities.
        * ``{"data": [float, float]}`` — common pooling envelope.
        * ``[{"label": "entailment", "score": 0.7}, ...]`` — labelled.
        """

        if isinstance(row, dict):
            scores = row.get("scores") or row.get("data") or row.get("probs")
            if scores is None:
                return [float(row.get("score", 0.0))]
            return VllmClassifyNLIProvider._row_probs(scores)
        if isinstance(row, list):
            if not row:
                return []
            if isinstance(row[0], dict):
                return [float(item.get("score", 0.0)) for item in row]
            return [float(value) for value in row]
        if isinstance(row, (int, float)):
            return [float(row)]
        raise TypeError(f"Unsupported /classify row shape: {type(row)!r}")

    def _to_triple(self, probs: List[float]) -> Tuple[float, float, float]:
        if not probs:
            return (0.0, 1.0, 0.0)
        if len(probs) >= 3:
            # Server already returns a 3-class head — pass through.
            return (float(probs[0]), float(probs[1]), float(probs[2]))
        entail = (
            float(probs[self.entail_index])
            if self.entail_index < len(probs)
            else float(probs[0])
        )
        return (entail, float(1.0 - entail), 0.0)

    def entail(
        self,
        premises: Sequence[str],
        hypotheses: Sequence[str],
    ) -> List[Tuple[float, float, float]]:
        if len(premises) != len(hypotheses):
            raise ValueError("premises and hypotheses must align")
        if not premises:
            return []

        client = self._get_http_client()
        inputs = [_xlm_r_pair(p or "", h or "") for p, h in zip(premises, hypotheses)]
        response = client.post(
            "/classify",
            json={"model": self.model, "input": inputs},
        )
        response.raise_for_status()
        payload = response.json()
        self._maybe_refresh_labels(payload)
        rows = payload.get("data") if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise TypeError(
                f"Unsupported /classify payload: {type(rows)!r}"
            )
        return [
            self._to_triple(self._row_probs(row))
            for row in rows
        ]

    async def entail_async(
        self,
        premises: Sequence[str],
        hypotheses: Sequence[str],
    ) -> List[Tuple[float, float, float]]:
        if len(premises) != len(hypotheses):
            raise ValueError("premises and hypotheses must align")
        if not premises:
            return []

        client = self._get_async_client()
        inputs = [_xlm_r_pair(p or "", h or "") for p, h in zip(premises, hypotheses)]
        response = await client.post(
            "/classify",
            json={"model": self.model, "input": inputs},
        )
        response.raise_for_status()
        payload = response.json()
        self._maybe_refresh_labels(payload)
        rows = payload.get("data") if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise TypeError(
                f"Unsupported /classify payload: {type(rows)!r}"
            )
        return [
            self._to_triple(self._row_probs(row))
            for row in rows
        ]


__all__ = ["VllmClassifyNLIProvider"]
