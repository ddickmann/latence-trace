"""HTTP-backed premise reranker for a vLLM-served cross-encoder.

The wire shape matches vLLM's OpenAI-compatible ``/v1/score`` endpoint
(``vllm serve <model> --task score`` or ``--task classify`` for
single-label cross-encoders):

    POST /v1/score
    { "model": "...", "text_1": ["query"], "text_2": ["doc1", "doc2", ...] }

For ``BAAI/bge-reranker-v2-m3`` (single-logit relevance head) the
response carries one float per ``text_2`` entry. We expose
:meth:`VllmRerankerProvider.score_pairs` so the call site in
:func:`latence_trace.core.nli.verify_claims` (which already pre-batches
every ``(atom, premise)`` pair across the request into one
``score_pairs`` call) can hit the reranker over HTTP without any
behaviour change.

Selection between this provider and the in-process
:class:`latence_trace.core.nli.CrossEncoderPremiseReranker` lives in
:func:`latence_trace.providers.reranker.resolve_reranker`. Set
``LATENCE_TRACE_RERANKER_ENDPOINT`` to opt in to the vLLM lane; leave
it empty to keep the transformers-direct path. This mirrors the NLI
endpoint convention.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from typing import Any, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


class VllmRerankerProvider:
    """Thin HTTP client for a vLLM-served bge-reranker-v2-m3 endpoint.

    The provider implements the
    :class:`latence_trace.core.nli.PremiseReranker` Protocol via
    :meth:`score_pairs` so the cross-claim batched hot path in
    ``verify_claims`` can drop it in for the in-process cross-encoder
    by setting ``LATENCE_TRACE_RERANKER_ENDPOINT``.
    """

    def __init__(
        self,
        endpoint: str,
        model: str,
        *,
        timeout: float = 60.0,
        health_timeout: float = 10.0,
        max_concurrency: int = 8,
        # vLLM's /v1/score expects a single text_1 per row when text_2
        # is a list. We convert flat ``(claim, premise)`` pairs into
        # one POST per claim so each claim's premises ride a single
        # HTTP round trip; the server batches them in one engine step.
        # When the upstream (verify_claims) has already pre-batched
        # across atoms we still group by claim text so the (1, N)
        # POST shape matches what vLLM optimises.
        max_pairs_per_post: int = 128,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.timeout = float(timeout)
        self.health_timeout = float(health_timeout)
        self.max_concurrency = max(1, int(max_concurrency))
        self.max_pairs_per_post = max(1, int(max_pairs_per_post))
        self._http_client: Any = None
        self._client_lock = threading.Lock()
        # Async transport is process-shared (see
        # :func:`latence_trace.providers.get_shared_async_client`). No
        # per-provider AsyncClient state lives here; the legacy sync
        # ``httpx.Client`` above is still per-provider for back-compat.

    def _get_http_client(self) -> Any:
        if self._http_client is not None:
            return self._http_client
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - optional install
            raise ImportError(
                "httpx is required for VllmRerankerProvider."
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
        """Return the process-shared async transport.

        See :func:`latence_trace.providers.get_shared_async_client` —
        the same client is reused by both NLI providers and this
        reranker, so concurrent verify_claims fan-out and reranker
        scoring share a single TCP keepalive pool instead of fighting
        for sockets across three independent pools.
        """

        from latence_trace.providers import get_shared_async_client

        return get_shared_async_client()

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
        # Shared async client is owned by the providers package and
        # closed via :func:`latence_trace.providers.close_shared_async_client`.

    async def aclose(self) -> None:
        """Async-aware close; safe to call from an asyncio coroutine."""

        with self._client_lock:
            if self._http_client is not None:
                try:
                    self._http_client.close()
                finally:
                    self._http_client = None

    @staticmethod
    def _coerce_score(item: Any) -> float:
        if isinstance(item, dict):
            for key in ("score", "relevance_score", "value"):
                if key in item:
                    return float(item[key])
            raise TypeError(
                f"Unsupported /v1/score row dict shape: {sorted(item.keys())!r}"
            )
        return float(item)

    def _post_one(
        self,
        text_1: str,
        text_2_list: Sequence[str],
    ) -> List[float]:
        client = self._get_http_client()
        response = client.post(
            "/v1/score",
            json={
                "model": self.model,
                "text_1": text_1,
                "text_2": list(text_2_list),
            },
        )
        response.raise_for_status()
        payload = response.json()
        rows = payload.get("data") if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise TypeError(
                f"Unsupported /v1/score payload: {type(rows)!r}"
            )
        return [self._coerce_score(row) for row in rows]

    async def _post_one_async(
        self,
        text_1: str,
        text_2_list: Sequence[str],
    ) -> List[float]:
        client = self._get_async_client()
        # Shared client carries no base_url; pass the absolute URL.
        response = await client.post(
            f"{self.endpoint}/v1/score",
            json={
                "model": self.model,
                "text_1": text_1,
                "text_2": list(text_2_list),
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        rows = payload.get("data") if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise TypeError(
                f"Unsupported /v1/score payload: {type(rows)!r}"
            )
        return [self._coerce_score(row) for row in rows]

    def score(self, claim: str, candidate_premises: Sequence[str]) -> List[float]:
        if not candidate_premises:
            return []
        # Honour ``max_pairs_per_post`` so a 500-premise request doesn't
        # blow past the server's ``--max-num-batched-tokens`` budget.
        out: List[float] = []
        premises = list(candidate_premises)
        for start in range(0, len(premises), self.max_pairs_per_post):
            chunk = premises[start : start + self.max_pairs_per_post]
            out.extend(self._post_one(claim, chunk))
        return out

    async def score_async(
        self, claim: str, candidate_premises: Sequence[str]
    ) -> List[float]:
        if not candidate_premises:
            return []
        premises = list(candidate_premises)
        # Within a single claim we still chunk by ``max_pairs_per_post``
        # to respect the server's ``--max-num-batched-tokens`` budget,
        # but we fan the chunks out concurrently so even a 500-premise
        # claim doesn't serialize on the wire.
        chunks = [
            premises[start : start + self.max_pairs_per_post]
            for start in range(0, len(premises), self.max_pairs_per_post)
        ]
        if len(chunks) == 1:
            return await self._post_one_async(claim, chunks[0])
        chunk_results = await asyncio.gather(
            *(self._post_one_async(claim, chunk) for chunk in chunks)
        )
        out: List[float] = []
        for result in chunk_results:
            out.extend(result)
        return out

    def score_pairs(
        self, pairs: Sequence[Tuple[str, str]]
    ) -> List[float]:
        """Sync entry point; concurrent under the hood via :meth:`score_pairs_async`.

        We keep this method as the protocol contract for in-process
        callers (``verify_claims``'s sync hot path) but route the work
        through :meth:`score_pairs_async` so the per-distinct-claim
        ``/v1/score`` POSTs fan out concurrently instead of looping
        sequentially. On a typical multi-atom request that turns N×
        round-trip latency into 1× round-trip + server-side coalescing
        (vLLM's continuous batching merges concurrent in-flight
        requests into the same engine step automatically).

        Result order is bit-identical to the previous sequential
        implementation — the per-pair index alignment is preserved
        because we still walk the original ``pairs`` order to fill the
        flat ``scores`` list at known positions.
        """

        if not pairs:
            return []
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # No active loop → safe to run our own.
            return asyncio.run(self.score_pairs_async(pairs))
        # If a caller invokes the sync API from an active event loop
        # (rare; mostly shows up in ad-hoc REPL sessions) we fall back
        # to the sequential code path so we don't deadlock the loop.
        # The async-native callers route through :meth:`score_pairs_async`
        # directly and never hit this branch.
        return self._score_pairs_sync_fallback(pairs)

    async def score_pairs_async(
        self, pairs: Sequence[Tuple[str, str]]
    ) -> List[float]:
        """Async fan-out variant: one POST per distinct claim, concurrent.

        Behaviour matches the legacy sync path bit-for-bit: same
        per-claim grouping, same soft-fail-on-error, same count-
        mismatch padding. Only the dispatch is concurrent.
        """

        if not pairs:
            return []

        positions_by_claim, premises_by_claim, order = self._group_pairs(pairs)

        async def _score_one_claim(
            claim_key: str,
        ) -> Tuple[str, Optional[List[float]], Optional[Exception]]:
            try:
                return claim_key, await self.score_async(
                    claim_key, premises_by_claim[claim_key]
                ), None
            except Exception as exc:  # noqa: BLE001
                return claim_key, None, exc

        results = await asyncio.gather(
            *(_score_one_claim(claim_key) for claim_key in order)
        )
        return self._stitch_results(pairs, positions_by_claim, results)

    def _score_pairs_sync_fallback(
        self, pairs: Sequence[Tuple[str, str]]
    ) -> List[float]:
        """Sequential implementation kept as a defensive fallback.

        Hit only when ``score_pairs`` is invoked from an already-running
        event loop. Production calls go through ``score_pairs_async``.
        """

        positions_by_claim, premises_by_claim, order = self._group_pairs(pairs)
        results: List[Tuple[str, Optional[List[float]], Optional[Exception]]] = []
        for claim_key in order:
            try:
                results.append((claim_key, self.score(claim_key, premises_by_claim[claim_key]), None))
            except Exception as exc:  # noqa: BLE001
                results.append((claim_key, None, exc))
        return self._stitch_results(pairs, positions_by_claim, results)

    @staticmethod
    def _group_pairs(
        pairs: Sequence[Tuple[str, str]],
    ) -> Tuple[dict[str, List[int]], dict[str, List[str]], List[str]]:
        positions_by_claim: dict[str, List[int]] = {}
        premises_by_claim: dict[str, List[str]] = {}
        order: List[str] = []
        for idx, (claim, premise) in enumerate(pairs):
            key = claim or ""
            if key not in positions_by_claim:
                positions_by_claim[key] = []
                premises_by_claim[key] = []
                order.append(key)
            positions_by_claim[key].append(idx)
            premises_by_claim[key].append(premise or "")
        return positions_by_claim, premises_by_claim, order

    def _stitch_results(
        self,
        pairs: Sequence[Tuple[str, str]],
        positions_by_claim: dict[str, List[int]],
        results: Sequence[Tuple[str, Optional[List[float]], Optional[Exception]]],
    ) -> List[float]:
        scores: List[Optional[float]] = [None] * len(pairs)
        for claim_key, claim_scores, exc in results:
            positions = positions_by_claim[claim_key]
            if exc is not None:
                logger.warning(
                    "vllm_reranker_score_failed",
                    extra={"claim": claim_key[:80], "error": str(exc)},
                )
                # Soft-fail per claim so the upstream pipeline falls
                # back to lexical for this claim only — same shape as
                # the in-process reranker's failure mode.
                for pos in positions:
                    scores[pos] = 0.0
                continue
            assert claim_scores is not None
            if len(claim_scores) != len(positions):
                logger.warning(
                    "vllm_reranker_score_count_mismatch",
                    extra={
                        "expected": len(positions),
                        "received": len(claim_scores),
                        "claim": claim_key[:80],
                    },
                )
                # Pad / truncate so the caller still gets one score per pair.
                claim_scores = list(claim_scores)[: len(positions)]
                while len(claim_scores) < len(positions):
                    claim_scores.append(0.0)
            for pos, value in zip(positions, claim_scores):
                scores[pos] = float(value)
        return [float(value) if value is not None else 0.0 for value in scores]


def resolve_reranker(
    *,
    fallback: Any = None,
) -> Any:
    """Pick the right premise reranker based on environment configuration.

    Priority:

    1. ``LATENCE_TRACE_RERANKER_ENDPOINT`` set → :class:`VllmRerankerProvider`.
       The healthcheck runs once at construction; on failure we fall
       through to the fallback so an unhealthy vLLM does not silently
       degrade groundedness.
    2. ``fallback`` (typically the in-process
       :class:`latence_trace.core.nli.CrossEncoderPremiseReranker`
       returned from
       :func:`latence_trace.core.nli.resolve_default_reranker`).
    3. ``None`` — premise selection falls back to lexical overlap, the
       same behaviour as before any reranker was wired.
    """

    endpoint = os.environ.get("LATENCE_TRACE_RERANKER_ENDPOINT", "").strip()
    if endpoint:
        model = os.environ.get(
            "LATENCE_TRACE_RERANKER_MODEL",
            "BAAI/bge-reranker-v2-m3",
        )
        try:
            provider = VllmRerankerProvider(
                endpoint=endpoint,
                model=model,
                timeout=_env_float("LATENCE_TRACE_RERANKER_TIMEOUT", 60.0),
                health_timeout=_env_float(
                    "LATENCE_TRACE_RERANKER_HEALTH_TIMEOUT", 10.0
                ),
                max_concurrency=_env_int(
                    "LATENCE_TRACE_RERANKER_MAX_CONCURRENCY", 8
                ),
                max_pairs_per_post=_env_int(
                    "LATENCE_TRACE_RERANKER_MAX_PAIRS_PER_POST", 128
                ),
            )
            provider.healthcheck()
            return provider
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "vllm_reranker_init_failed",
                extra={
                    "endpoint": endpoint,
                    "model": model,
                    "error": str(exc),
                },
            )
    return fallback


__all__ = ["VllmRerankerProvider", "resolve_reranker"]
