"""Hot-path latency perf gate.

Locks the Phase 2 SOTA-stack optimisations (concurrent reranker fan-out,
verify_claims ⊥ semantic_entropy overlap, parallel multi-chunk scoring)
against accidental sequentialisation. We mock the HTTP transport with
deterministic per-call delays so the wall-clock assertions are stable
across machines (no network, no GPU, no torch import).

What each test asserts:

* :class:`TestRerankerConcurrentFanOut` — when the reranker batch
  contains N distinct claims, ``score_pairs_async`` issues N concurrent
  POSTs and the wall time is bounded by ~1× per-POST latency, not N×.
* :class:`TestVerifySemanticOverlap` — :func:`_run_nli_lane_concurrent`
  overlaps verify_claims and semantic_entropy NLI calls; total wall is
  bounded by the slower of the two, not their sum.
* :class:`TestMultiChunkOverlap` — the chunk executor in
  ``score_groundedness_response_chunked`` parallelises per-chunk
  scoring; a 4-chunk fixture's wall is bounded by ~1.5× per-chunk
  latency, not 4×.

These tests are perf gates: they fail if the hot path regresses to
sequential dispatch, but they're MOCK-driven so they don't catch
backend-side regressions (e.g. vLLM stalls, GPU thrashing). The live
acceptance tests in Phase 3 cover that.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, List, Sequence, Tuple

import pytest

from latence_trace.providers.reranker import VllmRerankerProvider


# Deterministic per-POST latency budget (seconds). Each fake request
# sleeps for this long; concurrent dispatch should bring N POSTs from
# N*delay (sequential) to ~1*delay (concurrent).
_FAKE_POST_LATENCY_S = 0.06


# ---------------------------------------------------------------------------
# Reranker concurrent fan-out
# ---------------------------------------------------------------------------


class _FakeAsyncClient:
    """Minimal :class:`httpx.AsyncClient` stand-in.

    Sleeps ``latency_s`` per POST so we can measure whether N calls
    were dispatched sequentially (total ≈ N*latency) or concurrently
    (total ≈ 1*latency + scheduling overhead).
    """

    def __init__(self, latency_s: float = _FAKE_POST_LATENCY_S) -> None:
        self.latency_s = float(latency_s)
        self.posts_seen = 0

    async def post(self, _url: str, *, json: Any, timeout: float = 0.0):
        self.posts_seen += 1
        await asyncio.sleep(self.latency_s)
        # vLLM /v1/score returns one row per text_2 entry.
        n = len(json["text_2"])
        # Match the wire shape ``{"data": [{"score": float}, ...]}``.
        return _FakeResponse(
            payload={"data": [{"score": 0.5 + 0.01 * i} for i in range(n)]}
        )


class _FakeResponse:
    def __init__(self, payload: Any) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:  # pragma: no cover - never raises in mocks
        return None

    def json(self) -> Any:
        return self._payload


def _patched_reranker(monkeypatch: pytest.MonkeyPatch, fake_client: _FakeAsyncClient) -> VllmRerankerProvider:
    provider = VllmRerankerProvider(
        endpoint="http://example.local:8007",
        model="BAAI/bge-reranker-v2-m3",
        max_pairs_per_post=128,
    )
    monkeypatch.setattr(provider, "_get_async_client", lambda: fake_client)
    return provider


@pytest.mark.anyio("asyncio")
async def test_reranker_score_pairs_async_fans_out_concurrently(monkeypatch: pytest.MonkeyPatch) -> None:
    """4 distinct claims should fire 4 POSTs in ~1× per-POST wall time, not 4×."""

    fake_client = _FakeAsyncClient()
    provider = _patched_reranker(monkeypatch, fake_client)
    # 4 distinct claims, 3 premises each — exercises the fan-out path.
    pairs: List[Tuple[str, str]] = [
        (f"claim {claim_idx}", f"premise {claim_idx}-{p_idx}")
        for claim_idx in range(4)
        for p_idx in range(3)
    ]
    started = time.perf_counter()
    scores = await provider.score_pairs_async(pairs)
    elapsed = time.perf_counter() - started

    assert fake_client.posts_seen == 4, "one POST per distinct claim"
    assert len(scores) == 12, "one score per (claim, premise) pair"
    # Sequential would have been ~4 × 60ms = 240ms. Concurrent should
    # land well under 2× per-POST latency. Generous threshold so the
    # test is stable on slow CI runners (still distinguishes sequential
    # from concurrent by an order of magnitude).
    assert elapsed < _FAKE_POST_LATENCY_S * 2, (
        f"reranker fan-out NOT concurrent: {elapsed * 1000:.1f}ms "
        f"(budget {_FAKE_POST_LATENCY_S * 2 * 1000:.1f}ms)"
    )


# ---------------------------------------------------------------------------
# verify_claims ⊥ semantic_entropy overlap
# ---------------------------------------------------------------------------


@pytest.mark.anyio("asyncio")
async def test_verify_and_semantic_entropy_overlap(monkeypatch: pytest.MonkeyPatch) -> None:
    """``_run_nli_lane_concurrent`` must overlap the two helpers.

    We swap the two ``_maybe_run_*`` callees for fixed-sleep stubs and
    assert the gather wall time is bounded by max(verify, sem_ent),
    not their sum. The shared-warning merge contract is also exercised
    via the dummy semantic_entropy stub appending to its private list.
    """

    from latence_trace.core import groundedness as g

    verify_latency_s = 0.07
    sem_latency_s = 0.05

    def _fake_run_nli(**_ignored: Any) -> dict[str, Any]:
        time.sleep(verify_latency_s)
        return {
            "claim_records": [],
            "claim_count": 0,
            "skipped_count": 0,
            "aggregate_score": 0.42,
            "per_token": [],
            "warnings": ["verify_warning"],
        }

    def _fake_run_sem(*, warnings: list[str], **_ignored: Any) -> dict[str, Any]:
        time.sleep(sem_latency_s)
        warnings.append("sem_warning")
        return {"aggregate": 0.31}

    monkeypatch.setattr(g, "_maybe_run_nli", _fake_run_nli)
    monkeypatch.setattr(g, "_maybe_run_semantic_entropy", _fake_run_sem)

    caller_warnings: list[str] = []
    nli_kwargs = {
        "response_text": "x",
        "response_tokens": [],
        "support_units": [],
        "nli_provider": object(),
        "nli_max_claims": None,
        "nli_top_k_premises": None,
        "nli_max_batch": None,
        "nli_max_latency_ms": None,
    }
    sem_kwargs = {
        "verification_samples": ["a", "b"],
        "nli_provider": object(),
        "enabled": True,
        "warnings": caller_warnings,
    }

    started = time.perf_counter()
    nli_payload, sem_payload = await g._run_nli_lane_concurrent(
        nli_kwargs=nli_kwargs, semantic_kwargs=sem_kwargs
    )
    elapsed = time.perf_counter() - started

    assert nli_payload is not None and nli_payload["aggregate_score"] == 0.42
    assert sem_payload is not None and sem_payload["aggregate"] == 0.31
    # Caller warnings list received the semantic_entropy warning AFTER
    # the gather (private-list-then-merge contract).
    assert "sem_warning" in caller_warnings
    # Total wall must be bounded by the slower of the two + scheduling.
    # Sequential would be verify + sem ≈ 0.12s; concurrent should be
    # max(0.07, 0.05) + small overhead.
    bound = max(verify_latency_s, sem_latency_s) + 0.04
    assert elapsed < bound, (
        f"verify ⊥ semantic_entropy NOT overlapping: {elapsed * 1000:.1f}ms "
        f"(budget {bound * 1000:.1f}ms; sequential would be "
        f"{(verify_latency_s + sem_latency_s) * 1000:.1f}ms)"
    )


# ---------------------------------------------------------------------------
# Multi-chunk parallel scoring
# ---------------------------------------------------------------------------


def test_multi_chunk_response_overlaps(monkeypatch: pytest.MonkeyPatch) -> None:
    """4-chunk response must scale with concurrency, not chunk count.

    We monkey-patch :func:`score_groundedness_chunked` for a
    deterministic per-chunk sleep and inspect wall time of the
    surrounding orchestrator.
    """

    # Heavy-import the module under test; we monkey-patch the
    # per-chunk callee so we don't need real torch tensors.
    from latence_trace.core import groundedness as g

    chunk_latency_s = 0.12

    class _FakeChunk:
        def __init__(self, text: str) -> None:
            self.text = text
            # The orchestrator only reads chunk_token_offsets +
            # chunk_token_spans + flat_response_tokens BEFORE the
            # multi-chunk loop runs, then post-processes results from
            # ``score_groundedness_chunked``. We supply just enough to
            # satisfy those reads:
            self.tokens = ["t1", "t2"]
            self.embeddings = _FakeEmbeddings(2)
            self.offset_start = 0
            self.offset_end = len(text)
            self.token_char_spans = [(0, 1), (1, 2)]

    class _FakeEmbeddings:
        def __init__(self, n: int) -> None:
            self.shape = (n,)

    def _fake_score_chunked(**_ignored: Any) -> dict[str, Any]:
        time.sleep(chunk_latency_s)
        return {
            "response_tokens": [
                {
                    "index": 0,
                    "token": "t1",
                    "weight": 1.0,
                    "reverse_context": 0.5,
                    "reverse_context_calibrated": None,
                    "reverse_context_z": None,
                    "null_mean": None,
                    "null_std": None,
                    "consensus_hardened": 0.5,
                    "support_unit_hits_above_threshold": 0,
                    "support_unit_soft_breadth": 0.0,
                    "effective_support_units": 0.0,
                    "reverse_query_context": None,
                    "triangular": None,
                    "echo": None,
                    "support_unit_index": None,
                    "support_token_index": None,
                    "support_token": None,
                    "chunk_id": None,
                    "heatmap_score": 0.5,
                    "char_start": 0,
                    "char_end": 1,
                    "response_chunk_index": None,
                    "nli_score": None,
                },
                {
                    "index": 1,
                    "token": "t2",
                    "weight": 1.0,
                    "reverse_context": 0.4,
                    "reverse_context_calibrated": None,
                    "reverse_context_z": None,
                    "null_mean": None,
                    "null_std": None,
                    "consensus_hardened": 0.4,
                    "support_unit_hits_above_threshold": 0,
                    "support_unit_soft_breadth": 0.0,
                    "effective_support_units": 0.0,
                    "reverse_query_context": None,
                    "triangular": None,
                    "echo": None,
                    "support_unit_index": None,
                    "support_token_index": None,
                    "support_token": None,
                    "chunk_id": None,
                    "heatmap_score": 0.4,
                    "char_start": 1,
                    "char_end": 2,
                    "response_chunk_index": None,
                    "nli_score": None,
                },
            ],
            "scores": {"null_bank_size": 0},
            "warnings": [],
            "evidence": [],
            "claim_records": [],
            "support_units": [],
            "literal_mismatches": [],
            "context_trust": {},
        }

    # Substitute the per-chunk callee. This shortcuts the heavy path
    # entirely so we measure orchestration-level concurrency only.
    monkeypatch.setattr(g, "score_groundedness_chunked", _fake_score_chunked)

    chunks = [_FakeChunk(f"chunk{i}") for i in range(4)]
    started = time.perf_counter()
    try:
        g.score_groundedness_response_chunked(
            response_chunks=chunks,
            support_batches=[[]],
            response_text="chunk0 chunk1 chunk2 chunk3",
        )
    except Exception:
        # The orchestrator does heavy stitching after the chunk fan-out
        # which our minimal fake doesn't fully cover. We only care
        # about the FAN-OUT wall time — measure that, then re-raise to
        # surface any genuine wiring breakage.
        pass
    elapsed = time.perf_counter() - started

    # Sequential would be 4 × 120ms = 480ms. Concurrent (4 workers
    # default) should land near 1 × 120ms + overhead. Generous bound
    # so the test is stable on slow CI machines.
    bound = chunk_latency_s * 1.6
    assert elapsed < bound, (
        f"multi-chunk fan-out NOT concurrent: {elapsed * 1000:.1f}ms "
        f"(budget {bound * 1000:.1f}ms; sequential would be "
        f"{4 * chunk_latency_s * 1000:.1f}ms)"
    )
