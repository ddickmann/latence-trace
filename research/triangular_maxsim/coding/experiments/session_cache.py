"""Cross-turn support-unit embedding cache.

For the phantom-guard request lane, the *response* changes every turn
but the *context* support units change incrementally: the agent reads
maybe one new file and keeps the rest from the previous turn. The
dominant cost of ``score_groundedness_response_chunked`` is encoding the
support units, not scoring them. If we can skip re-encoding the 95% of
units that didn't change, the warm p95 latency drops dramatically.

The ``SessionContext`` here is a tiny LRU + signature cache keyed on
``support_unit_signature``. At API call time the client sends:

    mode = "code"
    session_id = "<agent_session>"
    context_files = [{path, content, optional_signature}, ...]

The server:

1. Derives signatures for each incoming support-unit candidate.
2. Looks them up in ``SessionContext(session_id).units``.
3. Encodes only the cache misses.
4. Stores the freshly-encoded misses back into the cache.

This module implements the pure Python side of that flow. Actual
wiring into the ``GroundednessService`` happens in the production
follow-up plan.

The cache is bounded per-session with an LRU eviction policy.
"""
from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch


def signature_for_text(path: str, content: str) -> str:
    """Deterministic signature for a (path, content) pair. Used as the
    cache key so identical content across turns resolves to the same
    embedding even if the client forgets to propagate a signature."""
    h = hashlib.blake2b(digest_size=16)
    h.update(path.encode("utf-8", "replace"))
    h.update(b"\x00")
    h.update(content.encode("utf-8", "replace"))
    return h.hexdigest()


@dataclass
class CachedSupportUnit:
    signature: str
    path: str
    tokens: List[str]
    embeddings: torch.Tensor
    last_accessed: float = field(default_factory=time.time)
    size_bytes: int = 0


class SessionContext:
    """Single-session LRU cache of encoded support units."""

    def __init__(
        self,
        session_id: str,
        *,
        max_units: int = 200,
        max_bytes: int = 512 * 1024 * 1024,  # 512 MiB cap per session
    ) -> None:
        self.session_id = session_id
        self.max_units = max_units
        self.max_bytes = max_bytes
        self._units: "OrderedDict[str, CachedSupportUnit]" = OrderedDict()
        self._lock = threading.RLock()
        self.stats = {"hits": 0, "misses": 0, "evictions": 0}

    def size_bytes(self) -> int:
        with self._lock:
            return sum(u.size_bytes for u in self._units.values())

    def get(self, signature: str) -> Optional[CachedSupportUnit]:
        with self._lock:
            unit = self._units.get(signature)
            if unit is not None:
                unit.last_accessed = time.time()
                self._units.move_to_end(signature)
                self.stats["hits"] += 1
            else:
                self.stats["misses"] += 1
            return unit

    def put(self, unit: CachedSupportUnit) -> None:
        with self._lock:
            self._units[unit.signature] = unit
            self._units.move_to_end(unit.signature)
            self._enforce_bounds()

    def _enforce_bounds(self) -> None:
        while len(self._units) > self.max_units or self.size_bytes() > self.max_bytes:
            if not self._units:
                return
            self._units.popitem(last=False)
            self.stats["evictions"] += 1

    def resolve_many(
        self,
        candidates: Sequence[Tuple[str, str, str]],  # (signature, path, content)
        *,
        encode_fn: Callable[[Sequence[Tuple[str, str]]], Sequence[CachedSupportUnit]],
    ) -> List[CachedSupportUnit]:
        """Return encoded units for every candidate, encoding cache misses
        once and caching the result.

        ``encode_fn`` is invoked only with the (path, content) of cache
        misses; it must return ``CachedSupportUnit`` objects with the
        correct ``signature`` field (use ``signature_for_text``)."""
        to_encode: List[Tuple[str, str, str]] = []
        resolved: Dict[str, CachedSupportUnit] = {}
        with self._lock:
            for signature, path, content in candidates:
                cached = self.get(signature)
                if cached is not None:
                    resolved[signature] = cached
                else:
                    to_encode.append((signature, path, content))

        if to_encode:
            encoded = list(
                encode_fn([(path, content) for _, path, content in to_encode])
            )
            if len(encoded) != len(to_encode):
                raise RuntimeError(
                    f"encode_fn returned {len(encoded)} units for "
                    f"{len(to_encode)} requests"
                )
            for (signature, path, _content), unit in zip(to_encode, encoded):
                if unit.signature != signature:
                    unit = CachedSupportUnit(
                        signature=signature,
                        path=unit.path,
                        tokens=unit.tokens,
                        embeddings=unit.embeddings,
                        size_bytes=unit.size_bytes,
                    )
                self.put(unit)
                resolved[signature] = unit

        return [resolved[sig] for sig, _, _ in candidates]


class SessionCacheRegistry:
    """Process-wide singleton of per-session caches."""

    def __init__(self, *, max_sessions: int = 64) -> None:
        self.max_sessions = max_sessions
        self._sessions: "OrderedDict[str, SessionContext]" = OrderedDict()
        self._lock = threading.RLock()

    def get_or_create(self, session_id: str, **kwargs) -> SessionContext:
        with self._lock:
            ctx = self._sessions.get(session_id)
            if ctx is None:
                ctx = SessionContext(session_id, **kwargs)
                self._sessions[session_id] = ctx
                self._sessions.move_to_end(session_id)
                while len(self._sessions) > self.max_sessions:
                    self._sessions.popitem(last=False)
            else:
                self._sessions.move_to_end(session_id)
            return ctx

    def get(self, session_id: str) -> Optional[SessionContext]:
        with self._lock:
            return self._sessions.get(session_id)

    def stats(self) -> Dict[str, Dict[str, int]]:
        with self._lock:
            return {sid: dict(ctx.stats) for sid, ctx in self._sessions.items()}


_GLOBAL_REGISTRY = SessionCacheRegistry()


def global_session_registry() -> SessionCacheRegistry:
    return _GLOBAL_REGISTRY


# --- 10-turn simulation -------------------------------------------------------


def simulate_cache_hit_rate(
    n_turns: int = 10,
    n_files_per_turn: int = 20,
    churn_rate: float = 0.05,
) -> Dict[str, float]:
    """Simulate a ``n_turns``-turn agent session where each turn carries
    ``n_files_per_turn`` files and ``churn_rate`` fraction of them change.

    Returns hit-rate statistics under the LRU policy.
    """
    import random

    random.seed(0)
    registry = SessionCacheRegistry(max_sessions=4)
    ctx = registry.get_or_create("sim")

    base_files = [(f"f{i:03d}", f"content_{i}_v1") for i in range(n_files_per_turn)]

    def _fake_encode(pairs: Sequence[Tuple[str, str]]) -> List[CachedSupportUnit]:
        out: List[CachedSupportUnit] = []
        for path, content in pairs:
            sig = signature_for_text(path, content)
            emb = torch.randn(40, 128) / 40 ** 0.5
            out.append(
                CachedSupportUnit(
                    signature=sig,
                    path=path,
                    tokens=[f"tok{i}" for i in range(40)],
                    embeddings=emb,
                    size_bytes=emb.numel() * emb.element_size(),
                )
            )
        return out

    total_hits = 0
    total_requests = 0
    total_encodes = 0
    timings: List[float] = []
    for turn in range(n_turns):
        if turn > 0:
            k = max(1, int(churn_rate * len(base_files)))
            idxs = random.sample(range(len(base_files)), k)
            for i in idxs:
                path, content = base_files[i]
                version = int(content.rsplit("_v", 1)[1]) + 1
                base_files[i] = (path, f"content_{path}_v{version}")
        candidates = [
            (signature_for_text(p, c), p, c) for p, c in base_files
        ]
        before = time.perf_counter()
        resolved = ctx.resolve_many(candidates, encode_fn=_fake_encode)
        timings.append((time.perf_counter() - before) * 1000.0)
        total_requests += len(candidates)
        turn_hits = ctx.stats["hits"] - total_hits
        total_hits = ctx.stats["hits"]
        turn_encodes = len(candidates) - turn_hits
        total_encodes += turn_encodes

    return {
        "hit_rate": float(total_hits) / max(1, total_requests),
        "total_requests": total_requests,
        "total_hits": total_hits,
        "total_encodes": total_encodes,
        "evictions": ctx.stats["evictions"],
        "turn_latencies_ms": timings,
    }


if __name__ == "__main__":
    import json

    result = simulate_cache_hit_rate(n_turns=10, n_files_per_turn=25, churn_rate=0.05)
    print(json.dumps(result, indent=2))
