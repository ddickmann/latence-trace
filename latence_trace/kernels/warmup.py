"""Triton-kernel warmup singletons for the standalone server.

Triton's JIT compiler caches the compiled kernel per (config, shape) tuple
on first call. The first request that hits the service therefore pays a
one-shot ~120 ms compile cost on top of the actual MaxSim work. By
running a tiny warmup pass on representative shape grids during process
startup, the cache is hot before any user request arrives and the
``/readyz`` probe only flips to 200 once the cache is primed - which
also gives Kubernetes a clean "drain warmup before serving traffic"
signal.

The warmup is best-effort: when CUDA is not available (CPU-only deploys,
test environments) the module short-circuits to a no-op success so the
same code path stays valid for both deployment shapes.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from latence_trace.kernels.triton_triangular_maxsim import (
    triangular_maxsim,
    triangular_maxsim_reference,
)

logger = logging.getLogger(__name__)


# Representative (S, T, U, H) shape grids for the three Pareto-optimal
# default profiles. Each tuple covers (query_tokens, context_tokens,
# response_tokens, embedding_dim). The numbers were sampled from the
# post-audit benchmark sweep so the JIT cache primes the same buckets the
# 50 RPS load test exercises.
#
# All profiles use the same H=128 because the multilingual ColBERT
# default emits 128-dim per-token vectors. If a customer swaps the
# encoder via VOYAGER_GROUNDEDNESS_MODEL the warmup still helps - the
# JIT only needs to recompile for the new H dimension on the first
# request to the new shape, which is exactly the cost we want to amortize
# during startup.
_PROFILE_SHAPE_GRIDS: Dict[str, List[Tuple[int, int, int, int]]] = {
    "fast": [
        (4, 32, 16, 128),
        (8, 64, 32, 128),
    ],
    "balanced": [
        (4, 32, 16, 128),
        (12, 128, 40, 128),
        (24, 256, 80, 128),
    ],
    "quality": [
        (8, 64, 32, 128),
        (24, 256, 80, 128),
        (48, 512, 160, 128),
    ],
}

# Default for unknown profile names.
_DEFAULT_GRID = _PROFILE_SHAPE_GRIDS["balanced"]


@dataclass
class WarmupResult:
    """Outcome of a warmup pass.

    ``ok`` is True when every shape compiled without raising. ``shapes``
    records the list of (S, T, U, H) tuples that were exercised.
    ``elapsed_ms`` records the total wall time. ``device`` records the
    CUDA device used (or ``"cpu"`` when warmup ran the reference path).
    """

    ok: bool = False
    profile: str = "balanced"
    shapes: List[Tuple[int, int, int, int]] = field(default_factory=list)
    elapsed_ms: float = 0.0
    device: str = "cpu"
    error: Optional[str] = None


# Process-wide warmup state. We use a lock so two concurrent worker
# startup paths (e.g. uvicorn workers + an ASGI lifespan callback) do not
# race and double-warm the kernels.
_WARMUP_LOCK = threading.Lock()
_WARMUP_STATE: Dict[str, WarmupResult] = {}
_WARMUP_DONE = threading.Event()


def is_warm() -> bool:
    """True once warmup has completed (success or failure)."""

    return _WARMUP_DONE.is_set()


def warmup_state() -> Dict[str, WarmupResult]:
    """Snapshot of the most recent warmup result per profile.

    Useful for ``/readyz`` to surface diagnostics, and for the CLI's
    ``latence-trace warm`` command to print human-readable output.
    """

    return dict(_WARMUP_STATE)


def reset_warmup_state_for_tests() -> None:
    """Test hook: clear the cached warmup result.

    Production code should never call this. Exposed only so the test
    suite can re-run the warmup path against a fresh state.
    """

    with _WARMUP_LOCK:
        _WARMUP_STATE.clear()
        _WARMUP_DONE.clear()


def warm_all(profile: str = "balanced", *, force: bool = False) -> WarmupResult:
    """Warm the Triton kernels for the given profile.

    Parameters
    ----------
    profile:
        One of ``fast``, ``balanced``, ``quality``. Unknown values fall
        back to the ``balanced`` shape grid so a custom-profile deploy
        still benefits from primed kernels.
    force:
        When False (the default) returns the cached result on subsequent
        calls so Uvicorn workers, the ASGI lifespan hook, and the CLI
        ``latence-trace warm`` subcommand all share the same warmup pass.
        When True, re-runs the warmup unconditionally - useful for tests
        and for swapping shape grids at runtime.
    """

    profile_key = (profile or "balanced").strip().lower()

    if not force:
        cached = _WARMUP_STATE.get(profile_key)
        if cached is not None:
            return cached

    with _WARMUP_LOCK:
        if not force:
            cached = _WARMUP_STATE.get(profile_key)
            if cached is not None:
                return cached

        grid = _PROFILE_SHAPE_GRIDS.get(profile_key, _DEFAULT_GRID)
        result = WarmupResult(profile=profile_key, shapes=list(grid))
        start = time.perf_counter()
        cuda_ok = torch.cuda.is_available()
        result.device = f"cuda:{torch.cuda.current_device()}" if cuda_ok else "cpu"

        try:
            for s, t, u, h in grid:
                if cuda_ok:
                    Q = torch.randn(s, h, device="cuda")
                    C = torch.randn(t, h, device="cuda")
                    R = torch.randn(u, h, device="cuda")
                    triangular_maxsim(Q, C, R, normalize=True, use_kernel=True)
                else:
                    # CPU-only deploys never JIT a Triton kernel; touch
                    # the reference path so any import-time errors in the
                    # PyTorch fallback also surface during startup
                    # rather than on the first request.
                    Q = torch.randn(s, h)
                    C = torch.randn(t, h)
                    R = torch.randn(u, h)
                    triangular_maxsim_reference(Q, C, R, normalize=True)
            result.ok = True
        except Exception as exc:
            # Warmup failure is logged but never crashes the process -
            # the first user request will still try to JIT and we want
            # the service to come up even when Triton is unavailable.
            result.ok = False
            result.error = str(exc)
            logger.warning(
                "kernel_warmup_failed",
                extra={"profile": profile_key, "error": str(exc)},
            )

        result.elapsed_ms = (time.perf_counter() - start) * 1000.0
        _WARMUP_STATE[profile_key] = result
        _WARMUP_DONE.set()

        logger.info(
            "kernel_warmup_complete",
            extra={
                "profile": profile_key,
                "ok": result.ok,
                "shapes": result.shapes,
                "elapsed_ms": round(result.elapsed_ms, 2),
                "device": result.device,
            },
        )
        return result


def warm_code_lane(*, force: bool = False) -> WarmupResult:
    """Prime the code-lane singletons.

    Loads tree-sitter grammars for every supported language and runs a
    tiny :class:`~latence_trace.core.code_lane.GPUScorer` pass so its
    dedicated CUDA stream is allocated before any user traffic arrives.

    Like :func:`warm_all`, failures are logged and do not crash the
    process — callers can inspect :attr:`WarmupResult.ok` on the
    ``code_lane`` key of :func:`warmup_state`.
    """

    cache_key = "code_lane"
    if not force:
        cached = _WARMUP_STATE.get(cache_key)
        if cached is not None:
            return cached

    with _WARMUP_LOCK:
        if not force:
            cached = _WARMUP_STATE.get(cache_key)
            if cached is not None:
                return cached

        result = WarmupResult(profile=cache_key, shapes=[])
        start = time.perf_counter()
        cuda_ok = torch.cuda.is_available()
        result.device = f"cuda:{torch.cuda.current_device()}" if cuda_ok else "cpu"

        try:
            from latence_trace.core.code_lane import (
                AstSymbolExtractor,
                GPUScorer,
                SUPPORTED_LANGUAGES,
            )

            extractor = AstSymbolExtractor(enabled=True)
            for language in SUPPORTED_LANGUAGES:
                extractor.extract_from_text(
                    "def sample():\n    pass\n", language_hint=language
                )

            device = "cuda" if cuda_ok else "cpu"
            scorer = GPUScorer(device=device)
            R = torch.randn(4, 8)
            C = torch.randn(4, 8)
            scorer(
                response_tokens=["a", "b", "c", "d"],
                response_embeddings=R,
                support_units=[(("a", "b", "c", "d"), C)],
                literal_tokens=["a"],
                query_embeddings=None,
            )
            result.ok = True
        except Exception as exc:
            result.ok = False
            result.error = str(exc)
            logger.warning("code_lane_warmup_failed", extra={"error": str(exc)})

        result.elapsed_ms = (time.perf_counter() - start) * 1000.0
        _WARMUP_STATE[cache_key] = result
        _WARMUP_DONE.set()
        logger.info(
            "code_lane_warmup_complete",
            extra={
                "ok": result.ok,
                "elapsed_ms": round(result.elapsed_ms, 2),
                "device": result.device,
            },
        )
        return result


__all__ = [
    "WarmupResult",
    "is_warm",
    "reset_warmup_state_for_tests",
    "warm_all",
    "warm_code_lane",
    "warmup_state",
]
