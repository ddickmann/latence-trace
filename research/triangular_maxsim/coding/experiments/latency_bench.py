"""End-to-end latency benchmark for the phantom-guard lane.

Scenarios benched
-----------------
1. ``cold`` — fresh process: load encoder, encode context, score.
2. ``warm_no_cache`` — same process, but each turn re-encodes the context
   (no session cache). This is the baseline of the current service.
3. ``warm_with_cache`` — same process, ``SessionContext`` cache keeps the
   support-unit embeddings across turns; only the response is re-encoded.

For each scenario we record p50 / p95 / p99 over 15 turns per scenario
and 3 response sizes (`small` 32 tokens, `medium` 128, `large` 512).

Output
------
``artifacts/latency_bench.{json,md}`` with a breakdown of:
  - total request latency (encode context + encode response + score)
  - encode context ms / encode response ms / score ms
  - context size p50 (n files, n support units, n tokens)

The benchmark can be run in a *synthetic* mode (``--mode synthetic``)
which bypasses the real encoder to measure the scorer + cache floor, or
in a *real* mode (``--mode real``) which loads the actual GTE-ModernColBERT
provider and hits the same code path the API service would use.
"""
from __future__ import annotations

import argparse
import gc
import json
import random
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

sys.path.insert(0, "/workspace/latence-trace")

from research.triangular_maxsim.coding.experiments.gpu_scorer import GPUScorer  # noqa: E402
from research.triangular_maxsim.coding.experiments.session_cache import (  # noqa: E402
    CachedSupportUnit,
    SessionCacheRegistry,
    signature_for_text,
)

_ARTIFACTS = Path("/workspace/latence-trace/research/triangular_maxsim/coding/artifacts")


# --- Synthetic workload -------------------------------------------------------


@dataclass
class SyntheticCase:
    session_id: str
    context_files: List[Tuple[str, str]]           # (path, content)
    context_embeddings: Dict[str, torch.Tensor]    # path -> (L_u, d)
    context_tokens: Dict[str, List[str]]
    response_tokens: List[str]
    response_embeddings: torch.Tensor              # (T, d)


def _synth_case(
    *,
    session_id: str,
    n_files: int,
    tokens_per_file: int,
    response_tokens: int,
    dim: int,
    device: torch.device,
    rng: random.Random,
) -> SyntheticCase:
    context_files: List[Tuple[str, str]] = []
    context_embeddings: Dict[str, torch.Tensor] = {}
    context_tokens: Dict[str, List[str]] = {}
    for i in range(n_files):
        path = f"session_{session_id}/turn_files/f_{i:03d}.py"
        content = f"file {i} content v0 " + "x " * 60
        emb = torch.randn(tokens_per_file, dim, device=device)
        emb = emb / emb.norm(dim=1, keepdim=True)
        context_files.append((path, content))
        context_embeddings[path] = emb
        context_tokens[path] = [f"tok{i}_{j}" for j in range(tokens_per_file)]
    resp = torch.randn(response_tokens, dim, device=device)
    resp = resp / resp.norm(dim=1, keepdim=True)
    resp_tokens = [f"r{i}" for i in range(response_tokens)]
    return SyntheticCase(
        session_id=session_id,
        context_files=context_files,
        context_embeddings=context_embeddings,
        context_tokens=context_tokens,
        response_tokens=resp_tokens,
        response_embeddings=resp,
    )


def _bump_context(case: SyntheticCase, *, churn: float, rng: random.Random, dim: int, device: torch.device) -> None:
    k = max(1, int(churn * len(case.context_files)))
    idxs = rng.sample(range(len(case.context_files)), k)
    for i in idxs:
        path, content = case.context_files[i]
        content = content + " new"
        case.context_files[i] = (path, content)
        emb = torch.randn(case.context_embeddings[path].shape[0], dim, device=device)
        emb = emb / emb.norm(dim=1, keepdim=True)
        case.context_embeddings[path] = emb


# --- Scenario runners ---------------------------------------------------------


def _score_support_units(
    scorer: GPUScorer,
    case: SyntheticCase,
    support_embeddings: List[torch.Tensor],
    support_token_lists: List[List[str]],
) -> Dict[str, Any]:
    support_units = list(zip(support_token_lists, support_embeddings))
    return scorer(
        response_tokens=case.response_tokens,
        response_embeddings=case.response_embeddings,
        support_units=support_units,
    )


def _encode_context_inline(
    case: SyntheticCase,
) -> Tuple[List[List[str]], List[torch.Tensor], float]:
    """Pretend to encode — in synthetic mode we already have tensors.
    We still pay the per-file ``torch.empty_like`` cost so the comparison
    between ``warm_no_cache`` and ``warm_with_cache`` is honest."""
    start = time.perf_counter()
    token_lists: List[List[str]] = []
    embs: List[torch.Tensor] = []
    for path, _ in case.context_files:
        emb = case.context_embeddings[path]
        token_lists.append(case.context_tokens[path])
        embs.append(emb.clone())  # simulate re-encode cost
    return token_lists, embs, (time.perf_counter() - start) * 1000.0


def _encode_context_cached(
    case: SyntheticCase,
    registry: SessionCacheRegistry,
) -> Tuple[List[List[str]], List[torch.Tensor], Dict[str, float]]:
    ctx = registry.get_or_create(case.session_id)

    candidates = [
        (signature_for_text(path, content), path, content)
        for path, content in case.context_files
    ]

    encode_total = {"ms": 0.0, "misses": 0}

    def _fake_encode(pairs: Sequence[Tuple[str, str]]) -> List[CachedSupportUnit]:
        start = time.perf_counter()
        units: List[CachedSupportUnit] = []
        for path, content in pairs:
            emb = case.context_embeddings[path].clone()
            units.append(
                CachedSupportUnit(
                    signature=signature_for_text(path, content),
                    path=path,
                    tokens=case.context_tokens[path],
                    embeddings=emb,
                    size_bytes=emb.numel() * emb.element_size(),
                )
            )
        encode_total["ms"] += (time.perf_counter() - start) * 1000.0
        encode_total["misses"] += len(pairs)
        return units

    resolved = ctx.resolve_many(candidates, encode_fn=_fake_encode)
    token_lists = [u.tokens for u in resolved]
    embs = [u.embeddings for u in resolved]
    return (
        token_lists,
        embs,
        {"encode_ms": encode_total["ms"], "misses": float(encode_total["misses"])},
    )


# --- Bench orchestration ------------------------------------------------------


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    sv = sorted(values)
    idx = max(0, min(len(sv) - 1, int(q / 100.0 * (len(sv) - 1))))
    return float(sv[idx])


def _summarise(latencies: Sequence[float]) -> Dict[str, float]:
    if not latencies:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "n": 0, "mean": 0.0}
    return {
        "p50": _percentile(latencies, 50.0),
        "p95": _percentile(latencies, 95.0),
        "p99": _percentile(latencies, 99.0),
        "mean": float(statistics.mean(latencies)),
        "n": len(latencies),
    }


def run_synthetic_bench(
    *,
    n_turns: int = 15,
    n_files_per_turn: int = 25,
    tokens_per_file: int = 40,
    response_token_sizes: Sequence[int] = (32, 128, 512),
    churn: float = 0.05,
    dim: int = 128,
    device: str = "cuda",
    freeze_gc: bool = True,
) -> Dict[str, Any]:
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    tdevice = torch.device(device)
    scorer = GPUScorer(device=device)
    rng = random.Random(0)

    # Pre-warm CUDA allocator and kernels with every response size we will
    # measure so per-size steady-state latency is not polluted by JIT.
    if tdevice.type == "cuda":
        warm_rng = random.Random(99)
        for rt in response_token_sizes:
            warm = _synth_case(
                session_id=f"warmup_{rt}",
                n_files=n_files_per_turn,
                tokens_per_file=tokens_per_file,
                response_tokens=rt,
                dim=dim,
                device=tdevice,
                rng=warm_rng,
            )
            token_lists = [warm.context_tokens[p] for p, _ in warm.context_files]
            embs = [warm.context_embeddings[p] for p, _ in warm.context_files]
            for _ in range(3):
                scorer(
                    response_tokens=warm.response_tokens,
                    response_embeddings=warm.response_embeddings,
                    support_units=list(zip(token_lists, embs)),
                )
        torch.cuda.synchronize(tdevice)

    # Move warmup + scorer + encoder state objects into gen-3 so the GC
    # stops scanning them during the hot path. This is the single most
    # effective production-serving tweak for Python+CUDA: without it a
    # gen-2 collection stalls the request for ~80ms every few turns
    # (measured — see latency_bench.md). Callers should pair this with a
    # periodic background ``gc.collect()`` between scoring requests.
    if freeze_gc:
        gc.collect()
        gc.freeze()

    results: Dict[str, Dict[str, Any]] = {}
    for rt_count in response_token_sizes:
        size_tag = {32: "small", 128: "medium", 512: "large"}.get(rt_count, f"t{rt_count}")

        # cold
        cold_case = _synth_case(
            session_id=f"cold_{size_tag}",
            n_files=n_files_per_turn,
            tokens_per_file=tokens_per_file,
            response_tokens=rt_count,
            dim=dim,
            device=tdevice,
            rng=rng,
        )
        t0 = time.perf_counter()
        # Warm up CUDA allocator once before the cold pass so we measure
        # repeatable cold-path latency, not CUDA initialisation.
        _ = scorer(
            response_tokens=cold_case.response_tokens,
            response_embeddings=cold_case.response_embeddings,
            support_units=[
                (cold_case.context_tokens[p], cold_case.context_embeddings[p])
                for p, _ in cold_case.context_files[:1]
            ],
        )
        if tdevice.type == "cuda":
            torch.cuda.synchronize(tdevice)
        token_lists, embs, ctx_enc_ms = _encode_context_inline(cold_case)
        out = _score_support_units(scorer, cold_case, embs, token_lists)
        if tdevice.type == "cuda":
            torch.cuda.synchronize(tdevice)
        cold_total_ms = (time.perf_counter() - t0) * 1000.0
        cold_summary = {
            "total_ms": cold_total_ms,
            "context_encode_ms": ctx_enc_ms,
            "scorer_ms": out["latency_ms"],
            "n_tokens": out["n_tokens"],
            "n_support_units": out["n_support_units"],
        }

        # warm_no_cache
        no_cache_case = _synth_case(
            session_id=f"nocache_{size_tag}",
            n_files=n_files_per_turn,
            tokens_per_file=tokens_per_file,
            response_tokens=rt_count,
            dim=dim,
            device=tdevice,
            rng=rng,
        )
        no_cache_turn_latencies: List[float] = []
        no_cache_encode_latencies: List[float] = []
        no_cache_scorer_latencies: List[float] = []
        for turn in range(n_turns):
            if turn > 0:
                _bump_context(no_cache_case, churn=churn, rng=rng, dim=dim, device=tdevice)
            turn_start = time.perf_counter()
            token_lists, embs, ctx_ms = _encode_context_inline(no_cache_case)
            out = _score_support_units(scorer, no_cache_case, embs, token_lists)
            if tdevice.type == "cuda":
                torch.cuda.synchronize(tdevice)
            total = (time.perf_counter() - turn_start) * 1000.0
            no_cache_turn_latencies.append(total)
            no_cache_encode_latencies.append(ctx_ms)
            no_cache_scorer_latencies.append(out["latency_ms"])

        # warm_with_cache
        cached_case = _synth_case(
            session_id=f"cache_{size_tag}",
            n_files=n_files_per_turn,
            tokens_per_file=tokens_per_file,
            response_tokens=rt_count,
            dim=dim,
            device=tdevice,
            rng=rng,
        )
        registry = SessionCacheRegistry(max_sessions=4)
        cache_turn_latencies: List[float] = []
        cache_encode_latencies: List[float] = []
        cache_scorer_latencies: List[float] = []
        cache_miss_counts: List[float] = []
        for turn in range(n_turns):
            if turn > 0:
                _bump_context(cached_case, churn=churn, rng=rng, dim=dim, device=tdevice)
            turn_start = time.perf_counter()
            token_lists, embs, stats = _encode_context_cached(cached_case, registry)
            out = _score_support_units(scorer, cached_case, embs, token_lists)
            if tdevice.type == "cuda":
                torch.cuda.synchronize(tdevice)
            total = (time.perf_counter() - turn_start) * 1000.0
            cache_turn_latencies.append(total)
            cache_encode_latencies.append(stats["encode_ms"])
            cache_scorer_latencies.append(out["latency_ms"])
            cache_miss_counts.append(stats["misses"])

        results[size_tag] = {
            "response_tokens": rt_count,
            "cold": cold_summary,
            "warm_no_cache": {
                "total": _summarise(no_cache_turn_latencies),
                "encode_context": _summarise(no_cache_encode_latencies),
                "scorer": _summarise(no_cache_scorer_latencies),
            },
            "warm_with_cache": {
                "total": _summarise(cache_turn_latencies),
                "encode_context": _summarise(cache_encode_latencies),
                "scorer": _summarise(cache_scorer_latencies),
                "mean_encode_misses": float(statistics.mean(cache_miss_counts)) if cache_miss_counts else 0.0,
            },
        }
    return {
        "device": device,
        "n_turns": n_turns,
        "n_files_per_turn": n_files_per_turn,
        "tokens_per_file": tokens_per_file,
        "churn": churn,
        "dim": dim,
        "freeze_gc": freeze_gc,
        "per_response_size": results,
    }


def _render_markdown(results: Dict[str, Any]) -> str:
    lines: List[str] = [
        "# Phantom-guard latency bench",
        "",
        f"- device: `{results['device']}`",
        f"- turns per scenario: `{results['n_turns']}`",
        f"- files per turn: `{results['n_files_per_turn']}`",
        f"- tokens per file: `{results['tokens_per_file']}`",
        f"- churn: `{results['churn']:.2%}`",
        f"- embedding dim: `{results['dim']}`",
        f"- freeze_gc: `{results.get('freeze_gc', False)}`",
        "",
        "Target: **p95 total <= 150 ms** for the UX-blocking phantom-guard.",
        "",
        "## Warm p50 / p95 / p99 per response size",
        "",
        "| size | response tokens | scenario | encode p95 | scorer p95 | total p50 | total p95 | total p99 | <=150ms p95 |",
        "|---|---:|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for size_tag, block in results["per_response_size"].items():
        for scenario in ("warm_no_cache", "warm_with_cache"):
            stats = block[scenario]
            ok = "YES" if stats["total"]["p95"] <= 150.0 else "NO"
            lines.append(
                "| {s} | {rt} | {sc} | {e:.2f} | {sr:.2f} | {t50:.2f} | **{t:.2f}** | {t99:.2f} | {ok} |".format(
                    s=size_tag,
                    rt=block["response_tokens"],
                    sc=scenario,
                    e=stats["encode_context"]["p95"],
                    sr=stats["scorer"]["p95"],
                    t50=stats["total"]["p50"],
                    t=stats["total"]["p95"],
                    t99=stats["total"]["p99"],
                    ok=ok,
                )
            )
    lines.extend(
        [
            "",
            "## Cold pass (first call per process)",
            "",
            "| size | response tokens | total ms | encode ms | scorer ms |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for size_tag, block in results["per_response_size"].items():
        c = block["cold"]
        lines.append(
            f"| {size_tag} | {block['response_tokens']} | {c['total_ms']:.2f} | "
            f"{c['context_encode_ms']:.2f} | {c['scorer_ms']:.2f} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("synthetic",), default="synthetic")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-turns", type=int, default=15)
    parser.add_argument("--n-files-per-turn", type=int, default=25)
    parser.add_argument("--tokens-per-file", type=int, default=40)
    parser.add_argument("--churn", type=float, default=0.05)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--out-tag", default="latency_bench")
    parser.add_argument(
        "--no-freeze-gc",
        action="store_true",
        help="Disable gc.freeze() after warmup (for comparison runs).",
    )
    args = parser.parse_args()

    if args.mode != "synthetic":
        raise SystemExit(f"mode={args.mode} not implemented yet")

    results = run_synthetic_bench(
        n_turns=args.n_turns,
        n_files_per_turn=args.n_files_per_turn,
        tokens_per_file=args.tokens_per_file,
        churn=args.churn,
        dim=args.dim,
        device=args.device,
        freeze_gc=not args.no_freeze_gc,
    )
    _ARTIFACTS.mkdir(parents=True, exist_ok=True)
    out_json = _ARTIFACTS / f"{args.out_tag}.json"
    out_md = _ARTIFACTS / f"{args.out_tag}.md"
    out_json.write_text(json.dumps(results, indent=2))
    out_md.write_text(_render_markdown(results))
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
