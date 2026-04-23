"""Noise-injection validation for the file-level dead-weight tracer.

Purpose
-------
Validate that ``file_attribution.attribute_file_usage`` correctly flags
unrelated context files as ``dead_weight`` while sparing genuinely useful
ones. The attribution logic is a pure rollup over ``usage_state`` labels
emitted by the GPU scorer, so we can exercise it end-to-end by:

1. Synthesising a "response" in an embedding space (``T`` normed vectors).
2. Synthesising ``n_signal`` signal files whose chunks are controllably
   close to the response tokens (cosine ≥ ``signal_cosine_min``).
3. Synthesising ``n_noise`` noise files whose chunks are drawn from the
   ambient distribution (cosine distribution centred near zero).
4. Running ``GPUScorer`` to get per-unit ``usage_state`` → rolling up with
   ``_file_usage_from_units`` → evaluating the ``dead_weight`` flag
   against the ground truth (noise files are dead weight, signal files
   are not).

Metrics
-------
- Per-setting precision / recall / F1 of the ``dead_weight`` flag.
- Mean dead-weight-ratio gap between signal and noise tiers.
- Swept across 4 workload mixes (25/50/75 % noise, strong/weak signal).

Output
------
``artifacts/noise_injection_validation.{json,md}``.

The script is self-contained: no runner JSON needed, no encoder weights
loaded. All ops run in the same tensor path the production phantom-guard
uses (``GPUScorer`` on CUDA if available, CPU otherwise)."""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch

sys.path.insert(0, "/workspace/latence-trace")

from research.triangular_maxsim.coding.experiments.file_attribution import (  # noqa: E402
    _file_usage_from_units,
)
from research.triangular_maxsim.coding.experiments.gpu_scorer import GPUScorer  # noqa: E402


_ARTIFACTS = Path(
    "/workspace/latence-trace/research/triangular_maxsim/coding/artifacts"
)


# --- Synthetic fixture -------------------------------------------------------


@dataclass
class FixtureCase:
    response_tokens: List[str]
    response_emb: torch.Tensor                # (T, d) L2-normed
    files: List[Dict[str, Any]]               # {path, is_signal, tokens[], emb}


def _norm(t: torch.Tensor) -> torch.Tensor:
    return t / t.norm(dim=-1, keepdim=True).clamp(min=1e-12)


def _sample_unit_aligned_to(
    response: torch.Tensor,
    *,
    unit_len: int,
    target_min_cosine: float,
    target_max_cosine: float = 1.0,
    rng: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    """Produce ``unit_len`` vectors each with cosine ∈ ``[min, max]``
    against a single anchor response token.

    Strategy: for every unit vector pick a random response token and a
    random orthogonal noise vector, then blend with angle ``theta``
    chosen so that ``cos(theta) ~ U[target_min_cosine, target_max_cosine]``."""
    T, d = response.shape
    picks = torch.randint(
        0, T, (unit_len,), device=device, generator=rng
    )
    anchors = response[picks]  # (unit_len, d)
    noise = torch.randn(unit_len, d, device=device, generator=rng)
    noise = _norm(noise - (noise * anchors).sum(dim=-1, keepdim=True) * anchors)
    cos = target_min_cosine + (target_max_cosine - target_min_cosine) * torch.rand(
        unit_len, device=device, generator=rng
    )
    sin = (1.0 - cos * cos).clamp(min=0.0).sqrt()
    vectors = anchors * cos.unsqueeze(-1) + noise * sin.unsqueeze(-1)
    return _norm(vectors)


def _sample_noise_unit(
    *,
    unit_len: int,
    d: int,
    rng: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    """Random normed vectors — ambient distribution so max cosine against
    the response is small-but-nonzero."""
    vectors = torch.randn(unit_len, d, device=device, generator=rng)
    return _norm(vectors)


def synthesise_case(
    *,
    case_id: str,
    n_signal_files: int,
    n_noise_files: int,
    chunks_per_file: int,
    tokens_per_chunk: int,
    response_len: int,
    dim: int,
    signal_cosine_min: float,
    signal_cosine_max: float,
    signal_chunk_signal_ratio: float,
    device: torch.device,
    seed: int,
) -> FixtureCase:
    rng = torch.Generator(device=device).manual_seed(seed)
    response_emb = _norm(
        torch.randn(response_len, dim, device=device, generator=rng)
    )
    response_tokens = [f"rtok_{i}" for i in range(response_len)]

    files: List[Dict[str, Any]] = []
    # How many chunks per signal file are actually aligned vs boilerplate.
    n_signal_chunks = max(
        1, int(round(signal_chunk_signal_ratio * chunks_per_file))
    )
    for i in range(n_signal_files):
        chunks: List[Tuple[List[str], torch.Tensor]] = []
        for c in range(chunks_per_file):
            if c < n_signal_chunks:
                unit = _sample_unit_aligned_to(
                    response_emb,
                    unit_len=tokens_per_chunk,
                    target_min_cosine=signal_cosine_min,
                    target_max_cosine=signal_cosine_max,
                    rng=rng,
                    device=device,
                )
            else:
                unit = _sample_noise_unit(
                    unit_len=tokens_per_chunk, d=dim, rng=rng, device=device
                )
            tokens = [f"sig{i}_{c}_{j}" for j in range(tokens_per_chunk)]
            chunks.append((tokens, unit))
        files.append(
            {
                "path": f"{case_id}/signal_{i:02d}.py",
                "is_signal": True,
                "chunks": chunks,
            }
        )
    for i in range(n_noise_files):
        chunks = []
        for c in range(chunks_per_file):
            unit = _sample_noise_unit(
                unit_len=tokens_per_chunk, d=dim, rng=rng, device=device
            )
            tokens = [f"noise{i}_{c}_{j}" for j in range(tokens_per_chunk)]
            chunks.append((tokens, unit))
        files.append(
            {
                "path": f"{case_id}/noise_{i:02d}.md",
                "is_signal": False,
                "chunks": chunks,
            }
        )
    return FixtureCase(
        response_tokens=response_tokens,
        response_emb=response_emb,
        files=files,
    )


# --- Scoring + attribution ---------------------------------------------------


def score_and_attribute(
    case: FixtureCase,
    *,
    scorer: GPUScorer,
    dead_weight_threshold: float,
) -> Dict[str, Any]:
    support_units: List[Tuple[List[str], torch.Tensor]] = []
    unit_origins: List[Tuple[str, bool]] = []  # (path, is_signal)
    for fobj in case.files:
        for tokens, emb in fobj["chunks"]:
            support_units.append((tokens, emb))
            unit_origins.append((fobj["path"], fobj["is_signal"]))

    out = scorer(
        response_tokens=case.response_tokens,
        response_embeddings=case.response_emb,
        support_units=support_units,
    )

    owner_counts = out.get("per_unit_owner_count") or [0] * len(support_units)
    units = []
    for i, ((path, _is_sig), state, score) in enumerate(
        zip(unit_origins, out["usage_states"], out["per_unit_max"])
    ):
        units.append(
            {
                "usage_state": state,
                "score": float(score),
                "owner_count": int(owner_counts[i]) if i < len(owner_counts) else 0,
                "metadata": {"path": path},
            }
        )

    attribution = _file_usage_from_units(
        units,
        dead_weight_threshold=dead_weight_threshold,
        total_response_tokens=int(out["n_tokens"]),
        min_owner_share=0.01,
    )

    ground_truth = {f["path"]: f["is_signal"] for f in case.files}

    tp = fp = fn = tn = 0
    for row in attribution["per_file"]:
        is_signal = ground_truth.get(row["path"], True)
        flagged_dead = bool(row["dead_weight"])
        true_dead = not is_signal  # noise files are dead weight
        if flagged_dead and true_dead:
            tp += 1
        elif flagged_dead and not true_dead:
            fp += 1
        elif not flagged_dead and true_dead:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )

    signal_coverages = [
        r["coverage"]
        for r in attribution["per_file"]
        if ground_truth.get(r["path"], False)
    ]
    noise_coverages = [
        r["coverage"]
        for r in attribution["per_file"]
        if not ground_truth.get(r["path"], False)
    ]

    return {
        "attribution": attribution,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "signal_coverage_mean": float(statistics.mean(signal_coverages))
        if signal_coverages
        else None,
        "signal_coverage_min": float(min(signal_coverages))
        if signal_coverages
        else None,
        "noise_coverage_mean": float(statistics.mean(noise_coverages))
        if noise_coverages
        else None,
        "noise_coverage_max": float(max(noise_coverages))
        if noise_coverages
        else None,
        "scorer_latency_ms": float(out["latency_ms"]),
    }


# --- Sweep -------------------------------------------------------------------


def run_sweep(
    *,
    device: torch.device,
    dim: int,
    response_len: int,
    chunks_per_file: int,
    tokens_per_chunk: int,
    dead_weight_threshold: float,
    n_cases_per_cell: int,
) -> Dict[str, Any]:
    scorer = GPUScorer(device=str(device))
    sweep: List[Dict[str, Any]] = []

    cells = [
        {"n_signal": 12, "n_noise": 4, "signal_cosine_min": 0.70, "signal_cosine_max": 1.0, "signal_chunk_signal_ratio": 1.0, "tag": "noise_25pct_strong_clean"},
        {"n_signal": 8, "n_noise": 8, "signal_cosine_min": 0.70, "signal_cosine_max": 1.0, "signal_chunk_signal_ratio": 1.0, "tag": "noise_50pct_strong_clean"},
        {"n_signal": 4, "n_noise": 12, "signal_cosine_min": 0.70, "signal_cosine_max": 1.0, "signal_chunk_signal_ratio": 1.0, "tag": "noise_75pct_strong_clean"},
        # Partial files: only 1/3 chunks per signal file is aligned, the
        # rest is boilerplate — tests that any-used-unit logic rescues the
        # file even when coverage is 33%.
        {"n_signal": 8, "n_noise": 8, "signal_cosine_min": 0.70, "signal_cosine_max": 1.0, "signal_chunk_signal_ratio": 1.0 / 3.0, "tag": "partial_signal_1_of_3_chunks"},
        # Adversarial: ALL signal tokens kept below the 0.55 "used" cut,
        # uniformly in [0.40, 0.54]. Coverage will be 0 on these files
        # because every unit is labelled "uncertain" → flag will fire.
        # This measures the FP rate on files that ARE in context and
        # weakly relevant but never quite "used".
        {"n_signal": 8, "n_noise": 8, "signal_cosine_min": 0.40, "signal_cosine_max": 0.54, "signal_chunk_signal_ratio": 1.0, "tag": "adversarial_all_uncertain_below_used"},
        # Even more adversarial: signal stays in [0.30, 0.44] — below the
        # "uncertain" floor (0.40) on most tokens → units labelled
        # "unused". This case mirrors a genuinely not-useful file that
        # happens to sit in the context window.
        {"n_signal": 8, "n_noise": 8, "signal_cosine_min": 0.30, "signal_cosine_max": 0.44, "signal_chunk_signal_ratio": 1.0, "tag": "adversarial_sub_uncertain"},
    ]

    corpus_stats: List[Dict[str, Any]] = []
    for cell in cells:
        cell_results: List[Dict[str, Any]] = []
        for seed in range(n_cases_per_cell):
            case = synthesise_case(
                case_id=f"{cell['tag']}_{seed:02d}",
                n_signal_files=cell["n_signal"],
                n_noise_files=cell["n_noise"],
                chunks_per_file=chunks_per_file,
                tokens_per_chunk=tokens_per_chunk,
                response_len=response_len,
                dim=dim,
                signal_cosine_min=cell["signal_cosine_min"],
                signal_cosine_max=cell["signal_cosine_max"],
                signal_chunk_signal_ratio=cell["signal_chunk_signal_ratio"],
                device=device,
                seed=seed * 13 + hash(cell["tag"]) & 0xFFFFFFFF,
            )
            res = score_and_attribute(
                case,
                scorer=scorer,
                dead_weight_threshold=dead_weight_threshold,
            )
            cell_results.append(res)

        def _mean(key: str) -> float:
            values = [r[key] for r in cell_results if r[key] is not None]
            return float(statistics.mean(values)) if values else 0.0

        stats = {
            "cell": cell["tag"],
            "config": {
                "n_signal": cell["n_signal"],
                "n_noise": cell["n_noise"],
                "signal_cosine_min": cell["signal_cosine_min"],
                "signal_cosine_max": cell["signal_cosine_max"],
                "signal_chunk_signal_ratio": cell["signal_chunk_signal_ratio"],
            },
            "n_cases": len(cell_results),
            "precision": _mean("precision"),
            "recall": _mean("recall"),
            "f1": _mean("f1"),
            "signal_coverage_mean": _mean("signal_coverage_mean"),
            "signal_coverage_min": _mean("signal_coverage_min"),
            "noise_coverage_mean": _mean("noise_coverage_mean"),
            "noise_coverage_max": _mean("noise_coverage_max"),
            "scorer_latency_ms_mean": _mean("scorer_latency_ms"),
            "confusion_sum": {
                "tp": sum(r["confusion"]["tp"] for r in cell_results),
                "fp": sum(r["confusion"]["fp"] for r in cell_results),
                "fn": sum(r["confusion"]["fn"] for r in cell_results),
                "tn": sum(r["confusion"]["tn"] for r in cell_results),
            },
        }
        corpus_stats.append(stats)
        sweep.append({"cell": cell["tag"], "per_case": cell_results})

    return {
        "dim": dim,
        "response_len": response_len,
        "chunks_per_file": chunks_per_file,
        "tokens_per_chunk": tokens_per_chunk,
        "dead_weight_threshold": dead_weight_threshold,
        "n_cases_per_cell": n_cases_per_cell,
        "device": str(device),
        "cells": corpus_stats,
    }


# --- Rendering ---------------------------------------------------------------


def render_markdown(payload: Dict[str, Any]) -> str:
    lines: List[str] = [
        "# Noise-injection validation — dead-weight tracer",
        "",
        f"- device: `{payload['device']}`",
        f"- dim: `{payload['dim']}`",
        f"- response tokens: `{payload['response_len']}`",
        f"- chunks/file: `{payload['chunks_per_file']}`, tokens/chunk: `{payload['tokens_per_chunk']}`",
        f"- dead-weight threshold (coverage <): `{payload['dead_weight_threshold']}`",
        f"- cases per cell: `{payload['n_cases_per_cell']}`",
        "",
        "## Cells",
        "",
        "| cell | n_sig | n_noise | sig cos range | sig chunk ratio | precision | recall | F1 | cov sig (min) | cov noise (max) | scorer p50 ms |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for c in payload["cells"]:
        cfg = c["config"]
        lines.append(
            "| {t} | {ns} | {nn} | [{smin:.2f},{smax:.2f}] | {cr:.2f} | **{p:.3f}** | **{r:.3f}** | **{f:.3f}** | "
            "{scov:.3f} | {ncov:.3f} | {lat:.2f} |".format(
                t=c["cell"],
                ns=cfg["n_signal"],
                nn=cfg["n_noise"],
                smin=cfg["signal_cosine_min"],
                smax=cfg["signal_cosine_max"],
                cr=cfg["signal_chunk_signal_ratio"],
                p=c["precision"],
                r=c["recall"],
                f=c["f1"],
                scov=c["signal_coverage_min"],
                ncov=c["noise_coverage_max"],
                lat=c["scorer_latency_ms_mean"],
            )
        )

    lines.extend(
        [
            "",
            "## Confusion totals (summed over cases)",
            "",
            "| cell | TP | FP | FN | TN |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for c in payload["cells"]:
        cm = c["confusion_sum"]
        lines.append(
            f"| {c['cell']} | {cm['tp']} | {cm['fp']} | {cm['fn']} | {cm['tn']} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


# --- Main --------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--response-len", type=int, default=96)
    parser.add_argument("--chunks-per-file", type=int, default=3)
    parser.add_argument("--tokens-per-chunk", type=int, default=40)
    parser.add_argument("--dead-weight-threshold", type=float, default=0.20)
    parser.add_argument("--n-cases-per-cell", type=int, default=10)
    parser.add_argument("--out-tag", default="noise_injection_validation")
    args = parser.parse_args()

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")

    payload = run_sweep(
        device=device,
        dim=args.dim,
        response_len=args.response_len,
        chunks_per_file=args.chunks_per_file,
        tokens_per_chunk=args.tokens_per_chunk,
        dead_weight_threshold=args.dead_weight_threshold,
        n_cases_per_cell=args.n_cases_per_cell,
    )

    _ARTIFACTS.mkdir(parents=True, exist_ok=True)
    json_path = _ARTIFACTS / f"{args.out_tag}.json"
    md_path = _ARTIFACTS / f"{args.out_tag}.md"
    json_path.write_text(json.dumps(payload, indent=2))
    md_path.write_text(render_markdown(payload))
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
