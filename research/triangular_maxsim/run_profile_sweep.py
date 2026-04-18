"""Run the 8-lane Pareto profile sweep for latence-trace.

Each lane invokes :py:func:`research.triangular_maxsim.groundedness_external_eval.main`
in-process with a different feature mask (NLI on/off, reranker on/off,
atomic on/off, semantic-entropy on/off, top-k variants), records peak
GPU VRAM around the run, and merges the resulting JSON report with a
``"vram"`` block. Final summary is written to ``profile_sweep_summary.json``.

Lane definitions match docs/plans/pareto-default-profiles.

Usage:
    python -m research.triangular_maxsim.run_profile_sweep \
        --pairs-per-stratum 20 --max-external-per-stratum 20 \
        --out-dir research/triangular_maxsim/reports
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from research.triangular_maxsim import groundedness_external_eval as harness  # noqa: E402


# Lane definitions: the harness CLI is the source of truth for flag
# semantics; we just preset the argv each lane needs.
def _lane_argv(lane: str, base: List[str]) -> List[str]:
    """Return CLI argv for a given lane, layered on top of common base flags."""

    if lane == "L0":
        # encoder + literal only
        return base
    if lane == "L1":
        # NLI on, no reranker, no atomic, top-k=3 default, no concat
        return base + ["--enable-nli", "--no-concat-premises", "--no-atomic-claims"]
    if lane == "L2":
        # L1 + atomic
        return base + ["--enable-nli", "--no-concat-premises", "--atomic-claims"]
    if lane == "L3":
        # L1 + reranker
        return base + [
            "--enable-nli",
            "--no-concat-premises",
            "--no-atomic-claims",
            "--reranker-model",
            "BAAI/bge-reranker-v2-m3",
        ]
    if lane == "L4":
        # NLI + reranker + atomic + concat, top-k=3 (full NLI stack, no SE)
        return base + [
            "--enable-nli",
            "--concat-premises",
            "--atomic-claims",
            "--reranker-model",
            "BAAI/bge-reranker-v2-m3",
        ]
    if lane == "L5":
        # NLI + reranker + atomic + concat, top-k=1 (cheap NLI variant)
        # Top-k is controlled via env var rather than CLI in the harness.
        return base + [
            "--enable-nli",
            "--concat-premises",
            "--atomic-claims",
            "--reranker-model",
            "BAAI/bge-reranker-v2-m3",
        ]
    if lane == "L6":
        # L1 + semantic entropy
        return base + [
            "--enable-nli",
            "--no-concat-premises",
            "--no-atomic-claims",
            "--enable-semantic-entropy",
        ]
    if lane == "L7":
        # L4 + semantic entropy (full quality stack)
        return base + [
            "--enable-nli",
            "--concat-premises",
            "--atomic-claims",
            "--reranker-model",
            "BAAI/bge-reranker-v2-m3",
            "--enable-semantic-entropy",
        ]
    raise ValueError("unknown lane: {0}".format(lane))


def _lane_env(lane: str) -> Dict[str, str]:
    """Per-lane environment overrides (top-k mostly)."""

    if lane == "L5":
        return {"VOYAGER_GROUNDEDNESS_NLI_TOP_K": "1"}
    return {"VOYAGER_GROUNDEDNESS_NLI_TOP_K": "3"}


def _set_env_for_lane(env: Dict[str, str], previous: Dict[str, Optional[str]]) -> None:
    """Apply env overrides, snapshotting previous values for restoration."""

    for key, value in env.items():
        previous[key] = os.environ.get(key)
        os.environ[key] = value


def _restore_env(previous: Dict[str, Optional[str]]) -> None:
    for key, prior in previous.items():
        if prior is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = prior


def _gpu_state(label: str) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        return {"label": label, "available": False}
    torch.cuda.synchronize()
    return {
        "label": label,
        "available": True,
        "allocated_mb": float(torch.cuda.memory_allocated() / (1024 ** 2)),
        "reserved_mb": float(torch.cuda.memory_reserved() / (1024 ** 2)),
        "peak_allocated_mb": float(torch.cuda.max_memory_allocated() / (1024 ** 2)),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="8-lane Pareto profile sweep")
    parser.add_argument("--pairs-per-stratum", type=int, default=20)
    parser.add_argument("--max-external-per-stratum", type=int, default=20)
    parser.add_argument(
        "--model",
        default=os.environ.get(
            "VOYAGER_GROUNDEDNESS_MODEL",
            "VAGOsolutions/SauerkrautLM-Multi-Reason-ModernColBERT",
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_HERE / "reports",
    )
    parser.add_argument(
        "--lanes",
        nargs="+",
        default=["L0", "L1", "L2", "L3", "L4", "L5", "L6", "L7"],
    )
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    base = [
        "--model", args.model,
        "--pairs-per-stratum", str(args.pairs_per_stratum),
        "--max-external-per-stratum", str(args.max_external_per_stratum),
    ]

    summary: Dict[str, Any] = {
        "model": args.model,
        "pairs_per_stratum": args.pairs_per_stratum,
        "max_external_per_stratum": args.max_external_per_stratum,
        "lanes": [],
    }

    for lane in args.lanes:
        out = args.out_dir / "profile_sweep_{0}.json".format(lane)
        argv_lane = _lane_argv(lane, base) + ["--out", str(out)]
        env_overrides = _lane_env(lane)
        previous: Dict[str, Optional[str]] = {}
        _set_env_for_lane(env_overrides, previous)
        # Reset VRAM peak counter so each lane reports its own footprint
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        baseline = _gpu_state("baseline_{0}".format(lane))
        t0 = time.perf_counter()
        try:
            harness.main(argv_lane)
            error: Optional[str] = None
        except SystemExit as exc:
            error = "SystemExit: {0}".format(exc)
        except Exception as exc:  # noqa: BLE001
            error = "{0}: {1}".format(type(exc).__name__, exc)
        elapsed = time.perf_counter() - t0
        peak = _gpu_state("peak_{0}".format(lane))
        _restore_env(previous)

        # Reload the JSON to read the saved metrics
        report: Dict[str, Any] = {}
        if out.exists():
            try:
                report = json.loads(out.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                error = error or "report_read_failed: {0}".format(exc)

        # Augment the on-disk report with VRAM and timing
        report.setdefault("vram", {})
        report["vram"]["baseline"] = baseline
        report["vram"]["peak"] = peak
        report["wall_time_s"] = elapsed
        report["lane"] = lane
        if error is not None:
            report["error"] = error
        out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

        # Summary row
        mp = report.get("minimal_pairs", {})
        per_stratum = mp.get("per_stratum", {})
        # Macro accuracy across strata (paired-ranking accuracy)
        macro_internal = (
            sum(s.get("paired_accuracy", 0.0) for s in per_stratum.values())
            / max(len(per_stratum), 1)
        )
        # Macro F1 across all external strata
        ext_f1s: List[float] = []
        ext_per_stratum: Dict[str, float] = {}
        for ext_name, ext in (report.get("external") or {}).items():
            if not ext:
                continue
            for stratum, row in (ext.get("per_stratum") or {}).items():
                ext_f1s.append(row.get("f1", 0.0))
                ext_per_stratum["{0}.{1}".format(ext_name, stratum)] = row.get("f1", 0.0)
        macro_external = sum(ext_f1s) / max(len(ext_f1s), 1) if ext_f1s else 0.0
        combined_f1 = 0.4 * macro_internal + 0.6 * macro_external
        # Per-stratum breakdown for the German lane (interesting in itself)
        de_acc = per_stratum.get("de_minimal_pairs", {}).get("paired_accuracy")

        summary["lanes"].append({
            "lane": lane,
            "argv": argv_lane,
            "env": env_overrides,
            "macro_internal_paired_accuracy": float(macro_internal),
            "macro_external_f1": float(macro_external),
            "combined_score": float(combined_f1),
            "de_minimal_pairs_paired_accuracy": de_acc,
            "encode_p95_ms": mp.get("encode_p95_ms"),
            "score_p95_ms": mp.get("score_p95_ms"),
            "latency_p95_ms": mp.get("latency_p95_ms"),
            "vram_peak_allocated_mb": peak.get("peak_allocated_mb"),
            "vram_peak_reserved_mb": peak.get("reserved_mb"),
            "wall_time_s": float(elapsed),
            "headline_verdict": report.get("headline_verdict"),
            "report_path": str(out),
            "external_per_stratum_f1": ext_per_stratum,
            "internal_per_stratum": {
                k: v.get("paired_accuracy") for k, v in per_stratum.items()
            },
            "error": error,
        })

    summary_path = args.out_dir / "profile_sweep_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "summary_path": str(summary_path),
        "lanes": [
            {
                "lane": row["lane"],
                "combined_score": row["combined_score"],
                "macro_internal": row["macro_internal_paired_accuracy"],
                "macro_external_f1": row["macro_external_f1"],
                "latency_p95_ms": row["latency_p95_ms"],
                "vram_peak_mb": row["vram_peak_allocated_mb"],
                "wall_time_s": row["wall_time_s"],
                "error": row["error"],
            }
            for row in summary["lanes"]
        ],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
