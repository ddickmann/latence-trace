"""PA3: profile-driven hot-path identification.

Runs ``cProfile`` against a representative groundedness scoring loop
(deterministic encoder, no GPU, 100 iterations at the ``balanced``
profile shape) and dumps the top sample-attributed Python functions to
``research/triangular_maxsim/reports/hot_path_profile.json``.

Usage:
    python scripts/profile_hot_paths.py [--iterations 100] [--profile balanced]

The output JSON drives the PA3 decision rules in
``docs/perf/hot_path_decisions.md``:

- p95 contribution < 2 ms : leave alone
- 2-5 ms and Python/numpy : optimize in-place
- > 5 ms and CPU-bound    : evaluate Rust PyO3 extension

This script is intentionally cheap (a few seconds on CPU) so it can
re-run on every release as a regression guard against silent perf
regressions.
"""

from __future__ import annotations

import argparse
import cProfile
import json
import os
import pstats
import re
import sys
import time
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

# Make the project root importable when run via "python scripts/...".
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))

from latence_trace.api.models import GroundednessRequest  # noqa: E402
from latence_trace.api.service import GroundednessService  # noqa: E402

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


class _DeterministicEncoder:
    """Tiny encoder so the profile is dominated by orchestration, not GPU calls."""

    model_name = "profile-stub"
    dim = 64
    _seed = 13

    def _vec(self, token: str) -> np.ndarray:
        h = abs(hash((self._seed, token.lower()))) % (2**31 - 1)
        rng = np.random.default_rng(h)
        v = rng.standard_normal(self.dim).astype(np.float32)
        n = float(np.linalg.norm(v))
        return v / n if n > 1e-9 else v

    def tokenize(self, text):
        return [t for t in _TOKEN_RE.findall(text) if t.strip()]

    def encode(self, texts, **_kwargs):
        if isinstance(texts, str):
            texts = [texts]
        out = []
        for text in texts:
            tokens = self.tokenize(text)
            if not tokens:
                out.append(np.zeros((1, self.dim), dtype=np.float32))
                continue
            out.append(np.stack([self._vec(t) for t in tokens], axis=0))
        return out


_REPRESENTATIVE_CONTEXT = (
    "Teardrops is a single by George Harrison, released on 20 July 1981 in the United States. "
    "The track reached number eight on the Billboard Hot 100 chart and was re-issued in 1989. "
    "Critics described it as a return to form for Harrison after his late-1970s catalog dip."
)
_REPRESENTATIVE_RESPONSE = (
    "Teardrops, a single by George Harrison, was released on 20 July 1981 in the US and "
    "reached number eight on the Billboard Hot 100 chart."
)
_REPRESENTATIVE_QUERY = "When was Teardrops released and where did it chart?"


def _run_iterations(service: GroundednessService, n: int) -> float:
    """Drive the service for ``n`` requests, returning total wall ms."""

    request = GroundednessRequest(
        raw_context=_REPRESENTATIVE_CONTEXT,
        response_text=_REPRESENTATIVE_RESPONSE,
        query_text=_REPRESENTATIVE_QUERY,
    )
    start = time.perf_counter()
    for _ in range(n):
        service.groundedness(request)
    return (time.perf_counter() - start) * 1000.0


def _summarize(profile: cProfile.Profile, top_k: int = 30) -> List[Dict[str, Any]]:
    buf = StringIO()
    stats = pstats.Stats(profile, stream=buf).sort_stats("cumulative")
    rows: List[Dict[str, Any]] = []
    rendered = stats.stats  # type: ignore[attr-defined]
    sorted_items = sorted(
        rendered.items(),
        key=lambda kv: kv[1][3],  # cumulative time
        reverse=True,
    )
    for func, (cc, nc, tt, ct, _callers) in sorted_items[:top_k]:
        filename, lineno, name = func
        if not isinstance(filename, str):
            continue
        rel = filename
        marker = "/latence_trace/"
        if marker in filename:
            rel = filename[filename.index(marker) + 1 :]
        rows.append(
            {
                "function": name,
                "file": rel,
                "line": lineno,
                "ncalls": int(nc),
                "tottime_ms": round(tt * 1000.0, 3),
                "cumtime_ms": round(ct * 1000.0, 3),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--profile", default="balanced")
    parser.add_argument(
        "--output",
        default=str(
            Path(__file__).resolve().parent.parent
            / "research"
            / "triangular_maxsim"
            / "reports"
            / "hot_path_profile.json"
        ),
    )
    args = parser.parse_args()

    # Disable NLI for the orchestration-only profile - we want to see
    # the latence-trace Python overhead, not HF transformers.
    os.environ.setdefault("VOYAGER_GROUNDEDNESS_NLI_ENABLED", "0")

    encoder = _DeterministicEncoder()
    service = GroundednessService(encoder_factory=lambda _name: encoder)

    # Warm one call so module-level caches (regex, spacy lazy load, etc)
    # do not dominate the profile.
    _run_iterations(service, 1)

    pr = cProfile.Profile()
    pr.enable()
    total_ms = _run_iterations(service, args.iterations)
    pr.disable()

    summary = {
        "schema_version": 1,
        "profile": args.profile,
        "iterations": args.iterations,
        "total_ms": round(total_ms, 2),
        "mean_per_request_ms": round(total_ms / args.iterations, 3),
        "top_functions": _summarize(pr, top_k=40),
        "encoder": "deterministic-stub",
        "torch_version": torch.__version__,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_path} ({summary['mean_per_request_ms']} ms / request)")


if __name__ == "__main__":
    main()
