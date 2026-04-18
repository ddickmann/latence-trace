"""Response-chunking microbench: short response parity + long-response scaling.

Runs three regimes against the same support corpus with the deterministic
stub encoder and reports per-request wall-clock latency:

1. ``short / 1 chunk``       - response fits in one window, chunking inert.
2. ``long / multi chunk``    - response sentence-packed into ~10 windows.
3. ``very_long / multi``     - response sentence-packed into ~40 windows.

The bench prints a table the response-chunking PR can paste into
``docs/perf/response_chunking_bench.md`` so reviewers can verify the
"short response: no perf regression" claim and see the linear-ish scaling
for long responses (per-chunk work is bounded by the support tensor and
each chunk does an independent matmul).

Pure CPU. No HF downloads. Runs in a few seconds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))

from latence_trace.core.groundedness import (  # noqa: E402
    SupportUnitInput,
    _build_response_chunks,
    encode_texts,
    partition_support_units,
    score_groundedness_response_chunked,
    segment_text,
    tokenize_text,
)


_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


class StubProvider:
    """Same deterministic stub used by the parity tests."""

    model_name = "stub-colbert"
    doc_maxlen = 64

    def __init__(self, dim: int = 16) -> None:
        self.dim = dim

    def tokenize(self, text: str, is_query: bool = False) -> List[str]:
        return _TOKEN_RE.findall(text)

    def encoded_token_count(self, text: str, is_query: bool = False) -> int:
        return len(self.tokenize(text))

    def _vec(self, token: str) -> np.ndarray:
        digest = hashlib.sha1(token.lower().encode("utf-8")).digest()
        v = np.zeros((self.dim,), dtype=np.float32)
        v[digest[0] % self.dim] = 1.0
        v[digest[1] % self.dim] = max(v[digest[1] % self.dim], 0.5)
        norm = np.linalg.norm(v) + 1e-8
        return v / norm

    def encode(self, inputs, **_kwargs):
        texts = [inputs] if isinstance(inputs, str) else list(inputs)
        return [
            np.stack([self._vec(t) for t in (self.tokenize(text) or ["<empty>"])]).astype(np.float32)
            for text in texts
        ]


def _make_support_units(provider: StubProvider, raw_context: str) -> List[SupportUnitInput]:
    segments = segment_text(
        raw_context,
        "sentence_packed",
        provider=provider,
        chunk_token_budget=32,
    )
    embeddings = encode_texts(provider, [s["text"] for s in segments], is_query=False)
    units: List[SupportUnitInput] = []
    for idx, (segment, emb) in enumerate(zip(segments, embeddings)):
        tokens = tokenize_text(provider, segment["text"], expected_len=int(emb.shape[0]))
        units.append(
            SupportUnitInput(
                support_id=f"raw-{idx}",
                text=segment["text"],
                embeddings=emb,
                tokens=tokens,
                source_mode="raw_context",
                offset_start=int(segment["offset_start"]),
                offset_end=int(segment["offset_end"]),
            )
        )
    return units


def _bench_one(
    label: str,
    *,
    provider: StubProvider,
    response_text: str,
    chunk_budget: int,
    support_units: List[SupportUnitInput],
    iterations: int,
) -> Dict[str, object]:
    chunks = _build_response_chunks(
        response_text,
        provider=provider,
        chunk_token_budget=chunk_budget,
        encode_fn=encode_texts,
    )
    support_batches = partition_support_units(support_units, batch_size=4)

    # Warm-up so JIT / lazy imports do not pollute the timed loop.
    score_groundedness_response_chunked(
        response_chunks=chunks,
        support_batches=support_batches,
        response_text=response_text,
        evidence_limit=8,
        primary_metric="reverse_context",
    )

    latencies_ms: List[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        score_groundedness_response_chunked(
            response_chunks=chunks,
            support_batches=support_batches,
            response_text=response_text,
            evidence_limit=8,
            primary_metric="reverse_context",
        )
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

    response_token_total = sum(int(c.embeddings.shape[0]) for c in chunks)
    latencies_ms.sort()
    return {
        "label": label,
        "response_token_total": response_token_total,
        "chunk_count": len(chunks),
        "support_unit_count": len(support_units),
        "iterations": iterations,
        "p50_ms": float(statistics.median(latencies_ms)),
        "p95_ms": float(latencies_ms[max(0, int(0.95 * len(latencies_ms)) - 1)]),
        "mean_ms": float(statistics.fmean(latencies_ms)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=200, help="Per-regime iteration count.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent.parent / "docs" / "perf" / "response_chunking_bench.json",
        help="JSON output path (default: docs/perf/response_chunking_bench.json).",
    )
    args = parser.parse_args()

    provider = StubProvider()
    raw_context_pieces = [
        "Berlin became the German capital in 1990.",
        "The Bundesbank is headquartered in Frankfurt.",
        "Paris is the French capital and home of the Eiffel Tower.",
        "Rome is the Italian capital and home of the Colosseum.",
        "Madrid is the Spanish capital.",
        "London is the British capital.",
        "Vienna is the Austrian capital.",
        "Bern is the Swiss capital.",
    ]
    raw_context = " ".join(raw_context_pieces)
    support_units = _make_support_units(provider, raw_context)

    short_response = " ".join(raw_context_pieces[:2])
    long_response = " ".join(raw_context_pieces * 5)
    very_long_response = " ".join(raw_context_pieces * 40)

    regimes = [
        # short_512 forces single-chunk fast path; short_256 uses the API
        # default and should also stay single-chunk.
        _bench_one(
            "short / 1 chunk (budget 512)",
            provider=provider,
            response_text=short_response,
            chunk_budget=512,
            support_units=support_units,
            iterations=args.iterations,
        ),
        _bench_one(
            "short / 1 chunk (budget 256, default)",
            provider=provider,
            response_text=short_response,
            chunk_budget=256,
            support_units=support_units,
            iterations=args.iterations,
        ),
        _bench_one(
            "long / ~10 chunks (budget 32)",
            provider=provider,
            response_text=long_response,
            chunk_budget=32,
            support_units=support_units,
            iterations=args.iterations,
        ),
        _bench_one(
            "very_long / ~40 chunks (budget 32)",
            provider=provider,
            response_text=very_long_response,
            chunk_budget=32,
            support_units=support_units,
            iterations=args.iterations,
        ),
    ]

    print("\n=== response chunking bench ===")
    print(
        f"{'regime':<48}{'tokens':>8}{'chunks':>8}{'p50_ms':>10}{'p95_ms':>10}{'mean_ms':>10}"
    )
    for r in regimes:
        print(
            f"{r['label']:<48}{r['response_token_total']:>8}{r['chunk_count']:>8}"
            f"{r['p50_ms']:>10.3f}{r['p95_ms']:>10.3f}{r['mean_ms']:>10.3f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"regimes": regimes}, indent=2) + "\n")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
