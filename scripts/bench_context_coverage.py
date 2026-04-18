"""Real-data benchmark for the context-coverage observability metric.

Loads ~150 HaluEval-QA samples from
``$VOYAGER_GROUNDEDNESS_HALUEVAL_DIR/qa_data.jsonl`` and runs each one
through the in-process groundedness scoring service twice — once with
the supplied "right" answer (grounded), once with the "hallucinated"
answer (not grounded). For every sample we record:

- ``context_coverage_ratio`` : the headline retrieval-efficiency metric
- ``context_attribution_ratio`` : the stricter argmax-based ratio
- ``support_units_total`` and ``support_units_used``

The benchmark proves, on real text:

1. **The two signals are well-defined on real data.** Both
   ``context_coverage_ratio`` and ``context_attribution_ratio`` stay
   inside ``[0, 1]`` for every sample.
2. **The two signals are independent observability axes.** Coverage
   answers "did the response have a strong match anywhere in this
   support unit?" (absolute strength). Attribution answers "did this
   unit win the argmax for at least one response token?" (competitive
   placement against sibling units). We do **not** claim a strict
   ordering between them — a unit can lose every argmax (low
   attribution) yet have one strong match (high coverage), and the
   reverse can hold under tight thresholds. The bench reports the
   distribution of both so operators can pick the one that maps to
   their retrieval-tuning question.
3. **The metric is cheap.** We report the median per-sample wall time so
   operators can see the coverage compute is dwarfed by encoder /
   scorer time.

The script is intentionally small (no NLI, no real ColBERT) because it
is a *contract* check on the coverage observability metric — not a
quality benchmark for the scoring engine itself. The HF benchmarks
(see ``research/triangular_maxsim/groundedness_external_eval.py``) are
the authoritative source of quality numbers.

Usage:
    python scripts/bench_context_coverage.py --samples 150 --output \
        research/triangular_maxsim/reports/coverage_real_data.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))

from latence_trace.api.models import GroundednessRequest  # noqa: E402
from latence_trace.api.service import GroundednessService  # noqa: E402


_TOKEN_RE = __import__("re").compile(r"\w+|[^\w\s]", __import__("re").UNICODE)


class _DeterministicEncoder:
    """Tiny CPU-only ColBERT-style encoder for harness use.

    Identical to the one in ``scripts/profile_hot_paths.py`` so behaviour
    here matches the perf bench. Dim 64 + L2-normalised so the resulting
    similarities live in roughly ``[-1, 1]`` and the documented 0.5
    coverage threshold is meaningful.
    """

    model_name = "coverage-bench-stub"
    dim = 64
    _seed = 17

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
        out: List[np.ndarray] = []
        for text in texts:
            tokens = self.tokenize(text)
            if not tokens:
                out.append(np.zeros((1, self.dim), dtype=np.float32))
                continue
            out.append(np.stack([self._vec(t) for t in tokens], axis=0))
        return out


def _load_halueval_qa(limit: int) -> List[Dict[str, Any]]:
    path = Path(os.environ["VOYAGER_GROUNDEDNESS_HALUEVAL_DIR"]) / "qa_data.jsonl"
    samples: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
            if len(samples) >= limit:
                break
    return samples


def _score(
    service: GroundednessService,
    knowledge: str,
    response: str,
    query: str,
    *,
    raw_context_chunk_tokens: int = 256,
) -> Dict[str, Any]:
    req = GroundednessRequest(
        raw_context=knowledge,
        response_text=response,
        query_text=query,
        raw_context_chunk_tokens=raw_context_chunk_tokens,
    )
    t0 = time.perf_counter()
    out = service.groundedness(req)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    scores = out.scores.model_dump()
    return {
        "context_coverage_ratio": scores.get("context_coverage_ratio"),
        "context_attribution_ratio": scores.get("context_attribution_ratio"),
        "support_units_total": scores.get("support_units_total"),
        "support_units_used": scores.get("support_units_used"),
        "elapsed_ms": elapsed_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=150)
    parser.add_argument(
        "--output",
        default=str(
            _HERE.parent.parent
            / "research"
            / "triangular_maxsim"
            / "reports"
            / "coverage_real_data.json"
        ),
    )
    args = parser.parse_args()

    raw_samples = _load_halueval_qa(args.samples)
    if not raw_samples:
        raise SystemExit("HaluEval QA data is empty / missing")

    encoder = _DeterministicEncoder()
    service = GroundednessService(encoder_factory=lambda _name=None: encoder)

    grounded_records: List[Dict[str, Any]] = []
    hallucinated_records: List[Dict[str, Any]] = []
    overfetch_records: List[Dict[str, Any]] = []
    range_failures: List[Dict[str, Any]] = []

    for sample in raw_samples:
        knowledge = sample.get("knowledge", "")
        question = sample.get("question", "")
        right = sample.get("right_answer") or ""
        wrong = sample.get("hallucinated_answer") or ""
        if not knowledge or not right or not wrong:
            continue

        grounded = _score(service, knowledge, right, question)
        wrong_record = _score(service, knowledge, wrong, question)
        grounded_records.append(grounded)
        hallucinated_records.append(wrong_record)

        for tag, record in (("grounded", grounded), ("hallucinated", wrong_record)):
            cov = record.get("context_coverage_ratio") or 0.0
            attr = record.get("context_attribution_ratio") or 0.0
            if not (0.0 - 1e-6 <= cov <= 1.0 + 1e-6) or not (
                0.0 - 1e-6 <= attr <= 1.0 + 1e-6
            ):
                range_failures.append(
                    {
                        "tag": tag,
                        "question": question,
                        "coverage": cov,
                        "attribution": attr,
                    }
                )

    # ------------------------------------------------------------------
    # Over-fetch scenario: simulate a noisy retriever that returns the
    # right context plus 4 unrelated chunks. We expect coverage_ratio to
    # drop sharply because only one of the 5 retrieved chunks is used.
    # ------------------------------------------------------------------
    overfetch_pairs = min(40, len(raw_samples) // 5)
    if overfetch_pairs > 0:
        for i in range(overfetch_pairs):
            primary = raw_samples[i]
            distractors = [raw_samples[(i + 1 + d) % len(raw_samples)] for d in range(4)]
            knowledge_chunks = [primary["knowledge"]] + [d["knowledge"] for d in distractors]
            joined_knowledge = "\n\n".join(knowledge_chunks)
            question = primary.get("question", "")
            right = primary.get("right_answer") or ""
            if not right:
                continue
            # Force one chunk per knowledge fragment so the over-fetch
            # signal isn't washed out by sentence packing.
            record = _score(
                service,
                joined_knowledge,
                right,
                question,
                raw_context_chunk_tokens=32,
            )
            overfetch_records.append(record)
            cov = record.get("context_coverage_ratio") or 0.0
            attr = record.get("context_attribution_ratio") or 0.0
            if not (0.0 - 1e-6 <= cov <= 1.0 + 1e-6) or not (
                0.0 - 1e-6 <= attr <= 1.0 + 1e-6
            ):
                range_failures.append(
                    {
                        "tag": "overfetch",
                        "question": question,
                        "coverage": cov,
                        "attribution": attr,
                    }
                )

    def _agg(records: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
        values = [r[key] for r in records if r.get(key) is not None]
        if not values:
            return {"n": 0}
        return {
            "n": len(values),
            "mean": float(statistics.fmean(values)),
            "median": float(statistics.median(values)),
            "p25": float(np.percentile(values, 25)),
            "p75": float(np.percentile(values, 75)),
            "min": float(min(values)),
            "max": float(max(values)),
        }

    elapsed_all = [
        r["elapsed_ms"]
        for r in grounded_records + hallucinated_records + overfetch_records
    ]
    report = {
        "samples_total": len(raw_samples),
        "samples_scored": len(grounded_records),
        "overfetch_samples_scored": len(overfetch_records),
        "encoder": encoder.model_name,
        "grounded": {
            "context_coverage_ratio": _agg(grounded_records, "context_coverage_ratio"),
            "context_attribution_ratio": _agg(grounded_records, "context_attribution_ratio"),
        },
        "hallucinated": {
            "context_coverage_ratio": _agg(hallucinated_records, "context_coverage_ratio"),
            "context_attribution_ratio": _agg(hallucinated_records, "context_attribution_ratio"),
        },
        "overfetch": {
            "context_coverage_ratio": _agg(overfetch_records, "context_coverage_ratio"),
            "context_attribution_ratio": _agg(overfetch_records, "context_attribution_ratio"),
            "support_units_total": _agg(overfetch_records, "support_units_total"),
            "support_units_used": _agg(overfetch_records, "support_units_used"),
        },
        "latency_ms_per_request": {
            "median": float(statistics.median(elapsed_all)),
            "p95": float(np.percentile(elapsed_all, 95)),
            "p99": float(np.percentile(elapsed_all, 99)),
            "mean": float(statistics.fmean(elapsed_all)),
        },
        "range_check": {
            "rule": "both context_coverage_ratio and context_attribution_ratio in [0, 1] for every sample",
            "violations": len(range_failures),
            "violation_examples": range_failures[:3],
        },
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("== Context-coverage real-data benchmark ==")
    print(f"  HaluEval QA samples scored : {report['samples_scored']}")
    print()
    print("  Grounded responses (right_answer):")
    g = report["grounded"]
    print(
        f"    coverage_ratio       median={g['context_coverage_ratio']['median']:.3f} "
        f"mean={g['context_coverage_ratio']['mean']:.3f}"
    )
    print(
        f"    attribution_ratio    median={g['context_attribution_ratio']['median']:.3f} "
        f"mean={g['context_attribution_ratio']['mean']:.3f}"
    )
    print()
    print("  Hallucinated responses:")
    h = report["hallucinated"]
    print(
        f"    coverage_ratio       median={h['context_coverage_ratio']['median']:.3f} "
        f"mean={h['context_coverage_ratio']['mean']:.3f}"
    )
    print(
        f"    attribution_ratio    median={h['context_attribution_ratio']['median']:.3f} "
        f"mean={h['context_attribution_ratio']['mean']:.3f}"
    )
    print()
    delta = (
        g["context_coverage_ratio"]["mean"]
        - h["context_coverage_ratio"]["mean"]
    )
    print(f"  Coverage gap (grounded - hallucinated, mean): {delta:+.3f}")
    print()
    print("  Over-fetch retriever scenario (1 relevant + 4 distractor chunks):")
    of = report["overfetch"]
    if of["context_coverage_ratio"]["n"] > 0:
        print(
            f"    coverage_ratio       median={of['context_coverage_ratio']['median']:.3f} "
            f"mean={of['context_coverage_ratio']['mean']:.3f}  "
            f"(target ~ 0.20 - one of five chunks used)"
        )
        print(
            f"    attribution_ratio    median={of['context_attribution_ratio']['median']:.3f} "
            f"mean={of['context_attribution_ratio']['mean']:.3f}"
        )
        print(
            f"    support_units_used   median={of['support_units_used']['median']:.1f} / "
            f"{of['support_units_total']['median']:.1f}"
        )
        print()
    rng = report["range_check"]
    print(
        f"  Range invariant 0 <= ratio <= 1     : "
        f"violations = {rng['violations']} / {2 * report['samples_scored']}"
    )
    print()
    print(f"  Median latency / request : {report['latency_ms_per_request']['median']:.2f} ms")
    print(f"  Report written to        : {out_path}")


if __name__ == "__main__":
    main()
