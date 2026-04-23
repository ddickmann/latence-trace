"""Re-score the transcripts_v2 bank with the production ``GPUScorer``.

The full ``coding_agent_validation`` runner persists a simplified score
dict that lacks ``per_token_stats`` (required for the composite phantom
guard). This helper loads the pre-encoded shared cache, runs only the
slim GPU scorer per case/tier, and writes a synthetic "runner-style"
JSON payload that includes every signal the composite needs:

- ``reverse_context``
- ``per_token_stats.p5|p10|min``
- ``literal_guarded`` (fraction of response literals matched in context)
- per-support-unit ``max_score`` and ``usage_state`` (for file attribution)

Output
------
``artifacts/v2_gpu_scorer_rescore.{json,md}``.

CLI
---

    python -m research.triangular_maxsim.coding.experiments.rescore_v2_with_gpu_scorer \
        --cache research/triangular_maxsim/coding/cache/<v2-sentence_packed>.pt \
        --device cuda
"""
from __future__ import annotations

import argparse
import json
import pickle
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch

sys.path.insert(0, "/workspace/latence-trace")

from research.triangular_maxsim.coding.code_segmenter import (  # noqa: E402
    render_context_files,
)
from research.triangular_maxsim.coding.coding_agent_validation import (  # noqa: E402,F401
    ResponseChunkInput,
    SharedEncodedCase,
    SupportUnitInput,
)
from research.triangular_maxsim.coding.experiments.file_attribution import (  # noqa: E402
    _file_usage_from_units,
)
from research.triangular_maxsim.coding.experiments.gpu_scorer import GPUScorer  # noqa: E402


def _build_file_boundaries(context_files) -> List[Dict[str, Any]]:
    """Return a list of ``{path, start, end}`` file spans in ``raw_context``.

    ``end`` is exclusive. The last entry covers up to the end of the
    rendered string so any trailing chunk still resolves.
    """
    _raw, rendered = render_context_files(context_files)
    spans: List[Dict[str, Any]] = []
    for i, entry in enumerate(rendered):
        start = int(entry.render_offset_start)
        if i + 1 < len(rendered):
            end = int(rendered[i + 1].render_offset_start)
        else:
            end = int(entry.code_offset_start) + len(entry.content)
        spans.append({"path": entry.path, "start": start, "end": end})
    return spans


def _path_for_offset(
    offset_start: int, offset_end: int, spans: Sequence[Dict[str, Any]]
) -> str:
    """Pick the file whose span contains the chunk midpoint. Ties break to
    the earlier file so header-only chunks attribute to the right file."""
    if not spans:
        return ""
    mid = (int(offset_start) + int(offset_end)) // 2 if offset_end else int(offset_start)
    for span in spans:
        if span["start"] <= mid < span["end"]:
            return span["path"]
    if mid >= spans[-1]["end"]:
        return spans[-1]["path"]
    return spans[0]["path"]


_ARTIFACTS = Path(
    "/workspace/latence-trace/research/triangular_maxsim/coding/artifacts"
)
_DEFAULT_CACHE = Path(
    "/workspace/latence-trace/research/triangular_maxsim/coding/cache/"
    "coding_shared_v5__sentence_packed__dual__cp_gte__bank_transcripts_v2__"
    "sw15__ctx256__resp256__pilot20.pt"
)


def _flatten_response(response_chunks) -> (List[str], torch.Tensor):
    tokens: List[str] = []
    embs: List[torch.Tensor] = []
    for chunk in response_chunks:
        tokens.extend(chunk.tokens)
        embs.append(chunk.embeddings)
    if not embs:
        raise ValueError("Empty response chunks")
    return tokens, torch.cat(embs, dim=0)


def _extract_literal_tokens(response_tokens: Sequence[str]) -> List[str]:
    """Treat alphanumeric-with-underscore tokens of length >= 3 as
    identifier-like and keep them as the literal set. This matches the
    production ``extract_literals`` intent without pulling in the full
    tokenizer module."""
    keep: List[str] = []
    for t in response_tokens:
        cleaned = t.strip().strip("Ġ ")
        if len(cleaned) < 3:
            continue
        if not any(ch.isalnum() or ch == "_" for ch in cleaned):
            continue
        keep.append(cleaned)
    return keep


def rescore_cache(
    cache_path: Path,
    *,
    device: str,
    encoder_key: str,
) -> Dict[str, Any]:
    with cache_path.open("rb") as handle:
        payload = pickle.load(handle)
    cases = payload["encoded_cases"]
    if not cases:
        raise RuntimeError(f"Cache {cache_path} has no encoded cases")

    tdevice = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
    scorer = GPUScorer(device=str(tdevice))

    rows: List[Dict[str, Any]] = []
    scorer_latency_ms: List[float] = []
    per_tier_rc: Dict[str, List[float]] = defaultdict(list)
    per_tier_p10: Dict[str, List[float]] = defaultdict(list)

    t_overall = time.perf_counter()
    for encoded in cases:
        tier = encoded.case.subcategory or encoded.case.label or "correct"
        if tier not in ("correct", "wrong", "ambiguous"):
            tier = "correct"

        support_units = encoded.support_units_by_encoder.get(encoder_key) or []
        response_chunks = encoded.response_chunks_by_encoder.get(encoder_key) or []
        if not support_units or not response_chunks:
            continue

        response_tokens, response_emb = _flatten_response(response_chunks)

        def _unit_tokens(u: SupportUnitInput) -> List[str]:
            return list(u.tokens or [])

        units_for_scorer = [
            (_unit_tokens(u), u.embeddings) for u in support_units
        ]

        literals = _extract_literal_tokens(response_tokens)

        out = scorer(
            response_tokens=response_tokens,
            response_embeddings=response_emb,
            support_units=units_for_scorer,
            literal_tokens=literals,
        )

        scorer_latency_ms.append(out["latency_ms"])

        # File-level attribution: resolve path from offsets for chunkers
        # (sentence_packed) that do not annotate metadata['path'] directly.
        file_spans = _build_file_boundaries(encoded.case.context_files)
        owner_counts = out.get("per_unit_owner_count") or [0] * len(support_units)
        unit_records: List[Dict[str, Any]] = []
        for i, (support_unit, state, score) in enumerate(
            zip(support_units, out["usage_states"], out["per_unit_max"])
        ):
            meta = dict(support_unit.metadata or {})
            path = meta.get("path") or meta.get("source_path") or support_unit.source_id
            if not path:
                # Fall back to offsets from SharedEncodedCase.support_metadata,
                # which retains the underscore-prefixed keys that the public
                # SupportUnitInput.metadata strips.
                raw_meta = (
                    encoded.support_metadata[i]
                    if i < len(encoded.support_metadata)
                    else {}
                )
                offset_start = raw_meta.get("_offset_start")
                offset_end = raw_meta.get("_offset_end", offset_start)
                if offset_start is not None:
                    path = _path_for_offset(
                        int(offset_start),
                        int(offset_end) if offset_end is not None else int(offset_start),
                        file_spans,
                    )
            unit_records.append(
                {
                    "support_id": support_unit.support_id,
                    "chunk_id": support_unit.chunk_id,
                    "usage_state": state,
                    "score": float(score),
                    "owner_count": int(owner_counts[i]) if i < len(owner_counts) else 0,
                    "metadata": {"path": str(path) if path else support_unit.chunk_id},
                }
            )
        attribution = _file_usage_from_units(
            unit_records,
            dead_weight_threshold=0.20,
            total_response_tokens=int(out["n_tokens"]),
            min_owner_share=0.01,
        )
        # Also expose per-unit records so downstream analyses (e.g., sweeping
        # dead-weight thresholds) do not have to re-derive them.
        attribution["unit_records"] = unit_records

        rows.append(
            {
                "id": encoded.case.id,
                "base_scenario_id": getattr(encoded.case, "base_scenario_id", None)
                or encoded.case.metadata.get("base_scenario_id"),
                "tier": tier,
                "subcategory": tier,
                "label": encoded.case.label,
                "chunker": encoded.chunker,
                "scorer_config": "gte_only_gpu_slim",
                "scores": {
                    "reverse_context": out["reverse_context"],
                    "literal_guarded": out["literal_guard"],
                    "per_token_stats": {
                        "p5": out["per_token_p5"],
                        "p10": out["per_token_p10"],
                        "min": out["per_token_min"],
                    },
                    "n_tokens": out["n_tokens"],
                    "n_support_units": out["n_support_units"],
                    "scorer_latency_ms": out["latency_ms"],
                },
                "file_attribution": attribution,
            }
        )
        per_tier_rc[tier].append(out["reverse_context"])
        per_tier_p10[tier].append(out["per_token_p10"])

    total_s = time.perf_counter() - t_overall

    def _summary(values: Sequence[float]) -> Dict[str, float]:
        if not values:
            return {"n": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0}
        sv = sorted(values)
        return {
            "n": len(values),
            "mean": float(statistics.mean(values)),
            "p50": float(sv[len(sv) // 2]),
            "p95": float(sv[int(0.95 * (len(sv) - 1))]),
        }

    return {
        "cache": str(cache_path),
        "encoder_key": encoder_key,
        "device": str(tdevice),
        "n_cases": len(rows),
        "rows": rows,
        "summary": {
            "tier_rc_mean": {t: statistics.mean(v) for t, v in per_tier_rc.items() if v},
            "tier_p10_mean": {t: statistics.mean(v) for t, v in per_tier_p10.items() if v},
            "scorer_latency_ms": _summary(scorer_latency_ms),
            "total_runtime_s": total_s,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=_DEFAULT_CACHE)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--encoder-key", default="gte")
    parser.add_argument("--out-tag", default="v2_gpu_scorer_rescore")
    args = parser.parse_args()

    payload = rescore_cache(args.cache, device=args.device, encoder_key=args.encoder_key)

    _ARTIFACTS.mkdir(parents=True, exist_ok=True)
    json_path = _ARTIFACTS / f"{args.out_tag}.json"
    md_path = _ARTIFACTS / f"{args.out_tag}.md"
    json_path.write_text(json.dumps(payload, indent=2))

    summary = payload["summary"]
    lines = [
        f"# GPU-slim rescore of v2 bank — `{args.out_tag}`",
        "",
        f"- cache: `{payload['cache']}`",
        f"- encoder key: `{payload['encoder_key']}`",
        f"- device: `{payload['device']}`",
        f"- n cases: `{payload['n_cases']}`",
        f"- total runtime: `{summary['total_runtime_s']:.2f}s`",
        "",
        "## Scorer latency",
        f"- n: `{summary['scorer_latency_ms']['n']}`, mean: `{summary['scorer_latency_ms']['mean']:.2f}ms`, p50: `{summary['scorer_latency_ms']['p50']:.2f}ms`, p95: `{summary['scorer_latency_ms']['p95']:.2f}ms`",
        "",
        "## Mean signals by tier",
        "",
        "| tier | reverse_context | per_token_p10 |",
        "|---|---:|---:|",
    ]
    for tier in ("correct", "ambiguous", "wrong"):
        rc = summary["tier_rc_mean"].get(tier)
        p10 = summary["tier_p10_mean"].get(tier)
        if rc is None or p10 is None:
            continue
        lines.append(f"| {tier} | {rc:.4f} | {p10:.4f} |")
    md_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
