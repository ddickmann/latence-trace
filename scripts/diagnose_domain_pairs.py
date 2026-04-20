#!/usr/bin/env python3
"""Run latence-trace `quality` profile against a paired domain dataset.

Each line of the input JSONL file is a paired sample::

    {
      "id": "...",
      "stratum": "...",
      "context": "...",
      "query": "...",
      "grounded_response":  "...",
      "ungrounded_response": "..."
    }

We score both responses against the same context with the *full*
production stack (response chunking, ColBERT MaxSim, multilingual NLI
peer with cross-encoder reranker, atomic-claim decomposition, fused
``groundedness_v2`` headline). Per-stratum we report:

- Paired ranking accuracy (P[ score(grounded) > score(ungrounded) ]),
  the only metric that is robust to threshold drift on small n.
- Mean grounded / ungrounded score and the delta.
- Risk-band classification accuracy under the shipped
  ``thresholds.quality.json`` (so callers see whether a vanilla deploy
  would route the bad answer to human review).
- Score range to spot saturation.
- Encode + score wall-clock so we can prove the latency budget holds.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

# Make the eval helpers importable.
sys.path.insert(0, "/workspace/latence-trace")
sys.path.insert(0, "/workspace/latence-trace/research/triangular_maxsim")

from research.triangular_maxsim.groundedness_external_eval import (  # noqa: E402
    _encode_pair_side,
    _score_pair_side,
    build_null_bank_embeddings,
)
from latence_trace.core.nli import (  # noqa: E402
    HuggingFaceNLIProvider,
    CrossEncoderPremiseReranker,
)
from latence_trace.core.thresholds import (  # noqa: E402
    classify_risk_band,
    load_risk_band_policy,
)


def _build_provider(model_id: str, device: str = "cuda") -> Any:
    from pylate import models  # noqa: PLC0415

    return models.ColBERT(
        model_name_or_path=model_id,
        device=device,
        do_query_expansion=False,
        trust_remote_code=True,
        model_kwargs={"torch_dtype": torch.bfloat16},
    )


def _classify_band(score: float, policy) -> str:
    """Reproduce the shipped per-band classification (uses the default
    -- i.e. hardest -- stratum threshold so the report mirrors what an
    operator without a custom calibration sees out of the box)."""
    if policy is None:
        return "unknown"
    return classify_risk_band(score, policy=policy)


def _load_samples(path: Path, limit: int | None) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    if limit is not None:
        out = out[:limit]
    return out


def _stratum_summary(rows: List[Dict[str, Any]], threshold_artifact) -> Dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"n_pairs": 0}

    grounded_scores = [row["grounded_score"] for row in rows]
    ungrounded_scores = [row["ungrounded_score"] for row in rows]

    wins = sum(
        1
        for g, u in zip(grounded_scores, ungrounded_scores)
        if g > u
    )
    ties = sum(
        1
        for g, u in zip(grounded_scores, ungrounded_scores)
        if g == u
    )

    grounded_bands = [_classify_band(s, threshold_artifact) for s in grounded_scores]
    ungrounded_bands = [_classify_band(s, threshold_artifact) for s in ungrounded_scores]

    grounded_green = sum(1 for b in grounded_bands if b == "green")
    ungrounded_red_or_amber = sum(
        1 for b in ungrounded_bands if b in {"amber", "red"}
    )

    return {
        "n_pairs": n,
        "paired_accuracy": (wins + ties / 2.0) / float(n),
        "wins_grounded_higher": wins,
        "ties": ties,
        "grounded_mean_score": sum(grounded_scores) / float(n),
        "ungrounded_mean_score": sum(ungrounded_scores) / float(n),
        "delta_mean": (
            sum(grounded_scores) - sum(ungrounded_scores)
        ) / float(n),
        "grounded_score_range": (
            min(grounded_scores),
            max(grounded_scores),
        ),
        "ungrounded_score_range": (
            min(ungrounded_scores),
            max(ungrounded_scores),
        ),
        "grounded_band_distribution": {
            band: grounded_bands.count(band)
            for band in sorted(set(grounded_bands))
        },
        "ungrounded_band_distribution": {
            band: ungrounded_bands.count(band)
            for band in sorted(set(ungrounded_bands))
        },
        "grounded_green_rate": grounded_green / float(n),
        "ungrounded_review_rate": ungrounded_red_or_amber / float(n),
    }


def main(args: argparse.Namespace) -> int:
    samples = _load_samples(Path(args.input), args.limit)
    if not samples:
        print(f"no samples in {args.input}", file=sys.stderr)
        return 2

    print(f"loaded {len(samples)} pairs from {args.input}", file=sys.stderr)
    print(f"colbert model: {args.colbert_model}", file=sys.stderr)
    print(f"nli model:     {args.nli_model}", file=sys.stderr)
    print(f"reranker:      {args.reranker_model}", file=sys.stderr)

    threshold_policy = None
    if args.thresholds_path:
        threshold_policy = load_risk_band_policy(Path(args.thresholds_path))

    provider = _build_provider(args.colbert_model)
    nli = (
        None
        if args.ablate == "no_nli"
        else HuggingFaceNLIProvider(model_id=args.nli_model)
    )
    reranker = (
        CrossEncoderPremiseReranker(model_id=args.reranker_model)
        if args.reranker_model and args.ablate not in {"no_nli", "no_reranker"}
        else None
    )
    null_bank = build_null_bank_embeddings(provider)
    if args.ablate == "no_structured":
        os.environ["VOYAGER_GROUNDEDNESS_STRUCTURED_GATE"] = "0"
    use_atomic = args.ablate != "no_atomic"

    # Quality profile fusion weights (0.2 literal / 0.8 nli) — see
    # latence_trace/api/service.py PROFILE_ENV_PRESETS["quality"].
    fusion_weights = {
        "calibrated": 0.0,
        "literal": 0.2,
        "nli": 0.8,
        "semantic_entropy": 0.0,
        "structured": 0.0,
    }

    rows_per_stratum: Dict[str, List[Dict[str, Any]]] = {}
    encode_ms_total = 0.0
    score_ms_total = 0.0
    started_all = time.perf_counter()
    for sample in samples:
        sid = sample["id"]
        stratum = sample.get("stratum", "default")
        context = sample["context"]
        grounded_text = sample["grounded_response"]
        ungrounded_text = sample["ungrounded_response"]

        # Side A — grounded.
        materials_g = _encode_pair_side(
            provider=provider, context=context, response=grounded_text
        )
        encode_ms_total += materials_g["encode_ms"]
        scored_g = _score_pair_side(
            materials=materials_g,
            response_text=grounded_text,
            null_bank_embeddings=null_bank,
            nli_provider=nli,
            nli_reranker=reranker,
            nli_concat_premises=True,
            nli_use_atomic_claims=use_atomic,
            fusion_weights=fusion_weights,
        )
        score_ms_total += scored_g["score_ms"]

        # Side B — ungrounded.
        materials_u = _encode_pair_side(
            provider=provider, context=context, response=ungrounded_text
        )
        encode_ms_total += materials_u["encode_ms"]
        scored_u = _score_pair_side(
            materials=materials_u,
            response_text=ungrounded_text,
            null_bank_embeddings=null_bank,
            nli_provider=nli,
            nli_reranker=reranker,
            nli_concat_premises=True,
            nli_use_atomic_claims=use_atomic,
            fusion_weights=fusion_weights,
        )
        score_ms_total += scored_u["score_ms"]

        g_band = _classify_band(
            float(scored_g["scores"].get("groundedness_v2") or 0.0),
            threshold_policy,
        )
        u_band = _classify_band(
            float(scored_u["scores"].get("groundedness_v2") or 0.0),
            threshold_policy,
        )
        rows_per_stratum.setdefault(stratum, []).append(
            {
                "id": sid,
                "flip_kind": sample.get("flip_kind"),
                "language": sample.get("language", "en"),
                "grounded_score": float(
                    scored_g["scores"].get("groundedness_v2") or 0.0
                ),
                "ungrounded_score": float(
                    scored_u["scores"].get("groundedness_v2") or 0.0
                ),
                "grounded_band": g_band,
                "ungrounded_band": u_band,
                "structured_grounded": (
                    None
                    if scored_g["scores"].get("structured_source") is None
                    else float(scored_g["scores"]["structured_source"])
                ),
                "structured_ungrounded": (
                    None
                    if scored_u["scores"].get("structured_source") is None
                    else float(scored_u["scores"]["structured_source"])
                ),
            }
        )
        if args.verbose:
            print(
                f"  {sid:>14}  grounded={scored_g['scores'].get('groundedness_v2'):.3f}"
                f"  ungrounded={scored_u['scores'].get('groundedness_v2'):.3f}",
                file=sys.stderr,
            )

    elapsed_total = time.perf_counter() - started_all
    n_responses = 2 * sum(len(rows) for rows in rows_per_stratum.values())
    summary_per_stratum = {
        stratum: _stratum_summary(rows, threshold_policy)
        for stratum, rows in rows_per_stratum.items()
    }

    all_rows = [row for rows in rows_per_stratum.values() for row in rows]
    macro = _stratum_summary(all_rows, threshold_policy)

    samples_out = [
        {"stratum": stratum, **row}
        for stratum, rows in rows_per_stratum.items()
        for row in rows
    ]

    payload = {
        "input": str(args.input),
        "n_pairs_total": len(samples),
        "colbert_model": args.colbert_model,
        "nli_model": args.nli_model,
        "reranker_model": args.reranker_model,
        "fusion_weights": fusion_weights,
        "threshold_artifact": (
            str(args.thresholds_path) if args.thresholds_path else None
        ),
        "wallclock": {
            "total_seconds": round(elapsed_total, 2),
            "encode_ms_per_response_avg": round(
                encode_ms_total / max(1, n_responses), 2
            ),
            "score_ms_per_response_avg": round(
                score_ms_total / max(1, n_responses), 2
            ),
        },
        "macro": macro,
        "per_stratum": summary_per_stratum,
        "samples": samples_out,
    }
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(rendered)
        print(f"wrote {args.out}", file=sys.stderr)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        help="JSONL file with paired samples (legal_pairs.jsonl, finance_pairs.jsonl, ...).",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--colbert-model",
        default="lightonai/GTE-ModernColBERT-v1",
        help="Production English default. Override for DE benchmarks.",
    )
    parser.add_argument(
        "--nli-model",
        default="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        help=(
            "English NLI peer is the recommended default for English-only "
            "deployments (+10pp paired acc on HaluEval QA / Summarization "
            "vs. the multilingual default). Override with the multilingual "
            "model for DE+EN deployments."
        ),
    )
    parser.add_argument(
        "--reranker-model",
        default="BAAI/bge-reranker-v2-m3",
        help="Cross-encoder premise reranker (quality profile default).",
    )
    parser.add_argument(
        "--thresholds-path",
        default=os.environ.get(
            "LATENCE_TRACE_QUALITY_THRESHOLDS",
            "/workspace/latence-trace/latence_trace/data/thresholds.quality.json",
        ),
        help="JSON artifact for risk-band classification.",
    )
    parser.add_argument("--out", default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--ablate",
        choices=["none", "no_nli", "no_atomic", "no_reranker", "no_structured"],
        default="none",
        help="Disable one component for ablation; default 'none' = full quality stack.",
    )
    return parser


if __name__ == "__main__":
    sys.exit(main(_build_parser().parse_args()))
