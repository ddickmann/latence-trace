#!/usr/bin/env python3
"""Diagnose HaluEval QA: does the scorer rank right_answer > hallucinated_answer?"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, "/workspace/latence-trace")
sys.path.insert(0, "/workspace/latence-trace/research/triangular_maxsim")

os.environ.setdefault("VOYAGER_GROUNDEDNESS_HALUEVAL_DIR", "/workspace/datasets/halueval")

from research.triangular_maxsim.groundedness_external_benchmarks import load_halueval
from research.triangular_maxsim.groundedness_external_eval import _encode_pair_side, _score_pair_side
from latence_trace.core.nli import HuggingFaceNLIProvider, CrossEncoderPremiseReranker


def _build_provider():
    from pylate import models
    return models.ColBERT(
        model_name_or_path="lightonai/GTE-ModernColBERT-v1",
        device="cuda",
        do_query_expansion=False,
        trust_remote_code=True,
        model_kwargs={"torch_dtype": torch.bfloat16},
    )


def main(nli_model: str, limit: int = 60, fusion_nli: float = 0.3, fusion_cal: float = 0.5, fusion_lit: float = 0.2) -> None:
    print(f"NLI model: {nli_model}")
    print(f"Fusion: cal={fusion_cal} lit={fusion_lit} nli={fusion_nli}")
    provider = _build_provider()
    nli = HuggingFaceNLIProvider(model_id=nli_model)
    reranker = CrossEncoderPremiseReranker(model_id="BAAI/bge-reranker-v2-m3")
    fusion_weights = {
        "calibrated": fusion_cal,
        "literal": fusion_lit,
        "nli": fusion_nli,
        "semantic_entropy": 0.0,
    }

    samples = load_halueval(max_samples_per_stratum=limit)
    by_stratum: dict[str, dict[str, list]] = {}
    for s in samples:
        by_stratum.setdefault(s.stratum, {"faithful": [], "hallucinated": []})
        materials = _encode_pair_side(provider=provider, context=s.context, response=s.response)
        scored = _score_pair_side(
            materials=materials,
            response_text=s.response,
            nli_provider=nli,
            nli_reranker=reranker,
            nli_concat_premises=True,
            nli_use_atomic_claims=True,
            fusion_weights=fusion_weights,
        )
        score = scored["scores"].get("groundedness_v2") or 0.0
        by_stratum[s.stratum][s.label].append((s.sample_id, float(score)))

    out = {}
    for stratum, lanes in by_stratum.items():
        # Pair right/halu by sample_id prefix (HaluEval emits both per source line)
        right = dict((sid.rsplit("-", 1)[0], sc) for sid, sc in lanes["faithful"])
        halu = dict((sid.rsplit("-", 1)[0], sc) for sid, sc in lanes["hallucinated"])
        common = sorted(set(right) & set(halu))
        wins = sum(1 for k in common if right[k] > halu[k])
        ties = sum(1 for k in common if right[k] == halu[k])
        n = len(common)
        right_scores = [right[k] for k in common]
        halu_scores = [halu[k] for k in common]
        out[stratum] = {
            "n_pairs": n,
            "wins_right_higher": wins,
            "ties": ties,
            "paired_accuracy": (wins + ties / 2.0) / max(1, n),
            "right_mean_score": sum(right_scores) / max(1, n),
            "halu_mean_score": sum(halu_scores) / max(1, n),
            "delta_mean": (sum(right_scores) - sum(halu_scores)) / max(1, n),
            "right_score_range": (min(right_scores) if right_scores else None, max(right_scores) if right_scores else None),
            "halu_score_range": (min(halu_scores) if halu_scores else None, max(halu_scores) if halu_scores else None),
        }
    payload = {
        "nli_model": nli_model,
        "fusion": {"calibrated": fusion_cal, "literal": fusion_lit, "nli": fusion_nli},
        "samples_per_label_per_stratum": limit,
        "per_stratum": out,
    }
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    out_path = os.environ.get("HALUEVAL_DIAGNOSE_OUT")
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(rendered)
        print(f"\nWrote {out_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--nli-model", default="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7")
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--fusion-nli", type=float, default=0.3)
    parser.add_argument("--fusion-cal", type=float, default=0.5)
    parser.add_argument("--fusion-lit", type=float, default=0.2)
    args = parser.parse_args()
    main(args.nli_model, args.limit, args.fusion_nli, args.fusion_cal, args.fusion_lit)
