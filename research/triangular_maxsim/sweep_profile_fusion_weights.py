"""Per-profile fusion-weight grid sweep for the Pareto profiles.

Reuses the encoder + NLI + reranker across the three profiles, scores
the minimal-pair fixture once per profile to capture the raw channel
table, then runs the offline grid search separately for each profile
and writes the best weights to
``latence_trace/data/fusion_weights.<profile>.json``.

Usage:

    python -m research.triangular_maxsim.sweep_profile_fusion_weights \
        --pairs-per-stratum 30 --steps 5 \
        --out-dir latence_trace/data
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from research.triangular_maxsim.groundedness_external_eval import (  # noqa: E402
    _load_nli_provider,
    _load_provider,
    _load_reranker,
    build_null_bank_embeddings,
)
from research.triangular_maxsim.groundedness_minimal_pairs import (  # noqa: E402
    build_minimal_pairs,
)
from research.triangular_maxsim.sweep_fusion_weights import (  # noqa: E402
    _collect_channel_samples,
    sweep,
)


PROFILES: Dict[str, Dict[str, bool]] = {
    "fast": {
        "use_nli": False,
        "use_reranker": False,
        "atomic": False,
        "concat": False,
        "include_semantic_entropy": False,
        "include_structured": True,
    },
    "balanced": {
        "use_nli": True,
        "use_reranker": False,
        "atomic": False,
        "concat": False,
        "include_semantic_entropy": False,
        "include_structured": True,
    },
    "quality": {
        "use_nli": True,
        "use_reranker": True,
        "atomic": True,
        "concat": True,
        "include_semantic_entropy": False,  # SE samples are synthetic in the harness
        "include_structured": True,
    },
}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Per-profile fusion-weight sweep")
    parser.add_argument(
        "--model",
        default=os.environ.get(
            "VOYAGER_GROUNDEDNESS_MODEL",
            "VAGOsolutions/SauerkrautLM-Multi-Reason-ModernColBERT",
        ),
    )
    parser.add_argument(
        "--nli-model",
        default="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
    )
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--pairs-per-stratum", type=int, default=30)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_REPO / "latence_trace" / "data",
    )
    parser.add_argument("--profiles", nargs="+", default=list(PROFILES.keys()))
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("loading encoder ...")
    provider = _load_provider(args.model)
    pairs = build_minimal_pairs(pairs_per_stratum=args.pairs_per_stratum)
    null_bank = build_null_bank_embeddings(provider)

    nli_cache = {"nli": None, "reranker": None}
    summary: Dict[str, Dict[str, object]] = {}

    for profile in args.profiles:
        if profile not in PROFILES:
            raise SystemExit("unknown profile: {0}".format(profile))
        spec = PROFILES[profile]
        nli_provider = None
        if spec["use_nli"]:
            if nli_cache["nli"] is None:
                print("loading NLI ({0}) ...".format(args.nli_model))
                nli_cache["nli"] = _load_nli_provider(args.nli_model)
            nli_provider = nli_cache["nli"]
        nli_reranker = None
        if spec["use_reranker"]:
            if nli_cache["reranker"] is None:
                print("loading reranker ({0}) ...".format(args.reranker_model))
                nli_cache["reranker"] = _load_reranker(args.reranker_model)
            nli_reranker = nli_cache["reranker"]

        print("scoring pairs for profile={0} ...".format(profile))
        rows = _collect_channel_samples(
            pairs=pairs,
            provider=provider,
            null_bank_embeddings=null_bank,
            nli_provider=nli_provider,
            nli_reranker=nli_reranker,
            nli_concat_premises=bool(spec["concat"]),
            nli_use_atomic_claims=bool(spec["atomic"]),
        )
        print("sweeping fusion grid for profile={0} ...".format(profile))
        report = sweep(
            rows,
            steps=args.steps,
            threshold=args.threshold,
            include_semantic_entropy=bool(spec["include_semantic_entropy"]),
            include_structured=bool(spec["include_structured"]),
        )
        report["profile"] = profile
        report["profile_spec"] = spec
        out_path = args.out_dir / "fusion_weights.{0}.json".format(profile)
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        best = report.get("best") or {}
        summary[profile] = {
            "out": str(out_path),
            "best_weights": best.get("weights"),
            "best_min_f1": best.get("min_f1"),
            "best_macro_f1": best.get("macro_f1"),
        }

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
