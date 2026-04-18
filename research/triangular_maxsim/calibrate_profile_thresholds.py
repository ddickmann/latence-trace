"""Per-profile threshold calibration for the Pareto sweep.

Loads the multilingual encoder, mDeBERTa NLI provider and bge reranker
once each and reuses them across the three profile calibrations
(`fast`, `balanced`, `quality`). Writes one JSON per profile under
``latence_trace/data/thresholds.<profile>.json``.

Usage:

    python -m research.triangular_maxsim.calibrate_profile_thresholds \
        --pairs-per-stratum 30 \
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

from research.triangular_maxsim.calibrate_thresholds import calibrate  # noqa: E402
from research.triangular_maxsim.groundedness_external_eval import (  # noqa: E402
    _load_nli_provider,
    _load_provider,
    _load_reranker,
    build_null_bank_embeddings,
)
from research.triangular_maxsim.groundedness_minimal_pairs import (  # noqa: E402
    build_minimal_pairs,
)


PROFILES = {
    "fast": {
        "use_nli": False,
        "use_reranker": False,
        "atomic": False,
        "concat": False,
        "fusion_weights": {
            "calibrated": 0.0,
            "literal": 1.0,
            "nli": 0.0,
            "semantic_entropy": 0.0,
            "structured": 0.0,
        },
    },
    "balanced": {
        "use_nli": True,
        "use_reranker": False,
        "atomic": False,
        "concat": False,
        "fusion_weights": {
            "calibrated": 0.0,
            "literal": 0.0,
            "nli": 1.0,
            "semantic_entropy": 0.0,
            "structured": 0.0,
        },
    },
    "quality": {
        "use_nli": True,
        "use_reranker": True,
        "atomic": True,
        "concat": True,
        "fusion_weights": {
            "calibrated": 0.0,
            "literal": 0.2,
            "nli": 0.7,
            "semantic_entropy": 0.1,
            "structured": 0.0,
        },
    },
}


_FUSION_ENV_KEYS = {
    "calibrated": "VOYAGER_GROUNDEDNESS_FUSION_W_CALIBRATED",
    "literal": "VOYAGER_GROUNDEDNESS_FUSION_W_LITERAL",
    "nli": "VOYAGER_GROUNDEDNESS_FUSION_W_NLI",
    "semantic_entropy": "VOYAGER_GROUNDEDNESS_FUSION_W_SEMANTIC_ENTROPY",
    "structured": "VOYAGER_GROUNDEDNESS_FUSION_W_STRUCTURED",
}


def _apply_fusion_weights(weights):
    saved = {}
    for channel, env_key in _FUSION_ENV_KEYS.items():
        saved[env_key] = os.environ.get(env_key)
        os.environ[env_key] = "{0}".format(float(weights.get(channel, 0.0)))
    return saved


def _restore_env(saved):
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Per-profile threshold calibration")
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
    parser.add_argument("--precision-target", type=float, default=0.75)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_REPO / "latence_trace" / "data",
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=list(PROFILES.keys()),
    )
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("loading encoder ({0}) ...".format(args.model))
    provider = _load_provider(args.model)
    pairs = build_minimal_pairs(pairs_per_stratum=args.pairs_per_stratum)
    print("encoded null bank ...")
    null_bank_embeddings = build_null_bank_embeddings(provider)

    nli_cache = {"nli": None, "reranker": None}

    summary: Dict[str, Dict[str, str]] = {}
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
        saved_env = _apply_fusion_weights(spec.get("fusion_weights", {}))
        try:
            print("calibrating profile={0} ...".format(profile))
            report = calibrate(
                pairs,
                provider,
                null_bank_embeddings=null_bank_embeddings,
                nli_provider=nli_provider,
                nli_reranker=nli_reranker,
                nli_concat_premises=bool(spec["concat"]),
                nli_use_atomic_claims=bool(spec["atomic"]),
                precision_target=args.precision_target,
            )
        finally:
            _restore_env(saved_env)
        report["profile"] = profile
        report["profile_spec"] = spec
        report["fusion_weights"] = dict(spec.get("fusion_weights", {}))
        out_path = args.out_dir / "thresholds.{0}.json".format(profile)
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        summary[profile] = {
            "out": str(out_path),
            "headline": report["headline"],
            "strata": {
                stratum: {
                    "green_min": info["green_min"],
                    "amber_min": info["amber_min"],
                    "precision_at_green": info["precision_at_green"],
                }
                for stratum, info in sorted(report["strata"].items())
            },
        }

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
