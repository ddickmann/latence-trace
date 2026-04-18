"""Warm-load all production models once and report VRAM headroom on the GPU.

Used by the Pareto-default-profiles sweep to confirm that the full
``quality`` lane (multilingual ColBERT + mDeBERTa NLI + bge reranker)
fits within the 24 GB hard SLO on a single RTX A5000 before we kick
off the 8-lane harness.

Writes a single JSON file with the per-model and cumulative VRAM
footprint (allocated + reserved). Run via:

    python -m research.triangular_maxsim.warm_models \
        --out research/triangular_maxsim/reports/vram_headroom.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def _gpu_state(label: str) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        return {"label": label, "available": False}
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
    return {
        "label": label,
        "available": True,
        "device": torch.cuda.get_device_name(0),
        "allocated_mb": float(allocated),
        "reserved_mb": float(reserved),
        "peak_allocated_mb": float(peak),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Warm-load production models and snapshot VRAM")
    parser.add_argument(
        "--encoder",
        default="VAGOsolutions/SauerkrautLM-Multi-Reason-ModernColBERT",
    )
    parser.add_argument(
        "--nli",
        default="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
    )
    parser.add_argument(
        "--reranker",
        default="BAAI/bge-reranker-v2-m3",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_HERE / "reports" / "vram_headroom.json",
    )
    args = parser.parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    snapshots = [_gpu_state("baseline")]
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # 1) Multilingual ColBERT (bf16)
    from pylate import models  # noqa: WPS433

    encoder = models.ColBERT(
        model_name_or_path=args.encoder,
        device="cuda",
        do_query_expansion=False,
        trust_remote_code=True,
        model_kwargs={"torch_dtype": torch.bfloat16},
    )
    _ = encoder.encode(["warmup query"], is_query=True, batch_size=1)
    _ = encoder.encode(["warmup document with several tokens"], is_query=False, batch_size=1)
    snapshots.append(_gpu_state("after_encoder"))

    # 2) Multilingual NLI (mDeBERTa)
    from latence_trace.core.nli import HuggingFaceNLIProvider  # noqa: WPS433

    nli = HuggingFaceNLIProvider(model_id=args.nli)
    _ = nli.entail(
        ["Berlin ist die Hauptstadt Deutschlands."],
        ["Berlin is the capital of Germany."],
    )
    snapshots.append(_gpu_state("after_nli"))

    # 3) Cross-encoder premise reranker
    from latence_trace.core.nli import CrossEncoderPremiseReranker  # noqa: WPS433

    reranker = CrossEncoderPremiseReranker(model_id=args.reranker)
    _ = reranker.score(
        "Berlin is the capital of Germany.",
        [
            "Berlin ist die Hauptstadt Deutschlands.",
            "Paris ist die Hauptstadt Frankreichs.",
        ],
    )
    snapshots.append(_gpu_state("after_reranker"))

    payload = {
        "encoder": args.encoder,
        "nli": args.nli,
        "reranker": args.reranker,
        "torch_dtype": "bfloat16",
        "vram_cap_mb": 24 * 1024,
        "snapshots": snapshots,
        "fits_within_cap": (
            snapshots[-1].get("reserved_mb", 0.0) <= 24 * 1024
            if torch.cuda.is_available()
            else None
        ),
    }
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
