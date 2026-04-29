"""Quick eval for a trained TRACE v2 student.

Not the full per-class, per-axis, 18-gate eval - that's ``eval.py``
and lives in ``s5``. This script is the "does the checkpoint work at
all?" sanity pass:

* Load a checkpoint.
* Run it on the ``val`` and ``test`` JSONL splits.
* Compute:
  - turn_band (3-way green/amber/red) accuracy + macro-F1.
  - binary grounded-vs-ungrounded accuracy (collapsing green -> grounded,
    amber/red -> ungrounded).
  - token_support precision + recall + F1.
  - dead_weight_unit precision + recall + F1.
  - coverage_unit MAE + Spearman.
* Write a ``quick_eval.json`` next to the checkpoint.

Run::

    python -m research.triangular_maxsim.student_v2.quick_eval \\
        --data-dir research/triangular_maxsim/student_v2/data \\
        --checkpoint research/triangular_maxsim/student_v2/checkpoints/student_final.pt \\
        --split test
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
from collections import defaultdict
from typing import Any, Iterator

logger = logging.getLogger("trace.v2.quick_eval")


def _iter_jsonl(path: pathlib.Path) -> Iterator[dict]:
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=pathlib.Path, required=True)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--split", default="test",
                        choices=["val", "test", "ood_eval"])
    parser.add_argument("--encoder",
                        default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-resp", type=int, default=128)
    parser.add_argument("--max-ev", type=int, default=256)
    parser.add_argument("--num-units", type=int, default=16)
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    import torch
    from transformers import AutoModel, AutoTokenizer  # type: ignore

    from research.triangular_maxsim.student_v2.architecture import (
        BiaffineStudent, StudentConfig,
    )
    from research.triangular_maxsim.student_v2.synthetic.labeler import (
        label_synthetic_row,
    )
    from research.triangular_maxsim.student_v2.training import build_batch
    from research.triangular_maxsim.student_v2.training.collate import (
        build_example,
    )

    split_path = args.data_dir / f"{args.split}.jsonl"
    rows = list(_iter_jsonl(split_path))
    if args.max_rows:
        rows = rows[: args.max_rows]
    logger.info("eval split=%s rows=%d", args.split, len(rows))

    tokenizer = AutoTokenizer.from_pretrained(args.encoder)
    encoder = AutoModel.from_pretrained(args.encoder)

    cfg = StudentConfig(hidden_dim=encoder.config.hidden_size)
    student = BiaffineStudent(encoder, cfg)

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    student.load_state_dict(state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student = student.to(device)
    student.eval()

    # Build examples.
    examples = []
    for r in rows:
        labelled = dict(r)
        if not labelled.get("token_support_labels"):
            labelled = label_synthetic_row(r).to_dict()
        examples.append(
            build_example(
                row=r, labelled=labelled, tokenizer=tokenizer,
                max_resp=args.max_resp, max_ev=args.max_ev,
                num_units=args.num_units,
            )
        )

    # Accumulators.
    band_pred_all: list[int] = []
    band_true_all: list[int] = []
    binary_pred_all: list[int] = []  # 0=grounded, 1=ungrounded
    binary_true_all: list[int] = []
    tok_sup_tp = tok_sup_fp = tok_sup_fn = 0
    dw_tp = dw_fp = dw_fn = 0
    cov_mae_num = 0.0
    cov_count = 0
    # Per-role breakdown.
    per_role_band_correct: dict[str, int] = defaultdict(int)
    per_role_band_total: dict[str, int] = defaultdict(int)

    with torch.no_grad():
        for start in range(0, len(examples), args.batch_size):
            batch_examples = examples[start : start + args.batch_size]
            batch = build_batch(batch_examples)
            batch_t = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            out = student(
                batch_t["resp_ids"], batch_t["resp_mask"],
                batch_t["ev_ids"], batch_t["ev_mask"],
                batch_t["phi"],
                unit_assign=batch_t["unit_assign"],
                num_units=args.num_units,
                class_idx=batch_t["class_idx"],
                hard_pool=True,
            )
            band_pred = out["turn_band_logits"].argmax(dim=-1)
            # Use gold_band (binary ground-truth) for primary eval.
            band_true = batch_t["gold_band"]
            band_pred_all.extend(band_pred.cpu().tolist())
            band_true_all.extend(band_true.cpu().tolist())

            for i, role in enumerate(batch["roles"]):
                per_role_band_total[role] += 1
                if band_pred[i].item() == band_true[i].item():
                    per_role_band_correct[role] += 1

            # Binary grounded / ungrounded: green=0, amber=red=1.
            b_pred = (band_pred != 0).long()
            b_true = (band_true != 0).long()
            binary_pred_all.extend(b_pred.cpu().tolist())
            binary_true_all.extend(b_true.cpu().tolist())

            # Token support P/R.
            ts_logits = out["token_support_logits"]
            ts_pred = (ts_logits > 0).float()
            ts_true = batch_t["token_support_aligned"]
            mask = batch_t["resp_mask"].float()
            tp = ((ts_pred == 1) & (ts_true == 1) & (mask > 0)).sum().item()
            fp = ((ts_pred == 1) & (ts_true == 0) & (mask > 0)).sum().item()
            fn = ((ts_pred == 0) & (ts_true == 1) & (mask > 0)).sum().item()
            tok_sup_tp += tp
            tok_sup_fp += fp
            tok_sup_fn += fn

            # Dead-weight.
            dw_logits = out["dead_weight_unit_logits"]
            if dw_logits is not None:
                dw_pred = (dw_logits > 0).float()
                dw_true = batch_t["dead_weight_unit_labels"]
                unit_mask = out["unit_mask"].float() if out["unit_mask"] is not None else torch.ones_like(dw_true)
                tp = ((dw_pred == 1) & (dw_true == 1) & (unit_mask > 0)).sum().item()
                fp = ((dw_pred == 1) & (dw_true == 0) & (unit_mask > 0)).sum().item()
                fn = ((dw_pred == 0) & (dw_true == 1) & (unit_mask > 0)).sum().item()
                dw_tp += tp
                dw_fp += fp
                dw_fn += fn

            # Coverage MAE.
            cov = out["coverage_unit_scores"]
            if cov is not None:
                cov_true = batch_t["coverage_unit_labels"]
                unit_mask = out["unit_mask"].float() if out["unit_mask"] is not None else torch.ones_like(cov_true)
                abs_err = ((cov - cov_true).abs() * unit_mask).sum().item()
                count = unit_mask.sum().item()
                cov_mae_num += abs_err
                cov_count += count

    # Compute metrics.
    n = len(band_pred_all)
    band_acc = sum(p == t for p, t in zip(band_pred_all, band_true_all)) / max(1, n)
    binary_acc = sum(p == t for p, t in zip(binary_pred_all, binary_true_all)) / max(1, n)
    tok_sup_p = tok_sup_tp / max(1, tok_sup_tp + tok_sup_fp)
    tok_sup_r = tok_sup_tp / max(1, tok_sup_tp + tok_sup_fn)
    tok_sup_f1 = 2 * tok_sup_p * tok_sup_r / max(1e-9, tok_sup_p + tok_sup_r)
    dw_p = dw_tp / max(1, dw_tp + dw_fp)
    dw_r = dw_tp / max(1, dw_tp + dw_fn)
    dw_f1 = 2 * dw_p * dw_r / max(1e-9, dw_p + dw_r)
    cov_mae = cov_mae_num / max(1, cov_count)

    results = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "n_rows": n,
        "hallucination": {
            "turn_band_3way_accuracy": round(band_acc, 4),
            "binary_grounded_accuracy": round(binary_acc, 4),
        },
        "token_support": {
            "precision": round(tok_sup_p, 4),
            "recall": round(tok_sup_r, 4),
            "f1": round(tok_sup_f1, 4),
        },
        "dead_weight": {
            "precision": round(dw_p, 4),
            "recall": round(dw_r, 4),
            "f1": round(dw_f1, 4),
        },
        "coverage": {
            "mae": round(cov_mae, 4),
            "units_scored": cov_count,
        },
        "per_role_band_accuracy": {
            role: round(per_role_band_correct[role] / max(1, total), 4)
            for role, total in per_role_band_total.items()
        },
    }

    out_path = args.checkpoint.parent / f"quick_eval_{args.split}.json"
    with out_path.open("w") as fh:
        json.dump(results, fh, indent=2)
    print(json.dumps(results, indent=2))
    logger.info("wrote: %s", out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
