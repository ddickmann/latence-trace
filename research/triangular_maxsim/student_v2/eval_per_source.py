"""Per-source breakdown of quick_eval numbers.

Runs the trained student on a split and reports band accuracy split by
``row["source"]`` so we can see synthetic-only vs RAGTruth-held-out
numbers side by side.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from collections import defaultdict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=pathlib.Path, required=True)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-units", type=int, default=16)
    args = parser.parse_args()

    import torch
    from transformers import AutoModel, AutoTokenizer

    from research.triangular_maxsim.student_v2.architecture import (
        BiaffineStudent, StudentConfig,
    )
    from research.triangular_maxsim.student_v2.training import build_batch
    from research.triangular_maxsim.student_v2.training.collate import (
        BAND_TO_IDX, build_example,
    )
    from research.triangular_maxsim.student_v2.synthetic.labeler import (
        label_synthetic_row,
    )

    split_path = args.data_dir / f"{args.split}.jsonl"
    rows: list[dict] = []
    with split_path.open() as fh:
        for line in fh:
            rows.append(json.loads(line))
    print(f"loaded {len(rows)} rows from {split_path}")

    tok = AutoTokenizer.from_pretrained(args.encoder)
    enc = AutoModel.from_pretrained(args.encoder)
    cfg = StudentConfig(hidden_dim=enc.config.hidden_size)
    student = BiaffineStudent(enc, cfg)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    student.load_state_dict(state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student = student.to(device).eval()

    examples: list[dict] = []
    sources: list[str] = []
    golds: list[int] = []
    for r in rows:
        labelled = dict(r)
        if not labelled.get("token_support_labels"):
            labelled = label_synthetic_row(r).to_dict()
        examples.append(
            build_example(
                row=r, labelled=labelled, tokenizer=tok,
                max_resp=128, max_ev=256, num_units=args.num_units,
            )
        )
        sources.append(str(r.get("source", "unknown")))
        golds.append(BAND_TO_IDX.get((r.get("gold_band") or "amber").lower(), 1))

    per_source_totals: dict[str, int] = defaultdict(int)
    per_source_correct_3way: dict[str, int] = defaultdict(int)
    per_source_correct_binary: dict[str, int] = defaultdict(int)
    per_role_totals: dict[tuple[str, str], int] = defaultdict(int)
    per_role_correct: dict[tuple[str, str], int] = defaultdict(int)

    with torch.no_grad():
        for start in range(0, len(examples), args.batch_size):
            ex = examples[start : start + args.batch_size]
            batch = build_batch(ex)
            bt = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                  for k, v in batch.items()}
            out = student(
                bt["resp_ids"], bt["resp_mask"], bt["ev_ids"], bt["ev_mask"],
                bt["phi"], unit_assign=bt["unit_assign"],
                num_units=args.num_units, class_idx=bt["class_idx"], hard_pool=True,
            )
            pred = out["turn_band_logits"].argmax(dim=-1).cpu().tolist()
            for i in range(len(ex)):
                idx = start + i
                s = sources[idx]
                g = golds[idx]
                per_source_totals[s] += 1
                if pred[i] == g:
                    per_source_correct_3way[s] += 1
                # binary: green -> 0, amber+red -> 1
                bp = 0 if pred[i] == 0 else 1
                bg = 0 if g == 0 else 1
                if bp == bg:
                    per_source_correct_binary[s] += 1
                role = batch["roles"][i]
                per_role_totals[(s, role)] += 1
                if pred[i] == g:
                    per_role_correct[(s, role)] += 1

    report = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "n_rows": len(rows),
        "by_source": {
            s: {
                "n": per_source_totals[s],
                "binary_grounded_accuracy": per_source_correct_binary[s] / per_source_totals[s],
                "turn_band_3way_accuracy": per_source_correct_3way[s] / per_source_totals[s],
            }
            for s in per_source_totals
        },
        "by_source_role": {
            f"{s}::{r}": {
                "n": n,
                "acc": per_role_correct[(s, r)] / n,
            }
            for (s, r), n in per_role_totals.items()
        },
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
