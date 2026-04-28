"""TRACE v2 biaffine student training loop (stub).

This file wires up the multi-task loss from ``architecture.py`` with
the distillation corpus produced by ``distill_dataset.py``.  It is
structured so CI can import it without needing a GPU attached - the
heavy ``transformers`` / ``torch`` imports are deferred into the
``main()`` function.

The training is intentionally small-model-small-corpus:

* 6-layer MiniLM-style encoder at fp32 (optionally FSDP fp16 on >=2x
  A10);
* batch 32 / sequence 256 / warmup 2 epochs / cosine decay;
* multi-task loss weights from ``architecture.loss_multi_task``;
* per-source reweighting so the tiny Veracier split does not get
  swamped by HaluEval volume.

Before shipping to production we must hit:

* HaluEval QA red precision >= 0.90  (test split)
* RAGTruth QA red precision >= 0.80  (test split)
* Veracier green precision >= 0.97   (test split)
* Veracier red precision >= 0.95     (test split)

If any of those miss, we stay on v1.

Run::

    python -m research.triangular_maxsim.student_v2.train \\
        --data-dir research/triangular_maxsim/student_v2/data \\
        --out-dir /tmp/trace_v2_student \\
        --encoder sentence-transformers/all-MiniLM-L6-v2 \\
        --epochs 6 --batch-size 32
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib
import random
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("trace.v2.train")


BAND_TO_IDX = {"green": 0, "amber": 1, "red": 2}


@dataclass
class TrainConfig:
    data_dir: pathlib.Path
    out_dir: pathlib.Path
    encoder: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_response_tokens: int = 128
    max_evidence_tokens: int = 256
    epochs: int = 6
    batch_size: int = 32
    lr: float = 3e-5
    weight_decay: float = 0.01
    pair_margin: float = 0.2
    seed: int = 42


def _iter_jsonl(path: pathlib.Path):
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _band_label(row: dict[str, Any]) -> int | None:
    """Pick the best band label: gold wins over teacher."""

    for key in ("gold_band", "turn_band"):
        val = (row.get(key) or "").lower()
        if val in BAND_TO_IDX:
            return BAND_TO_IDX[val]
    return None


def _phi_vector(row: dict[str, Any]) -> list[float]:
    phi = row.get("phi_channels") or {}
    source_type = phi.get("source_type") or "prose"
    # one-hot: {code, markdown, passage_enum, prose, other}
    type_ohe = [
        1.0 if source_type == "code" else 0.0,
        1.0 if source_type == "markdown" else 0.0,
        1.0 if source_type == "passage_enum" else 0.0,
        1.0 if source_type == "prose" else 0.0,
    ]
    return [
        float(phi.get("exact_overlap") or 0.0),
        float(phi.get("numeric_overlap") or 0.0),
        float(phi.get("identifier_overlap") or 0.0),
        0.0,  # reserved for lemmatised overlap bit
        *type_ohe,
    ]


def _build_phi_tensor(response: str, evidence: str, row: dict[str, Any]):
    """Broadcast the per-row phi vector to (Tr, Te, F) for the student.

    v0: constant across all token pairs (fast and still informative).
    v1: per-pair exact-match bit.  Left as a TODO; requires the
    tokeniser offsets so we can mark aligned tokens.
    """
    import torch  # lazy

    feats = _phi_vector(row)
    return torch.tensor(feats, dtype=torch.float32)


def _score_to_band(score: float | None) -> int:
    if score is None:
        return BAND_TO_IDX["amber"]
    if score >= 0.80:
        return BAND_TO_IDX["green"]
    if score < 0.60:
        return BAND_TO_IDX["red"]
    return BAND_TO_IDX["amber"]


def load_rows(data_dir: pathlib.Path) -> dict[str, list[dict]]:
    splits: dict[str, list[dict]] = {}
    for name in ("train", "val", "test"):
        path = data_dir / f"{name}.jsonl"
        splits[name] = list(_iter_jsonl(path)) if path.exists() else []
    return splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=pathlib.Path, required=True)
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    parser.add_argument("--encoder", default=TrainConfig.encoder)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--lr", type=float, default=TrainConfig.lr)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run a single forward+backward on the first batch and exit.",
    )
    args = parser.parse_args()

    # Deferred imports so this file can be imported without torch.
    import torch
    from torch.optim import AdamW
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModel, AutoTokenizer  # type: ignore

    from research.triangular_maxsim.student_v2.architecture import (
        BiaffineStudent,
        StudentConfig,
        loss_multi_task,
    )

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows = load_rows(args.data_dir)
    if not rows.get("train"):
        raise SystemExit(f"no training rows under {args.data_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.encoder)
    encoder = AutoModel.from_pretrained(args.encoder)

    # Read encoder hidden size dynamically.
    cfg = StudentConfig(hidden_dim=encoder.config.hidden_size)
    student = BiaffineStudent(encoder, cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student = student.to(device)
    student.train()

    class TraceDataset(Dataset):
        def __init__(self, items: list[dict]) -> None:
            self.items = [it for it in items if _band_label(it) is not None]

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(self, idx: int) -> dict:
            row = self.items[idx]
            resp = row.get("response_text") or ""
            ev = row.get("evidence_text") or ""
            enc_r = tokenizer(
                resp,
                max_length=128,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            enc_e = tokenizer(
                ev,
                max_length=256,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            band_idx = _band_label(row)
            return {
                "resp_ids": enc_r["input_ids"].squeeze(0),
                "resp_mask": enc_r["attention_mask"].squeeze(0),
                "ev_ids": enc_e["input_ids"].squeeze(0),
                "ev_mask": enc_e["attention_mask"].squeeze(0),
                "phi_vector": _build_phi_tensor(resp, ev, row),
                "turn_band": torch.tensor(band_idx, dtype=torch.long),
                "turn_score": torch.tensor(
                    float(row.get("turn_score") or 0.5), dtype=torch.float32
                ),
                "source": row.get("source") or "unknown",
                "pair_id": row.get("pair_id"),
            }

    train_ds = TraceDataset(rows["train"])
    val_ds = TraceDataset(rows.get("val", []))

    def collate(batch: list[dict]) -> dict:
        out: dict[str, Any] = {}
        for key in ("resp_ids", "resp_mask", "ev_ids", "ev_mask", "turn_band", "turn_score"):
            out[key] = torch.stack([b[key] for b in batch])
        # Broadcast phi vector to the (Tr, Te, F) shape the student expects.
        B = out["resp_ids"].size(0)
        Tr = out["resp_ids"].size(1)
        Te = out["ev_ids"].size(1)
        phi = torch.stack([b["phi_vector"] for b in batch])  # (B, F)
        out["phi"] = phi[:, None, None, :].expand(B, Tr, Te, phi.size(-1))
        # Token support labels: use the v1 teacher's per-token heatmap when
        # available.  For this scaffold we zero them out - ``distill_dataset.py``
        # will be extended once we have audit-log top-k attributions.
        out["token_support"] = torch.zeros(B, Tr)
        return out

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    if val_ds:
        DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    optim = AdamW(student.parameters(), lr=args.lr, weight_decay=TrainConfig.weight_decay)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        batch = next(iter(train_loader))
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        out = student(
            batch["resp_ids"],
            batch["resp_mask"],
            batch["ev_ids"],
            batch["ev_mask"],
            batch["phi"],
        )
        losses = loss_multi_task(
            out,
            token_support_labels=batch["token_support"],
            resp_mask=batch["resp_mask"],
            turn_band_labels=batch["turn_band"],
            turn_score_labels=batch["turn_score"],
        )
        losses["total"].backward()
        optim.step()
        logger.info(
            "dry-run ok batch_size=%d loss=%.4f",
            batch["resp_ids"].size(0),
            losses["total"].item(),
        )
        return

    for epoch in range(args.epochs):
        for step, batch in enumerate(train_loader):
            batch = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }
            out = student(
                batch["resp_ids"],
                batch["resp_mask"],
                batch["ev_ids"],
                batch["ev_mask"],
                batch["phi"],
            )
            losses = loss_multi_task(
                out,
                token_support_labels=batch["token_support"],
                resp_mask=batch["resp_mask"],
                turn_band_labels=batch["turn_band"],
                turn_score_labels=batch["turn_score"],
            )
            losses["total"].backward()
            optim.step()
            optim.zero_grad()
            if step % 50 == 0:
                logger.info(
                    "epoch=%d step=%d loss_total=%.4f loss_band=%.4f loss_score=%.4f",
                    epoch,
                    step,
                    losses["total"].item(),
                    losses["turn_band"].item(),
                    losses["turn_score"].item(),
                )
        ckpt = args.out_dir / f"student_epoch{epoch}.pt"
        torch.save(student.state_dict(), ckpt)
        logger.info("saved checkpoint: %s", ckpt)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
