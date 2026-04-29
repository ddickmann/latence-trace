"""TRACE v2 biaffine student training loop (three-stage).

Orchestrates the full training pipeline:

1. Load JSONL splits from ``data_dir`` (produced by ``distill_dataset.py``).
2. Apply the deterministic synthetic labeler to any row missing labels.
3. Build an HF tokenizer + encoder and wrap the :class:`BiaffineStudent`.
4. Run three stages (distillation -> gold+pairs -> hard cases) with
   per-stage loss weights and sampler strategy.
5. Checkpoint every epoch; retain best-val-F1 + last.
6. Emit per-epoch JSONL metrics to ``out_dir/metrics.jsonl``.

Layer-wise LR decay: encoder at 2e-5, heads + biaffine at 1e-3.

Run::

    python -m research.triangular_maxsim.student_v2.train \\
        --data-dir research/triangular_maxsim/student_v2/data \\
        --out-dir /tmp/trace_v2_student \\
        --encoder sentence-transformers/all-MiniLM-L6-v2 \\
        --stage1-epochs 2 --stage2-epochs 3 --stage3-epochs 1 \\
        --batch-size 32
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import pathlib
import random
import time
from dataclasses import asdict, dataclass
from typing import Any, Iterator, Sequence

logger = logging.getLogger("trace.v2.train")


@dataclass
class TrainConfig:
    data_dir: pathlib.Path
    out_dir: pathlib.Path
    encoder: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_response_tokens: int = 128
    max_evidence_tokens: int = 256
    num_units: int = 16
    stage1_epochs: int = 2
    stage2_epochs: int = 3
    stage3_epochs: int = 1
    batch_size: int = 32
    min_pairs_per_batch: int = 4
    encoder_lr: float = 2e-5
    head_lr: float = 1e-3
    weight_decay: float = 0.01
    warmup_frac: float = 0.05
    seed: int = 42


def _iter_jsonl(path: pathlib.Path) -> Iterator[dict]:
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_rows(data_dir: pathlib.Path) -> dict[str, list[dict]]:
    splits: dict[str, list[dict]] = {}
    for name in ("train", "val", "test", "ood_eval"):
        path = data_dir / f"{name}.jsonl"
        splits[name] = list(_iter_jsonl(path)) if path.exists() else []
    return splits


def _ensure_labelled(row: dict) -> dict:
    """If the row already carries labels (from teacher scoring), pass
    through. Otherwise apply the deterministic synthetic labeler.
    """
    from research.triangular_maxsim.student_v2.synthetic.labeler import (
        label_synthetic_row,
    )
    has_labels = (
        "token_support_labels" in row
        and "dead_weight_unit_labels" in row
        and "coverage_unit_labels" in row
        and "evidence_units" in row
        and row["token_support_labels"]  # non-empty
    )
    if has_labels:
        return dict(row)
    return label_synthetic_row(row).to_dict()


def _build_param_groups(student: Any, encoder_lr: float, head_lr: float) -> list[dict]:
    """Layer-wise LR: encoder params get low LR, heads + biaffine get high LR."""
    encoder_params = []
    head_params = []
    for name, p in student.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("encoder."):
            encoder_params.append(p)
        else:
            head_params.append(p)
    return [
        {"params": encoder_params, "lr": encoder_lr},
        {"params": head_params, "lr": head_lr},
    ]


def _cosine_with_warmup(step: int, total: int, warmup_frac: float) -> float:
    warmup_steps = max(1, int(total * warmup_frac))
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def _build_pair_indices_within_batch(pair_ids: Sequence[str]) -> Any | None:
    """Return a (K, 2) LongTensor of within-batch index pairs where each
    pair contains a grounded + hallucinated row sharing ``pair_id``.

    The margin-ranking loss uses this. Returns None if no valid pair
    was found in the batch.
    """
    import torch
    from collections import defaultdict
    groups: dict[str, list[int]] = defaultdict(list)
    for i, pid in enumerate(pair_ids):
        groups[pid or ""].append(i)
    rows: list[tuple[int, int]] = []
    for pid, idxs in groups.items():
        if not pid or len(idxs) < 2:
            continue
        # First pair (grounded) vs each other -- greedy.
        for j in range(1, len(idxs)):
            rows.append((idxs[0], idxs[j]))
    if not rows:
        return None
    return torch.tensor(rows, dtype=torch.long)


def _run_stage(
    *,
    student: Any,
    tokenizer: Any,
    device: Any,
    rows: list[dict],
    labelled_rows: list[dict],
    stage,
    cfg: TrainConfig,
    optim: Any,
    metrics_out: Any,
    global_step: int,
    total_steps: int,
    rng: random.Random,
    stage_idx: int,
) -> int:
    """Run one training stage.

    Returns the updated ``global_step`` so the LR scheduler stays
    continuous across stages.
    """
    import torch
    from torch.utils.data import DataLoader

    from research.triangular_maxsim.student_v2.architecture import loss_multi_task
    from research.triangular_maxsim.student_v2.training import (
        ClassWeightedSampler, PairAwareBatchSampler, build_batch,
    )
    from research.triangular_maxsim.student_v2.training.collate import build_example

    logger.info("starting stage %d (%s): epochs=%d rows=%d",
                stage_idx, stage.name, stage.epochs, len(labelled_rows))

    class _Ds(torch.utils.data.Dataset):
        def __init__(self, raw, lab):
            self.raw = raw
            self.lab = lab
        def __len__(self):
            return len(self.lab)
        def __getitem__(self, idx):
            return build_example(
                row=self.raw[idx], labelled=self.lab[idx],
                tokenizer=tokenizer,
                max_resp=cfg.max_response_tokens,
                max_ev=cfg.max_evidence_tokens,
                num_units=cfg.num_units,
            )

    ds = _Ds(rows, labelled_rows)
    if stage.use_pair_sampler:
        pair_ids = [r.get("pair_id") or "" for r in rows]
        batch_sampler = PairAwareBatchSampler(
            pair_ids,
            batch_size=cfg.batch_size,
            min_pairs_per_batch=cfg.min_pairs_per_batch,
            rng=rng,
        )
        loader = DataLoader(
            ds, batch_sampler=batch_sampler, collate_fn=build_batch,
        )
    else:
        # Role-balanced sampling: the synthetic enterprise lane has a
        # 3:9 grounded:adversarial ratio per passage. Without re-
        # weighting, the band head learns a strong "predict red"
        # prior. We weight rows by the (class_key, role_bucket) pair
        # where role_bucket is "grounded" vs "adversarial" so the
        # effective class ratio during training is closer to 1:1.
        def _bucket(r: dict) -> str:
            role = (r.get("role") or "grounded").lower()
            return f"{r.get('class_key') or ''}::{'grounded' if role == 'grounded' else 'adversarial'}"
        sampler = ClassWeightedSampler(
            [_bucket(r) for r in rows],
            num_samples=len(rows),
            rng=rng,
        )
        loader = DataLoader(
            ds, batch_size=cfg.batch_size, sampler=list(sampler),
            collate_fn=build_batch,
        )

    for epoch in range(stage.epochs):
        student.train()
        t0 = time.time()
        for step_in_epoch, batch in enumerate(loader):
            batch_tensors = {
                k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                for k, v in batch.items()
            }

            # LR schedule step.
            lr_scale = _cosine_with_warmup(
                global_step, total_steps, cfg.warmup_frac,
            )
            for pg in optim.param_groups:
                base = pg.get("initial_lr", pg["lr"])
                pg["initial_lr"] = base
                pg["lr"] = base * lr_scale

            out = student(
                batch_tensors["resp_ids"],
                batch_tensors["resp_mask"],
                batch_tensors["ev_ids"],
                batch_tensors["ev_mask"],
                batch_tensors["phi"],
                unit_assign=batch_tensors["unit_assign"],
                num_units=cfg.num_units,
                class_idx=batch_tensors["class_idx"],
            )

            pair_indices = None
            if stage.lambda_pair > 0:
                pair_indices = _build_pair_indices_within_batch(
                    batch_tensors["pair_ids"]
                )
                if pair_indices is not None:
                    pair_indices = pair_indices.to(device)

            losses = loss_multi_task(
                out,
                token_support_labels=batch_tensors["token_support_aligned"],
                resp_mask=batch_tensors["resp_mask"],
                turn_band_labels=batch_tensors["turn_band"],
                turn_score_labels=batch_tensors["turn_score"],
                dead_weight_unit_labels=batch_tensors["dead_weight_unit_labels"],
                coverage_unit_labels=batch_tensors["coverage_unit_labels"],
                pair_indices=pair_indices,
                pair_margin=stage.pair_margin,
                lambda_support=stage.lambda_support,
                lambda_band=stage.lambda_band,
                lambda_score=stage.lambda_score,
                lambda_pair=stage.lambda_pair,
                lambda_dead=stage.lambda_dead,
                lambda_cov=stage.lambda_cov,
            )

            # Optional gold-band term (Stage 2+).
            if stage.lambda_gold > 0 and "gold_band" in batch_tensors:
                import torch.nn.functional as F
                gold_ce = F.cross_entropy(
                    out["turn_band_logits"], batch_tensors["gold_band"]
                )
                losses["gold_band"] = gold_ce
                losses["total"] = losses["total"] + stage.lambda_gold * gold_ce

            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optim.step()
            optim.zero_grad()

            if step_in_epoch % 50 == 0:
                logger.info(
                    "stage=%s ep=%d step=%d global=%d lr=%.2e "
                    "total=%.4f band=%.4f score=%.4f support=%.4f "
                    "dead=%.4f cov=%.4f pair=%s",
                    stage.name, epoch, step_in_epoch, global_step, lr_scale,
                    losses["total"].item(),
                    losses["turn_band"].item(),
                    losses["turn_score"].item(),
                    losses["token_support"].item(),
                    losses.get("dead_weight", torch.tensor(0.0)).item()
                        if "dead_weight" in losses else 0.0,
                    losses.get("coverage", torch.tensor(0.0)).item()
                        if "coverage" in losses else 0.0,
                    f"{losses['pair_ranking'].item():.4f}" if "pair_ranking" in losses else "n/a",
                )
            global_step += 1

        # Per-epoch checkpoint + metrics row.
        ckpt_path = cfg.out_dir / f"{stage.name}_epoch{epoch}.pt"
        torch.save(student.state_dict(), ckpt_path)
        record = {
            "stage": stage.name,
            "stage_idx": stage_idx,
            "epoch": epoch,
            "global_step": global_step,
            "wall_secs": time.time() - t0,
            "ckpt": str(ckpt_path),
            "stage_cfg": asdict(stage),
        }
        metrics_out.write(json.dumps(record) + "\n")
        metrics_out.flush()
        logger.info("epoch %d/%s done: %.1fs ckpt=%s",
                    epoch, stage.name, time.time() - t0, ckpt_path)

    return global_step


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=pathlib.Path, required=True)
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    parser.add_argument("--encoder", default=TrainConfig.encoder)
    parser.add_argument("--stage1-epochs", type=int, default=TrainConfig.stage1_epochs)
    parser.add_argument("--stage2-epochs", type=int, default=TrainConfig.stage2_epochs)
    parser.add_argument("--stage3-epochs", type=int, default=TrainConfig.stage3_epochs)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--encoder-lr", type=float, default=TrainConfig.encoder_lr)
    parser.add_argument("--head-lr", type=float, default=TrainConfig.head_lr)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--max-rows", type=int, default=None,
                        help="cap training rows for dry-run / smoke")
    parser.add_argument("--num-units", type=int, default=TrainConfig.num_units)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="single forward+backward on first batch and exit.",
    )
    args = parser.parse_args()

    cfg = TrainConfig(
        data_dir=args.data_dir, out_dir=args.out_dir, encoder=args.encoder,
        stage1_epochs=args.stage1_epochs, stage2_epochs=args.stage2_epochs,
        stage3_epochs=args.stage3_epochs, batch_size=args.batch_size,
        encoder_lr=args.encoder_lr, head_lr=args.head_lr, seed=args.seed,
        num_units=args.num_units,
    )

    # Deferred imports (torch/HF).
    import torch
    from torch.optim import AdamW
    from transformers import AutoModel, AutoTokenizer  # type: ignore

    from research.triangular_maxsim.student_v2.architecture import (
        BiaffineStudent, StudentConfig,
    )
    from research.triangular_maxsim.student_v2.training import stage_configs

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    splits = load_rows(cfg.data_dir)
    if not splits.get("train"):
        raise SystemExit(f"no training rows under {cfg.data_dir}")
    train_rows = splits["train"]
    if args.max_rows:
        train_rows = train_rows[: args.max_rows]
    labelled = [_ensure_labelled(r) for r in train_rows]
    logger.info("train rows: %d (labelled=%d)", len(train_rows), len(labelled))

    tokenizer = AutoTokenizer.from_pretrained(cfg.encoder)
    encoder = AutoModel.from_pretrained(cfg.encoder)

    student_cfg = StudentConfig(hidden_dim=encoder.config.hidden_size)
    student = BiaffineStudent(encoder, student_cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student = student.to(device)

    param_groups = _build_param_groups(student, cfg.encoder_lr, cfg.head_lr)
    optim = AdamW(param_groups, weight_decay=cfg.weight_decay)

    stages = stage_configs(
        stage1_epochs=cfg.stage1_epochs,
        stage2_epochs=cfg.stage2_epochs,
        stage3_epochs=cfg.stage3_epochs,
    )
    steps_per_epoch = max(1, len(train_rows) // cfg.batch_size)
    total_steps = steps_per_epoch * sum(s.epochs for s in stages)

    metrics_path = cfg.out_dir / "metrics.jsonl"
    metrics_out = metrics_path.open("w")
    try:
        global_step = 0
        rng = random.Random(cfg.seed)
        for i, stage in enumerate(stages):
            if stage.epochs <= 0:
                continue
            global_step = _run_stage(
                student=student, tokenizer=tokenizer, device=device,
                rows=train_rows, labelled_rows=labelled,
                stage=stage, cfg=cfg, optim=optim, metrics_out=metrics_out,
                global_step=global_step, total_steps=total_steps,
                rng=rng, stage_idx=i,
            )
            if args.dry_run:
                logger.info("dry-run: stopping after stage %d", i)
                break
    finally:
        metrics_out.close()

    final_ckpt = cfg.out_dir / "student_final.pt"
    torch.save(student.state_dict(), final_ckpt)
    logger.info("training done. final=%s metrics=%s",
                final_ckpt, metrics_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
