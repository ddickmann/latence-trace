"""Fit and persist the calibrated ``composite_logistic_v3.json`` artifact.

The v3 quality-boost sprint signed off shipping a calibrated
``LogisticRegression`` composite alongside the deterministic
``LinearComposite``. The artifact must live next to ``composite.py`` so
``default_composite()`` returns the logistic path at request time.

Training data
-------------
We pool two labeled banks from
``research/triangular_maxsim/coding/artifacts/``:

- ``experimentB_phantom_bank.json`` — hand-curated v1 *phantom* bank
  (AUROC of ``reverse_context`` already >= 0.95 on correct-vs-phantom).
- ``v2_full_gpu_scorer_rescore.json`` — 180 rows of v2 transcripts
  (correct / ambiguous / wrong).

Labels follow the same convention as the orchestrator: ``phantom=1``
means the turn should be flagged, ``0`` means grounded. For v2 we map
``subcategory=="correct"`` -> 0, ``subcategory in {ambiguous, wrong}``
-> 1. The v1 bank contributes one positive (phantom) and one negative
(correct) row per scenario.

Features
--------
We train on the three signals captured in both banks:
``reverse_context``, ``per_token_p10``, ``literal_guard``, plus the
interaction ``per_token_p10 * literal_guard``. AST / NLI / literal-
novelty features are not present in the training banks; they are
folded into the logistic output at runtime via the linear contribution
term in ``orchestrator.py`` (see ``LogisticComposite.score``).

Usage
-----

    python -m scripts.fit_composite_logistic_v3
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from latence_trace.core.code_lane.composite import (  # noqa: E402
    CompositeFeatures,
    LogisticArtifact,
    fit_logistic_from_payloads,
    persist_artifact,
)


_ARTIFACT_DIR = REPO_ROOT / "research" / "triangular_maxsim" / "coding" / "artifacts"
_V1_PHANTOM_BANK = _ARTIFACT_DIR / "experimentB_phantom_bank.json"
_V2_RESCORE = _ARTIFACT_DIR / "v2_full_gpu_scorer_rescore.json"


def _literal_guard_from(mismatch: Any, total: Any) -> float:
    if total in (None, 0):
        return 1.0
    if mismatch is None:
        return 1.0
    return max(0.0, 1.0 - float(mismatch) / float(total))


def _rows_from_v1(
    bank_path: Path,
) -> List[Tuple[CompositeFeatures, int]]:
    payload = json.loads(bank_path.read_text())
    rows: List[Tuple[CompositeFeatures, int]] = []
    for chunker, records in payload.get("records_by_chunker", {}).items():
        for rec in records:
            correct_agg = rec.get("correct_rescored_aggs") or rec.get("correct_baseline_aggs") or {}
            per_token_correct = rec.get("correct_rescored_per_token") or {}
            correct_rc = correct_agg.get("reverse_context")
            correct_p10 = per_token_correct.get("p10")
            correct_lg = _literal_guard_from(
                rec.get("correct_rescored_literal_mismatch"),
                rec.get("correct_rescored_literal_total"),
            )
            if correct_rc is not None and correct_p10 is not None:
                rows.append(
                    (
                        CompositeFeatures(
                            reverse_context=float(correct_rc),
                            per_token_p10=float(correct_p10),
                            literal_guard=float(correct_lg),
                        ),
                        0,
                    )
                )

            phantom = rec.get("phantom") or {}
            p_rc = phantom.get("reverse_context")
            p_p10 = (phantom.get("per_token_stats") or {}).get("p10")
            p_lg = _literal_guard_from(
                phantom.get("literal_mismatch_count"),
                phantom.get("literal_total_count"),
            )
            if p_rc is not None and p_p10 is not None:
                rows.append(
                    (
                        CompositeFeatures(
                            reverse_context=float(p_rc),
                            per_token_p10=float(p_p10),
                            literal_guard=float(p_lg),
                        ),
                        1,
                    )
                )
    return rows


def _rows_from_v2(
    rescore_path: Path,
) -> List[Tuple[CompositeFeatures, int]]:
    payload = json.loads(rescore_path.read_text())
    rows: List[Tuple[CompositeFeatures, int]] = []
    for row in payload.get("rows", []):
        scores = row.get("scores") or {}
        rc = scores.get("reverse_context")
        lg = scores.get("literal_guarded")
        p10 = (scores.get("per_token_stats") or {}).get("p10")
        if rc is None or lg is None or p10 is None:
            continue
        label = 0 if row.get("subcategory") == "correct" else 1
        rows.append(
            (
                CompositeFeatures(
                    reverse_context=float(rc),
                    per_token_p10=float(p10),
                    literal_guard=float(lg),
                ),
                label,
            )
        )
    return rows


def main() -> None:
    if not _V1_PHANTOM_BANK.exists():
        raise SystemExit(f"missing training bank: {_V1_PHANTOM_BANK}")
    if not _V2_RESCORE.exists():
        raise SystemExit(f"missing training bank: {_V2_RESCORE}")

    v1_rows = _rows_from_v1(_V1_PHANTOM_BANK)
    v2_rows = _rows_from_v2(_V2_RESCORE)
    all_rows = v1_rows + v2_rows

    positives = sum(1 for _, y in all_rows if y == 1)
    negatives = sum(1 for _, y in all_rows if y == 0)
    print(
        f"training set: v1={len(v1_rows)} rows, v2={len(v2_rows)} rows, "
        f"pooled={len(all_rows)} (phantom={positives}, grounded={negatives})"
    )

    feature_order: Sequence[str] = (
        "reverse_context",
        "per_token_p10",
        "literal_guard",
    )
    interactions: Sequence[Tuple[str, str]] = (
        ("per_token_p10", "literal_guard"),
    )

    artifact = fit_logistic_from_payloads(
        all_rows,
        feature_order=feature_order,
        interactions=interactions,
        threshold=0.5,
        trained_on=(
            "pooled v1 phantom_bank + v2_full_gpu_scorer_rescore "
            f"({len(all_rows)} rows)"
        ),
        version="v3.0",
    )
    out = persist_artifact(artifact)
    print(f"wrote artifact: {out}")
    print(
        "feature_order=",
        artifact.feature_order,
        "\ninteractions=",
        artifact.interactions,
        "\ncoef=",
        [round(c, 4) for c in artifact.coef],
        "\nintercept=",
        round(artifact.intercept, 4),
        "\nauroc_pooled=",
        None if artifact.auroc_pooled is None else round(artifact.auroc_pooled, 4),
    )


if __name__ == "__main__":
    main()
