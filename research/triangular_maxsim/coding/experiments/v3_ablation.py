"""Phased ablation matrix for the code-lane quality-boost sprint.

Walks the following phases on the pre-scored ``v2_full_gpu_scorer_rescore.json``
bank of 180 transcripts (60 correct / 60 ambiguous / 60 wrong):

1. ``baseline``            — ``reverse_context`` only.
2. ``+literal_guard``      — linear combination of ``reverse_context`` +
   ``literal_guarded`` using the v2 production weights.
3. ``+per_token_p10``      — adds the per-token p10 floor (the v2
   linear composite input).
4. ``+literal_novelty``    — literal-novelty-at-high-confidence
   strengthening of ``literal_guard``. Requires offline re-scoring, so
   this phase is flagged ``requires_full_rerun`` when the signal is
   unavailable and is reported as ``None``.
5. ``+ast``                — AST-grounded phantom / literal-drift
   symbol counts. Same caveat as the previous phase.
6. ``+nli_cascade``        — ambiguity-triggered NLI contradiction
   probability on top-3 evidence units. Same caveat.
7. ``+logistic``           — calibrated logistic regression over the
   three always-available baseline features (``reverse_context``,
   ``literal_guarded``, ``per_token_p10``) plus an interaction term;
   this provides the *offline* lower bound on what the calibrated
   composite can squeeze out of the already-captured signals.

For each phase the script reports:

- ``auroc_correct_vs_wrong`` — the drift lift we ultimately care about.
- ``auroc_correct_vs_ambiguous`` — hallucination-vs-soft-drift.
- ``auroc_grounded_vs_ungrounded`` — correct + ambiguous considered
  grounded, wrong considered ungrounded (the top-level composite target).
- ``monotonicity_rate`` — fraction of base scenarios where
  ``score(correct) > score(ambiguous) > score(wrong)``.

Artefacts are written to
``research/triangular_maxsim/coding/artifacts/v3_ablation.{json,md}``.
Rerun with ``python -m research.triangular_maxsim.coding.experiments.v3_ablation``
or ``python research/triangular_maxsim/coding/experiments/v3_ablation.py``.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

ARTIFACTS = Path(__file__).resolve().parent.parent / "artifacts"
RESCORE_PATH = ARTIFACTS / "v2_full_gpu_scorer_rescore.json"
OUT_JSON = ARTIFACTS / "v3_ablation.json"
OUT_MD = ARTIFACTS / "v3_ablation.md"


def _auroc(scores_pos: Sequence[float], scores_neg: Sequence[float]) -> Optional[float]:
    """Mann-Whitney-U AUROC.

    ``pos`` means the *higher* score is correct; ``neg`` means the
    *lower* score is correct. Used here with higher=more-phantom, so
    callers pass ``pos=wrong_scores`` and ``neg=correct_scores`` for
    correct-vs-wrong.
    """
    if not scores_pos or not scores_neg:
        return None
    tie = 0.0
    wins = 0.0
    for p in scores_pos:
        for n in scores_neg:
            if p > n:
                wins += 1.0
            elif p == n:
                tie += 1.0
    denom = float(len(scores_pos) * len(scores_neg))
    if denom == 0.0:
        return None
    return (wins + 0.5 * tie) / denom


def _monotonicity_rate(
    correct: Dict[str, float],
    ambiguous: Dict[str, float],
    wrong: Dict[str, float],
) -> float:
    """Fraction of base scenarios where c > a > w on the *grounded* axis."""
    shared = set(correct) & set(ambiguous) & set(wrong)
    if not shared:
        return 0.0
    mono = 0
    for sid in shared:
        if correct[sid] > ambiguous[sid] > wrong[sid]:
            mono += 1
    return mono / len(shared)


def _phase_metrics(
    *,
    rows: Iterable[dict],
    score_fn,
    direction: str = "higher_is_grounded",
) -> dict:
    """Compute AUROC / monotonicity for a per-row score function.

    ``direction`` may be ``higher_is_grounded`` (score for correct cases
    should be larger than for wrong ones — our default) or
    ``higher_is_phantom`` (score for wrong cases should dominate).
    The AUROC formulation always treats the numerically-larger score as
    the *positive* class, so we flip the inputs internally.
    """
    correct_scores: Dict[str, float] = {}
    ambiguous_scores: Dict[str, float] = {}
    wrong_scores: Dict[str, float] = {}

    for row in rows:
        base = row["base_scenario_id"]
        subcat = row["subcategory"]
        value = score_fn(row)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        if subcat == "correct":
            correct_scores[base] = float(value)
        elif subcat == "ambiguous":
            ambiguous_scores[base] = float(value)
        elif subcat == "wrong":
            wrong_scores[base] = float(value)

    if direction == "higher_is_grounded":
        auroc_cw = _auroc(
            list(correct_scores.values()), list(wrong_scores.values())
        )
        auroc_ca = _auroc(
            list(correct_scores.values()), list(ambiguous_scores.values())
        )
        grounded = list(correct_scores.values()) + list(ambiguous_scores.values())
        auroc_gu = _auroc(grounded, list(wrong_scores.values()))
        mono = _monotonicity_rate(correct_scores, ambiguous_scores, wrong_scores)
    else:
        auroc_cw = _auroc(
            list(wrong_scores.values()), list(correct_scores.values())
        )
        auroc_ca = _auroc(
            list(ambiguous_scores.values()), list(correct_scores.values())
        )
        grounded = list(correct_scores.values()) + list(ambiguous_scores.values())
        auroc_gu = _auroc(list(wrong_scores.values()), grounded)
        mono = _monotonicity_rate(correct_scores, ambiguous_scores, wrong_scores)

    return {
        "auroc_correct_vs_wrong": auroc_cw,
        "auroc_correct_vs_ambiguous": auroc_ca,
        "auroc_grounded_vs_ungrounded": auroc_gu,
        "monotonicity_rate": mono,
        "n_correct": len(correct_scores),
        "n_ambiguous": len(ambiguous_scores),
        "n_wrong": len(wrong_scores),
    }


def _features(row: dict) -> dict:
    scores = row.get("scores") or {}
    per = (scores.get("per_token_stats") or {})
    return {
        "reverse_context": scores.get("reverse_context"),
        "literal_guarded": scores.get("literal_guarded"),
        "per_token_p10": per.get("p10"),
        "per_token_min": per.get("min"),
    }


def _phase_baseline(row: dict) -> Optional[float]:
    return _features(row).get("reverse_context")


def _phase_literal_guard(row: dict) -> Optional[float]:
    f = _features(row)
    rc, lg = f.get("reverse_context"), f.get("literal_guarded")
    if rc is None or lg is None:
        return None
    # v2 production linear weights: 0.7 * reverse_context + 0.3 * literal_guarded.
    return 0.7 * float(rc) + 0.3 * float(lg)


def _phase_per_token_floor(row: dict) -> Optional[float]:
    f = _features(row)
    rc, lg, pt = (
        f.get("reverse_context"),
        f.get("literal_guarded"),
        f.get("per_token_p10"),
    )
    if rc is None or lg is None or pt is None:
        return None
    # Canonical v2 composite weights (see composite.py LinearComposite).
    return 0.55 * float(rc) + 0.25 * float(lg) + 0.20 * float(pt)


def _phase_logistic(rows: List[dict]) -> Optional[dict]:
    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
    except Exception:
        return None

    X: List[List[float]] = []
    y: List[int] = []
    keep_rows: List[dict] = []
    for row in rows:
        f = _features(row)
        if (
            f["reverse_context"] is None
            or f["literal_guarded"] is None
            or f["per_token_p10"] is None
        ):
            continue
        X.append(
            [
                float(f["reverse_context"]),
                float(f["literal_guarded"]),
                float(f["per_token_p10"]),
                float(f["reverse_context"]) * float(f["literal_guarded"]),
            ]
        )
        y.append(0 if row["subcategory"] == "correct" else 1)
        keep_rows.append(row)

    if not X:
        return None
    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.int64)
    model = LogisticRegression(
        max_iter=500, class_weight="balanced", C=1.0, solver="lbfgs"
    )
    model.fit(X_arr, y_arr)
    probs = model.predict_proba(X_arr)[:, 1]  # P(phantom)

    def _lookup(row: dict) -> Optional[float]:
        for r, p in zip(keep_rows, probs):
            if r["id"] == row["id"]:
                return 1.0 - float(p)
        return None

    pool_auroc = float(roc_auc_score(y_arr, probs))
    metrics = _phase_metrics(rows=rows, score_fn=_lookup)
    metrics["pooled_train_auroc"] = pool_auroc
    metrics["coef"] = model.coef_[0].tolist()
    metrics["intercept"] = float(model.intercept_[0])
    return metrics


def main() -> None:
    if not RESCORE_PATH.exists():
        raise SystemExit(f"Missing rescore artefact: {RESCORE_PATH}")
    data = json.loads(RESCORE_PATH.read_text())
    rows = data["rows"]

    phases: Dict[str, dict] = {}
    phases["baseline"] = {
        "description": "reverse_context only",
        "metrics": _phase_metrics(rows=rows, score_fn=_phase_baseline),
        "available": True,
    }
    phases["+literal_guard"] = {
        "description": "0.7*reverse_context + 0.3*literal_guarded",
        "metrics": _phase_metrics(rows=rows, score_fn=_phase_literal_guard),
        "available": True,
    }
    phases["+per_token_p10"] = {
        "description": "v2 linear composite (0.55/0.25/0.20)",
        "metrics": _phase_metrics(rows=rows, score_fn=_phase_per_token_floor),
        "available": True,
    }
    phases["+literal_novelty"] = {
        "description": "literal-novelty-at-high-confidence (requires full rerun)",
        "metrics": None,
        "available": False,
        "note": (
            "literal_novelty_min is computed only by the production "
            "code-lane orchestrator; this rescore bank predates the "
            "signal. Rerun score_code_groundedness over transcripts_v2 "
            "to activate this phase."
        ),
    }
    phases["+ast"] = {
        "description": "tree-sitter AST drift + phantom symbol counts",
        "metrics": None,
        "available": False,
        "note": (
            "ast_literal_drift_count / ast_phantom_symbol_count require "
            "the multi-language extractor from "
            "latence_trace.core.code_lane.ast_grounding; not captured "
            "in the pre-sprint rescore bank."
        ),
    }
    phases["+nli_cascade"] = {
        "description": "ambiguity-triggered NLI on top-3 evidence",
        "metrics": None,
        "available": False,
        "note": (
            "Requires a live vLLM NLI server; run the full code lane "
            "against transcripts_v2 with NLI enabled to collect this "
            "column."
        ),
    }
    logistic = _phase_logistic(rows)
    phases["+logistic"] = {
        "description": (
            "LogisticRegression over (reverse_context, literal_guarded, "
            "per_token_p10, reverse_context*literal_guarded); trained "
            "in-sample (correct=0, ambiguous+wrong=1)"
        ),
        "metrics": logistic,
        "available": logistic is not None,
    }

    payload = {
        "source": str(RESCORE_PATH.relative_to(REPO_ROOT)),
        "n_rows": len(rows),
        "label_counts": {
            label: sum(1 for r in rows if r["label"] == label)
            for label in sorted({r["label"] for r in rows})
        },
        "phases": phases,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2))

    lines: List[str] = [
        "# Code-lane v3 ablation matrix",
        "",
        f"Source: `{payload['source']}` ({payload['n_rows']} rows).",
        "",
        "| phase | AUROC correct-vs-wrong | AUROC correct-vs-ambiguous | AUROC grounded-vs-ungrounded | monotonicity |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, entry in phases.items():
        m = entry.get("metrics") or {}
        def _fmt(v):
            return f"{v:.3f}" if isinstance(v, float) else "—"
        lines.append(
            "| {phase} | {cw} | {ca} | {gu} | {mono} |".format(
                phase=name,
                cw=_fmt(m.get("auroc_correct_vs_wrong")),
                ca=_fmt(m.get("auroc_correct_vs_ambiguous")),
                gu=_fmt(m.get("auroc_grounded_vs_ungrounded")),
                mono=_fmt(m.get("monotonicity_rate")),
            )
        )
    lines.extend(
        [
            "",
            "## Phase notes",
            "",
        ]
    )
    for name, entry in phases.items():
        note = entry.get("note")
        if note:
            lines.append(f"- **{name}** — {entry['description']}. {note}")
        else:
            lines.append(f"- **{name}** — {entry['description']}.")
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_JSON}")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
