"""Composite phantom-API guard score.

Motivation
----------
Experiment B (see ``experimentB_phantom_bank.{json,md}``) showed that single
MaxSim aggregates like ``reverse_context`` are not well-separated on v1 when
the only signal you have is "correct vs hand-authored wrong". They are,
however, strongly separated between correct and *genuinely phantom*
responses (AUROC ~0.95-1.0 on the 8-base v1 bank).

For the production phantom-API guard we fuse three complementary signals
into a single composite that is calibrated once and applied at request time.

The three signals
-----------------
1. ``reverse_context`` — global MaxSim aggregate, catches topic drift.
2. ``per_token_p10``    — 10th-percentile per-token MaxSim, catches a few
                          phantom tokens drowning in a mostly-grounded reply.
3. ``literal_guard_strength`` — fraction of literal identifiers in the
                                response that match any support-unit chunk;
                                low value → phantom imports / made-up names.

Score
-----
``composite = w_rc * reverse_context + w_pt * per_token_p10 + w_lg * literal_guard``

Weights are fit by searching a small simplex so that
``AUROC(composite, correct_vs_phantom)`` is maximised on v1. The same
simplex search also emits a threshold that controls the operational
precision target (phantom-rate <= ``target_fp`` on correct responses).

Outputs
-------
- ``composite_phantom_score.json`` — weights, threshold, AUROC on v1 and,
  if present, v2.
- ``composite_phantom_score.md``   — human-readable digest.
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

sys.path.insert(0, "/workspace/latence-trace")

from research.triangular_maxsim.coding.coding_agent_validation import auroc  # noqa: E402


_ROOT = Path("/workspace/latence-trace/research/triangular_maxsim/coding")
_ARTIFACTS = _ROOT / "artifacts"
_V1_PHANTOM_BANK = _ARTIFACTS / "experimentB_phantom_bank.json"


# --- Signal extraction --------------------------------------------------------


def _literal_guard_strength(
    literal_mismatch: Optional[int], literal_total: Optional[int]
) -> Optional[float]:
    if literal_total is None or literal_total == 0:
        return 1.0  # no literals → no evidence of phantom imports
    if literal_mismatch is None:
        return None
    return max(0.0, 1.0 - float(literal_mismatch) / float(literal_total))


def _v1_rows_for_contrast(
    records: Sequence[Dict[str, Any]]
) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    """From the v1 phantom bank, return (correct_rows, phantom_rows).

    Each row is a dict of {rc, per_token_p10, literal_guard}.
    """
    correct_rows: List[Dict[str, float]] = []
    phantom_rows: List[Dict[str, float]] = []
    for rec in records:
        correct_agg = rec.get("correct_rescored_aggs") or rec.get("correct_baseline_aggs") or {}
        per_token = rec.get("correct_rescored_per_token") or {}
        lg = _literal_guard_strength(
            rec.get("correct_rescored_literal_mismatch"),
            rec.get("correct_rescored_literal_total"),
        )
        if lg is None:
            lg = 1.0
        rc = correct_agg.get("reverse_context")
        p10 = per_token.get("p10")
        if rc is None or p10 is None:
            continue
        correct_rows.append({"rc": float(rc), "p10": float(p10), "lg": float(lg)})

        phantom = rec.get("phantom") or {}
        p_rc = phantom.get("reverse_context")
        p_p10 = (phantom.get("per_token_stats") or {}).get("p10")
        p_lg = _literal_guard_strength(
            phantom.get("literal_mismatch_count"),
            phantom.get("literal_total_count"),
        )
        if p_rc is None or p_p10 is None:
            continue
        if p_lg is None:
            p_lg = 1.0
        phantom_rows.append({"rc": float(p_rc), "p10": float(p_p10), "lg": float(p_lg)})
    return correct_rows, phantom_rows


def _v2_rows_from_run(payload: Dict[str, Any]) -> Dict[str, List[Dict[str, float]]]:
    """From a full runner JSON (``--case-bank transcripts_v2``), return rows
    grouped by tier (``correct`` / ``wrong`` / ``ambiguous``).

    ``per_token_stats`` is not always persisted by the runner; when
    missing we fall back to ``reverse_context_calibrated`` as a proxy
    for p10 (they track closely). Rows missing both signals are skipped.
    """
    rows_by_tier: Dict[str, List[Dict[str, float]]] = {"correct": [], "wrong": [], "ambiguous": []}
    for row in payload.get("rows") or []:
        tier = row.get("tier") or (row.get("subcategory") or "correct")
        if tier not in rows_by_tier:
            continue
        scores = row.get("scores") or {}
        rc = scores.get("reverse_context")
        per_token = scores.get("per_token_stats") or {}
        p10 = per_token.get("p10")
        if p10 is None:
            # Fallback: reverse_context_calibrated is a smoothed variant
            # and tracks the lower tail of per-token MaxSim in practice.
            p10 = scores.get("reverse_context_calibrated")
        lg = scores.get("literal_guarded")
        if rc is None or p10 is None:
            continue
        if lg is None:
            lg = 1.0
        rows_by_tier[tier].append({"rc": float(rc), "p10": float(p10), "lg": float(lg)})
    return rows_by_tier


# --- Fitting ------------------------------------------------------------------


def _composite(rows: Sequence[Dict[str, float]], w: Tuple[float, float, float]) -> List[float]:
    w_rc, w_pt, w_lg = w
    return [w_rc * r["rc"] + w_pt * r["p10"] + w_lg * r["lg"] for r in rows]


def _auroc_for(
    pos_rows: Sequence[Dict[str, float]],
    neg_rows: Sequence[Dict[str, float]],
    w: Tuple[float, float, float],
) -> float:
    """Treats *phantom* as the positive class: composite ranks higher for
    *grounded* text, so we flip the label so AUROC corresponds to
    "grounded > phantom"."""
    scores = _composite(list(pos_rows) + list(neg_rows), w)
    labels = [1] * len(pos_rows) + [0] * len(neg_rows)
    return auroc(scores, labels)


def _simplex_weights(step: float = 0.1) -> List[Tuple[float, float, float]]:
    grid: List[Tuple[float, float, float]] = []
    points = [round(i * step, 4) for i in range(int(1 / step) + 1)]
    for a in points:
        for b in points:
            c = round(1.0 - a - b, 4)
            if c < 0 or c > 1:
                continue
            grid.append((a, b, c))
    return grid


def fit_weights(
    correct_rows: Sequence[Dict[str, float]],
    phantom_rows: Sequence[Dict[str, float]],
    step: float = 0.05,
) -> Tuple[Tuple[float, float, float], float]:
    best_w = (0.34, 0.33, 0.33)
    best_auroc = -1.0
    for w in _simplex_weights(step):
        a = _auroc_for(correct_rows, phantom_rows, w)
        if a > best_auroc + 1e-6:
            best_auroc = a
            best_w = w
    return best_w, best_auroc


def lock_threshold(
    correct_rows: Sequence[Dict[str, float]],
    phantom_rows: Sequence[Dict[str, float]],
    w: Tuple[float, float, float],
    target_fp: float = 0.05,
) -> Dict[str, Any]:
    """Pick a threshold on the composite so that the false-phantom rate on
    ``correct_rows`` is <= ``target_fp`` (grounded *correct* responses mis-
    flagged as phantom). Report the resulting recall on phantoms."""
    correct_scores = sorted(_composite(correct_rows, w))
    phantom_scores = _composite(phantom_rows, w)
    if not correct_scores:
        return {"threshold": 0.0, "target_fp": target_fp}
    # We flag as phantom if composite < threshold. To keep FP <= target_fp
    # on grounded, we choose threshold = target_fp-quantile of correct scores.
    idx = max(0, int(target_fp * len(correct_scores)) - 1)
    threshold = correct_scores[idx]
    tp = sum(1 for s in phantom_scores if s < threshold)
    fp = sum(1 for s in correct_scores if s < threshold)
    recall = tp / max(1, len(phantom_scores))
    actual_fp = fp / max(1, len(correct_scores))
    return {
        "threshold": float(threshold),
        "target_fp": float(target_fp),
        "actual_fp": float(actual_fp),
        "recall": float(recall),
        "flagged_phantoms": int(tp),
        "flagged_correct": int(fp),
    }


# --- Main ---------------------------------------------------------------------


def evaluate_on_v1(
    fit_step: float = 0.05,
    target_fp: float = 0.05,
) -> Dict[str, Any]:
    if not _V1_PHANTOM_BANK.exists():
        raise FileNotFoundError(f"v1 phantom bank not found: {_V1_PHANTOM_BANK}")
    with _V1_PHANTOM_BANK.open() as handle:
        payload = json.load(handle)

    per_chunker_results: Dict[str, Dict[str, Any]] = {}
    for chunker, records in payload["records_by_chunker"].items():
        correct_rows, phantom_rows = _v1_rows_for_contrast(records)
        if not correct_rows or not phantom_rows:
            per_chunker_results[chunker] = {"error": "missing rows"}
            continue
        w, fit_auroc = fit_weights(correct_rows, phantom_rows, step=fit_step)
        lock = lock_threshold(correct_rows, phantom_rows, w, target_fp=target_fp)
        baselines = {
            "rc_alone_auroc": float(
                auroc(
                    [r["rc"] for r in correct_rows + phantom_rows],
                    [1] * len(correct_rows) + [0] * len(phantom_rows),
                )
            ),
            "p10_alone_auroc": float(
                auroc(
                    [r["p10"] for r in correct_rows + phantom_rows],
                    [1] * len(correct_rows) + [0] * len(phantom_rows),
                )
            ),
            "lg_alone_auroc": float(
                auroc(
                    [r["lg"] for r in correct_rows + phantom_rows],
                    [1] * len(correct_rows) + [0] * len(phantom_rows),
                )
            ),
        }
        per_chunker_results[chunker] = {
            "weights": {"w_rc": w[0], "w_pt": w[1], "w_lg": w[2]},
            "composite_auroc_v1": float(fit_auroc),
            "baselines": baselines,
            "threshold_lock": lock,
            "n_correct": len(correct_rows),
            "n_phantom": len(phantom_rows),
        }
    return {"per_chunker": per_chunker_results}


def apply_to_v2(
    v2_payload_path: Path,
    *,
    weights: Tuple[float, float, float],
    threshold: float,
) -> Dict[str, Any]:
    with v2_payload_path.open() as handle:
        payload = json.load(handle)
    rows_by_tier = _v2_rows_from_run(payload)
    if not rows_by_tier["correct"]:
        return {"error": "no correct rows in v2 payload"}

    def _scores(rs: Sequence[Dict[str, float]]) -> List[float]:
        return _composite(rs, weights)

    correct = rows_by_tier["correct"]
    wrong = rows_by_tier["wrong"]
    ambiguous = rows_by_tier["ambiguous"]

    def _contrast(pos: Sequence[Dict[str, float]], neg: Sequence[Dict[str, float]]) -> Optional[float]:
        if not pos or not neg:
            return None
        return float(auroc(_scores(list(pos) + list(neg)), [1] * len(pos) + [0] * len(neg)))

    flagged_counts = {
        tier: sum(1 for s in _scores(rs) if s < threshold)
        for tier, rs in rows_by_tier.items()
    }
    # v2-calibrated threshold so the production service has a ready-to-use
    # operating point tuned to the v2-style drift distribution. Sweep a
    # few FP budgets so the on-call operator can pick a cut.
    v2_lock = lock_threshold(correct, wrong, weights, target_fp=0.05) if wrong else None
    v2_sweep = (
        [
            {"target_fp": fp, **lock_threshold(correct, wrong, weights, target_fp=fp)}
            for fp in (0.02, 0.05, 0.10, 0.20)
        ]
        if wrong
        else []
    )
    return {
        "auroc_correct_vs_wrong": _contrast(correct, wrong),
        "auroc_correct_vs_ambiguous": _contrast(correct, ambiguous),
        "flagged_at_v1_threshold": flagged_counts,
        "counts_by_tier": {k: len(v) for k, v in rows_by_tier.items()},
        "v2_threshold_lock": v2_lock,
        "v2_threshold_sweep": v2_sweep,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-step", type=float, default=0.05)
    parser.add_argument("--target-fp", type=float, default=0.05)
    parser.add_argument(
        "--v2-payload",
        type=Path,
        default=None,
        help="Optional path to a transcripts_v2 runner JSON; if present, the fitted weights are applied to it.",
    )
    parser.add_argument("--out-tag", default="composite_phantom_score")
    args = parser.parse_args()

    v1_results = evaluate_on_v1(fit_step=args.fit_step, target_fp=args.target_fp)

    v2_results: Optional[Dict[str, Any]] = None
    if args.v2_payload is not None:
        # Pick the chunker cell with the best fitted AUROC on v1.
        chunker_choice = max(
            v1_results["per_chunker"].items(),
            key=lambda kv: kv[1].get("composite_auroc_v1", -1.0),
        )[0]
        weights_dict = v1_results["per_chunker"][chunker_choice]["weights"]
        weights = (weights_dict["w_rc"], weights_dict["w_pt"], weights_dict["w_lg"])
        threshold = v1_results["per_chunker"][chunker_choice]["threshold_lock"]["threshold"]
        v2_results = apply_to_v2(args.v2_payload, weights=weights, threshold=threshold)
        v2_results["source_chunker_for_weights"] = chunker_choice
        v2_results["weights"] = weights_dict
        v2_results["threshold"] = threshold

    payload = {"v1": v1_results, "v2": v2_results, "fit_step": args.fit_step, "target_fp": args.target_fp}
    json_path = _ARTIFACTS / f"{args.out_tag}.json"
    md_path = _ARTIFACTS / f"{args.out_tag}.md"
    json_path.write_text(json.dumps(payload, indent=2))

    lines: List[str] = [
        f"# Composite phantom-guard score — `{args.out_tag}`",
        "",
        f"- fit step: `{args.fit_step}`",
        f"- target FP (false-phantom on grounded): `{args.target_fp}`",
        "",
        "## v1 fit",
        "",
        "| chunker | w_rc | w_pt | w_lg | AUROC composite | AUROC rc | AUROC p10 | AUROC lg | threshold | actual FP | recall |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for ck, res in v1_results["per_chunker"].items():
        if "error" in res:
            lines.append(f"| {ck} | - | - | - | error | - | - | - | - | - | - |")
            continue
        w = res["weights"]
        b = res["baselines"]
        lk = res["threshold_lock"]
        lines.append(
            "| {ck} | {wrc:.2f} | {wpt:.2f} | {wlg:.2f} | **{ca:.3f}** | {ba:.3f} | {bp:.3f} | {bl:.3f} | {th:.4f} | {fp:.3f} | {rc:.3f} |".format(
                ck=ck,
                wrc=w["w_rc"],
                wpt=w["w_pt"],
                wlg=w["w_lg"],
                ca=res["composite_auroc_v1"],
                ba=b["rc_alone_auroc"],
                bp=b["p10_alone_auroc"],
                bl=b["lg_alone_auroc"],
                th=lk["threshold"],
                fp=lk["actual_fp"],
                rc=lk["recall"],
            )
        )

    if v2_results is not None and "error" not in v2_results:
        lines.extend(
            [
                "",
                "## v2 application (weights fit on v1)",
                "",
                f"- weights source chunker: `{v2_results['source_chunker_for_weights']}`",
                f"- weights: `{v2_results['weights']}`",
                f"- threshold: `{v2_results['threshold']:.4f}`",
                "",
                f"- AUROC correct vs wrong:     **{v2_results['auroc_correct_vs_wrong']}**",
                f"- AUROC correct vs ambiguous: {v2_results['auroc_correct_vs_ambiguous']}",
                f"- counts: {v2_results['counts_by_tier']}",
                f"- flagged at v1 threshold: {v2_results['flagged_at_v1_threshold']}",
                "",
                "### v2-locked threshold sweep (wrong = positive)",
                "",
                "| target FP | threshold | actual FP | recall | flagged wrong | flagged correct |",
                "|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in v2_results.get("v2_threshold_sweep") or []:
            lines.append(
                "| {tfp:.2f} | {th:.4f} | {afp:.3f} | {r:.3f} | {fw} | {fc} |".format(
                    tfp=row.get("target_fp", 0.0),
                    th=row.get("threshold", 0.0),
                    afp=row.get("actual_fp", 0.0),
                    r=row.get("recall", 0.0),
                    fw=row.get("flagged_phantoms", 0),
                    fc=row.get("flagged_correct", 0),
                )
            )
        lines.extend(
            [
                "",
            ]
        )
    elif v2_results is not None:
        lines.extend(["", f"## v2 application — error: {v2_results['error']}"])

    md_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
