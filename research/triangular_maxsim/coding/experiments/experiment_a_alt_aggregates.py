"""Experiment A (isolated): alternative aggregates on the same transcripts_v1 rows.

Loads ``artifacts/transcripts_v1.json`` produced by the main benchmark and
recomputes AUROC / separation / monotonicity using every per-row scalar
aggregate that's already been captured (no re-scoring needed for round 1).

Round 2: also recomputes a worst-token aggregate by re-running the scorer
on the cached encoded cases and keeping per-token reverse_context arrays.
We pack both rounds into one artifact.

Does NOT modify any production code. Writes to
``artifacts/experimentA_alt_aggregates.{json,md}``.
"""
from __future__ import annotations

import json
import math
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, "/workspace/latence-trace")

from research.triangular_maxsim.coding.coding_agent_validation import (  # noqa: E402
    SharedEncodedCase,
    _score_batch_units,
    auroc,
)
from latence_trace.core.groundedness import (  # noqa: E402
    partition_support_units,
    score_groundedness_response_chunked,
)

ROOT = Path("/workspace/latence-trace/research/triangular_maxsim/coding")
INPUT_JSON = ROOT / "artifacts" / "transcripts_v1.json"
OUT_JSON = ROOT / "artifacts" / "experimentA_alt_aggregates.json"
OUT_MD = ROOT / "artifacts" / "experimentA_alt_aggregates.md"
CACHE_DIR = ROOT / "cache"


# --- Round 1 helpers ----------------------------------------------------------

AGG_FIELDS: Tuple[Tuple[str, str, bool], ...] = (
    # (name in scores dict, label, higher_is_more_grounded)
    ("reverse_context", "reverse_context (baseline)", True),
    ("reverse_context_calibrated", "reverse_context_calibrated", True),
    ("literal_guarded", "literal_guarded", True),
    ("consensus_hardened", "consensus_hardened", True),
    ("triangular", "triangular", True),
    ("echo_mean", "echo_mean", True),
    ("grounded_coverage", "grounded_coverage", True),
    ("reverse_query_context", "reverse_query_context", True),
    ("groundedness_v2", "groundedness_v2", True),
    ("max_top_evidence_score", "max_top_evidence_score", True),
    ("context_usage_ratio", "context_usage_ratio", True),
    ("context_attribution_ratio", "context_attribution_ratio", True),
)


def _pull(row: Dict[str, Any], name: str) -> Optional[float]:
    if name == "max_top_evidence_score":
        val = row.get("max_top_evidence_score")
    else:
        val = row["scores"].get(name)
    if val is None:
        return None
    try:
        val = float(val)
    except (TypeError, ValueError):
        return None
    if math.isnan(val) or math.isinf(val):
        return None
    return val


def _auroc(scores: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    if len(scores) < 2 or len(set(labels)) < 2:
        return None
    return auroc(scores, labels)


def _separation(scores: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return None
    return float(sum(pos) / len(pos) - sum(neg) / len(neg))


def _monotonicity(
    rows: List[Dict[str, Any]],
    aggregate_name: str,
    chunker: str,
    scorer_config: str,
) -> Tuple[Optional[float], Optional[float]]:
    """Fraction of bases where score(correct) > score(ambiguous) > score(wrong),
    and partial (correct > wrong)."""

    by_base: Dict[str, Dict[str, float]] = {}
    for row in rows:
        if row["chunker"] != chunker or row["scorer_config"] != scorer_config:
            continue
        base = row.get("base_scenario_id")
        tier = row.get("tier")
        if not base or not tier:
            continue
        value = _pull(row, aggregate_name)
        if value is None:
            continue
        by_base.setdefault(base, {})[tier] = value

    full_hits = 0
    full_total = 0
    partial_hits = 0
    partial_total = 0
    for tiers in by_base.values():
        if "correct" in tiers and "wrong" in tiers:
            partial_total += 1
            if tiers["correct"] > tiers["wrong"]:
                partial_hits += 1
        if {"correct", "ambiguous", "wrong"}.issubset(tiers):
            full_total += 1
            if tiers["correct"] > tiers["ambiguous"] > tiers["wrong"]:
                full_hits += 1
    full = full_hits / full_total if full_total else None
    partial = partial_hits / partial_total if partial_total else None
    return full, partial


def evaluate_round1(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """For every (chunker, scorer_config, aggregate) triple, compute metrics."""

    cells = sorted({(r["chunker"], r["scorer_config"]) for r in rows})
    results: List[Dict[str, Any]] = []
    for chunker, scorer in cells:
        cell_rows = [r for r in rows if r["chunker"] == chunker and r["scorer_config"] == scorer]
        anchor_rows = [r for r in cell_rows if r.get("label") in {"grounded", "ungrounded"}]
        correct_rows = [r for r in cell_rows if r.get("tier") == "correct"]
        wrong_rows = [r for r in cell_rows if r.get("tier") == "wrong"]
        ambi_rows = [r for r in cell_rows if r.get("tier") == "ambiguous"]
        for agg_name, agg_label, higher in AGG_FIELDS:
            def _collect(bucket: List[Dict[str, Any]]) -> Tuple[List[float], List[int]]:
                vals, labs = [], []
                for row in bucket:
                    v = _pull(row, agg_name)
                    if v is None:
                        continue
                    vals.append(v if higher else -v)
                    if row.get("label") == "grounded":
                        lab = 1
                    elif row.get("label") == "ungrounded":
                        lab = 0
                    else:
                        lab = -1
                    labs.append(lab)
                return vals, labs

            anchor_vals, anchor_labs = _collect(anchor_rows)
            anchor_vals_for_auroc = [v for v, l in zip(anchor_vals, anchor_labs) if l in (0, 1)]
            anchor_labs_for_auroc = [l for l in anchor_labs if l in (0, 1)]
            anchor_auroc = _auroc(anchor_vals_for_auroc, anchor_labs_for_auroc)
            sep = _separation(anchor_vals_for_auroc, anchor_labs_for_auroc)

            cw_vals, cw_labs = [], []
            for row in correct_rows + wrong_rows:
                v = _pull(row, agg_name)
                if v is None:
                    continue
                cw_vals.append(v if higher else -v)
                cw_labs.append(1 if row.get("tier") == "correct" else 0)
            cw_auroc = _auroc(cw_vals, cw_labs)

            ca_vals, ca_labs = [], []
            for row in correct_rows + ambi_rows:
                v = _pull(row, agg_name)
                if v is None:
                    continue
                ca_vals.append(v if higher else -v)
                ca_labs.append(1 if row.get("tier") == "correct" else 0)
            ca_auroc = _auroc(ca_vals, ca_labs)

            mono, partial = _monotonicity(rows, agg_name, chunker, scorer)

            results.append(
                {
                    "chunker": chunker,
                    "scorer_config": scorer,
                    "aggregate": agg_label,
                    "agg_field": agg_name,
                    "higher_is_grounded": higher,
                    "auroc_grounded_vs_ungrounded": anchor_auroc,
                    "separation": sep,
                    "auroc_correct_vs_wrong": cw_auroc,
                    "auroc_correct_vs_ambiguous": ca_auroc,
                    "monotonicity": mono,
                    "partial_monotonicity": partial,
                    "n_anchor": len(anchor_labs_for_auroc),
                    "n_correct": len(correct_rows),
                    "n_wrong": len(wrong_rows),
                    "n_ambiguous": len(ambi_rows),
                }
            )
    return results


# --- Round 2 helpers: rescoring for per-token worst-k aggregates ---------------


def _latest_cache_path(chunker: str) -> Optional[Path]:
    glob_prefix = f"coding_shared_v5__{chunker}__dual__cp_gte__bank_transcripts_v1__"
    candidates = sorted(CACHE_DIR.glob(f"{glob_prefix}*.pt"))
    if not candidates:
        return None
    return candidates[-1]


def _pertoken_aggregates(per_token: Sequence[float]) -> Dict[str, float]:
    if not per_token:
        return {"min": 0.0, "p5": 0.0, "p10": 0.0, "mean_weakest_5": 0.0, "mean_minus_min": 0.0}
    sorted_scores = sorted(float(x) for x in per_token if x is not None and not (isinstance(x, float) and (math.isnan(x) or math.isinf(x))))
    if not sorted_scores:
        return {"min": 0.0, "p5": 0.0, "p10": 0.0, "mean_weakest_5": 0.0, "mean_minus_min": 0.0}
    n = len(sorted_scores)
    mn = sorted_scores[0]
    mean = sum(sorted_scores) / n
    p5 = sorted_scores[max(0, int(0.05 * n) - 1)] if n >= 20 else sorted_scores[0]
    p10 = sorted_scores[max(0, int(0.10 * n) - 1)] if n >= 10 else sorted_scores[0]
    weakest_k = sorted_scores[: min(5, n)]
    mean_weakest = sum(weakest_k) / len(weakest_k)
    return {
        "min": mn,
        "p5": p5,
        "p10": p10,
        "mean_weakest_5": mean_weakest,
        "mean_minus_min": mean - mn,
    }


def evaluate_round2(rows_by_chunker: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Re-score each cached encoded case and compute per-token aggregates.

    Only runs the GTE scorer (best baseline). Scores all 20 cases on each
    chunker, captures per-token reverse_context, and computes alt aggregates.
    """

    results: List[Dict[str, Any]] = []
    for chunker, prior_rows in rows_by_chunker.items():
        cache_path = _latest_cache_path(chunker)
        if cache_path is None or not cache_path.exists():
            print(f"[round2] no cache for {chunker!r}, skipping")
            continue
        print(f"[round2] {chunker}: loading {cache_path.name} ...")
        with open(cache_path, "rb") as handle:
            blob = pickle.load(handle)
        encoded_cases: List[SharedEncodedCase] = blob["encoded_cases"]

        by_id: Dict[str, Dict[str, Any]] = {row["id"]: row for row in prior_rows}

        t0 = time.perf_counter()
        collected: List[Dict[str, Any]] = []
        for case in encoded_cases:
            sus = case.support_units_by_encoder["gte"]
            rcs = case.response_chunks_by_encoder["gte"]
            qe = case.query_embeddings_by_encoder.get("gte")
            qt = case.query_tokens_by_encoder.get("gte")
            batches = partition_support_units(sus, batch_size=_score_batch_units())
            scored = score_groundedness_response_chunked(
                response_chunks=rcs,
                support_batches=batches,
                response_text=case.case.response,
                query_embeddings=qe,
                query_tokens=qt,
                evidence_limit=5,
                primary_metric="reverse_context",
                debug_dense_matrices=False,
                structured_enabled=False,
            )
            per_token = [
                float(tok["reverse_context"])
                for tok in scored.get("response_tokens") or []
                if tok.get("reverse_context") is not None
            ]
            agg = _pertoken_aggregates(per_token)
            prior = by_id.get(case.case.id, {})
            collected.append(
                {
                    "id": case.case.id,
                    "chunker": chunker,
                    "scorer_config": "gte_only",
                    "label": prior.get("label"),
                    "tier": prior.get("tier"),
                    "base_scenario_id": prior.get("base_scenario_id"),
                    "n_tokens": len(per_token),
                    **{f"agg_{k}": v for k, v in agg.items()},
                }
            )
        dt = time.perf_counter() - t0
        print(f"[round2] {chunker}: {len(collected)} cases rescored in {dt:.1f}s")

        # Evaluate each of the 5 aggregates
        alt_names = ("min", "p5", "p10", "mean_weakest_5", "mean_minus_min")
        correct = [r for r in collected if r.get("tier") == "correct"]
        wrong = [r for r in collected if r.get("tier") == "wrong"]
        ambi = [r for r in collected if r.get("tier") == "ambiguous"]
        grounded = [r for r in collected if r.get("label") == "grounded"]
        ungrounded = [r for r in collected if r.get("label") == "ungrounded"]
        for name in alt_names:
            field = f"agg_{name}"
            # higher = more grounded (these measure minimum similarity; higher = better)
            def _vals(bucket):
                return [float(r[field]) for r in bucket if r.get(field) is not None]

            vg, vu = _vals(grounded), _vals(ungrounded)
            auc = _auroc(vg + vu, [1] * len(vg) + [0] * len(vu))
            sep = (sum(vg) / len(vg) - sum(vu) / len(vu)) if vg and vu else None
            vc, vw = _vals(correct), _vals(wrong)
            cw_auc = _auroc(vc + vw, [1] * len(vc) + [0] * len(vw))
            va = _vals(ambi)
            ca_auc = _auroc(vc + va, [1] * len(vc) + [0] * len(va))

            by_base: Dict[str, Dict[str, float]] = {}
            for row in collected:
                base = row.get("base_scenario_id")
                tier = row.get("tier")
                if not base or not tier or row.get(field) is None:
                    continue
                by_base.setdefault(base, {})[tier] = float(row[field])
            full_hits = full_total = partial_hits = partial_total = 0
            for tiers in by_base.values():
                if "correct" in tiers and "wrong" in tiers:
                    partial_total += 1
                    if tiers["correct"] > tiers["wrong"]:
                        partial_hits += 1
                if {"correct", "ambiguous", "wrong"}.issubset(tiers):
                    full_total += 1
                    if tiers["correct"] > tiers["ambiguous"] > tiers["wrong"]:
                        full_hits += 1

            results.append(
                {
                    "chunker": chunker,
                    "scorer_config": "gte_only",
                    "aggregate": f"per_token_{name}",
                    "agg_field": field,
                    "higher_is_grounded": True,
                    "auroc_grounded_vs_ungrounded": auc,
                    "separation": sep,
                    "auroc_correct_vs_wrong": cw_auc,
                    "auroc_correct_vs_ambiguous": ca_auc,
                    "monotonicity": (full_hits / full_total) if full_total else None,
                    "partial_monotonicity": (partial_hits / partial_total) if partial_total else None,
                    "n_anchor": len(vg) + len(vu),
                    "n_correct": len(correct),
                    "n_wrong": len(wrong),
                    "n_ambiguous": len(ambi),
                }
            )
    return results


# --- Rendering ----------------------------------------------------------------


def _fmt(x: Optional[float], nd: int = 4) -> str:
    if x is None:
        return "n/a"
    return f"{x:.{nd}f}"


def render_markdown(round1: List[Dict[str, Any]], round2: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append("# Experiment A (isolated): alternative aggregates on transcripts_v1")
    lines.append("")
    lines.append("> Same rows as the production `transcripts_v1` run. Tests whether an aggregate other than `reverse_context` (weighted mean) separates grounded-vs-ungrounded / correct-vs-wrong better. No production code changed.")
    lines.append("")
    lines.append("Gate: `AUROC grounded_vs_ungrounded >= 0.90` for at least one (chunker, scorer, aggregate) triple.")
    lines.append("")

    def render_section(rows: List[Dict[str, Any]], title: str) -> None:
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| chunker | scorer | aggregate | AUROC anchor | separation | AUROC c-vs-w | AUROC c-vs-a | full mono | partial mono |")
        lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
        for r in rows:
            lines.append(
                "| {chunker} | `{scorer}` | {agg} | {auc} | {sep} | {cw} | {ca} | {mono} | {pm} |".format(
                    chunker=r["chunker"],
                    scorer=r["scorer_config"],
                    agg=r["aggregate"],
                    auc=_fmt(r["auroc_grounded_vs_ungrounded"]),
                    sep=_fmt(r["separation"], 5),
                    cw=_fmt(r["auroc_correct_vs_wrong"]),
                    ca=_fmt(r["auroc_correct_vs_ambiguous"]),
                    mono=_fmt(r["monotonicity"]),
                    pm=_fmt(r["partial_monotonicity"]),
                )
            )
        lines.append("")

    # Split round1: only show gte_only to keep signal-to-noise high; code_only
    # and fusion rules look like duplicates or dominated shadows of gte_only.
    r1_focus = [r for r in round1 if r["scorer_config"] in {"gte_only"}]
    render_section(r1_focus, "Round 1 — existing scalar aggregates (`gte_only`)")

    render_section(round2, "Round 2 — per-token worst-k aggregates (`gte_only`)")

    # Ranking
    all_rows = round1 + round2
    ranked = sorted(
        (r for r in all_rows if r["auroc_grounded_vs_ungrounded"] is not None),
        key=lambda r: r["auroc_grounded_vs_ungrounded"],
        reverse=True,
    )
    lines.append("## Top 10 aggregates by AUROC grounded-vs-ungrounded")
    lines.append("")
    lines.append("| rank | chunker | scorer | aggregate | AUROC | separation | AUROC c-vs-w |")
    lines.append("|---:|---|---|---|---:|---:|---:|")
    for idx, r in enumerate(ranked[:10], start=1):
        lines.append(
            "| {r} | {chunker} | `{scorer}` | {agg} | {auc} | {sep} | {cw} |".format(
                r=idx,
                chunker=r["chunker"],
                scorer=r["scorer_config"],
                agg=r["aggregate"],
                auc=_fmt(r["auroc_grounded_vs_ungrounded"]),
                sep=_fmt(r["separation"], 5),
                cw=_fmt(r["auroc_correct_vs_wrong"]),
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    with INPUT_JSON.open() as handle:
        data = json.load(handle)
    rows = data["rows"]
    print(f"Loaded {len(rows)} rows from {INPUT_JSON.name}")

    print("\n=== Round 1: existing scalar aggregates ===")
    round1 = evaluate_round1(rows)
    print(f"Round 1: {len(round1)} (cell, aggregate) evaluations")

    print("\n=== Round 2: per-token worst-k aggregates (rescoring) ===")
    rows_by_chunker: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        rows_by_chunker.setdefault(row["chunker"], []).append(row)
    round2 = evaluate_round2(rows_by_chunker)
    print(f"Round 2: {len(round2)} evaluations")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w") as handle:
        json.dump(
            {
                "source": str(INPUT_JSON),
                "round1": round1,
                "round2": round2,
            },
            handle,
            indent=2,
        )
    print(f"Wrote {OUT_JSON}")

    md = render_markdown(round1, round2)
    OUT_MD.write_text(md)
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
