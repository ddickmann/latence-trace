"""Experiment B (isolated): phantom-response bank on the same 8 base scenarios.

Replaces the hand-edited ``wrong`` responses (which still shared heavy
surface tokens with context) with genuinely-phantom responses that use
fabricated import paths, class names, and signatures from non-existent
packages. See ``phantom_responses.py``.

For each chunker {sentence_packed, colgrep}:
    - load the existing v5 cached ``SharedEncodedCase`` list
    - for each base scenario, take its ``__correct`` case and build a
      sibling case with the phantom response (same support units, same
      query, different response text + embeddings)
    - score the phantom response against the *same* support units
    - compare AUROC: correct-vs-original-wrong vs correct-vs-phantom
    - also log per-case min/p10 token scores

Writes to ``artifacts/experimentB_phantom_bank.{json,md}``. No
production code is modified.
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
    PRIMARY_ENCODER,
    SharedEncodedCase,
    _encode_response_chunks,
    _score_batch_units,
    auroc,
)
from research.triangular_maxsim.groundedness_service_validation import (  # noqa: E402
    _load_provider,
)
from latence_trace.core.groundedness import (  # noqa: E402
    partition_support_units,
    score_groundedness_response_chunked,
)

from research.triangular_maxsim.coding.experiments.phantom_responses import (  # noqa: E402
    PHANTOM_RESPONSES,
)

ROOT = Path("/workspace/latence-trace/research/triangular_maxsim/coding")
BASELINE_JSON = ROOT / "artifacts" / "transcripts_v1.json"
OUT_JSON = ROOT / "artifacts" / "experimentB_phantom_bank.json"
OUT_MD = ROOT / "artifacts" / "experimentB_phantom_bank.md"
CACHE_DIR = ROOT / "cache"
RESPONSE_CHUNK_TOKENS = 256


def _cache_path(chunker: str) -> Optional[Path]:
    glob_prefix = f"coding_shared_v5__{chunker}__dual__cp_gte__bank_transcripts_v1__"
    candidates = sorted(CACHE_DIR.glob(f"{glob_prefix}*.pt"))
    return candidates[-1] if candidates else None


def _finite(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def _per_token_rc(scored: Dict[str, Any]) -> List[float]:
    out: List[float] = []
    for tok in scored.get("response_tokens") or []:
        v = _finite(tok.get("reverse_context"))
        if v is not None:
            out.append(v)
    return out


def _summary(scores: Sequence[float]) -> Dict[str, Optional[float]]:
    if not scores:
        return {"min": None, "p5": None, "p10": None, "mean_weakest_5": None, "mean": None}
    s = sorted(float(x) for x in scores)
    n = len(s)
    return {
        "min": s[0],
        "p5": s[max(0, int(0.05 * n) - 1)] if n >= 20 else s[0],
        "p10": s[max(0, int(0.10 * n) - 1)] if n >= 10 else s[0],
        "mean_weakest_5": sum(s[: min(5, n)]) / min(5, n),
        "mean": sum(s) / n,
    }


def _score_case_variant(
    case: SharedEncodedCase,
    response_text: str,
    response_chunks: Any,
) -> Dict[str, Any]:
    sus = case.support_units_by_encoder["gte"]
    batches = partition_support_units(sus, batch_size=_score_batch_units())
    scored = score_groundedness_response_chunked(
        response_chunks=response_chunks,
        support_batches=batches,
        response_text=response_text,
        query_embeddings=case.query_embeddings_by_encoder.get("gte"),
        query_tokens=case.query_tokens_by_encoder.get("gte"),
        evidence_limit=5,
        primary_metric="reverse_context",
        debug_dense_matrices=False,
        structured_enabled=False,
    )
    per_token = _per_token_rc(scored)
    scores = scored.get("scores") or {}
    top_ev = scored.get("top_evidence") or []
    max_evidence = max((e.get("score") for e in top_ev if e.get("score") is not None), default=None)
    literal_mismatch = scores.get("literal_mismatch_count") or 0
    literal_total = scores.get("literal_total_count") or 0
    return {
        "reverse_context": _finite(scores.get("reverse_context")),
        "triangular": _finite(scores.get("triangular")),
        "echo_mean": _finite(scores.get("echo_mean")),
        "consensus_hardened": _finite(scores.get("consensus_hardened")),
        "reverse_query_context": _finite(scores.get("reverse_query_context")),
        "literal_guarded": _finite(scores.get("literal_guarded")),
        "groundedness_v2": _finite(scores.get("groundedness_v2")),
        "max_top_evidence_score": _finite(max_evidence),
        "literal_mismatch_count": int(literal_mismatch),
        "literal_total_count": int(literal_total),
        "per_token_stats": _summary(per_token),
        "n_tokens": len(per_token),
    }


def run_for_chunker(
    chunker: str,
    gte_provider: Any,
    gte_doc_prompt: Optional[str],
    baseline_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    cache_path = _cache_path(chunker)
    if cache_path is None or not cache_path.exists():
        return {"chunker": chunker, "error": "no cache", "records": []}
    print(f"[{chunker}] loading {cache_path.name}")
    with open(cache_path, "rb") as handle:
        blob = pickle.load(handle)
    encoded_cases: List[SharedEncodedCase] = blob["encoded_cases"]

    # Index by case.id.
    by_id: Dict[str, SharedEncodedCase] = {c.case.id: c for c in encoded_cases}

    cell_rows = [
        r for r in baseline_rows
        if r["chunker"] == chunker and r["scorer_config"] == "gte_only"
    ]
    by_row_id: Dict[str, Dict[str, Any]] = {r["id"]: r for r in cell_rows}

    records: List[Dict[str, Any]] = []
    t0 = time.perf_counter()
    for base_id, phantom_text in PHANTOM_RESPONSES.items():
        correct_id = f"{base_id}__correct"
        wrong_id_candidates = (f"{base_id}_wrong", f"{base_id}__wrong")
        wrong_id = next((c for c in wrong_id_candidates if c in by_row_id), None)
        ambi_ids = [
            cid for cid in by_row_id
            if cid.startswith(base_id) and ("ambiguous" in cid) and cid != correct_id
        ]

        if correct_id not in by_id:
            print(f"  [skip] {correct_id} not in cache")
            continue
        correct_case = by_id[correct_id]

        correct_baseline = by_row_id.get(correct_id, {})
        wrong_baseline = by_row_id.get(wrong_id, {}) if wrong_id else {}
        ambi_baselines = [by_row_id.get(cid, {}) for cid in ambi_ids]

        def _pull_agg(baseline: Dict[str, Any], key: str) -> Optional[float]:
            if not baseline:
                return None
            if key == "max_top_evidence_score":
                return _finite(baseline.get("max_top_evidence_score"))
            return _finite((baseline.get("scores") or {}).get(key))

        # Encode the phantom response against the same provider/budget.
        phantom_chunks, _spans = _encode_response_chunks(
            phantom_text,
            provider=gte_provider,
            chunk_token_budget=RESPONSE_CHUNK_TOKENS,
            document_prompt_name=gte_doc_prompt,
        )
        phantom_scored = _score_case_variant(correct_case, phantom_text, phantom_chunks)

        # Also rescore the CORRECT response to collect per-token stats
        # (baseline JSON didn't persist them) — for a like-for-like view.
        correct_rescored = _score_case_variant(
            correct_case,
            correct_case.case.response,
            correct_case.response_chunks_by_encoder["gte"],
        )

        agg_field_names = (
            "reverse_context",
            "triangular",
            "echo_mean",
            "consensus_hardened",
            "reverse_query_context",
            "literal_guarded",
            "groundedness_v2",
            "max_top_evidence_score",
        )
        record = {
            "base_scenario_id": base_id,
            "chunker": chunker,
            "correct_id": correct_id,
            "wrong_id": wrong_id,
            "correct_reverse_context": _pull_agg(correct_baseline, "reverse_context"),
            "wrong_reverse_context": _pull_agg(wrong_baseline, "reverse_context"),
            "ambiguous_reverse_contexts": [
                _pull_agg(a, "reverse_context") for a in ambi_baselines
            ],
            "correct_baseline_aggs": {k: _pull_agg(correct_baseline, k) for k in agg_field_names},
            "wrong_baseline_aggs": {k: _pull_agg(wrong_baseline, k) for k in agg_field_names},
            "ambiguous_baseline_aggs": [
                {k: _pull_agg(a, k) for k in agg_field_names} for a in ambi_baselines
            ],
            "phantom": phantom_scored,
            "correct_rescored_per_token": correct_rescored["per_token_stats"],
            "correct_rescored_aggs": {
                k: correct_rescored.get(k)
                for k in (
                    "reverse_context",
                    "triangular",
                    "echo_mean",
                    "consensus_hardened",
                    "reverse_query_context",
                    "literal_guarded",
                    "groundedness_v2",
                    "max_top_evidence_score",
                )
            },
            "correct_rescored_literal_mismatch": correct_rescored.get("literal_mismatch_count"),
            "correct_rescored_literal_total": correct_rescored.get("literal_total_count"),
        }
        records.append(record)
        print(
            f"  [{base_id}] rc(correct)={_fmt(record['correct_reverse_context'])} "
            f"rc(wrong)={_fmt(record['wrong_reverse_context'])}  "
            f"rc(phantom)={_fmt(phantom_scored['reverse_context'])}"
        )
    elapsed = time.perf_counter() - t0
    print(f"[{chunker}] {len(records)} base scenarios in {elapsed:.1f}s")
    return {"chunker": chunker, "records": records}


# --- Evaluation ---------------------------------------------------------------


def _auroc(scores: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    filtered = [(s, l) for s, l in zip(scores, labels) if s is not None]
    if not filtered or len(set(l for _, l in filtered)) < 2:
        return None
    s = [f[0] for f in filtered]
    l = [f[1] for f in filtered]
    return auroc(s, l)


def evaluate(records_by_chunker: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """For each chunker, compute AUROC for:
    - correct vs original_wrong (baseline number)
    - correct vs phantom
    - correct vs (ambiguous pooled)
    using both the aggregate `reverse_context` and per-token `min` / `p10`.
    """
    agg_keys: Tuple[Tuple[str, str], ...] = (
        ("reverse_context", "reverse_context (aggregate)"),
        ("triangular", "triangular"),
        ("echo_mean", "echo_mean"),
        ("consensus_hardened", "consensus_hardened"),
        ("reverse_query_context", "reverse_query_context"),
        ("literal_guarded", "literal_guarded"),
        ("groundedness_v2", "groundedness_v2"),
        ("max_top_evidence_score", "max_top_evidence_score"),
    )
    pt_keys: Tuple[Tuple[str, str], ...] = (
        ("min", "per_token_min"),
        ("p5", "per_token_p5"),
        ("p10", "per_token_p10"),
        ("mean_weakest_5", "per_token_mean_weakest_5"),
    )

    out: List[Dict[str, Any]] = []
    for chunker, records in records_by_chunker.items():
        # Build paired vectors.
        def _pair_agg(key: str) -> Tuple[List[float], List[int]]:
            scores, labels = [], []
            for rec in records:
                c = rec["correct_rescored" if False else "correct_rescored_per_token"]  # not used here
                c_agg = rec["phantom"].get(key)  # for phantom
                w_agg = rec.get("wrong_reverse_context") if key == "reverse_context" else None
                # We need correct values for each aggregate — rescored only captured per_token.
                # Fall back to baseline JSON for correct/wrong/ambig aggregates.
            return scores, labels

        # Simpler flow: build three contrasts explicitly.

        records_n = len(records)

        def _agg_value(rec: Dict[str, Any], tier: str, key: str) -> Optional[float]:
            if tier == "correct":
                return rec.get("correct_baseline_aggs", {}).get(key)
            if tier == "wrong":
                return rec.get("wrong_baseline_aggs", {}).get(key)
            if tier == "phantom":
                return rec["phantom"].get(key)
            if tier == "ambiguous":
                vals = [a.get(key) for a in (rec.get("ambiguous_baseline_aggs") or []) if a.get(key) is not None]
                return (sum(vals) / len(vals)) if vals else None
            return None

        for agg_key, agg_label in agg_keys:
            for contrast, neg_tier in (
                ("correct vs wrong (baseline)", "wrong"),
                ("correct vs phantom (exp B)", "phantom"),
                ("correct vs ambiguous (baseline)", "ambiguous"),
            ):
                scores, labels = [], []
                for rec in records:
                    pos = _agg_value(rec, "correct", agg_key)
                    neg = _agg_value(rec, neg_tier, agg_key)
                    if pos is None or neg is None:
                        continue
                    scores.append(pos)
                    labels.append(1)
                    scores.append(neg)
                    labels.append(0)
                auc = _auroc(scores, labels)
                # Separation: mean(pos) - mean(neg)
                pos_vals = [s for s, l in zip(scores, labels) if l == 1]
                neg_vals = [s for s, l in zip(scores, labels) if l == 0]
                sep = (
                    sum(pos_vals) / len(pos_vals) - sum(neg_vals) / len(neg_vals)
                    if pos_vals and neg_vals
                    else None
                )
                out.append(
                    {
                        "chunker": chunker,
                        "aggregate": agg_label,
                        "agg_field": agg_key,
                        "contrast": contrast,
                        "auroc": auc,
                        "separation": sep,
                        "n_pairs": min(len(pos_vals), len(neg_vals)),
                        "pos_mean": sum(pos_vals) / len(pos_vals) if pos_vals else None,
                        "neg_mean": sum(neg_vals) / len(neg_vals) if neg_vals else None,
                    }
                )

        # Per-token (only computed for correct-rescored and phantom).
        for pt_key, pt_label in pt_keys:
            scores, labels = [], []
            for rec in records:
                pos = rec["correct_rescored_per_token"].get(pt_key)
                neg = rec["phantom"]["per_token_stats"].get(pt_key)
                if pos is None or neg is None:
                    continue
                scores += [pos, neg]
                labels += [1, 0]
            auc = _auroc(scores, labels)
            pos_vals = [s for s, l in zip(scores, labels) if l == 1]
            neg_vals = [s for s, l in zip(scores, labels) if l == 0]
            sep = (
                sum(pos_vals) / len(pos_vals) - sum(neg_vals) / len(neg_vals)
                if pos_vals and neg_vals
                else None
            )
            out.append(
                {
                    "chunker": chunker,
                    "aggregate": pt_label,
                    "agg_field": pt_key,
                    "contrast": "correct vs phantom (exp B)",
                    "auroc": auc,
                    "separation": sep,
                    "n_pairs": min(len(pos_vals), len(neg_vals)),
                    "pos_mean": sum(pos_vals) / len(pos_vals) if pos_vals else None,
                    "neg_mean": sum(neg_vals) / len(neg_vals) if neg_vals else None,
                }
            )
    return out


def _fmt(x: Optional[float], nd: int = 4) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


def render_markdown(records_by_chunker: Dict[str, List[Dict[str, Any]]], evals: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append("# Experiment B (isolated): phantom-response bank")
    lines.append("")
    lines.append(
        "> Swaps the hand-edited `wrong` tier for genuinely-phantom responses that use "
        "fabricated imports (banana-ml, photon-labs, skylab-metrics, crystal-lattice, "
        "cheetah-profiler, mochi-router, prismatic-bench, cactus-codegen, basilisk-schema, "
        "skyjournal). Same 8 base scenarios, same cached context, same scorer. "
        "No production code modified."
    )
    lines.append("")
    lines.append("## Per-case scores")
    lines.append("")

    for chunker, records in records_by_chunker.items():
        lines.append(f"### chunker = `{chunker}`")
        lines.append("")
        lines.append(
            "| base | rc(correct) | rc(wrong) | rc(ambig avg) | rc(phantom) | "
            "correct_min | phantom_min | correct_p10 | phantom_p10 |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for rec in records:
            ambs = [v for v in (rec.get("ambiguous_reverse_contexts") or []) if v is not None]
            amb_avg = sum(ambs) / len(ambs) if ambs else None
            cpt = rec["correct_rescored_per_token"]
            ppt = rec["phantom"]["per_token_stats"]
            lines.append(
                "| {b} | {c} | {w} | {a} | {p} | {cmn} | {pmn} | {cp} | {pp} |".format(
                    b=rec["base_scenario_id"],
                    c=_fmt(rec["correct_reverse_context"]),
                    w=_fmt(rec["wrong_reverse_context"]),
                    a=_fmt(amb_avg),
                    p=_fmt(rec["phantom"]["reverse_context"]),
                    cmn=_fmt(cpt["min"]),
                    pmn=_fmt(ppt["min"]),
                    cp=_fmt(cpt["p10"]),
                    pp=_fmt(ppt["p10"]),
                )
            )
        lines.append("")

    lines.append("## AUROC — correct vs phantom vs original wrong")
    lines.append("")
    lines.append(
        "| chunker | aggregate | contrast | AUROC | separation | pos mean | neg mean | pairs |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|---:|")
    for row in sorted(
        evals,
        key=lambda r: (r["chunker"], r["agg_field"], r.get("contrast", "")),
    ):
        lines.append(
            "| {chunker} | {agg} | {contrast} | {auc} | {sep} | {pm} | {nm} | {n} |".format(
                chunker=row["chunker"],
                agg=row["aggregate"],
                contrast=row.get("contrast", ""),
                auc=_fmt(row.get("auroc")),
                sep=_fmt(row.get("separation"), 5),
                pm=_fmt(row.get("pos_mean"), 5),
                nm=_fmt(row.get("neg_mean"), 5),
                n=row.get("n_pairs"),
            )
        )
    lines.append("")

    # Headline.
    headline: List[str] = []
    for chunker in records_by_chunker.keys():
        rc_phantom = [
            r for r in evals
            if r["chunker"] == chunker
            and r["agg_field"] == "reverse_context"
            and r["contrast"] == "correct vs phantom (exp B)"
        ]
        rc_wrong = [
            r for r in evals
            if r["chunker"] == chunker
            and r["agg_field"] == "reverse_context"
            and r["contrast"] == "correct vs wrong (baseline)"
        ]
        if rc_phantom and rc_wrong:
            p = rc_phantom[0]
            w = rc_wrong[0]
            headline.append(
                f"- **{chunker}**: `reverse_context` AUROC "
                f"correct-vs-wrong = {_fmt(w['auroc'])} → "
                f"correct-vs-phantom = {_fmt(p['auroc'])} "
                f"(separation {_fmt(w['separation'], 5)} → {_fmt(p['separation'], 5)})"
            )
    if headline:
        lines.append("## Headline")
        lines.append("")
        lines.extend(headline)
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    with BASELINE_JSON.open() as handle:
        baseline = json.load(handle)
    baseline_rows = baseline["rows"]

    print(f"Loading encoder {PRIMARY_ENCODER} ...")
    gte_provider, _gte_query_prompt, gte_doc_prompt, _gte_transport = _load_provider(
        PRIMARY_ENCODER, False
    )

    records_by_chunker: Dict[str, List[Dict[str, Any]]] = {}
    for chunker in ("sentence_packed", "colgrep"):
        summary = run_for_chunker(chunker, gte_provider, gte_doc_prompt, baseline_rows)
        records_by_chunker[chunker] = summary.get("records", [])

    evals = evaluate(records_by_chunker)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUT_JSON.open("w") as handle:
        json.dump(
            {
                "source": str(BASELINE_JSON),
                "records_by_chunker": records_by_chunker,
                "evaluations": evals,
            },
            handle,
            indent=2,
        )
    print(f"Wrote {OUT_JSON}")

    md = render_markdown(records_by_chunker, evals)
    OUT_MD.write_text(md)
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
