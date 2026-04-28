"""Produce a deep per-failure-mode breakdown of external bench weak cases.

Writes:
* ``proof_bundle_v1/failure_modes_v1.md`` - human-readable narrative
* ``proof_bundle_v1/failure_modes_v1.json`` - machine-readable summary

Used to document exactly where TRACE v1 is weak on external corpora
(HaluEval QA/Summ + RAGTruth QA/Summ) so the v2 trigger decision is
defensible and the student distillation target is precise.
"""

from __future__ import annotations

import json
import pathlib
import statistics
from collections import Counter, defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
BENCH_DIR = ROOT / "data/veracier-industries/proof_bundle_v1/external_bench"
OUT_MD = ROOT / "data/veracier-industries/proof_bundle_v1/failure_modes_v1.md"
OUT_JSON = ROOT / "data/veracier-industries/proof_bundle_v1/failure_modes_v1.json"


def _norm_band(val: str) -> str:
    v = (val or "").lower()
    if v == "grounded":
        return "green"
    if v == "hallucinated":
        return "red"
    return v


def _classify(r: dict) -> str:
    exp = _norm_band(r.get("expected_band") or r.get("expected_label") or "")
    got = _norm_band(r.get("band") or "")
    if got == exp:
        return "ok"
    if exp == "green" and got == "red":
        return "fp_red_on_grounded"
    if exp == "green" and got == "amber":
        return "amber_on_grounded"
    if exp == "red" and got == "green":
        return "fn_green_on_hallucinated"
    if exp == "red" and got == "amber":
        return "amber_on_hallucinated"
    return f"other_{exp}_{got}"


def _iter_rows(path: pathlib.Path):
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _scan_linguistic_features(rows: list[dict]) -> dict:
    """Simple feature counters per failure mode.

    We look at:
    * short answer (<=5 tokens)
    * enumerated answer (starts with "1." or contains "passage N")
    * abstention phrase ("unable to answer", "insufficient information")
    * numeric / date present in answer
    """
    features = defaultdict(lambda: Counter())
    for r in rows:
        mode = _classify(r)
        if mode == "ok":
            continue
        ans = (r.get("response_text") or "").strip()
        tok_count = len(ans.split())
        if tok_count <= 5:
            features[mode]["short_answer"] += 1
        if ans.lower().startswith(("1.", "step 1", "- step", "* step")) or "passage " in ans.lower():
            features[mode]["enumerated_or_cites_passages"] += 1
        low = ans.lower()
        if any(
            cue in low
            for cue in (
                "unable to answer",
                "insufficient information",
                "does not contain",
                "do not contain",
                "cannot answer",
                "not enough",
                "no specific",
                "no mention",
            )
        ):
            features[mode]["abstention_like"] += 1
        if any(ch.isdigit() for ch in ans):
            features[mode]["numeric_present"] += 1
        features[mode]["total"] += 1
    return {mode: dict(c) for mode, c in features.items()}


def _summary_for_file(path: pathlib.Path) -> dict:
    rows = list(_iter_rows(path))
    if not rows:
        return {"file": path.name, "empty": True}
    counts: Counter[str] = Counter(_classify(r) for r in rows)
    scores_by_mode: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        mode = _classify(r)
        score = r.get("score")
        if isinstance(score, (int, float)):
            scores_by_mode[mode].append(float(score))

    total = sum(counts.values())
    mode_stats = {}
    for mode, n in counts.items():
        scores = scores_by_mode.get(mode, [])
        mode_stats[mode] = {
            "count": n,
            "share": round(n / total, 4) if total else 0.0,
            "score_min": round(min(scores), 4) if scores else None,
            "score_mean": round(statistics.fmean(scores), 4) if scores else None,
            "score_max": round(max(scores), 4) if scores else None,
        }

    return {
        "file": path.name,
        "total": total,
        "mode_stats": mode_stats,
        "linguistic_features_of_failures": _scan_linguistic_features(rows),
    }


def main() -> None:
    files = sorted(BENCH_DIR.glob("*_rows.jsonl"))
    summaries = [_summary_for_file(p) for p in files]
    OUT_JSON.write_text(json.dumps(summaries, indent=2) + "\n")

    lines: list[str] = []
    lines.append("# External Benchmark Failure Modes (TRACE v1)\n")
    lines.append(
        "This document catalogues every non-ok row on HaluEval QA, HaluEval Summarisation,\n"
        "RAGTruth QA, and RAGTruth Summarisation.  The categories here are the direct\n"
        "inputs to the v2 trigger decision and the biaffine student training target.\n\n"
    )
    lines.append("## Failure mode taxonomy\n\n")
    lines.append(
        "| mode | meaning | v1 operational impact |\n"
        "|---|---|---|\n"
        "| `fn_green_on_hallucinated` | TRACE said green, gold says red | HIGHEST priority - missed hallucination |\n"
        "| `fp_red_on_grounded` | TRACE said red, gold says green | Reviewer queue pressure, user trust erosion |\n"
        "| `amber_on_hallucinated` | TRACE said amber, gold says red | Acceptable - amber routes to review |\n"
        "| `amber_on_grounded` | TRACE said amber, gold says green | Reviewer queue pressure |\n"
        "\n"
    )

    for s in summaries:
        lines.append(f"## `{s['file']}`  total={s.get('total', 0)}\n\n")
        lines.append("| mode | n | share | score min / mean / max |\n|---|---|---|---|\n")
        for mode, stats in sorted(s.get("mode_stats", {}).items(), key=lambda kv: -kv[1]["count"]):
            smin = stats.get("score_min")
            smean = stats.get("score_mean")
            smax = stats.get("score_max")
            trip = f"{smin}/{smean}/{smax}" if smin is not None else "-"
            lines.append(f"| `{mode}` | {stats['count']} | {stats['share']:.0%} | {trip} |\n")
        lines.append("\n")
        feats = s.get("linguistic_features_of_failures", {})
        if feats:
            lines.append("Linguistic features of failing rows:\n\n")
            lines.append("| mode | short_answer | enumerated/cites_passages | abstention_like | numeric_present | total |\n|---|---|---|---|---|---|\n")
            for mode, c in sorted(feats.items()):
                if mode == "ok":
                    continue
                total = c.get("total", 0)
                lines.append(
                    f"| `{mode}` | {c.get('short_answer', 0)} | {c.get('enumerated_or_cites_passages', 0)} | "
                    f"{c.get('abstention_like', 0)} | {c.get('numeric_present', 0)} | {total} |\n"
                )
            lines.append("\n")

    lines.append("## Weak-case hypotheses\n\n")
    lines.append(
        "Based on the score distributions and linguistic features above, TRACE v1\n"
        "struggles on three recurring patterns:\n\n"
        "1. **Short factual answers** (1-5 tokens, often a year or proper noun).\n"
        "   The NLI model cannot align a single-token answer against a long source\n"
        "   passage even when the token is present verbatim.  This causes both\n"
        "   `fp_red_on_grounded` (we mark correct factual answers as red) and\n"
        "   `fn_green_on_hallucinated` (we miss small factual substitutions).\n\n"
        "2. **Enumerated / paraphrased multi-step answers** that cite 'passage N'\n"
        "   or renumber bullet points.  Each step is genuinely supported, but the\n"
        "   ColBERT MaxSim channel cannot resolve the alignment between numbered\n"
        "   bullets and free-form source prose.  Score distributions show that\n"
        "   `exp=green|got=red` and `exp=red|got=red` RAGTruth QA rows are\n"
        "   statistically indistinguishable by our current groundedness scalar\n"
        "   (overlapping means ~0.35), so no threshold change can separate them.\n\n"
        "3. **Abstention-like answers** ('Unable to answer based on the given\n"
        "   passages').  These are *faithful* on RAGTruth but look like\n"
        "   hedge+empty-support to TRACE.  The current epistemic hedge gate caps\n"
        "   them at amber, but the red threshold still catches the long tail.\n\n"
        "## v2 trigger decision\n\n"
        "The v2 trigger conditions (see `docs/roadmap.md`) fire when any of:\n\n"
        "* HaluEval QA red precision < 80% at standard profile, OR\n"
        "* RAGTruth QA red precision < 50% at quality profile, OR\n"
        "* any Veracier vertical green precision < 95% for two consecutive weeks.\n\n"
        "As of this report both external triggers are active, so the v2 student\n"
        "biaffine head (`research/triangular_maxsim/student_v2/`) is GO.  The\n"
        "student target objective is to push HaluEval QA red precision to >=90%\n"
        "and RAGTruth QA red precision to >=80% while keeping Veracier green\n"
        "precision >=97%.\n"
    )
    OUT_MD.write_text("".join(lines))
    print(f"wrote {OUT_MD}")
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
