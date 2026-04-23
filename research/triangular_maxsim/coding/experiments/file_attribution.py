"""File-level attribution: which context files are dead weight this turn?

Given a scored groundedness payload (one case → many ``support_units``
with ``metadata['path']`` and ``usage_state``), we roll up to files and
emit a single ``file_usage_score`` per file. A file is flagged
``dead_weight`` if its score is below a threshold *and* it contributes
zero unit with ``usage_state == "used"``.

Scoring per file
----------------
For each file ``f`` with support units ``U_f``:

    used_f       = sum(u.usage_state == "used")
    total_f      = len(U_f)
    max_evidence = max(u.score)             # best MaxSim this turn
    coverage_f   = used_f / max(1, total_f)
    dead_weight  = coverage_f < threshold AND used_f == 0

Public API
----------
``attribute_file_usage(scored_payload, *, dead_weight_threshold)``
returns::

    {
        "per_file": [{"path": ..., "n_units": ..., "used": ..., "coverage": ...,
                      "max_evidence": ..., "dead_weight": bool}, ...],
        "dead_weight_files": [paths...],
        "dead_weight_ratio": float,
    }

The module also exposes a CLI that accepts an existing runner JSON
(``--case-bank transcripts_v2``) and writes a per-case file-attribution
report.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


_ROOT = Path("/workspace/latence-trace/research/triangular_maxsim/coding")
_ARTIFACTS = _ROOT / "artifacts"


def _file_path_for_unit(unit: Dict[str, Any]) -> Optional[str]:
    """Extract the file path the support unit originates from."""
    meta = unit.get("metadata") or {}
    path = meta.get("path") or unit.get("chunk_id")
    return str(path) if path else None


def _file_usage_from_units(
    support_units: Sequence[Dict[str, Any]],
    *,
    dead_weight_threshold: float,
    total_response_tokens: Optional[int] = None,
    min_owner_share: float = 0.01,
) -> Dict[str, Any]:
    """Roll up support-unit stats to per-file attribution.

    Two complementary signals are tracked per file:

    - **coverage** = fraction of units tagged ``usage_state == "used"``.
      Saturates easily on permissive thresholds (most units hit cosine ~1
      somewhere against a large response) so only used as a secondary
      guard.
    - **owner_share** = fraction of response tokens whose *global* argmax
      across all support units lands inside any unit of this file. This
      directly answers "did this file contribute evidence to any response
      token?" — robust to per-unit saturation.

    A file is flagged ``dead_weight`` when
    ``owner_share < min_owner_share`` AND ``used == 0`` (both ownership
    and permissive coverage agree the file is unused).
    """
    buckets: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "n_units": 0,
            "used": 0,
            "uncertain": 0,
            "unused": 0,
            "max_evidence": 0.0,
            "sum_score": 0.0,
            "owner_tokens": 0,
        }
    )
    total_owner_tokens = 0
    for unit in support_units:
        path = _file_path_for_unit(unit)
        if path is None:
            continue
        bucket = buckets[path]
        bucket["n_units"] += 1
        state = str(unit.get("usage_state") or "").lower()
        if state == "used":
            bucket["used"] += 1
        elif state == "uncertain":
            bucket["uncertain"] += 1
        else:
            bucket["unused"] += 1
        score = float(unit.get("score") or 0.0)
        bucket["sum_score"] += score
        if score > bucket["max_evidence"]:
            bucket["max_evidence"] = score
        owner_count = int(unit.get("owner_count") or 0)
        bucket["owner_tokens"] += owner_count
        total_owner_tokens += owner_count

    # Denominator for owner_share: prefer the explicit response token
    # count (from the scorer). Fall back to the sum of owner tokens which
    # approximates the response length when every token has an owner.
    denom = int(total_response_tokens) if total_response_tokens else total_owner_tokens
    denom = max(1, denom)

    per_file: List[Dict[str, Any]] = []
    dead_weight_files: List[str] = []
    dw_by_coverage: List[str] = []
    for path, b in buckets.items():
        n = max(1, int(b["n_units"]))
        coverage = float(b["used"]) / n
        mean_score = b["sum_score"] / n
        owner_share = float(b["owner_tokens"]) / denom
        # Primary (strict) flag: no response token's global argmax ever
        # lands inside this file. Robust to per-unit cosine saturation.
        is_dead = owner_share < min_owner_share
        # Secondary (legacy) flag retained for transparency: permissive
        # coverage below threshold AND no "used" units.
        is_dead_cov = (coverage < dead_weight_threshold) and (b["used"] == 0)
        per_file.append(
            {
                "path": path,
                "n_units": b["n_units"],
                "used": b["used"],
                "uncertain": b["uncertain"],
                "unused": b["unused"],
                "coverage": float(coverage),
                "mean_score": float(mean_score),
                "max_evidence": float(b["max_evidence"]),
                "owner_tokens": int(b["owner_tokens"]),
                "owner_share": float(owner_share),
                "dead_weight": bool(is_dead),
                "dead_weight_by_coverage": bool(is_dead_cov),
            }
        )
        if is_dead:
            dead_weight_files.append(path)
        if is_dead_cov:
            dw_by_coverage.append(path)

    per_file.sort(key=lambda r: (r["owner_share"], r["coverage"]))
    total = max(1, len(per_file))
    return {
        "per_file": per_file,
        "dead_weight_files": dead_weight_files,
        "dead_weight_files_by_coverage": dw_by_coverage,
        "dead_weight_ratio": float(len(dead_weight_files)) / total,
        "dead_weight_ratio_by_coverage": float(len(dw_by_coverage)) / total,
        "n_files": len(per_file),
        "n_response_tokens": int(total_response_tokens or total_owner_tokens),
        "min_owner_share": float(min_owner_share),
        "coverage_threshold": float(dead_weight_threshold),
    }


def attribute_file_usage(
    scored_payload: Dict[str, Any],
    *,
    dead_weight_threshold: float = 0.20,
    min_owner_share: float = 0.01,
) -> Dict[str, Any]:
    """Accept either a raw ``score_groundedness_response_chunked`` output
    (with ``support_units`` at top level) or a runner row."""
    units = (
        scored_payload.get("support_units")
        or scored_payload.get("support_units_detail")
        or []
    )
    if not units and "scored" in scored_payload:
        units = scored_payload["scored"].get("support_units") or []
    total_tokens = (
        scored_payload.get("n_response_tokens")
        or (scored_payload.get("scores") or {}).get("n_tokens")
    )
    return _file_usage_from_units(
        units,
        dead_weight_threshold=dead_weight_threshold,
        total_response_tokens=total_tokens,
        min_owner_share=min_owner_share,
    )


def attribute_run_payload(
    payload: Dict[str, Any],
    *,
    dead_weight_threshold: float = 0.20,
    min_owner_share: float = 0.01,
) -> Dict[str, Any]:
    """Operate on a full runner JSON. For each row attaches the
    file-level attribution under ``row['file_attribution']`` and returns
    a corpus-level summary."""
    rows = payload.get("rows") or []
    if not rows:
        return {"error": "no rows in payload"}

    per_case_summaries: List[Dict[str, Any]] = []
    all_dw_ratios: List[float] = []
    for row in rows:
        units = (row.get("scores") or {}).get("support_units_detail")
        if units is None:
            units = row.get("support_units") or []
        # The GPU rescore payload stores the per-unit records inside
        # ``row['file_attribution']['unit_records']``. Use those when no
        # higher-priority source is available so the CLI can re-roll with
        # a different threshold.
        if not units:
            pre = row.get("file_attribution") or {}
            units = pre.get("unit_records") or []
        # Fall back to ids-only held_out if no detail available.
        if not units:
            held = row.get("held_out") or {}
            ids_unused = set(held.get("held_out_ids") or [])
            ids_used = set(held.get("used_ids") or [])
            ids_uncertain = set(held.get("uncertain_ids") or [])
            units = []
            for sid in ids_unused | ids_used | ids_uncertain:
                state = (
                    "used"
                    if sid in ids_used
                    else "uncertain"
                    if sid in ids_uncertain
                    else "unused"
                )
                units.append(
                    {
                        "support_id": sid,
                        "chunk_id": sid,
                        "usage_state": state,
                        "score": 0.0,
                        "metadata": {"path": sid.split("-")[0]},
                    }
                )

        total_tokens = (row.get("scores") or {}).get("n_tokens")
        attribution = _file_usage_from_units(
            units,
            dead_weight_threshold=dead_weight_threshold,
            total_response_tokens=total_tokens,
            min_owner_share=min_owner_share,
        )
        row["file_attribution"] = attribution
        per_case_summaries.append(
            {
                "id": row.get("id"),
                "tier": row.get("tier") or row.get("subcategory"),
                "chunker": row.get("chunker"),
                "scorer": row.get("scorer_config"),
                "dead_weight_files": attribution["dead_weight_files"],
                "dead_weight_ratio": attribution["dead_weight_ratio"],
                "n_files": attribution["n_files"],
            }
        )
        all_dw_ratios.append(attribution["dead_weight_ratio"])

    return {
        "per_case": per_case_summaries,
        "mean_dead_weight_ratio": (
            float(sum(all_dw_ratios)) / len(all_dw_ratios) if all_dw_ratios else 0.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_json", type=Path)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.20,
        help="coverage threshold under which a file is flagged dead weight (default 0.20)",
    )
    parser.add_argument(
        "--min-owner-share",
        type=float,
        default=0.01,
        help=(
            "minimum share of response tokens a file must own (global argmax) "
            "to escape dead-weight flag (default 0.01 ~= 1 token in a 100-token "
            "response)"
        ),
    )
    parser.add_argument("--out-tag", default=None)
    args = parser.parse_args()

    with args.input_json.open() as handle:
        payload = json.load(handle)

    result = attribute_run_payload(
        payload,
        dead_weight_threshold=args.threshold,
        min_owner_share=args.min_owner_share,
    )

    tag = args.out_tag or f"{args.input_json.stem}__file_attribution"
    out_json = _ARTIFACTS / f"{tag}.json"
    out_json.write_text(json.dumps(result, indent=2))
    md_lines = [
        f"# File-level attribution — `{tag}`",
        "",
        f"- input: `{args.input_json.name}`",
        f"- dead-weight threshold: `{args.threshold:.2f}`",
        f"- mean dead-weight ratio: **{result.get('mean_dead_weight_ratio', 0):.3f}**",
        "",
        "## Per-case (first 30)",
        "",
        "| id | tier | chunker | scorer | #files | dw ratio | dead-weight paths (first 3) |",
        "|---|---|---|---|---:|---:|---|",
    ]
    for entry in (result.get("per_case") or [])[:30]:
        dw_preview = ", ".join(entry["dead_weight_files"][:3]) or "-"
        md_lines.append(
            f"| {entry['id']} | {entry['tier']} | {entry['chunker']} | {entry['scorer']} | "
            f"{entry['n_files']} | {entry['dead_weight_ratio']:.3f} | {dw_preview} |"
        )
    (_ARTIFACTS / f"{tag}.md").write_text("\n".join(md_lines) + "\n")
    print(f"Wrote {out_json}")
    print(f"Wrote {_ARTIFACTS / (tag + '.md')}")


if __name__ == "__main__":
    main()
