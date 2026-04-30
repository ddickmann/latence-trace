"""Generate trajectory-native supervision for the agentic TRACE head.

These rows model whether an agent's final claim is supported by the ordered
tool/file/test trace, not whether the prose is semantically similar to a code
context. The banks are non-public internal training/evaluation data.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = ROOT / "research/triangular_maxsim/coding/artifacts"
DEFAULT_OUT_DIR = ARTIFACTS / "trajectory_native_v1"

FAILURE_FAMILIES = (
    "wrong_file",
    "symbol_api_drift",
    "test_outcome_mismatch",
    "patch_result_contradiction",
    "temporal_order_error",
    "partial_support",
    "near_miss_overlap",
)


def _split_for_index(index: int, per_family: int) -> str:
    ratio = index / max(1, per_family)
    if ratio < 0.60:
        return "train"
    if ratio < 0.80:
        return "val"
    return "test"


def _event(kind: str, text: str, **metadata: Any) -> dict[str, Any]:
    return {
        "kind": kind,
        "text": text,
        "metadata": metadata,
    }


def _row(
    *,
    row_id: str,
    base_scenario_id: str,
    split: str,
    label: str,
    failure_family: str,
    claim: str,
    trace_events: list[dict[str, Any]],
    evidence_spans: list[dict[str, Any]],
    expected: dict[str, Any],
    observed: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "trace_trajectory_native_row.v1",
        "row_id": row_id,
        "base_scenario_id": base_scenario_id,
        "split": split,
        "label": label,
        "tier": "correct" if label == "grounded" else "wrong",
        "failure_family": failure_family,
        "claim": claim,
        "trace_events": trace_events,
        "evidence_spans": evidence_spans,
        "expected": expected,
        "observed": observed,
    }


def generate_rows(seed: int = 23, per_family: int = 90) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for family in FAILURE_FAMILIES:
        for index in range(per_family):
            rows.append(_make_pair_member(rng, family, index, per_family, grounded=True))
            rows.append(_make_pair_member(rng, family, index, per_family, grounded=False))
    return rows


def _make_pair_member(
    rng: random.Random,
    family: str,
    index: int,
    per_family: int,
    *,
    grounded: bool,
) -> dict[str, Any]:
    module = rng.choice(["auth", "billing", "router", "worker", "search", "cache"])
    ext = rng.choice(["py", "ts", "go"])
    file_path = f"src/{module}.{ext}"
    wrong_file = f"src/{module}_settings.{ext}"
    symbol = rng.choice(["validate_token", "sync_invoice", "route_request", "flush_cache", "parse_event"])
    wrong_symbol = f"{symbol}_v9"
    api = rng.choice(["RetryPolicy", "AsyncClient", "ProjectStore", "EventParser"])
    wrong_api = f"{api}Beta"
    test_cmd = rng.choice(["pytest tests/test_auth.py", "npm test -- router", "go test ./worker"])

    expected = {
        "files": [file_path],
        "symbols": [symbol, api],
        "test_command": test_cmd,
        "test_passed": True,
        "patch_applied": True,
        "order": ["edit", "test", "final"],
    }
    observed = {
        "files": [file_path],
        "symbols": [symbol, api],
        "test_command": test_cmd,
        "test_passed": True,
        "patch_applied": True,
        "order": ["edit", "test", "final"],
    }

    claim_file = file_path
    claim_symbol = symbol
    claim_api = api
    claim_test_passed = True
    claim_patch_applied = True
    claim_order = ["edit", "test", "final"]

    if not grounded:
        if family == "wrong_file":
            claim_file = wrong_file
        elif family == "symbol_api_drift":
            claim_symbol = wrong_symbol
            claim_api = wrong_api
        elif family == "test_outcome_mismatch":
            observed["test_passed"] = False
            claim_test_passed = True
        elif family == "patch_result_contradiction":
            observed["patch_applied"] = False
            claim_patch_applied = True
        elif family == "temporal_order_error":
            observed["order"] = ["test", "edit", "final"]
            claim_order = ["edit", "test", "final"]
        elif family == "partial_support":
            claim_symbol = wrong_symbol
        elif family == "near_miss_overlap":
            claim_api = wrong_api
            claim_file = file_path

    status_text = "passed" if observed["test_passed"] else "failed"
    patch_text = "applied" if observed["patch_applied"] else "reverted"
    trace_events = [
        _event("edit", f"Edited {file_path}; added {symbol} using {api}.", file_path=file_path, symbols=[symbol, api]),
        _event("test", f"Ran {test_cmd}; tests {status_text}.", command=test_cmd, passed=observed["test_passed"]),
        _event("patch", f"Patch was {patch_text}.", applied=observed["patch_applied"]),
        _event("final", "Agent produced final response.", order=observed["order"]),
    ]
    if observed["order"][0] == "test":
        trace_events = [trace_events[1], trace_events[0], trace_events[2], trace_events[3]]

    claim = (
        f"The agent edited {claim_file}, added {claim_symbol} with {claim_api}, "
        f"then ran `{test_cmd}` and the tests {'passed' if claim_test_passed else 'failed'}; "
        f"the patch was {'applied' if claim_patch_applied else 'reverted'}."
    )
    evidence_spans = [
        {"claim_atom": "file", "event_index": 0 if observed["order"][0] == "edit" else 1},
        {"claim_atom": "symbol", "event_index": 0 if observed["order"][0] == "edit" else 1},
        {"claim_atom": "test", "event_index": 1 if observed["order"][0] == "edit" else 0},
        {"claim_atom": "patch", "event_index": 2},
    ]
    label = "grounded" if grounded else "ungrounded"
    base = f"native_{family}_{index:04d}"
    return _row(
        row_id=f"{base}__{'correct' if grounded else 'wrong'}",
        base_scenario_id=base,
        split=_split_for_index(index, per_family),
        label=label,
        failure_family=family,
        claim=claim,
        trace_events=trace_events,
        evidence_spans=evidence_spans,
        expected={
            **expected,
            "claim_files": [claim_file],
            "claim_symbols": [claim_symbol, claim_api],
            "claim_test_passed": claim_test_passed,
            "claim_patch_applied": claim_patch_applied,
            "claim_order": claim_order,
        },
        observed=observed,
    )


def write_banks(rows: list[dict[str, Any]], out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for split in ("train", "val", "test"):
        split_rows = [row for row in rows if row["split"] == split]
        path = out_dir / f"trajectory_native_{split}.json"
        payload = {
            "schema": "trace_trajectory_native_bank.v1",
            "split": split,
            "rows": split_rows,
            "metadata": {
                "rows": len(split_rows),
                "failure_families": list(FAILURE_FAMILIES),
                "leakage_guard": "base_scenario_id is disjoint across train/val/test",
            },
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        paths[split] = path
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--per-family", type=int, default=90)
    args = parser.parse_args()
    paths = write_banks(generate_rows(seed=args.seed, per_family=args.per_family), Path(args.out_dir))
    for split, path in paths.items():
        print(f"{split}: {path}")


if __name__ == "__main__":
    main()
