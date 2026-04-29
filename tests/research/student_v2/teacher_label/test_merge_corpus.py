"""Integration test for ``merge_corpus``: fuse synthetic + teacher rows."""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys


def _write(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def _base_row(*, pair_id: str, pair_row_id: str, **extra) -> dict:
    return {
        "pair_id": pair_id,
        "pair_row_id": pair_row_id,
        "class_key": "x",
        "split_hint": "train",
        "role": "grounded",
        "response_text": "resp",
        "evidence_text": "ev",
        "gold_band": "green",
        "source": "synthetic",
        "turn_score": 0.9,
        "turn_band": 0,
        "token_support_labels": [1, 1],
        "dead_weight_unit_labels": [0],
        "coverage_unit_labels": [1],
        **extra,
    }


def test_merge_corpus_dedups_and_splits(tmp_path: pathlib.Path) -> None:
    synth = tmp_path / "syn_train.jsonl"
    teacher_a = tmp_path / "teacher_a.jsonl"
    teacher_b = tmp_path / "teacher_b.jsonl"
    out = tmp_path / "out"

    # 10 synthetic rows, 10 teacher_a rows, 10 teacher_b rows — one
    # deliberate duplicate in teacher_a to exercise dedup.
    _write(synth, [_base_row(pair_id=f"syn-{i}", pair_row_id=f"syn-{i}") for i in range(10)])
    _write(
        teacher_a,
        [_base_row(pair_id=f"ta-{i}", pair_row_id=f"ta-{i}", source="ragtruth") for i in range(10)]
        + [_base_row(pair_id="syn-0", pair_row_id="syn-0", source="ragtruth")],
    )
    _write(
        teacher_b,
        [_base_row(pair_id=f"tb-{i}", pair_row_id=f"tb-{i}", source="veracier") for i in range(10)],
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "research.triangular_maxsim.student_v2.merge_corpus",
            "--synthetic-train",
            str(synth),
            "--teacher-jsonls",
            str(teacher_a),
            str(teacher_b),
            "--out-dir",
            str(out),
            "--seed",
            "42",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    train = [json.loads(l) for l in (out / "train.jsonl").read_text().splitlines()]
    val = [json.loads(l) for l in (out / "val.jsonl").read_text().splitlines()]
    test = [json.loads(l) for l in (out / "test.jsonl").read_text().splitlines()]
    ood = [json.loads(l) for l in (out / "ood_eval.jsonl").read_text().splitlines()]
    manifest = json.loads((out / "manifest.json").read_text())

    total = len(train) + len(val) + len(test) + len(ood)
    assert total == 30, (
        f"expected 30 rows after dedup (10 syn + 10 ta + 10 tb - 1 dup), got {total}"
    )
    assert manifest["split_sizes"]["train"] + manifest["split_sizes"]["val"] + manifest[
        "split_sizes"
    ]["test"] == 30

    # No row id should appear in more than one split.
    all_ids: list[str] = []
    for split in (train, val, test):
        all_ids.extend(r["pair_row_id"] for r in split)
    assert len(all_ids) == len(set(all_ids)), "splits leak pair_row_id across boundaries"

    # Pair-aware: rows sharing a pair_id must be co-located. Here every
    # pair has a single row so the invariant is trivially satisfied,
    # but we still verify structure is consistent.
    assert manifest["sources"]  # non-empty source breakdown
    assert manifest["bands"]


def test_merge_corpus_emits_manifest_inputs(tmp_path: pathlib.Path) -> None:
    synth = tmp_path / "syn.jsonl"
    teacher = tmp_path / "t.jsonl"
    out = tmp_path / "out"
    _write(synth, [_base_row(pair_id="a", pair_row_id="a")])
    _write(teacher, [_base_row(pair_id="b", pair_row_id="b", source="ragtruth")])

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "research.triangular_maxsim.student_v2.merge_corpus",
            "--synthetic-train",
            str(synth),
            "--teacher-jsonls",
            str(teacher),
            "--out-dir",
            str(out),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["inputs"]["synthetic_train"].endswith("syn.jsonl")
    assert len(manifest["inputs"]["teacher_jsonls"]) == 1
