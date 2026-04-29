"""Unit tests for the teacher-labeler raw-benchmark loaders.

We write tiny synthetic fixtures and assert the loader normalizes to
the student's common row schema correctly, with an emphasis on the
fields that matter downstream (``pair_id``, ``role``, ``gold_band``).
"""

from __future__ import annotations

import json
import pathlib

import pytest

from research.triangular_maxsim.student_v2.teacher_label.loaders import (
    iter_ragtruth_train,
    iter_veracier_curated,
)


def _write(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


@pytest.fixture
def ragtruth_fixture(tmp_path: pathlib.Path) -> pathlib.Path:
    base = tmp_path / "RAGTruth"
    _write(
        base / "source_info.jsonl",
        [
            {
                "source_id": "100",
                "task_type": "QA",
                "source_info": "The Eiffel Tower is in Paris.",
                "prompt": "Where is the Eiffel Tower?",
            },
            {
                "source_id": "200",
                "task_type": "Summary",
                "source_info": "Apples are red.",
                "prompt": "Summarize.",
            },
        ],
    )
    _write(
        base / "response.jsonl",
        [
            {
                "id": "r1",
                "source_id": "100",
                "split": "train",
                "model": "gpt-3.5",
                "response": "The Eiffel Tower is in Paris.",
                "labels": [],
            },
            {
                "id": "r2",
                "source_id": "100",
                "split": "train",
                "model": "gpt-4",
                "response": "The Eiffel Tower is in Berlin.",
                "labels": [
                    {
                        "start": 0,
                        "end": 30,
                        "text": "Eiffel in Berlin",
                        "meta": "…",
                        "label_type": "Evident Conflict",
                    }
                ],
            },
            {
                "id": "r3",
                "source_id": "100",
                "split": "test",
                "model": "gpt-4",
                "response": "...",
                "labels": [],
            },
            {
                "id": "r4",
                "source_id": "200",
                "split": "train",
                "model": "mistral",
                "response": "Apples are red.",
                "labels": [
                    {"label_type": "Subtle Conflict", "start": 0, "end": 3, "text": "X"}
                ],
            },
        ],
    )
    return base


def test_ragtruth_returns_only_train_split(ragtruth_fixture: pathlib.Path) -> None:
    rows = list(iter_ragtruth_train(root=ragtruth_fixture))
    ids = {r["meta"]["ragtruth_id"] for r in rows}
    assert ids == {"r1", "r2", "r4"}
    for r in rows:
        assert r["split_hint"] == "train"


def test_ragtruth_normalizes_band_and_role(ragtruth_fixture: pathlib.Path) -> None:
    rows = {r["meta"]["ragtruth_id"]: r for r in iter_ragtruth_train(root=ragtruth_fixture)}
    assert rows["r1"]["gold_band"] == "green"
    assert rows["r1"]["role"] == "grounded"
    assert rows["r2"]["gold_band"] == "red"
    assert rows["r2"]["role"] == "hallucinated"
    assert rows["r4"]["gold_band"] == "amber"
    assert rows["r4"]["role"] == "partial_support"


def test_ragtruth_shared_pair_id_for_same_source(ragtruth_fixture: pathlib.Path) -> None:
    rows = list(iter_ragtruth_train(root=ragtruth_fixture))
    pair_ids_by_source = {}
    for r in rows:
        pair_ids_by_source.setdefault(r["meta"]["source_id"], set()).add(r["pair_id"])
    for src, pids in pair_ids_by_source.items():
        assert len(pids) == 1, f"source {src} emitted >1 pair_id: {pids}"


def test_ragtruth_class_key_tracks_task_type(ragtruth_fixture: pathlib.Path) -> None:
    by_id = {r["meta"]["ragtruth_id"]: r for r in iter_ragtruth_train(root=ragtruth_fixture)}
    assert by_id["r1"]["class_key"] == "ragtruth::qa"
    assert by_id["r4"]["class_key"] == "ragtruth::summary"


def test_ragtruth_task_filter(ragtruth_fixture: pathlib.Path) -> None:
    only_qa = list(iter_ragtruth_train(root=ragtruth_fixture, tasks=("QA",)))
    assert {r["meta"]["task_type"] for r in only_qa} == {"QA"}


def test_ragtruth_missing_fixture_raises(tmp_path: pathlib.Path) -> None:
    with pytest.raises(FileNotFoundError):
        list(iter_ragtruth_train(root=tmp_path / "nope"))


@pytest.fixture
def veracier_fixture(tmp_path: pathlib.Path) -> pathlib.Path:
    p = tmp_path / "variants.curated.jsonl"
    _write(
        p,
        [
            {
                "question_id": "Q1",
                "example_id": "Q1:perfect",
                "mutation_type": "perfect",
                "expected_band": "green",
                "raw_context": "Context A.",
                "response_text": "Faithful A.",
                "query_text": "What is A?",
                "verticals": ["finance"],
            },
            {
                "question_id": "Q1",
                "example_id": "Q1:ambiguous",
                "mutation_type": "ambiguous",
                "expected_band": "amber",
                "raw_context": "Context A.",
                "response_text": "Kind of A.",
                "query_text": "What is A?",
                "verticals": ["finance"],
            },
            {
                "question_id": "Q1",
                "example_id": "Q1:contradiction",
                "mutation_type": "contradiction",
                "expected_band": "red",
                "raw_context": "Context A.",
                "response_text": "Not A.",
                "query_text": "What is A?",
                "verticals": ["finance"],
            },
        ],
    )
    return p


def test_veracier_yields_one_pair_id_per_question(veracier_fixture: pathlib.Path) -> None:
    rows = list(iter_veracier_curated(path=veracier_fixture))
    assert {r["pair_id"] for r in rows} == {"veracier::Q1"}


def test_veracier_normalizes_mutations_to_roles(veracier_fixture: pathlib.Path) -> None:
    by_id = {r["meta"]["example_id"]: r for r in iter_veracier_curated(path=veracier_fixture)}
    assert by_id["Q1:perfect"]["role"] == "grounded"
    assert by_id["Q1:perfect"]["gold_band"] == "green"
    assert by_id["Q1:ambiguous"]["role"] == "partial_support"
    assert by_id["Q1:ambiguous"]["gold_band"] == "amber"
    assert by_id["Q1:contradiction"]["role"] == "hallucinated"
    assert by_id["Q1:contradiction"]["gold_band"] == "red"


def test_veracier_skips_empty_rows(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "v.jsonl"
    _write(
        p,
        [
            {"question_id": "Q", "example_id": "x", "raw_context": "", "response_text": "y"},
            {"question_id": "Q", "example_id": "y", "raw_context": "y", "response_text": ""},
            {
                "question_id": "Q",
                "example_id": "z",
                "raw_context": "ctx",
                "response_text": "resp",
                "expected_band": "green",
                "mutation_type": "perfect",
            },
        ],
    )
    rows = list(iter_veracier_curated(path=p))
    assert len(rows) == 1
    assert rows[0]["meta"]["example_id"] == "z"
