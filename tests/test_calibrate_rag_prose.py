"""Unit tests for the ``rag_prose`` threshold calibrator."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load_calibrator_module() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / "calibrate_rag_prose.py"
    spec = importlib.util.spec_from_file_location("calibrate_rag_prose", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


calib = _load_calibrator_module()


def _example(
    idx: int,
    label: str,
    archetype: str = "finance_tax",
) -> "calib.CalibrationExample":  # type: ignore[name-defined]
    return calib.CalibrationExample(
        id=f"E-{label}-{idx:03d}",
        archetype=archetype,
        language="fr",
        label=label,
        query_text="q",
        response_text=f"response {idx}",
        raw_context="context",
    )


# ---------------------------------------------------------------------------
# Seed loading
# ---------------------------------------------------------------------------


def test_load_calibration_set_parses_checked_in_seed() -> None:
    seed = (
        Path(__file__).resolve().parents[1]
        / "latence_trace"
        / "data"
        / "calibration"
        / "rag_prose.jsonl"
    )
    assert seed.exists(), "checked-in seed file must exist"
    rows = calib.load_calibration_set(seed)
    assert len(rows) >= 20
    labels = {row.label for row in rows}
    assert labels == {"green", "amber", "red"}
    archetypes = {row.archetype for row in rows}
    assert "finance_tax" in archetypes
    assert "legal_contracts" in archetypes


def test_load_calibration_set_rejects_invalid_label(tmp_path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "X-1",
                "archetype": "finance_tax",
                "label": "yellow",
                "query_text": "q",
                "response_text": "r",
                "raw_context": "c",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        calib.load_calibration_set(path)


# ---------------------------------------------------------------------------
# Threshold fitter (pure logic)
# ---------------------------------------------------------------------------


def test_fit_rag_prose_thresholds_separates_green_from_red() -> None:
    examples = []
    scores = {}
    for idx in range(40):
        green = _example(idx, "green")
        examples.append(green)
        scores[green.id] = 0.90 + (idx % 5) * 0.005
    for idx in range(40):
        red = _example(idx, "red")
        examples.append(red)
        scores[red.id] = 0.25 + (idx % 5) * 0.005
    for idx in range(20):
        amber = _example(idx, "amber")
        examples.append(amber)
        scores[amber.id] = 0.60 + (idx % 5) * 0.005

    fit = calib.fit_rag_prose_thresholds(
        examples,
        scores,
        profile="standard",
        green_precision_target=0.97,
        red_precision_target=0.95,
        holdout_fraction=0.25,
        seed=7,
    )
    assert fit.profile == "standard"
    assert 0.55 <= fit.amber_min <= fit.green_min <= 1.0
    assert fit.precision_at_green >= 0.97
    assert fit.red_precision_at_amber >= 0.95
    assert fit.class_counts == {"green": 40, "red": 40, "amber": 20}
    assert fit.sample_count == 100


def test_fit_rag_prose_thresholds_squashes_inverted_thresholds() -> None:
    # A pathological case where all classes overlap and the fitter could
    # otherwise return amber_min >= green_min.
    examples = []
    scores = {}
    for idx in range(6):
        green = _example(idx, "green")
        examples.append(green)
        scores[green.id] = 0.70
    for idx in range(6):
        red = _example(idx, "red")
        examples.append(red)
        scores[red.id] = 0.70
    for idx in range(4):
        amber = _example(idx, "amber")
        examples.append(amber)
        scores[amber.id] = 0.70

    fit = calib.fit_rag_prose_thresholds(
        examples,
        scores,
        profile="standard",
        green_precision_target=0.97,
        red_precision_target=0.95,
        holdout_fraction=0.5,
        seed=0,
    )
    assert fit.amber_min < fit.green_min


# ---------------------------------------------------------------------------
# Threshold file writer
# ---------------------------------------------------------------------------


def test_update_thresholds_file_rewrites_rag_prose(tmp_path) -> None:
    sample = {
        "schema_version": 1,
        "headline": "groundedness_v2",
        "nli_enabled": True,
        "precision_target": 0.75,
        "strata": {
            "default": {"amber_min": 0.6, "green_min": 0.8},
            "rag_prose": {"amber_min": 0.55, "green_min": 0.75},
        },
    }
    path = tmp_path / "thresholds.json"
    path.write_text(json.dumps(sample), encoding="utf-8")

    fit = calib.FitResult(
        profile="quality",
        green_min=0.78,
        amber_min=0.52,
        precision_at_green=0.98,
        recall_at_green=0.9,
        red_precision_at_amber=0.96,
        red_recall_at_amber=0.85,
        sample_count=120,
        holdout_sample_count=36,
        class_counts={"green": 60, "amber": 24, "red": 36},
        positive_median=0.9,
        negative_median=0.4,
    )
    result = calib.update_thresholds_file(path, fit)
    assert result["entry"]["green_min"] == pytest.approx(0.78)

    reloaded = json.loads(path.read_text(encoding="utf-8"))
    rag_prose = reloaded["strata"]["rag_prose"]
    for key in (
        "amber_min",
        "green_min",
        "precision_at_green",
        "recall_at_green",
        "red_precision_at_amber",
        "red_recall_at_amber",
        "sample_count",
        "holdout_sample_count",
        "positive_median",
        "negative_median",
    ):
        assert key in rag_prose, f"missing metadata key {key}"
    assert reloaded["rag_prose_fit"]["profile"] == "quality"
    assert reloaded["rag_prose_fit"]["class_counts"] == fit.class_counts


# ---------------------------------------------------------------------------
# Build-seed subcommand
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_build_seed_from_bench_promotes_matching_rows(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    variants = [
        {
            "example_id": "FIN-01:perfect",
            "question_id": "FIN-01",
            "mutation_type": "perfect",
            "expected_band": "green",
            "generation_archetype": {"id": "finance_tax"},
            "query_text": "q",
            "response_text": "r",
            "raw_context": "c",
            "context_doc_ids": ["FIN-01:DOC-1"],
        },
        {
            "example_id": "FIN-01:wrong",
            "question_id": "FIN-01",
            "mutation_type": "wrong",
            "expected_band": "red",
            "generation_archetype": {"id": "finance_tax"},
            "query_text": "q",
            "response_text": "r",
            "raw_context": "c",
        },
        {
            "example_id": "FIN-01:ambiguous",
            "question_id": "FIN-01",
            "mutation_type": "ambiguous",
            "expected_band": "amber",
            "generation_archetype": {"id": "finance_tax"},
            "ambiguous_unstable": True,
            "query_text": "q",
            "response_text": "r",
            "raw_context": "c",
        },
    ]
    trace_results = [
        {
            "example_id": "FIN-01:perfect",
            "profile": "standard",
            "trace": {"band": "green"},
        },
        {
            "example_id": "FIN-01:wrong",
            "profile": "standard",
            "trace": {"band": "amber"},
        },
    ]
    _write_jsonl(run_dir / "variants.jsonl", variants)
    _write_jsonl(run_dir / "trace_results.jsonl", trace_results)

    out_path = tmp_path / "rag_prose.jsonl"
    summary = calib.build_seed_from_bench(
        benchmark_run=run_dir,
        out_path=out_path,
        label_from="expected-band",
        keep_unstable_amber=False,
    )
    assert summary["kept"] == 1  # only the perfect row survives
    reasons = {item["reason"] for item in summary["skipped_rows"]}
    assert any("ambiguous_unstable" in reason for reason in reasons)
    assert any("live band amber" in reason for reason in reasons)

    rows = calib.load_calibration_set(out_path)
    assert [row.label for row in rows] == ["green"]
