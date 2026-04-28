"""Unit tests for the customer-defensible report helpers in the Veracier bench."""

from __future__ import annotations

import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path
from types import ModuleType


def _load_bench_module() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / "bench_veracier_rag_validation.py"
    spec = importlib.util.spec_from_file_location("bench_veracier_rag_validation", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bench = _load_bench_module()


def _trace_row(
    *,
    example_id: str,
    expected: str,
    observed: str,
    archetype_id: str = "finance_tax",
    archetype_label: str = "Finance / tax",
    profile: str = "standard",
    unstable: bool = False,
    score: float = 0.8,
    wall_ms: float = 1200.0,
) -> dict:
    return {
        "example_id": example_id,
        "question_id": example_id.split(":")[0],
        "generation_archetype": {"id": archetype_id, "label": archetype_label},
        "mutation_type": "perfect" if expected == "green" else (
            "ambiguous" if expected == "amber" else "wrong"
        ),
        "expected_band": expected,
        "expected_groundedness_range": {"min": 0.0, "max": 1.0},
        "ambiguous_unstable": unstable,
        "profile": profile,
        "wall_ms": wall_ms,
        "trace": {
            "band": observed,
            "score": score,
            "nli_aggregate": score,
            "context_coverage_ratio": 0.75,
            "context_usage_ratio": 0.55,
            "epistemic_hedge_gate": None,
        },
    }


def test_compute_headline_metrics_matches_precision_definitions() -> None:
    traces = [
        # Green: 8 TP, 1 FP (expected amber observed green)
        *[
            _trace_row(example_id=f"FIN:perfect:{i}", expected="green", observed="green")
            for i in range(8)
        ],
        _trace_row(example_id="FIN:amber:1", expected="amber", observed="green"),
        # Red: 5 TP, 0 FP
        *[
            _trace_row(example_id=f"FIN:wrong:{i}", expected="red", observed="red")
            for i in range(5)
        ],
        # Amber: 4 expected amber (1 already counted above as misclassified green,
        # so 3 more), 3 observed amber of these
        _trace_row(example_id="FIN:amber:2", expected="amber", observed="amber"),
        _trace_row(example_id="FIN:amber:3", expected="amber", observed="amber"),
        _trace_row(example_id="FIN:amber:4", expected="amber", observed="red"),
        # One unstable amber row that should be excluded from amber denominator
        _trace_row(
            example_id="FIN:amber:unstable",
            expected="amber",
            observed="red",
            unstable=True,
        ),
    ]
    metrics = bench._compute_headline_metrics(traces)
    assert metrics["green"]["true_positive"] == 8
    assert metrics["green"]["false_positive"] == 1
    assert metrics["green"]["precision"] == 8 / 9
    # Non-unstable amber row observed as red is a false positive for red.
    assert metrics["red"]["true_positive"] == 5
    assert metrics["red"]["false_positive"] == 1
    assert metrics["red"]["denominator"] == 6
    # Amber: expected 4 (minus the 1 unstable) = 4, observed amber = 2 out of 4
    assert metrics["amber"]["expected"] == 4
    assert metrics["amber"]["match"] == 2
    assert metrics["amber"]["excluded_unstable"] == 1


def test_compute_archetype_metrics_groups_by_archetype() -> None:
    traces = [
        _trace_row(
            example_id="FIN:perfect:1",
            expected="green",
            observed="green",
            archetype_id="finance_tax",
            archetype_label="Finance / tax",
        ),
        _trace_row(
            example_id="FIN:wrong:1",
            expected="red",
            observed="red",
            archetype_id="finance_tax",
            archetype_label="Finance / tax",
        ),
        _trace_row(
            example_id="LEG:perfect:1",
            expected="green",
            observed="amber",
            archetype_id="legal_contracts",
            archetype_label="Legal contracts",
        ),
    ]
    entries = bench._compute_archetype_metrics(traces)
    by_id = {entry["id"]: entry for entry in entries}
    assert set(by_id.keys()) == {"finance_tax", "legal_contracts"}
    assert by_id["finance_tax"]["headline_accuracy"] == 1.0
    assert by_id["legal_contracts"]["headline_accuracy"] == 0.0
    assert by_id["legal_contracts"]["label"] == "Legal contracts"


def test_compute_latency_summary_reports_percentiles() -> None:
    traces = [
        _trace_row(
            example_id=f"FIN:perfect:{i}",
            expected="green",
            observed="green",
            profile="standard",
            wall_ms=wall,
        )
        for i, wall in enumerate((1000, 1500, 2000, 2500, 3000), start=1)
    ]
    summary = bench._compute_latency_summary(traces)
    assert "standard" in summary
    assert summary["standard"]["count"] == 5
    assert summary["standard"]["p50_ms"] == 2000
    assert summary["standard"]["mean_ms"] == 2000


def test_write_report_emits_markdown_and_json(tmp_path) -> None:
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    (output_dir / "evidence_manifest.json").write_text(
        json.dumps(
            {
                "scope": "pilot",
                "use_cases": ["FIN-01", "LEG-01"],
                "cases": [
                    {
                        "use_case_id": "FIN-01",
                        "generation_archetype": {"id": "finance_tax"},
                        "document_count": 2,
                        "context_char_count": 12345,
                        "documents": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    variants = [
        {
            "example_id": "FIN-01:perfect",
            "question_id": "FIN-01",
            "generation_archetype": {"id": "finance_tax", "label": "Finance / tax"},
            "mutation_type": "perfect",
            "expected_band": "green",
            "query_text": "Quels sont les principaux elements du dossier ?",
            "response_text": "Reponse extractive avec deux citations verbatim.",
            "raw_context": "DOC-1 ...",
        },
        {
            "example_id": "FIN-01:wrong",
            "question_id": "FIN-01",
            "generation_archetype": {"id": "finance_tax", "label": "Finance / tax"},
            "mutation_type": "wrong",
            "expected_band": "red",
            "query_text": "Quels sont les principaux elements du dossier ?",
            "response_text": "Reponse non etayee avec valeurs inventees.",
            "raw_context": "DOC-1 ...",
        },
    ]
    with (output_dir / "variants.jsonl").open("w", encoding="utf-8") as fp:
        for row in variants:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")
    trace_rows = [
        _trace_row(
            example_id="FIN-01:perfect", expected="green", observed="green", profile="standard"
        ),
        _trace_row(
            example_id="FIN-01:perfect",
            expected="green",
            observed="green",
            profile="quality",
        ),
        _trace_row(
            example_id="FIN-01:wrong", expected="red", observed="red", profile="standard"
        ),
        _trace_row(
            example_id="FIN-01:wrong",
            expected="red",
            observed="red",
            profile="quality",
        ),
    ]
    with (output_dir / "trace_results.jsonl").open("w", encoding="utf-8") as fp:
        for row in trace_rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")

    args = Namespace(output_dir=output_dir, scope="pilot")
    report_path = bench.write_report(args)

    markdown = report_path.read_text(encoding="utf-8")
    for header in (
        "## 1. Scope and dataset",
        "## 2. Headline metrics",
        "## 3. Archetype breakdown",
        "## 4. Latency summary by profile",
        "## 5. Annotated examples",
        "## 6. Amber is a reviewer queue",
    ):
        assert header in markdown, f"report missing section: {header}"
    assert "Green precision" in markdown
    assert "Red precision" in markdown
    assert "Amber agreement" in markdown

    summary_path = output_dir / "validation_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["headline_metrics"]["green"]["precision"] == 1.0
    assert summary["headline_metrics"]["red"]["precision"] == 1.0
    assert summary["headline_targets"]["green_precision_target"] == 0.97
    assert summary["trace_counts"]["total"] == 4
    assert summary["variant_counts"]["by_mutation_type"] == {"perfect": 1, "wrong": 1}
