from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType


def _load_script(name: str) -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bench = _load_script("bench_granite_parity")
calib = _load_script("calibrate_granite_parity")


def _record(dataset: str, idx: int, supported: bool, score: float) -> dict:
    return {
        "suite": "lm-aggrefact",
        "dataset": dataset,
        "row_id": f"{dataset}-{idx}",
        "supported": supported,
        "score": score,
        "predicted_supported": score >= 0.5,
        "latency_ms": 1.0,
        "error": None,
        "metadata": {},
        "audit": {
            "runtime_decision": {
                "score": score,
                "head_score": score,
                "class_key": "rag.prose.multi_claim",
            },
            "corpus_route": {"corpus_type": "rag.prose.multi_claim"},
            "score_channels": {"groundedness_v2": score, "primary": score},
            "scores": {"groundedness_v2": score},
            "warnings": [],
        },
    }


def test_filter_records_to_targets_deduplicates_latest() -> None:
    target = [
        bench.BenchRow("lm-aggrefact", "A", "A-1", "", "", "", True),
        bench.BenchRow("lm-aggrefact", "A", "A-2", "", "", "", False),
    ]
    first = bench._record_to_scored_row(_record("A", 1, True, 0.1))
    latest = bench._record_to_scored_row(_record("A", 1, True, 0.9))
    second = bench._record_to_scored_row(_record("A", 2, False, 0.2))

    filtered = bench._filter_records_to_targets([first, second, latest], target)

    assert [item.row.row_id for item in filtered] == ["A-1", "A-2"]
    assert filtered[0].score == 0.9


def test_calibrator_selects_production_safe_threshold_rule(tmp_path: Path) -> None:
    rows = []
    for idx in range(40):
        rows.append(_record("A", idx, True, 0.85 + (idx % 5) * 0.01))
        rows.append(_record("A", idx + 100, False, 0.10 + (idx % 5) * 0.01))
        rows.append(_record("B", idx, True, 0.80 + (idx % 5) * 0.01))
        rows.append(_record("B", idx + 100, False, 0.20 + (idx % 5) * 0.01))
    path = tmp_path / "rows.jsonl"
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )

    loaded = calib._load_rows(path)
    artifact = calib.fit_calibrator(loaded, holdout_fraction=0.25, seed=7)

    assert artifact["selected_validation"]["macro_balanced_accuracy"] == 1.0
    assert artifact["selected_candidate"]["production_safe"] is True
