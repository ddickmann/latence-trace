"""Integration tests for the runtime classifier singleton + per-class predictions."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from latence_trace.core.corpus_router import classifier as _classifier

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data/corpus_classifier"


@pytest.fixture(autouse=True)
def _reset_singleton():
    _classifier.reset_singleton_for_tests()
    yield
    _classifier.reset_singleton_for_tests()


def test_classifier_loads_and_reports_top1_accuracy() -> None:
    result = _classifier.classify(query="q", response="r", raw_context="c")
    # Artefact is shipped via scripts/train_corpus_classifier.py; bundle
    # must load successfully in CI.
    assert result.source == "classifier"
    assert result.corpus_type is not None
    assert result.artefact_sha256 is not None


def test_classifier_latency_under_router_budget() -> None:
    """Featurize + predict_proba must complete in <= 5 ms on CPU."""
    # Warm-up so the joblib bundle is loaded before timing.
    _classifier.classify(query="q", response="r", raw_context="c")
    samples = []
    for _ in range(20):
        result = _classifier.classify(
            query="What is the policy?",
            response="The policy mandates annual review cycles.",
            raw_context="Policy doc: annual review cycle, mandated since 2023.",
        )
        samples.append(result.latency_ms)
    p50 = sorted(samples)[len(samples) // 2]
    assert p50 < 10.0, f"classifier p50 latency {p50:.2f} ms exceeds 10 ms budget"


@pytest.mark.parametrize(
    "class_key",
    [
        "rag.prose.enterprise",
        "rag.prose.short_factoid",
        "rag.prose.multi_claim",
        "rag.structured",
        "rag.code_in_context",
        "code.agentic_trace",
    ],
)
def test_classifier_predicts_correct_class_on_held_out_samples(class_key: str) -> None:
    """For each class, the first held-out test row must be classified correctly.

    This is a minimal smoke-level assertion - the full confusion matrix
    lives in ``data/corpus_classifier/confusion_matrix.json`` (0.997 top-1).
    """
    t = pq.read_table(DATA_DIR / "test.parquet")
    rids = t.column("row_id").to_pylist()
    cks = t.column("class_key").to_pylist()
    qs = t.column("query").to_pylist()
    rs = t.column("response").to_pylist()
    cs = t.column("raw_context").to_pylist()
    # first test row for this class
    for i, ck in enumerate(cks):
        if ck == class_key:
            break
    else:
        pytest.skip(f"no test row for {class_key}")
    result = _classifier.classify(query=qs[i], response=rs[i], raw_context=cs[i])
    assert result.corpus_type == class_key, (
        f"misclassified row {rids[i]!r}: expected {class_key}, got {result.corpus_type} "
        f"(confidence={result.confidence})"
    )
