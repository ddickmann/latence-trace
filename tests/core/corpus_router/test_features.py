"""Unit tests for the corpus-router feature extractor."""

from __future__ import annotations

import time

import pytest

from latence_trace.core.corpus_router.features import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    featurize,
    featurize_batch,
)


def test_feature_names_are_locked_and_unique() -> None:
    # Schema version is bumped by any feature set change.
    assert FEATURE_SCHEMA_VERSION == 2
    assert len(FEATURE_NAMES) == 27
    assert len(set(FEATURE_NAMES)) == 27


def test_featurize_returns_locked_length_vector() -> None:
    vec = featurize(query="q", response="r", raw_context="c")
    assert tuple(vec.names) == FEATURE_NAMES
    assert len(vec.values) == len(FEATURE_NAMES)
    assert all(isinstance(v, float) for v in vec.values)


def test_featurize_is_deterministic_across_calls() -> None:
    v1 = featurize(query="What is X?", response="X is Y.", raw_context="X is Y. Y is Z.")
    v2 = featurize(query="What is X?", response="X is Y.", raw_context="X is Y. Y is Z.")
    assert tuple(v1.values) == tuple(v2.values)


def test_fenced_code_detected_and_counted() -> None:
    response = """Here is the fix:

```python
def foo(x):
    return x + 1


def bar(y):
    return y - 1
```

That should do it."""
    vec = featurize(query="fix foo", response=response, raw_context="").as_dict()
    assert vec["response_fenced_code_blocks"] >= 1
    assert vec["response_fenced_max_nonblank_lines"] >= 4


def test_prose_without_fence_has_zero_code_features() -> None:
    vec = featurize(
        query="explain",
        response="The policy mandates an annual review cycle.",
        raw_context="Policy document: annual review.",
    ).as_dict()
    assert vec["response_fenced_code_blocks"] == 0.0
    assert vec["response_fenced_max_nonblank_lines"] == 0.0
    assert vec["response_has_diff_headers"] == 0.0


def test_structured_context_triggers_json_hint_density() -> None:
    ctx = '{"name": "Apna", "city": "Santa Barbara", "state": "CA"}'
    vec = featurize(query="summary", response="Apna is in SB.", raw_context=ctx).as_dict()
    assert vec["context_json_hint_density"] > 0.0


def test_file_headers_and_multiple_chunks() -> None:
    ctx = "=== src/foo.py ===\ndef foo():...\n=== src/bar.py ===\ndef bar():..."
    vec = featurize(query="q", response="r", raw_context=ctx).as_dict()
    assert vec["context_file_header_count"] == 2
    assert vec["context_has_multiple_chunks"] == 1.0


def test_featurize_latency_under_budget_for_typical_payloads() -> None:
    """Typical production request: ~2 KB response + ~20 KB context."""
    response = "The policy requires review. " * 50
    context = "Chapter 1: overview\n" + ("X" * 1000 + "\n") * 20
    start = time.perf_counter()
    for _ in range(20):
        featurize(query="q", response=response, raw_context=context)
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / 20
    assert elapsed_ms < 3.0, (
        f"featurize p50 was {elapsed_ms:.3f} ms on a typical payload, budget is 3 ms"
    )


def test_featurize_latency_budget_for_extreme_payloads() -> None:
    """Extreme payload stress test - accept up to 25 ms per call on 100 KB."""
    response = "The policy requires review. " * 2000
    context = "X" * 100_000
    start = time.perf_counter()
    for _ in range(5):
        featurize(query="q", response=response, raw_context=context)
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / 5
    assert elapsed_ms < 25.0, (
        f"featurize p50 was {elapsed_ms:.3f} ms on extreme payload, budget is 25 ms"
    )


def test_featurize_batch_preserves_input_order() -> None:
    rows = [
        {"query": "q1", "response": "r1", "raw_context": "c1"},
        {"query": "q2", "response": "r2 with more words", "raw_context": "c2"},
    ]
    matrix, names = featurize_batch(rows)
    assert names == FEATURE_NAMES
    assert len(matrix) == 2
    assert matrix[0][FEATURE_NAMES.index("response_len_tokens_log")] < matrix[1][
        FEATURE_NAMES.index("response_len_tokens_log")
    ]


def test_unicode_non_ascii_response_lowers_ascii_ratio() -> None:
    german = featurize(query="q", response="Das ist ein Überblick", raw_context="c").as_dict()
    english = featurize(query="q", response="This is an overview", raw_context="c").as_dict()
    assert german["response_ascii_ratio"] < english["response_ascii_ratio"]
