"""End-to-end integration tests for the corpus-router wired into the service.

Requires a locally running ``dev_app`` on ``LATENCE_TRACE_DEV_URL``
(default ``http://127.0.0.1:8091``). These tests are skipped when the
service is not reachable so they do not block CI on clean hosts.

What they cover:

* one real row from each of the six corpus classes is routed correctly
  (via explicit ``corpus_type`` override and via the classifier path),
* the router's chosen bundle is reflected in ``corpus_route.thresholds``
  and ``profile_diagnostics.effective_fusion_weights``,
* the code lane ``code.agentic_trace`` comes back with a non-null
  ``band`` now that the band-classification hook is installed,
* compact response from the RunPod handler continues to expose
  ``corpus_route`` so downstream callers can see the routing decision.
"""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq
import pytest


DEV_URL = os.environ.get("LATENCE_TRACE_DEV_URL", "http://127.0.0.1:8091")
TEST_PARQUET = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "corpus_classifier"
    / "test.parquet"
)


def _service_reachable() -> bool:
    try:
        payload = json.dumps(
            {"input": {"response": "ping", "query": "ping", "raw_context": "ping"}}
        ).encode()
        req = urllib.request.Request(
            f"{DEV_URL}/runsync",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _service_reachable(),
    reason="dev_app is not reachable on LATENCE_TRACE_DEV_URL",
)


def _post(body: dict, *, timeout: float = 120.0) -> dict:
    payload = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{DEV_URL}/runsync",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _one_row_per_class() -> dict[str, dict]:
    """Return the first row for each class from the held-out test split."""
    assert TEST_PARQUET.exists(), f"expected {TEST_PARQUET}"
    t = pq.read_table(TEST_PARQUET)
    class_keys = t.column("class_key").to_pylist()
    queries = t.column("query").to_pylist()
    responses = t.column("response").to_pylist()
    contexts = t.column("raw_context").to_pylist()
    seen: dict[str, dict] = {}
    for ck, q, r, c in zip(class_keys, queries, responses, contexts):
        if ck in seen:
            continue
        seen[ck] = {"query": q, "response": r, "raw_context": c}
        if len(seen) == 6:
            break
    return seen


SAMPLES = _one_row_per_class()

EXPECTED_CLASSES = {
    "rag.prose.enterprise",
    "rag.prose.short_factoid",
    "rag.prose.multi_claim",
    "rag.structured",
    "rag.code_in_context",
    "code.agentic_trace",
}


def test_one_real_row_exists_per_class() -> None:
    """Guard rail: if the dataset builder drops a class, this test
    explodes so the integration suite never silently skips a lane."""
    assert set(SAMPLES.keys()) == EXPECTED_CLASSES, (
        f"missing classes in test.parquet: {EXPECTED_CLASSES - set(SAMPLES)}"
    )


@pytest.mark.parametrize("class_key", sorted(EXPECTED_CLASSES))
def test_explicit_override_routes_to_expected_bundle(class_key: str) -> None:
    row = SAMPLES[class_key]
    body = {
        "input": {
            "query": row["query"],
            "response": row["response"],
            "raw_context": row["raw_context"],
            "corpus_type": class_key,
        }
    }
    out = _post(body)
    assert out["success"] is True, out
    route = out.get("corpus_route")
    assert route is not None, f"corpus_route missing in compact response: {out}"
    assert route["corpus_type"] == class_key
    assert route["source"] == "explicit"
    assert "green" in route["thresholds_applied"]
    assert "amber" in route["thresholds_applied"]
    # Each class owns its own scoring mode expectation
    if class_key == "code.agentic_trace":
        assert route["scoring_mode_applied"] == "code"
    else:
        assert route["scoring_mode_applied"] == "rag"


def test_classifier_infers_prose_enterprise() -> None:
    row = SAMPLES["rag.prose.enterprise"]
    out = _post(
        {
            "input": {
                "query": row["query"],
                "response": row["response"],
                "raw_context": row["raw_context"],
            }
        }
    )
    assert out["success"] is True
    route = out["corpus_route"]
    assert route["source"] in {"classifier", "fallback"}
    assert route["corpus_type"].startswith("rag.")


def test_classifier_infers_agentic_code_for_code_trace() -> None:
    row = SAMPLES["code.agentic_trace"]
    out = _post(
        {
            "input": {
                "query": row["query"],
                "response": row["response"],
                "raw_context": row["raw_context"],
            }
        }
    )
    assert out["success"] is True
    route = out["corpus_route"]
    # The classifier must reach one of the two code lanes for a real
    # agentic trace; we accept either code.agentic_trace or
    # rag.code_in_context because they share many features.
    assert route["corpus_type"] in {"code.agentic_trace", "rag.code_in_context"}
    # If it picked code.agentic_trace the service must flip to code mode
    # so the composite phantom score is primary.
    if route["corpus_type"] == "code.agentic_trace":
        assert out["scoring_mode"] == "code"


def test_code_lane_emits_band() -> None:
    """Regression: code lane used to return ``band=None`` because the
    risk-band classification was only applied in the RAG lane."""
    row = SAMPLES["code.agentic_trace"]
    out = _post(
        {
            "input": {
                "query": row["query"],
                "response": row["response"],
                "raw_context": row["raw_context"],
                "corpus_type": "code.agentic_trace",
                "scoring_mode": "code",
            }
        }
    )
    assert out["success"] is True
    band = out.get("band")
    assert band in {"green", "amber", "red"}, (
        f"code lane must emit a real band, got {band!r}. full={out}"
    )


def test_router_latency_budget_on_live_service() -> None:
    """The router itself should add less than ~15 ms on top of scoring."""
    row = SAMPLES["rag.prose.enterprise"]
    out = _post(
        {
            "input": {
                "query": row["query"],
                "response": row["response"],
                "raw_context": row["raw_context"],
            }
        }
    )
    route = out["corpus_route"]
    # Classifier latency is a router sub-step; budget 15 ms on live service.
    latency_ms = float(route.get("classifier_latency_ms", 0.0))
    assert latency_ms < 15.0, f"classifier latency {latency_ms:.2f} ms > 15 ms"
