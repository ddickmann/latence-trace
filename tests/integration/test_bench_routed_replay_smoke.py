"""Smoke test for ``scripts/bench_routed_replay.py``.

Runs the per-class deck with a 2-row-per-class cap against the live
``dev_app``. Asserts the proof bundle writes all expected files and
that every row came back with a band and a routed bundle.
"""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

import pytest


DEV_URL = os.environ.get("LATENCE_TRACE_DEV_URL", "http://127.0.0.1:8091")


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


def test_bench_routed_replay_smoke(tmp_path: Path) -> None:
    from scripts import bench_routed_replay as m

    exit_code = m.main(
        [
            "--url",
            f"{DEV_URL}/runsync",
            "--per-class-cap",
            "2",
            "--latency-sample",
            "1",
            "--concurrency",
            "2",
            "--skip-veracier",
            "--out-dir",
            str(tmp_path),
        ]
    )
    assert exit_code == 0
    assert (tmp_path / "PROOF_SUMMARY.json").exists()
    assert (tmp_path / "PROOF_REPORT.md").exists()
    assert (tmp_path / "per_class.jsonl").exists()
    assert (tmp_path / "classifier_confusion.json").exists()
    assert (tmp_path / "router_latency_histogram.json").exists()

    rows = [
        json.loads(line)
        for line in (tmp_path / "per_class.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(rows) >= 12, f"expected 2 rows x 6 classes = 12, got {len(rows)}"
    # Every row must either carry a band OR document an error so we
    # never silently drop predictions.
    for r in rows:
        if r.get("error"):
            continue
        assert r.get("band") in {"green", "amber", "red"}, r
        route = r.get("corpus_route")
        assert route is not None and route.get("corpus_type"), r
