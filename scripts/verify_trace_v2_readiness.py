"""Reproducible TRACE v2 readiness checks.

This harness is intentionally lightweight enough for CI. It verifies that the
runtime registry, head artifacts, policy, and focused tests agree before a
RunPod rebuild is promoted. Live RunPod checks remain opt-in via env vars.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "latence_trace/data/runtime_head_registry.root_cause_solution_v1.json"
POLICY = ROOT / "latence_trace/data/runtime_policy.optimized_v1_plus_calibrator.json"
REQUIRED_CLASSES = {
    "rag.prose.enterprise",
    "rag.prose.short_factoid",
    "rag.prose.multi_claim",
    "rag.structured",
    "rag.code_in_context",
    "code.agentic_trace",
}
FOCUSED_TESTS = [
    "tests/research/test_coding_trajectory_head.py",
    "tests/research/test_root_cause_solution_tracks.py",
    "tests/core/test_runtime_decision.py",
    "tests/test_runpod_handler.py",
]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_artifacts() -> list[str]:
    errors: list[str] = []
    registry = _read_json(REGISTRY).get("runtime_head_registry") or {}
    policy = _read_json(POLICY).get("classes") or {}
    missing = REQUIRED_CLASSES - set(registry)
    if missing:
        errors.append(f"registry missing classes: {sorted(missing)}")
    for class_key in sorted(REQUIRED_CLASSES):
        entry = registry.get(class_key) or {}
        if not entry.get("enabled"):
            errors.append(f"{class_key}: registry disabled")
        artifact_path = ROOT / str(entry.get("artifact_path") or "")
        if not artifact_path.exists():
            errors.append(f"{class_key}: missing artifact {artifact_path}")
            continue
        actual_sha = _sha256(artifact_path)
        if actual_sha != entry.get("artifact_sha256"):
            errors.append(f"{class_key}: checksum mismatch")
        artifact = _read_json(artifact_path)
        strategy = artifact.get("score_strategy") or {}
        strategy_type = strategy.get("type")
        if strategy_type == "descriptor_only_until_promoted":
            errors.append(f"{class_key}: descriptor-only strategy")
        if strategy_type not in {
            "response_score_passthrough",
            "linear_feature_model",
            "tfidf_linear_model",
            "rule_feature_model",
        }:
            errors.append(f"{class_key}: unsupported strategy {strategy_type!r}")
        class_policy = policy.get(class_key) or {}
        if class_policy.get("allow_disabled") or class_policy.get("block_disabled"):
            errors.append(f"{class_key}: policy not allow/block enabled")
        if int(class_policy.get("allowed") or 0) <= 0:
            errors.append(f"{class_key}: no safe autonomous allows recorded")
        if int(class_policy.get("blocked") or 0) <= 0:
            errors.append(f"{class_key}: no safe autonomous blocks recorded")
        if class_policy.get("status") == "runtime_head_promoted_block_only":
            errors.append(f"{class_key}: block-only promotion status is not autonomous-ready")
    return errors


def run_pytest() -> int:
    return subprocess.call([sys.executable, "-m", "pytest", *FOCUSED_TESTS], cwd=ROOT)


def run_live_smoke() -> int:
    if not (os.environ.get("RUNPOD_ENDPOINT_ID") and os.environ.get("RUNPOD_API_KEY")):
        print("live RunPod smoke skipped: set RUNPOD_ENDPOINT_ID and RUNPOD_API_KEY", file=sys.stderr)
        return 0
    return subprocess.call(
        [sys.executable, "-m", "pytest", "tests/test_runpod_live_endpoint.py", "-q"],
        cwd=ROOT,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-pytest", action="store_true")
    parser.add_argument("--live-runpod", action="store_true")
    args = parser.parse_args()

    errors = check_artifacts()
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("artifact/policy readiness checks passed")

    if not args.skip_pytest:
        rc = run_pytest()
        if rc != 0:
            return rc
    if args.live_runpod:
        return run_live_smoke()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
