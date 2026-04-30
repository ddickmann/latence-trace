from __future__ import annotations

import json
from types import SimpleNamespace

from latence_trace.core import runtime_decision


def _response(score: float, class_key: str = "rag.prose.enterprise") -> SimpleNamespace:
    return SimpleNamespace(
        scores=SimpleNamespace(
            groundedness_v2=score,
            primary_score=score,
            primary_name="groundedness_v2",
            risk_band="green" if score >= 0.83 else "red",
            composite_phantom_score=None,
        ),
        corpus_route=SimpleNamespace(corpus_type=class_key),
        support_units=[
            SimpleNamespace(
                index=0,
                support_id="doc-1",
                text="The contract says renewal requires written approval.",
                coverage_score=0.91,
                usage_confidence=0.8,
                usage_state=SimpleNamespace(value="used"),
            )
        ],
        response_tokens=[
            SimpleNamespace(
                index=0,
                token="phantom",
                heatmap_score=0.2,
                nli_score=-0.7,
                reverse_context_calibrated=0.3,
                char_start=0,
                char_end=7,
            )
        ],
        file_attribution=None,
        reason=None,
    )


def test_runtime_decision_disabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("LATENCE_TRACE_RUNTIME_DECISION_ENABLED", raising=False)
    runtime_decision.reset_policy_cache_for_tests()

    assert runtime_decision.build_runtime_decision(_response(0.99)) is None


def test_enterprise_policy_can_allow_and_block(monkeypatch) -> None:
    monkeypatch.setenv("LATENCE_TRACE_RUNTIME_DECISION_ENABLED", "1")
    runtime_decision.reset_policy_cache_for_tests()

    allowed = runtime_decision.build_runtime_decision(_response(0.91))
    blocked = runtime_decision.build_runtime_decision(_response(0.42))

    assert allowed is not None
    assert allowed["action"] == "allow"
    assert allowed["class_key"] == "rag.prose.enterprise"
    assert allowed["head_id"] == "optimized_calibrator"
    assert allowed["head_version"] == "root_cause_solution_v1"
    assert allowed["head_enabled"] is True
    assert allowed["head_score"] == 0.91
    assert allowed["head_features_used"] == ["groundedness_v2"]
    assert "enterprise_optimized_calibrator_active" in allowed["head_reason_codes"]
    assert allowed["head_registry_sha256"]
    assert allowed["evidence"][0]["support_id"] == "doc-1"
    assert allowed["unsupported_spans"][0]["token"] == "phantom"

    assert blocked is not None
    assert blocked["action"] == "block"
    assert blocked["rollback_safe"] is True


def test_underproven_classes_are_auto_repair_only(monkeypatch) -> None:
    monkeypatch.setenv("LATENCE_TRACE_RUNTIME_DECISION_ENABLED", "1")
    runtime_decision.reset_policy_cache_for_tests()

    record = runtime_decision.build_runtime_decision(
        _response(0.99, class_key="code.agentic_trace")
    )

    assert record is not None
    assert record["action"] == "auto_repair"
    assert record["head_id"] == "trajectory_ranker"
    assert record["head_enabled"] is False
    assert record["head_score"] is None
    assert "head_disabled_repair_only" in record["head_reason_codes"]
    assert record["allow_disabled"] is True
    assert record["block_disabled"] is True


def test_missing_head_artifact_falls_back(monkeypatch, tmp_path) -> None:
    registry = {
        "runtime_head_registry": {
            "rag.prose.enterprise": {
                "class_key": "rag.prose.enterprise",
                "enabled": True,
                "head_id": "missing_head",
                "version": "test",
                "artifact_path": str(tmp_path / "missing.json"),
            }
        }
    }
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    monkeypatch.setenv("LATENCE_TRACE_RUNTIME_DECISION_ENABLED", "1")
    monkeypatch.setenv("LATENCE_TRACE_RUNTIME_HEAD_REGISTRY_PATH", str(registry_path))
    runtime_decision.reset_policy_cache_for_tests()

    record = runtime_decision.build_runtime_decision(_response(0.91))

    assert record is not None
    assert record["action"] == "allow"
    assert record["head_id"] == "missing_head"
    assert record["head_enabled"] is False
    assert record["head_score"] is None
    assert any("head_artifact_unavailable" in code for code in record["head_reason_codes"])


def test_corrupt_head_artifact_falls_back(monkeypatch, tmp_path) -> None:
    artifact_path = tmp_path / "head.json"
    artifact_path.write_text("{not-json", encoding="utf-8")
    registry = {
        "runtime_head_registry": {
            "rag.prose.enterprise": {
                "class_key": "rag.prose.enterprise",
                "enabled": True,
                "head_id": "corrupt_head",
                "version": "test",
                "artifact_path": str(artifact_path),
            }
        }
    }
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    monkeypatch.setenv("LATENCE_TRACE_RUNTIME_DECISION_ENABLED", "1")
    monkeypatch.setenv("LATENCE_TRACE_RUNTIME_HEAD_REGISTRY_PATH", str(registry_path))
    runtime_decision.reset_policy_cache_for_tests()

    record = runtime_decision.build_runtime_decision(_response(0.91))

    assert record is not None
    assert record["action"] == "allow"
    assert record["head_enabled"] is False
    assert any("load_failed" in code for code in record["head_reason_codes"])
