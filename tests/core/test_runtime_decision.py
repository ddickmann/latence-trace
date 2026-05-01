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


def _trajectory_response(features: dict[str, float]) -> SimpleNamespace:
    response = _response(0.5, class_key="code.agentic_trace")
    response.runtime_head_features = features
    return response


def _head_feature_response(class_key: str, features: dict[str, float]) -> SimpleNamespace:
    response = _response(0.5, class_key=class_key)
    response.runtime_head_features = features
    response.response_text = (
        "The answer cites the exact supported atom, matching schema cell, "
        "and verified symbol from the evidence."
    )
    return response


def _good_root_cause_features() -> dict[str, float]:
    return {
        "v1_score": 0.99,
        "v1_nli_aggregate": 0.9,
        "reverse_context": 0.99,
        "groundedness_v2": 0.99,
        "literal_guarded": 0.99,
        "literal_mismatch_count": 0.0,
        "context_coverage_ratio": 1.0,
        "context_unused_ratio": 0.0,
        "dead_weight_ratio": 0.0,
        "token_mean": 0.99,
        "token_bottom10": 0.99,
        "token_saturation_rate": 0.0,
        "calibrated_mean": 0.99,
        "nli_token_mean": 0.9,
        "literal_coverage": 1.0,
        "atom_match": 1.0,
        "claim_count": 1.0,
        "unsupported_claim_fraction": 0.0,
        "schema_match": 1.0,
        "api_symbol_match": 1.0,
        "trajectory_order_match": 1.0,
        "test_outcome_match": 1.0,
        "numeric_coverage": 1.0,
        "coverage_label_mean": 1.0,
        "row_alignment": 1.0,
        "column_alignment": 1.0,
        "value_alignment": 1.0,
        "unit_alignment": 1.0,
        "date_alignment": 1.0,
        "numeric_tolerance_match": 1.0,
        "schema_alias_match": 1.0,
        "cell_provenance_match": 1.0,
    }


def _bad_root_cause_features() -> dict[str, float]:
    features = {key: 0.0 for key in _good_root_cause_features()}
    features.update(
        {
            "literal_mismatch_count": 10.0,
            "context_unused_ratio": 1.0,
            "dead_weight_ratio": 1.0,
            "token_saturation_rate": 1.0,
            "unsupported_claim_fraction": 1.0,
            "claim_count": 1.0,
            "numeric_count": 1.0,
        }
    )
    return features


def test_runtime_decision_uses_production_default(monkeypatch) -> None:
    monkeypatch.delenv("LATENCE_TRACE_RUNTIME_DECISION_ENABLED", raising=False)
    runtime_decision.reset_policy_cache_for_tests()

    assert runtime_decision.build_runtime_decision(_response(0.99)) is not None


def test_runtime_decision_can_be_disabled_by_rollback_switch(monkeypatch) -> None:
    monkeypatch.setenv("LATENCE_TRACE_RUNTIME_DECISION_ENABLED", "0")
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


def test_agentic_trace_policy_uses_promoted_head(monkeypatch) -> None:
    monkeypatch.setenv("LATENCE_TRACE_RUNTIME_DECISION_ENABLED", "1")
    runtime_decision.reset_policy_cache_for_tests()

    grounded_features = {
        "file_alignment": 1.0,
        "symbol_alignment": 1.0,
        "test_outcome_alignment": 1.0,
        "patch_alignment": 1.0,
        "temporal_order_alignment": 1.0,
        "claim_atom_coverage": 1.0,
        "unsupported_atom_rate": 0.0,
        "phantom_symbol_rate": 0.0,
        "missing_command_evidence": 0.0,
        "literal_match_rate": 1.0,
        "literal_mismatch_rate": 0.0,
        "identifier_query_overlap": 1.0,
        "identifier_query_absent_rate": 0.0,
        "warning_identifier_rate": 0.0,
        "reverse_context": 0.95,
        "consensus_hardened": 0.95,
        "groundedness_v2": 0.95,
        "triangular": 0.95,
        "context_attribution_ratio": 1.0,
        "context_uncertain_ratio": 0.0,
        "dead_weight_ratio": 0.0,
        "support_usage_rate": 1.0,
        "context_token_log": 5.0,
        "multi_cell_reverse_min": 0.95,
        "multi_cell_reverse_max": 0.95,
        "multi_cell_reverse_std": 0.0,
    }
    ungrounded_features = {key: 1.0 - value for key, value in grounded_features.items()}
    ungrounded_features["context_token_log"] = 5.0
    ungrounded_features["missing_command_evidence"] = 0.0

    allowed = runtime_decision.build_runtime_decision(_trajectory_response(grounded_features))
    blocked = runtime_decision.build_runtime_decision(_trajectory_response(ungrounded_features))
    missing_features = runtime_decision.build_runtime_decision(_response(0.99, class_key="code.agentic_trace"))

    assert allowed is not None
    assert allowed["action"] == "allow"
    assert allowed["band"] == "green"
    assert allowed["head_id"] == "trajectory_ranker"
    assert allowed["head_enabled"] is True
    assert allowed["head_score"] is not None
    assert "trajectory_native_head_active" in allowed["head_reason_codes"]
    assert allowed["score_channel"] == "head:trajectory_ranker"
    assert allowed["allow_disabled"] is False
    assert allowed["block_disabled"] is False

    assert blocked is not None
    assert blocked["action"] == "block"
    assert blocked["band"] == "red"
    assert blocked["head_enabled"] is True

    assert missing_features is not None
    assert missing_features["action"] == "auto_repair"
    assert missing_features["band"] == "amber"
    assert missing_features["head_enabled"] is False
    assert "trajectory_features_missing_repair_only" in missing_features["head_reason_codes"]


def test_all_promoted_heads_are_executable_or_feature_gated(monkeypatch) -> None:
    monkeypatch.setenv("LATENCE_TRACE_RUNTIME_DECISION_ENABLED", "1")
    runtime_decision.reset_policy_cache_for_tests()

    feature_classes = {
        "rag.code_in_context",
        "rag.prose.multi_claim",
        "rag.prose.short_factoid",
        "rag.structured",
    }
    for class_key in feature_classes:
        record = runtime_decision.build_runtime_decision(
            _head_feature_response(class_key, _good_root_cause_features())
        )
        assert record is not None
        assert record["head_enabled"] is True
        assert record["head_score"] is not None
        assert record["score_channel"].startswith("head:")
        assert record["allow_disabled"] is False
        assert record["block_disabled"] is False

        blocked = runtime_decision.build_runtime_decision(
            _head_feature_response(
                class_key,
                _bad_root_cause_features(),
            )
        )
        assert blocked is not None
        assert blocked["action"] == "block"
        assert blocked["head_enabled"] is True

        missing = runtime_decision.build_runtime_decision(_response(0.99, class_key=class_key))
        assert missing is not None
        assert missing["action"] == "auto_repair"
        assert missing["head_enabled"] is False
        assert "head_features_missing_repair_only" in missing["head_reason_codes"]


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
