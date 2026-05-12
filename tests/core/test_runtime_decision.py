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


def _trajectory_response(
    features: dict[str, float],
    *,
    risk_band: str | None = None,
) -> SimpleNamespace:
    """Build a fixture for the head-driven trajectory ranker.

    Production responses always carry a calibration ``risk_band`` aligned
    with the underlying score. Test stubs that exercise the head path in
    isolation must therefore stamp a calibration band consistent with
    the head's intended outcome (grounded -> green, ungrounded -> red);
    otherwise the runtime decision's band-coercion stage will quite
    correctly override the head's verdict to match calibration.
    """

    response = _response(0.5, class_key="code.agentic_trace")
    response.runtime_head_features = features
    if risk_band is not None:
        response.scores.risk_band = risk_band
    return response


def _head_feature_response(
    class_key: str,
    features: dict[str, float],
    *,
    risk_band: str | None = None,
) -> SimpleNamespace:
    response = _response(0.5, class_key=class_key)
    response.runtime_head_features = features
    response.response_text = (
        "The answer cites the exact supported atom, matching schema cell, "
        "and verified symbol from the evidence."
    )
    if risk_band is not None:
        response.scores.risk_band = risk_band
    return response


def _structured_response_with_warning(warning: str | None = None) -> SimpleNamespace:
    # The clean (no-warning) case has a fully-grounded response so
    # calibration is green; the warning cases carry a specific literal
    # mismatch which calibration would correctly mark as amber. Setting
    # the band on the fixture matches the production (calibration ≡
    # head verdict severity) contract — without the band stamp the test
    # would simultaneously claim "no risk" via calibration and "must
    # auto-repair" via the structured guard, which never happens in
    # production.
    risk_band = "green" if warning is None else "amber"
    response = _head_feature_response(
        "rag.structured",
        _structured_allow_features(),
        risk_band=risk_band,
    )
    response.warnings = [] if warning is None else [warning]
    return response


def _structured_allow_features() -> dict[str, float]:
    features = _good_root_cause_features()
    features.update(
        {
            "v1_score": 0.0,
            "reverse_context": 0.0,
            "groundedness_v2": 0.0,
            "literal_guarded": 0.0,
            "claim_count": 3.0,
        }
    )
    return features


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

    # In production the calibration band would also be green for a
    # grounded response (and red for an ungrounded one); stamp the
    # consistent band on the fixture so the head's verdict is not
    # later coerced by the band-coercion stage.
    allowed = runtime_decision.build_runtime_decision(
        _trajectory_response(grounded_features, risk_band="green")
    )
    blocked = runtime_decision.build_runtime_decision(
        _trajectory_response(ungrounded_features, risk_band="red")
    )
    # Missing features path: head is disabled. We stamp an amber
    # calibration band to match the production "we don't have enough
    # signal to make a confident allow/block call" semantics; without
    # that explicit amber the band-coercion stage would (correctly)
    # promote auto_repair to allow when calibration claims the response
    # is fully grounded.
    missing_features_fixture = _response(0.99, class_key="code.agentic_trace")
    missing_features_fixture.scores.risk_band = "amber"
    missing_features = runtime_decision.build_runtime_decision(missing_features_fixture)

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
        # Calibration band on a fully-grounded fixture is green.
        record = runtime_decision.build_runtime_decision(
            _head_feature_response(class_key, _good_root_cause_features(), risk_band="green")
        )
        assert record is not None
        assert record["head_enabled"] is True
        assert record["head_score"] is not None
        assert record["score_channel"].startswith("head:")
        assert record["allow_disabled"] is False
        assert record["block_disabled"] is False

        # Bad features → calibration band is red; both head and
        # calibration converge on block.
        blocked = runtime_decision.build_runtime_decision(
            _head_feature_response(
                class_key,
                _bad_root_cause_features(),
                risk_band="red",
            )
        )
        assert blocked is not None
        assert blocked["action"] == "block"
        assert blocked["head_enabled"] is True

        # Missing-feature path: stamp an explicit amber calibration band
        # so the repair-only verdict survives the band-coercion stage
        # (without it, calibration's green from the placeholder score
        # would correctly promote auto_repair to allow).
        missing_fixture = _response(0.99, class_key=class_key)
        missing_fixture.scores.risk_band = "amber"
        missing = runtime_decision.build_runtime_decision(missing_fixture)
        assert missing is not None
        assert missing["action"] == "auto_repair"
        assert missing["head_enabled"] is False
        assert "head_features_missing_repair_only" in missing["head_reason_codes"]


def test_guardian_green_hot_path_is_not_demoted_by_nli_era_head(monkeypatch) -> None:
    monkeypatch.setenv("LATENCE_TRACE_RUNTIME_DECISION_ENABLED", "1")
    runtime_decision.reset_policy_cache_for_tests()

    response = _response(0.99, class_key="rag.prose.multi_claim")
    response.scores.risk_band = "green"
    response.scores.guardian_aggregate = 0.99
    response.scores.nli_aggregate = None

    record = runtime_decision.build_runtime_decision(response)

    assert record is not None
    assert record["action"] == "allow"
    assert record["band"] == "green"
    assert record["head_enabled"] is False
    assert "head_features_missing_repair_only" in record["head_reason_codes"]
    assert "guardian_green_calibration_allow" in record["reason_codes"]


def test_structured_measurement_literal_mismatch_forces_repair_without_blocking_clean_allow(
    monkeypatch,
) -> None:
    monkeypatch.setenv("LATENCE_TRACE_RUNTIME_DECISION_ENABLED", "1")
    runtime_decision.reset_policy_cache_for_tests()

    clean = runtime_decision.build_runtime_decision(_structured_response_with_warning())
    mismatch = runtime_decision.build_runtime_decision(
        _structured_response_with_warning(
            "literal_mismatch: 2 response literal(s) not present in support: "
            "identifier=p95, measurement=120 seconds"
        )
    )
    number_mismatch = runtime_decision.build_runtime_decision(
        _structured_response_with_warning(
            "literal_mismatch: 1 response literal(s) not present in support: number=8420"
        )
    )
    currency_mismatch = runtime_decision.build_runtime_decision(
        _structured_response_with_warning(
            "literal_mismatch: 1 response literal(s) not present in support: currency=8420 EUR"
        )
    )
    identifier_only = runtime_decision.build_runtime_decision(
        _structured_response_with_warning(
            "literal_mismatch: 1 response literal(s) not present in support: identifier=p95"
        )
    )

    assert clean is not None
    assert clean["action"] == "allow"
    assert clean["band"] == "green"

    assert mismatch is not None
    assert mismatch["action"] == "auto_repair"
    assert mismatch["band"] == "amber"
    assert "structured_measurement_literal_mismatch_repair_only" in mismatch["reason_codes"]

    assert number_mismatch is not None
    assert number_mismatch["action"] == "auto_repair"
    assert number_mismatch["band"] == "amber"
    assert "structured_numeric_literal_mismatch_repair_only" in number_mismatch["reason_codes"]

    assert currency_mismatch is not None
    assert currency_mismatch["action"] == "auto_repair"
    assert currency_mismatch["band"] == "amber"
    assert "structured_currency_literal_mismatch_repair_only" in currency_mismatch["reason_codes"]

    assert identifier_only is not None
    assert identifier_only["action"] == "allow"


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


def test_red_calibration_band_overrides_head_driven_auto_repair(monkeypatch) -> None:
    """Regression for the German Kafka case.

    The runtime head can return a borderline ``head_score`` that maps
    to ``auto_repair`` even when the per-class calibration bundle
    correctly bands the response as ``red``. Before the band-coercion
    stage shipped, the live overlay would show ``Amber`` while the
    event log showed ``Red`` for the same response. Pin the new
    one-way coercion: red calibration band → ``block`` action even
    when heads would otherwise repair.
    """

    monkeypatch.setenv("LATENCE_TRACE_RUNTIME_DECISION_ENABLED", "1")
    runtime_decision.reset_policy_cache_for_tests()

    # Simulate the Kafka case: groundedness_v2 mid-range, calibration
    # band red (per the rag.prose.multi_claim DE bundle, amber=0.90).
    response = _response(0.5, class_key="rag.prose.multi_claim")
    response.scores.risk_band = "red"
    response.runtime_head_features = _good_root_cause_features()

    record = runtime_decision.build_runtime_decision(response)

    assert record is not None
    assert record["band"] == "red", "band must reflect the calibration verdict"
    assert record["action"] == "block", "red calibration must block, not auto_repair"
    assert any(
        code.startswith("calibration_band_coercion:red") for code in record["reason_codes"]
    ), "expected reason_codes to record the coercion event"


def test_red_calibration_falls_back_to_auto_repair_when_block_disabled(
    monkeypatch,
    tmp_path,
) -> None:
    """When operators disable block on a class, red calibration must
    drop to ``auto_repair`` rather than ignoring the disable flag."""

    policy = {
        "channel": "test_block_disabled",
        "default": {
            "allow_threshold": 1.0,
            "block_threshold": 0.0,
            "allow_disabled": False,
            "block_disabled": True,
        },
        "classes": {
            "rag.prose.enterprise": {
                "allow_threshold": 0.9,
                "block_threshold": 0.1,
                "allow_disabled": False,
                "block_disabled": True,
            }
        },
    }
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    monkeypatch.setenv("LATENCE_TRACE_RUNTIME_DECISION_ENABLED", "1")
    monkeypatch.setenv("LATENCE_TRACE_RUNTIME_POLICY_PATH", str(policy_path))
    runtime_decision.reset_policy_cache_for_tests()

    response = _response(0.5, class_key="rag.prose.enterprise")
    response.scores.risk_band = "red"
    record = runtime_decision.build_runtime_decision(response)

    assert record is not None
    assert record["band"] == "red"
    assert record["action"] == "auto_repair", "block_disabled must downgrade block to auto_repair"


def test_amber_calibration_does_not_disturb_head_driven_allow(monkeypatch) -> None:
    """Amber calibration leaves head/policy verdicts untouched.

    The coercion is intentionally one-way: only red overrides. Amber
    calibration must NOT downgrade a head-driven ``allow`` to
    ``auto_repair`` because the head's structured guards are precisely
    the lane that catches finer-grained issues calibration may have
    missed; a blanket amber→auto_repair would erase that signal.
    """

    monkeypatch.setenv("LATENCE_TRACE_RUNTIME_DECISION_ENABLED", "1")
    runtime_decision.reset_policy_cache_for_tests()

    response = _head_feature_response(
        "rag.structured",
        _structured_allow_features(),
        risk_band="amber",
    )
    record = runtime_decision.build_runtime_decision(response)

    assert record is not None
    assert record["band"] == "amber", "band reflects calibration"
    assert record["action"] == "allow", "amber must not coerce head-driven allow"
