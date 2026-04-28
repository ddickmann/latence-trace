"""Unit tests for latence_trace.middleware.amber_escalation (Phase 2)."""

from __future__ import annotations

from latence_trace.middleware.amber_escalation import (
    AmberEscalationConfig,
    AmberEscalationPayload,
    build_payload,
    escalate,
)


def test_config_from_env_defaults_off() -> None:
    cfg = AmberEscalationConfig.from_env({})
    assert cfg.enabled is False
    assert cfg.provider == "offline"


def test_config_from_env_enables_offline_even_without_api_key() -> None:
    cfg = AmberEscalationConfig.from_env(
        {"LATENCE_TRACE_AUTO_DECIDE_ENABLED": "1"}
    )
    assert cfg.enabled is True
    assert cfg.provider == "offline"


def test_config_from_env_openai_without_key_falls_back_to_offline() -> None:
    cfg = AmberEscalationConfig.from_env(
        {
            "LATENCE_TRACE_AUTO_DECIDE_ENABLED": "1",
            "LATENCE_TRACE_AUTO_DECIDE_PROVIDER": "openai",
            "OPENAI_API_KEY": "",
        }
    )
    assert cfg.provider == "offline"


def test_config_from_env_openai_with_key() -> None:
    cfg = AmberEscalationConfig.from_env(
        {
            "LATENCE_TRACE_AUTO_DECIDE_ENABLED": "1",
            "LATENCE_TRACE_AUTO_DECIDE_PROVIDER": "openai",
            "OPENAI_API_KEY": "sk-test",
            "LATENCE_TRACE_AUTO_DECIDE_MODEL": "gpt-4o-mini",
        }
    )
    assert cfg.enabled is True
    assert cfg.provider == "openai"
    assert cfg.model == "gpt-4o-mini"
    assert cfg.api_key == "sk-test"


def _offline_cfg(**kwargs: object) -> AmberEscalationConfig:
    return AmberEscalationConfig(
        enabled=True,
        provider="offline",
        model="heuristic",
        **kwargs,  # type: ignore[arg-type]
    )


def test_escalate_disabled_returns_amber() -> None:
    cfg = AmberEscalationConfig(enabled=False, provider="offline")
    payload = AmberEscalationPayload(query="q", response_text="r")
    verdict = escalate(payload, cfg)
    assert verdict.verdict == "amber"
    assert verdict.reasoning == "auto_decide_disabled"


def test_offline_heuristic_returns_red_on_high_contradiction() -> None:
    cfg = _offline_cfg()
    payload = AmberEscalationPayload(
        query="q",
        response_text="r",
        claims=[
            {"claim": "c1", "entailment": 0.1, "contradiction": 0.8, "score": 0.2}
        ],
    )
    v = escalate(payload, cfg)
    assert v.verdict == "red"
    assert "contradiction" in v.reasoning


def test_offline_heuristic_returns_green_on_high_entailment() -> None:
    cfg = _offline_cfg()
    payload = AmberEscalationPayload(
        query="q",
        response_text="r",
        claims=[
            {"claim": "c1", "entailment": 0.9, "contradiction": 0.05, "score": 0.8}
        ],
    )
    v = escalate(payload, cfg)
    assert v.verdict == "green"


def test_offline_heuristic_uses_fallback_score_in_hedge() -> None:
    cfg = _offline_cfg()
    payload = AmberEscalationPayload(
        query="q",
        response_text="r",
        claims=[
            {"claim": "c1", "entailment": 0.3, "contradiction": 0.1, "score": 0.4}
        ],
    )
    # entailment < 0.6, contradiction < 0.5, but fallback score >= 0.55 => green
    v = escalate(payload, cfg, fallback_score=0.7)
    assert v.verdict == "green"

    v2 = escalate(payload, cfg, fallback_score=0.3)
    assert v2.verdict == "red"


def test_build_payload_respects_max_claims_and_evidence() -> None:
    cfg = AmberEscalationConfig(
        enabled=True, provider="offline", max_claims=2, max_evidence_chars=10
    )
    nli_claims = [
        {
            "text": f"claim {i}",
            "score": 0.5,
            "entailment": 0.4,
            "contradiction": 0.2,
            "premises": ["this is a very long premise"],
            "atoms": [{"text": f"atom {j}"} for j in range(5)],
        }
        for i in range(5)
    ]
    payload = build_payload(
        query_text="q",
        response_text="r",
        nli_claims=nli_claims,
        config=cfg,
    )
    assert len(payload.claims) == 2
    first = payload.claims[0]
    assert first["claim"] == "claim 0"
    assert first["evidence"] is not None
    assert len(first["evidence"]) <= 10
    assert len(first["atoms"]) <= 3


def test_build_payload_skips_empty_claim_text() -> None:
    cfg = AmberEscalationConfig(enabled=True, provider="offline")
    payload = build_payload(
        query_text="q",
        response_text="r",
        nli_claims=[{"text": "", "score": 0.9}, {"text": "c1", "score": 0.8}],
        config=cfg,
    )
    assert [c["claim"] for c in payload.claims] == ["c1"]


def test_clip_and_verdict_parse() -> None:
    from latence_trace.middleware.amber_escalation import _clip, _parse_verdict_json

    assert _clip("abc", 10) == "abc"
    assert _clip("abcdef", 5).endswith("\u2026")
    # Parses plain JSON
    r = _parse_verdict_json('{"verdict": "green", "reasoning": "ok"}')
    assert r == ("green", "ok")
    # Parses JSON in code fence
    r2 = _parse_verdict_json('```json\n{"verdict":"red","reasoning":"x"}\n```')
    assert r2 == ("red", "x")
    # Rejects unknown verdict
    assert _parse_verdict_json('{"verdict":"blue"}') is None
    # Rejects non-JSON
    assert _parse_verdict_json("hello") is None
