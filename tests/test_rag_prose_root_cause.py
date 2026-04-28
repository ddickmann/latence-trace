"""Unit tests for the RAG-prose root-cause fix.

These tests pin down the two changes that unblock enterprise RAG on
multilingual legal / cyber prose:

1. ``detect_source_format`` no longer trips on sentence-shaped chunks with
   a handful of incidental ``Label number`` patterns (prose-safety guard);
2. ``fuse_groundedness_v2`` only collapses the headline via
   ``min(narrative, structured_min)`` for true structured sources or
   strongly-aligned typed lanes — prose is folded as a weighted channel.

Also covers the new ``structured_verification`` opt-out, the near-
verbatim / lexical rescue floors, and the new ``rag_prose`` threshold
stratum.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from latence_trace.core.structured import (
    detect_source_format,
    resolve_structured_mode,
    _support_is_prose,
)
from latence_trace.core.nli import fuse_groundedness_v2
from latence_trace.core.groundedness import (
    _lexical_rescue_floor,
    _resolve_effective_stratum,
    _verbatim_support_floor,
)
from latence_trace.core.thresholds import (
    RiskBandPolicy,
    classify_risk_band,
    load_risk_band_policy,
)


# ----------------------------------------------------------------------
# detect_source_format & prose-safety guard
# ----------------------------------------------------------------------


LEGAL_PROSE_FR = (
    "L'article 32 du RGPD impose la mise en place de mesures techniques et "
    "organisationnelles appropriees afin de garantir un niveau de securite "
    "adapte au risque. Les responsables de traitement doivent notamment "
    "chiffrer les donnees, pseudonymiser les identifiants et preserver la "
    "confidentialite des systemes. En cas de violation, l'autorite de "
    "controle doit etre notifiee dans les 72 heures."
)


CYBER_PROSE_EN = (
    "The NIS2 directive requires entities to apply proportionate and "
    "risk-based security measures across their network and information "
    "systems. Incidents that cause a significant impact must be reported "
    "to the CSIRT within 24 hours, with a final report following inside "
    "72 hours. Affected entities should also coordinate with vendors to "
    "patch affected components within a reasonable timeframe."
)


ENTERPRISE_RAG_PROSE_WITH_METRICS = (
    "Revenue grew across all regions with Americas at 42%, Europe at 38%, "
    "and Asia at 21%, reflecting strong demand from enterprise customers. "
    "Management expects the trend to continue into the next quarter."
)


TRUE_PROSE_TABLE = (
    "Segment revenue Q3 FY24: Americas $37,678; Europe $21,883; "
    "Greater China $14,728; Japan $6,253; Rest of Asia Pacific $6,665; "
    "Wearables $7,829; Services $22,314."
)


TRUE_KV_BLOCK = (
    "Revenue: 1200\n"
    "Operating margin: 38%\n"
    "Gross margin: 45%\n"
    "EPS: 2.10\n"
    "Headcount: 12500\n"
    "Cash position: 28400"
)


TRUE_MARKDOWN_TABLE = (
    "| Segment | Q3 | Q4 |\n"
    "| --- | --- | --- |\n"
    "| Cloud | 42 | 45 |\n"
    "| Ads | 58 | 62 |"
)


TRUE_JSON = '{"revenue": 1200, "margin": 0.38, "segment": "Cloud"}'


def test_support_is_prose_detects_paragraph_prose() -> None:
    assert _support_is_prose(LEGAL_PROSE_FR) is True
    assert _support_is_prose(CYBER_PROSE_EN) is True
    assert _support_is_prose(ENTERPRISE_RAG_PROSE_WITH_METRICS) is True


def test_support_is_prose_rejects_structured_sources() -> None:
    assert _support_is_prose(TRUE_KV_BLOCK) is False
    assert _support_is_prose(TRUE_MARKDOWN_TABLE) is False


def test_detect_source_format_skips_prose_with_incidental_numbers() -> None:
    """Regression: plain RAG prose with 3 metrics must not be 'prose_table'."""
    assert detect_source_format(ENTERPRISE_RAG_PROSE_WITH_METRICS) is None
    assert detect_source_format(CYBER_PROSE_EN) is None
    assert detect_source_format(LEGAL_PROSE_FR) is None


def test_detect_source_format_still_catches_true_tables() -> None:
    assert detect_source_format(TRUE_JSON) == "json"
    assert detect_source_format(TRUE_MARKDOWN_TABLE) == "markdown_table"
    assert detect_source_format(TRUE_PROSE_TABLE) == "prose_table"
    assert detect_source_format(TRUE_KV_BLOCK) == "prose_table"


def test_detect_source_format_hint_on_forces_detection() -> None:
    """``content_type`` hint overrides the prose-safety guard for true tables."""
    assert (
        detect_source_format(TRUE_MARKDOWN_TABLE, content_type="text/markdown")
        == "markdown_table"
    )


def test_detect_source_format_numeric_fact_honours_response_length() -> None:
    """``numeric_fact`` suppressed when the response is long narrative."""
    short_fact = "Senkung des Einlagesatzes um 25 Basispunkte auf 3,50 %."
    long_response = " ".join(["detail"] * 120)
    assert detect_source_format(short_fact, response_text="OK") == "numeric_fact"
    assert detect_source_format(short_fact, response_text=long_response) is None


# ----------------------------------------------------------------------
# resolve_structured_mode
# ----------------------------------------------------------------------


def test_resolve_structured_mode_defaults_to_auto() -> None:
    assert resolve_structured_mode(None) == "auto"
    assert resolve_structured_mode("") == "auto"
    assert resolve_structured_mode("weird") == "auto"


def test_resolve_structured_mode_parses_on_off() -> None:
    assert resolve_structured_mode("on") == "on"
    assert resolve_structured_mode("off") == "off"
    assert resolve_structured_mode("0") == "off"
    assert resolve_structured_mode("force") == "on"


# ----------------------------------------------------------------------
# fuse_groundedness_v2 precise gate
# ----------------------------------------------------------------------


_GREEN_WEIGHTS = {
    "calibrated": 0.0,
    "literal": 0.0,
    "nli": 1.0,
    "semantic_entropy": 0.0,
    "structured": 0.0,
}


def test_fuse_does_not_apply_min_gate_on_prose_with_zero_typed_alignment() -> None:
    """Prose-shaped typed score must not collapse the narrative."""
    fused = fuse_groundedness_v2(
        reverse_context_calibrated=0.95,
        literal_guarded=0.95,
        nli_aggregate=0.95,
        typed_structured=0.05,
        typed_structured_gate=True,
        source_format="prose_table",
        typed_claims_matched=0,
        weights=_GREEN_WEIGHTS,
    )
    assert fused == pytest.approx(0.95)


def test_fuse_applies_min_gate_on_true_json_source() -> None:
    fused = fuse_groundedness_v2(
        reverse_context_calibrated=0.95,
        literal_guarded=0.95,
        nli_aggregate=0.95,
        typed_structured=0.10,
        typed_structured_gate=True,
        source_format="json",
        typed_claims_matched=0,
        weights=_GREEN_WEIGHTS,
    )
    assert fused == pytest.approx(0.10)


def test_fuse_applies_min_gate_on_strong_typed_alignment() -> None:
    fused = fuse_groundedness_v2(
        reverse_context_calibrated=0.90,
        literal_guarded=0.90,
        nli_aggregate=0.90,
        typed_structured=0.30,
        typed_structured_gate=True,
        source_format="prose_table",
        typed_claims_matched=5,
        weights=_GREEN_WEIGHTS,
    )
    assert fused == pytest.approx(0.30)


def test_fuse_backwards_compatible_when_gate_flags_missing() -> None:
    fused = fuse_groundedness_v2(
        reverse_context_calibrated=0.80,
        literal_guarded=0.80,
        nli_aggregate=0.80,
        typed_structured=None,
        weights=_GREEN_WEIGHTS,
    )
    assert fused == pytest.approx(0.80)


# ----------------------------------------------------------------------
# Lexical rescue / verbatim floor
# ----------------------------------------------------------------------


def test_verbatim_support_floor_near_verbatim_fr() -> None:
    response = (
        "Les responsables de traitement doivent chiffrer les donnees et "
        "pseudonymiser les identifiants."
    )
    support = [
        {
            "text": (
                "Les responsables de traitement doivent chiffrer les donnees "
                "et pseudonymiser les identifiants pour respecter l'article 32."
            )
        }
    ]
    assert _verbatim_support_floor(response, support) == pytest.approx(0.95)


def test_lexical_rescue_fires_when_lexical_channels_agree() -> None:
    response = (
        "Les entites concernees doivent notifier la violation dans les 72 heures "
        "a l'autorite de controle designee."
    )
    support = [
        {
            "text": (
                "En cas de violation, l'autorite de controle doit etre notifiee "
                "dans les 72 heures par les entites concernees."
            )
        }
    ]

    rescue = _lexical_rescue_floor(
        response,
        support,
        reverse_context_calibrated=0.90,
        literal_guarded=0.88,
        nli_aggregate=0.60,
    )
    assert rescue == pytest.approx(0.85)


def test_lexical_rescue_stays_silent_when_nli_already_green() -> None:
    rescue = _lexical_rescue_floor(
        "response that mentions incident reporting rules",
        [{"text": "support that mentions incident reporting rules"}],
        reverse_context_calibrated=0.95,
        literal_guarded=0.95,
        nli_aggregate=0.90,
    )
    assert rescue is None


def test_lexical_rescue_rejects_low_lexical_overlap() -> None:
    rescue = _lexical_rescue_floor(
        "the response discusses a completely different subject",
        [{"text": "support about network security policies"}],
        reverse_context_calibrated=0.90,
        literal_guarded=0.90,
        nli_aggregate=0.65,
    )
    assert rescue is None


def test_lexical_rescue_rejects_strong_nli_contradiction() -> None:
    """NLI aggregate below 0.50 must never be overridden by lexical overlap."""
    rescue = _lexical_rescue_floor(
        "Les entites concernees doivent notifier la violation dans les 72 heures "
        "a l'autorite de controle designee.",
        [
            {
                "text": (
                    "En cas de violation, l'autorite de controle doit etre notifiee "
                    "dans les 72 heures par les entites concernees."
                )
            }
        ],
        reverse_context_calibrated=0.90,
        literal_guarded=0.88,
        nli_aggregate=0.25,
    )
    assert rescue is None


def test_lexical_rescue_rejects_injected_negation_en() -> None:
    """A response that inserts a negation where support has none must not be lifted."""
    rescue = _lexical_rescue_floor(
        "The directive does not require reporting incidents to the CSIRT within 24 hours.",
        [{"text": "The directive requires reporting incidents to the CSIRT within 24 hours."}],
        reverse_context_calibrated=0.90,
        literal_guarded=0.85,
        nli_aggregate=0.55,
    )
    assert rescue is None


def test_lexical_rescue_rejects_injected_negation_fr() -> None:
    rescue = _lexical_rescue_floor(
        "Les entites ne doivent pas notifier la violation dans les 72 heures.",
        [{"text": "Les entites doivent notifier la violation dans les 72 heures."}],
        reverse_context_calibrated=0.88,
        literal_guarded=0.85,
        nli_aggregate=0.55,
    )
    assert rescue is None


def test_lexical_rescue_rejects_injected_negation_de() -> None:
    rescue = _lexical_rescue_floor(
        "Die Einrichtung muss keine erheblichen Vorfaelle innerhalb von 24 Stunden melden.",
        [{"text": "Die Einrichtung muss erhebliche Vorfaelle innerhalb von 24 Stunden melden."}],
        reverse_context_calibrated=0.88,
        literal_guarded=0.85,
        nli_aggregate=0.55,
    )
    assert rescue is None


def test_lexical_rescue_allows_symmetric_negation() -> None:
    """A response that keeps the same negation cue as support is still rescued."""
    rescue = _lexical_rescue_floor(
        "The provider must not share credentials outside of the approved tunnel.",
        [{"text": "The provider must not share credentials outside of the approved tunnel."}],
        reverse_context_calibrated=0.92,
        literal_guarded=0.92,
        nli_aggregate=0.60,
    )
    assert rescue == pytest.approx(0.85)


# ----------------------------------------------------------------------
# rag_prose stratum
# ----------------------------------------------------------------------


def _load_policy(path: Path) -> RiskBandPolicy:
    return load_risk_band_policy(path=path)


def test_thresholds_have_rag_prose_stratum() -> None:
    root = Path(__file__).resolve().parent.parent / "latence_trace" / "data"
    for file_name in ("thresholds.balanced.json", "thresholds.quality.json"):
        payload = json.loads((root / file_name).read_text())
        assert "rag_prose" in payload["strata"], file_name
        assert payload["strata"]["rag_prose"]["green_min"] == pytest.approx(0.75)
        assert payload["strata"]["rag_prose"]["amber_min"] == pytest.approx(0.55)


def test_rag_prose_stratum_routes_0_76_to_green() -> None:
    root = Path(__file__).resolve().parent.parent / "latence_trace" / "data"
    policy = _load_policy(root / "thresholds.balanced.json")
    assert classify_risk_band(0.76, stratum="rag_prose", policy=policy) == "green"
    assert classify_risk_band(0.76, stratum="default", policy=policy) == "amber"


def test_resolve_effective_stratum_auto_routes_prose_to_rag_prose() -> None:
    assert (
        _resolve_effective_stratum(
            risk_band_stratum=None,
            structured_source_format=None,
            structured_verification=None,
        )
        == "rag_prose"
    )


def test_resolve_effective_stratum_keeps_default_for_tables() -> None:
    assert (
        _resolve_effective_stratum(
            risk_band_stratum=None,
            structured_source_format="json",
            structured_verification=None,
        )
        is None
    )
    assert (
        _resolve_effective_stratum(
            risk_band_stratum=None,
            structured_source_format="prose_table",
            structured_verification=None,
        )
        is None
    )


def test_resolve_effective_stratum_respects_explicit_caller_stratum() -> None:
    assert (
        _resolve_effective_stratum(
            risk_band_stratum="entity_swap",
            structured_source_format=None,
            structured_verification=None,
        )
        == "entity_swap"
    )


def test_resolve_effective_stratum_forced_on_returns_default() -> None:
    """`structured_verification="on"` keeps the hardest (default) threshold."""
    assert (
        _resolve_effective_stratum(
            risk_band_stratum=None,
            structured_source_format=None,
            structured_verification="on",
        )
        is None
    )
