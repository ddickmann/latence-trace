"""Unit tests for the epistemic-hedge gate in ``latence_trace.core.groundedness``.

The gate is a pure function that takes a fused ``groundedness_v2`` score plus
per-claim NLI records and returns a cap/floor diagnostic. These tests exercise
the decision matrix directly, so they do not need embeddings or an NLI model.

Covered scenarios (per archetype archetype proxies — finance, legal, cyber —
each with three cases):

* amber-leaning green: supported + hedged claim, score would have been green
  without the gate. Gate caps to just below ``green_min``.
* amber-leaning red: supported + hedged claim, score would have been red
  without the gate. Gate floors to ``amber_min``.
* legitimate red with contradiction: hedged-looking text but one claim has
  strong contradiction. Gate must NOT rescue (``applied=False``/None).
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from latence_trace.core.groundedness import (
    EPISTEMIC_HEDGE_CUES,
    _epistemic_hedge_gate,
)
from latence_trace.core.thresholds import RiskBandPolicy


def _policy() -> RiskBandPolicy:
    return RiskBandPolicy(
        strata={
            "default": {"green_min": 0.80, "amber_min": 0.60},
            "rag_prose": {"green_min": 0.75, "amber_min": 0.55},
        }
    )


def _claim(
    *,
    index: int,
    text: str,
    entailment: float,
    contradiction: float = 0.05,
    skipped: bool = False,
) -> Dict[str, Any]:
    return {
        "index": index,
        "text": text,
        "entailment": entailment,
        "contradiction": contradiction,
        "neutral": max(0.0, 1.0 - entailment - contradiction),
        "score": entailment,
        "skipped": skipped,
    }


def _mixed_records(
    *,
    supported_text: str,
    hedge_text: str,
    contradiction: float = 0.05,
) -> List[Dict[str, Any]]:
    return [
        _claim(index=0, text=supported_text, entailment=0.82, contradiction=0.02),
        _claim(index=1, text=hedge_text, entailment=0.15, contradiction=contradiction),
    ]


# ---------------------------------------------------------------------------
# Finance archetype proxy
# ---------------------------------------------------------------------------


def test_finance_amber_leaning_green_is_capped() -> None:
    records = _mixed_records(
        supported_text=(
            "The transfer pricing master file describes intra-group services,"
            " licensing and contract manufacturing."
        ),
        hedge_text=(
            "The documentation does not show full per-exercise coverage and"
            " the scope remains unclear."
        ),
    )
    result = _epistemic_hedge_gate(
        groundedness_v2=0.92,
        claim_records=records,
        effective_stratum="rag_prose",
        risk_band_policy=_policy(),
    )
    assert result is not None
    assert result["fired"] is True
    assert result["applied"] is True
    assert result["claim_indices"] == [1]
    assert result["cap"] == pytest.approx(0.55 + 0.20 * 0.95)
    assert result["floor"] == pytest.approx(0.55)
    assert result["adjusted_score"] == pytest.approx(0.55 + 0.20 * 0.95)
    assert result["original_score"] == pytest.approx(0.92)


def test_finance_amber_leaning_red_is_floored() -> None:
    records = _mixed_records(
        supported_text=(
            "The master file confirms intra-group services, licensing and"
            " contract manufacturing activities."
        ),
        hedge_text=(
            "Confirmation of the full filing history ne permet pas"
            " d'établir la couverture par exercice."
        ),
    )
    result = _epistemic_hedge_gate(
        groundedness_v2=0.42,
        claim_records=records,
        effective_stratum="rag_prose",
        risk_band_policy=_policy(),
    )
    assert result is not None
    assert result["applied"] is True
    assert result["adjusted_score"] == pytest.approx(0.55)
    assert result["original_score"] == pytest.approx(0.42)


def test_finance_legitimate_red_with_contradiction_not_rescued() -> None:
    records = [
        _claim(
            index=0,
            text=(
                "The master file confirms intra-group services and contract"
                " manufacturing activities."
            ),
            entailment=0.80,
            contradiction=0.03,
        ),
        _claim(
            index=1,
            text=(
                "The royalty rate is 9% and ten complete fiscal years have"
                " been filed, which is unclear to validate."
            ),
            entailment=0.12,
            contradiction=0.78,
        ),
    ]
    result = _epistemic_hedge_gate(
        groundedness_v2=0.35,
        claim_records=records,
        effective_stratum="rag_prose",
        risk_band_policy=_policy(),
    )
    assert result is None, (
        "A claim with strong contradiction must not be treated as an"
        " epistemic hedge and the gate must refuse to floor the score."
    )


# ---------------------------------------------------------------------------
# Legal archetype proxy
# ---------------------------------------------------------------------------


def test_legal_amber_leaning_green_is_capped() -> None:
    records = _mixed_records(
        supported_text=(
            "The engagement letter sets a 30-day notice period before"
            " termination of the mandate."
        ),
        hedge_text=(
            "The applicability of clause 7.2 to mid-contract modifications"
            " is not established from the evidence."
        ),
    )
    result = _epistemic_hedge_gate(
        groundedness_v2=0.88,
        claim_records=records,
        effective_stratum="rag_prose",
        risk_band_policy=_policy(),
    )
    assert result is not None
    assert result["applied"] is True
    assert result["adjusted_score"] < 0.75


def test_legal_amber_leaning_red_is_floored() -> None:
    records = _mixed_records(
        supported_text=(
            "The engagement letter sets a 30-day notice period and a fixed"
            " monthly retainer."
        ),
        hedge_text=(
            "The scope of clause 7.2 remains incertain and cannot be"
            " confirmed from the supplied pack."
        ),
    )
    result = _epistemic_hedge_gate(
        groundedness_v2=0.48,
        claim_records=records,
        effective_stratum="rag_prose",
        risk_band_policy=_policy(),
    )
    assert result is not None
    assert result["applied"] is True
    assert result["adjusted_score"] == pytest.approx(0.55)


def test_legal_legitimate_red_with_contradiction_not_rescued() -> None:
    records = [
        _claim(
            index=0,
            text=(
                "The engagement letter sets a 30-day notice period before"
                " termination."
            ),
            entailment=0.78,
            contradiction=0.02,
        ),
        _claim(
            index=1,
            text=(
                "The notice period is actually 180 days; the applicable"
                " scope is unclear."
            ),
            entailment=0.10,
            contradiction=0.72,
        ),
    ]
    result = _epistemic_hedge_gate(
        groundedness_v2=0.28,
        claim_records=records,
        effective_stratum="rag_prose",
        risk_band_policy=_policy(),
    )
    assert result is None


# ---------------------------------------------------------------------------
# Cyber archetype proxy
# ---------------------------------------------------------------------------


def test_cyber_amber_leaning_green_is_capped() -> None:
    records = _mixed_records(
        supported_text=(
            "The incident response runbook mandates a 24-hour CISO"
            " notification after detection."
        ),
        hedge_text=(
            "Whether the same SLA applies to subcontractor-originated"
            " incidents is nicht klar from the material."
        ),
    )
    result = _epistemic_hedge_gate(
        groundedness_v2=0.90,
        claim_records=records,
        effective_stratum="rag_prose",
        risk_band_policy=_policy(),
    )
    assert result is not None
    assert result["applied"] is True
    assert result["adjusted_score"] < 0.75


def test_cyber_amber_leaning_red_is_floored() -> None:
    records = _mixed_records(
        supported_text=(
            "The runbook describes a 24-hour CISO notification workflow"
            " after detection."
        ),
        hedge_text=(
            "The SLA for subcontractor-originated incidents is nicht belegt"
            " in the supplied policy bundle."
        ),
    )
    result = _epistemic_hedge_gate(
        groundedness_v2=0.40,
        claim_records=records,
        effective_stratum="rag_prose",
        risk_band_policy=_policy(),
    )
    assert result is not None
    assert result["applied"] is True
    assert result["adjusted_score"] == pytest.approx(0.55)


def test_cyber_legitimate_red_with_contradiction_not_rescued() -> None:
    records = [
        _claim(
            index=0,
            text=(
                "The runbook mandates a 24-hour CISO notification after"
                " incident detection."
            ),
            entailment=0.78,
            contradiction=0.03,
        ),
        _claim(
            index=1,
            text=(
                "The actual SLA is 72 hours, though the precise window is"
                " unclear."
            ),
            entailment=0.10,
            contradiction=0.68,
        ),
    ]
    result = _epistemic_hedge_gate(
        groundedness_v2=0.30,
        claim_records=records,
        effective_stratum="rag_prose",
        risk_band_policy=_policy(),
    )
    assert result is None


# ---------------------------------------------------------------------------
# Guardrails: wrong stratum, missing support, skipped records, already in band
# ---------------------------------------------------------------------------


def test_gate_never_fires_outside_rag_prose_stratum() -> None:
    records = _mixed_records(
        supported_text="The master file confirms intra-group services.",
        hedge_text="The coverage is unclear for older fiscal years.",
    )
    for stratum in ("default", "number_swap", None, "quality"):
        result = _epistemic_hedge_gate(
            groundedness_v2=0.91,
            claim_records=records,
            effective_stratum=stratum,
            risk_band_policy=_policy(),
        )
        assert result is None, f"Gate must not fire on stratum={stratum!r}"


def test_gate_requires_supported_and_hedged_claim() -> None:
    only_hedge = [
        _claim(
            index=0,
            text="The coverage is unclear and not established.",
            entailment=0.15,
            contradiction=0.05,
        )
    ]
    result = _epistemic_hedge_gate(
        groundedness_v2=0.85,
        claim_records=only_hedge,
        effective_stratum="rag_prose",
        risk_band_policy=_policy(),
    )
    assert result is None

    only_supported = [
        _claim(
            index=0,
            text="The master file confirms intra-group services.",
            entailment=0.86,
        )
    ]
    assert (
        _epistemic_hedge_gate(
            groundedness_v2=0.85,
            claim_records=only_supported,
            effective_stratum="rag_prose",
            risk_band_policy=_policy(),
        )
        is None
    )


def test_gate_is_noop_when_score_already_in_amber() -> None:
    records = _mixed_records(
        supported_text="The master file confirms intra-group services.",
        hedge_text="Coverage by exercise is unclear from the pack.",
    )
    result = _epistemic_hedge_gate(
        groundedness_v2=0.65,
        claim_records=records,
        effective_stratum="rag_prose",
        risk_band_policy=_policy(),
    )
    assert result is not None
    assert result["fired"] is True
    assert result["applied"] is False
    assert result["adjusted_score"] == pytest.approx(0.65)


def test_gate_ignores_skipped_claims() -> None:
    records = [
        _claim(
            index=0,
            text="The master file confirms intra-group services.",
            entailment=0.85,
        ),
        _claim(
            index=1,
            text="Coverage is unclear across fiscal years.",
            entailment=0.1,
            contradiction=0.05,
            skipped=True,
        ),
    ]
    result = _epistemic_hedge_gate(
        groundedness_v2=0.90,
        claim_records=records,
        effective_stratum="rag_prose",
        risk_band_policy=_policy(),
    )
    assert result is None


def test_cue_list_covers_common_multilingual_hedges() -> None:
    must_contain = {
        "unclear",
        "not established",
        "ne permet pas",
        "nicht klar",
        "nicht belegt",
    }
    assert must_contain.issubset(set(EPISTEMIC_HEDGE_CUES))
