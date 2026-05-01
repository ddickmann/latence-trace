from __future__ import annotations

from types import SimpleNamespace

from latence_trace.core.groundedness import (
    _business_policy_rescue_floor,
    _lexical_rescue_floor,
    diff_literals,
    extract_literals,
)


def test_lexical_rescue_accepts_mild_multilingual_nli_underconfidence() -> None:
    floor = _lexical_rescue_floor(
        "Die Drohne darf starten wenn der Akku mindestens 80 Prozent hat und der Pilot die Checkliste bestaetigt.",
        [
            SimpleNamespace(
                text=(
                    "Eine Inspektionsdrohne darf starten, wenn der Akkustand mindestens "
                    "80 Prozent betraegt und der Pilot die Checkliste digital bestaetigt."
                )
            )
        ],
        reverse_context_calibrated=0.96,
        literal_guarded=0.97,
        nli_aggregate=0.47,
    )

    assert floor == 0.85


def test_lexical_rescue_still_refuses_strong_nli_contradiction() -> None:
    floor = _lexical_rescue_floor(
        "Die Drohne darf nicht starten wenn der Akku mindestens 80 Prozent hat.",
        [SimpleNamespace(text="Die Drohne darf starten, wenn der Akku mindestens 80 Prozent hat.")],
        reverse_context_calibrated=0.96,
        literal_guarded=0.97,
        nli_aggregate=0.44,
    )

    assert floor is None


def test_numeric_literals_normalize_space_grouping_and_currency_variants() -> None:
    support = [SimpleNamespace(text="The finance approval limit is EUR 120 000.")]
    response_literals, mismatches, matches = diff_literals(
        "The finance approval limit is EUR 120,000.",
        support,
    )

    assert response_literals
    assert matches
    assert mismatches == []
    assert extract_literals("The limit is 120 000 EUR.")[0]["normalized"] == "\u20ac120000"


def test_query_scoped_business_numeric_literals_do_not_trigger_mismatch() -> None:
    support = [
        SimpleNamespace(
            text=(
                "Vendor payments above 100000 EUR require approval from the finance "
                "director and treasury operations."
            )
        )
    ]

    _, mismatches, matches = diff_literals(
        "A 120000 EUR vendor payment requires finance director and treasury approval.",
        support,
        query_text="Can we approve a 120000 EUR vendor payment?",
    )

    assert matches
    assert mismatches == []


def test_lexical_rescue_accepts_only_clause_negated_paraphrase() -> None:
    floor = _lexical_rescue_floor(
        "The supplier cannot terminate for convenience.",
        [
            SimpleNamespace(
                text=(
                    "The supplier may terminate only for material breach. "
                    "The customer may terminate for convenience."
                )
            )
        ],
        reverse_context_calibrated=0.94,
        literal_guarded=0.94,
        nli_aggregate=0.52,
    )

    assert floor == 0.85


def test_business_policy_rescue_sets_floor_for_exclusive_clause() -> None:
    floor = _business_policy_rescue_floor(
        "The supplier cannot terminate for convenience.",
        [
            SimpleNamespace(
                text=(
                    "The supplier may terminate only for material breach. "
                    "The customer may terminate for convenience."
                )
            )
        ],
        query_text="Can the supplier terminate for convenience?",
        literal_mismatches=[],
    )

    assert floor == 0.85


def test_lexical_rescue_refuses_negating_allowed_only_clause_reason() -> None:
    floor = _lexical_rescue_floor(
        "The supplier cannot terminate for material breach.",
        [SimpleNamespace(text="The supplier may terminate only for material breach.")],
        reverse_context_calibrated=0.94,
        literal_guarded=0.94,
        nli_aggregate=0.52,
    )

    assert floor is None


def test_business_policy_rescue_sets_floor_for_supported_threshold_case() -> None:
    floor = _business_policy_rescue_floor(
        "A 120000 EUR vendor payment requires approval from the finance director and treasury operations.",
        [
            SimpleNamespace(
                text=(
                    "Corporate treasury policy: vendor payments above 100000 EUR require "
                    "approval from the finance director and treasury operations. Payments "
                    "up to 100000 EUR require only cost-center owner approval."
                )
            )
        ],
        query_text="Who must approve a 120000 EUR vendor payment?",
        literal_mismatches=[],
    )

    assert floor == 0.85


def test_business_policy_rescue_refuses_unmatched_literals() -> None:
    floor = _business_policy_rescue_floor(
        "A 250000 EUR vendor payment requires approval from the finance director.",
        [
            SimpleNamespace(
                text="Vendor payments above 100000 EUR require finance director approval."
            )
        ],
        query_text="Who must approve a 120000 EUR vendor payment?",
        literal_mismatches=[{"kind": "currency", "value": "250000 EUR"}],
    )

    assert floor is None
