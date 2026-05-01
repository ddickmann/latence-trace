from __future__ import annotations

from types import SimpleNamespace

from latence_trace.core.groundedness import _lexical_rescue_floor


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
