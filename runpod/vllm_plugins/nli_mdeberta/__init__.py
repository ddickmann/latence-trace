"""latence-trace NLI BYOP plugin."""

from forge.registration import register_plugin

from .config import NLIDebertaV2Config
from .model import NLIDebertaV2Model


def register() -> None:
    register_plugin(
        "nli_mdeberta",
        NLIDebertaV2Config,
        "NLIDebertaV2Model",
        NLIDebertaV2Model,
        aliases=["DebertaV2ForSequenceClassification"],
    )


register()

__all__ = ["NLIDebertaV2Config", "NLIDebertaV2Model", "register"]
