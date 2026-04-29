"""Deterministic paraphrase variants for synthetic responses.

Three variants are emitted per ``(passage, response, shape)`` pair,
labelled ``v0``, ``v1``, ``v2``:

* ``v0`` — the unmodified response (baseline).
* ``v1`` — lead-with-metric: rotate key numeric or named-entity tokens
  to the start of the response; swap a handful of stock connectors.
* ``v2`` — tighten-and-reorder: apply a small set of deterministic
  lexical substitutions ("year over year" -> "YoY", "respectively"
  -> "each", etc.), collapse redundant spacing, and rotate the
  sentence order of multi-sentence responses by one.

These are cheap surface rewrites, not semantic rewrites. The goal is
to inject *lexical* variation (so the biaffine MaxSim head cannot
memorise surface forms) while keeping the *factual content* identical
— which is exactly what the training loop needs, because the three-
axis labels are tied to the content, not the phrasing. No LLM is
invoked, so the paraphrase step is hermetic and reproducible.

Numeric values are never touched (so that adversarial ``numeric_flip``
is the only source of numeric-flip signal in the dataset).
"""

from __future__ import annotations

import random
import re


# Stock connector swaps applied deterministically in v2.
_LEXICAL_SUBS_V2: tuple[tuple[str, str], ...] = (
    ("year over year", "YoY"),
    ("year-over-year", "YoY"),
    ("respectively", "each"),
    ("approximately", "~"),
    ("continued to", "kept on"),
    ("according to", "per"),
    ("with a ", "at a "),
    ("which was", "— the"),
    ("and also", "and"),
    ("in addition,", "additionally,"),
)


# Leading-connector swaps for v1.
_LEAD_CONNECTORS_V1: tuple[tuple[str, str], ...] = (
    ("In ", "During "),
    ("The ", "A "),
    ("Under ", "Per "),
    ("During ", "In "),
    ("Across ", "Over "),
)


def paraphrase_v0(response: str, *, seed: int) -> str:
    return response


def paraphrase_v1(response: str, *, seed: int) -> str:
    """Lead-with-phrase swap: rotate a leading connector.

    Safe no-op when the response does not start with a known
    connector. Deterministic given ``seed``.
    """
    rng = random.Random(seed ^ 0xA11C)
    out = response
    for src, dst in _LEAD_CONNECTORS_V1:
        if out.startswith(src):
            out = dst + out[len(src):]
            break
    # Replace one "and" with a semicolon if more than one exists, to
    # inject modest lexical variation. Pick deterministically.
    ands = [m.start() for m in re.finditer(r" and ", out)]
    if len(ands) >= 2:
        pick = ands[rng.randrange(len(ands))]
        out = out[:pick] + "; " + out[pick + 5 :]
    return out


def paraphrase_v2(response: str, *, seed: int) -> str:
    """Tighten-and-reorder: deterministic lexical subs + sentence rotation."""
    rng = random.Random(seed ^ 0xB22D)
    out = response
    for src, dst in _LEXICAL_SUBS_V2:
        out = out.replace(src, dst)
    # Collapse double spaces introduced by substitutions.
    out = re.sub(r"\s+", " ", out).strip()
    # If the response has > 1 sentence, rotate the first two.
    parts = re.split(r"(?<=[.!?])\s+", out.strip())
    parts = [p for p in parts if p]
    if len(parts) >= 2 and rng.random() < 0.8:
        parts[0], parts[1] = parts[1], parts[0]
        out = " ".join(parts)
    return out


PARAPHRASE_VARIANTS: tuple[str, ...] = ("v0", "v1", "v2")

_VARIANTS = {
    "v0": paraphrase_v0,
    "v1": paraphrase_v1,
    "v2": paraphrase_v2,
}


def paraphrase(variant: str, response: str, *, seed: int) -> str:
    if variant not in _VARIANTS:
        raise KeyError(
            f"unknown paraphrase variant {variant!r}; "
            f"valid: {PARAPHRASE_VARIANTS}"
        )
    return _VARIANTS[variant](response, seed=seed)


__all__ = [
    "PARAPHRASE_VARIANTS",
    "paraphrase",
    "paraphrase_v0",
    "paraphrase_v1",
    "paraphrase_v2",
]
