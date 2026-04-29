"""Quality gates for synthetic-lane rows.

Two gates are enforced at dataset-build time before a row enters the
training manifest:

* **Jaccard overlap**: the grounded response and its adversarial
  counterpart must share at least ``min_jaccard`` (default 0.75) of
  their token set, so adversarials stay on-topic and do not degenerate
  into off-topic distractors.
* **Teacher verdict**: the teacher must rate grounded responses as
  ``green`` and adversarial responses as ``red`` or ``amber`` for the
  row to be kept. Otherwise the pair is logged to
  ``ambiguous.jsonl`` and excluded from training (but kept for hard-
  case eval).

Neither gate is enforced here directly — this module exposes the
primitives. The orchestration (and the teacher call) lives in
``distill_dataset.py`` so this file remains hermetic and
unit-testable without a live teacher.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenise(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text or "")}


def jaccard(a: str, b: str) -> float:
    """Classic Jaccard on case-folded alnum tokens.

    Returns 0.0 if either side is empty to match set-theoretic convention.
    """
    ta, tb = _tokenise(a), _tokenise(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    if union == 0:
        return 0.0
    return inter / union


@dataclass(frozen=True)
class JaccardGateResult:
    kept: bool
    score: float
    min_required: float


def jaccard_gate(
    grounded_response: str,
    adversarial_response: str,
    *,
    min_jaccard: float = 0.75,
) -> JaccardGateResult:
    """Enforce the 0.75 floor on grounded-vs-adversarial overlap."""
    score = jaccard(grounded_response, adversarial_response)
    return JaccardGateResult(
        kept=score >= min_jaccard,
        score=score,
        min_required=min_jaccard,
    )


def teacher_verdict_consistent(
    role: str,  # "grounded" | "entity_swap" | "numeric_flip" | "support_drop"
    teacher_band: str,  # "green" | "amber" | "red" | ...
) -> bool:
    """Return True iff the teacher's band matches the intended label.

    * ``grounded`` rows should score green.
    * Adversarial rows (any non-grounded role) should score red or
      amber (both count as "flagged as ungrounded").
    """
    band = (teacher_band or "").lower().strip()
    if role == "grounded":
        return band == "green"
    return band in {"red", "amber"}


__all__ = [
    "JaccardGateResult",
    "jaccard",
    "jaccard_gate",
    "teacher_verdict_consistent",
]
