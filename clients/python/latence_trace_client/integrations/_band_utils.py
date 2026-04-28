"""Shared helpers for framework integration adapters.

The wire has two slightly different flat response shapes:

1. RunPod handler (``_compact_response``) emits a top-level ``band``
   key and ``groundedness`` scalar.
2. Public REST ``/groundedness`` returns a pydantic
   ``GroundednessResponse`` with ``risk_band`` and
   ``scores.groundedness_v2``.

The Python client model accepts extra fields (``extra="allow"``) so
the RunPod shape survives deserialisation - but older callers that
only set ``risk_band`` would leave ``response.band`` unset.  These
helpers let every adapter resolve band / score uniformly.
"""

from __future__ import annotations

from typing import Any


def resolve_band(response: Any) -> str:
    """Return the lowercase band for ``response`` from any shape."""

    band = getattr(response, "band", None)
    if band:
        return str(band).lower()
    risk_band = getattr(response, "risk_band", None)
    if risk_band is None:
        return "unknown"
    value = getattr(risk_band, "value", None)
    if value:
        return str(value).lower()
    return str(risk_band).lower()


def resolve_score(response: Any) -> float:
    """Return the headline groundedness scalar from any shape."""

    direct = getattr(response, "groundedness", None)
    if direct is not None:
        try:
            return float(direct)
        except (TypeError, ValueError):
            pass
    scores = getattr(response, "scores", None)
    for name in ("groundedness_v2", "groundedness_v1"):
        val = getattr(scores, name, None) if scores is not None else None
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                continue
    return 0.0
