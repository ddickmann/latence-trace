"""Shared dataclasses for the code lane.

Keeping the shared types in one small module avoids circular imports
between the scorer, attribution, and orchestrator layers. Everything
here is frozen or immutable-by-convention so the code lane stays
thread-safe when the singletons serve concurrent requests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence, Tuple

import torch


@dataclass(frozen=True)
class SupportUnitPack:
    """Compact, hashable bundle of a single support unit.

    Matches the subset of :class:`latence_trace.core.groundedness.SupportUnitInput`
    the code-lane hot path actually reads. By not carrying the full
    groundedness struct we keep object churn low in the scorer loop and
    the orchestrator free to reuse the upstream type without importing
    the heavy core module.

    Attributes
    ----------
    support_id:
        Stable identifier echoed back in the response (matches the
        ``support_id`` on the corresponding ``GroundednessSupportUnit``).
    path:
        Best-effort origin file path for per-file attribution. ``None``
        when the unit comes from an anonymous ``raw_context`` blob with
        no path header.
    tokens:
        Tokenised premise used for literal-guard matching. Lower-cased
        downstream.
    embeddings:
        L2-normalised ColBERT token matrix ``(L_u, d)``.
    offset_start, offset_end:
        Character offsets into the rendered context; used to expose
        chunk-level ownership to callers that want to drop sub-file
        regions.
    metadata:
        Echo of caller-supplied metadata; unused by the scorer.
    """

    support_id: str
    path: Optional[str]
    tokens: Tuple[str, ...]
    embeddings: torch.Tensor
    offset_start: Optional[int] = None
    offset_end: Optional[int] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
