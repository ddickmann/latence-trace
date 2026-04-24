"""Back-compat re-export of the lane-neutral attribution kernel.

The implementation lives at
:mod:`latence_trace.core.attribution.file_attribution` so the RAG lane
can reuse it. This module stays around so existing imports of the form
``from latence_trace.core.code_lane.file_attribution import ...`` keep
working without modification.
"""

from __future__ import annotations

from latence_trace.core.attribution.file_attribution import (
    FileAttributionResult,
    PerFileUsage,
    PerUnitOwnership,
    ReasonCode,
    attribute_files,
    build_per_unit_records,
    resolve_attribution_key,
)

__all__ = [
    "FileAttributionResult",
    "PerFileUsage",
    "PerUnitOwnership",
    "ReasonCode",
    "attribute_files",
    "build_per_unit_records",
    "resolve_attribution_key",
]
