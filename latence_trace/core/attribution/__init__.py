"""Lane-neutral file attribution primitives.

Both the code-lane orchestrator and the RAG scorer project their
per-unit argmax evidence through :func:`attribute_files` to produce
per-file dead-weight rollups with reason codes and a small histogram.

The module was lifted out of ``latence_trace.core.code_lane`` so the
RAG lane could share the same contract; the old import path still
works as a thin re-export to keep downstream callers un-broken.
"""

from __future__ import annotations

from .file_attribution import (
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
