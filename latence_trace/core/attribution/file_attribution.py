"""Per-file / per-unit attribution with reason codes and query awareness.

Originally promoted from
``research/triangular_maxsim/coding/experiments/file_attribution.py``
into the code lane, this module now sits under
:mod:`latence_trace.core.attribution` so both lanes can share the
same kernel. It is intentionally pure-Python; the heavy lifting
already happened in the GPU scorer (for the code lane) or in the
MaxSim / usage classifier (for the RAG lane).

What it produces
----------------

- **Per-file rollup** (``PerFileUsage``) with coverage, ownership,
  query-aware ownership, dead-weight bit, and reason codes.
- **Per-unit export** (``PerUnitOwnership``) so IDE plugins can drop
  sub-file regions at the 200k-token wall.
- **Reason-code histogram** across files so callers can say
  "53 % of the waste is ``dominated_by_single_file``" without
  rolling their own aggregation.

The module is thread-safe and has no I/O or global state.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence


class ReasonCode(str, Enum):
    """Why a file was flagged as dead weight this turn.

    Multiple reasons can apply to one file — the attribution layer
    emits the full sorted list so operators can tune thresholds.
    """

    NEVER_WON_ARGMAX = "never_won_argmax"
    ALL_TOKENS_BELOW_0_40 = "all_tokens_below_0_40"
    DOMINATED_BY_SINGLE_FILE = "dominated_by_single_file"
    QUERY_RELEVANT_BUT_IGNORED = "query_relevant_but_ignored"
    NOT_QUERY_RELEVANT = "not_query_relevant"


@dataclass
class PerUnitOwnership:
    """Chunk-level ownership record.

    One per support unit so callers with very large files (>200 kB)
    can sub-divide and keep only the high-ownership slices.
    """

    support_id: str
    path: Optional[str]
    unit_index: int
    max_cos: float
    response_owner_count: int
    query_owner_count: int
    response_owner_share: float
    query_owner_share: float
    offset_start: Optional[int]
    offset_end: Optional[int]
    usage_state: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "support_id": self.support_id,
            "path": self.path,
            "unit_index": int(self.unit_index),
            "max_cos": float(self.max_cos),
            "response_owner_count": int(self.response_owner_count),
            "query_owner_count": int(self.query_owner_count),
            "response_owner_share": float(self.response_owner_share),
            "query_owner_share": float(self.query_owner_share),
            "offset_start": self.offset_start,
            "offset_end": self.offset_end,
            "usage_state": self.usage_state,
        }


@dataclass
class PerFileUsage:
    """Per-file rollup with reason codes."""

    path: str
    n_units: int
    used: int
    uncertain: int
    unused: int
    coverage: float
    mean_score: float
    max_evidence: float
    owner_tokens: int
    owner_share: float
    query_owner_tokens: int
    query_owner_share: float
    dead_weight: bool
    reason_codes: List[ReasonCode] = field(default_factory=list)
    dominating_peer: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "n_units": int(self.n_units),
            "used": int(self.used),
            "uncertain": int(self.uncertain),
            "unused": int(self.unused),
            "coverage": float(self.coverage),
            "mean_score": float(self.mean_score),
            "max_evidence": float(self.max_evidence),
            "owner_tokens": int(self.owner_tokens),
            "owner_share": float(self.owner_share),
            "query_owner_tokens": int(self.query_owner_tokens),
            "query_owner_share": float(self.query_owner_share),
            "dead_weight": bool(self.dead_weight),
            "reason_codes": [rc.value for rc in self.reason_codes],
            "dominating_peer": self.dominating_peer,
        }


@dataclass
class FileAttributionResult:
    """Full attribution bundle for one turn."""

    per_file: List[PerFileUsage]
    per_unit: List[PerUnitOwnership]
    dead_weight_files: List[str]
    dead_weight_ratio: float
    n_files: int
    n_response_tokens: int
    n_query_tokens: int
    min_owner_share: float
    coverage_threshold: float
    low_cosine_threshold: float
    reason_code_histogram: Dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "per_file": [p.as_dict() for p in self.per_file],
            "per_unit": [u.as_dict() for u in self.per_unit],
            "dead_weight_files": list(self.dead_weight_files),
            "dead_weight_ratio": float(self.dead_weight_ratio),
            "n_files": int(self.n_files),
            "n_response_tokens": int(self.n_response_tokens),
            "n_query_tokens": int(self.n_query_tokens),
            "min_owner_share": float(self.min_owner_share),
            "coverage_threshold": float(self.coverage_threshold),
            "low_cosine_threshold": float(self.low_cosine_threshold),
            "reason_code_histogram": dict(self.reason_code_histogram),
        }


# ---------------------------------------------------------------------------
# Lane-neutral helpers
# ---------------------------------------------------------------------------


def resolve_attribution_key(unit: Any) -> Optional[str]:
    """Pick the best grouping key for a support unit.

    Preference order, tight by design so both lanes agree:

    1. Explicit ``path`` attribute on the unit (code-lane
       :class:`SupportUnitPack`).
    2. ``metadata.path`` or ``metadata.source`` (enterprise RAG stacks
       that label chunks by filesystem path or source document).
    3. ``metadata.source_id`` (opaque document IDs).
    4. ``support_id`` / ``chunk_id`` (safe fallback).

    Returns ``None`` only when the unit has literally no identifier —
    the caller should treat that as a pipeline bug.
    """
    path = getattr(unit, "path", None)
    if path:
        return str(path)
    metadata: Optional[Mapping[str, Any]] = getattr(unit, "metadata", None)
    if metadata and isinstance(metadata, Mapping):
        for key in ("path", "source", "source_id"):
            value = metadata.get(key)
            if value:
                return str(value)
    support_id = getattr(unit, "support_id", None) or getattr(unit, "chunk_id", None)
    return str(support_id) if support_id else None


def build_per_unit_records(
    *,
    units: Sequence[Any],
    per_unit_max: Sequence[float],
    per_unit_owner_count: Sequence[int],
    per_unit_query_owner_count: Sequence[int],
    usage_states: Sequence[str],
    resp_denom: int,
    query_denom: int,
) -> List[PerUnitOwnership]:
    """Project scorer outputs into a list of :class:`PerUnitOwnership`.

    Lane-neutral: works as long as the units expose ``support_id``
    and optionally ``path`` / ``metadata`` / ``offset_start`` /
    ``offset_end``. Both code-lane ``SupportUnitPack`` and the RAG
    ``SupportUnitInput`` satisfy the contract.
    """
    records: List[PerUnitOwnership] = []
    for idx, unit in enumerate(units):
        support_id = str(getattr(unit, "support_id", f"support-{idx}"))
        path = resolve_attribution_key(unit)
        max_cos = float(per_unit_max[idx]) if idx < len(per_unit_max) else 0.0
        rc = int(per_unit_owner_count[idx]) if idx < len(per_unit_owner_count) else 0
        qc = (
            int(per_unit_query_owner_count[idx])
            if idx < len(per_unit_query_owner_count)
            else 0
        )
        usage = (
            str(usage_states[idx])
            if idx < len(usage_states) and usage_states[idx] is not None
            else "uncertain"
        )
        records.append(
            PerUnitOwnership(
                support_id=support_id,
                path=path,
                unit_index=idx,
                max_cos=max_cos,
                response_owner_count=rc,
                query_owner_count=qc,
                response_owner_share=float(rc) / resp_denom,
                query_owner_share=float(qc) / query_denom,
                offset_start=getattr(unit, "offset_start", None),
                offset_end=getattr(unit, "offset_end", None),
                usage_state=usage,
            )
        )
    return records


# ---------------------------------------------------------------------------
# Core rollup
# ---------------------------------------------------------------------------


def attribute_files(
    *,
    units: Sequence[Any],
    per_unit_max: Sequence[float],
    per_unit_owner_count: Sequence[int],
    per_unit_query_owner_count: Sequence[int],
    usage_states: Sequence[str],
    n_response_tokens: int,
    n_query_tokens: int,
    min_owner_share: float = 0.01,
    coverage_threshold: float = 0.20,
    low_cosine_threshold: float = 0.40,
) -> FileAttributionResult:
    """Roll up per-unit scorer stats to files and emit reason codes.

    Parameters
    ----------
    units:
        Sequence in scorer order. Each element must expose
        ``support_id``, optionally ``path``, ``offset_start``,
        ``offset_end``, and ``metadata``. Both the code-lane
        :class:`SupportUnitPack` and the RAG
        :class:`SupportUnitInput` satisfy the contract.
    per_unit_max, per_unit_owner_count, per_unit_query_owner_count,
    usage_states:
        Parallel sequences from the GPU scorer (code lane) or the
        tri-state usage classifier (RAG).
    """
    if len(units) != len(per_unit_max):
        raise ValueError(
            "units and per_unit_max length mismatch: "
            f"{len(units)} vs {len(per_unit_max)}"
        )

    resp_denom = max(1, int(n_response_tokens) or sum(per_unit_owner_count) or 1)
    query_denom = max(1, int(n_query_tokens) or sum(per_unit_query_owner_count) or 1)

    per_unit_records = build_per_unit_records(
        units=units,
        per_unit_max=per_unit_max,
        per_unit_owner_count=per_unit_owner_count,
        per_unit_query_owner_count=per_unit_query_owner_count,
        usage_states=usage_states,
        resp_denom=resp_denom,
        query_denom=query_denom,
    )

    # -- Per-file rollup ------------------------------------------------
    buckets: Dict[str, Dict[str, Any]] = {}
    for rec in per_unit_records:
        path = rec.path or rec.support_id
        bucket = buckets.setdefault(
            path,
            {
                "n_units": 0,
                "used": 0,
                "uncertain": 0,
                "unused": 0,
                "sum_score": 0.0,
                "max_evidence": 0.0,
                "owner_tokens": 0,
                "query_owner_tokens": 0,
                "all_max_cos": [],
                "unit_indices": [],
            },
        )
        bucket["n_units"] += 1
        state = rec.usage_state.lower()
        if state == "used":
            bucket["used"] += 1
        elif state == "uncertain":
            bucket["uncertain"] += 1
        else:
            bucket["unused"] += 1
        bucket["sum_score"] += rec.max_cos
        if rec.max_cos > bucket["max_evidence"]:
            bucket["max_evidence"] = rec.max_cos
        bucket["owner_tokens"] += rec.response_owner_count
        bucket["query_owner_tokens"] += rec.query_owner_count
        bucket["all_max_cos"].append(rec.max_cos)
        bucket["unit_indices"].append(rec.unit_index)

    # Pre-compute: which file *does* own most response tokens so we can
    # mark ``dominated_by_single_file`` on peer units.
    top_owner_path: Optional[str] = None
    top_owner_tokens = -1
    for path, b in buckets.items():
        if b["owner_tokens"] > top_owner_tokens:
            top_owner_tokens = int(b["owner_tokens"])
            top_owner_path = path

    per_file_records: List[PerFileUsage] = []
    dead_weight_files: List[str] = []
    for path, b in buckets.items():
        n = max(1, int(b["n_units"]))
        coverage = float(b["used"]) / n
        owner_share = float(b["owner_tokens"]) / resp_denom
        query_share = float(b["query_owner_tokens"]) / query_denom

        reasons: List[ReasonCode] = []
        if b["owner_tokens"] == 0:
            reasons.append(ReasonCode.NEVER_WON_ARGMAX)
        if b["all_max_cos"] and max(b["all_max_cos"]) < low_cosine_threshold:
            reasons.append(ReasonCode.ALL_TOKENS_BELOW_0_40)
        dominating_peer: Optional[str] = None
        if (
            top_owner_path
            and path != top_owner_path
            and owner_share < min_owner_share
            and top_owner_tokens >= resp_denom // 2
        ):
            reasons.append(ReasonCode.DOMINATED_BY_SINGLE_FILE)
            dominating_peer = top_owner_path
        # Query-aware categorisation:
        if query_share >= min_owner_share and owner_share < min_owner_share:
            reasons.append(ReasonCode.QUERY_RELEVANT_BUT_IGNORED)
        elif query_share < min_owner_share and owner_share < min_owner_share:
            reasons.append(ReasonCode.NOT_QUERY_RELEVANT)

        is_dead = owner_share < min_owner_share
        per_file_records.append(
            PerFileUsage(
                path=path,
                n_units=int(b["n_units"]),
                used=int(b["used"]),
                uncertain=int(b["uncertain"]),
                unused=int(b["unused"]),
                coverage=float(coverage),
                mean_score=float(b["sum_score"] / n),
                max_evidence=float(b["max_evidence"]),
                owner_tokens=int(b["owner_tokens"]),
                owner_share=float(owner_share),
                query_owner_tokens=int(b["query_owner_tokens"]),
                query_owner_share=float(query_share),
                dead_weight=bool(is_dead),
                reason_codes=list(dict.fromkeys(reasons)),
                dominating_peer=dominating_peer,
            )
        )
        if is_dead:
            dead_weight_files.append(path)

    per_file_records.sort(key=lambda r: (r.owner_share, r.coverage))
    total_files = max(1, len(per_file_records))
    histogram = Counter(
        rc.value for rec in per_file_records for rc in rec.reason_codes
    )
    return FileAttributionResult(
        per_file=per_file_records,
        per_unit=per_unit_records,
        dead_weight_files=dead_weight_files,
        dead_weight_ratio=float(len(dead_weight_files)) / total_files,
        n_files=len(per_file_records),
        n_response_tokens=int(n_response_tokens),
        n_query_tokens=int(n_query_tokens),
        min_owner_share=float(min_owner_share),
        coverage_threshold=float(coverage_threshold),
        low_cosine_threshold=float(low_cosine_threshold),
        reason_code_histogram=dict(histogram),
    )
