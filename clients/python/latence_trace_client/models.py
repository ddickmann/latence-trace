"""SDK-facing pydantic models.

These mirror the server schema but are intentionally a *subset* -- we
expose the fields production callers actually consume and let the rest
ride along in ``GroundednessResponse.raw`` so server-side additions
never break the typed surface.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, List, Mapping, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field


class RiskBand(str, Enum):
    GREEN = "green"
    AMBER = "amber"
    RED = "red"
    UNKNOWN = "unknown"
    UNSUPPORTED = "unsupported"


class AttributionMode(str, Enum):
    CLOSED_BOOK = "closed_book"
    OPEN_DOMAIN = "open_domain"


class SupportUnit(BaseModel):
    """One structured premise (sentence/paragraph + attribution).

    Optional ``source_id`` / ``speaker`` / ``timestamp`` propagate to
    the per-unit attribution slot in the response so callers can render
    a citation panel without re-resolving the source.
    """

    model_config = ConfigDict(extra="allow")
    text: str
    source_id: Optional[str] = None
    speaker: Optional[str] = None
    timestamp: Optional[str] = None
    metadata: Optional[Mapping[str, Any]] = None


class TokenScore(BaseModel):
    model_config = ConfigDict(extra="allow")
    token: str
    char_start: int
    char_end: int
    g_t: float = Field(..., description="Per-token groundedness signal.")
    e_t: Optional[float] = Field(default=None, description="Per-token query echo.")
    j_star: Optional[int] = None


class NLIVerdict(BaseModel):
    model_config = ConfigDict(extra="allow")
    claim: str
    label: str
    score: float
    premise_index: Optional[int] = None


class GroundednessScores(BaseModel):
    model_config = ConfigDict(extra="allow")
    groundedness_v1: Optional[float] = None
    groundedness_v2: Optional[float] = None
    coverage_score_u: Optional[float] = None
    semantic_entropy: Optional[float] = None
    context_coverage_ratio: Optional[float] = None
    context_attribution_ratio: Optional[float] = None
    support_units_used: Optional[int] = None
    support_units_total: Optional[int] = None


class GroundednessRequest(BaseModel):
    """Wire-compatible request body.

    Use exactly one premise lane: ``chunk_ids`` (vector fast path),
    ``raw_context`` (free-text), or ``support_units`` (structured).
    """

    model_config = ConfigDict(extra="allow")
    query: Optional[str] = None
    response_text: str
    chunk_ids: Optional[List[str]] = None
    raw_context: Optional[List[str]] = None
    support_units: Optional[List[SupportUnit]] = None
    attribution_mode: AttributionMode = AttributionMode.CLOSED_BOOK
    primary_metric: Optional[str] = None
    coverage_threshold: Optional[float] = None
    chunk_token_budget: Optional[int] = None
    chunk_token_overlap: Optional[int] = None
    locale: Optional[str] = None
    runtime_head_features: Optional[Mapping[str, float]] = None
    trajectory_features: Optional[Mapping[str, float]] = None


class RuntimeDecision(BaseModel):
    model_config = ConfigDict(extra="allow")
    action: str
    score: float
    score_channel: str
    class_key: str
    head_id: Optional[str] = None
    head_enabled: Optional[bool] = None
    head_score: Optional[float] = None
    head_features_used: Sequence[str] = Field(default_factory=list)
    head_reason_codes: Sequence[str] = Field(default_factory=list)


class GroundednessResponse(BaseModel):
    """Wire-compatible response body."""

    model_config = ConfigDict(extra="allow")
    risk_band: RiskBand
    risk_reason: Optional[str] = None
    scores: GroundednessScores = Field(default_factory=GroundednessScores)
    response_tokens: Sequence[TokenScore] = Field(default_factory=list)
    nli: Sequence[NLIVerdict] = Field(default_factory=list)
    support_units: Sequence[Mapping[str, Any]] = Field(default_factory=list)
    runtime_decision: Optional[RuntimeDecision] = None
    runtime_head_features: Optional[Mapping[str, float]] = None
    request_id: Optional[str] = None
    raw: Optional[Mapping[str, Any]] = None
