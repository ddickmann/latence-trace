"""SDK-facing pydantic models.

These mirror the server schema but are intentionally a *subset* -- we
expose the fields production callers actually consume and let the rest
ride along in ``GroundednessResponse.raw`` so server-side additions
never break the typed surface.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RiskBand(str, Enum):
    GREEN = "green"
    AMBER = "amber"
    RED = "red"
    UNKNOWN = "unknown"
    UNSUPPORTED = "unsupported"


class AttributionMode(str, Enum):
    CLOSED_BOOK = "closed_book"
    OPEN_DOMAIN = "open_domain"


class ComplianceLabelMode(str, Enum):
    OPEN = "open"
    CATEGORY = "category"


class ComplianceRedactionMode(str, Enum):
    MASK = "mask"
    REPLACE = "replace"


class SupportUnit(BaseModel):
    """One structured premise (sentence/paragraph + attribution).

    Optional ``source_id`` / ``speaker`` / ``timestamp`` propagate to
    the per-unit attribution slot in the response so callers can render
    a citation panel without re-resolving the source.
    """

    model_config = ConfigDict(extra="allow")
    text: str
    source_id: str | None = None
    speaker: str | None = None
    timestamp: str | None = None
    metadata: Mapping[str, Any] | None = None


class TokenScore(BaseModel):
    model_config = ConfigDict(extra="allow")
    token: str
    char_start: int
    char_end: int
    g_t: float = Field(..., description="Per-token groundedness signal.")
    e_t: float | None = Field(default=None, description="Per-token query echo.")
    j_star: int | None = None


class NLIVerdict(BaseModel):
    model_config = ConfigDict(extra="allow")
    claim: str
    label: str
    score: float
    premise_index: int | None = None


class GroundednessScores(BaseModel):
    model_config = ConfigDict(extra="allow")
    groundedness_v1: float | None = None
    groundedness_v2: float | None = None
    coverage_score_u: float | None = None
    semantic_entropy: float | None = None
    context_coverage_ratio: float | None = None
    context_attribution_ratio: float | None = None
    support_units_used: int | None = None
    support_units_total: int | None = None
    context_trust_score: float | None = None
    context_trust_suspicious_count: int | None = None
    context_trust_blocked_count: int | None = None
    context_trust_max_risk: float | None = None
    risk_band: str | None = None


class GroundednessRequest(BaseModel):
    """Wire-compatible request body.

    Use exactly one premise lane: ``chunk_ids`` (vector fast path),
    ``raw_context`` (free-text), or ``support_units`` (structured).
    """

    model_config = ConfigDict(extra="allow")
    query_text: str | None = None
    response_text: str
    chunk_ids: list[str] | None = None
    raw_context: str | None = None
    support_units: list[SupportUnit] | None = None
    attribution_mode: AttributionMode = AttributionMode.CLOSED_BOOK
    primary_metric: str | None = None
    coverage_threshold: float | None = None
    raw_context_chunk_tokens: int | None = None
    response_chunk_tokens: int | None = None
    locale: str | None = None
    context_trust_enabled: bool | None = None
    runtime_head_features: Mapping[str, float] | None = None
    trajectory_features: Mapping[str, float] | None = None


class RuntimeDecision(BaseModel):
    model_config = ConfigDict(extra="allow")
    action: str
    score: float
    score_channel: str
    class_key: str
    head_id: str | None = None
    head_enabled: bool | None = None
    head_score: float | None = None
    head_features_used: Sequence[str] = Field(default_factory=list)
    head_reason_codes: Sequence[str] = Field(default_factory=list)


class GroundednessResponse(BaseModel):
    """Wire-compatible response body."""

    model_config = ConfigDict(extra="allow")
    risk_band: RiskBand | None = None
    risk_reason: str | None = None
    scores: GroundednessScores = Field(default_factory=GroundednessScores)
    response_tokens: Sequence[TokenScore] = Field(default_factory=list)
    nli: Sequence[NLIVerdict] = Field(default_factory=list)
    support_units: Sequence[Mapping[str, Any]] = Field(default_factory=list)
    context_trust_diagnostics: Mapping[str, Any] | None = None
    runtime_decision: RuntimeDecision | None = None
    runtime_head_features: Mapping[str, float] | None = None
    request_id: str | None = None
    raw: Mapping[str, Any] | None = None

    @model_validator(mode="after")
    def fill_native_risk_band(self) -> GroundednessResponse:
        """Accept both native FastAPI and compact/gateway response shapes."""

        if self.risk_band is not None:
            return self
        raw_band = self.scores.risk_band
        if raw_band is None:
            return self
        normalized = {
            "low": "green",
            "medium": "amber",
            "high": "red",
        }.get(str(raw_band).lower(), str(raw_band).lower())
        if normalized in {item.value for item in RiskBand}:
            self.risk_band = RiskBand(normalized)
        return self


class ComplianceCustomLabel(BaseModel):
    label_name: str
    extractor: str


class ComplianceEntity(BaseModel):
    model_config = ConfigDict(extra="allow")
    start: int
    end: int
    text: str
    label: str
    score: float = 0.0
    source: str = "model"
    redacted_value: str | None = None
    redaction_mode: str | None = None
    metadata: Mapping[str, Any] | None = None


class ComplianceUsage(BaseModel):
    model_config = ConfigDict(extra="allow")
    chunks_processed: int
    labels_used: int
    entity_count: int = 0
    unique_labels: list[str] = Field(default_factory=list)
    redaction_mode: ComplianceRedactionMode | None = None
    redacted: bool = False
    mode: ComplianceLabelMode
    categories: list[str] = Field(default_factory=list)


class ComplianceRedactionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    text: str
    mode: ComplianceLabelMode = ComplianceLabelMode.OPEN
    categories: list[str] = Field(default_factory=list)
    labels: list[str] | None = None
    threshold: float = 0.5
    redact: bool = True
    redaction_mode: ComplianceRedactionMode = ComplianceRedactionMode.MASK
    custom_labels: list[ComplianceCustomLabel] = Field(default_factory=list)
    country: str | None = None
    flat_ner: bool = True
    multi_label: bool = False
    include_original_text: bool = False


class ComplianceRedactionResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    success: bool = True
    original_text: str | None = None
    entities: list[ComplianceEntity] = Field(default_factory=list)
    entity_count: int
    unique_labels: list[str] = Field(default_factory=list)
    redacted_text: str | None = None
    chunks_processed: int
    labels_used: list[str] = Field(default_factory=list)
    label_mode: ComplianceLabelMode
    selected_categories: list[str] = Field(default_factory=list)
    processing_time_ms: float
    timings_ms: Mapping[str, float] = Field(default_factory=dict)
    usage: ComplianceUsage


class CompressionSpan(BaseModel):
    model_config = ConfigDict(extra="allow")
    start: int
    end: int
    text: str
    keep_score: float


class CompressionResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    compressed_text: str
    original_tokens: int | None = None
    compressed_tokens: int | None = None
    compression_ratio: float | None = None
    compression_percentage: float = 0.0
    tokens_saved: int = 0
    preserved_terms: list[str] = Field(default_factory=list)
    compressed_messages: list[dict[str, Any]] | None = None
    spans: list[CompressionSpan] = Field(default_factory=list)
    provider: str | None = None
    diagnostics: Mapping[str, Any] = Field(default_factory=dict)
    request_id: str | None = None
    raw: Mapping[str, Any] | None = None


class MemoryUpdateResponse(BaseModel):
    model_config = ConfigDict(extra="allow")
    next_memory_state: Mapping[str, Any]
    hot_context: str | None = None
    hot_context_preview: str | None = None
    actions: list[Mapping[str, Any]] = Field(default_factory=list)
    diagnostics: Mapping[str, Any] = Field(default_factory=dict)
    request_id: str | None = None
    raw: Mapping[str, Any] | None = None
