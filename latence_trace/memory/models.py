"""Data models for TRACE Memory / InfiniMem."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

SpanType = Literal[
    "goal",
    "constraint",
    "decision",
    "evidence",
    "code_symbol",
    "code_fragment",
    "error",
    "tool_result",
    "retrieval_chunk",
    "summary",
    "filler",
]
MemoryLayer = Literal["hot", "warm", "cold", "tombstone"]
CompressionLevel = Literal["exact", "extractive", "summary", "fact", "tombstone"]
MemoryActionType = Literal["added", "kept", "demoted", "removed", "deduped", "superseded", "anchored"]


class SpanSignature(BaseModel):
    normalized_hash: str
    typed_key: str
    identifiers: list[str] = Field(default_factory=list)
    file_paths: list[str] = Field(default_factory=list)
    symbols: list[str] = Field(default_factory=list)
    numbers: list[str] = Field(default_factory=list)
    dates: list[str] = Field(default_factory=list)
    rare_terms: list[str] = Field(default_factory=list)


class SpanScores(BaseModel):
    salience: float = 0.0
    relevance: float = 0.0
    attribution: float = 0.0
    redundancy: float = 0.0
    dead_weight: float = 0.0
    staleness: float = 0.0
    exact_critical: float = 0.0
    survival_value: float = 0.0
    learned_survival: float = 0.0
    survival_horizon_turns: float = 0.0
    domain_decay_pressure: float = 0.0


class SpanRecord(BaseModel):
    id: str
    text: str
    span_type: SpanType = "filler"
    layer: MemoryLayer = "warm"
    compression_level: CompressionLevel = "exact"
    signature: SpanSignature
    scores: SpanScores = Field(default_factory=SpanScores)
    token_count: int = 0
    created_turn: int = 0
    last_seen_turn: int = 0
    source: str = "turn"
    provenance: dict[str, Any] = Field(default_factory=dict)
    supersedes: list[str] = Field(default_factory=list)


class MemoryState(BaseModel):
    version: str = "infinimem.v1"
    turn_index: int = 0
    spans: list[SpanRecord] = Field(default_factory=list)
    cold_provenance: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryPolicy(BaseModel):
    hot_token_budget: int = Field(default=800, ge=1)
    warm_token_budget: int = Field(default=4000, ge=1)
    max_spans: int = Field(default=256, ge=1)
    exact_critical_floor: float = Field(default=0.72, ge=0.0, le=1.0)
    rho: float = Field(default=0.7, ge=0.1, le=2.0)
    search_horizon_turns: int = Field(default=2, ge=1)
    rag_horizon_turns: int = Field(default=3, ge=1)
    tool_horizon_turns: int = Field(default=6, ge=1)
    chat_horizon_turns: int = Field(default=10, ge=1)
    code_horizon_turns: int = Field(default=18, ge=1)
    code_anchor_horizon_turns: int = Field(default=36, ge=1)


class MemoryAction(BaseModel):
    action: MemoryActionType
    span_id: str
    reason: str
    from_layer: MemoryLayer | None = None
    to_layer: MemoryLayer | None = None


class MemoryDiagnostics(BaseModel):
    actions: list[MemoryAction] = Field(default_factory=list)
    hot_tokens: int = 0
    warm_tokens: int = 0
    cold_tokens: int = 0
    removed_tokens: int = 0
    exact_critical_spans: int = 0
    top_survival_causes: list[dict[str, Any]] = Field(default_factory=list)


class MemoryUpdateRequest(BaseModel):
    turn_text: str = ""
    query_text: str | None = None
    response_text: str | None = None
    raw_context: str | None = None
    memory_domain: str | None = None
    prior_memory_state: MemoryState | None = None
    trace_response: dict[str, Any] | None = None
    trace_signals: dict[str, Any] | None = None
    memory_ranker_weights: dict[str, float] | None = None
    memory_policy: MemoryPolicy = Field(default_factory=MemoryPolicy)
    enable_ingress_compression: bool = False
    ingress_compression_min_tokens: int = Field(default=512, ge=1)
    ingress_compression_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    ingress_apply_toon: bool | None = None


class MemoryUpdateResponse(BaseModel):
    next_memory_state: MemoryState
    hot_context: str
    actions: list[MemoryAction] = Field(default_factory=list)
    diagnostics: MemoryDiagnostics = Field(default_factory=MemoryDiagnostics)
