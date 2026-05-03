"""Pydantic contracts for stateful TRACE sessions."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from latence_trace.memory.models import MemoryDiagnostics, MemoryPolicy, MemoryState

TraceSessionKind = Literal["code", "rag", "general"]
TraceSessionStatus = Literal["active", "closed", "expired", "tombstone"]
TraceSessionBackend = Literal["server", "client"]
TraceSessionPrivacyMode = Literal["standard", "redacted", "metadata_only"]
TraceSessionEventType = Literal[
    "user_message",
    "assistant_response",
    "tool_result",
    "file_read",
    "retrieval_result",
    "code_diff",
    "error",
    "decision",
    "constraint",
    "observation",
]
TraceSessionLane = Literal["code", "rag"]


class TraceSessionState(BaseModel):
    """Unified persisted state for Core TRACE + InfiniMem."""

    session_id: str
    kind: TraceSessionKind = "general"
    status: TraceSessionStatus = "active"
    state_backend: TraceSessionBackend = "server"
    privacy_mode: TraceSessionPrivacyMode = "standard"
    created_at: str | None = None
    updated_at: str | None = None
    event_count: int = 0
    last_event_id: str | None = None
    memory_state: MemoryState | None = None
    memory_policy: MemoryPolicy = Field(default_factory=MemoryPolicy)
    code_session_state: dict[str, Any] | None = None
    rollup_turns: list[dict[str, Any]] = Field(default_factory=list)
    counters: dict[str, int | float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    redaction_policy: dict[str, Any] = Field(default_factory=dict)
    retention_policy: dict[str, Any] = Field(default_factory=dict)


class TraceSessionCreateRequest(BaseModel):
    session_id: str | None = Field(default=None, description="Optional caller-defined opaque id.")
    kind: TraceSessionKind = "general"
    metadata: dict[str, Any] = Field(default_factory=dict)
    memory_policy: MemoryPolicy | None = None
    redaction_policy: dict[str, Any] = Field(default_factory=dict)
    retention_policy: dict[str, Any] = Field(default_factory=dict)
    state_backend: TraceSessionBackend = "server"
    privacy_mode: TraceSessionPrivacyMode = "standard"


class TraceSessionCreateResponse(BaseModel):
    session: TraceSessionState


class TraceSessionGetResponse(BaseModel):
    session: TraceSessionState


class TraceSessionEvent(BaseModel):
    event_id: str | None = None
    event_type: TraceSessionEventType = "observation"
    content: str = ""
    role: str | None = None
    source: str | None = None
    timestamp: str | None = None
    raw_context: str | None = None
    query_text: str | None = None
    response_text: str | None = None
    trace_signals: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    idempotency_key: str | None = None


class TraceSessionEventRequest(BaseModel):
    event: TraceSessionEvent
    memory_domain: str | None = None
    return_context: bool = True


class TraceSessionEventResponse(BaseModel):
    session: TraceSessionState
    event: TraceSessionEvent
    hot_context: str = ""
    memory_diagnostics: MemoryDiagnostics | None = None


class TraceSessionScoreRequest(BaseModel):
    lane: TraceSessionLane | None = None
    trace_request: dict[str, Any] | None = None
    event: TraceSessionEvent | None = None
    memory_domain: str | None = None
    return_context: bool = True
    idempotency_key: str | None = None


class TraceSessionScoreResponse(BaseModel):
    session: TraceSessionState
    trace_response: dict[str, Any]
    hot_context: str = ""
    memory_diagnostics: MemoryDiagnostics | None = None


class TraceSessionContextResponse(BaseModel):
    session_id: str
    kind: TraceSessionKind
    status: TraceSessionStatus
    hot_context: str = ""
    warm_tokens: int = 0
    hot_tokens: int = 0
    cold_tokens: int = 0
    memory_state: MemoryState | None = None
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class TraceSessionRollupRequest(BaseModel):
    heatmap_format: Literal["none", "data", "html"] = "data"
    extra_turns: list[dict[str, Any]] = Field(default_factory=list)


class TraceSessionRollupResponse(BaseModel):
    session: TraceSessionState
    rollup: dict[str, Any]


class TraceSessionCloseResponse(BaseModel):
    session: TraceSessionState
