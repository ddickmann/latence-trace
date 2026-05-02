"""Canonical trajectory schema for dataset-backed TRACE Memory evaluation."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from latence_trace.memory.models import SpanType

TrajectoryDomain = Literal["chat", "rag", "grounding", "code", "tool"]
OutcomeStatus = Literal["success", "failure", "unknown", "partial"]


class CanonicalSpan(BaseModel):
    id: str
    text: str
    source: str
    turn_index: int = 0
    span_type: SpanType = "evidence"
    speaker: str | None = None
    tool: str | None = None
    exact_critical_terms: list[str] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CanonicalTurn(BaseModel):
    turn_index: int
    query_text: str | None = None
    turn_text: str = ""
    response_text: str | None = None
    raw_context: str | None = None
    retrieved_context: list[CanonicalSpan] = Field(default_factory=list)
    tool_outputs: list[CanonicalSpan] = Field(default_factory=list)
    candidate_spans: list[CanonicalSpan] = Field(default_factory=list)
    gold: dict[str, Any] = Field(default_factory=dict)
    trace_scores: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CanonicalOutcome(BaseModel):
    status: OutcomeStatus = "unknown"
    score: float | None = None
    passed_tests: bool | None = None
    policy_success: bool | None = None
    answer_correct: bool | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CanonicalTrajectory(BaseModel):
    dataset: str
    case_id: str
    domain: TrajectoryDomain
    turns: list[CanonicalTurn]
    candidate_spans: list[CanonicalSpan] = Field(default_factory=list)
    responses: list[str] = Field(default_factory=list)
    retrieved_context: list[CanonicalSpan] = Field(default_factory=list)
    tool_outputs: list[CanonicalSpan] = Field(default_factory=list)
    outcome: CanonicalOutcome = Field(default_factory=CanonicalOutcome)
    gold: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_turns(self) -> CanonicalTrajectory:
        if not self.turns:
            raise ValueError("CanonicalTrajectory requires at least one turn")
        return self

    @property
    def critical_terms(self) -> list[str]:
        terms: list[str] = []
        if self.domain == "tool":
            for span in [*self.candidate_spans, *self.tool_outputs]:
                terms.extend(span.exact_critical_terms)
        elif self.domain == "code":
            for span in self.candidate_spans:
                terms.extend(span.exact_critical_terms)
        for turn in self.turns:
            if self.domain == "tool":
                for span in [*turn.candidate_spans, *turn.tool_outputs]:
                    terms.extend(span.exact_critical_terms)
            elif self.domain == "code":
                for span in turn.candidate_spans:
                    terms.extend(span.exact_critical_terms)
            terms.extend(str(term) for term in turn.gold.get("critical_terms", []) or [])
        terms.extend(str(term) for term in self.gold.get("critical_terms", []) or [])
        return sorted({term for term in terms if term})

    def full_history_text(self) -> str:
        parts: list[str] = []
        for turn in self.turns:
            parts.extend(
                text
                for text in [turn.query_text, turn.turn_text, turn.response_text, turn.raw_context]
                if text
            )
            parts.extend(span.text for span in turn.retrieved_context)
            parts.extend(span.text for span in turn.tool_outputs)
        return "\n".join(parts)


def coerce_trajectory(row: dict[str, Any]) -> CanonicalTrajectory:
    """Accept old synthetic rows but normalize everything to canonical form."""

    if "dataset" in row and "case_id" in row and "domain" in row:
        return CanonicalTrajectory.model_validate(row)

    turns = []
    for idx, turn in enumerate(row.get("turns", []), start=1):
        turns.append(
            {
                "turn_index": int(turn.get("turn_index") or idx),
                "query_text": turn.get("query_text"),
                "turn_text": turn.get("turn_text") or "",
                "response_text": turn.get("response_text"),
                "raw_context": turn.get("raw_context"),
                "gold": {"critical_terms": row.get("critical_terms", [])},
            }
        )
    return CanonicalTrajectory(
        dataset=str(row.get("dataset") or "synthetic"),
        case_id=str(row.get("case_id") or row.get("id") or "case"),
        domain=row.get("domain") or _infer_domain(row),
        turns=turns,
        gold={"critical_terms": row.get("critical_terms", [])},
        metadata={key: value for key, value in row.items() if key not in {"turns", "critical_terms"}},
    )


def _infer_domain(row: dict[str, Any]) -> TrajectoryDomain:
    text = " ".join(
        str(value)
        for turn in row.get("turns", [])
        for value in turn.values()
        if isinstance(value, str)
    ).lower()
    if "src/" in text or "pytest" in text or ".py" in text:
        return "code"
    if "tool" in text or "api" in text:
        return "tool"
    if "retrieved" in text or "chunk" in text:
        return "rag"
    return "chat"
