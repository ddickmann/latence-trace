"""Pydantic models for the compliance redaction runtime."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from latence_trace.compliance.labels import (
    GDPR_CATEGORIES,
    all_gdpr_labels,
    model_alias_metadata,
)


class CustomLabelInput(BaseModel):
    label_name: str = Field(..., min_length=1)
    extractor: str = Field(..., min_length=1)


class ComplianceRedactionRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "text": "Jane Doe was born on 1980-01-01. Email jane@example.com.",
                    "mode": "open",
                    "redact": True,
                    "redaction_mode": "mask",
                },
                {
                    "text": "Employee EMP-123 has account DE89370400440532013000.",
                    "mode": "category",
                    "categories": ["financial", "employment_and_education"],
                },
            ]
        }
    )

    text: str = Field(..., min_length=1)
    mode: Literal["open", "category"] = "open"
    categories: list[str] = Field(default_factory=list)
    labels: list[str] | None = None
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    redact: bool = False
    redaction_mode: Literal["mask", "replace"] = "mask"
    custom_labels: list[CustomLabelInput] = Field(default_factory=list)
    country: str | None = None
    flat_ner: bool = True
    multi_label: bool = False
    include_original_text: bool = True

    @model_validator(mode="after")
    def _validate_mode(self) -> ComplianceRedactionRequest:
        if self.mode == "category" and not self.categories and not self.labels:
            raise ValueError("category mode requires categories or an explicit labels allowlist")
        return self


class ComplianceEntity(BaseModel):
    start: int
    end: int
    text: str
    label: str
    score: float = 0.0
    source: str = "model"
    redacted_value: str | None = None
    redaction_mode: str | None = None
    metadata: dict[str, Any] | None = None


class ComplianceUsage(BaseModel):
    chunks_processed: int
    labels_used: int
    entity_count: int = 0
    unique_labels: list[str] = Field(default_factory=list)
    redaction_mode: Literal["mask", "replace"] | None = None
    redacted: bool = False
    mode: Literal["open", "category"]
    categories: list[str]


class ComplianceRedactionResponse(BaseModel):
    success: bool = True
    original_text: str | None = None
    entities: list[ComplianceEntity]
    entity_count: int
    unique_labels: list[str]
    redacted_text: str | None = None
    chunks_processed: int
    labels_used: list[str]
    label_mode: Literal["open", "category"]
    selected_categories: list[str]
    processing_time_ms: float
    timings_ms: dict[str, float]
    usage: ComplianceUsage


def compliance_schema_metadata() -> dict[str, Any]:
    return {
        "categories": GDPR_CATEGORIES,
        "labels": all_gdpr_labels(),
        "model_label_aliases": model_alias_metadata(),
        "modes": ["open", "category"],
    }
