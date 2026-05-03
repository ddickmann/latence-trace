"""Public models for standalone TRACE compression."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class CompressionRequest(BaseModel):
    """Request for SuperPod-compatible LLMLingua2 compression."""

    action: Literal["compress", "compress_messages"] = Field(default="compress")
    text: str | None = Field(default=None, description="Single text payload to compress.")
    messages: list[dict[str, Any]] | None = Field(
        default=None,
        description="Optional chat-style messages. The service compresses their joined content.",
    )
    target_token_ratio: float = Field(
        default=0.6,
        ge=0.05,
        le=1.0,
        description="Approximate fraction of whitespace tokens to keep.",
    )
    compression_rate: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="SuperPod semantics: fraction of tokens to remove. Overrides target_token_ratio.",
    )
    chunk_size: int = Field(default=4096, ge=512, le=16384)
    force_tokens: list[str] = Field(
        default_factory=list,
        description="Exact tokens/strings that the LLMLingua2 postprocessor must preserve.",
    )
    preserve_tokens: list[str] = Field(
        default_factory=lambda: [
            "```",
            "`",
            "|",
            "|---",
            "| ---",
            "#",
            "##",
            "###",
            "- ",
            "* ",
        ],
        description=(
            "Structural tokens to preserve by default for code fences, inline code, "
            "Markdown headings/lists, and tables."
        ),
    )
    auto_preserve_structural_tokens: bool = Field(
        default=True,
        description="Automatically preserve code-like, Markdown, and table tokens found in the input.",
    )
    force_preserve_digit: bool = True
    fallback_mode: bool = True
    apply_toon: bool = True
    toon_encoding: bool = True
    target_compression: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Message compression rate for newest compressible message.",
    )
    max_compression: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Message compression rate for oldest compressible message.",
    )
    preserve_exact: list[str] = Field(
        default_factory=list,
        description="Exact strings that should survive fallback compression when present.",
    )
    mode: Literal["extractive", "token_classification", "llmlingua2"] = Field(default="llmlingua2")

    @model_validator(mode="after")
    def validate_payload(self) -> CompressionRequest:
        if self.action == "compress_messages" and not self.messages:
            raise ValueError("Provide messages for compress_messages")
        if self.action == "compress" and not (self.text or "").strip() and not self.messages:
            raise ValueError("Provide either text or messages")
        if self.messages and self.action == "compress":
            self.action = "compress_messages"
        return self

    @property
    def effective_compression_rate(self) -> float:
        if self.compression_rate is not None:
            return self.compression_rate
        return max(0.0, min(1.0, 1.0 - self.target_token_ratio))

    @property
    def effective_force_tokens(self) -> list[str]:
        return sorted(
            {token for token in [*self.force_tokens, *self.preserve_tokens, *self.preserve_exact] if token}
        )


class CompressionSpan(BaseModel):
    start: int
    end: int
    text: str
    keep_score: float


class CompressionResponse(BaseModel):
    compressed_text: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    compression_percentage: float = 0.0
    tokens_saved: int = 0
    preserved_terms: list[str] = Field(default_factory=list)
    compressed_messages: list[dict[str, Any]] | None = None
    spans: list[CompressionSpan] = Field(default_factory=list)
    provider: Literal["superpod_vllm", "fallback"] = "fallback"
    diagnostics: dict[str, Any] = Field(default_factory=dict)
