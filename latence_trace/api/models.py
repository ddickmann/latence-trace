"""Pydantic models for the latence-trace Groundedness Tracker (Beta).

These were extracted verbatim from the upstream voyager-index API surface and
re-homed to keep latence-trace fully self-contained. The schema is unchanged
so callers using the historical voyager-index endpoint can swap to
latence-trace without touching their request/response shapes.
"""

from enum import Enum
from typing import List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CollectionKind(str, Enum):
    """Collection kind tag retained on the response for compatibility.

    Standalone latence-trace deployments will typically emit ``raw_context``
    or ``embeddings`` here; the enum is preserved so schema consumers do not
    have to change their parsers.
    """

    DENSE = "dense"
    LATE_INTERACTION = "late_interaction"
    MULTIMODAL = "multimodal"
    SHARD = "shard"


class GroundednessSegmentationMode(str, Enum):
    SENTENCE = "sentence"
    SENTENCE_PACKED = "sentence_packed"
    PARAGRAPH = "paragraph"


class GroundednessPrimaryMetric(str, Enum):
    REVERSE_CONTEXT = "reverse_context"
    TRIANGULAR = "triangular"


class GroundednessRequest(BaseModel):
    """Beta post-generation groundedness scoring request."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "chunk_ids": ["doc-1", "doc-7"],
                "query_text": "When was Teardrops released in the United States?",
                "response_text": "Teardrops was released in the United States on 20 July 1981.",
                "evidence_limit": 5,
                "primary_metric": "reverse_context",
                "raw_context_chunk_tokens": 256,
            }
        }
    )

    response_text: str = Field(
        ...,
        min_length=1,
        description="Generated response text to score for groundedness / hallucination detection.",
    )
    query_text: Optional[str] = Field(
        default=None,
        description="Optional query text used for query-conditioned diagnostic channels and grounded coverage.",
    )
    chunk_ids: Optional[List[Union[str, int]]] = Field(
        default=None,
        description="Preferred production fast path: external chunk ids whose stored support vectors should be reused.",
    )
    raw_context: Optional[str] = Field(
        default=None,
        description="Compatibility fallback: raw context text to segment and re-encode on demand.",
    )
    segmentation_mode: GroundednessSegmentationMode = Field(
        default=GroundednessSegmentationMode.SENTENCE_PACKED,
        description=(
            "How raw_context should be segmented into support units before scoring. "
            "The default sentence_packed mode packs adjacent sentences into token-budgeted windows."
        ),
    )
    raw_context_chunk_tokens: int = Field(
        default=256,
        ge=1,
        le=8192,
        description=(
            "Approximate token budget for packed raw_context support windows. "
            "Applies to raw_context only and is used by sentence_packed mode. "
            "Budgets above the active encoder limit may trigger warnings and truncation."
        ),
    )
    primary_metric: GroundednessPrimaryMetric = Field(
        default=GroundednessPrimaryMetric.REVERSE_CONTEXT,
        description="Primary scalar score exposed as the headline groundedness metric. The shipped Beta default is reverse_context.",
    )
    evidence_limit: int = Field(
        default=8,
        ge=1,
        le=128,
        description="Maximum top evidence links to return in the bounded sparse response.",
    )
    debug_dense_matrices: bool = Field(
        default=False,
        description="Include dense token similarity matrices when payload size permits. Intended for debugging, not default UI payloads.",
    )
    include_triangular_diagnostics: bool = Field(
        default=True,
        description="When query_text is provided, include optional query-conditioned diagnostics such as triangular groundedness, echo, and grounded coverage.",
    )
    model: Optional[str] = Field(
        default=None,
        description="Optional groundedness encoder override for response/query/raw-context encoding.",
    )
    query_prompt_name: Optional[str] = Field(
        default=None,
        description="Optional asymmetric prompt name for query encoding, e.g. query.",
    )
    document_prompt_name: Optional[str] = Field(
        default=None,
        description="Optional asymmetric prompt name for response/raw_context encoding, e.g. document.",
    )
    verification_samples: Optional[List[str]] = Field(
        default=None,
        description=(
            "Optional caller-supplied alternate responses drawn from the same generator "
            "at sampling temperature > 0. When present and semantic-entropy fusion is "
            "configured (VOYAGER_GROUNDEDNESS_FUSION_W_SEMANTIC_ENTROPY > 0 plus an NLI "
            "model), the service clusters the samples by bidirectional entailment and "
            "surfaces a semantic-entropy channel."
        ),
    )
    content_type: Optional[str] = Field(
        default=None,
        description=(
            "Optional structured-source hint used by the structured-source adapter "
            "(Phase I). Known values: application/json, text/markdown, application/json+schema. "
            "When omitted the adapter auto-detects JSON and markdown table sources."
        ),
    )
    risk_band_stratum: Optional[str] = Field(
        default=None,
        description=(
            "Optional failure-mode hint consumed by the calibrated risk-band "
            "classifier (Phase H). Allowed values match the calibrated strata, "
            "typically one of: 'entity_swap', 'date_swap', 'number_swap', "
            "'unit_swap', 'negation', 'role_swap', 'partial', or 'default' "
            "(the conservative maximum). When omitted the classifier uses the "
            "hardest calibrated threshold so the green band stays honest."
        ),
    )

    @model_validator(mode="after")
    def validate_input_modes(self) -> "GroundednessRequest":
        has_chunk_ids = bool(self.chunk_ids)
        has_raw_context = bool((self.raw_context or "").strip())
        if has_chunk_ids == has_raw_context:
            raise ValueError("Provide exactly one of 'chunk_ids' or 'raw_context'")
        if self.primary_metric == GroundednessPrimaryMetric.TRIANGULAR and not (self.query_text or "").strip():
            raise ValueError("triangular primary_metric requires query_text")
        return self


class GroundednessScores(BaseModel):
    """Aggregate groundedness scores returned by the Beta endpoint."""

    primary_name: str
    primary_score: float
    reverse_context: float
    reverse_context_calibrated: Optional[float] = None
    literal_guarded: Optional[float] = None
    literal_mismatch_count: Optional[int] = None
    literal_match_count: Optional[int] = None
    literal_total_count: Optional[int] = None
    nli_aggregate: Optional[float] = None
    nli_claim_count: Optional[int] = None
    nli_skipped_count: Optional[int] = None
    groundedness_v2: Optional[float] = None
    consensus_hardened: Optional[float] = None
    reverse_query_context: Optional[float] = None
    triangular: Optional[float] = None
    echo_mean: Optional[float] = None
    grounded_coverage: Optional[float] = None
    null_bank_size: Optional[int] = None
    semantic_entropy_aggregate: Optional[float] = None
    semantic_entropy_raw: Optional[float] = None
    semantic_entropy_sample_count: Optional[int] = None
    structured_source_guarded: Optional[float] = None
    structured_source_detected: Optional[bool] = None
    risk_band: Optional[str] = None


class GroundednessLiteral(BaseModel):
    """A narrow-scope literal extracted from response or support text."""

    kind: str
    value: str
    normalized: str
    start: int
    end: int


class GroundednessLiteralDiagnostics(BaseModel):
    """Per-request literal extraction and matching diagnostics."""

    response_literals: List[GroundednessLiteral] = Field(default_factory=list)
    matches: List[GroundednessLiteral] = Field(default_factory=list)
    mismatches: List[GroundednessLiteral] = Field(default_factory=list)


class GroundednessNLIAtom(BaseModel):
    """Per-atom entailment record produced by atomic-fact decomposition (Phase F3)."""

    atom_index: int
    text: str
    char_start: int
    char_end: int
    entailment: float
    neutral: float
    contradiction: float
    score: float
    skipped: bool
    skip_reason: Optional[str] = None
    premise_count: int


class GroundednessNLIClaim(BaseModel):
    """Per-claim entailment record returned by the NLI verifier."""

    index: int
    text: str
    char_start: int
    char_end: int
    entailment: float
    neutral: float
    contradiction: float
    score: float
    skipped: bool
    skip_reason: Optional[str] = None
    premise_count: int
    atoms: List[GroundednessNLIAtom] = Field(default_factory=list)


class GroundednessNLIDiagnostics(BaseModel):
    """Per-request NLI verification diagnostics."""

    aggregate_score: Optional[float] = None
    claims: List[GroundednessNLIClaim] = Field(default_factory=list)


class GroundednessSemanticEntropyCluster(BaseModel):
    """One equivalence cluster emitted by the semantic-entropy peer."""

    cluster_id: int
    size: int
    representative: str


class GroundednessSemanticEntropyDiagnostics(BaseModel):
    """Per-request semantic-entropy (Phase G) diagnostics."""

    aggregate_score: Optional[float] = None
    entropy_raw: Optional[float] = None
    sample_count: int = 0
    cluster_count: int = 0
    clusters: List[GroundednessSemanticEntropyCluster] = Field(default_factory=list)


class GroundednessStructuredTripleMatch(BaseModel):
    """A source/response triple match emitted by the structured-source adapter."""

    subject: str
    predicate: str
    object: str
    matched: bool
    mismatch_kind: Optional[str] = None


class GroundednessStructuredDiagnostics(BaseModel):
    """Per-request structured-source (Phase I) diagnostics."""

    source_format: Optional[str] = None
    source_triple_count: int = 0
    response_triple_count: int = 0
    matches: List[GroundednessStructuredTripleMatch] = Field(default_factory=list)
    mismatches: List[GroundednessStructuredTripleMatch] = Field(default_factory=list)


class GroundednessQueryToken(BaseModel):
    """Per-query-token grounded coverage diagnostics."""

    index: int
    token: str
    coverage: float


class GroundednessResponseToken(BaseModel):
    """Per-response-token groundedness diagnostics."""

    index: int
    token: str
    weight: float
    reverse_context: float
    reverse_context_calibrated: Optional[float] = None
    reverse_context_z: Optional[float] = None
    null_mean: Optional[float] = None
    null_std: Optional[float] = None
    nli_score: Optional[float] = None
    consensus_hardened: Optional[float] = None
    support_unit_hits_above_threshold: Optional[int] = None
    support_unit_soft_breadth: Optional[float] = None
    effective_support_units: Optional[float] = None
    reverse_query_context: Optional[float] = None
    triangular: Optional[float] = None
    echo: Optional[float] = None
    support_unit_index: Optional[int] = None
    support_token_index: Optional[int] = None
    support_token: Optional[str] = None
    chunk_id: Optional[Union[str, int]] = None
    heatmap_score: float


class GroundednessSupportUnit(BaseModel):
    """Support unit returned for chunk- or raw-context-mode groundedness."""

    index: int
    support_id: str
    chunk_id: Optional[Union[str, int]] = None
    source_mode: str
    text: str
    offset_start: Optional[int] = None
    offset_end: Optional[int] = None
    token_count: int
    tokens: List[str]
    token_scores: List[float]
    score: float
    matched_response_tokens: int


class GroundednessEvidence(BaseModel):
    """Top evidence alignment between a response token and a support token."""

    response_token_index: int
    response_token: str
    support_unit_index: int
    support_token_index: int
    support_token: str
    chunk_id: Optional[Union[str, int]] = None
    metric: str
    score: float


class GroundednessEligibility(BaseModel):
    """Eligibility and fidelity metadata for groundedness trust boundaries."""

    collection_kind: CollectionKind
    vector_source: str
    storage_compression: Optional[str] = None
    quantization_mode: Optional[str] = None
    dequantized: bool
    user_facing_supported: bool
    warnings: List[str] = Field(default_factory=list)


class GroundednessDebugPayload(BaseModel):
    """Optional dense matrices for debugging or custom heatmaps."""

    response_to_support: Optional[List[List[float]]] = None
    response_to_query: Optional[List[List[float]]] = None
    triangular_gated: Optional[List[List[float]]] = None


class GroundednessResponse(BaseModel):
    """Beta groundedness scoring response with heatmap-ready sparse data."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "collection": "tutorial-li",
                "mode": "chunk_ids",
                "model": "lightonai/GTE-ModernColBERT-v1",
                "scores": {
                    "primary_name": "reverse_context",
                    "primary_score": 0.97,
                    "reverse_context": 0.97,
                    "consensus_hardened": 0.96,
                    "reverse_query_context": 0.98,
                    "triangular": 0.82,
                },
                "response_tokens": [
                    {
                        "index": 0,
                        "token": "Teardrops",
                        "weight": 1.0,
                        "reverse_context": 0.99,
                        "consensus_hardened": 0.98,
                        "support_unit_hits_above_threshold": 2,
                        "support_unit_soft_breadth": 1.7,
                        "effective_support_units": 1.8,
                        "heatmap_score": 0.99,
                        "support_unit_index": 0,
                        "support_token_index": 3,
                        "support_token": "Teardrops",
                        "chunk_id": "doc-7",
                    }
                ],
                "support_units": [
                    {
                        "index": 0,
                        "support_id": "doc-7",
                        "chunk_id": "doc-7",
                        "source_mode": "chunk_ids",
                        "text": "Teardrops is a single by George Harrison, released on 20 July 1981 in the United States.",
                        "token_count": 16,
                        "tokens": ["Teardrops", "is", "a", "single"],
                        "token_scores": [0.99, 0.35, 0.0, 0.71],
                        "score": 0.93,
                        "matched_response_tokens": 4,
                    }
                ],
                "top_evidence": [
                    {
                        "response_token_index": 0,
                        "response_token": "Teardrops",
                        "support_unit_index": 0,
                        "support_token_index": 3,
                        "support_token": "Teardrops",
                        "chunk_id": "doc-7",
                        "metric": "reverse_context",
                        "score": 0.99,
                    }
                ],
                "eligibility": {
                    "collection_kind": "late_interaction",
                    "vector_source": "stored_vectors",
                    "dequantized": True,
                    "user_facing_supported": True,
                    "warnings": [],
                },
                "time_ms": 4.2,
            }
        }
    )

    collection: str
    mode: str
    model: Optional[str] = None
    scores: GroundednessScores
    response_tokens: List[GroundednessResponseToken]
    support_units: List[GroundednessSupportUnit]
    top_evidence: List[GroundednessEvidence]
    eligibility: GroundednessEligibility
    query_tokens: Optional[List[GroundednessQueryToken]] = None
    debug: Optional[GroundednessDebugPayload] = None
    warnings: List[str] = Field(default_factory=list)
    literal_diagnostics: Optional[GroundednessLiteralDiagnostics] = None
    nli_diagnostics: Optional[GroundednessNLIDiagnostics] = None
    semantic_entropy_diagnostics: Optional[GroundednessSemanticEntropyDiagnostics] = None
    structured_diagnostics: Optional[GroundednessStructuredDiagnostics] = None
    time_ms: float
