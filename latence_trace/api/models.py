"""Pydantic models for the latence-trace Groundedness Tracker.

These were extracted verbatim from the upstream voyager-index API surface and
re-homed to keep latence-trace fully self-contained. The schema is unchanged
so callers using the historical voyager-index endpoint can swap to
latence-trace without touching their request/response shapes.
"""

from enum import Enum
from typing import Dict, List, Literal, Optional, Union

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

from latence_trace.memory.models import MemoryDiagnostics, MemoryPolicy, MemoryState


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


class ScoringMode(str, Enum):
    """Top-level scoring lane selector.

    ``rag`` (default) keeps the long-standing retrieval-augmented
    groundedness behaviour used by enterprise RAG pipelines — every
    existing client continues to work untouched when ``scoring_mode``
    is omitted.

    ``code`` routes through the code-lane orchestrator
    (:mod:`latence_trace.core.code_lane.orchestrator`) which layers
    AST drift detection, per-identifier novelty, an ambiguity-gated
    NLI cascade, optional semantic entropy, a calibrated composite
    score, and per-file / per-unit attribution with reason codes on
    top of the shared ColBERT MaxSim backbone. Pick this lane when
    scoring agentic coding turns (Cursor, Claude Code, OpenAI Codex,
    OpenCode, aider, ...).
    """

    RAG = "rag"
    CODE = "code"


class CorpusType(str, Enum):
    """Caller-declared or inferred corpus type for per-class calibration.

    Set by the corpus router middleware
    (:mod:`latence_trace.middleware.corpus_router`); callers may also
    pre-declare the class when they know it (e.g. a structured-reporting
    tenant that always routes through ``rag.structured``).

    Six classes are shipped in v1, each with its own calibrated
    ``(fusion_weights, thresholds)`` bundle in
    ``latence_trace/data/calibration.<class>.json``:

    * ``rag.prose.enterprise``    - Veracier-style enterprise RAG prose
    * ``rag.prose.short_factoid`` - HaluEval QA-style short factoid QA
    * ``rag.prose.multi_claim``   - RAGTruth-style multi-claim summaries
    * ``rag.structured``          - Data2Text / tabular / schema-shaped
    * ``rag.code_in_context``     - prose-about-code agent turns
    * ``code.agentic_trace``      - agentic coding turns with fenced code
    """

    RAG_PROSE_ENTERPRISE = "rag.prose.enterprise"
    RAG_PROSE_SHORT_FACTOID = "rag.prose.short_factoid"
    RAG_PROSE_MULTI_CLAIM = "rag.prose.multi_claim"
    RAG_STRUCTURED = "rag.structured"
    RAG_CODE_IN_CONTEXT = "rag.code_in_context"
    CODE_AGENTIC_TRACE = "code.agentic_trace"


class TraceRuntimeProfile(str, Enum):
    """Hosted runtime profile selected per request.

    ``standard`` preserves the process/default Trace configuration. ``quality``
    opts into the stronger groundedness profile used for higher-cost hosted
    requests.
    """

    STANDARD = "standard"
    QUALITY = "quality"


class GroundednessSegmentationMode(str, Enum):
    SENTENCE = "sentence"
    SENTENCE_PACKED = "sentence_packed"
    PARAGRAPH = "paragraph"


class GroundednessPrimaryMetric(str, Enum):
    REVERSE_CONTEXT = "reverse_context"
    TRIANGULAR = "triangular"


class AttributionMode(str, Enum):
    """Caller-declared evidence policy for a groundedness request.

    ``closed_book`` (the default) tells the service that everything required
    to ground the response must be supplied in the request itself
    (``chunk_ids`` / ``raw_context`` / ``support_units``). When the premise
    set is empty the service returns ``risk_band="unknown"`` with
    ``reason="no_premise_supplied"`` instead of inventing a score from a
    zero-evidence input.

    ``open_domain`` is a forward-compatible reservation for the K5
    retrieval-callback lane that ships post-v1. In v1 the service accepts
    the field for schema stability and returns ``risk_band="unsupported"``
    with ``reason="open_domain_pending_v1_next"`` so callers can detect the
    feature gate in production.
    """

    CLOSED_BOOK = "closed_book"
    OPEN_DOMAIN = "open_domain"


class GroundednessUsageState(str, Enum):
    """Tri-state precision-first support-unit usage label."""

    USED = "used"
    UNUSED = "unused"
    UNCERTAIN = "uncertain"


class GroundednessSupportUnitInput(BaseModel):
    """Caller-supplied structured premise for the ``support_units`` lane.

    ``support_units[]`` is the third premise-supplying lane alongside
    ``chunk_ids`` and ``raw_context``. Use it whenever the caller already has
    multi-source / multi-speaker premises (sales-call transcripts, multi-doc
    RAG with explicit per-doc IDs, knowledge-base passages with passage IDs)
    and wants the per-unit attribution to flow back through the response so
    each surviving claim can be attributed to its originating speaker /
    source / turn.

    Backward compatibility: legacy callers that set ``raw_context`` or
    ``chunk_ids`` see no behavior change.
    """

    text: str = Field(..., min_length=1, description="Premise text for this support unit.")
    source_id: Optional[str] = Field(
        default=None,
        description="Stable, caller-defined identifier (document id, passage id, transcript turn id, ...).",
    )
    speaker: Optional[str] = Field(
        default=None,
        description="Speaker / author label for dialogue or multi-author premises.",
    )
    timestamp: Optional[str] = Field(
        default=None,
        description="Optional ISO-8601 timestamp echoed back in the response for time-aware audit trails.",
    )
    metadata: Optional[dict] = Field(
        default=None,
        description="Free-form caller metadata returned verbatim on the matching response support unit.",
    )


class GroundednessRequest(BaseModel):
    """Post-generation groundedness scoring request."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "summary": "RAG lane (default) — enterprise retrieval grounding",
                    "value": {
                        "chunk_ids": ["doc-1", "doc-7"],
                        "query_text": "When was Teardrops released in the United States?",
                        "response_text": "Teardrops was released in the United States on 20 July 1981.",
                        "evidence_limit": 5,
                        "primary_metric": "reverse_context",
                        "raw_context_chunk_tokens": 256,
                    },
                },
                {
                    "summary": "Code lane — agentic coding turn",
                    "value": {
                        "scoring_mode": "code",
                        "session_id": "hashed-session-abc123",
                        "response_language_hint": "python",
                        "query_text": "Add retry logic to fetch_user",
                        "response_text": "```python\nfrom httpx import AsyncClient\n\nasync def fetch_user(id: int):\n    client = AsyncClient()\n    return await client.get(f'/users/{id}')\n```",
                        "raw_context": "# fetch_user.py\nasync def fetch_user(id: int):\n    async with httpx.AsyncClient() as client:\n        return await client.get(f'/users/{id}')\n",
                        "emit_chunk_ownership": True,
                        "evidence_limit": 5,
                    },
                },
            ]
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
    support_units: Optional[List[GroundednessSupportUnitInput]] = Field(
        default=None,
        description=(
            "Structured premise lane: list of caller-supplied support units, each with text, "
            "optional source_id, speaker, timestamp, and metadata. Each surviving response unit "
            "in the response carries the matching source_id/speaker/timestamp so callers can show "
            "'this answer was grounded in turn 4 by Dr. X'. Mutually exclusive with chunk_ids and raw_context."
        ),
    )
    attribution_mode: AttributionMode = Field(
        default=AttributionMode.CLOSED_BOOK,
        description=(
            "Evidence policy. ``closed_book`` (default) requires premises in the request and refuses "
            "to score with risk_band='unknown' / reason='no_premise_supplied' when none are provided. "
            "``open_domain`` is reserved for the post-v1 retrieval-callback lane and currently returns "
            "risk_band='unsupported' / reason='open_domain_pending_v1_next' for forward schema compatibility."
        ),
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
    response_chunk_tokens: int = Field(
        default=256,
        ge=1,
        le=8192,
        description=(
            "Approximate token budget for packed response windows. Mirrors "
            "raw_context_chunk_tokens on the response side so long responses "
            "(beyond the encoder's max sequence length) are sentence-packed into "
            "windows and scored chunk-by-chunk against the full support set. "
            "When the encoded response fits in a single window the orchestrator "
            "skips chunking entirely (parity-preserving fast path). "
            "Budgets above the active encoder limit may trigger warnings and "
            "truncation."
        ),
    )
    coverage_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Threshold on per-support-unit reverse-context similarity used "
            "to flag a unit as 'used' in the context-coverage observability "
            "signal. A unit's coverage_score is the maximum similarity any "
            "response token had to that unit's tokens; when it crosses this "
            "threshold the unit is counted as having contributed strongly to "
            "the response. Defaults to 0.5, a conservative cutoff for "
            "ColBERT-style normalized embeddings (raise to 0.6-0.7 for "
            "stricter retrieval-efficiency reporting; lower to 0.3-0.4 for "
            "tolerant scoring on noisy multilingual encoders). The threshold "
            "is reported back in scores.context_coverage_threshold."
        ),
    )
    primary_metric: GroundednessPrimaryMetric = Field(
        default=GroundednessPrimaryMetric.REVERSE_CONTEXT,
        description="Primary scalar score exposed as the headline groundedness metric. The shipped default is reverse_context.",
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
    context_trust_enabled: bool = Field(
        default=True,
        validation_alias=AliasChoices("context_trust_enabled", "guard_check_enabled"),
        description=(
            "Enable the context-trust / prompt-guard scan for retrieved support "
            "context. Defaults to true. Set false only for trusted internal "
            "benchmarks or latency isolation tests; operator-level deployment "
            "configuration still controls which provider is used."
        ),
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
    structured_verification: Optional[str] = Field(
        default=None,
        description=(
            "Controls the structured-source verification lane. 'auto' (default) "
            "runs the detector but suppresses the typed lane on prose-shaped "
            "chunks to avoid false positives. 'on' forces the typed lane on the "
            "raw support (useful when the caller knows the content is tabular). "
            "'off' disables the lane entirely so the fused headline relies on "
            "narrative channels (NLI / literal / ColBERT). Recommended for "
            "enterprise RAG on multilingual legal or cyber prose where numeric "
            "key-value patterns can appear incidentally."
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
    runtime_head_features: Optional[Dict[str, float]] = Field(
        default=None,
        description=(
            "Optional explicit feature map for promoted runtime heads. Use this "
            "for agentic trace or root-cause heads when the caller already has "
            "file, symbol, test, patch, order, or claim-atom features. Missing "
            "required features keep the affected head rollback-safe repair-only."
        ),
    )
    trajectory_features: Optional[Dict[str, float]] = Field(
        default=None,
        description=(
            "Alias for runtime_head_features used by agentic coding clients that "
            "emit trajectory-native feature maps."
        ),
    )
    scoring_mode: ScoringMode = Field(
        default=ScoringMode.RAG,
        description=(
            "Lane selector. ``rag`` (default) keeps the existing enterprise "
            "RAG groundedness pipeline untouched. ``code`` routes through the "
            "code lane — AST drift, literal novelty, ambiguity-gated NLI, "
            "semantic entropy, calibrated composite score, and per-file / "
            "per-unit attribution with reason codes — designed for agentic "
            "coding harnesses like Cursor, Claude Code, OpenAI Codex, and "
            "OpenCode. Backwards compatible: clients that do not set this "
            "field stay on the RAG lane."
        ),
    )
    profile: Optional[TraceRuntimeProfile] = Field(
        default=None,
        description=(
            "Hosted Trace runtime profile. Omitted or ``standard`` preserves "
            "the deployed/default scoring configuration; explicit ``quality`` "
            "uses the stronger NLI/fusion profile and is billed at the hosted "
            "quality rate by the gateway."
        ),
    )
    auto_decide: Optional[bool] = Field(
        default=None,
        description=(
            "Opt into zero-human-in-the-loop mode. When True and the initial "
            "scoring band is ``amber`` (hedge-gate), the service runs one "
            "pinned LLM-judge call over the atomic-claim + evidence pairs and "
            "collapses the band to ``green`` or ``red``. Budget is capped at "
            "one judge call per request; per-tenant daily spend is enforced "
            "at the gateway. Defaults to the tenant-level default "
            "(``VOYAGER_TRACE_AUTO_DECIDE_DEFAULT``); set to False to force "
            "the conservative amber hedge even on auto-decide tenants."
        ),
    )
    session_id: Optional[str] = Field(
        default=None,
        description=(
            "Optional opaque session identifier echoed back in the response "
            "and structured logs so plugins can correlate turn events for "
            "multi-turn EMA / z-score analysis client-side. The server is "
            "stateless — this field is a passthrough label, nothing more. "
            "PII concerns: do *not* put user text in here; use a hashed "
            "session token."
        ),
    )
    response_language_hint: Optional[str] = Field(
        default=None,
        description=(
            "Optional programming-language hint for the code lane. Accepts "
            "any of ``python``, ``typescript``, ``tsx``, ``javascript``, "
            "``jsx``, ``go``, ``rust`` and their short aliases. When omitted "
            "the AST extractor falls back to the language declared on the "
            "response's first fenced code block, then to content heuristics. "
            "Ignored by the RAG lane."
        ),
    )
    corpus_type: Optional[CorpusType] = Field(
        default=None,
        description=(
            "Optional tenant-declared corpus type. Forces the corpus "
            "router to use the matching per-class calibration bundle "
            "(``latence_trace/data/calibration.<class>.json``) regardless "
            "of what the classifier would infer. When omitted, the "
            "classifier infers the class from request features in "
            "<= 3 ms on CPU."
        ),
    )
    language: Optional[Literal["en", "de", "auto"]] = Field(
        default=None,
        description=(
            "Optional caller-declared content language. Drives two things: "
            "(1) which per-language calibration bundle the corpus router "
            "loads (``calibration.<class>.<lang>.json`` falling back to "
            "the English bundle when a German bundle is missing), and "
            "(2) the default NLI premise-selection strategy for the claim "
            "verification step. ``\"de\"`` switches to balanced German "
            "defaults (``top_k_premises=2, premise_concat=False, "
            "premise_aggregate=\"max\"``) which avoid the multi-premise "
            "concatenation dilution observed on long German contexts with "
            "mDeBERTa-xnli. ``\"en\"`` keeps the historical English "
            "defaults (``top_k=3, concat=True``) untouched. ``\"auto\"`` "
            "or ``None`` (default) runs the lightweight ``langdetect`` "
            "probe over the response / query / context (in that order, "
            "200 chars each) and picks ``de`` only when the probe is "
            "high-confidence German. Anything else collapses to ``en``."
        ),
    )
    nli_max_claims: Optional[int] = Field(
        default=None,
        ge=0,
        description=(
            "Optional per-request override for the maximum number of "
            "claims (sentences in quality mode) that NLI verifies. "
            "Defaults to ``0`` (unlimited) so a long enterprise RAG "
            "answer is analysed end-to-end and the heatmap covers "
            "every sentence. The ``nli_max_latency_ms`` budget is the "
            "real safety net for very long answers: tail claims that "
            "would exceed the budget come back as ``skipped=True, "
            "skip_reason=\"latency_budget\"`` and the diagnostics "
            "surface ``claims_total`` and ``claims_skipped_for_budget`` "
            "so operators can see exactly how many sentences were "
            "fully scored. Set a positive integer here to force a hard "
            "cap (e.g. for cost-sensitive batch jobs); ``0`` means "
            "no cap."
        ),
    )
    nli_top_k_premises: Optional[int] = Field(
        default=None,
        ge=1,
        le=16,
        description=(
            "Optional per-request override for the number of top-reranked "
            "support units passed to NLI per claim. Defaults to 3 for "
            "English, 2 for German. Lower values reduce premise dilution "
            "on long contexts; higher values preserve correctness on "
            "claims that legitimately span multiple support units. "
            "Bounded to ``[1, 16]``."
        ),
    )
    nli_premise_concat: Optional[bool] = Field(
        default=None,
        description=(
            "Optional per-request override for premise concatenation. "
            "When ``True`` (English default) the top-k reranked premises "
            "are joined into one composite premise per atom and a single "
            "NLI call is made. When ``False`` (German default) each "
            "premise is scored independently and the per-premise NLI "
            "scores are aggregated. Forcing ``False`` on languages where "
            "the multilingual NLI model is sensitive to long mixed "
            "premises (notably German with mDeBERTa-xnli) avoids the "
            "dilution failure mode where a verbatim-supported claim is "
            "contradicted because the composite premise mixed in "
            "off-topic units."
        ),
    )
    nli_premise_aggregate: Optional[Literal["mean", "max", "min"]] = Field(
        default=None,
        description=(
            "Optional per-request override for the per-premise NLI "
            "aggregation mode used when ``nli_premise_concat=False``. "
            "Defaults to ``\"max\"`` for both English and German (the "
            "shipped ``_aggregate_premise_scores`` aggregator), which "
            "lets a single supporting / refuting premise carry the "
            "claim. ``\"mean\"`` and ``\"min\"`` are reserved for "
            "future tuning experiments and currently fall back to "
            "``\"max\"`` until additional aggregators ship."
        ),
    )
    emit_chunk_ownership: bool = Field(
        default=False,
        description=(
            "Code lane only. When True the response carries the full "
            "per-unit ownership table (one entry per support chunk with "
            "max cosine, owner counts, offsets, and usage state) so IDE "
            "plugins can drop sub-file regions at the 200k-token wall. "
            "Costs a few extra KB over the wire; keep False for UI-only "
            "dashboards that just need the file rollup."
        ),
    )
    heatmap_format: Literal["none", "data", "html"] = Field(
        default="data",
        description=(
            "Controls the ``heatmap`` convenience field on the response. "
            "``'data'`` (default) emits a compact structured payload "
            "(tokens + files + summary + thresholds) that any HTML "
            "template can bind to. ``'html'`` additionally returns "
            "``heatmap_html`` — a self-contained ``<div>`` with inline "
            "CSS that renders without a frontend dev. ``'none'`` "
            "disables both fields and saves a few KB for callers that "
            "render their own visualisations from ``scores`` + "
            "``file_attribution``."
        ),
    )
    session_state: Optional["SessionStatePayload"] = Field(
        default=None,
        description=(
            "Caller-portable session blob for code-lane temporal signals "
            "(drift, EMA groundedness, dead-weight streaks, "
            "recommendations). The server is stateless: echo back the "
            "``next_session_state`` from the previous response on each "
            "subsequent turn. Omit on the first turn to initialise. "
            "Ignored by the RAG lane. See ``docs/session_semantics.md``."
        ),
    )
    memory_state: Optional[MemoryState] = Field(
        default=None,
        description=(
            "Optional caller-portable InfiniMem span state. Echo the previous "
            "response's next_memory_state here to update memory in shadow mode."
        ),
    )
    memory_policy: Optional[MemoryPolicy] = Field(
        default=None,
        description=(
            "Optional hot/warm/cold memory budgets and survival policy. "
            "For large-context models, set context_window_tokens, "
            "memory_context_ratio, and target_token_reduction so adaptive "
            "memory can choose the smallest quality-gated budget instead of "
            "using only fixed token caps."
        ),
    )
    enable_memory_shadow: bool = Field(
        default=False,
        description=(
            "When true, update InfiniMem in shadow mode and return "
            "next_memory_state, hot_context_preview, and diagnostics. "
            "Default scoring inputs remain unchanged."
        ),
    )
    apply_memory_context: bool = Field(
        default=False,
        description=(
            "Reserved opt-in for future production coupling. V1 does not rewrite "
            "raw_context during scoring even when this is true."
        ),
    )

    @model_validator(mode="after")
    def validate_input_modes(self) -> "GroundednessRequest":
        has_chunk_ids = bool(self.chunk_ids)
        has_raw_context = bool((self.raw_context or "").strip())
        has_support_units = bool(self.support_units)
        premise_lanes = [has_chunk_ids, has_raw_context, has_support_units]
        active_lanes = sum(premise_lanes)
        if active_lanes > 1:
            raise ValueError(
                "Provide exactly one of 'chunk_ids', 'raw_context', or 'support_units'"
            )
        if active_lanes == 0:
            # K1: closed_book + zero premises is a *valid* request. The
            # service short-circuits it to risk_band="unknown" /
            # reason="no_premise_supplied" instead of inventing a score.
            # open_domain + zero premises is reserved for the K5
            # retrieval-callback lane (post-v1) and the service emits a
            # forward-compat refusal there as well, so we accept the input
            # at the schema layer in both modes.
            pass
        if self.primary_metric == GroundednessPrimaryMetric.TRIANGULAR and not (self.query_text or "").strip():
            raise ValueError("triangular primary_metric requires query_text")
        return self


class GroundednessScores(BaseModel):
    """Aggregate groundedness scores returned by the /groundedness endpoint."""

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
    structured_source: Optional[float] = Field(
        default=None,
        description=(
            "Typed structured-evidence (AND-gate) score: min over per-claim "
            "(entity_align, value_match, unit_match, sign_match) for every "
            "response claim that aligned to a typed source cell. Returns "
            "``None`` for pure-prose responses where no claim could be "
            "aligned. When non-null and the AND-gate is enabled, the "
            "headline groundedness becomes ``min(narrative, structured)`` "
            "so a single broken cell collapses support."
        ),
    )
    structured_source_typed_aligned: Optional[int] = Field(
        default=None,
        description=(
            "Number of response claims that the typed structured-evidence "
            "lane aligned to a typed source cell."
        ),
    )
    structured_source_typed_count: Optional[int] = Field(
        default=None,
        description=(
            "Total number of typed claims extracted from the response, "
            "including those dropped because no cell could be aligned."
        ),
    )
    risk_band: Optional[str] = None
    context_coverage_ratio: Optional[float] = Field(
        default=None,
        description=(
            "Retrieval-efficiency observability metric: the fraction of "
            "support units whose coverage_score >= context_coverage_threshold. "
            "Range [0, 1]. Higher values mean the retriever's chunks were "
            "actually useful to the response. A value of 0.4 means 60% of the "
            "fetched chunks were dead weight — strong signal that the "
            "retrieval k or query expansion is over-fetching."
        ),
    )
    context_coverage_threshold: Optional[float] = Field(
        default=None,
        description=(
            "Threshold on per-unit reverse-context similarity used to flag a "
            "support unit as 'used'. Default 0.5 — a conservative cutoff for "
            "ColBERT-style normalized embeddings. Configurable per request "
            "via coverage_threshold."
        ),
    )
    support_units_used: Optional[int] = Field(
        default=None,
        description=(
            "Count of support units with coverage_score >= "
            "context_coverage_threshold. Numerator of context_coverage_ratio."
        ),
    )
    support_units_total: Optional[int] = Field(
        default=None,
        description=(
            "Total support units evaluated. Denominator of "
            "context_coverage_ratio."
        ),
    )
    context_attribution_ratio: Optional[float] = Field(
        default=None,
        description=(
            "Stricter retrieval-efficiency signal than context_coverage_ratio: "
            "the fraction of support units that were the *argmax* support for "
            "at least one response token. Always <= context_coverage_ratio. "
            "Use the gap between the two ratios to identify units that were "
            "semantically relevant but lost to a sibling chunk — those are "
            "candidates for retrieval deduplication."
        ),
    )
    context_attribution_used_count: Optional[int] = Field(
        default=None,
        description=(
            "Count of support units with matched_response_tokens > 0. "
            "Numerator of context_attribution_ratio."
        ),
    )
    support_units_usage_used: Optional[int] = Field(
        default=None,
        description=(
            "Count of support units whose precision-first tri-state "
            "``usage_state`` is ``used``. Unlike ``support_units_used`` this "
            "can include low-overlap units rescued by NLI evidence."
        ),
    )
    support_units_unused: Optional[int] = Field(
        default=None,
        description=(
            "Count of support units whose tri-state ``usage_state`` is "
            "``unused``. This label is precision-first: ambiguous or "
            "redundant units should fall into ``uncertain`` instead."
        ),
    )
    support_units_uncertain: Optional[int] = Field(
        default=None,
        description=(
            "Count of support units whose tri-state ``usage_state`` is "
            "``uncertain`` because the engine abstained instead of forcing a "
            "binary used/unused verdict."
        ),
    )
    context_usage_ratio: Optional[float] = Field(
        default=None,
        description=(
            "Fraction of support units whose tri-state ``usage_state`` is "
            "``used``."
        ),
    )
    context_unused_ratio: Optional[float] = Field(
        default=None,
        description=(
            "Fraction of support units emitted as high-confidence "
            "``usage_state = unused``."
        ),
    )
    context_uncertain_ratio: Optional[float] = Field(
        default=None,
        description=(
            "Fraction of support units emitted as ``usage_state = uncertain``."
        ),
    )
    context_trust_score: Optional[float] = Field(
        default=None,
        description=(
            "Average context-trust risk across support units. Diagnostic only; "
            "does not lower groundedness_v2 in v1."
        ),
    )
    context_trust_suspicious_count: Optional[int] = Field(
        default=None,
        description="Count of support units with suspicious context-trust state.",
    )
    context_trust_blocked_count: Optional[int] = Field(
        default=None,
        description="Count of support units with blocked context-trust state.",
    )
    context_trust_max_risk: Optional[float] = Field(
        default=None,
        description="Maximum context-trust risk observed across support units.",
    )
    # --- Code-lane additions ------------------------------------------
    composite_phantom_score: Optional[float] = Field(
        default=None,
        description=(
            "Code lane only. Calibrated composite phantom-guard score in "
            "``[0, 1]``; 1.0 = maximally grounded, 0.0 = flagged as phantom. "
            "Combines reverse_context, per_token_p10, literal_guard, "
            "literal_novelty_min, AST drift/phantom counts, and (when "
            "triggered) NLI contradiction. Threshold defaults to 0.5."
        ),
    )
    composite_phantom_probability: Optional[float] = Field(
        default=None,
        description=(
            "Code lane only. Calibrated probability that the turn is a "
            "phantom / hallucination. This is ``1 - composite_phantom_score`` "
            "when the logistic composite is active."
        ),
    )
    composite_phantom_verdict: Optional[bool] = Field(
        default=None,
        description=(
            "Code lane only. True when "
            "``composite_phantom_probability > threshold``. Designed to be "
            "the one-bit signal IDE plugins consume."
        ),
    )
    literal_novelty_min: Optional[float] = Field(
        default=None,
        description=(
            "Code lane only. Minimum first-match score over every response "
            "identifier. Low values mean at least one identifier only matches "
            "low-cosine support units — a classic phantom-import signature."
        ),
    )
    literal_novelty_missing_count: Optional[int] = Field(
        default=None,
        description=(
            "Code lane only. Count of response identifiers that had no match "
            "in any support unit."
        ),
    )
    ast_phantom_symbol_count: Optional[int] = Field(
        default=None,
        description=(
            "Code lane only. Count of imports / classes the response uses "
            "that do not appear in any context unit. Precision-1.0 phantom "
            "signal when > 0."
        ),
    )
    ast_literal_drift_count: Optional[int] = Field(
        default=None,
        description=(
            "Code lane only. Count of response identifiers (class / function "
            "/ method / kwarg) not present in any context unit's symbol "
            "table. Catches identifier-swap drift that the literal guard "
            "misses because sub-word tokens overlap."
        ),
    )
    ast_phantom_verdict: Optional[bool] = Field(
        default=None,
        description=(
            "Code lane only. True when at least one phantom import / class "
            "was detected."
        ),
    )
    nli_contradiction_prob_max: Optional[float] = Field(
        default=None,
        description=(
            "Code lane only. Maximum per-claim NLI contradiction probability "
            "produced by the ambiguity-triggered cascade. ``None`` when the "
            "cascade did not fire (composite outside ``[0.65, 0.90]``)."
        ),
    )
    nli_cascade_triggered: Optional[bool] = Field(
        default=None,
        description=(
            "Code lane only. True when the NLI cascade ran for this turn."
        ),
    )
    dead_weight_ratio: Optional[float] = Field(
        default=None,
        description=(
            "Code lane only. Fraction of context files whose "
            "``owner_share < min_owner_share`` — i.e. contributed zero (or "
            "negligible) evidence to the response."
        ),
    )
    dead_weight_file_count: Optional[int] = Field(
        default=None,
        description=(
            "Code lane only. Raw count of context files flagged as dead "
            "weight this turn."
        ),
    )


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
    support_ids: List[str] = Field(default_factory=list)
    support_unit_indices: List[int] = Field(default_factory=list)


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
    support_ids: List[str] = Field(default_factory=list)
    support_unit_indices: List[int] = Field(default_factory=list)
    atoms: List[GroundednessNLIAtom] = Field(default_factory=list)


class GroundednessNLIDiagnostics(BaseModel):
    """Per-request NLI verification diagnostics."""

    aggregate_score: Optional[float] = Field(
        default=None,
        validation_alias=AliasChoices("aggregate_score", "aggregate"),
    )
    claims: List[GroundednessNLIClaim] = Field(default_factory=list)
    # Coverage observability. Without these fields the heatmap frontend
    # cannot tell "every sentence was fully scored" apart from "the
    # latency budget deferred the tail of a long answer", so a
    # 30-sentence response with 14 deferred claims would silently look
    # the same as a 16-sentence response with full coverage. The
    # runtime now reports the full sentence count (``claims_total``),
    # how many got a real entailment verdict (``claims_scored``), and
    # the per-reason breakdown of the rest. ``aggregate_score`` is
    # still computed only over the scored claims.
    claims_total: Optional[int] = Field(
        default=None,
        ge=0,
        description=(
            "Total number of sentences/claims the splitter produced. "
            "Equal to len(claims) for backwards compatibility."
        ),
    )
    claims_scored: Optional[int] = Field(
        default=None,
        ge=0,
        description=(
            "Subset of ``claims_total`` that received a real NLI "
            "entailment verdict (i.e. not skipped). Used by the "
            "demo coverage chip."
        ),
    )
    claims_skipped_for_budget: Optional[int] = Field(
        default=None,
        ge=0,
        description=(
            "Claims that were left unscored because the per-request "
            "``nli_max_latency_ms`` budget would have been exceeded. "
            "These appear in the heatmap with a neutral ``skipped`` "
            "band so the response stays visually fully covered."
        ),
    )
    claims_skipped_for_no_premises: Optional[int] = Field(
        default=None,
        ge=0,
        description=(
            "Claims that had zero retrieved premises and so could not "
            "be entailment-scored. Usually indicates an empty or "
            "completely off-topic context."
        ),
    )


class ContextTrustLabel(BaseModel):
    """Normalized context-trust label emitted for a support unit."""

    label: str
    score: float
    severity: str
    count: int = 1
    source: str = "heuristic"


class ContextTrustSpan(BaseModel):
    """Span-level context-trust evidence inside a support unit."""

    label: str
    start: int
    end: int
    text: str
    score: float
    severity: str
    source: str = "heuristic"
    pattern_id: Optional[str] = None
    metadata: Dict[str, object] = Field(default_factory=dict)


class ContextTrustDiagnostics(BaseModel):
    """Aggregate context safety diagnostics for RAG/support-unit context."""

    enabled: bool = True
    provider: str = "heuristic"
    support_unit_count: int = 0
    trusted_count: int = 0
    suspicious_count: int = 0
    blocked_count: int = 0
    score: float = 0.0
    max_risk: float = 0.0
    suspicious_support_ids: List[str] = Field(default_factory=list)
    blocked_support_ids: List[str] = Field(default_factory=list)
    labels: List[ContextTrustLabel] = Field(default_factory=list)
    skipped_reason: Optional[str] = None
    suspicious_threshold: float = 0.20
    blocked_threshold: float = 0.60


class CorpusRouteDiagnostics(BaseModel):
    """Per-request corpus router diagnostics.

    Emitted when the router produced a class assignment for the request
    (whether from the classifier or a tenant-supplied override). Carries
    enough information for dashboards / auditors to reproduce the
    scoring configuration that was actually applied.
    """

    corpus_type: str = Field(
        ..., description="Inferred or declared corpus class (e.g. 'rag.prose.enterprise')."
    )
    source: str = Field(
        ...,
        description=(
            "Where the class came from. ``classifier`` = inferred from "
            "request features by the LR classifier; ``rule`` = a high-"
            "precision structural rule in the overlay fired before the "
            "classifier (see ``rule_reason``); ``explicit`` = caller "
            "supplied ``corpus_type`` on the request; ``fallback`` = "
            "classifier unavailable, default bundle used."
        ),
    )
    confidence: Optional[float] = Field(
        default=None,
        description="Classifier posterior probability for the selected class (None for explicit / fallback).",
    )
    classifier_latency_ms: float = Field(
        default=0.0,
        description="Wall-clock latency of the featurize + predict_proba pass.",
    )
    artefact_sha256: Optional[str] = Field(
        default=None,
        description="SHA256 of the corpus_classifier.joblib artefact actually loaded at serving time.",
    )
    rule_reason: Optional[str] = Field(
        default=None,
        description=(
            "Human-readable reason string when a high-precision rule in "
            "the router overlay fired (e.g. ``rule:json_rooted_context``). "
            "Null when the LR classifier or a fallback path produced the "
            "decision."
        ),
    )
    classifier_top_classes: List[Dict[str, Union[float, str]]] = Field(
        default_factory=list,
        description="Top classifier classes and probabilities, ordered by probability.",
    )
    classifier_probabilities: Dict[str, float] = Field(
        default_factory=dict,
        description="Full classifier probability vector when a classifier artefact was used.",
    )
    fusion_weights_applied: Optional[Dict[str, float]] = Field(
        default=None,
        description="Fusion weights from the calibration bundle that were layered on top of the runtime profile.",
    )
    thresholds_applied: Optional[Dict[str, float]] = Field(
        default=None,
        description="Green / amber thresholds applied by the scoring layer for this class.",
    )
    scoring_mode_applied: Optional[str] = Field(
        default=None,
        description="Scoring mode actually used after router override (``rag`` or ``code``).",
    )
    bundle_metric: Optional[str] = Field(
        default=None,
        description="Objective name from the calibration bundle (``f1_at_best_threshold`` / ``veracier_composite`` / ``paired_accuracy``).",
    )
    bundle_metric_value: Optional[float] = Field(
        default=None,
        description="Recorded value of that objective at calibration time (for transparency).",
    )


class RuntimeUnsupportedSpan(BaseModel):
    """Token-level unsupported span emitted by the runtime decision gate."""

    token_index: int
    token: str
    heatmap_score: Optional[float] = None
    nli_score: Optional[float] = None
    reverse_context_calibrated: Optional[float] = None
    char_start: Optional[int] = None
    char_end: Optional[int] = None


class RuntimeEvidence(BaseModel):
    """Compact evidence record used by automatic decision clients."""

    index: int
    support_id: Optional[str] = None
    text: str
    coverage_score: Optional[float] = None
    usage_state: Optional[str] = None
    context_trust_state: Optional[str] = None
    context_trust_score: Optional[float] = None
    context_trust_labels: List[str] = Field(default_factory=list)


class RuntimeDecisionRecord(BaseModel):
    """Stable allow / repair / block decision record for agentic runtimes."""

    policy_version: str
    policy_sha256: Optional[str] = None
    head_id: Optional[str] = None
    head_version: Optional[str] = None
    head_registry_sha256: Optional[str] = None
    head_enabled: Optional[bool] = None
    head_score: Optional[float] = None
    head_features_used: List[str] = Field(default_factory=list)
    head_reason_codes: List[str] = Field(default_factory=list)
    class_key: str
    score: float
    score_channel: str
    band: str
    action: Literal["allow", "auto_repair", "block"]
    evidence: List[RuntimeEvidence] = Field(default_factory=list)
    unsupported_spans: List[RuntimeUnsupportedSpan] = Field(default_factory=list)
    reason_codes: List[str] = Field(default_factory=list)
    allow_disabled: Optional[bool] = None
    block_disabled: Optional[bool] = None
    allow_threshold: Optional[float] = None
    block_threshold: Optional[float] = None
    rollback_safe: bool = True


class AmberEscalationDiagnostics(BaseModel):
    """Per-request amber auto-decide diagnostics (Phase 2)."""

    original_band: str = Field(
        ...,
        description="Band TRACE emitted before auto-decide ran. Always 'amber' when escalation fires.",
    )
    final_band: str = Field(
        ...,
        description="Band after the judge ran. Either 'green', 'red', or 'amber' (if the judge failed).",
    )
    judge_verdict: str = Field(
        ...,
        description="Judge verdict as emitted: 'green', 'red', or 'amber' when the judge failed to produce one.",
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Short judge-provided reasoning string, clipped to 600 chars.",
    )
    judge_provider: str = Field(
        ...,
        description="Provider used: 'openai', 'anthropic', or 'offline' (deterministic heuristic).",
    )
    judge_model: Optional[str] = Field(default=None)
    judge_latency_ms: float = Field(default=0.0)
    judge_cost_usd: Optional[float] = Field(default=None)
    judge_error: Optional[str] = Field(
        default=None,
        description="Populated when the judge call failed; the final_band stays 'amber'.",
    )


class GroundednessSemanticEntropyCluster(BaseModel):
    """One equivalence cluster emitted by the semantic-entropy peer."""

    cluster_id: int
    size: int
    representative: str


class GroundednessSemanticEntropyDiagnostics(BaseModel):
    """Per-request semantic-entropy (Phase G) diagnostics."""

    aggregate_score: Optional[float] = Field(
        default=None,
        validation_alias=AliasChoices("aggregate_score", "aggregate"),
    )
    entropy_raw: Optional[float] = None
    sample_count: int = 0
    cluster_count: int = 0
    skipped_reason: Optional[str] = None
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


# ---------------------------------------------------------------------------
# Code-lane diagnostics
# ---------------------------------------------------------------------------


class CodeLaneCompositeContributions(BaseModel):
    """Per-feature contribution breakdown of the composite score."""

    name: str = Field(..., description="Feature or interaction term name.")
    value: float = Field(..., description="Signed contribution to the composite.")


class CodeLaneCompositeDiagnostics(BaseModel):
    """Diagnostics for the calibrated composite phantom-guard score."""

    kind: str = Field(..., description="'linear' or 'logistic' depending on which composite ran.")
    composite_score: float
    phantom_probability: float
    verdict: bool
    threshold: float
    contributions: List[CodeLaneCompositeContributions] = Field(default_factory=list)


class CodeLaneAstDiagnostics(BaseModel):
    """Multi-language AST drift diagnostics."""

    language: Optional[str] = Field(
        default=None,
        description="Detected language (python/typescript/javascript/go/rust) or None.",
    )
    parser_backend: str = Field(
        ...,
        description="'tree_sitter' when a grammar loaded, 'regex_fallback' otherwise, 'disabled' when the AST pass is off.",
    )
    ast_literal_drift_count: int = 0
    ast_phantom_symbol_count: int = 0
    ast_phantom_verdict: bool = False
    drift_symbols: List[str] = Field(default_factory=list)
    phantom_symbols: List[str] = Field(default_factory=list)
    latency_ms: float = 0.0


class CodeLaneNLICascade(BaseModel):
    """Ambiguity-gated NLI cascade diagnostics."""

    triggered: bool
    skipped_reason: Optional[str] = None
    nli_contradiction_prob_max: float = 0.0
    nli_entailment_prob_mean: float = 0.0
    nli_aggregate: Optional[float] = None
    claim_count: int = 0
    verified_claim_count: int = 0
    latency_ms: float = 0.0


class CodeLaneLiteralNovelty(BaseModel):
    """Per-identifier first-match novelty diagnostics."""

    literal_novelty_min: float = 1.0
    literal_novelty_mean: float = 1.0
    literal_novelty_count: int = 0
    missing_literal_count: int = 0


class CodeLanePerUnitOwnership(BaseModel):
    """Chunk-level ownership record."""

    support_id: str
    path: Optional[str] = None
    unit_index: int
    max_cos: float
    response_owner_count: int
    query_owner_count: int
    response_owner_share: float
    query_owner_share: float
    offset_start: Optional[int] = None
    offset_end: Optional[int] = None
    usage_state: str


class CodeLanePerFileUsage(BaseModel):
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
    reason_codes: List[str] = Field(default_factory=list)
    dominating_peer: Optional[str] = None


class CodeLaneFileAttribution(BaseModel):
    """File-level attribution bundle (shared between code and RAG lanes).

    Despite the historical ``CodeLane`` prefix this model is lane-neutral
    — both ``scoring_mode == "code"`` and ``scoring_mode == "rag"``
    populate it via the shared
    :mod:`latence_trace.core.attribution.file_attribution` kernel.

    ``reason_code_histogram`` rolls the per-file ``reason_codes`` up into
    a small aggregate so dashboards can render the "53 % of the waste is
    ``dominated_by_single_file``" narrative without custom aggregation.
    """

    per_file: List[CodeLanePerFileUsage] = Field(default_factory=list)
    per_unit: List[CodeLanePerUnitOwnership] = Field(default_factory=list)
    dead_weight_files: List[str] = Field(default_factory=list)
    dead_weight_ratio: float = 0.0
    n_files: int = 0
    n_response_tokens: int = 0
    n_query_tokens: int = 0
    min_owner_share: float = 0.01
    coverage_threshold: float = 0.20
    low_cosine_threshold: float = 0.40
    reason_code_histogram: Dict[str, int] = Field(
        default_factory=dict,
        description=(
            "Aggregate count of reason codes across files. "
            "Keys are ReasonCode wire values (never_won_argmax, "
            "all_tokens_below_0_40, dominated_by_single_file, "
            "query_relevant_but_ignored, not_query_relevant)."
        ),
    )


# Lane-neutral alias — same schema, product-friendly name. Use this in
# new code (e.g. the top-level ``GroundednessResponse.file_attribution``
# field) so downstream consumers are not confused by the ``CodeLane``
# prefix when the data is coming from the RAG path.
FileAttributionDiagnostics = CodeLaneFileAttribution


class CodeLaneDiagnostics(BaseModel):
    """Full diagnostics payload for ``scoring_mode == "code"`` responses."""

    config: dict = Field(default_factory=dict, description="Snapshot of the code-lane config used.")
    total_latency_ms: float = 0.0
    component_latency_ms: dict = Field(
        default_factory=dict,
        description="Per-component latency breakdown: scorer_ms, literal_novelty_ms, ast_ms, nli_ms, file_attribution_ms.",
    )
    composite: Optional[CodeLaneCompositeDiagnostics] = None
    ast: Optional[CodeLaneAstDiagnostics] = None
    literal_novelty: Optional[CodeLaneLiteralNovelty] = None
    nli_cascade: Optional[CodeLaneNLICascade] = None
    file_attribution: Optional[CodeLaneFileAttribution] = None


class RollingStatsPayload(BaseModel):
    """Welford-style running mean + variance accumulator (wire shape)."""

    n: int = 0
    mean: float = 0.0
    m2: float = 0.0


class FileSessionStatsPayload(BaseModel):
    """Per-file rolling stats inside the portable session blob."""

    ema_owner_share: float = 0.0
    ema_query_owner_share: float = 0.0
    dead_turns: int = 0
    total_turns: int = 0


class SessionStatePayload(BaseModel):
    """Caller-portable session blob for temporal code-lane signals.

    The API is stateless for per-turn measurement. Temporal judgments
    (drift, EMA groundedness, dead-weight streaks, recommendations)
    require memory *somewhere* — this payload makes that memory a
    first-class, **caller-carried** concern.

    Protocol
    --------
    - Turn 1: omit ``session_state`` entirely. The response carries a
      freshly initialised ``next_session_state``.
    - Turn N+1: echo the previous ``next_session_state`` back as
      ``session_state``. The server runs a pure deterministic
      transform and returns an updated blob.
    - The server never persists this blob. It round-trips verbatim
      on the request/response boundary.

    Schema is versioned via ``schema_version``; mismatched versions
    are treated as a fresh session (safe default).

    See :mod:`latence_trace.core.code_lane.session` for the transform
    and ``docs/session_semantics.md`` for the full protocol spec.
    """

    model_config = ConfigDict(extra="ignore")

    schema_version: int = Field(
        default=1,
        description="Session-state schema version. Bumped on breaking changes.",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Opaque session identifier. Not used for lookup — just echoed.",
    )
    total_turns: int = 0
    cascade_fires: int = 0
    per_token_rolling: RollingStatsPayload = Field(default_factory=RollingStatsPayload)
    composite_rolling: RollingStatsPayload = Field(default_factory=RollingStatsPayload)
    groundedness_rolling: RollingStatsPayload = Field(
        default_factory=RollingStatsPayload
    )
    groundedness_baseline: RollingStatsPayload = Field(
        default_factory=RollingStatsPayload
    )
    ema_groundedness: Optional[float] = None
    file_stats: Dict[str, FileSessionStatsPayload] = Field(default_factory=dict)
    phantom_trail: List[Optional[bool]] = Field(default_factory=list)
    risk_band_trail: List[str] = Field(default_factory=list)
    ema_half_life_turns: int = 5
    groundedness_ema_half_life: int = 3
    verdict_window: int = 20
    risk_band_window: int = 20


class SessionSignals(BaseModel):
    """Derived session-level signals emitted next to ``next_session_state``.

    These are cheap rollups the caller can render directly without
    cracking the opaque state blob open. They are *advisory* — every
    per-turn metric in ``scores`` / ``code_lane_diagnostics`` is still
    available independently.
    """

    total_turns: int = 0
    drift_z_score: float = Field(
        default=0.0,
        description=(
            "Absolute per-session z-score of the current ``per_token_p10`` "
            "against the session's rolling baseline. A value >= 2.0 is a "
            "strong drift signal."
        ),
    )
    ema_groundedness: Optional[float] = Field(
        default=None,
        description="EMA of the turn-level groundedness headline over the session.",
    )
    groundedness_drift: float = Field(
        default=0.0,
        description=(
            "``ema_groundedness - baseline_mean`` where the baseline is "
            "the mean of the first ~10 turns. Negative = session is "
            "degrading vs its early behaviour."
        ),
    )
    dead_file_candidates: List[str] = Field(
        default_factory=list,
        description=(
            "Files whose EMA owner_share has stayed below the dead-weight "
            "threshold for at least 5 consecutive turns. Safe to evict "
            "from the agent's context window."
        ),
    )
    dead_weight_streak: int = Field(
        default=0,
        description="Length of the longest active dead-turn run across tracked files.",
    )
    cascade_density: float = Field(
        default=0.0,
        description=(
            "Fraction of turns that triggered the NLI cascade. A sustained "
            "value > 0.5 suggests the session has entered an ambiguous regime."
        ),
    )
    phantom_rate: float = Field(
        default=0.0,
        description="Rolling fraction of turns flagged ``ast_phantom_verdict=True``.",
    )
    red_streak: int = Field(
        default=0,
        description="Trailing count of consecutive ``red`` risk-band turns.",
    )
    recommendation: str = Field(
        default="continue",
        description=(
            "Session-level recommendation derived from the signals above. "
            "One of ``continue``, ``re-anchor``, ``fresh-chat``."
        ),
    )


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
    char_start: Optional[int] = Field(
        default=None,
        description=(
            "Character start offset within the original response_text for this "
            "token. Best-effort: populated when the encoder's tokenizer exposes "
            "an offset_mapping (modern HuggingFace fast tokenizers). When None, "
            "the UI must reconstruct positions by walking response_text."
        ),
    )
    char_end: Optional[int] = Field(
        default=None,
        description=(
            "Character end offset within the original response_text for this "
            "token. Best-effort, see char_start."
        ),
    )
    response_chunk_index: Optional[int] = Field(
        default=None,
        description=(
            "Index of the response chunk that produced this token, when "
            "response chunking is active. None for single-chunk (parity) "
            "fast path scoring."
        ),
    )


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
    coverage_score: float = Field(
        default=0.0,
        description=(
            "Per-unit context-coverage score: the maximum reverse-context "
            "similarity any response token had to this support unit (range "
            "[0, 1]). Independent of argmax attribution — a unit can have "
            "high coverage yet matched_response_tokens=0 when sibling units "
            "scored even higher. Use this with the global "
            "scores.context_coverage_threshold to decide whether the unit "
            "actually contributed to the response."
        ),
    )
    used: bool = Field(
        default=False,
        description=(
            "True when coverage_score >= scores.context_coverage_threshold. "
            "Compatibility coverage view only. Retrieval-efficiency "
            "observability: units with used=False were fetched by the "
            "retriever but contributed nothing strong to the response. Use "
            "``usage_state`` for the newer precision-first tri-state contract."
        ),
    )
    usage_state: GroundednessUsageState = Field(
        default=GroundednessUsageState.UNCERTAIN,
        description=(
            "Precision-first tri-state usage label. ``unused`` is emitted only "
            "for high-confidence negatives; borderline cases fall into "
            "``uncertain`` instead of forcing a binary verdict."
        ),
    )
    usage_confidence: Optional[float] = Field(
        default=None,
        description=(
            "Confidence in the emitted ``usage_state``. Range ``[0, 1]``."
        ),
    )
    unused_confidence: Optional[float] = Field(
        default=None,
        description=(
            "Confidence that this support unit is truly unused. Especially "
            "useful for sorting or filtering units with "
            "``usage_state = unused``."
        ),
    )
    context_trust_state: Optional[Literal["trusted", "suspicious", "blocked"]] = Field(
        default=None,
        description=(
            "Instruction-contamination state for this support unit. Diagnostic "
            "only; high risk is surfaced to runtime_decision rather than "
            "lowering groundedness_v2 directly."
        ),
    )
    context_trust_score: Optional[float] = Field(
        default=None,
        description="Per-support-unit context-trust risk in [0, 1].",
    )
    context_trust_labels: List[ContextTrustLabel] = Field(default_factory=list)
    context_trust_spans: List[ContextTrustSpan] = Field(default_factory=list)
    source_id: Optional[str] = Field(
        default=None,
        description="Echoed from the matching support_units[] request entry, when supplied.",
    )
    speaker: Optional[str] = Field(
        default=None,
        description="Echoed from the matching support_units[] request entry, when supplied.",
    )
    timestamp: Optional[str] = Field(
        default=None,
        description="Echoed from the matching support_units[] request entry, when supplied.",
    )
    metadata: Optional[dict] = Field(
        default=None,
        description="Echoed verbatim from the matching support_units[] request entry, when supplied.",
    )


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


class HeatmapToken(BaseModel):
    """Per-response-token visualisation record.

    ``band`` is one of ``"green"`` / ``"amber"`` / ``"red"`` and maps
    directly to a CSS class name in the bundled HTML template.
    """

    index: int
    token: str
    score: float
    band: str


class HeatmapFile(BaseModel):
    """Per-file visualisation record."""

    path: str
    owner_share: float
    band: str
    reason_codes: List[str] = Field(default_factory=list)
    dead_weight: bool = False


class HeatmapSummary(BaseModel):
    """Headline numbers rendered at the top of the heatmap card."""

    headline_score: Optional[float] = None
    risk_band: Optional[str] = None
    recommendation: Optional[str] = None
    groundedness_pct: Optional[float] = None
    dead_weight_pct: Optional[float] = None
    reason_code_histogram: Dict[str, int] = Field(default_factory=dict)


class HeatmapThresholds(BaseModel):
    """Exact cut-offs used to bucket tokens / files into colour bands.

    Exposed on the response so callers can reproduce the exact
    bucketing server-side or explain it in their UI.
    """

    token_green_min: float = 0.60
    token_amber_min: float = 0.35
    file_green_min: float = 0.20
    file_amber_min: float = 0.05


class HeatmapPayload(BaseModel):
    """Compact visualisation bundle ready for copy-paste HTML rendering.

    See ``docs/heatmap.md`` for the reference template and CSS.
    """

    summary: HeatmapSummary = Field(default_factory=HeatmapSummary)
    tokens: List[HeatmapToken] = Field(default_factory=list)
    files: List[HeatmapFile] = Field(default_factory=list)
    thresholds: HeatmapThresholds = Field(default_factory=HeatmapThresholds)


class GroundednessResponse(BaseModel):
    """Groundedness scoring response with heatmap-ready sparse data."""

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
                    "support_units_usage_used": 1,
                    "support_units_unused": 0,
                    "support_units_uncertain": 0,
                    "context_usage_ratio": 1.0,
                    "context_unused_ratio": 0.0,
                    "context_uncertain_ratio": 0.0,
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
                        "coverage_score": 0.99,
                        "used": True,
                        "usage_state": "used",
                        "usage_confidence": 0.98,
                        "unused_confidence": 0.01,
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
    context_trust_diagnostics: Optional[ContextTrustDiagnostics] = None
    semantic_entropy_diagnostics: Optional[GroundednessSemanticEntropyDiagnostics] = None
    structured_diagnostics: Optional[GroundednessStructuredDiagnostics] = None
    code_lane_diagnostics: Optional[CodeLaneDiagnostics] = Field(
        default=None,
        description=(
            "Populated when ``scoring_mode == 'code'``. Carries every "
            "code-lane signal: composite score, AST drift, literal novelty, "
            "NLI cascade, and per-file / per-unit attribution with reason "
            "codes. RAG-lane responses leave this field ``None``."
        ),
    )
    file_attribution: Optional[FileAttributionDiagnostics] = Field(
        default=None,
        description=(
            "Lane-neutral per-file attribution with reason codes and a "
            "reason-code histogram. Populated for both ``scoring_mode == "
            "'code'`` and ``scoring_mode == 'rag'``. For the code lane "
            "this echoes ``code_lane_diagnostics.file_attribution`` so "
            "downstream dashboards have one canonical place to read from; "
            "for the RAG lane it is the primary surface (chunks are "
            "grouped by ``metadata.path`` / ``metadata.source`` / "
            "``source_id``, falling back to ``support_id``)."
        ),
    )
    heatmap: Optional["HeatmapPayload"] = Field(
        default=None,
        description=(
            "Compact visual summary of the turn: per-token band, "
            "per-file band, and headline numbers. Emitted when the "
            "request ``heatmap_format`` is ``'data'`` (default) or "
            "``'html'``. Renders copy-paste into any HTML page using "
            "the template documented in ``docs/heatmap.md``."
        ),
    )
    heatmap_html: Optional[str] = Field(
        default=None,
        description=(
            "Self-contained HTML fragment (a single ``<div>`` with "
            "inline CSS) that renders the ``heatmap`` payload without "
            "any frontend work. Emitted only when the request "
            "``heatmap_format`` is ``'html'``."
        ),
    )
    time_ms: float
    scoring_mode: ScoringMode = Field(
        default=ScoringMode.RAG,
        description="Echo of the request scoring_mode so callers can tell which lane ran.",
    )
    profile: Optional[TraceRuntimeProfile] = Field(
        default=None,
        description="Echo of the requested hosted Trace profile when supplied.",
    )
    effective_profile: TraceRuntimeProfile = Field(
        default=TraceRuntimeProfile.STANDARD,
        description="Runtime profile that actually configured this scoring request.",
    )
    profile_diagnostics: Dict[str, object] = Field(
        default_factory=dict,
        description="Non-PII diagnostics proving which profile features were requested and available.",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Echo of the request session_id when supplied; used by IDE plugins for turn-event correlation.",
    )
    attribution_mode: Optional[AttributionMode] = Field(
        default=None,
        description="Echo of the request attribution_mode for downstream auditing.",
    )
    corpus_route: Optional["CorpusRouteDiagnostics"] = Field(
        default=None,
        description=(
            "Populated when the corpus router is enabled. Records the "
            "inferred (or explicitly declared) corpus type, the source "
            "of the decision (``classifier`` vs ``explicit``), the "
            "classifier latency, and the calibrated (fusion_weights, "
            "thresholds) bundle actually applied by the scoring service."
        ),
    )
    amber_escalation: Optional[AmberEscalationDiagnostics] = Field(
        default=None,
        description=(
            "Populated when the request opted into ``auto_decide`` and the "
            "initial band was ``amber``. Records the pinned LLM judge's "
            "verdict, reasoning, latency, and cost so the caller can audit "
            "the zero-human-in-the-loop path."
        ),
    )
    runtime_decision: Optional[RuntimeDecisionRecord] = Field(
        default=None,
        description=(
            "Stable automatic decision record emitted when "
            "LATENCE_TRACE_RUNTIME_DECISION_ENABLED is set. Contains the "
            "allow / auto_repair / block action, policy version, evidence, "
            "unsupported spans, and rollback-safe thresholds."
        ),
    )
    runtime_head_features: Optional[Dict[str, float]] = Field(
        default=None,
        description="Echoed runtime-head features when supplied by the caller.",
    )
    runtime_feature_source: Optional[str] = Field(
        default=None,
        description=(
            "Source of runtime-head features used for the automatic decision: "
            "'client', 'synthesized', 'partial', or 'missing'."
        ),
    )
    runtime_feature_missing_groups: List[str] = Field(
        default_factory=list,
        description="Feature/evidence groups that were unavailable for server-side synthesis.",
    )
    reason: Optional[str] = Field(
        default=None,
        description=(
            "Machine-readable reason code for non-scored responses. "
            "Currently emitted: 'no_premise_supplied' (closed_book + zero premises) and "
            "'open_domain_pending_v1_next' (open_domain not yet wired in v1)."
        ),
    )
    next_session_state: Optional[SessionStatePayload] = Field(
        default=None,
        description=(
            "Updated caller-portable session blob. Populated when the "
            "caller either supplied ``session_state`` or a ``session_id`` "
            "on the request — opt-in in both directions. Echo this "
            "verbatim as ``session_state`` on the next turn; the server "
            "never persists it."
        ),
    )
    session_signals: Optional[SessionSignals] = Field(
        default=None,
        description=(
            "Derived session-level signals (drift z-score, EMA "
            "groundedness, dead-file candidates, recommendation). "
            "Populated alongside ``next_session_state``. Purely "
            "advisory — every per-turn metric remains available in "
            "``scores`` and ``code_lane_diagnostics``."
        ),
    )
    next_memory_state: Optional[MemoryState] = Field(
        default=None,
        description="Updated caller-carried InfiniMem state when memory shadow mode is enabled.",
    )
    hot_context_preview: Optional[str] = Field(
        default=None,
        description="Budgeted hot memory context preview. Never applied to scoring by default.",
    )
    memory_diagnostics: Optional[MemoryDiagnostics] = Field(
        default=None,
        description="Explainable span survival, dedup, demotion, and budget diagnostics.",
    )


# ``GroundednessRequest.session_state`` uses a forward reference to
# ``SessionStatePayload`` (defined further down for schema readability);
# rebuild the model so pydantic resolves the annotation.
GroundednessRequest.model_rebuild()
# ``GroundednessResponse.heatmap`` is a forward reference to
# ``HeatmapPayload`` defined below the response model; rebuild once the
# target class exists so pydantic resolves the annotation.
GroundednessResponse.model_rebuild()


# ---------------------------------------------------------------------------
# Rollup (stateless session-level aggregation)
# ---------------------------------------------------------------------------


class RollupTurnInput(BaseModel):
    """Compact record for one turn in a rollup request.

    Every field is optional so callers can feed the exact slice of the
    per-turn response they already have. The rollup transform skips
    anything it does not recognise; garbage in -> sensible zeros out.
    """

    scores: Optional[Dict[str, Optional[float]]] = Field(
        default=None,
        description=(
            "Subset of ``GroundednessScores`` fields. At minimum the "
            "transform reads ``groundedness_v2`` / "
            "``composite_phantom_score`` / ``dead_weight_ratio`` / "
            "``dead_weight_file_count``."
        ),
    )
    session_signals: Optional[SessionSignals] = None
    file_attribution: Optional[FileAttributionDiagnostics] = None
    risk_band: Optional[str] = None
    recommendation: Optional[str] = None
    timestamp: Optional[str] = None


class DriftTrend(BaseModel):
    """Tiny summary of the drift-z-score trajectory."""

    min: float = 0.0
    max: float = 0.0
    mean: float = 0.0
    last: float = 0.0


class RollupTopDeadFile(BaseModel):
    path: str
    dead_turns: int = 0
    ema_owner_share: float = 0.0


class RollupRequest(BaseModel):
    """Stateless aggregation over a sequence of per-turn records.

    The server does not persist anything; the caller owns the list of
    turns. Heavy fields like chunk-level ownership are NOT required —
    this endpoint exists to let IDE plugins / dashboards roll up a
    conversation into business-grade metrics with one HTTP call.
    """

    turns: List[RollupTurnInput] = Field(default_factory=list)
    session_id: Optional[str] = None
    heatmap_format: Literal["none", "data", "html"] = Field(
        default="none",
        description=(
            "When ``'data'`` or ``'html'`` the rollup also emits a "
            "conversation-level heatmap (session signals + top dead "
            "files + reason-code histogram) ready to render."
        ),
    )


class RollupResponse(BaseModel):
    """Session-level aggregates emitted by :meth:`GroundednessService.rollup`.

    Every percentage is bounded to ``[0, 1]``. The full per-turn trails
    are returned so clients can draw sparklines without hitting the
    service again.
    """

    turns: int = 0
    noise_pct: float = 0.0
    model_drift_pct: float = 0.0
    retrieval_waste_pct: float = 0.0
    reason_code_histogram: Dict[str, int] = Field(default_factory=dict)
    recommendations: List[str] = Field(default_factory=list)
    risk_band_trail: List[str] = Field(default_factory=list)
    drift_trend: DriftTrend = Field(default_factory=DriftTrend)
    top_dead_files: List[RollupTopDeadFile] = Field(default_factory=list)
    session_id: Optional[str] = None
    heatmap: Optional[HeatmapPayload] = None
    heatmap_html: Optional[str] = None
