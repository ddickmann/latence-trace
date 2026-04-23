"""Top-level entry point for ``scoring_mode == "code"``.

The orchestrator takes *already-encoded* support units and response
embeddings (produced by the RAG lane's encoder in the service layer)
and layers the code-specific signals on top: GPU MaxSim + AST drift +
literal novelty + NLI cascade + semantic entropy + composite score +
file-level attribution with reason codes.

Design principles
-----------------
- **Pure compute** — no I/O other than the NLI provider call. The
  function is safe to call from a thread pool.
- **Composable** — :class:`CodeLaneConfig` toggles every feature
  independently so the ablation harness can peel them back.
- **Low latency** — the MaxSim + literal + per-token stats run on GPU
  in one pass; AST parsing runs in a background thread via
  ``asyncio.to_thread`` when called from the async handler; NLI only
  fires when ambiguous.
- **Zero breaking changes** — the RAG lane is untouched; the code lane
  emits a strict superset of diagnostics the response model already
  understands plus a new :class:`CodeLaneResult` for the code-only
  fields.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from latence_trace.core.semantic_entropy import (
    SemanticEntropyResult,
    compute_semantic_entropy,
)

from .ast_grounding import (
    AstGroundingResult,
    AstSymbolExtractor,
    extract_symbols,
)
from .composite import (
    CompositeFeatures,
    CompositeResult,
    LinearComposite,
    LogisticComposite,
    default_composite,
)
from .file_attribution import FileAttributionResult, attribute_files
from .gpu_scorer import GPUScorer, ScorerOutput, get_default_scorer
from .literal_novelty import LiteralNoveltyResult, compute_literal_novelty
from .nli_cascade import NLICascade, NLICascadeResult
from .types import SupportUnitPack

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class CodeLaneConfig:
    """Feature toggles for the code lane.

    Defaults ship the full stack; the ablation harness flips features
    off one at a time to measure per-signal lift.
    """

    enable_ast: bool = True
    enable_literal_novelty: bool = True
    enable_nli_cascade: bool = True
    enable_semantic_entropy: bool = True
    nli_lower_band: float = 0.65
    nli_upper_band: float = 0.90
    min_owner_share: float = 0.01
    low_cosine_threshold: float = 0.40
    coverage_threshold: float = 0.20
    response_language_hint: Optional[str] = None
    use_logistic_composite: bool = True

    def as_dict(self) -> Dict[str, Any]:
        return {
            "enable_ast": bool(self.enable_ast),
            "enable_literal_novelty": bool(self.enable_literal_novelty),
            "enable_nli_cascade": bool(self.enable_nli_cascade),
            "enable_semantic_entropy": bool(self.enable_semantic_entropy),
            "nli_lower_band": float(self.nli_lower_band),
            "nli_upper_band": float(self.nli_upper_band),
            "min_owner_share": float(self.min_owner_share),
            "low_cosine_threshold": float(self.low_cosine_threshold),
            "coverage_threshold": float(self.coverage_threshold),
            "response_language_hint": self.response_language_hint,
            "use_logistic_composite": bool(self.use_logistic_composite),
        }


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass
class CodeLaneResult:
    """Bundle returned by :func:`score_code_groundedness`.

    Every component ``result`` is also exposed as ``as_dict`` for
    observability — downstream callers only need to stringify the full
    bundle once.
    """

    scorer: ScorerOutput
    composite: CompositeResult
    ast: Optional[AstGroundingResult]
    literal_novelty: Optional[LiteralNoveltyResult]
    nli_cascade: Optional[NLICascadeResult]
    semantic_entropy: Optional[SemanticEntropyResult]
    file_attribution: FileAttributionResult
    config: CodeLaneConfig
    total_latency_ms: float
    component_latency_ms: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "scorer": self.scorer.as_dict(),
            "composite": self.composite.as_dict(),
            "ast": None
            if self.ast is None
            else {
                "language": self.ast.language,
                "parser_backend": self.ast.parser_backend,
                "ast_literal_drift_count": self.ast.ast_literal_drift_count,
                "ast_phantom_symbol_count": self.ast.ast_phantom_symbol_count,
                "ast_phantom_verdict": self.ast.ast_phantom_verdict,
                "drift_symbols": list(self.ast.drift_symbols),
                "phantom_symbols": list(self.ast.phantom_symbols),
                "latency_ms": self.ast.latency_ms,
            },
            "literal_novelty": None
            if self.literal_novelty is None
            else self.literal_novelty.as_dict(),
            "nli_cascade": None
            if self.nli_cascade is None
            else self.nli_cascade.as_dict(),
            "semantic_entropy": None
            if self.semantic_entropy is None
            else {
                "aggregate_score": self.semantic_entropy.aggregate,
                "entropy_raw": self.semantic_entropy.entropy_raw,
                "cluster_count": int(self.semantic_entropy.cluster_count),
                "sample_count": int(self.semantic_entropy.sample_count),
                "skipped_reason": self.semantic_entropy.skipped_reason,
                "latency_ms": float(self.semantic_entropy.latency_ms),
            },
            "file_attribution": self.file_attribution.as_dict(),
            "config": self.config.as_dict(),
            "total_latency_ms": float(self.total_latency_ms),
            "component_latency_ms": {
                k: float(v) for k, v in self.component_latency_ms.items()
            },
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect_literal_tokens(pack: SupportUnitPack) -> List[str]:
    return [str(t).lower() for t in pack.tokens if t]


def _extract_response_literals(
    response_tokens: Sequence[str],
    min_len: int = 3,
) -> List[str]:
    """Pull identifier-like tokens out of the response token list.

    ColBERT tokenisers produce sub-word pieces prefixed with ``Ġ`` or
    ``▁``; we strip those and keep only alphanumeric pieces of length
    >= ``min_len`` so common stopwords don't swamp the novelty guard.
    """
    from latence_trace.core.groundedness import _strip_marker

    literals: List[str] = []
    seen: set[str] = set()
    for token in response_tokens:
        if not token:
            continue
        stripped = _strip_marker(token).lower()
        if len(stripped) < min_len:
            continue
        if not (stripped.isalnum() or "_" in stripped):
            continue
        if stripped in seen:
            continue
        seen.add(stripped)
        literals.append(stripped)
    return literals


def _units_as_nli_shim(packs: Sequence[SupportUnitPack]) -> List[Any]:
    """Adapt :class:`SupportUnitPack` to the NLI verifier contract.

    :func:`latence_trace.core.nli.verify_claims` reads ``.text`` and
    ``.support_id`` off the support-unit objects. We build a minimal
    namespace object per pack that exposes both.
    """

    class _Unit:
        __slots__ = ("text", "support_id")

        def __init__(self, text: str, support_id: str) -> None:
            self.text = text
            self.support_id = support_id

    shims: List[Any] = []
    for idx, pack in enumerate(packs):
        text = " ".join(pack.tokens) if pack.tokens else ""
        metadata = pack.metadata if isinstance(pack.metadata, Mapping) else {}
        raw_text = metadata.get("text") if metadata else None
        shims.append(_Unit(text=str(raw_text or text), support_id=pack.support_id))
    return shims


def _context_texts_for_ast(packs: Sequence[SupportUnitPack]) -> List[str]:
    texts: List[str] = []
    for pack in packs:
        metadata = pack.metadata if isinstance(pack.metadata, Mapping) else {}
        raw_text = metadata.get("text") if metadata else None
        if raw_text:
            texts.append(str(raw_text))
        elif pack.tokens:
            texts.append(" ".join(pack.tokens))
    return texts


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def score_code_groundedness(
    *,
    response_text: str,
    response_tokens: Sequence[str],
    response_embeddings: torch.Tensor,
    support_units: Sequence[SupportUnitPack],
    query_text: Optional[str] = None,
    query_embeddings: Optional[torch.Tensor] = None,
    verification_samples: Optional[Sequence[str]] = None,
    config: Optional[CodeLaneConfig] = None,
    scorer: Optional[GPUScorer] = None,
    composite: Optional[Any] = None,
    ast_extractor: Optional[AstSymbolExtractor] = None,
    nli_cascade: Optional[NLICascade] = None,
    nli_provider: Optional[Any] = None,
) -> CodeLaneResult:
    """Score one turn through the code lane.

    Parameters mirror the request + service-layer helpers so the RunPod
    handler can call this with the same artefacts it already builds for
    the RAG lane.

    Either ``scorer`` or the module-level singleton is used; passing an
    instance in lets tests pin the CPU device.
    """
    start = time.perf_counter()
    config = config or CodeLaneConfig()
    scorer = scorer or get_default_scorer()
    composite = composite or default_composite()
    component_latency: Dict[str, float] = {}

    # -- 1. GPU MaxSim pass --------------------------------------------
    literal_tokens: List[str] = []
    for pack in support_units:
        literal_tokens.extend(_collect_literal_tokens(pack))
    support_pairs: List[Tuple[Sequence[str], torch.Tensor]] = [
        (pack.tokens, pack.embeddings) for pack in support_units
    ]
    scorer_start = time.perf_counter()
    scorer_output = scorer(
        response_tokens=list(response_tokens),
        response_embeddings=response_embeddings,
        support_units=support_pairs,
        literal_tokens=literal_tokens,
        query_embeddings=query_embeddings,
    )
    component_latency["scorer_ms"] = (time.perf_counter() - scorer_start) * 1000.0

    # -- 2. Literal novelty ---------------------------------------------
    literal_novelty_result: Optional[LiteralNoveltyResult] = None
    if config.enable_literal_novelty:
        ln_start = time.perf_counter()
        response_literals = _extract_response_literals(response_tokens)
        literal_novelty_result = compute_literal_novelty(
            response_literals=response_literals,
            support_units=support_units,
            per_unit_score=scorer_output.per_unit_max,
        )
        component_latency["literal_novelty_ms"] = (
            time.perf_counter() - ln_start
        ) * 1000.0

    # -- 3. AST grounding ----------------------------------------------
    ast_result: Optional[AstGroundingResult] = None
    if config.enable_ast:
        ast_start = time.perf_counter()
        ast_result = extract_symbols(
            response_text=response_text,
            context_texts=_context_texts_for_ast(support_units),
            extractor=ast_extractor,
            language_hint=config.response_language_hint,
        )
        component_latency["ast_ms"] = (time.perf_counter() - ast_start) * 1000.0

    # -- 4. Composite (pre-NLI) -----------------------------------------
    features = CompositeFeatures(
        reverse_context=scorer_output.reverse_context,
        per_token_p10=scorer_output.per_token_p10,
        literal_guard=scorer_output.literal_guard,
        literal_novelty_min=(
            literal_novelty_result.literal_novelty_min
            if literal_novelty_result is not None
            else 1.0
        ),
        ast_phantom_symbol_count=(ast_result.ast_phantom_symbol_count if ast_result else 0),
        ast_literal_drift_count=(ast_result.ast_literal_drift_count if ast_result else 0),
        nli_contradiction_prob_max=0.0,
        semantic_entropy_aggregate=None,
    )
    pre_nli_start = time.perf_counter()
    pre_nli_composite = composite.score(features)
    composite_ms_acc = (time.perf_counter() - pre_nli_start) * 1000.0

    # -- 5. NLI cascade (only when ambiguous) ---------------------------
    nli_result: Optional[NLICascadeResult] = None
    effective_nli_cascade = nli_cascade
    if (
        effective_nli_cascade is None
        and config.enable_nli_cascade
        and nli_provider is not None
    ):
        effective_nli_cascade = NLICascade(
            nli_provider,
            lower_band=config.nli_lower_band,
            upper_band=config.nli_upper_band,
        )
    if config.enable_nli_cascade and effective_nli_cascade is not None:
        nli_start = time.perf_counter()
        nli_result = effective_nli_cascade.run(
            response_text=response_text,
            support_units=_units_as_nli_shim(support_units),
            composite_score=pre_nli_composite.composite_score,
        )
        component_latency["nli_ms"] = (time.perf_counter() - nli_start) * 1000.0
        features = CompositeFeatures(
            reverse_context=features.reverse_context,
            per_token_p10=features.per_token_p10,
            literal_guard=features.literal_guard,
            literal_novelty_min=features.literal_novelty_min,
            ast_phantom_symbol_count=features.ast_phantom_symbol_count,
            ast_literal_drift_count=features.ast_literal_drift_count,
            nli_contradiction_prob_max=nli_result.nli_contradiction_prob_max,
            semantic_entropy_aggregate=features.semantic_entropy_aggregate,
        )

    # -- 6. Semantic entropy (opt-in, needs samples) --------------------
    se_result: Optional[SemanticEntropyResult] = None
    if (
        config.enable_semantic_entropy
        and verification_samples
        and len(verification_samples) >= 2
        and (nli_provider is not None or effective_nli_cascade is not None)
    ):
        se_start = time.perf_counter()
        se_provider = nli_provider
        if se_provider is None and effective_nli_cascade is not None:
            se_provider = effective_nli_cascade.provider
        try:
            se_result = compute_semantic_entropy(
                verification_samples, se_provider  # type: ignore[arg-type]
            )
        except Exception as exc:
            logger.warning("semantic_entropy_failed", extra={"error": str(exc)})
            se_result = None
        component_latency["semantic_entropy_ms"] = (
            time.perf_counter() - se_start
        ) * 1000.0
        if se_result is not None and se_result.aggregate is not None:
            features = CompositeFeatures(
                reverse_context=features.reverse_context,
                per_token_p10=features.per_token_p10,
                literal_guard=features.literal_guard,
                literal_novelty_min=features.literal_novelty_min,
                ast_phantom_symbol_count=features.ast_phantom_symbol_count,
                ast_literal_drift_count=features.ast_literal_drift_count,
                nli_contradiction_prob_max=features.nli_contradiction_prob_max,
                semantic_entropy_aggregate=se_result.aggregate,
            )

    # -- 7. Final composite with all features --------------------------
    final_start = time.perf_counter()
    final_composite = composite.score(features)
    composite_ms_acc += (time.perf_counter() - final_start) * 1000.0
    component_latency["composite_ms"] = composite_ms_acc

    # -- 8. File attribution -------------------------------------------
    fa_start = time.perf_counter()
    n_query_tokens = int(query_embeddings.shape[0]) if query_embeddings is not None else 0
    file_attribution = attribute_files(
        units=support_units,
        per_unit_max=scorer_output.per_unit_max,
        per_unit_owner_count=scorer_output.per_unit_owner_count,
        per_unit_query_owner_count=scorer_output.per_unit_query_owner_count,
        usage_states=scorer_output.usage_states,
        n_response_tokens=int(scorer_output.n_tokens),
        n_query_tokens=n_query_tokens,
        min_owner_share=config.min_owner_share,
        coverage_threshold=config.coverage_threshold,
        low_cosine_threshold=config.low_cosine_threshold,
    )
    component_latency["file_attribution_ms"] = (time.perf_counter() - fa_start) * 1000.0

    total_latency_ms = (time.perf_counter() - start) * 1000.0

    return CodeLaneResult(
        scorer=scorer_output,
        composite=final_composite,
        ast=ast_result,
        literal_novelty=literal_novelty_result,
        nli_cascade=nli_result,
        semantic_entropy=se_result,
        file_attribution=file_attribution,
        config=config,
        total_latency_ms=total_latency_ms,
        component_latency_ms=component_latency,
    )
