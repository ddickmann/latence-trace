"""Code-lane groundedness package.

Public surface for the scoring mode ``code`` shipped alongside the
original RAG lane. Everything in this package is importable and
stateless so callers can embed it in long-running workers, FastAPI
handlers, or offline validators interchangeably.

The lane is built out of small cooperating modules:

- :mod:`.gpu_scorer` — GPU-resident MaxSim scorer producing per-token
  stats, per-unit ownership counts, and literal guard.
- :mod:`.ast_grounding` — multi-language tree-sitter symbol extractor
  that flags drift and phantom imports in fenced code blocks.
- :mod:`.nli_cascade` — ambiguity-gated NLI fallback that reuses the
  existing :mod:`latence_trace.core.nli` primitives.
- :mod:`.literal_novelty` — strengthens the literal guard by scoring
  *where* each identifier first matched.
- :mod:`.composite` — linear + logistic-regression calibrated
  phantom-guard score.
- :mod:`.file_attribution` — per-file and per-unit ownership rollup
  with reason codes and query-aware attribution.
- :mod:`.orchestrator` — ``score_code_groundedness`` entry point.

The lane reuses — never re-implements — the shared primitives in
:mod:`latence_trace.core.nli` (NLI provider, ``verify_claims``,
``aggregate_nli_score``) and :mod:`latence_trace.core.semantic_entropy`
(Farquhar-style SE on caller-supplied samples).
"""

from __future__ import annotations

from .ast_grounding import (
    AstGroundingResult,
    AstSymbolExtractor,
    SUPPORTED_LANGUAGES,
    detect_language,
    extract_symbols,
    extract_symbols_async,
)
from .composite import (
    CompositeFeatures,
    CompositeResult,
    LinearComposite,
    LogisticComposite,
    default_composite,
)
from .file_attribution import (
    FileAttributionResult,
    PerFileUsage,
    PerUnitOwnership,
    ReasonCode,
    attribute_files,
)
from .gpu_scorer import GPUScorer, ScorerOutput, UsageThresholds
from .literal_novelty import LiteralNoveltyResult, compute_literal_novelty
from .nli_cascade import NLICascade, NLICascadeResult
from .orchestrator import CodeLaneConfig, CodeLaneResult, score_code_groundedness
from .session import (
    DEFAULT_EMA_HALF_LIFE_TURNS,
    DEFAULT_GROUNDEDNESS_EMA_HALF_LIFE,
    DEFAULT_VERDICT_WINDOW,
    MAX_TRACKED_FILES,
    SESSION_STATE_SCHEMA_VERSION,
    FileSessionStats,
    Recommendation,
    RollingStats,
    SessionSignals,
    SessionState,
    TurnMetrics,
    file_attribution_to_turn_inputs,
    update_session_state,
)
from .types import SupportUnitPack

__all__ = [
    "AstGroundingResult",
    "AstSymbolExtractor",
    "CodeLaneConfig",
    "CodeLaneResult",
    "CompositeFeatures",
    "CompositeResult",
    "DEFAULT_EMA_HALF_LIFE_TURNS",
    "DEFAULT_GROUNDEDNESS_EMA_HALF_LIFE",
    "DEFAULT_VERDICT_WINDOW",
    "FileAttributionResult",
    "FileSessionStats",
    "GPUScorer",
    "LinearComposite",
    "LiteralNoveltyResult",
    "LogisticComposite",
    "MAX_TRACKED_FILES",
    "NLICascade",
    "NLICascadeResult",
    "PerFileUsage",
    "PerUnitOwnership",
    "ReasonCode",
    "Recommendation",
    "RollingStats",
    "SESSION_STATE_SCHEMA_VERSION",
    "SUPPORTED_LANGUAGES",
    "ScorerOutput",
    "SessionSignals",
    "SessionState",
    "SupportUnitPack",
    "TurnMetrics",
    "UsageThresholds",
    "attribute_files",
    "compute_literal_novelty",
    "default_composite",
    "detect_language",
    "extract_symbols",
    "extract_symbols_async",
    "file_attribution_to_turn_inputs",
    "score_code_groundedness",
    "update_session_state",
]
