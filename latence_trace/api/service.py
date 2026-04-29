"""Standalone groundedness service for latence-trace.

This module replaces the multi-collection groundedness layer that used to live
inside :mod:`voyager_index._internal.server.api.service` with a single,
self-contained ``GroundednessService`` class. The standalone product does not
own a vector store, so the ``chunk_ids`` fast path delegates vector lookup to
a caller-supplied resolver callback. The ``raw_context`` path is fully
self-contained: text is segmented into token-budgeted support windows and
re-encoded on demand by the configured encoder provider.

The scoring logic itself is unchanged from the upstream Phase J implementation
in ``latence_trace.core.groundedness`` (which was ported verbatim from
voyager-index). Only the surrounding orchestration was extracted here so the
service can run as a standalone FastAPI process or be embedded in another
host application.

In addition to the runtime classes the module exposes :data:`PROFILE_NAMES`
and :func:`apply_profile`, which materialise the three Pareto-optimal default
configurations (``fast``, ``balanced``, ``quality``) selected by the offline
profile sweep documented in
``research/triangular_maxsim/reports/profile_pareto.md``. Profile presets are
expressed as environment-variable overlays so they compose cleanly with the
existing env-driven configuration surface and so any operator-set environment
variable always wins (the loader never overwrites an explicit override).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from latence_trace.api.models import (
    AmberEscalationDiagnostics,
    AttributionMode,
    CodeLaneAstDiagnostics,
    CodeLaneCompositeContributions,
    CodeLaneCompositeDiagnostics,
    CodeLaneDiagnostics,
    CodeLaneFileAttribution,
    CodeLaneLiteralNovelty,
    CodeLaneNLICascade,
    CodeLanePerFileUsage,
    CodeLanePerUnitOwnership,
    CollectionKind,
    DriftTrend,
    FileAttributionDiagnostics,
    FileSessionStatsPayload,
    GroundednessEligibility,
    GroundednessRequest,
    GroundednessResponse,
    GroundednessScores,
    HeatmapPayload,
    RollingStatsPayload,
    RollupRequest,
    RollupResponse,
    RollupTopDeadFile,
    RollupTurnInput,
    ScoringMode,
    SessionSignals as SessionSignalsPayload,
    SessionStatePayload,
    TraceRuntimeProfile,
)
from latence_trace.api.heatmap import build_heatmap, render_heatmap_html
from latence_trace.api.rollup import aggregate_turns
from latence_trace.middleware.amber_escalation import (
    AmberEscalationConfig,
    build_payload as build_amber_payload,
    escalate as run_amber_escalate,
)
from latence_trace.core.groundedness import (
    SupportUnitInput,
    _build_null_bank_pack,
    _build_response_chunks,
    default_null_bank_texts,
    encode_texts,
    partition_support_units,
    provider_token_limit,
    score_groundedness_response_chunked,
    segment_text,
    split_raw_context_by_file_headers,
    tokenize_text,
)
from latence_trace.core.code_lane import (
    AstSymbolExtractor,
    CodeLaneConfig,
    CodeLaneResult,
    GPUScorer,
    NLICascade,
    SESSION_STATE_SCHEMA_VERSION,
    SessionSignals as SessionSignalsData,
    SessionState as SessionStateData,
    SupportUnitPack,
    TurnMetrics,
    file_attribution_to_turn_inputs,
    score_code_groundedness,
    update_session_state,
)
from latence_trace.core.nli import (
    CrossEncoderPremiseReranker,
    HuggingFaceNLIProvider,
    default_max_batch as nli_default_max_batch,
    default_max_claims as nli_default_max_claims,
    default_max_latency_ms as nli_default_max_latency_ms,
    default_premise_concat_word_budget as nli_default_premise_concat_word_budget,
    default_top_k_premises as nli_default_top_k_premises,
    fusion_weights_from_env as nli_fusion_weights_from_env,
    is_atomic_enabled as nli_is_atomic_enabled,
    is_enabled as nli_is_enabled,
    is_premise_concat_enabled as nli_is_premise_concat_enabled,
    resolve_default_provider as nli_resolve_default_provider,
    resolve_default_reranker as nli_resolve_default_reranker,
)
from latence_trace.core.thresholds import RiskBandPolicy, load_risk_band_policy

import contextvars

from latence_trace.middleware import corpus_router as _corpus_router_middleware

logger = logging.getLogger(__name__)


# Request-scoped corpus-router decision. Populated at the top of
# :meth:`groundedness()` before we dispatch to ``_score_rag`` /
# ``_score_code`` so ``resolve_request_runtime_profile`` can layer the
# class-specific fusion weights + thresholds on top of the hosted
# profile preset without changing any callsite signatures.
_ACTIVE_ROUTE_DECISION: "contextvars.ContextVar[Optional[_corpus_router_middleware.CorpusRouteDecision]]" = (
    contextvars.ContextVar("_ACTIVE_ROUTE_DECISION", default=None)
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ServiceError(Exception):
    """Base service error.

    All service-layer exceptions carry a ``status_code`` (HTTP), an
    ``error_code`` (stable machine-readable identifier), and a ``hint``
    (actionable next step for the caller / AI agent). The structured
    error envelope built by :func:`latence_trace.api.routes._raise_service_error`
    surfaces all three so consumers never have to scrape the message
    string.
    """

    status_code = 500
    error_code = "service_error"
    hint = "An unexpected error occurred. Inspect logs and retry."

    def __init__(self, message: str, *, hint: Optional[str] = None) -> None:
        super().__init__(message)
        if hint is not None:
            # Per-instance override wins over the class default. Useful
            # for context-specific guidance ("supply a chunk_resolver",
            # "raw_context produced no support windows", ...).
            self.hint = hint


class ValidationError(ServiceError):
    status_code = 400
    error_code = "validation_error"
    hint = (
        "Check the request shape against /agent-help or the OpenAPI schema; "
        "every field is documented with the value range it accepts."
    )


class NotFoundError(ServiceError):
    status_code = 404
    error_code = "not_found"
    hint = (
        "The referenced chunk_id / model is not registered with this service. "
        "Confirm the chunk_resolver is wired or that the model is loaded."
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# Pareto-optimal default profiles
# ---------------------------------------------------------------------------


PROFILE_NAMES: Tuple[str, ...] = ("fast", "balanced", "quality")

DEFAULT_PROFILE: str = "balanced"

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _profile_data_path(filename: str) -> str:
    return str(_DATA_DIR / filename)


# Each preset captures the environment overlay the standalone server should
# install when the corresponding profile is selected. Values are intentionally
# strings so they can be merged into ``os.environ`` without coercion.
#
# The presets reflect the per-profile sweep winners documented in
# ``research/triangular_maxsim/reports/profile_pareto.md`` and the per-profile
# threshold + fusion-weight artefacts under ``latence_trace/data/``. Operators
# that need a different mix can simply export the underlying environment
# variables before launching the server - :func:`apply_profile` never
# overwrites a value that is already present in ``os.environ``.
PROFILE_ENV_PRESETS: Dict[str, Dict[str, str]] = {
    "fast": {
        # Encoder + literal guardrails only. NLI and the cross-encoder
        # reranker are kept off so the lane stays under ~160 ms p95 on
        # an A5000.
        "VOYAGER_GROUNDEDNESS_NLI_ENABLED": "0",
        "VOYAGER_GROUNDEDNESS_NLI_ATOMIC_CLAIMS": "0",
        "VOYAGER_GROUNDEDNESS_NLI_PREMISE_CONCAT": "0",
        "VOYAGER_GROUNDEDNESS_NLI_PREMISE_RERANKER_MODEL": "",
        # Sweep winner: literal-only fusion. When literals are absent the
        # fuse helper falls back to the calibrated reverse-context MaxSim
        # score automatically (see ``_resolve_headline_for_risk_band``).
        "VOYAGER_GROUNDEDNESS_FUSION_W_CALIBRATED": "0.0",
        "VOYAGER_GROUNDEDNESS_FUSION_W_LITERAL": "1.0",
        "VOYAGER_GROUNDEDNESS_FUSION_W_NLI": "0.0",
        "VOYAGER_GROUNDEDNESS_FUSION_W_SEMANTIC_ENTROPY": "0.0",
        "VOYAGER_GROUNDEDNESS_FUSION_W_STRUCTURED": "0.0",
        # Per-profile risk-band thresholds calibrated under the same fusion
        # weights as above so the runtime distribution matches.
        "VOYAGER_GROUNDEDNESS_THRESHOLDS_PATH": _profile_data_path("thresholds.fast.json"),
    },
    "balanced": {
        # Default profile: NLI peer with multi-premise concatenation off
        # and no cross-encoder reranker. ~190 ms p95 on an A5000.
        "VOYAGER_GROUNDEDNESS_NLI_ENABLED": "1",
        "VOYAGER_GROUNDEDNESS_NLI_ATOMIC_CLAIMS": "0",
        "VOYAGER_GROUNDEDNESS_NLI_PREMISE_CONCAT": "0",
        "VOYAGER_GROUNDEDNESS_NLI_PREMISE_RERANKER_MODEL": "",
        "VOYAGER_GROUNDEDNESS_FUSION_W_CALIBRATED": "0.0",
        "VOYAGER_GROUNDEDNESS_FUSION_W_LITERAL": "0.0",
        "VOYAGER_GROUNDEDNESS_FUSION_W_NLI": "1.0",
        "VOYAGER_GROUNDEDNESS_FUSION_W_SEMANTIC_ENTROPY": "0.0",
        "VOYAGER_GROUNDEDNESS_FUSION_W_STRUCTURED": "0.0",
        "VOYAGER_GROUNDEDNESS_THRESHOLDS_PATH": _profile_data_path("thresholds.balanced.json"),
    },
    "quality": {
        # Full stack: NLI + cross-encoder reranker + atomic-claim
        # decomposition + multi-premise concatenation. Fusion weights
        # follow the L7 sweep winner exactly (``literal=0.2 / nli=0.8``)
        # so the per-stratum F1 numbers in
        # ``latence_trace/data/fusion_weights.quality.json`` apply to
        # what the runtime actually ships. Semantic entropy is left at
        # zero by default because the L7 sweep ran with
        # ``include_semantic_entropy=False`` - operators who want a SE
        # channel should re-run the sweep with their LLM ensemble
        # provider attached and override
        # ``VOYAGER_GROUNDEDNESS_FUSION_W_SEMANTIC_ENTROPY`` from the
        # resulting calibration artefact. The fuse helper renormalises
        # cleanly either way (see ``fuse_groundedness_v2``).
        "VOYAGER_GROUNDEDNESS_NLI_ENABLED": "1",
        "VOYAGER_GROUNDEDNESS_NLI_ATOMIC_CLAIMS": "1",
        "VOYAGER_GROUNDEDNESS_NLI_PREMISE_CONCAT": "1",
        "VOYAGER_GROUNDEDNESS_NLI_PREMISE_RERANKER_MODEL": "BAAI/bge-reranker-v2-m3",
        "VOYAGER_GROUNDEDNESS_FUSION_W_CALIBRATED": "0.0",
        "VOYAGER_GROUNDEDNESS_FUSION_W_LITERAL": "0.2",
        "VOYAGER_GROUNDEDNESS_FUSION_W_NLI": "0.8",
        "VOYAGER_GROUNDEDNESS_FUSION_W_SEMANTIC_ENTROPY": "0.0",
        "VOYAGER_GROUNDEDNESS_FUSION_W_STRUCTURED": "0.0",
        "VOYAGER_GROUNDEDNESS_THRESHOLDS_PATH": _profile_data_path("thresholds.quality.json"),
    },
}


_HOSTED_PROFILE_TO_PRESET: Dict[TraceRuntimeProfile, str] = {
    TraceRuntimeProfile.STANDARD: DEFAULT_PROFILE,
    TraceRuntimeProfile.QUALITY: "quality",
}


@dataclass(frozen=True)
class RequestRuntimeProfile:
    """Resolved request-scoped runtime profile.

    The resolver reads the same preset table used by ``apply_profile`` but never
    writes to ``os.environ``. Hosted ``standard`` and ``quality`` requests both
    use their bundled presets so a worker launched with a different process
    profile cannot silently change the externally billed profile semantics.
    """

    requested_profile: Optional[TraceRuntimeProfile]
    effective_profile: TraceRuntimeProfile
    preset_name: str
    nli_enabled: bool
    nli_use_atomic_claims: bool
    nli_concat_premises: bool
    nli_reranker_model: Optional[str]
    fusion_weights: Dict[str, float]
    thresholds_path: Optional[str]
    risk_band_policy: Optional[RiskBandPolicy]
    fusion_substitute_missing_channels_threshold: float = 0.0

    @property
    def is_quality(self) -> bool:
        return self.effective_profile == TraceRuntimeProfile.QUALITY


def _coerce_request_profile(
    profile: Optional[TraceRuntimeProfile],
) -> TraceRuntimeProfile:
    return TraceRuntimeProfile.QUALITY if profile == TraceRuntimeProfile.QUALITY else TraceRuntimeProfile.STANDARD


def _profile_env_value(runtime: RequestRuntimeProfile, key: str) -> Optional[str]:
    preset = PROFILE_ENV_PRESETS.get(runtime.preset_name, {})
    if key in preset:
        return preset[key]
    if key in os.environ:
        return os.environ[key]
    return None


def _profile_bool(runtime: RequestRuntimeProfile, key: str, default: bool) -> bool:
    raw = _profile_env_value(runtime, key)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _profile_int(runtime: RequestRuntimeProfile, key: str, default: int) -> int:
    raw = _profile_env_value(runtime, key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _profile_float(runtime: RequestRuntimeProfile, key: str, default: float) -> float:
    raw = _profile_env_value(runtime, key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _profile_fusion_weights(runtime: RequestRuntimeProfile) -> Dict[str, float]:
    base = nli_fusion_weights_from_env()
    preset = PROFILE_ENV_PRESETS.get(runtime.preset_name, {})
    for channel, key in (
        ("calibrated", "VOYAGER_GROUNDEDNESS_FUSION_W_CALIBRATED"),
        ("literal", "VOYAGER_GROUNDEDNESS_FUSION_W_LITERAL"),
        ("nli", "VOYAGER_GROUNDEDNESS_FUSION_W_NLI"),
        ("semantic_entropy", "VOYAGER_GROUNDEDNESS_FUSION_W_SEMANTIC_ENTROPY"),
        ("structured", "VOYAGER_GROUNDEDNESS_FUSION_W_STRUCTURED"),
    ):
        if key in preset:
            try:
                base[channel] = float(preset[key])
            except ValueError:
                continue
    return base


def resolve_request_runtime_profile(
    profile: Optional[TraceRuntimeProfile],
) -> RequestRuntimeProfile:
    """Resolve request-level hosted profile without mutating process env."""

    effective = _coerce_request_profile(profile)
    preset_name = _HOSTED_PROFILE_TO_PRESET[effective]

    provisional = RequestRuntimeProfile(
        requested_profile=profile,
        effective_profile=effective,
        preset_name=preset_name,
        nli_enabled=True,
        nli_use_atomic_claims=False,
        nli_concat_premises=False,
        nli_reranker_model=None,
        fusion_weights={},
        thresholds_path=None,
        risk_band_policy=None,
    )
    nli_reranker_model = (_profile_env_value(
        provisional, "VOYAGER_GROUNDEDNESS_NLI_PREMISE_RERANKER_MODEL"
    ) or "").strip() or None
    thresholds_path = _profile_env_value(
        provisional, "VOYAGER_GROUNDEDNESS_THRESHOLDS_PATH"
    )
    risk_band_policy: Optional[RiskBandPolicy] = None
    if thresholds_path:
        try:
            risk_band_policy = load_risk_band_policy(path=Path(thresholds_path))
        except Exception as exc:  # pragma: no cover - defensive diagnostics
            logger.warning(
                "request_profile_thresholds_load_failed",
                extra={
                    "profile": effective.value,
                    "thresholds_path": thresholds_path,
                    "error": str(exc),
                },
            )

    fusion_weights = _profile_fusion_weights(provisional)
    # Production fusion hardening: a non-zero threshold tells
    # ``fuse_groundedness_v2`` to substitute a 0.5 uncertainty prior for
    # any channel that has non-trivial weight but came back ``None``
    # (e.g. NLI engine crashed mid-request). Drift-safe default is 0.0
    # so direct in-process callers keep legacy drop-and-renormalise
    # behaviour. The router flips it on below for every routed request.
    fusion_substitute_threshold = 0.0

    # Layer in the per-class calibration bundle if the corpus router
    # produced one for this request. The router's decision lives on a
    # context variable so both ``_score_rag`` and ``_score_code`` pick it
    # up without threading a new argument through every callsite.
    route_decision = _ACTIVE_ROUTE_DECISION.get()
    if route_decision is not None and route_decision.bundle is not None:
        bundle = route_decision.bundle
        if bundle.scoring_mode == "rag" and bundle.fusion_weights:
            # Overwrite channel-by-channel so unspecified channels fall
            # through to the hosted preset (mostly a no-op since bundles
            # cover all five channels).
            fusion_weights = dict(fusion_weights)
            for channel, value in bundle.fusion_weights.items():
                fusion_weights[channel] = float(value)
        # Build a single-stratum RiskBandPolicy from the bundle so the
        # downstream classifier honours class-specific green / amber
        # thresholds. Preserving the source string ("calibration_bundle:<class>")
        # keeps provenance traceable in profile_diagnostics.
        risk_band_policy = RiskBandPolicy(
            headline="groundedness_v2",
            precision_target=0.75,
            nli_enabled=provisional.nli_enabled,
            strata={
                "default": {
                    "green_min": float(bundle.thresholds.get("green", 0.80)),
                    "amber_min": float(bundle.thresholds.get("amber", 0.60)),
                }
            },
            source=f"calibration_bundle:{bundle.class_key}",
            schema_version=1,
        )
        thresholds_path = f"calibration_bundle:{bundle.class_key}"
        # Enable missing-channel substitution for every routed request.
        # 0.2 = "only substitute channels the bundle actually relies on"
        # (all six shipped bundles assign at least 0.2 to any channel
        # they keep), which avoids penalising channels the bundle
        # already dropped to zero. Override via
        # ``LATENCE_TRACE_FUSION_SUBSTITUTE_MISSING_THRESHOLD``.
        fusion_substitute_threshold = float(
            os.environ.get(
                "LATENCE_TRACE_FUSION_SUBSTITUTE_MISSING_THRESHOLD",
                "0.2",
            )
        )

    return RequestRuntimeProfile(
        requested_profile=profile,
        effective_profile=effective,
        preset_name=preset_name,
        nli_enabled=_profile_bool(
            provisional,
            "VOYAGER_GROUNDEDNESS_NLI_ENABLED",
            nli_is_enabled(),
        ),
        nli_use_atomic_claims=_profile_bool(
            provisional,
            "VOYAGER_GROUNDEDNESS_NLI_ATOMIC_CLAIMS",
            nli_is_atomic_enabled(),
        ),
        nli_concat_premises=_profile_bool(
            provisional,
            "VOYAGER_GROUNDEDNESS_NLI_PREMISE_CONCAT",
            nli_is_premise_concat_enabled(),
        ),
        nli_reranker_model=nli_reranker_model,
        fusion_weights=fusion_weights,
        thresholds_path=thresholds_path,
        risk_band_policy=risk_band_policy,
        fusion_substitute_missing_channels_threshold=fusion_substitute_threshold,
    )


def _profile_diagnostics(
    *,
    runtime_profile: RequestRuntimeProfile,
    nli_provider: Any,
    nli_reranker: Any,
    verification_samples: Optional[Sequence[str]],
) -> Dict[str, Any]:
    semantic_entropy_skipped_reason = None
    if runtime_profile.is_quality and not verification_samples:
        semantic_entropy_skipped_reason = "semantic_entropy_skipped_no_samples"
    return {
        "requested_profile": (
            runtime_profile.requested_profile.value
            if runtime_profile.requested_profile
            else None
        ),
        "effective_profile": runtime_profile.effective_profile.value,
        "preset": runtime_profile.preset_name,
        "nli_requested": bool(runtime_profile.nli_enabled),
        "nli_provider_available": nli_provider is not None,
        "nli_atomic_claims": bool(runtime_profile.nli_use_atomic_claims),
        "nli_premise_concat": bool(runtime_profile.nli_concat_premises),
        "nli_reranker_model": runtime_profile.nli_reranker_model,
        "nli_reranker_available": nli_reranker is not None,
        "fusion_weights": dict(runtime_profile.fusion_weights),
        "thresholds_path": runtime_profile.thresholds_path,
        "thresholds_source": (
            runtime_profile.risk_band_policy.source
            if runtime_profile.risk_band_policy is not None
            else None
        ),
        "verification_sample_count": len(verification_samples or []),
        "semantic_entropy_skipped_reason": semantic_entropy_skipped_reason,
    }


def _auto_decide_requested(request: GroundednessRequest) -> bool:
    if request.auto_decide is not None:
        return bool(request.auto_decide)
    raw = os.environ.get("VOYAGER_TRACE_AUTO_DECIDE_DEFAULT", "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _maybe_auto_decide(
    response: GroundednessResponse, request: GroundednessRequest
) -> None:
    """Collapse amber -> green/red via a pinned LLM judge when requested.

    No-op when:
    * The request did not opt into ``auto_decide`` (explicitly or via the
      tenant default env knob).
    * The scored band is not ``amber`` (green/red stay as they are; red is
      already an accept/reject signal for the caller).
    * The escalation middleware is not enabled (provider misconfigured or
      API key missing).

    On judge success we rewrite the risk band on the response in-place so the
    existing response-serialisation path stays untouched. On judge failure
    the band stays amber and the caller sees the error in the diagnostics.
    """
    if not _auto_decide_requested(request):
        return
    try:
        band_value = str(response.scores.risk_band or "").lower()
    except Exception:  # pragma: no cover - defensive
        band_value = ""
    if band_value != "amber":
        return
    config = AmberEscalationConfig.from_env()
    if not config.enabled:
        return
    nli_diag = getattr(response, "nli_diagnostics", None)
    claim_payloads: List[Dict[str, Any]] = []
    if nli_diag is not None:
        for claim in getattr(nli_diag, "claims", []) or []:
            claim_dict = (
                claim.model_dump() if hasattr(claim, "model_dump") else dict(claim)
            )
            claim_payloads.append(claim_dict)
    payload = build_amber_payload(
        query_text=str(request.query_text or ""),
        response_text=str(request.response_text or ""),
        nli_claims=claim_payloads,
        config=config,
    )
    scores_obj = response.scores
    fallback_score = getattr(scores_obj, "groundedness_v2", None)
    if fallback_score is None:
        fallback_score = getattr(scores_obj, "primary_score", None)
    verdict = run_amber_escalate(payload, config, fallback_score=fallback_score)
    final_band = verdict.verdict if verdict.verdict in ("green", "red") else "amber"
    if final_band in ("green", "red"):
        response.scores.risk_band = final_band
    response.amber_escalation = AmberEscalationDiagnostics(
        original_band="amber",
        final_band=final_band,
        judge_verdict=verdict.verdict,
        reasoning=verdict.reasoning,
        judge_provider=verdict.judge_provider,
        judge_model=verdict.judge_model,
        judge_latency_ms=verdict.judge_latency_ms,
        judge_cost_usd=verdict.judge_cost_usd,
        judge_error=verdict.error,
    )


@dataclass(frozen=True)
class ProfileApplication:
    """Result of :func:`apply_profile` describing which keys were touched."""

    profile: str
    applied: Dict[str, str] = field(default_factory=dict)
    skipped: Dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "profile": self.profile,
            "applied": dict(self.applied),
            "skipped": dict(self.skipped),
        }


def apply_profile(
    profile: Optional[str],
    *,
    env: Optional[Dict[str, str]] = None,
    overrides: Optional[Mapping[str, str]] = None,
    refresh_thresholds: bool = True,
) -> ProfileApplication:
    """Materialise the env overlay for the given profile.

    Parameters
    ----------
    profile:
        One of :data:`PROFILE_NAMES` or ``None``. Empty strings, ``None``
        and the literal value ``"none"`` are treated as no-ops.
    env:
        Optional mutable mapping (defaults to :data:`os.environ`). The
        overlay is applied in-place so the rest of the process picks up
        the new values via the existing env-var readers in
        :mod:`latence_trace.core.nli`,
        :mod:`latence_trace.core.semantic_entropy` and
        :mod:`latence_trace.core.thresholds`.
    overrides:
        Optional explicit overrides applied on top of the preset. Useful
        for the standalone server's CLI flags (e.g. forcing a custom
        thresholds path).
    refresh_thresholds:
        When True (the default) the cached
        :class:`~latence_trace.core.thresholds.RiskBandPolicy` is
        rebuilt so the per-profile JSON file takes effect immediately.

    Returns
    -------
    :class:`ProfileApplication`
        Diagnostics describing which environment variables were applied
        versus skipped because the operator already exported them.

    Notes
    -----
    The function never overwrites an existing environment variable. This
    is intentional: profiles are convenience defaults for new operators,
    but any explicit env var on the host wins so production tuning
    decisions remain authoritative.
    """

    target_env = os.environ if env is None else env
    if profile is None or not str(profile).strip() or str(profile).strip().lower() == "none":
        return ProfileApplication(profile="none")

    name = str(profile).strip().lower()
    if name not in PROFILE_ENV_PRESETS:
        raise ValueError(
            "Unknown groundedness profile '{0}'. Choose one of: {1}".format(
                profile, ", ".join(sorted(PROFILE_ENV_PRESETS.keys()))
            )
        )

    preset: Dict[str, str] = dict(PROFILE_ENV_PRESETS[name])
    if overrides:
        for key, value in overrides.items():
            preset[str(key)] = str(value)

    applied: Dict[str, str] = {}
    skipped: Dict[str, str] = {}
    for key, value in preset.items():
        if key in target_env:
            skipped[key] = target_env[key]
            continue
        target_env[key] = value
        applied[key] = value

    # Always advertise the active profile so downstream diagnostics can
    # surface it even when no env vars were touched.
    target_env.setdefault("LATENCE_TRACE_ACTIVE_PROFILE", name)

    if refresh_thresholds:
        try:
            from latence_trace.core.thresholds import get_risk_band_policy

            get_risk_band_policy(refresh=True)
        except Exception as exc:  # pragma: no cover - defensive logging only
            logger.warning(
                "profile_thresholds_refresh_failed",
                extra={"profile": name, "error": str(exc)},
            )

    logger.info(
        "groundedness_profile_applied",
        extra={"profile": name, "applied_keys": sorted(applied.keys()), "skipped_keys": sorted(skipped.keys())},
    )
    return ProfileApplication(profile=name, applied=applied, skipped=skipped)


# ---------------------------------------------------------------------------
# chunk_ids resolver protocol
# ---------------------------------------------------------------------------


@dataclass
class ResolvedChunk:
    """A pre-fetched support unit supplied by the caller for a chunk_id.

    Standalone latence-trace deployments that want to expose the chunk_ids
    fast path (e.g. when sitting next to a retrieval engine like
    voyager-index) wire a resolver callback that turns a chunk_id into the
    supporting text plus its multi-vector embedding tensor as it would be
    produced by the configured encoder.
    """

    chunk_id: Any
    text: str
    embeddings: torch.Tensor
    metadata: Optional[Dict[str, Any]] = field(default=None)


ChunkResolver = Callable[[Sequence[Any]], List[ResolvedChunk]]


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class GroundednessService:
    """Standalone groundedness scoring service.

    Parameters
    ----------
    encoder_factory:
        Callable that returns an encoder provider for a given optional model
        name. The default factory honours ``VOYAGER_GROUNDEDNESS_VLLM_ENDPOINT``
        and ``VOYAGER_GROUNDEDNESS_MODEL`` and otherwise falls back to a
        local pylate ``models.ColBERT`` instance.
    chunk_resolver:
        Optional callback that resolves a list of chunk_ids into pre-fetched
        ``ResolvedChunk`` objects. When ``None`` (the default), requests using
        the ``chunk_ids`` path will be rejected with a clear validation error.
        Most standalone deployments use the ``raw_context`` path directly.
    device:
        Device hint passed through to local encoder providers.
    collection_label:
        Cosmetic label echoed back in the ``collection`` response field so
        existing voyager-index clients continue to see a stable shape.
    """

    def __init__(
        self,
        *,
        encoder_factory: Optional[Callable[[Optional[str]], Any]] = None,
        chunk_resolver: Optional[ChunkResolver] = None,
        device: str = "cpu",
        collection_label: str = "latence-trace",
    ) -> None:
        self._encoder_factory = encoder_factory or _default_encoder_factory(device)
        self._chunk_resolver = chunk_resolver
        self.device = device
        self._collection_label = collection_label
        self._cached_groundedness_providers: Dict[str, Any] = {}
        self._cached_groundedness_null_banks: Dict[str, List[torch.Tensor]] = {}
        # PA3 hot-path optimization: cache the pre-normalized, pre-stacked
        # null bank alongside the raw tensors so compute_null_distribution
        # collapses from N matmul + N max calls to one batched matmul +
        # one segment-max per request.
        self._cached_groundedness_null_packs: Dict[str, Any] = {}
        self._nli_provider: Any = None
        self._nli_provider_resolved = False
        self._nli_reranker: Any = None
        self._nli_reranker_resolved = False
        self._nli_rerankers_by_model: Dict[str, Any] = {}
        # Concurrent first requests must not double-resolve the NLI provider
        # or cross-encoder reranker - both are expensive HuggingFace loads.
        self._nli_provider_lock = threading.Lock()
        self._nli_reranker_lock = threading.Lock()
        self._null_bank_lock = threading.Lock()
        # Code-lane singletons. Lazy + lock-protected so two concurrent
        # cold code-lane requests don't each build their own GPU scorer
        # or tree-sitter extractor.
        self._code_scorer: Optional[GPUScorer] = None
        self._code_ast_extractor: Optional[AstSymbolExtractor] = None
        self._code_scorer_lock = threading.Lock()
        self._code_ast_extractor_lock = threading.Lock()

    # --- provider plumbing ----------------------------------------------------

    def _get_groundedness_provider(self, *, model_name: Optional[str] = None) -> Any:
        provider = self._encoder_factory(model_name)
        if provider is None:
            raise ValidationError(
                "No groundedness model configured. Set VOYAGER_GROUNDEDNESS_VLLM_ENDPOINT + "
                "VOYAGER_GROUNDEDNESS_VLLM_MODEL for the vLLM-factory production path, "
                "or VOYAGER_GROUNDEDNESS_MODEL for a local pylate model, or wire a custom "
                "encoder_factory when constructing the service."
            )
        return provider

    def _resolve_nli_provider_unchecked(self):
        """Resolve the default NLI provider even for request-level quality.

        ``latence_trace.core.nli.resolve_default_provider`` intentionally gates
        on the process-level env flag. Request-level quality must be able to use
        the same provider without flipping that global flag for concurrent
        standard requests.
        """

        model_id = os.environ.get(
            "VOYAGER_GROUNDEDNESS_NLI_MODEL",
            "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
        )
        vllm_endpoint = os.environ.get("LATENCE_TRACE_NLI_VLLM_ENDPOINT", "").strip()
        if vllm_endpoint:
            try:
                from latence_trace.providers.nli import VllmFactoryNLIProvider

                provider = VllmFactoryNLIProvider(
                    endpoint=vllm_endpoint,
                    model=os.environ.get("LATENCE_TRACE_NLI_VLLM_MODEL", model_id),
                    timeout=_env_float(
                        "LATENCE_TRACE_NLI_VLLM_TIMEOUT",
                        nli_default_max_latency_ms() / 1000.0,
                    ),
                    health_timeout=_env_float(
                        "LATENCE_TRACE_NLI_VLLM_HEALTH_TIMEOUT", 10.0
                    ),
                    max_concurrency=_env_int(
                        "LATENCE_TRACE_NLI_VLLM_MAX_CONCURRENCY",
                        nli_default_max_batch(),
                    ),
                )
                provider.healthcheck()
                return provider
            except Exception as exc:
                logger.warning(
                    "nli_vllm_provider_init_failed",
                    extra={
                        "endpoint": vllm_endpoint,
                        "model": model_id,
                        "error": str(exc),
                    },
                )
        try:
            return HuggingFaceNLIProvider(
                model_id=model_id,
                max_length=_env_int("VOYAGER_GROUNDEDNESS_NLI_MAX_TOKENS", 384),
            )
        except Exception as exc:
            logger.warning("nli_provider_init_failed", extra={"error": str(exc)})
            return None

    def _get_nli_provider(self, *, force_enabled: bool = False):
        if not (force_enabled or nli_is_enabled()):
            return None
        if self._nli_provider_resolved:
            return self._nli_provider
        with self._nli_provider_lock:
            if self._nli_provider_resolved:
                return self._nli_provider
            try:
                self._nli_provider = (
                    self._resolve_nli_provider_unchecked()
                    if force_enabled and not nli_is_enabled()
                    else nli_resolve_default_provider()
                )
            except Exception as exc:
                logger.warning("nli_resolve_failed", extra={"error": str(exc)})
                self._nli_provider = None
            self._nli_provider_resolved = True
            return self._nli_provider

    def _get_nli_reranker(self, *, model_id: Optional[str] = None, force_enabled: bool = False):
        if not (force_enabled or nli_is_enabled()):
            return None
        if model_id and not os.environ.get(
            "VOYAGER_GROUNDEDNESS_NLI_PREMISE_RERANKER_MODEL"
        ):
            if model_id in self._nli_rerankers_by_model:
                return self._nli_rerankers_by_model[model_id]
            with self._nli_reranker_lock:
                if model_id in self._nli_rerankers_by_model:
                    return self._nli_rerankers_by_model[model_id]
                try:
                    reranker = CrossEncoderPremiseReranker(
                        model_id=model_id,
                        max_length=_env_int(
                            "VOYAGER_GROUNDEDNESS_NLI_PREMISE_RERANKER_MAX_TOKENS",
                            512,
                        ),
                        batch_size=_env_int(
                            "VOYAGER_GROUNDEDNESS_NLI_PREMISE_RERANKER_BATCH",
                            32,
                        ),
                    )
                except Exception as exc:
                    logger.warning("nli_reranker_resolve_failed", extra={"error": str(exc)})
                    reranker = None
                self._nli_rerankers_by_model[model_id] = reranker
                return reranker
        if self._nli_reranker_resolved:
            return self._nli_reranker
        with self._nli_reranker_lock:
            if self._nli_reranker_resolved:
                return self._nli_reranker
            try:
                self._nli_reranker = nli_resolve_default_reranker()
            except Exception as exc:
                logger.warning("nli_reranker_resolve_failed", extra={"error": str(exc)})
                self._nli_reranker = None
            self._nli_reranker_resolved = True
            return self._nli_reranker

    def _groundedness_null_bank_pack(
        self,
        provider: Any,
        *,
        prompt_name: Optional[str],
    ) -> Any:
        """Return the cached pre-stacked null bank for ``provider`` (or None).

        Triggers a (locked) encode if the bank has not been cached yet so
        callers can use the pack as a single drop-in argument to
        :func:`compute_null_distribution`.
        """

        bank = self._groundedness_null_bank_embeddings(provider, prompt_name=prompt_name)
        if not bank:
            return None
        provider_key = (
            getattr(provider, "model_name", None)
            or getattr(provider, "model_name_or_path", None)
            or getattr(provider, "model", None)
            or repr(provider)
        )
        cache_key = f"{provider_key}::{id(provider)}::{prompt_name or ''}::v{len(bank)}"
        return self._cached_groundedness_null_packs.get(cache_key)

    def _groundedness_null_bank_embeddings(
        self,
        provider: Any,
        *,
        prompt_name: Optional[str],
    ) -> List[torch.Tensor]:
        if os.environ.get("VOYAGER_GROUNDEDNESS_DISABLE_CALIBRATION", "").lower() in {"1", "true", "yes"}:
            return []
        bank_texts = default_null_bank_texts()
        if not bank_texts:
            return []
        provider_key = (
            getattr(provider, "model_name", None)
            or getattr(provider, "model_name_or_path", None)
            or getattr(provider, "model", None)
            or repr(provider)
        )
        cache_key = f"{provider_key}::{id(provider)}::{prompt_name or ''}::v{len(bank_texts)}"
        cached = self._cached_groundedness_null_banks.get(cache_key)
        if cached is not None:
            return cached
        # Concurrent cold requests would otherwise each pay for a full
        # encode of the bilingual null bank; serialize with a lock so the
        # second caller sees the cached result.
        with self._null_bank_lock:
            cached = self._cached_groundedness_null_banks.get(cache_key)
            if cached is not None:
                return cached
            try:
                tensors = encode_texts(
                    provider,
                    list(bank_texts),
                    is_query=False,
                    prompt_name=prompt_name,
                )
            except Exception as exc:
                logger.warning("groundedness_null_bank_encode_failed", extra={"error": str(exc)})
                self._cached_groundedness_null_banks[cache_key] = []
                self._cached_groundedness_null_packs[cache_key] = None
                return []
            bank: List[torch.Tensor] = [torch.as_tensor(t, dtype=torch.float32) for t in tensors]
            self._cached_groundedness_null_banks[cache_key] = bank
            # PA3: pre-normalize and stack the bank once so each request
            # uses a single batched matmul instead of N matmuls.
            try:
                self._cached_groundedness_null_packs[cache_key] = _build_null_bank_pack(bank)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "groundedness_null_bank_pack_failed", extra={"error": str(exc)}
                )
                self._cached_groundedness_null_packs[cache_key] = None
            return bank

    # --- chunk_ids resolution -------------------------------------------------

    def _resolve_chunk_ids(self, chunk_ids: Sequence[Any]) -> List[ResolvedChunk]:
        if self._chunk_resolver is None:
            raise ValidationError(
                "chunk_ids fast path requires a chunk_resolver to be configured on "
                "GroundednessService. Either inject one (e.g. wired to voyager-index) "
                "or use the raw_context path instead."
            )
        resolved = list(self._chunk_resolver(chunk_ids))
        seen: set[str] = set()
        deduped: List[ResolvedChunk] = []
        for item in resolved:
            key = str(item.chunk_id)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        if len(deduped) < len(set(str(c) for c in chunk_ids)):
            missing = sorted(set(str(c) for c in chunk_ids) - seen)
            raise NotFoundError(f"Unknown chunk ids for groundedness: {missing}")
        return deduped

    # --- eligibility ----------------------------------------------------------

    @staticmethod
    def _eligibility(*, use_stored_vectors: bool) -> GroundednessEligibility:
        if use_stored_vectors:
            return GroundednessEligibility(
                collection_kind=CollectionKind.LATE_INTERACTION,
                vector_source="stored_vectors",
                storage_compression=None,
                quantization_mode=None,
                dequantized=True,
                user_facing_supported=True,
                warnings=[],
            )
        return GroundednessEligibility(
            collection_kind=CollectionKind.LATE_INTERACTION,
            vector_source="encoded_raw_context",
            storage_compression=None,
            quantization_mode=None,
            dequantized=True,
            user_facing_supported=True,
            warnings=[],
        )

    # --- lane singletons (lazy, lock-protected) -------------------------

    def _get_code_scorer(self) -> GPUScorer:
        """Return the process-wide :class:`GPUScorer` singleton.

        Two concurrent cold code-lane requests must not each build their
        own GPU scorer (dedicated CUDA stream creation is measurable).
        Gate the construction the same way :meth:`_get_nli_provider` does.
        """
        if self._code_scorer is not None:
            return self._code_scorer
        with self._code_scorer_lock:
            if self._code_scorer is not None:
                return self._code_scorer
            device = os.environ.get("LATENCE_TRACE_CODE_DEVICE") or (
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            self._code_scorer = GPUScorer(device=device)
            return self._code_scorer

    def _get_code_ast_extractor(self) -> AstSymbolExtractor:
        if self._code_ast_extractor is not None:
            return self._code_ast_extractor
        with self._code_ast_extractor_lock:
            if self._code_ast_extractor is not None:
                return self._code_ast_extractor
            self._code_ast_extractor = AstSymbolExtractor()
            return self._code_ast_extractor

    # --- main entry point -----------------------------------------------------

    def groundedness(self, request: GroundednessRequest) -> GroundednessResponse:
        """Score one groundedness request.

        Dispatches by :attr:`GroundednessRequest.scoring_mode`:

        - ``rag`` (default) — unchanged enterprise RAG pipeline.
        - ``code`` — routes through
          :func:`latence_trace.core.code_lane.score_code_groundedness`
          after the shared encoder pass so all code-specific signals
          (AST drift, literal novelty, NLI cascade, composite score,
          file attribution) can layer on top of the MaxSim backbone.

        The corpus-type router runs first: it infers the corpus class
        (or accepts ``request.corpus_type`` as an explicit override) and
        loads the matching calibration bundle. The bundle's fusion
        weights + thresholds are layered on top of the hosted profile
        via a request-scoped ``contextvars.ContextVar`` so every
        downstream scorer sees a single, coherent runtime profile.
        """
        decision = _corpus_router_middleware.route(request)
        # Router may override the scoring mode (e.g. agentic-coding
        # classifier chose code.agentic_trace even though the caller
        # left scoring_mode unset on default). Respect explicit caller
        # values (CODE was passed explicitly) unless the explicit
        # corpus_type path told us otherwise.
        effective_scoring_mode = request.scoring_mode
        if decision.bundle is not None:
            bundle_mode = decision.bundle.scoring_mode
            if bundle_mode == "code":
                effective_scoring_mode = ScoringMode.CODE
            elif bundle_mode == "rag":
                # Only downgrade CODE->RAG if the router (not the caller)
                # picked the class. Explicit caller-set CODE always wins.
                if decision.source != "explicit" and request.scoring_mode == ScoringMode.RAG:
                    effective_scoring_mode = ScoringMode.RAG
        if effective_scoring_mode != request.scoring_mode:
            request = request.model_copy(update={"scoring_mode": effective_scoring_mode})
        token = _ACTIVE_ROUTE_DECISION.set(decision)
        try:
            if request.scoring_mode == ScoringMode.CODE:
                response = self._score_code(request)
            else:
                response = self._score_rag(request)
        finally:
            _ACTIVE_ROUTE_DECISION.reset(token)
        # Attach router diagnostics for audit / dashboards. Kept as a
        # best-effort attach — a missing CorpusRouteDiagnostics field on
        # the response model would fail import, not runtime.
        try:
            from latence_trace.api.models import CorpusRouteDiagnostics
            response.corpus_route = CorpusRouteDiagnostics.model_validate(
                _corpus_router_middleware.build_diagnostics(decision)
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("corpus_router: failed to attach diagnostics: %r", exc)
        return response

    def rollup(self, request: RollupRequest) -> RollupResponse:
        """Aggregate a sequence of per-turn records into session metrics.

        Stateless and CPU-only. The service holds no memory between
        calls — the caller owns the list of turns. Typical wall-clock
        is sub-millisecond for reasonable session lengths (<200
        turns). See :mod:`latence_trace.api.rollup` for the aggregation
        formulas.
        """
        response = aggregate_turns(
            list(request.turns or []),
            session_id=request.session_id,
        )
        # Optional conversation-level heatmap: synthesize a compact
        # payload from the rollup aggregates so dashboards can render
        # the session story in the same visual vocabulary as the
        # per-turn response.
        fmt = (request.heatmap_format or "none").lower()
        if fmt in {"data", "html"}:
            scores_dict: Dict[str, Any] = {
                "composite_phantom_score": 1.0 - response.model_drift_pct,
                "groundedness_v2": 1.0 - response.noise_pct,
                "primary_score": 1.0 - response.noise_pct,
                "dead_weight_ratio": response.noise_pct,
                "risk_band": _risk_band_for_rollup(response),
            }
            # Synthetic "files" view using the top dead-file candidates.
            per_file_stubs = [
                CodeLanePerFileUsage(
                    path=entry.path,
                    n_units=0,
                    used=0,
                    uncertain=0,
                    unused=0,
                    coverage=0.0,
                    mean_score=0.0,
                    max_evidence=0.0,
                    owner_tokens=0,
                    owner_share=float(entry.ema_owner_share),
                    query_owner_tokens=0,
                    query_owner_share=0.0,
                    dead_weight=entry.dead_turns > 0,
                    reason_codes=(
                        ["never_won_argmax"] if entry.dead_turns > 0 else []
                    ),
                    dominating_peer=None,
                )
                for entry in response.top_dead_files
            ]
            fa_stub = CodeLaneFileAttribution(
                per_file=per_file_stubs,
                per_unit=[],
                dead_weight_files=[f.path for f in response.top_dead_files if f.dead_turns > 0],
                dead_weight_ratio=float(response.noise_pct),
                n_files=len(response.top_dead_files),
                n_response_tokens=0,
                n_query_tokens=0,
                reason_code_histogram=dict(response.reason_code_histogram),
            )
            payload = build_heatmap(
                scores_dict=scores_dict,
                file_attribution=fa_stub,
                response_tokens=[],
                session_signals=None,
            )
            response.heatmap = payload
            if fmt == "html":
                response.heatmap_html = render_heatmap_html(
                    payload, title="session rollup"
                )
        return response

    def _score_rag(self, request: GroundednessRequest) -> GroundednessResponse:
        start = time.perf_counter()
        runtime_profile = resolve_request_runtime_profile(request.profile)

        # K1: refuse-to-score gates. Both branches return a fully-typed
        # GroundednessResponse with risk_band="unknown" / "unsupported"
        # plus a machine-readable reason so callers (and AI agents) can
        # detect the no-evidence case without having to special-case
        # exception handling.
        has_chunk_ids = bool(request.chunk_ids)
        has_raw_context = bool((request.raw_context or "").strip())
        has_support_units = bool(request.support_units)
        if request.attribution_mode == AttributionMode.OPEN_DOMAIN:
            return self._refusal_response(
                request=request,
                started_at=start,
                mode="open_domain",
                risk_band="unsupported",
                reason="open_domain_pending_v1_next",
                warning="open_domain attribution_mode is reserved for the post-v1 retrieval-callback lane",
            )
        if not (has_chunk_ids or has_raw_context or has_support_units):
            return self._refusal_response(
                request=request,
                started_at=start,
                mode="closed_book",
                risk_band="unknown",
                reason="no_premise_supplied",
                warning=(
                    "No premise supplied. Provide one of chunk_ids, raw_context, or support_units. "
                    "closed_book attribution_mode refuses to score zero-evidence inputs by design."
                ),
            )

        if has_chunk_ids:
            mode = "chunk_ids"
        elif has_support_units:
            mode = "support_units"
        else:
            mode = "raw_context"
        use_stored_vectors = mode == "chunk_ids"
        eligibility = self._eligibility(use_stored_vectors=use_stored_vectors)
        provider = self._get_groundedness_provider(model_name=request.model)

        warnings: List[str] = list(eligibility.warnings)
        support_units: List[SupportUnitInput] = []

        if request.chunk_ids:
            for resolved in self._resolve_chunk_ids(request.chunk_ids):
                tensor = torch.as_tensor(resolved.embeddings, dtype=torch.float32)
                tokens = tokenize_text(
                    provider,
                    resolved.text or "",
                    expected_len=int(tensor.shape[0]),
                    is_query=False,
                )
                if not (resolved.text or "").strip():
                    warnings.append(
                        f"chunk_id '{resolved.chunk_id}' has no text/content payload; placeholder tokens will be used"
                    )
                support_units.append(
                    SupportUnitInput(
                        support_id=str(resolved.chunk_id),
                        chunk_id=resolved.chunk_id,
                        source_mode="chunk_ids",
                        text=resolved.text or "",
                        embeddings=tensor,
                        tokens=tokens,
                        metadata=dict(resolved.metadata or {}),
                    )
                )
        elif request.support_units:
            unit_inputs = list(request.support_units)
            unit_texts = [(u.text or "") for u in unit_inputs]
            unit_embeddings = encode_texts(
                provider,
                unit_texts,
                is_query=False,
                prompt_name=request.document_prompt_name,
            )
            for idx, (unit, tensor) in enumerate(zip(unit_inputs, unit_embeddings)):
                tokens = tokenize_text(
                    provider,
                    unit.text or "",
                    expected_len=int(tensor.shape[0]),
                    is_query=False,
                )
                # Caller-supplied source_id wins over a synthetic
                # "support-{idx}" id so downstream attribution is stable
                # across calls. The synthetic id is only used when the
                # caller did not bind one (e.g. ad-hoc dialogue turns).
                support_id = (unit.source_id or f"support-{idx}").strip() or f"support-{idx}"
                support_units.append(
                    SupportUnitInput(
                        support_id=support_id,
                        chunk_id=None,
                        source_mode="support_units",
                        text=unit.text or "",
                        embeddings=tensor,
                        tokens=tokens,
                        source_id=unit.source_id,
                        speaker=unit.speaker,
                        timestamp=unit.timestamp,
                        metadata=dict(unit.metadata) if unit.metadata else {},
                    )
                )
        else:
            encoder_token_limit = provider_token_limit(provider, is_query=False)
            if (
                request.segmentation_mode.value == "sentence_packed"
                and encoder_token_limit is not None
                and request.raw_context_chunk_tokens > encoder_token_limit
            ):
                warnings.append(
                    "raw_context_chunk_tokens={} exceeds the groundedness encoder token limit {}; "
                    "support windows may be truncated during encoding. Lower the budget or use a longer-context encoder.".format(
                        request.raw_context_chunk_tokens,
                        encoder_token_limit,
                    )
                )
            # File-header-aware segmentation so multi-file raw_context
            # bundles emit per-file support units (populates n_files,
            # dead_weight_files, per_file.owner_share instead of
            # collapsing to a single anonymous bucket). Collapses back
            # to the legacy single-block path when no headers are
            # present.
            file_blocks = split_raw_context_by_file_headers(
                request.raw_context or ""
            )
            segments: List[Dict[str, Any]] = []
            segment_paths: List[Optional[str]] = []
            for block in file_blocks:
                block_segments = segment_text(
                    block["text"],
                    request.segmentation_mode.value,
                    provider=provider,
                    chunk_token_budget=request.raw_context_chunk_tokens,
                )
                block_offset = int(block["offset_start"])
                for seg in block_segments:
                    rebased = dict(seg)
                    rebased["offset_start"] = int(seg["offset_start"]) + block_offset
                    rebased["offset_end"] = int(seg["offset_end"]) + block_offset
                    segments.append(rebased)
                    segment_paths.append(block["path"])
            if not segments:
                raise ValidationError("raw_context did not produce any support units")
            segment_texts = [segment["text"] for segment in segments]
            segment_embeddings = encode_texts(
                provider,
                segment_texts,
                is_query=False,
                prompt_name=request.document_prompt_name,
            )
            for idx, (segment, tensor) in enumerate(zip(segments, segment_embeddings)):
                tokens = tokenize_text(
                    provider,
                    segment["text"],
                    expected_len=int(tensor.shape[0]),
                    is_query=False,
                )
                path = segment_paths[idx]
                metadata: Dict[str, Any] = {"path": path} if path else {}
                support_units.append(
                    SupportUnitInput(
                        support_id=f"raw-{idx}",
                        chunk_id=None,
                        source_mode="raw_context",
                        text=segment["text"],
                        embeddings=tensor,
                        tokens=tokens,
                        offset_start=int(segment["offset_start"]),
                        offset_end=int(segment["offset_end"]),
                        metadata=metadata,
                    )
                )

        # Sentence-pack the response into windows that fit the encoder's
        # max sequence length. The orchestrator auto-bypasses chunking when
        # the response fits in one window (parity-preserving fast path) and
        # otherwise scores each window against the full support set,
        # stitching per-token rows back to global positions.
        encoder_token_limit_doc = provider_token_limit(provider, is_query=False)
        if (
            encoder_token_limit_doc is not None
            and request.response_chunk_tokens > encoder_token_limit_doc
        ):
            warnings.append(
                "response_chunk_tokens={} exceeds the groundedness encoder token limit {}; "
                "response windows may be truncated during encoding. Lower the budget or use a longer-context encoder.".format(
                    request.response_chunk_tokens,
                    encoder_token_limit_doc,
                )
            )
        response_chunks = _build_response_chunks(
            request.response_text,
            provider=provider,
            chunk_token_budget=request.response_chunk_tokens,
            encode_fn=encode_texts,
            document_prompt_name=request.document_prompt_name,
        )
        if not response_chunks:
            raise ValidationError("response_text did not produce any embeddings")

        include_triangular = bool(
            request.include_triangular_diagnostics and (request.query_text or "").strip()
        )
        if request.primary_metric.value == "triangular":
            include_triangular = True

        query_embeddings: Optional[torch.Tensor] = None
        query_tokens: Optional[List[str]] = None
        if include_triangular and request.query_text:
            query_embeddings = encode_texts(
                provider,
                [request.query_text],
                is_query=True,
                prompt_name=request.query_prompt_name,
            )[0]
            query_tokens = tokenize_text(
                provider,
                request.query_text,
                expected_len=int(query_embeddings.shape[0]),
                is_query=True,
            )

        # PA3 fast path: feed score_groundedness the pre-stacked pack
        # whenever it is available so compute_null_distribution skips the
        # per-request normalize + dispatch loop. Falls back to the raw
        # tensor list when the pack failed to build (extremely rare).
        null_bank_pack = self._groundedness_null_bank_pack(
            provider,
            prompt_name=request.document_prompt_name,
        )
        if null_bank_pack is not None:
            null_bank_embeddings = null_bank_pack
        else:
            null_bank_embeddings = self._groundedness_null_bank_embeddings(
                provider,
                prompt_name=request.document_prompt_name,
            )
        nli_provider = self._get_nli_provider(
            force_enabled=runtime_profile.nli_enabled
        )
        nli_reranker = (
            self._get_nli_reranker(
                model_id=runtime_profile.nli_reranker_model,
                force_enabled=runtime_profile.nli_enabled,
            )
            if nli_provider is not None and runtime_profile.nli_enabled
            else None
        )
        nli_kwargs: Dict[str, Any] = {}
        if nli_provider is not None:
            nli_kwargs = {
                "nli_provider": nli_provider,
                "nli_max_claims": _profile_int(
                    runtime_profile,
                    "VOYAGER_GROUNDEDNESS_NLI_MAX_CLAIMS",
                    nli_default_max_claims(),
                ),
                "nli_top_k_premises": _profile_int(
                    runtime_profile,
                    "VOYAGER_GROUNDEDNESS_NLI_TOP_K",
                    nli_default_top_k_premises(),
                ),
                "nli_max_batch": _profile_int(
                    runtime_profile,
                    "VOYAGER_GROUNDEDNESS_NLI_BATCH",
                    nli_default_max_batch(),
                ),
                "nli_max_latency_ms": _profile_float(
                    runtime_profile,
                    "VOYAGER_GROUNDEDNESS_NLI_LATENCY_MS",
                    nli_default_max_latency_ms(),
                ),
                "nli_reranker": nli_reranker,
                "nli_concat_premises": runtime_profile.nli_concat_premises,
                "nli_premise_concat_word_budget": _profile_int(
                    runtime_profile,
                    "VOYAGER_GROUNDEDNESS_NLI_PREMISE_CONCAT_BUDGET",
                    nli_default_premise_concat_word_budget(),
                ),
                "nli_use_atomic_claims": runtime_profile.nli_use_atomic_claims,
                "fusion_weights": runtime_profile.fusion_weights,
                "fusion_substitute_missing_channels_threshold": (
                    runtime_profile.fusion_substitute_missing_channels_threshold
                ),
                "risk_band_policy": runtime_profile.risk_band_policy,
            }
        elif runtime_profile.is_quality:
            warnings.append("quality_profile_nli_unavailable")
        if request.verification_samples:
            nli_kwargs.setdefault("fusion_weights", runtime_profile.fusion_weights)
            nli_kwargs.setdefault(
                "fusion_substitute_missing_channels_threshold",
                runtime_profile.fusion_substitute_missing_channels_threshold,
            )
            nli_kwargs["verification_samples"] = list(request.verification_samples)
            nli_kwargs["semantic_entropy_enabled"] = True
        elif runtime_profile.is_quality:
            warnings.append("semantic_entropy_skipped_no_samples")
        if request.risk_band_stratum:
            nli_kwargs["risk_band_stratum"] = str(request.risk_band_stratum)
        if request.content_type:
            nli_kwargs["content_type"] = str(request.content_type)
        if request.structured_verification:
            nli_kwargs["structured_verification"] = str(request.structured_verification)
        if request.raw_context:
            nli_kwargs["structured_support_text"] = request.raw_context
        nli_kwargs.setdefault("risk_band_policy", runtime_profile.risk_band_policy)

        if request.chunk_ids:
            # chunk_ids path: single support batch (caller-supplied
            # embeddings), routed through the response-chunked orchestrator
            # so long responses still get the chunked path while short
            # responses hit the single-chunk parity fast path.
            support_batches = [list(support_units)]
        else:
            # Both raw_context and support_units share the chunked scoring
            # path so very large inputs (e.g. a transcript of dozens of
            # speaker turns supplied via support_units[]) are batched the
            # same way segmented raw_context already is.
            support_batches = partition_support_units(
                support_units,
                batch_size=_env_int("VOYAGER_GROUNDEDNESS_SCORE_BATCH_UNITS", 64),
            )
        scored = score_groundedness_response_chunked(
            response_chunks=response_chunks,
            support_batches=support_batches,
            response_text=request.response_text,
            query_embeddings=query_embeddings,
            query_tokens=query_tokens,
            evidence_limit=request.evidence_limit,
            primary_metric=request.primary_metric.value,
            debug_dense_matrices=request.debug_dense_matrices,
            null_bank_embeddings=null_bank_embeddings or None,
            coverage_threshold=float(request.coverage_threshold),
            **nli_kwargs,
        )
        if scored.get("_response_chunk_count", 1) > 1:
            warnings.append(
                "response_chunked: response_text was sentence-packed into {} windows of <= {} tokens; per-token scores stitched back to global positions.".format(
                    int(scored["_response_chunk_count"]),
                    int(request.response_chunk_tokens),
                )
            )
        scored.pop("_response_chunk_count", None)
        warnings.extend(scored.pop("warnings", []))
        warnings = list(dict.fromkeys(warnings))

        model_name = (
            getattr(provider, "model_name", None)
            or getattr(provider, "model_name_or_path", None)
            or request.model
            or os.environ.get("VOYAGER_GROUNDEDNESS_MODEL")
            or os.environ.get("VOYAGER_ENCODE_MODEL")
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        file_attribution_wire = _file_attribution_to_wire(
            scored.get("file_attribution"),
            emit_chunk_ownership=bool(request.emit_chunk_ownership),
        )
        heatmap_payload, heatmap_html_str = _build_heatmap_fields(
            heatmap_format=request.heatmap_format,
            scores_dict=scored["scores"],
            file_attribution=file_attribution_wire,
            response_tokens=scored.get("response_tokens") or [],
            session_signals=None,
        )
        response = GroundednessResponse(
            collection=self._collection_label,
            mode=mode,
            model=model_name,
            scores=scored["scores"],
            response_tokens=scored["response_tokens"],
            support_units=scored["support_units"],
            top_evidence=scored["top_evidence"],
            eligibility=eligibility,
            query_tokens=scored.get("query_tokens"),
            debug=scored.get("debug"),
            warnings=warnings,
            literal_diagnostics=scored.get("literal_diagnostics"),
            nli_diagnostics=scored.get("nli_diagnostics"),
            semantic_entropy_diagnostics=(
                scored.get("semantic_entropy_diagnostics")
                or (
                    {
                        "aggregate_score": None,
                        "entropy_raw": None,
                        "sample_count": 0,
                        "cluster_count": 0,
                        "clusters": [],
                        "skipped_reason": "semantic_entropy_skipped_no_samples",
                    }
                    if runtime_profile.is_quality
                    and not request.verification_samples
                    else None
                )
            ),
            structured_diagnostics=scored.get("structured_diagnostics"),
            file_attribution=file_attribution_wire,
            heatmap=heatmap_payload,
            heatmap_html=heatmap_html_str,
            time_ms=elapsed_ms,
            scoring_mode=ScoringMode.RAG,
            profile=request.profile,
            effective_profile=runtime_profile.effective_profile,
            profile_diagnostics=_profile_diagnostics(
                runtime_profile=runtime_profile,
                nli_provider=nli_provider,
                nli_reranker=nli_reranker,
                verification_samples=request.verification_samples,
            ),
            session_id=request.session_id,
            attribution_mode=request.attribution_mode,
        )
        _maybe_auto_decide(response, request)
        return response

    # --- code-lane ------------------------------------------------------

    def _score_code(self, request: GroundednessRequest) -> GroundednessResponse:
        """Score a request through the code lane.

        The encoder pass reuses the same provider, segmentation, and
        support-batch plumbing as the RAG lane so the MaxSim backbone is
        bit-identical. On top of that the code lane runs:

        - GPU MaxSim scorer with query-aware ownership
          (:class:`latence_trace.core.code_lane.GPUScorer`)
        - AST symbol extractor (multi-language tree-sitter)
        - Literal novelty (per-identifier first-match unit score)
        - Ambiguity-gated NLI cascade
        - Optional semantic entropy when ``verification_samples`` is set
        - Calibrated composite phantom score (linear or logistic)
        - Per-file and per-unit attribution with reason codes
        """
        start = time.perf_counter()
        runtime_profile = resolve_request_runtime_profile(request.profile)

        has_chunk_ids = bool(request.chunk_ids)
        has_raw_context = bool((request.raw_context or "").strip())
        has_support_units = bool(request.support_units)
        if request.attribution_mode == AttributionMode.OPEN_DOMAIN:
            return self._refusal_response(
                request=request,
                started_at=start,
                mode="open_domain",
                risk_band="unsupported",
                reason="open_domain_pending_v1_next",
                warning=(
                    "open_domain attribution_mode is reserved for the post-v1 "
                    "retrieval-callback lane"
                ),
            )
        if not (has_chunk_ids or has_raw_context or has_support_units):
            return self._refusal_response(
                request=request,
                started_at=start,
                mode="closed_book",
                risk_band="unknown",
                reason="no_premise_supplied",
                warning=(
                    "No premise supplied. Provide one of chunk_ids, raw_context, "
                    "or support_units. closed_book attribution_mode refuses to "
                    "score zero-evidence inputs by design."
                ),
            )

        if has_chunk_ids:
            mode = "chunk_ids"
        elif has_support_units:
            mode = "support_units"
        else:
            mode = "raw_context"
        eligibility = self._eligibility(use_stored_vectors=(mode == "chunk_ids"))
        provider = self._get_groundedness_provider(model_name=request.model)
        warnings: List[str] = list(eligibility.warnings)

        support_units: List[SupportUnitInput] = self._encode_support_units(
            request=request,
            mode=mode,
            provider=provider,
            warnings=warnings,
        )

        response_chunks = _build_response_chunks(
            request.response_text,
            provider=provider,
            chunk_token_budget=request.response_chunk_tokens,
            encode_fn=encode_texts,
            document_prompt_name=request.document_prompt_name,
        )
        if not response_chunks:
            raise ValidationError("response_text did not produce any embeddings")

        query_embeddings: Optional[torch.Tensor] = None
        query_tokens: Optional[List[str]] = None
        if (request.query_text or "").strip():
            query_embeddings = encode_texts(
                provider,
                [request.query_text],
                is_query=True,
                prompt_name=request.query_prompt_name,
            )[0]
            query_tokens = tokenize_text(
                provider,
                request.query_text,
                expected_len=int(query_embeddings.shape[0]),
                is_query=True,
            )

        response_tokens_flat, response_embeddings_flat, response_token_char_spans = (
            _stitch_response_chunks(response_chunks)
        )

        support_packs = _support_units_to_packs(support_units)

        code_config = CodeLaneConfig(
            response_language_hint=request.response_language_hint,
            coverage_threshold=float(request.coverage_threshold),
        )

        scorer = self._get_code_scorer()
        ast_extractor = (
            self._get_code_ast_extractor() if code_config.enable_ast else None
        )
        nli_provider = (
            self._get_nli_provider(force_enabled=runtime_profile.nli_enabled)
            if code_config.enable_nli_cascade
            else None
        )
        if runtime_profile.is_quality and nli_provider is None:
            warnings.append("quality_profile_nli_unavailable")
        if runtime_profile.is_quality and not request.verification_samples:
            warnings.append("semantic_entropy_skipped_no_samples")

        code_result: CodeLaneResult = score_code_groundedness(
            response_text=request.response_text,
            response_tokens=response_tokens_flat,
            response_embeddings=response_embeddings_flat,
            support_units=support_packs,
            query_text=request.query_text,
            query_embeddings=query_embeddings,
            verification_samples=(
                list(request.verification_samples)
                if request.verification_samples
                else None
            ),
            config=code_config,
            scorer=scorer,
            ast_extractor=ast_extractor,
            nli_provider=nli_provider,
        )

        # Build the API response payload from the code-lane result.
        model_name = (
            getattr(provider, "model_name", None)
            or getattr(provider, "model_name_or_path", None)
            or request.model
            or os.environ.get("VOYAGER_GROUNDEDNESS_MODEL")
            or os.environ.get("VOYAGER_ENCODE_MODEL")
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        diagnostics = _code_lane_diagnostics(
            code_result=code_result,
            emit_chunk_ownership=bool(request.emit_chunk_ownership),
        )

        scores = _code_lane_scores(
            code_result=code_result,
            coverage_threshold=float(request.coverage_threshold),
        )

        # Caller-portable session state — opt-in via either an explicit
        # ``session_state`` payload or just ``session_id``. The server is
        # stateless: we run a pure deterministic transform and round-trip
        # the blob back. Nothing is persisted anywhere.
        next_session_state, session_signals_payload = self._advance_session_state(
            request=request,
            scores=scores,
            code_result=code_result,
        )

        # Apply router-supplied thresholds to the code lane's
        # ``composite_phantom_score`` so ``code.agentic_trace`` requests
        # come back with a band (previously only the RAG lane did). When
        # the router didn't install a policy, stay None so callers can
        # tell the difference between "no band" and "green".
        if (
            runtime_profile.risk_band_policy is not None
            and scores.composite_phantom_score is not None
        ):
            from latence_trace.core.thresholds import classify_risk_band

            scores.risk_band = classify_risk_band(
                float(scores.composite_phantom_score),
                policy=runtime_profile.risk_band_policy,
            )

        # Mirror the code-lane file-attribution onto the lane-neutral
        # top-level field so downstream dashboards have one canonical
        # place to read from regardless of ``scoring_mode``.
        top_level_file_attribution = (
            diagnostics.file_attribution if diagnostics is not None else None
        )
        heatmap_payload, heatmap_html_str = _build_heatmap_fields(
            heatmap_format=request.heatmap_format,
            scores_dict=scores.model_dump(exclude_none=True),
            file_attribution=top_level_file_attribution,
            response_tokens=[],
            session_signals=session_signals_payload,
        )

        return GroundednessResponse(
            collection=self._collection_label,
            mode=mode,
            model=model_name,
            scores=scores,
            response_tokens=[],
            support_units=[],
            top_evidence=[],
            eligibility=eligibility,
            query_tokens=None,
            debug=None,
            warnings=warnings,
            literal_diagnostics=None,
            nli_diagnostics=None,
            semantic_entropy_diagnostics=None,
            structured_diagnostics=None,
            code_lane_diagnostics=diagnostics,
            file_attribution=top_level_file_attribution,
            heatmap=heatmap_payload,
            heatmap_html=heatmap_html_str,
            time_ms=elapsed_ms,
            scoring_mode=ScoringMode.CODE,
            profile=request.profile,
            effective_profile=runtime_profile.effective_profile,
            profile_diagnostics=_profile_diagnostics(
                runtime_profile=runtime_profile,
                nli_provider=nli_provider,
                nli_reranker=None,
                verification_samples=request.verification_samples,
            ),
            session_id=request.session_id,
            attribution_mode=request.attribution_mode,
            next_session_state=next_session_state,
            session_signals=session_signals_payload,
        )

    @staticmethod
    def _advance_session_state(
        *,
        request: GroundednessRequest,
        scores: GroundednessScores,
        code_result: CodeLaneResult,
    ) -> Tuple[Optional[SessionStatePayload], Optional[SessionSignalsPayload]]:
        """Run the pure session-state transform and map the result back.

        Opt-in rule: callers get ``next_session_state`` + ``session_signals``
        if they either supplied a ``session_state`` payload (a real
        multi-turn client) *or* a ``session_id`` (a first-turn client
        that wants to bootstrap the session blob). Callers that pass
        neither keep the previous code-lane contract byte-for-byte.
        """
        if request.session_state is None and not request.session_id:
            return None, None

        prior = _session_state_from_payload(request.session_state)
        turn_metrics = _turn_metrics_from(scores=scores, code_result=code_result)

        try:
            next_state, signals = update_session_state(
                prior,
                turn_metrics,
                session_id=request.session_id,
            )
        except Exception:  # pragma: no cover - defensive
            logger.exception("session-state update failed; dropping session signals")
            return None, None

        return (
            _session_state_to_payload(next_state),
            _session_signals_to_payload(signals),
        )

    def _encode_support_units(
        self,
        *,
        request: GroundednessRequest,
        mode: str,
        provider: Any,
        warnings: List[str],
    ) -> List[SupportUnitInput]:
        """Shared encoder pass for both lanes.

        Extracted from :meth:`_score_rag` verbatim so the code lane
        reuses the identical segmentation + encoding logic — only the
        downstream scoring orchestration diverges.
        """
        support_units: List[SupportUnitInput] = []
        if mode == "chunk_ids":
            for resolved in self._resolve_chunk_ids(request.chunk_ids or []):
                tensor = torch.as_tensor(resolved.embeddings, dtype=torch.float32)
                tokens = tokenize_text(
                    provider,
                    resolved.text or "",
                    expected_len=int(tensor.shape[0]),
                    is_query=False,
                )
                if not (resolved.text or "").strip():
                    warnings.append(
                        f"chunk_id '{resolved.chunk_id}' has no text/content payload; placeholder tokens will be used"
                    )
                support_units.append(
                    SupportUnitInput(
                        support_id=str(resolved.chunk_id),
                        chunk_id=resolved.chunk_id,
                        source_mode="chunk_ids",
                        text=resolved.text or "",
                        embeddings=tensor,
                        tokens=tokens,
                        metadata=dict(resolved.metadata or {}),
                    )
                )
            return support_units

        if mode == "support_units" and request.support_units:
            unit_inputs = list(request.support_units)
            unit_texts = [(u.text or "") for u in unit_inputs]
            unit_embeddings = encode_texts(
                provider,
                unit_texts,
                is_query=False,
                prompt_name=request.document_prompt_name,
            )
            for idx, (unit, tensor) in enumerate(zip(unit_inputs, unit_embeddings)):
                tokens = tokenize_text(
                    provider,
                    unit.text or "",
                    expected_len=int(tensor.shape[0]),
                    is_query=False,
                )
                support_id = (unit.source_id or f"support-{idx}").strip() or f"support-{idx}"
                support_units.append(
                    SupportUnitInput(
                        support_id=support_id,
                        chunk_id=None,
                        source_mode="support_units",
                        text=unit.text or "",
                        embeddings=tensor,
                        tokens=tokens,
                        source_id=unit.source_id,
                        speaker=unit.speaker,
                        timestamp=unit.timestamp,
                        metadata=dict(unit.metadata) if unit.metadata else {},
                    )
                )
            return support_units

        # raw_context path
        encoder_token_limit = provider_token_limit(provider, is_query=False)
        if (
            request.segmentation_mode.value == "sentence_packed"
            and encoder_token_limit is not None
            and request.raw_context_chunk_tokens > encoder_token_limit
        ):
            warnings.append(
                "raw_context_chunk_tokens={} exceeds the groundedness encoder token limit {}; "
                "support windows may be truncated during encoding. Lower the budget or use a longer-context encoder.".format(
                    request.raw_context_chunk_tokens,
                    encoder_token_limit,
                )
            )
        # Split the raw_context on file headers first so per-file
        # attribution (``n_files``, ``dead_weight_files``, owner_share,
        # per-file coverage) works on multi-file bundles. When no
        # headers are present this collapses back to the legacy single-
        # block path with ``path=None`` - existing callers are
        # unaffected. Patterns recognised: ``# file: <path>``,
        # ``=== <path> ===``, ``--- a/<path>``, ``+++ b/<path>``.
        file_blocks = split_raw_context_by_file_headers(
            request.raw_context or ""
        )
        segments: List[Dict[str, Any]] = []
        segment_paths: List[Optional[str]] = []
        for block in file_blocks:
            block_segments = segment_text(
                block["text"],
                request.segmentation_mode.value,
                provider=provider,
                chunk_token_budget=request.raw_context_chunk_tokens,
            )
            block_offset = int(block["offset_start"])
            for seg in block_segments:
                # Rebase segment offsets onto the full raw_context so
                # per-unit offsets match pre-splitter behaviour.
                rebased = dict(seg)
                rebased["offset_start"] = int(seg["offset_start"]) + block_offset
                rebased["offset_end"] = int(seg["offset_end"]) + block_offset
                segments.append(rebased)
                segment_paths.append(block["path"])
        if not segments:
            raise ValidationError("raw_context did not produce any support units")
        segment_texts = [segment["text"] for segment in segments]
        segment_embeddings = encode_texts(
            provider,
            segment_texts,
            is_query=False,
            prompt_name=request.document_prompt_name,
        )
        for idx, (segment, tensor) in enumerate(zip(segments, segment_embeddings)):
            tokens = tokenize_text(
                provider,
                segment["text"],
                expected_len=int(tensor.shape[0]),
                is_query=False,
            )
            path = segment_paths[idx]
            metadata: Dict[str, Any] = {"path": path} if path else {}
            support_units.append(
                SupportUnitInput(
                    support_id=f"raw-{idx}",
                    chunk_id=None,
                    source_mode="raw_context",
                    text=segment["text"],
                    embeddings=tensor,
                    tokens=tokens,
                    offset_start=int(segment["offset_start"]),
                    offset_end=int(segment["offset_end"]),
                    metadata=metadata,
                )
            )
        return support_units

    def _refusal_response(
        self,
        *,
        request: GroundednessRequest,
        started_at: float,
        mode: str,
        risk_band: str,
        reason: str,
        warning: str,
    ) -> GroundednessResponse:
        """Build a structured zero-evidence response (K1 refuse-to-score path).

        Returned for ``closed_book`` calls with no premise and for the
        forward-compat ``open_domain`` lane. We return a fully-typed
        :class:`GroundednessResponse` instead of an HTTP error so callers
        and AI agents can branch on ``risk_band`` / ``reason`` without
        special-casing exception handling.
        """

        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        runtime_profile = resolve_request_runtime_profile(request.profile)
        scores = GroundednessScores(
            primary_name=request.primary_metric.value,
            primary_score=0.0,
            reverse_context=0.0,
            risk_band=risk_band,
        )
        return GroundednessResponse(
            collection=self._collection_label,
            mode=mode,
            model=None,
            scores=scores,
            response_tokens=[],
            support_units=[],
            top_evidence=[],
            eligibility=self._eligibility(use_stored_vectors=False),
            query_tokens=None,
            debug=None,
            warnings=[warning],
            literal_diagnostics=None,
            nli_diagnostics=None,
            semantic_entropy_diagnostics=None,
            structured_diagnostics=None,
            time_ms=elapsed_ms,
            scoring_mode=request.scoring_mode,
            profile=request.profile,
            effective_profile=runtime_profile.effective_profile,
            profile_diagnostics=_profile_diagnostics(
                runtime_profile=runtime_profile,
                nli_provider=None,
                nli_reranker=None,
                verification_samples=request.verification_samples,
            ),
            session_id=request.session_id,
            attribution_mode=request.attribution_mode,
            reason=reason,
        )


# ---------------------------------------------------------------------------
# Code-lane helpers
# ---------------------------------------------------------------------------


def _stitch_response_chunks(
    response_chunks: Sequence[Any],
) -> Tuple[List[str], torch.Tensor, List[Tuple[Optional[int], Optional[int]]]]:
    """Concatenate per-chunk response embeddings/tokens into a flat sequence.

    The code-lane scorer operates on a single ``(T, d)`` response
    embedding and a parallel ``List[str]`` of tokens. The RAG lane
    ships response chunks as a list, so here we flatten — preserving
    chunk order — and accumulate each token's global char span so the
    downstream attribution layer can locate evidence inside the
    original response text.
    """
    tokens: List[str] = []
    embeddings: List[torch.Tensor] = []
    char_spans: List[Tuple[Optional[int], Optional[int]]] = []
    for chunk in response_chunks:
        chunk_tokens = list(getattr(chunk, "tokens", []) or [])
        tokens.extend(chunk_tokens)
        embeddings.append(torch.as_tensor(chunk.embeddings, dtype=torch.float32))
        spans = getattr(chunk, "token_char_spans", None)
        base = int(getattr(chunk, "offset_start", 0) or 0)
        if spans:
            for span in spans:
                if span is None:
                    char_spans.append((None, None))
                else:
                    char_spans.append((base + int(span[0]), base + int(span[1])))
        else:
            char_spans.extend([(None, None)] * len(chunk_tokens))
    if not embeddings:
        return [], torch.zeros((0, 0), dtype=torch.float32), []
    return tokens, torch.cat(embeddings, dim=0), char_spans


def _support_units_to_packs(
    support_units: Sequence[SupportUnitInput],
) -> List[SupportUnitPack]:
    """Adapt the shared :class:`SupportUnitInput` to :class:`SupportUnitPack`.

    Best-effort path extraction: caller-supplied ``metadata.path`` wins,
    otherwise we look for a ``# path/to/file.py`` header in the text,
    otherwise the support_id is used as the attribution key.
    """
    packs: List[SupportUnitPack] = []
    for unit in support_units:
        metadata_map: Dict[str, Any] = dict(unit.metadata or {})
        metadata_map.setdefault("text", unit.text)
        path: Optional[str] = None
        raw_path = metadata_map.get("path")
        if raw_path:
            path = str(raw_path)
        elif unit.text:
            first_line = unit.text.split("\n", 1)[0].strip()
            if first_line.startswith("#"):
                candidate = first_line.lstrip("# ").strip()
                if candidate and (
                    "/" in candidate
                    or "." in candidate
                    or candidate.endswith((".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs"))
                ):
                    path = candidate
        packs.append(
            SupportUnitPack(
                support_id=unit.support_id,
                path=path,
                tokens=tuple(unit.tokens or ()),
                embeddings=torch.as_tensor(unit.embeddings, dtype=torch.float32),
                offset_start=unit.offset_start,
                offset_end=unit.offset_end,
                metadata=metadata_map,
            )
        )
    return packs


# ---------------------------------------------------------------------------
# Session-state marshalling (pure, lives next to the other mappers)
# ---------------------------------------------------------------------------


def _session_state_from_payload(
    payload: Optional[SessionStatePayload],
) -> Optional[SessionStateData]:
    if payload is None:
        return None
    # Treat version mismatches as a fresh session — safer than trying to
    # upgrade across schema bumps on the hot path.
    if int(payload.schema_version) != SESSION_STATE_SCHEMA_VERSION:
        return None
    return SessionStateData(
        schema_version=SESSION_STATE_SCHEMA_VERSION,
        session_id=payload.session_id,
        total_turns=int(payload.total_turns),
        cascade_fires=int(payload.cascade_fires),
        per_token_rolling=_rolling_from_payload(payload.per_token_rolling),
        composite_rolling=_rolling_from_payload(payload.composite_rolling),
        groundedness_rolling=_rolling_from_payload(payload.groundedness_rolling),
        groundedness_baseline=_rolling_from_payload(payload.groundedness_baseline),
        ema_groundedness=payload.ema_groundedness,
        file_stats={
            path: _file_stats_from_payload(entry)
            for path, entry in (payload.file_stats or {}).items()
        },
        phantom_trail=list(payload.phantom_trail or []),
        risk_band_trail=list(payload.risk_band_trail or []),
        ema_half_life_turns=int(payload.ema_half_life_turns or 5),
        groundedness_ema_half_life=int(payload.groundedness_ema_half_life or 3),
        verdict_window=int(payload.verdict_window or 20),
        risk_band_window=int(payload.risk_band_window or 20),
    )


def _rolling_from_payload(payload: RollingStatsPayload):  # type: ignore[no-untyped-def]
    from latence_trace.core.code_lane.session import RollingStats

    return RollingStats(
        n=int(payload.n),
        mean=float(payload.mean),
        m2=float(payload.m2),
    )


def _file_stats_from_payload(payload: FileSessionStatsPayload):  # type: ignore[no-untyped-def]
    from latence_trace.core.code_lane.session import FileSessionStats

    return FileSessionStats(
        ema_owner_share=float(payload.ema_owner_share),
        ema_query_owner_share=float(payload.ema_query_owner_share),
        dead_turns=int(payload.dead_turns),
        total_turns=int(payload.total_turns),
    )


def _session_state_to_payload(state: SessionStateData) -> SessionStatePayload:
    return SessionStatePayload(
        schema_version=state.schema_version,
        session_id=state.session_id,
        total_turns=state.total_turns,
        cascade_fires=state.cascade_fires,
        per_token_rolling=RollingStatsPayload(
            n=state.per_token_rolling.n,
            mean=state.per_token_rolling.mean,
            m2=state.per_token_rolling.m2,
        ),
        composite_rolling=RollingStatsPayload(
            n=state.composite_rolling.n,
            mean=state.composite_rolling.mean,
            m2=state.composite_rolling.m2,
        ),
        groundedness_rolling=RollingStatsPayload(
            n=state.groundedness_rolling.n,
            mean=state.groundedness_rolling.mean,
            m2=state.groundedness_rolling.m2,
        ),
        groundedness_baseline=RollingStatsPayload(
            n=state.groundedness_baseline.n,
            mean=state.groundedness_baseline.mean,
            m2=state.groundedness_baseline.m2,
        ),
        ema_groundedness=state.ema_groundedness,
        file_stats={
            path: FileSessionStatsPayload(
                ema_owner_share=stats.ema_owner_share,
                ema_query_owner_share=stats.ema_query_owner_share,
                dead_turns=stats.dead_turns,
                total_turns=stats.total_turns,
            )
            for path, stats in state.file_stats.items()
        },
        phantom_trail=list(state.phantom_trail),
        risk_band_trail=list(state.risk_band_trail),
        ema_half_life_turns=state.ema_half_life_turns,
        groundedness_ema_half_life=state.groundedness_ema_half_life,
        verdict_window=state.verdict_window,
        risk_band_window=state.risk_band_window,
    )


def _session_signals_to_payload(
    signals: SessionSignalsData,
) -> SessionSignalsPayload:
    return SessionSignalsPayload(
        total_turns=signals.total_turns,
        drift_z_score=signals.drift_z_score,
        ema_groundedness=signals.ema_groundedness,
        groundedness_drift=signals.groundedness_drift,
        dead_file_candidates=list(signals.dead_file_candidates),
        dead_weight_streak=signals.dead_weight_streak,
        cascade_density=signals.cascade_density,
        phantom_rate=signals.phantom_rate,
        red_streak=signals.red_streak,
        recommendation=signals.recommendation,
    )


def _turn_metrics_from(
    *,
    scores: GroundednessScores,
    code_result: CodeLaneResult,
) -> TurnMetrics:
    attribution = code_result.file_attribution
    file_updates = file_attribution_to_turn_inputs(
        [
            {
                "path": entry.path,
                "owner_share": entry.owner_share,
                "query_owner_share": entry.query_owner_share,
                "dead_weight": entry.dead_weight,
            }
            for entry in attribution.per_file
        ]
    )
    ast_result = code_result.ast
    cascade = code_result.nli_cascade
    # On the code lane the headline groundedness is the calibrated
    # composite score (range [0, 1]); fall back to the legacy
    # ``groundedness_v2`` field if downstream mappers expose it so the
    # session EMA and drift signals remain meaningful for the RAG lane
    # too once it opts in.
    groundedness = getattr(scores, "groundedness_v2", None)
    if groundedness is None:
        groundedness = getattr(scores, "composite_phantom_score", None)
    return TurnMetrics(
        groundedness=(None if groundedness is None else float(groundedness)),
        per_token_p10=code_result.scorer.per_token_p10,
        composite=code_result.composite.composite_score,
        dead_weight_ratio=attribution.dead_weight_ratio,
        cascade_fired=bool(cascade and cascade.triggered),
        phantom_verdict=(
            None if ast_result is None else bool(ast_result.ast_phantom_verdict)
        ),
        phantom_probability=(
            None
            if cascade is None
            else float(cascade.nli_contradiction_prob_max)
        ),
        risk_band=_risk_band_from_scores(scores),
        file_updates=file_updates,
    )


def _risk_band_for_rollup(response: RollupResponse) -> str:
    """Coarse session-level risk banding for the rollup heatmap.

    The rollup is a summary, not a scored turn, so we synthesise a
    single risk band from ``model_drift_pct`` + ``noise_pct`` using the
    same thresholds the token/file bands document.
    """
    drift = float(response.model_drift_pct or 0.0)
    noise = float(response.noise_pct or 0.0)
    worst = max(drift, noise)
    if worst >= 0.50:
        return "red"
    if worst >= 0.25:
        return "amber"
    return "green"


def _risk_band_from_scores(scores: GroundednessScores) -> Optional[str]:
    band = getattr(scores, "risk_band", None)
    if band is None:
        return None
    return str(band.value) if hasattr(band, "value") else str(band)


def _file_attribution_to_wire(
    result: Any,
    *,
    emit_chunk_ownership: bool = False,
) -> Optional[FileAttributionDiagnostics]:
    """Project a :class:`FileAttributionResult` into the wire model.

    Shared between lanes: the RAG scorer returns a
    :class:`latence_trace.core.attribution.file_attribution.FileAttributionResult`
    directly; the code lane already goes through ``_code_lane_diagnostics``
    which builds its own wire model. This helper keeps the projection in
    one place so both lanes produce identical-shape JSON.
    """
    if result is None:
        return None

    per_file = [
        CodeLanePerFileUsage(
            path=rec.path,
            n_units=int(rec.n_units),
            used=int(rec.used),
            uncertain=int(rec.uncertain),
            unused=int(rec.unused),
            coverage=float(rec.coverage),
            mean_score=float(rec.mean_score),
            max_evidence=float(rec.max_evidence),
            owner_tokens=int(rec.owner_tokens),
            owner_share=float(rec.owner_share),
            query_owner_tokens=int(rec.query_owner_tokens),
            query_owner_share=float(rec.query_owner_share),
            dead_weight=bool(rec.dead_weight),
            reason_codes=[rc.value for rc in rec.reason_codes],
            dominating_peer=rec.dominating_peer,
        )
        for rec in result.per_file
    ]
    per_unit = (
        [
            CodeLanePerUnitOwnership(
                support_id=rec.support_id,
                path=rec.path,
                unit_index=int(rec.unit_index),
                max_cos=float(rec.max_cos),
                response_owner_count=int(rec.response_owner_count),
                query_owner_count=int(rec.query_owner_count),
                response_owner_share=float(rec.response_owner_share),
                query_owner_share=float(rec.query_owner_share),
                offset_start=rec.offset_start,
                offset_end=rec.offset_end,
                usage_state=rec.usage_state,
            )
            for rec in result.per_unit
        ]
        if emit_chunk_ownership
        else []
    )
    return CodeLaneFileAttribution(
        per_file=per_file,
        per_unit=per_unit,
        dead_weight_files=list(result.dead_weight_files),
        dead_weight_ratio=float(result.dead_weight_ratio),
        n_files=int(result.n_files),
        n_response_tokens=int(result.n_response_tokens),
        n_query_tokens=int(result.n_query_tokens),
        min_owner_share=float(result.min_owner_share),
        coverage_threshold=float(result.coverage_threshold),
        low_cosine_threshold=float(result.low_cosine_threshold),
        reason_code_histogram=dict(result.reason_code_histogram or {}),
    )


def _build_heatmap_fields(
    *,
    heatmap_format: str,
    scores_dict: Mapping[str, Any],
    file_attribution: Optional[FileAttributionDiagnostics],
    response_tokens: Sequence[Any],
    session_signals: Optional[SessionSignalsPayload],
) -> Tuple[Optional[HeatmapPayload], Optional[str]]:
    """Build the (heatmap, heatmap_html) pair for the response.

    ``heatmap_format`` follows ``GroundednessRequest.heatmap_format`` —
    ``"none"`` returns ``(None, None)``, ``"data"`` returns
    ``(HeatmapPayload, None)``, and ``"html"`` returns
    ``(HeatmapPayload, html_fragment)``.
    """
    fmt = (heatmap_format or "data").lower()
    if fmt == "none":
        return None, None
    payload = build_heatmap(
        scores_dict=scores_dict,
        file_attribution=file_attribution,
        response_tokens=response_tokens,
        session_signals=session_signals,
    )
    if fmt == "html":
        return payload, render_heatmap_html(payload)
    return payload, None


def _code_lane_diagnostics(
    *,
    code_result: CodeLaneResult,
    emit_chunk_ownership: bool,
) -> CodeLaneDiagnostics:
    """Project a :class:`CodeLaneResult` into the API diagnostics payload."""
    composite_result = code_result.composite
    composite_diag = CodeLaneCompositeDiagnostics(
        kind=composite_result.kind,
        composite_score=float(composite_result.composite_score),
        phantom_probability=float(composite_result.phantom_probability),
        verdict=bool(composite_result.verdict),
        threshold=float(composite_result.threshold),
        contributions=[
            CodeLaneCompositeContributions(name=name, value=float(value))
            for name, value in (composite_result.contributions or {}).items()
        ],
    )

    ast_diag: Optional[CodeLaneAstDiagnostics] = None
    if code_result.ast is not None:
        ast_diag = CodeLaneAstDiagnostics(
            language=code_result.ast.language,
            parser_backend=code_result.ast.parser_backend,
            ast_literal_drift_count=int(code_result.ast.ast_literal_drift_count),
            ast_phantom_symbol_count=int(code_result.ast.ast_phantom_symbol_count),
            ast_phantom_verdict=bool(code_result.ast.ast_phantom_verdict),
            drift_symbols=list(code_result.ast.drift_symbols),
            phantom_symbols=list(code_result.ast.phantom_symbols),
            latency_ms=float(code_result.ast.latency_ms),
        )

    ln_diag: Optional[CodeLaneLiteralNovelty] = None
    if code_result.literal_novelty is not None:
        ln_diag = CodeLaneLiteralNovelty(
            literal_novelty_min=float(code_result.literal_novelty.literal_novelty_min),
            literal_novelty_mean=float(code_result.literal_novelty.literal_novelty_mean),
            literal_novelty_count=int(code_result.literal_novelty.literal_novelty_count),
            missing_literal_count=int(
                code_result.literal_novelty.missing_literal_count
            ),
        )

    nli_diag: Optional[CodeLaneNLICascade] = None
    if code_result.nli_cascade is not None:
        nli_diag = CodeLaneNLICascade(
            triggered=bool(code_result.nli_cascade.triggered),
            skipped_reason=code_result.nli_cascade.skipped_reason,
            nli_contradiction_prob_max=float(
                code_result.nli_cascade.nli_contradiction_prob_max
            ),
            nli_entailment_prob_mean=float(
                code_result.nli_cascade.nli_entailment_prob_mean
            ),
            nli_aggregate=(
                None
                if code_result.nli_cascade.nli_aggregate is None
                else float(code_result.nli_cascade.nli_aggregate)
            ),
            claim_count=int(code_result.nli_cascade.claim_count),
            verified_claim_count=int(code_result.nli_cascade.verified_claim_count),
            latency_ms=float(code_result.nli_cascade.latency_ms),
        )

    fa = code_result.file_attribution
    per_file = [
        CodeLanePerFileUsage(
            path=rec.path,
            n_units=int(rec.n_units),
            used=int(rec.used),
            uncertain=int(rec.uncertain),
            unused=int(rec.unused),
            coverage=float(rec.coverage),
            mean_score=float(rec.mean_score),
            max_evidence=float(rec.max_evidence),
            owner_tokens=int(rec.owner_tokens),
            owner_share=float(rec.owner_share),
            query_owner_tokens=int(rec.query_owner_tokens),
            query_owner_share=float(rec.query_owner_share),
            dead_weight=bool(rec.dead_weight),
            reason_codes=[rc.value for rc in rec.reason_codes],
            dominating_peer=rec.dominating_peer,
        )
        for rec in fa.per_file
    ]
    per_unit = (
        [
            CodeLanePerUnitOwnership(
                support_id=rec.support_id,
                path=rec.path,
                unit_index=int(rec.unit_index),
                max_cos=float(rec.max_cos),
                response_owner_count=int(rec.response_owner_count),
                query_owner_count=int(rec.query_owner_count),
                response_owner_share=float(rec.response_owner_share),
                query_owner_share=float(rec.query_owner_share),
                offset_start=rec.offset_start,
                offset_end=rec.offset_end,
                usage_state=rec.usage_state,
            )
            for rec in fa.per_unit
        ]
        if emit_chunk_ownership
        else []
    )

    fa_diag = CodeLaneFileAttribution(
        per_file=per_file,
        per_unit=per_unit,
        dead_weight_files=list(fa.dead_weight_files),
        dead_weight_ratio=float(fa.dead_weight_ratio),
        n_files=int(fa.n_files),
        n_response_tokens=int(fa.n_response_tokens),
        n_query_tokens=int(fa.n_query_tokens),
        min_owner_share=float(fa.min_owner_share),
        coverage_threshold=float(fa.coverage_threshold),
        low_cosine_threshold=float(fa.low_cosine_threshold),
        reason_code_histogram=dict(fa.reason_code_histogram),
    )

    return CodeLaneDiagnostics(
        config=code_result.config.as_dict(),
        total_latency_ms=float(code_result.total_latency_ms),
        component_latency_ms={
            k: float(v) for k, v in code_result.component_latency_ms.items()
        },
        composite=composite_diag,
        ast=ast_diag,
        literal_novelty=ln_diag,
        nli_cascade=nli_diag,
        file_attribution=fa_diag,
    )


def _code_lane_scores(
    *,
    code_result: CodeLaneResult,
    coverage_threshold: float,
) -> GroundednessScores:
    """Project code-lane diagnostics into the top-level scores payload.

    Fills in the primary metric + back-compatible coverage fields used
    by the RAG UI plus the new code-specific headline numbers (composite
    verdict, AST phantom counts, literal novelty, dead-weight ratio) so
    IDE plugins can render a one-bit "green / yellow / red" indicator
    without reading the nested diagnostics bundle.
    """
    scorer = code_result.scorer
    composite = code_result.composite
    fa = code_result.file_attribution
    literal = code_result.literal_novelty
    ast_res = code_result.ast
    nli = code_result.nli_cascade

    used_units = sum(1 for s in scorer.usage_states if s == "used")
    unused_units = sum(1 for s in scorer.usage_states if s == "unused")
    uncertain_units = sum(1 for s in scorer.usage_states if s == "uncertain")
    total_units = max(1, len(scorer.usage_states))

    return GroundednessScores(
        primary_name="composite_phantom_score",
        primary_score=float(composite.composite_score),
        reverse_context=float(scorer.reverse_context),
        literal_guarded=float(scorer.literal_guard),
        context_coverage_ratio=float(used_units) / total_units,
        context_coverage_threshold=float(coverage_threshold),
        support_units_used=int(used_units),
        support_units_total=int(total_units),
        support_units_usage_used=int(used_units),
        support_units_unused=int(unused_units),
        support_units_uncertain=int(uncertain_units),
        context_usage_ratio=float(used_units) / total_units,
        context_unused_ratio=float(unused_units) / total_units,
        context_uncertain_ratio=float(uncertain_units) / total_units,
        composite_phantom_score=float(composite.composite_score),
        composite_phantom_probability=float(composite.phantom_probability),
        composite_phantom_verdict=bool(composite.verdict),
        literal_novelty_min=(
            None if literal is None else float(literal.literal_novelty_min)
        ),
        literal_novelty_missing_count=(
            None if literal is None else int(literal.missing_literal_count)
        ),
        ast_phantom_symbol_count=(
            None if ast_res is None else int(ast_res.ast_phantom_symbol_count)
        ),
        ast_literal_drift_count=(
            None if ast_res is None else int(ast_res.ast_literal_drift_count)
        ),
        ast_phantom_verdict=(
            None if ast_res is None else bool(ast_res.ast_phantom_verdict)
        ),
        nli_contradiction_prob_max=(
            None
            if nli is None or not nli.triggered
            else float(nli.nli_contradiction_prob_max)
        ),
        nli_cascade_triggered=(None if nli is None else bool(nli.triggered)),
        dead_weight_ratio=float(fa.dead_weight_ratio),
        dead_weight_file_count=int(len(fa.dead_weight_files)),
    )


# ---------------------------------------------------------------------------
# Default encoder factory (vLLM-factory or local pylate)
# ---------------------------------------------------------------------------


# Multilingual ModernColBERT default. SauerkrautLM-Multi-Reason-ModernColBERT
# is a German+English ColBERT fine-tune that keeps the standard ModernColBERT
# tokenizer and IO contract, so it is a drop-in replacement for the previous
# English-only GTE-ModernColBERT-v1 default. We load it in bf16 by default to
# halve VRAM with negligible quality impact on cosine MaxSim. Override at
# runtime with VOYAGER_GROUNDEDNESS_MODEL or per-request ``model``.
DEFAULT_GROUNDEDNESS_MODEL = "VAGOsolutions/SauerkrautLM-Multi-Reason-ModernColBERT"
DEFAULT_GROUNDEDNESS_TORCH_DTYPE = "bfloat16"


def _resolve_torch_dtype(name: Optional[str]) -> Optional[Any]:
    """Map a string dtype to a real ``torch.dtype`` or ``None``.

    Returns ``None`` when ``name`` is empty or ``"default"``/``"none"``,
    which lets pylate pick the model card's preferred precision (typically
    fp32 on CPU and fp16/bf16 on GPU per the model config).
    """

    if not name:
        return None
    label = name.strip().lower()
    if label in {"", "default", "none", "auto"}:
        return None
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
        "full": torch.float32,
    }
    if label not in mapping:
        raise ValidationError(
            f"Unsupported VOYAGER_GROUNDEDNESS_TORCH_DTYPE='{name}'. "
            "Use one of bfloat16, float16, float32, or 'default'."
        )
    return mapping[label]


def _default_encoder_factory(device: str) -> Callable[[Optional[str]], Any]:
    # PA4: now that /groundedness runs in the FastAPI threadpool with a
    # bounded inflight semaphore, two concurrent cold-start requests can
    # both arrive here before the cache is primed. Serialize the
    # construction so only the first caller pays the healthcheck cost
    # and every subsequent request hits the warm cache.
    cache: Dict[str, Any] = {}
    cache_lock = threading.Lock()

    def factory(model_name: Optional[str]) -> Any:
        vllm_endpoint = os.environ.get("VOYAGER_GROUNDEDNESS_VLLM_ENDPOINT")
        if vllm_endpoint:
            resolved_model_name = (
                model_name
                or os.environ.get("VOYAGER_GROUNDEDNESS_VLLM_MODEL")
                or os.environ.get("VOYAGER_GROUNDEDNESS_MODEL")
                or os.environ.get("VOYAGER_ENCODE_MODEL")
                or DEFAULT_GROUNDEDNESS_MODEL
            )
            cache_key = f"vllm_factory:{vllm_endpoint}:{resolved_model_name}"
            with cache_lock:
                cached = cache.get(cache_key)
                if cached is not None:
                    return cached
                try:
                    from latence_trace.providers.encoders import VllmFactoryModernColBERTProvider
                except ImportError as exc:
                    raise ValidationError(
                        "Groundedness vLLM integration requires the http dependencies to be installed."
                    ) from exc
                try:
                    provider = VllmFactoryModernColBERTProvider(
                        endpoint=vllm_endpoint,
                        model=resolved_model_name,
                        timeout=_env_float("VOYAGER_GROUNDEDNESS_VLLM_TIMEOUT", 60.0),
                        health_timeout=_env_float("VOYAGER_GROUNDEDNESS_VLLM_HEALTH_TIMEOUT", 10.0),
                        batch_size=_env_int("VOYAGER_GROUNDEDNESS_VLLM_BATCH_SIZE", 16),
                        max_concurrency=_env_int("VOYAGER_GROUNDEDNESS_VLLM_MAX_CONCURRENCY", 8),
                    )
                    provider.healthcheck()
                except Exception as exc:
                    raise ValidationError(
                        f"Failed to initialize groundedness vLLM provider '{resolved_model_name}' at '{vllm_endpoint}': {exc}"
                    ) from exc
                cache[cache_key] = provider
                return provider

        resolved = (
            model_name
            or os.environ.get("VOYAGER_GROUNDEDNESS_MODEL")
            or os.environ.get("VOYAGER_ENCODE_MODEL")
            or DEFAULT_GROUNDEDNESS_MODEL
        )
        dtype_name = os.environ.get("VOYAGER_GROUNDEDNESS_TORCH_DTYPE", DEFAULT_GROUNDEDNESS_TORCH_DTYPE)
        torch_dtype = _resolve_torch_dtype(dtype_name)
        cache_key = f"local:{resolved}:{dtype_name}"
        with cache_lock:
            cached = cache.get(cache_key)
            if cached is not None:
                return cached
            try:
                from pylate import models
            except ImportError as exc:
                raise ValidationError(
                    "Local groundedness encoder requires pylate. Install pylate or set VOYAGER_GROUNDEDNESS_VLLM_ENDPOINT."
                ) from exc
            model_kwargs: Dict[str, Any] = {}
            if torch_dtype is not None:
                model_kwargs["torch_dtype"] = torch_dtype
            try:
                provider = models.ColBERT(
                    model_name_or_path=resolved,
                    device=device,
                    do_query_expansion=False,
                    trust_remote_code=True,
                    model_kwargs=model_kwargs or None,
                )
            except TypeError:
                # Older pylate releases without model_kwargs / trust_remote_code
                # kwargs - fall back to the minimal signature.
                try:
                    provider = models.ColBERT(
                        model_name_or_path=resolved,
                        device=device,
                        do_query_expansion=False,
                    )
                except Exception as exc:
                    raise ValidationError(f"Failed to load groundedness model '{resolved}': {exc}") from exc
            except Exception as exc:
                raise ValidationError(f"Failed to load groundedness model '{resolved}': {exc}") from exc
            cache[cache_key] = provider
            return provider

    return factory


__all__ = [
    "ChunkResolver",
    "DEFAULT_GROUNDEDNESS_MODEL",
    "DEFAULT_GROUNDEDNESS_TORCH_DTYPE",
    "DEFAULT_PROFILE",
    "GroundednessService",
    "NotFoundError",
    "PROFILE_ENV_PRESETS",
    "PROFILE_NAMES",
    "ProfileApplication",
    "ResolvedChunk",
    "ServiceError",
    "ValidationError",
    "apply_profile",
]
