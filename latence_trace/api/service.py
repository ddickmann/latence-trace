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
    AttributionMode,
    CollectionKind,
    GroundednessEligibility,
    GroundednessRequest,
    GroundednessResponse,
    GroundednessScores,
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
    tokenize_text,
)
from latence_trace.core.nli import (
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

logger = logging.getLogger(__name__)


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
        # Concurrent first requests must not double-resolve the NLI provider
        # or cross-encoder reranker - both are expensive HuggingFace loads.
        self._nli_provider_lock = threading.Lock()
        self._nli_reranker_lock = threading.Lock()
        self._null_bank_lock = threading.Lock()

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

    def _get_nli_provider(self):
        if not nli_is_enabled():
            return None
        if self._nli_provider_resolved:
            return self._nli_provider
        with self._nli_provider_lock:
            if self._nli_provider_resolved:
                return self._nli_provider
            try:
                self._nli_provider = nli_resolve_default_provider()
            except Exception as exc:
                logger.warning("nli_resolve_failed", extra={"error": str(exc)})
                self._nli_provider = None
            self._nli_provider_resolved = True
            return self._nli_provider

    def _get_nli_reranker(self):
        if not nli_is_enabled():
            return None
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

    # --- main entry point -----------------------------------------------------

    def groundedness(self, request: GroundednessRequest) -> GroundednessResponse:
        start = time.perf_counter()

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
            segments = segment_text(
                request.raw_context or "",
                request.segmentation_mode.value,
                provider=provider,
                chunk_token_budget=request.raw_context_chunk_tokens,
            )
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
        nli_provider = self._get_nli_provider()
        nli_reranker = self._get_nli_reranker() if nli_provider is not None else None
        nli_kwargs: Dict[str, Any] = {}
        if nli_provider is not None:
            nli_kwargs = {
                "nli_provider": nli_provider,
                "nli_max_claims": nli_default_max_claims(),
                "nli_top_k_premises": nli_default_top_k_premises(),
                "nli_max_batch": nli_default_max_batch(),
                "nli_max_latency_ms": nli_default_max_latency_ms(),
                "nli_reranker": nli_reranker,
                "nli_concat_premises": nli_is_premise_concat_enabled(),
                "nli_premise_concat_word_budget": nli_default_premise_concat_word_budget(),
                "nli_use_atomic_claims": nli_is_atomic_enabled(),
                "fusion_weights": nli_fusion_weights_from_env(),
            }
        if request.verification_samples:
            nli_kwargs.setdefault("fusion_weights", nli_fusion_weights_from_env())
            nli_kwargs["verification_samples"] = list(request.verification_samples)
            nli_kwargs["semantic_entropy_enabled"] = True
        if request.risk_band_stratum:
            nli_kwargs["risk_band_stratum"] = str(request.risk_band_stratum)
        if request.content_type:
            nli_kwargs["content_type"] = str(request.content_type)
        if request.raw_context:
            nli_kwargs["structured_support_text"] = request.raw_context

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
        return GroundednessResponse(
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
            semantic_entropy_diagnostics=scored.get("semantic_entropy_diagnostics"),
            structured_diagnostics=scored.get("structured_diagnostics"),
            time_ms=elapsed_ms,
            attribution_mode=request.attribution_mode,
        )

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
            attribution_mode=request.attribution_mode,
            reason=reason,
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
