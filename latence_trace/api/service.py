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
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch

from latence_trace.api.models import (
    CollectionKind,
    GroundednessEligibility,
    GroundednessRequest,
    GroundednessResponse,
)
from latence_trace.core.groundedness import (
    SupportUnitInput,
    default_null_bank_texts,
    encode_texts,
    partition_support_units,
    provider_token_limit,
    score_groundedness,
    score_groundedness_chunked,
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
    status_code = 500
    error_code = "service_error"


class ValidationError(ServiceError):
    status_code = 400
    error_code = "validation_error"


class NotFoundError(ServiceError):
    status_code = 404
    error_code = "not_found"


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
        self._nli_provider: Any = None
        self._nli_provider_resolved = False
        self._nli_reranker: Any = None
        self._nli_reranker_resolved = False

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
        try:
            self._nli_reranker = nli_resolve_default_reranker()
        except Exception as exc:
            logger.warning("nli_reranker_resolve_failed", extra={"error": str(exc)})
            self._nli_reranker = None
        self._nli_reranker_resolved = True
        return self._nli_reranker

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
            return []
        bank: List[torch.Tensor] = [torch.as_tensor(t, dtype=torch.float32) for t in tensors]
        self._cached_groundedness_null_banks[cache_key] = bank
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
        mode = "chunk_ids" if request.chunk_ids else "raw_context"
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

        response_embeddings = encode_texts(
            provider,
            [request.response_text],
            is_query=False,
            prompt_name=request.document_prompt_name,
        )[0]
        response_tokens = tokenize_text(
            provider,
            request.response_text,
            expected_len=int(response_embeddings.shape[0]),
            is_query=False,
        )

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
            scored = score_groundedness(
                support_units=support_units,
                response_embeddings=response_embeddings,
                response_tokens=response_tokens,
                query_embeddings=query_embeddings,
                query_tokens=query_tokens,
                evidence_limit=request.evidence_limit,
                primary_metric=request.primary_metric.value,
                debug_dense_matrices=request.debug_dense_matrices,
                null_bank_embeddings=null_bank_embeddings or None,
                response_text=request.response_text,
                **nli_kwargs,
            )
        else:
            support_batches = partition_support_units(
                support_units,
                batch_size=_env_int("VOYAGER_GROUNDEDNESS_SCORE_BATCH_UNITS", 64),
            )
            scored = score_groundedness_chunked(
                support_batches=support_batches,
                response_embeddings=response_embeddings,
                response_tokens=response_tokens,
                query_embeddings=query_embeddings,
                query_tokens=query_tokens,
                evidence_limit=request.evidence_limit,
                primary_metric=request.primary_metric.value,
                debug_dense_matrices=request.debug_dense_matrices,
                null_bank_embeddings=null_bank_embeddings or None,
                response_text=request.response_text,
                **nli_kwargs,
            )
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
        )


# ---------------------------------------------------------------------------
# Default encoder factory (vLLM-factory or local pylate)
# ---------------------------------------------------------------------------


def _default_encoder_factory(device: str) -> Callable[[Optional[str]], Any]:
    cache: Dict[str, Any] = {}

    def factory(model_name: Optional[str]) -> Any:
        vllm_endpoint = os.environ.get("VOYAGER_GROUNDEDNESS_VLLM_ENDPOINT")
        if vllm_endpoint:
            resolved_model_name = (
                model_name
                or os.environ.get("VOYAGER_GROUNDEDNESS_VLLM_MODEL")
                or os.environ.get("VOYAGER_GROUNDEDNESS_MODEL")
                or os.environ.get("VOYAGER_ENCODE_MODEL")
            )
            if not resolved_model_name:
                raise ValidationError(
                    "VOYAGER_GROUNDEDNESS_VLLM_ENDPOINT is set but no model id was provided. "
                    "Set VOYAGER_GROUNDEDNESS_VLLM_MODEL, VOYAGER_GROUNDEDNESS_MODEL, "
                    "VOYAGER_ENCODE_MODEL, or pass request.model."
                )
            cache_key = f"vllm_factory:{vllm_endpoint}:{resolved_model_name}"
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

        resolved = model_name or os.environ.get("VOYAGER_GROUNDEDNESS_MODEL") or os.environ.get("VOYAGER_ENCODE_MODEL")
        if not resolved:
            raise ValidationError(
                "No groundedness encoder configured. Set VOYAGER_GROUNDEDNESS_MODEL or pass request.model."
            )
        cached = cache.get(resolved)
        if cached is not None:
            return cached
        try:
            from pylate import models
        except ImportError as exc:
            raise ValidationError(
                "Local groundedness encoder requires pylate. Install pylate or set VOYAGER_GROUNDEDNESS_VLLM_ENDPOINT."
            ) from exc
        try:
            provider = models.ColBERT(
                model_name_or_path=resolved,
                device=device,
                do_query_expansion=False,
            )
        except Exception as exc:
            raise ValidationError(f"Failed to load groundedness model '{resolved}': {exc}") from exc
        cache[resolved] = provider
        return provider

    return factory


__all__ = [
    "ChunkResolver",
    "GroundednessService",
    "NotFoundError",
    "ResolvedChunk",
    "ServiceError",
    "ValidationError",
]
