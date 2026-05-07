"""Language-aware NLI provider registry.

Resolves the right :class:`latence_trace.core.nli.NLIProvider` for a
given request language. Selection is env-driven so callers never need
to know which backend serves which language; flipping a single
endpoint env var swaps a transformers-direct provider for a vLLM-served
one with no behaviour change.

Selection priority per language ``L`` (where L is one of ``en``, ``de``,
or ``None`` for the legacy single-NLI behaviour):

0. **Granite Guardian** — ``LATENCE_TRACE_GUARDIAN_ENDPOINT`` set.
   :class:`~latence_trace.providers.granite_guardian.GraniteGuardianNLIProvider`
   is multilingual and supersedes all per-language providers. It talks
   to a vLLM ``/v1/completions`` endpoint and extracts continuous
   grounding scores from token logprobs.
1. **Per-language vLLM endpoint** — ``LATENCE_TRACE_NLI_<L>_ENDPOINT``
   set. The provider class is picked from
   ``LATENCE_TRACE_NLI_<L>_PROTOCOL`` which defaults to ``pooling``
   (``VllmFactoryNLIProvider`` — the wire shape today's mDeBERTa server
   and the future MiniCheck BYOP plugin both use). Set to ``classify``
   for vLLM-native /v1/classify servers (bge-m3-zeroshot today).
2. **Per-language transformers fallback** — when no endpoint is
   configured we boot the matching in-process model so the system stays
   SOTA without any vLLM plugin work. ``en`` defaults to
   :class:`MiniCheckNLIProvider`, ``de`` to
   :class:`BgeM3ZeroShotNLIProvider`. Operators can override via
   ``LATENCE_TRACE_NLI_<L>_TRANSFORMERS_MODEL``.
3. **Legacy single-NLI provider** — falls through to
   :func:`latence_trace.core.nli.resolve_default_provider` (mDeBERTa
   over /pooling, or its HuggingFace fallback) so deployments that
   haven't enabled the dual-NLI setup behave exactly like before.

Resolved providers are cached process-wide, keyed by ``language``, and
re-used across requests. Failed initialisations (e.g. server unhealthy)
are logged and the next bullet down is tried; if everything fails we
return ``None`` and the caller surfaces ``quality_profile_nli_unavailable``
the same way it does today.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Tracked languages today. Adding a new one is a tuple here + matching
# env-var fan-out + a default model.
_SUPPORTED_LANGUAGES: tuple[str, ...] = ("en", "de")

# In-process defaults from the SOTA Phase 0 verdict (PROOF.md):
# * MiniCheck-Flan-T5-Large is the published LLM-AggreFact SOTA
#   (Tang et al. 2024) for grounded fact-checking in English.
# * bge-m3-zeroshot-v2.0 is the strongest off-the-shelf multilingual
#   zero-shot NLI model for German.
_DEFAULT_TRANSFORMERS_MODEL: dict[str, str] = {
    "en": "lytang/MiniCheck-Flan-T5-Large",
    "de": "MoritzLaurer/bge-m3-zeroshot-v2.0",
}

# Map a language to its in-process provider class. New languages land
# here when we extend coverage beyond {en, de}.
_TRANSFORMERS_PROVIDER_BY_LANGUAGE: dict[str, str] = {
    "en": "MiniCheckNLIProvider",
    "de": "BgeM3ZeroShotNLIProvider",
}

# Lazy per-language cache. ``None`` is a sentinel meaning "tried, failed,
# don't retry" so a downed vLLM server doesn't reload the heavy
# transformers fallback on every request.
_provider_cache: dict[str, Any] = {}
_provider_cache_resolved: dict[str, bool] = {}
_provider_cache_lock = threading.Lock()


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


def _normalise_language(language: Optional[str]) -> str:
    """Map ``language`` to a registry key.

    ``None`` and unsupported languages collapse to the empty string
    which represents the legacy single-provider lane. We keep ``en``
    and ``de`` distinct so the dual-NLI setup picks the right model.
    """

    if language is None:
        return ""
    normalised = str(language).lower().strip()
    if normalised in _SUPPORTED_LANGUAGES:
        return normalised
    return ""


def _build_vllm_provider(language: str) -> Optional[Any]:
    """Try the vLLM endpoint for ``language``; return None if not configured."""

    upper = language.upper()
    endpoint = os.environ.get(f"LATENCE_TRACE_NLI_{upper}_ENDPOINT", "").strip()
    if not endpoint:
        return None
    model = os.environ.get(
        f"LATENCE_TRACE_NLI_{upper}_MODEL",
        _DEFAULT_TRANSFORMERS_MODEL[language],
    )
    protocol = os.environ.get(
        f"LATENCE_TRACE_NLI_{upper}_PROTOCOL", "pooling"
    ).lower()
    timeout = _env_float(f"LATENCE_TRACE_NLI_{upper}_TIMEOUT", 60.0)
    health_timeout = _env_float(
        f"LATENCE_TRACE_NLI_{upper}_HEALTH_TIMEOUT", 10.0
    )
    max_concurrency = _env_int(
        f"LATENCE_TRACE_NLI_{upper}_MAX_CONCURRENCY", 8
    )
    try:
        if protocol == "classify":
            from latence_trace.providers.nli_classify import (
                VllmClassifyNLIProvider,
            )

            provider = VllmClassifyNLIProvider(
                endpoint=endpoint,
                model=model,
                timeout=timeout,
                health_timeout=health_timeout,
                max_concurrency=max_concurrency,
            )
        else:  # pooling — same wire shape as nli_mdeberta + future minicheck_t5
            from latence_trace.providers.nli import VllmFactoryNLIProvider

            provider = VllmFactoryNLIProvider(
                endpoint=endpoint,
                model=model,
                timeout=timeout,
                health_timeout=health_timeout,
                max_concurrency=max_concurrency,
            )
        provider.healthcheck()
        return provider
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "language_nli_vllm_provider_init_failed",
            extra={
                "language": language,
                "endpoint": endpoint,
                "model": model,
                "protocol": protocol,
                "error": str(exc),
            },
        )
        return None


def _build_transformers_provider(language: str) -> Optional[Any]:
    """Boot the in-process MiniCheck / bge-m3-zs fallback for ``language``.

    The provider is only constructed; the model is not loaded until the
    first ``entail`` call so service startup stays fast even when
    multiple languages are configured.
    """

    model_id = os.environ.get(
        f"LATENCE_TRACE_NLI_{language.upper()}_TRANSFORMERS_MODEL",
        _DEFAULT_TRANSFORMERS_MODEL[language],
    )
    provider_name = _TRANSFORMERS_PROVIDER_BY_LANGUAGE[language]
    try:
        from latence_trace.providers import nli_transformers

        cls = getattr(nli_transformers, provider_name)
        return cls(model_id=model_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "language_nli_transformers_provider_init_failed",
            extra={
                "language": language,
                "model": model_id,
                "provider": provider_name,
                "error": str(exc),
            },
        )
        return None


def _build_legacy_provider() -> Optional[Any]:
    """Fall back to the legacy ``resolve_default_provider`` lane."""

    try:
        from latence_trace.core.nli import resolve_default_provider

        return resolve_default_provider()
    except Exception as exc:  # noqa: BLE001
        logger.warning("legacy_nli_resolve_failed", extra={"error": str(exc)})
        return None


def _build_guardian_provider() -> Optional[Any]:
    """Try the Granite Guardian vLLM endpoint; return None if not configured.

    When ``LATENCE_TRACE_GUARDIAN_ENDPOINT`` is set, Granite Guardian
    replaces all per-language NLI providers — it is multilingual and
    handles both English and German out of the box.
    """
    endpoint = os.environ.get("LATENCE_TRACE_GUARDIAN_ENDPOINT", "").strip()
    if not endpoint:
        return None
    model = os.environ.get(
        "LATENCE_TRACE_GUARDIAN_MODEL",
        "latence/granite-4.1-guardian-W8A16",
    )
    timeout = _env_float("LATENCE_TRACE_GUARDIAN_TIMEOUT", 120.0)
    tokenizer_id = os.environ.get(
        "LATENCE_TRACE_GUARDIAN_TOKENIZER", ""
    ).strip() or None
    try:
        from latence_trace.providers.granite_guardian import (
            GraniteGuardianNLIProvider,
        )

        provider = GraniteGuardianNLIProvider(
            endpoint=endpoint,
            model=model,
            timeout=timeout,
            tokenizer_id=tokenizer_id,
        )
        provider.healthcheck()
        return provider
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "granite_guardian_provider_init_failed",
            extra={
                "endpoint": endpoint,
                "model": model,
                "error": str(exc),
            },
        )
        return None


def _resolve_for_language(language: str) -> Optional[Any]:
    # Granite Guardian takes priority — it is multilingual and
    # supersedes the per-language MiniCheck / bge-m3-zs providers.
    guardian = _build_guardian_provider()
    if guardian is not None:
        return guardian

    if language in _SUPPORTED_LANGUAGES:
        provider = _build_vllm_provider(language)
        if provider is not None:
            return provider
        # Per-language transformers fallback only kicks in when the
        # operator has explicitly opted into the dual-NLI setup. We
        # detect opt-in by the presence of *any* per-language env var
        # so deployments that have not flipped the new flag behave
        # exactly like before.
        if _per_language_setup_enabled(language):
            provider = _build_transformers_provider(language)
            if provider is not None:
                return provider
    # Legacy single-provider lane (mDeBERTa or its HF fallback).
    return _build_legacy_provider()


def _per_language_setup_enabled(language: str) -> bool:
    """Return True when the per-language SOTA stack is opted into.

    Reading this flag is what differentiates "user explicitly enabled
    the dual-NLI setup" from "user just hasn't migrated yet" — the
    latter must keep falling through to the legacy single-NLI lane so
    the deploy surface stays unchanged for everyone else.
    """

    upper = language.upper()
    explicit = os.environ.get(f"LATENCE_TRACE_NLI_{upper}_ENABLED", "").lower()
    if explicit in {"1", "true", "yes", "on"}:
        return True
    if explicit in {"0", "false", "no", "off"}:
        return False
    # Implicit opt-in: any per-language endpoint, model override, or
    # transformers-model override turns the lane on.
    for suffix in ("ENDPOINT", "MODEL", "TRANSFORMERS_MODEL"):
        if os.environ.get(f"LATENCE_TRACE_NLI_{upper}_{suffix}", "").strip():
            return True
    # Process-wide opt-in.
    process_wide = os.environ.get("LATENCE_TRACE_NLI_DUAL_ENABLED", "").lower()
    return process_wide in {"1", "true", "yes", "on"}


def resolve_nli_provider(language: Optional[str] = None) -> Optional[Any]:
    """Return the right NLI provider for ``language`` (cached)."""

    key = _normalise_language(language)
    cached_resolved = _provider_cache_resolved.get(key, False)
    if cached_resolved:
        return _provider_cache.get(key)
    with _provider_cache_lock:
        if _provider_cache_resolved.get(key, False):
            return _provider_cache.get(key)
        provider = _resolve_for_language(key)
        _provider_cache[key] = provider
        _provider_cache_resolved[key] = True
        return provider


def reset_cache() -> None:
    """Clear the per-language provider cache.

    Used by tests and by ``service`` when the env config is mutated at
    runtime (e.g. the demo-bridge swapping endpoints between requests
    in dev). Production stays warm — the cache is process-lived.
    """

    with _provider_cache_lock:
        _provider_cache.clear()
        _provider_cache_resolved.clear()


__all__ = [
    "reset_cache",
    "resolve_nli_provider",
]
