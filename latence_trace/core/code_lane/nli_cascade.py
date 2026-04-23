"""Ambiguity-triggered NLI cascade for the code lane.

Motivation
----------
The existing RAG path runs NLI unconditionally (driven by the
``balanced`` / ``quality`` profile fusion weights). For the code lane
we only want to pay the NLI cost when the composite phantom-guard is
genuinely ambiguous — typically the ``[0.65, 0.90]`` band. That keeps
the code lane under the ~150 ms p95 SLO while still catching the
failure mode the composite is weakest on (identifier swaps that the
literal guard cannot tell apart from overlap).

The module *reuses* :func:`latence_trace.core.nli.verify_claims` and
:func:`latence_trace.core.nli.aggregate_nli_score` so the actual
entailment algorithm, batching, and reranker code are identical to the
RAG lane. Only the trigger policy is new.

Contract
--------
``NLICascade(provider, reranker=None).run(response_text, support_units,
                                           composite_score,
                                           lower=0.65, upper=0.90)``
returns an :class:`NLICascadeResult` with the contradiction probability
maximum over every verified claim. Callers OR this into the final
composite only when ``result.triggered`` is True.

The cascade is thread-safe — the underlying provider is expected to be
a shared vLLM client (an ``httpx.AsyncClient`` under the hood) and the
verify call itself is already reentrant.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional, Sequence, Tuple

from latence_trace.core.nli import (
    ClaimVerification,
    aggregate_nli_score,
    default_max_batch,
    default_max_claims,
    default_max_latency_ms,
    default_premise_concat_word_budget,
    default_top_k_premises,
    verify_claims,
)

logger = logging.getLogger(__name__)


DEFAULT_LOWER_BAND: float = 0.65
DEFAULT_UPPER_BAND: float = 0.90
DEFAULT_MAX_CLAIMS: int = 4
DEFAULT_TOP_K_PREMISES: int = 3


@dataclass
class NLICascadeResult:
    """Diagnostics returned by the cascade."""

    triggered: bool
    skipped_reason: Optional[str] = None
    nli_contradiction_prob_max: float = 0.0
    nli_entailment_prob_mean: float = 0.0
    nli_aggregate: Optional[float] = None
    claim_count: int = 0
    verified_claim_count: int = 0
    verifications: List[ClaimVerification] = field(default_factory=list)
    latency_ms: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "triggered": bool(self.triggered),
            "skipped_reason": self.skipped_reason,
            "nli_contradiction_prob_max": float(self.nli_contradiction_prob_max),
            "nli_entailment_prob_mean": float(self.nli_entailment_prob_mean),
            "nli_aggregate": (
                None if self.nli_aggregate is None else float(self.nli_aggregate)
            ),
            "claim_count": int(self.claim_count),
            "verified_claim_count": int(self.verified_claim_count),
            "latency_ms": float(self.latency_ms),
        }


class NLICascade:
    """Ambiguity-gated NLI wrapper around :func:`verify_claims`.

    The cascade holds a reference to an already-resolved ``NLIProvider``
    plus an optional cross-encoder reranker (matches the RAG
    ``_get_nli_provider`` / ``_get_nli_reranker`` pattern). Because
    ``verify_claims`` runs on the caller's thread it releases the GIL
    during the HTTPX round-trip — so multiple code-lane requests can
    overlap without a dedicated executor.
    """

    def __init__(
        self,
        provider: Any,
        *,
        reranker: Optional[Any] = None,
        lower_band: float = DEFAULT_LOWER_BAND,
        upper_band: float = DEFAULT_UPPER_BAND,
        max_claims: int = DEFAULT_MAX_CLAIMS,
        top_k_premises: int = DEFAULT_TOP_K_PREMISES,
        max_batch: Optional[int] = None,
        max_latency_ms: Optional[float] = None,
    ) -> None:
        if lower_band > upper_band:
            raise ValueError("lower_band must be <= upper_band")
        self.provider = provider
        self.reranker = reranker
        self.lower_band = float(lower_band)
        self.upper_band = float(upper_band)
        self.max_claims = int(max_claims or default_max_claims())
        self.top_k_premises = int(top_k_premises or default_top_k_premises())
        self.max_batch = int(max_batch or default_max_batch())
        self.max_latency_ms = float(
            max_latency_ms if max_latency_ms is not None else default_max_latency_ms()
        )

    def should_trigger(self, composite_score: float) -> bool:
        """Return True when ``composite_score`` falls in the ambiguity band."""
        return self.lower_band <= composite_score <= self.upper_band

    def run(
        self,
        *,
        response_text: str,
        support_units: Sequence[Any],
        composite_score: float,
        force: bool = False,
    ) -> NLICascadeResult:
        """Run NLI when ambiguous, otherwise short-circuit.

        Parameters
        ----------
        response_text:
            Same ``response_text`` that fed the scorer; passed as-is to
            :func:`verify_claims` which handles claim splitting.
        support_units:
            Sequence of objects carrying ``.text`` and ``.support_id``
            attributes (the code lane builds these from
            :class:`~latence_trace.core.code_lane.types.SupportUnitPack`
            plus a lightweight shim; the RAG lane passes
            :class:`~latence_trace.core.groundedness.SupportUnitInput`).
        composite_score:
            Scalar composite phantom-guard score. Used only for the
            trigger decision — the NLI call itself is independent.
        force:
            Skip the band check and always run. Used by the ablation
            harness.
        """
        if self.provider is None:
            return NLICascadeResult(
                triggered=False, skipped_reason="nli_provider_unavailable"
            )
        if not force and not self.should_trigger(composite_score):
            reason = (
                "composite_above_upper_band"
                if composite_score > self.upper_band
                else "composite_below_lower_band"
            )
            return NLICascadeResult(triggered=False, skipped_reason=reason)

        start = time.perf_counter()
        try:
            verifications, _warnings = verify_claims(
                response_text,
                support_units,
                self.provider,
                max_claims=self.max_claims,
                top_k_premises=self.top_k_premises,
                max_batch=self.max_batch,
                max_latency_ms=self.max_latency_ms,
                reranker=self.reranker,
            )
        except Exception as exc:
            logger.warning("nli_cascade_failed", extra={"error": str(exc)})
            return NLICascadeResult(
                triggered=True,
                skipped_reason=f"nli_exception:{type(exc).__name__}",
                latency_ms=(time.perf_counter() - start) * 1000.0,
            )

        verified = [v for v in verifications if not v.skipped]
        if verified:
            contradiction_max = max(v.contradiction for v in verified)
            entailment_mean = sum(v.entailment for v in verified) / float(len(verified))
        else:
            contradiction_max = 0.0
            entailment_mean = 0.0

        aggregate = aggregate_nli_score(verifications)
        latency_ms = (time.perf_counter() - start) * 1000.0
        return NLICascadeResult(
            triggered=True,
            nli_contradiction_prob_max=float(contradiction_max),
            nli_entailment_prob_mean=float(entailment_mean),
            nli_aggregate=(None if aggregate is None else float(aggregate)),
            claim_count=len(verifications),
            verified_claim_count=len(verified),
            verifications=list(verifications),
            latency_ms=float(latency_ms),
        )
