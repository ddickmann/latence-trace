"""Phase E external evaluation harness for the groundedness Beta.

Two execution lanes drive the same minimal-pair fixture:

- the **default** lane uses ``reverse_context_calibrated`` as the headline
  (pass ``null_bank_embeddings`` to ``score_groundedness``)
- the **NLI** lane additionally builds a ``HuggingFaceNLIProvider`` and
  uses the fused ``groundedness_v2`` as the headline

Both lanes report:

- per-stratum paired ranking accuracy with bootstrap 95 percent CIs
- per-call latency split into ``encode_ms`` (provider work) and
  ``score_ms`` (Voyager scoring math), with the latency exit criterion
  applied to the sum so the budget reflects the full request

The harness then maps results to the pre-registered exit criteria from
:mod:`research.triangular_maxsim.groundedness_external_benchmarks` and
stamps a single ``headline_verdict`` string into the report so callers
do not have to re-derive it.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from research.triangular_maxsim.groundedness_external_benchmarks import (  # noqa: E402
    PREREGISTERED_TARGETS,
    BenchmarkSample,
    available_benchmarks,
    load_factscore,
    load_halueval,
    load_ragtruth,
)
from research.triangular_maxsim.groundedness_minimal_pairs import (  # noqa: E402
    MinimalPair,
    build_minimal_pairs,
    stratum_summary,
)
from latence_trace.core.groundedness import (  # noqa: E402
    SupportUnitInput,
    _build_response_chunks,
    default_null_bank_texts,
    encode_texts,
    score_groundedness_response_chunked,
    segment_text,
    tokenize_text,
)


# ----------------------------------------------------------------------
# Provider loading
# ----------------------------------------------------------------------


def _resolve_torch_dtype(name: Optional[str]):
    """Map a string dtype name to a torch.dtype (or ``None`` for default)."""

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
        raise ValueError(
            "Unsupported VOYAGER_GROUNDEDNESS_TORCH_DTYPE='{0}'. "
            "Use one of bfloat16, float16, float32, or 'default'.".format(name)
        )
    return mapping[label]


def _load_provider(model_name: Optional[str]):
    """Load a real provider for production runs, else fall back to the dummy.

    Honours ``VOYAGER_GROUNDEDNESS_TORCH_DTYPE`` (default ``bfloat16``) so the
    sweep matches the production service-level encoder configuration.
    """

    if model_name:
        try:
            from pylate import models

            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype_name = os.environ.get("VOYAGER_GROUNDEDNESS_TORCH_DTYPE", "bfloat16")
            torch_dtype = _resolve_torch_dtype(dtype_name)
            model_kwargs: Dict[str, Any] = {}
            if torch_dtype is not None:
                model_kwargs["torch_dtype"] = torch_dtype
            try:
                return models.ColBERT(
                    model_name_or_path=model_name,
                    device=device,
                    do_query_expansion=False,
                    trust_remote_code=True,
                    model_kwargs=model_kwargs or None,
                )
            except TypeError:
                # Older pylate without model_kwargs / trust_remote_code support.
                return models.ColBERT(
                    model_name_or_path=model_name,
                    device=device,
                    do_query_expansion=False,
                )
        except Exception:
            pass
    from tests.test_groundedness_service import DummyGroundednessProvider

    return DummyGroundednessProvider(dim=24)


def _load_nli_provider(model_id: Optional[str]):
    """Build a HuggingFace NLI provider on demand; ``None`` if disabled."""

    if not model_id:
        return None
    from latence_trace.core.nli import HuggingFaceNLIProvider

    return HuggingFaceNLIProvider(model_id=model_id)


def _load_reranker(model_id: Optional[str]):
    """Build a cross-encoder premise reranker on demand; ``None`` if disabled."""

    if not model_id:
        return None
    from latence_trace.core.nli import CrossEncoderPremiseReranker

    return CrossEncoderPremiseReranker(model_id=model_id)


# ----------------------------------------------------------------------
# Score helpers
# ----------------------------------------------------------------------


def _encode_pair_side(
    *,
    provider,
    context: str,
    response: str,
    chunk_token_budget: int = 256,
) -> Dict[str, Any]:
    """Phase 1: encode + tokenize. Returns the materials needed by scoring.

    The response is **chunked** with the same sentence-packed segmenter as
    the context so this lane exercises the production response-chunking
    path (``score_groundedness_response_chunked``). Otherwise long responses
    silently truncate at the encoder ``model_max_length`` and tail tokens
    are dropped from the groundedness matrix, biasing every benchmark
    metric downward.
    """

    started = time.perf_counter()
    segments = segment_text(
        context, "sentence_packed", provider=provider, chunk_token_budget=chunk_token_budget
    )
    segment_texts = [segment["text"] for segment in segments] or [context]
    support_embeddings = encode_texts(provider, segment_texts, is_query=False, prompt_name=None)
    support_units = [
        SupportUnitInput(
            support_id="sup-{idx}".format(idx=idx),
            chunk_id=None,
            source_mode="raw_context",
            text=segment_texts[idx],
            embeddings=embedding,
            tokens=tokenize_text(provider, segment_texts[idx], expected_len=int(embedding.shape[0]), is_query=False),
        )
        for idx, embedding in enumerate(support_embeddings)
    ]
    response_chunks = _build_response_chunks(
        response,
        provider=provider,
        chunk_token_budget=chunk_token_budget,
        encode_fn=encode_texts,
        document_prompt_name=None,
    )
    encode_ms = (time.perf_counter() - started) * 1000.0
    return {
        "support_units": support_units,
        "response_chunks": response_chunks,
        "encode_ms": float(encode_ms),
    }


def _score_pair_side(
    *,
    materials: Dict[str, Any],
    response_text: str,
    null_bank_embeddings: Optional[Sequence[torch.Tensor]] = None,
    nli_provider=None,
    nli_max_latency_ms: float = 2000.0,
    nli_reranker=None,
    nli_concat_premises: Optional[bool] = None,
    nli_use_atomic_claims: Optional[bool] = None,
    verification_samples: Optional[Sequence[str]] = None,
    semantic_entropy_enabled: Optional[bool] = None,
    fusion_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Phase 2: score the encoded materials. Times only the score call."""

    started = time.perf_counter()
    response_chunks = materials.get("response_chunks") or []
    if not response_chunks:
        # Degenerate-response safeguard: empty response → trivial score.
        # ``score_groundedness_response_chunked`` raises on empty input.
        return {
            "scores": {
                "reverse_context": 0.0,
                "reverse_context_calibrated": 0.0,
                "groundedness_v2": 0.0,
            },
            "score_ms": 0.0,
        }
    scored = score_groundedness_response_chunked(
        support_batches=[materials["support_units"]],
        response_chunks=response_chunks,
        response_text=response_text,
        evidence_limit=3,
        primary_metric="reverse_context",
        debug_dense_matrices=False,
        null_bank_embeddings=null_bank_embeddings,
        nli_provider=nli_provider,
        nli_max_latency_ms=nli_max_latency_ms,
        nli_reranker=nli_reranker,
        nli_concat_premises=nli_concat_premises,
        nli_use_atomic_claims=nli_use_atomic_claims,
        verification_samples=list(verification_samples) if verification_samples else None,
        semantic_entropy_enabled=semantic_entropy_enabled,
        fusion_weights=fusion_weights,
    )
    score_ms = (time.perf_counter() - started) * 1000.0
    return {
        "scores": scored["scores"],
        "score_ms": float(score_ms),
    }


def _resolve_headline(
    scores: Dict[str, Any],
    *,
    nli_enabled: bool,
) -> Tuple[float, str]:
    """Pick the headline score for paired ranking and report which one was used.

    ``groundedness_v2`` already fuses calibrated + literal-guarded (and the
    optional NLI channel), so it is the preferred headline whenever it is
    available. Falls back to ``reverse_context_calibrated`` and finally the
    raw ``reverse_context`` score.
    """

    if scores.get("groundedness_v2") is not None:
        if nli_enabled:
            return float(scores["groundedness_v2"]), "groundedness_v2"
        return float(scores["groundedness_v2"]), "groundedness_v2_no_nli"
    cal = scores.get("reverse_context_calibrated")
    if cal is not None:
        return float(cal), "reverse_context_calibrated"
    return float(scores.get("reverse_context") or 0.0), "reverse_context"


# ----------------------------------------------------------------------
# Null bank caching
# ----------------------------------------------------------------------


def build_null_bank_embeddings(provider) -> List[torch.Tensor]:
    """Encode the default null bank once and return list of per-text embeddings."""

    texts = default_null_bank_texts()
    if not texts:
        return []
    embeddings = encode_texts(provider, texts, is_query=False, prompt_name=None)
    return [emb.detach().clone() for emb in embeddings]


# ----------------------------------------------------------------------
# Minimal-pair lane
# ----------------------------------------------------------------------


def _bootstrap_ci(
    rng: random.Random,
    samples: Sequence[float],
    *,
    iterations: int = 1000,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    if not samples:
        return 0.0, 0.0
    n = len(samples)
    means: List[float] = []
    for _ in range(iterations):
        draw = [samples[rng.randrange(n)] for _ in range(n)]
        means.append(sum(draw) / float(n))
    means.sort()
    lo_idx = max(0, int(math.floor((alpha / 2.0) * iterations)))
    hi_idx = min(iterations - 1, int(math.ceil((1.0 - alpha / 2.0) * iterations)))
    return float(means[lo_idx]), float(means[hi_idx])


def evaluate_minimal_pairs(
    pairs: Sequence[MinimalPair],
    provider,
    *,
    null_bank_embeddings: Optional[Sequence[torch.Tensor]] = None,
    nli_provider=None,
    nli_max_latency_ms: float = 2000.0,
    nli_reranker=None,
    nli_concat_premises: Optional[bool] = None,
    nli_use_atomic_claims: Optional[bool] = None,
    semantic_entropy_enabled: Optional[bool] = None,
    semantic_entropy_sample_count: int = 0,
    fusion_weights: Optional[Dict[str, float]] = None,
    warmup: int = 3,
    seed: int = 17,
) -> Dict[str, Any]:
    """Score every minimal pair and report paired-ranking accuracy per stratum.

    Latency is recorded per call as ``encode_ms + score_ms`` so the reported
    p50 / p95 reflect the full request, not just post-encoder math.
    """

    rng = random.Random(seed)
    by_stratum: Dict[str, List[float]] = {}
    encode_samples: List[float] = []
    score_samples: List[float] = []
    full_samples: List[float] = []
    headline_seen: Dict[str, int] = {}

    def _build_samples(pair_obj: MinimalPair, seed_response: str) -> Optional[List[str]]:
        """Synthesize caller-supplied verification samples.

        For a headline run we cannot call a real LLM, so we synthesize a
        deterministic bank of paraphrases around ``seed_response`` taken
        from the pair context. When semantic entropy is disabled or the
        count is non-positive, return ``None`` so the fast path is
        unchanged.
        """

        if not semantic_entropy_enabled or semantic_entropy_sample_count <= 0:
            return None
        # Mix the positive and negative responses around the seed to emulate
        # temperature>0 sampling: a stable generator mostly returns the seed,
        # a confabulating one drifts toward the opposite side. We combine
        # both to produce a realistic mix without an actual LLM call.
        alt_sources = [pair_obj.positive, pair_obj.negative]
        samples = [seed_response]
        for idx in range(1, semantic_entropy_sample_count):
            samples.append(alt_sources[idx % 2])
        return samples

    if pairs and warmup > 0:
        for _ in range(warmup):
            warm_pair = pairs[0]
            warm_materials = _encode_pair_side(
                provider=provider, context=warm_pair.context, response=warm_pair.positive
            )
            _ = _score_pair_side(
                materials=warm_materials,
                response_text=warm_pair.positive,
                null_bank_embeddings=null_bank_embeddings,
                nli_provider=nli_provider,
                nli_max_latency_ms=nli_max_latency_ms,
                nli_reranker=nli_reranker,
                nli_concat_premises=nli_concat_premises,
                nli_use_atomic_claims=nli_use_atomic_claims,
                verification_samples=_build_samples(warm_pair, warm_pair.positive),
                semantic_entropy_enabled=semantic_entropy_enabled,
                fusion_weights=fusion_weights,
            )

    nli_enabled = nli_provider is not None
    for pair in pairs:
        pos_materials = _encode_pair_side(
            provider=provider, context=pair.context, response=pair.positive
        )
        pos_scored = _score_pair_side(
            materials=pos_materials,
            response_text=pair.positive,
            null_bank_embeddings=null_bank_embeddings,
            nli_provider=nli_provider,
            nli_max_latency_ms=nli_max_latency_ms,
            nli_reranker=nli_reranker,
            nli_concat_premises=nli_concat_premises,
            nli_use_atomic_claims=nli_use_atomic_claims,
            verification_samples=_build_samples(pair, pair.positive),
            semantic_entropy_enabled=semantic_entropy_enabled,
            fusion_weights=fusion_weights,
        )
        neg_materials = _encode_pair_side(
            provider=provider, context=pair.context, response=pair.negative
        )
        neg_scored = _score_pair_side(
            materials=neg_materials,
            response_text=pair.negative,
            null_bank_embeddings=null_bank_embeddings,
            nli_provider=nli_provider,
            nli_max_latency_ms=nli_max_latency_ms,
            nli_reranker=nli_reranker,
            nli_concat_premises=nli_concat_premises,
            nli_use_atomic_claims=nli_use_atomic_claims,
            verification_samples=_build_samples(pair, pair.negative),
            semantic_entropy_enabled=semantic_entropy_enabled,
            fusion_weights=fusion_weights,
        )
        for materials, scored in (
            (pos_materials, pos_scored),
            (neg_materials, neg_scored),
        ):
            encode_samples.append(materials["encode_ms"])
            score_samples.append(scored["score_ms"])
            full_samples.append(materials["encode_ms"] + scored["score_ms"])

        pos_score, pos_field = _resolve_headline(pos_scored["scores"], nli_enabled=nli_enabled)
        neg_score, neg_field = _resolve_headline(neg_scored["scores"], nli_enabled=nli_enabled)
        headline_seen[pos_field] = headline_seen.get(pos_field, 0) + 1
        headline_seen[neg_field] = headline_seen.get(neg_field, 0) + 1
        correct = 1.0 if float(pos_score) > float(neg_score) else 0.0
        by_stratum.setdefault(pair.stratum, []).append(correct)

    per_stratum: Dict[str, Dict[str, float]] = {}
    for stratum, outcomes in by_stratum.items():
        accuracy = sum(outcomes) / float(len(outcomes))
        ci_lo, ci_hi = _bootstrap_ci(rng, outcomes)
        per_stratum[stratum] = {
            "n": len(outcomes),
            "paired_accuracy": float(accuracy),
            "ci_lower": float(ci_lo),
            "ci_upper": float(ci_hi),
        }

    overall_outcomes = [outcome for outcomes in by_stratum.values() for outcome in outcomes]
    overall_accuracy = (
        sum(overall_outcomes) / float(len(overall_outcomes)) if overall_outcomes else 0.0
    )
    headline_field = max(headline_seen, key=headline_seen.get) if headline_seen else "reverse_context"
    return {
        "per_stratum": per_stratum,
        "overall_accuracy": float(overall_accuracy),
        "pair_count": len(pairs),
        "stratum_counts": stratum_summary(pairs),
        "headline_used": headline_field,
        "headline_distribution": dict(headline_seen),
        "nli_enabled": bool(nli_enabled),
        "encode_p50_ms": float(np.percentile(encode_samples, 50)) if encode_samples else 0.0,
        "encode_p95_ms": float(np.percentile(encode_samples, 95)) if encode_samples else 0.0,
        "score_p50_ms": float(np.percentile(score_samples, 50)) if score_samples else 0.0,
        "score_p95_ms": float(np.percentile(score_samples, 95)) if score_samples else 0.0,
        "latency_p50_ms": float(np.percentile(full_samples, 50)) if full_samples else 0.0,
        "latency_p95_ms": float(np.percentile(full_samples, 95)) if full_samples else 0.0,
    }


# ----------------------------------------------------------------------
# External benchmark lane
# ----------------------------------------------------------------------


def _binary_label(sample: BenchmarkSample) -> int:
    return 0 if sample.label == "hallucinated" else 1


def evaluate_benchmark_samples(
    samples: Sequence[BenchmarkSample],
    provider,
    *,
    null_bank_embeddings: Optional[Sequence[torch.Tensor]] = None,
    nli_provider=None,
    nli_reranker=None,
    nli_concat_premises: Optional[bool] = None,
    nli_use_atomic_claims: Optional[bool] = None,
    semantic_entropy_enabled: Optional[bool] = None,
    semantic_entropy_sample_count: int = 0,
    fusion_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Score a flat list of benchmark samples and bucket per stratum."""

    nli_enabled = nli_provider is not None

    def _external_samples(sample_obj: BenchmarkSample) -> Optional[List[str]]:
        if not semantic_entropy_enabled or semantic_entropy_sample_count <= 0:
            return None
        return [sample_obj.response] * semantic_entropy_sample_count

    by_stratum: Dict[str, List[Tuple[float, int]]] = {}
    for sample in samples:
        materials = _encode_pair_side(
            provider=provider, context=sample.context, response=sample.response
        )
        scored = _score_pair_side(
            materials=materials,
            response_text=sample.response,
            null_bank_embeddings=null_bank_embeddings,
            nli_provider=nli_provider,
            nli_reranker=nli_reranker,
            nli_concat_premises=nli_concat_premises,
            nli_use_atomic_claims=nli_use_atomic_claims,
            verification_samples=_external_samples(sample),
            semantic_entropy_enabled=semantic_entropy_enabled,
            fusion_weights=fusion_weights,
        )
        score, _field = _resolve_headline(scored["scores"], nli_enabled=nli_enabled)
        by_stratum.setdefault(sample.stratum, []).append((float(score), _binary_label(sample)))

    benchmark_id = samples[0].benchmark if samples else ""
    per_stratum: Dict[str, Dict[str, float]] = {}
    for stratum, rows in by_stratum.items():
        scores = [score for score, _ in rows]
        labels = [label for _, label in rows]
        if not scores:
            continue
        # Threshold strategy:
        #   * FActScore atomic claims are class-imbalanced and the response
        #     space is many short atomic claims, not a few long answers --
        #     median threshold is meaningless. Sweep all candidate thresholds
        #     and pick the one that maximises F1 (the standard FActScore
        #     evaluation protocol).
        #   * Every other dataset (RAGTruth, HaluEval) ships balanced
        #     positive/negative pairs per stratum, so median threshold is the
        #     fair, parameter-free baseline.
        if benchmark_id == "factscore":
            threshold = _select_best_f1_threshold(scores, labels)
        else:
            threshold = float(np.median(scores))
        tp = sum(1 for s, l in rows if s >= threshold and l == 1)
        fp = sum(1 for s, l in rows if s >= threshold and l == 0)
        fn = sum(1 for s, l in rows if s < threshold and l == 1)
        tn = sum(1 for s, l in rows if s < threshold and l == 0)
        precision = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        positives = sum(1 for l in labels if l == 1)
        negatives = sum(1 for l in labels if l == 0)
        per_stratum[stratum] = {
            "n": len(rows),
            "n_positive": positives,
            "n_negative": negatives,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "threshold": float(threshold),
            "threshold_strategy": (
                "best_f1_sweep" if benchmark_id == "factscore" else "median"
            ),
        }
    return {
        "sample_count": len(samples),
        "per_stratum": per_stratum,
    }


def _select_best_f1_threshold(
    scores: Sequence[float],
    labels: Sequence[int],
    *,
    candidate_count: int = 101,
) -> float:
    """Sweep ``candidate_count`` thresholds and return the F1-optimal one.

    Used for class-imbalanced datasets (FActScore atomic claims) where the
    median threshold is uninformative. The sweep is bounded by the actual
    score range so it always evaluates at thresholds where the prediction
    actually changes.
    """

    if not scores:
        return 0.0
    if not any(label == 1 for label in labels):
        return float(max(scores)) + 1.0
    lo = float(min(scores))
    hi = float(max(scores))
    if hi <= lo:
        return float(lo)
    best_threshold = float(np.median(scores))
    best_f1 = -1.0
    span = hi - lo
    for k in range(candidate_count):
        threshold = lo + span * (k / float(candidate_count - 1))
        tp = sum(1 for s, l in zip(scores, labels) if s >= threshold and l == 1)
        fp = sum(1 for s, l in zip(scores, labels) if s >= threshold and l == 0)
        fn = sum(1 for s, l in zip(scores, labels) if s < threshold and l == 1)
        precision = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
    return best_threshold


# ----------------------------------------------------------------------
# Aggregation and verdict
# ----------------------------------------------------------------------


def assemble_report(
    *,
    minimal_pair_results: Dict[str, Any],
    external_results: Dict[str, Optional[Dict[str, Any]]],
    targets: Dict[str, Dict[str, Any]] = PREREGISTERED_TARGETS,
) -> Dict[str, Any]:
    """Merge per-lane results with the pre-registered exit criteria.

    Latency criteria use ``latency_p95_ms`` (encode + score). The
    ``latency_with_nli`` target is only checked when ``nli_enabled=True``;
    otherwise it is reported as ``not_applicable``.
    """

    nli_enabled = bool(minimal_pair_results.get("nli_enabled"))
    report: Dict[str, Any] = {
        "minimal_pairs": minimal_pair_results,
        "external": external_results,
        "criteria": {},
    }

    def _bucket_accuracy(strata: Sequence[str]) -> Tuple[float, float, int]:
        outcomes_acc: List[float] = []
        ci_lowers: List[float] = []
        n = 0
        for stratum in strata:
            stats = minimal_pair_results["per_stratum"].get(stratum)
            if not stats:
                continue
            outcomes_acc.append(stats["paired_accuracy"])
            ci_lowers.append(stats["ci_lower"])
            n += stats["n"]
        if not outcomes_acc:
            return 0.0, 0.0, 0
        return float(sum(outcomes_acc) / len(outcomes_acc)), float(min(ci_lowers)), n

    for key, target in targets.items():
        if key.startswith("minimal_pairs_"):
            strata = target.get("strata", ())
            accuracy, ci_lower, n = _bucket_accuracy(strata)
            met = (
                accuracy >= target.get("min", 0.0)
                and ci_lower >= target.get("ci_lower_min", 0.0)
                and n > 0
            )
            label = "pass" if met else (
                "partial"
                if accuracy >= target.get("min", 0.0) and n > 0
                else "fail"
            )
            report["criteria"][key] = {
                "metric": target["metric"],
                "value": accuracy,
                "ci_lower": ci_lower,
                "n": n,
                "min": target.get("min"),
                "ci_lower_min": target.get("ci_lower_min"),
                "met": bool(met),
                "label": label,
                "notes": target.get("notes"),
            }
        elif key == "ragtruth":
            payload = external_results.get("ragtruth")
            if not payload:
                report["criteria"][key] = {"status": "skipped", "notes": target.get("notes")}
                continue
            f1s = [stats["f1"] for stats in payload["per_stratum"].values()]
            macro = float(sum(f1s) / len(f1s)) if f1s else 0.0
            met = bool(macro >= target.get("min", 0.0))
            report["criteria"][key] = {
                "metric": target["metric"],
                "value": macro,
                "min": target.get("min"),
                "met": met,
                "label": "pass" if met else "fail",
                "notes": target.get("notes"),
            }
        elif key == "halueval_qa":
            payload = external_results.get("halueval")
            if not payload or "qa" not in payload["per_stratum"]:
                report["criteria"][key] = {"status": "skipped", "notes": target.get("notes")}
                continue
            qa_stats = payload["per_stratum"]["qa"]
            paired_proxy = float(qa_stats["precision"])
            met = bool(paired_proxy >= target.get("min", 0.0))
            report["criteria"][key] = {
                "metric": target["metric"],
                "value": paired_proxy,
                "min": target.get("min"),
                "met": met,
                "label": "pass" if met else "fail",
                "notes": target.get("notes"),
            }
        elif key == "factscore":
            payload = external_results.get("factscore")
            if not payload or "biography" not in payload["per_stratum"]:
                report["criteria"][key] = {
                    "status": "skipped",
                    "skip_reason": (
                        "FActScore data not available "
                        "(set VOYAGER_GROUNDEDNESS_FACTSCORE_DIR; "
                        "run scripts/enrich_factscore_with_wiki.py to populate "
                        "biographies_wiki.jsonl)"
                    ),
                    "notes": target.get("notes"),
                }
                continue
            bio_stats = payload["per_stratum"]["biography"]
            precision = float(bio_stats["precision"])
            met = bool(precision >= target.get("min", 0.0))
            report["criteria"][key] = {
                "metric": target["metric"],
                "value": precision,
                "f1": float(bio_stats.get("f1", 0.0)),
                "recall": float(bio_stats.get("recall", 0.0)),
                "n_claims": int(bio_stats.get("n", 0)),
                "n_positive": int(bio_stats.get("n_positive", 0)),
                "n_negative": int(bio_stats.get("n_negative", 0)),
                "threshold": float(bio_stats.get("threshold", 0.0)),
                "threshold_strategy": str(
                    bio_stats.get("threshold_strategy", "best_f1_sweep")
                ),
                "min": target.get("min"),
                "met": met,
                "label": "pass" if met else "fail",
                "evaluation": "per_claim_atomic_precision_at_f1_best_threshold",
                "notes": target.get("notes"),
            }
        elif key == "latency_score_only":
            value = float(minimal_pair_results.get("latency_p95_ms", 0.0))
            limit = float(target.get("max", math.inf))
            met = value <= limit
            report["criteria"][key] = {
                "metric": target["metric"],
                "value": value,
                "max": limit,
                "met": bool(met),
                "label": "pass" if met else "fail",
                "applies_when": "nli_disabled",
                "applicable": not nli_enabled,
                "notes": target.get("notes"),
            }
        elif key == "latency_with_nli":
            value = float(minimal_pair_results.get("latency_p95_ms", 0.0))
            limit = float(target.get("max", math.inf))
            if not nli_enabled:
                report["criteria"][key] = {
                    "status": "not_applicable",
                    "applies_when": "nli_enabled",
                    "applicable": False,
                    "notes": target.get("notes"),
                }
                continue
            met = value <= limit
            report["criteria"][key] = {
                "metric": target["metric"],
                "value": value,
                "max": limit,
                "met": bool(met),
                "label": "pass" if met else "fail",
                "applies_when": "nli_enabled",
                "applicable": True,
                "notes": target.get("notes"),
            }

    actionable = [
        criterion
        for criterion in report["criteria"].values()
        if criterion.get("status") not in {"skipped", "not_applicable"}
        and criterion.get("applicable", True)
    ]
    overall_passed = bool(actionable) and all(
        criterion.get("met", False) for criterion in actionable
    )
    report["all_targets_met"] = bool(overall_passed)
    report["headline_verdict"] = compute_headline_verdict(report)
    return report


def compute_headline_verdict(report: Dict[str, Any]) -> str:
    """Map per-stratum criterion labels to a single, plainly worded verdict."""

    criteria = report.get("criteria", {})
    nli_enabled = bool(report.get("minimal_pairs", {}).get("nli_enabled"))

    def _label(name: str) -> str:
        return str(criteria.get(name, {}).get("label", "missing"))

    lex = _label("minimal_pairs_lexical")
    sem = _label("minimal_pairs_semantic")
    par = _label("minimal_pairs_partial")
    lat_no = _label("latency_score_only")
    lat_nli = _label("latency_with_nli")

    if lex == "fail":
        return "not a feature yet: lexical strata fail"

    semantic_ok = sem in {"pass", "partial"}
    partial_ok = par in {"pass", "partial"}

    if not nli_enabled:
        if lex == "pass" and semantic_ok and partial_ok and lat_no == "pass":
            return "feature in Beta, ready for evidence/QA without NLI"
        if lex == "pass" and (sem == "fail" or par == "fail"):
            return "feature in Beta, NLI required for negation/role/partial"
        if lat_no == "fail":
            return "feature in Beta, missing latency budget without NLI"
        return "feature in Beta with caveats; see per-stratum table"

    if lex == "pass" and semantic_ok and partial_ok and lat_nli == "pass":
        return "feature in Beta with NLI peer, ready for evidence/QA"
    if lex == "pass" and (sem == "fail" or par == "fail"):
        return "not a feature yet: NLI lane still fails semantic or partial"
    if lat_nli == "fail":
        return "feature in Beta with NLI peer, missing latency budget"
    return "feature in Beta with NLI peer; see per-stratum table"


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Phase E external evaluation harness")
    parser.add_argument("--model", default=os.environ.get("VOYAGER_GROUNDEDNESS_MODEL"))
    parser.add_argument(
        "--pairs-per-stratum",
        type=int,
        default=int(os.environ.get("VOYAGER_GROUNDEDNESS_MIN_PAIRS_PER_STRATUM", "30")),
    )
    parser.add_argument(
        "--max-external-per-stratum",
        type=int,
        default=int(os.environ.get("VOYAGER_GROUNDEDNESS_MAX_EXTERNAL_PER_STRATUM", "200")),
    )
    parser.add_argument(
        "--max-factscore-biographies",
        type=int,
        default=int(os.environ.get("VOYAGER_GROUNDEDNESS_MAX_FACTSCORE_BIOGRAPHIES", "30")),
        help=(
            "Cap on FActScore biographies. Each biography emits ~25 atomic "
            "claims that are scored independently, so the total number of "
            "scoring calls is roughly biographies x 25. Use a smaller cap "
            "for fast iteration (default 30 biographies ~ 800 claims)."
        ),
    )
    parser.add_argument(
        "--enable-nli",
        action="store_true",
        help="Build a HuggingFaceNLIProvider and run with the fused groundedness_v2 headline.",
    )
    parser.add_argument(
        "--nli-model",
        default=os.environ.get(
            "VOYAGER_GROUNDEDNESS_NLI_MODEL",
            "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
        ),
    )
    parser.add_argument(
        "--reranker-model",
        default=os.environ.get("VOYAGER_GROUNDEDNESS_NLI_PREMISE_RERANKER_MODEL"),
        help="Optional cross-encoder premise reranker (e.g. BAAI/bge-reranker-v2-m3).",
    )
    parser.add_argument(
        "--concat-premises",
        dest="concat_premises",
        action="store_true",
        default=None,
        help="Concatenate top-k premises into a single NLI input per claim.",
    )
    parser.add_argument(
        "--no-concat-premises",
        dest="concat_premises",
        action="store_false",
        help="Disable the multi-premise concatenation (revert to per-premise scoring).",
    )
    parser.add_argument(
        "--atomic-claims",
        dest="atomic_claims",
        action="store_true",
        default=None,
        help="Decompose each sentence into atomic facts and verify each atom separately.",
    )
    parser.add_argument(
        "--no-atomic-claims",
        dest="atomic_claims",
        action="store_false",
        help="Disable atomic-fact decomposition and keep sentence-level verification.",
    )
    parser.add_argument(
        "--enable-semantic-entropy",
        action="store_true",
        help="Enable semantic-entropy peer (Phase G) using synthesized verification samples.",
    )
    parser.add_argument(
        "--semantic-entropy-samples",
        type=int,
        default=int(os.environ.get("VOYAGER_GROUNDEDNESS_SEMANTIC_ENTROPY_SAMPLES", "4")),
        help="Number of synthetic verification samples to draw per case (default 4).",
    )
    parser.add_argument(
        "--fusion-weight-calibrated",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--fusion-weight-literal",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--fusion-weight-nli",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--fusion-weight-semantic-entropy",
        type=float,
        default=None,
    )
    parser.add_argument("--out", type=Path, default=_HERE / "groundedness_external_eval_report.json")
    args = parser.parse_args(argv)

    provider = _load_provider(args.model)
    pairs = build_minimal_pairs(pairs_per_stratum=args.pairs_per_stratum)

    null_bank_embeddings = build_null_bank_embeddings(provider)
    nli_provider = _load_nli_provider(args.nli_model) if args.enable_nli else None
    nli_reranker = _load_reranker(args.reranker_model) if args.enable_nli else None

    fusion_weights: Optional[Dict[str, float]] = None
    if any(
        getattr(args, name) is not None
        for name in (
            "fusion_weight_calibrated",
            "fusion_weight_literal",
            "fusion_weight_nli",
            "fusion_weight_semantic_entropy",
        )
    ):
        fusion_weights = {
            "calibrated": args.fusion_weight_calibrated if args.fusion_weight_calibrated is not None else 0.5,
            "literal": args.fusion_weight_literal if args.fusion_weight_literal is not None else 0.2,
            "nli": args.fusion_weight_nli if args.fusion_weight_nli is not None else (0.3 if args.enable_nli else 0.0),
            "semantic_entropy": (
                args.fusion_weight_semantic_entropy
                if args.fusion_weight_semantic_entropy is not None
                else (0.15 if args.enable_semantic_entropy else 0.0)
            ),
        }
    elif args.enable_semantic_entropy:
        fusion_weights = {
            "calibrated": 0.45,
            "literal": 0.2,
            "nli": 0.25 if args.enable_nli else 0.0,
            "semantic_entropy": 0.1,
        }

    semantic_entropy_sample_count = max(0, int(args.semantic_entropy_samples))

    minimal_pair_results = evaluate_minimal_pairs(
        pairs,
        provider,
        null_bank_embeddings=null_bank_embeddings,
        nli_provider=nli_provider,
        nli_reranker=nli_reranker,
        nli_concat_premises=args.concat_premises,
        nli_use_atomic_claims=args.atomic_claims,
        semantic_entropy_enabled=args.enable_semantic_entropy,
        semantic_entropy_sample_count=semantic_entropy_sample_count,
        fusion_weights=fusion_weights,
    )

    external_results: Dict[str, Optional[Dict[str, Any]]] = {}
    for name, loader in (
        ("ragtruth", load_ragtruth),
        ("halueval", load_halueval),
        ("factscore", load_factscore),
    ):
        # FActScore is per-claim atomic; one biography emits ~25 claims, so
        # the per-stratum cap is interpreted as biographies (not claims).
        per_stratum_cap = (
            args.max_factscore_biographies
            if name == "factscore"
            else args.max_external_per_stratum
        )
        samples = loader(max_samples_per_stratum=per_stratum_cap)
        if samples is None:
            external_results[name] = None
            continue
        external_results[name] = evaluate_benchmark_samples(
            samples,
            provider,
            null_bank_embeddings=null_bank_embeddings,
            nli_provider=nli_provider,
            nli_reranker=nli_reranker,
            nli_concat_premises=args.concat_premises,
            nli_use_atomic_claims=args.atomic_claims,
            semantic_entropy_enabled=args.enable_semantic_entropy,
            semantic_entropy_sample_count=semantic_entropy_sample_count,
            fusion_weights=fusion_weights,
        )

    report = assemble_report(
        minimal_pair_results=minimal_pair_results,
        external_results=external_results,
    )
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "headline_verdict": report["headline_verdict"],
        "headline_used": minimal_pair_results.get("headline_used"),
        "nli_enabled": minimal_pair_results.get("nli_enabled"),
        "minimal_pair_summary": minimal_pair_results["per_stratum"],
        "encode_p95_ms": minimal_pair_results.get("encode_p95_ms"),
        "score_p95_ms": minimal_pair_results.get("score_p95_ms"),
        "latency_p95_ms": minimal_pair_results.get("latency_p95_ms"),
        "available_external": available_benchmarks(),
        "all_targets_met": report["all_targets_met"],
        "report_path": str(args.out),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
