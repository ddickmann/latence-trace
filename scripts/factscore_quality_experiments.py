#!/usr/bin/env python3
"""Standalone FActScore / HaluEval quality experiments.

This script is intentionally research-only: it does not change production
defaults. It probes three hypotheses:

1. Query-aware scoring for FActScore via the existing triangular query/context
   path.
2. Paragraph-level Wikipedia selection before scoring atomic claims.
3. English NLI + small fusion-weight sweeps, reusing component scores instead
   of re-running the model for every weight vector.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from latence_trace.core.groundedness import (  # noqa: E402
    SupportUnitInput,
    _build_response_chunks,
    encode_texts,
    score_groundedness_response_chunked,
    segment_text,
    tokenize_text,
)
from latence_trace.core.nli import (  # noqa: E402
    CrossEncoderPremiseReranker,
    HuggingFaceNLIProvider,
    fuse_groundedness_v2,
)
from research.triangular_maxsim.groundedness_external_benchmarks import (  # noqa: E402
    BenchmarkSample,
    load_factscore,
    load_halueval,
)
from research.triangular_maxsim.groundedness_external_eval import (  # noqa: E402
    _load_provider,
    build_null_bank_embeddings,
)


_WORD_RE = re.compile(r"[A-Za-z0-9]+")


@dataclass(frozen=True)
class Variant:
    name: str
    paragraph_top_k: Optional[int] = None
    query_metric: bool = False


FUSION_GRID: Dict[str, Dict[str, float]] = {
    "default_0.5_0.2_0.3": {"calibrated": 0.5, "literal": 0.2, "nli": 0.3},
    "nli_heavy_0.35_0.15_0.50": {"calibrated": 0.35, "literal": 0.15, "nli": 0.50},
    "nli_light_0.65_0.20_0.15": {"calibrated": 0.65, "literal": 0.20, "nli": 0.15},
    "literal_heavy_0.45_0.35_0.20": {"calibrated": 0.45, "literal": 0.35, "nli": 0.20},
    "calibrated_only": {"calibrated": 1.0, "literal": 0.0, "nli": 0.0},
}


def _tokens(text: str) -> set[str]:
    return {m.group(0).lower() for m in _WORD_RE.finditer(text)}


def _paragraphs(context: str) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n+", context) if p.strip()]
    if len(paras) >= 2:
        return paras
    # Fallback for contexts without blank-line structure.
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", context) if p.strip()]
    return parts or [context]


def _select_paragraphs(context: str, query: str, *, top_k: int) -> tuple[str, list[int]]:
    query_tokens = _tokens(query)
    paragraphs = _paragraphs(context)
    scored: list[tuple[float, int, str]] = []
    for idx, paragraph in enumerate(paragraphs):
        p_tokens = _tokens(paragraph)
        if not p_tokens:
            score = 0.0
        else:
            overlap = len(query_tokens & p_tokens)
            # Length-normalized lexical prior: enough to pull the right section
            # up without always preferring the longest paragraph.
            score = overlap / math.sqrt(len(p_tokens))
        scored.append((float(score), idx, paragraph))
    chosen = sorted(scored, key=lambda row: (-row[0], row[1]))[:top_k]
    chosen_sorted = sorted(chosen, key=lambda row: row[1])
    return "\n\n".join(row[2] for row in chosen_sorted), [row[1] for row in chosen_sorted]


def _select_factscore_samples(samples: Sequence[BenchmarkSample], *, per_label: int, seed: int) -> list[BenchmarkSample]:
    rng = random.Random(seed)
    by_label: dict[str, list[BenchmarkSample]] = defaultdict(list)
    for sample in samples:
        by_label[sample.label].append(sample)
    selected: list[BenchmarkSample] = []
    for label in ("faithful", "hallucinated"):
        rows = list(by_label[label])
        rows.sort(key=lambda s: s.sample_id)
        rng.shuffle(rows)
        selected.extend(rows[:per_label])
    selected.sort(key=lambda s: s.sample_id)
    return selected


def _select_halueval_samples(samples: Sequence[BenchmarkSample], *, pairs_per_stratum: int) -> list[BenchmarkSample]:
    # The loader emits paired `*-pos` and `*-neg` sample ids in file order.
    selected: list[BenchmarkSample] = []
    by_stratum: dict[str, dict[str, BenchmarkSample]] = defaultdict(dict)
    for sample in samples:
        if sample.stratum not in {"qa", "summarization"}:
            continue
        pair_id = sample.sample_id.rsplit("-", 1)[0]
        by_stratum[sample.stratum][f"{pair_id}:{sample.label}"] = sample
    for stratum in ("qa", "summarization"):
        emitted = 0
        pair_ids = sorted({key.split(":", 1)[0] for key in by_stratum[stratum]})
        for pair_id in pair_ids:
            pos = by_stratum[stratum].get(f"{pair_id}:faithful")
            neg = by_stratum[stratum].get(f"{pair_id}:hallucinated")
            if pos is None or neg is None:
                continue
            selected.extend([pos, neg])
            emitted += 1
            if emitted >= pairs_per_stratum:
                break
    return selected


def _build_materials(
    provider: Any,
    *,
    context: str,
    response: str,
    query: Optional[str],
    chunk_token_budget: int,
) -> dict[str, Any]:
    segments = segment_text(
        context,
        "sentence_packed",
        provider=provider,
        chunk_token_budget=chunk_token_budget,
    )
    segment_texts = [str(segment["text"]) for segment in segments] or [context]
    support_embeddings = encode_texts(provider, segment_texts, is_query=False, prompt_name=None)
    support_units = [
        SupportUnitInput(
            support_id=f"sup-{idx}",
            chunk_id=None,
            source_mode="raw_context",
            text=segment_texts[idx],
            embeddings=embedding,
            tokens=tokenize_text(
                provider,
                segment_texts[idx],
                expected_len=int(embedding.shape[0]),
                is_query=False,
            ),
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

    query_embeddings = None
    query_tokens = None
    if query and query.strip():
        query_embedding = encode_texts(provider, [query], is_query=True, prompt_name=None)[0]
        query_embeddings = query_embedding
        query_tokens = tokenize_text(
            provider,
            query,
            expected_len=int(query_embedding.shape[0]),
            is_query=True,
        )

    return {
        "support_units": support_units,
        "response_chunks": response_chunks,
        "query_embeddings": query_embeddings,
        "query_tokens": query_tokens,
    }


def _score_sample(
    provider: Any,
    sample: BenchmarkSample,
    *,
    variant: Variant,
    null_bank_embeddings: Optional[Sequence[torch.Tensor]],
    nli_provider: Optional[HuggingFaceNLIProvider],
    nli_reranker: Optional[CrossEncoderPremiseReranker],
    chunk_token_budget: int,
) -> dict[str, Any]:
    query_text = " ".join(part for part in (sample.query, sample.response) if part)
    context = sample.context
    selected_paragraphs: list[int] = []
    if variant.paragraph_top_k is not None:
        context, selected_paragraphs = _select_paragraphs(
            sample.context,
            query_text or sample.response,
            top_k=variant.paragraph_top_k,
        )

    started = time.perf_counter()
    materials = _build_materials(
        provider,
        context=context,
        response=sample.response,
        query=query_text if variant.query_metric else None,
        chunk_token_budget=chunk_token_budget,
    )
    if not materials["response_chunks"]:
        scores: dict[str, Any] = {
            "reverse_context": 0.0,
            "reverse_context_calibrated": 0.0,
            "literal_guarded": 0.0,
            "nli_aggregate": None,
            "groundedness_v2": 0.0,
            "triangular": None,
        }
    else:
        scored = score_groundedness_response_chunked(
            support_batches=[materials["support_units"]],
            response_chunks=materials["response_chunks"],
            response_text=sample.response,
            query_embeddings=materials["query_embeddings"],
            query_tokens=materials["query_tokens"],
            evidence_limit=3,
            primary_metric="triangular" if variant.query_metric else "reverse_context",
            null_bank_embeddings=null_bank_embeddings,
            nli_provider=nli_provider,
            nli_reranker=nli_reranker,
            nli_concat_premises=True if nli_provider is not None else None,
            nli_use_atomic_claims=True if nli_provider is not None else None,
        )
        scores = scored["scores"]

    latency_ms = (time.perf_counter() - started) * 1000.0
    query_triangular = scores.get("triangular")
    default_v2 = scores.get("groundedness_v2")
    triangular_blend = (
        0.7 * float(default_v2) + 0.3 * float(query_triangular)
        if default_v2 is not None and query_triangular is not None
        else None
    )

    return {
        "benchmark": sample.benchmark,
        "sample_id": sample.sample_id,
        "stratum": sample.stratum,
        "label": sample.label,
        "is_supported": sample.label == "faithful",
        "topic": sample.raw.get("topic") or sample.query,
        "claim": sample.response,
        "variant": variant.name,
        "paragraph_top_k": variant.paragraph_top_k,
        "selected_paragraphs": selected_paragraphs,
        "context_chars": len(context),
        "latency_ms": float(latency_ms),
        "scores": {
            "groundedness_v2": _float_or_none(default_v2),
            "reverse_context_calibrated": _float_or_none(scores.get("reverse_context_calibrated")),
            "literal_guarded": _float_or_none(scores.get("literal_guarded")),
            "nli_aggregate": _float_or_none(scores.get("nli_aggregate")),
            "triangular": _float_or_none(query_triangular),
            "triangular_blend_0.7v2_0.3tri": _float_or_none(triangular_blend),
        },
    }


def _float_or_none(value: Any) -> Optional[float]:
    return None if value is None else float(value)


def _labels_and_scores(rows: Sequence[dict[str, Any]], metric: str) -> tuple[list[int], list[float]]:
    labels: list[int] = []
    scores: list[float] = []
    for row in rows:
        value = row["scores"].get(metric)
        if value is None:
            continue
        labels.append(1 if row["is_supported"] else 0)
        scores.append(float(value))
    return labels, scores


def _best_threshold(scores: Sequence[float], labels: Sequence[int], *, candidate_count: int = 101) -> float:
    if not scores:
        return 0.0
    lo = min(scores)
    hi = max(scores)
    if hi <= lo:
        return float(lo)
    best_threshold = float(np.median(scores))
    best_f1 = -1.0
    for idx in range(candidate_count):
        threshold = lo + (hi - lo) * (idx / float(candidate_count - 1))
        stats = _classification_at_threshold(scores, labels, threshold)
        if stats["f1"] > best_f1:
            best_f1 = stats["f1"]
            best_threshold = float(threshold)
    return best_threshold


def _classification_at_threshold(
    scores: Sequence[float],
    labels: Sequence[int],
    threshold: float,
) -> dict[str, Any]:
    tp = sum(1 for score, label in zip(scores, labels) if score >= threshold and label == 1)
    fp = sum(1 for score, label in zip(scores, labels) if score >= threshold and label == 0)
    fn = sum(1 for score, label in zip(scores, labels) if score < threshold and label == 1)
    tn = sum(1 for score, label in zip(scores, labels) if score < threshold and label == 0)
    precision = tp / float(tp + fp) if tp + fp else 0.0
    recall = tp / float(tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "threshold": float(threshold),
    }


def _factscore_metrics(rows: Sequence[dict[str, Any]], metric: str) -> dict[str, Any]:
    labels, scores = _labels_and_scores(rows, metric)
    threshold = _best_threshold(scores, labels)
    stats = _classification_at_threshold(scores, labels, threshold)
    pos_scores = [s for s, label in zip(scores, labels) if label == 1]
    neg_scores = [s for s, label in zip(scores, labels) if label == 0]
    stats.update(
        {
            "n": len(scores),
            "n_positive": sum(labels),
            "n_negative": len(labels) - sum(labels),
            "metric": metric,
            "mean_supported": statistics.fmean(pos_scores) if pos_scores else None,
            "mean_unsupported": statistics.fmean(neg_scores) if neg_scores else None,
            "mean_margin": (
                statistics.fmean(pos_scores) - statistics.fmean(neg_scores)
                if pos_scores and neg_scores
                else None
            ),
        }
    )
    return stats


def _fused_metric_rows(rows: Sequence[dict[str, Any]], weights: dict[str, float], metric_name: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        scores = row["scores"]
        fused = fuse_groundedness_v2(
            reverse_context_calibrated=scores.get("reverse_context_calibrated"),
            literal_guarded=scores.get("literal_guarded"),
            nli_aggregate=scores.get("nli_aggregate"),
            weights=weights,
        )
        clone = dict(row)
        clone["scores"] = dict(scores)
        clone["scores"][metric_name] = fused
        out.append(clone)
    return out


def _summarize_factscore(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_variant[row["variant"]].append(row)

    summary: dict[str, Any] = {}
    for variant, variant_rows in by_variant.items():
        metrics = {
            "groundedness_v2": _factscore_metrics(variant_rows, "groundedness_v2"),
            "reverse_context_calibrated": _factscore_metrics(variant_rows, "reverse_context_calibrated"),
        }
        if any(row["scores"].get("triangular") is not None for row in variant_rows):
            metrics["triangular"] = _factscore_metrics(variant_rows, "triangular")
            metrics["triangular_blend_0.7v2_0.3tri"] = _factscore_metrics(
                variant_rows,
                "triangular_blend_0.7v2_0.3tri",
            )
        for name, weights in FUSION_GRID.items():
            metric_name = f"fusion::{name}"
            fused_rows = _fused_metric_rows(variant_rows, weights, metric_name)
            metrics[metric_name] = _factscore_metrics(fused_rows, metric_name)
        summary[variant] = {
            "n": len(variant_rows),
            "avg_latency_ms": statistics.fmean(row["latency_ms"] for row in variant_rows),
            "p95_latency_ms": float(np.percentile([row["latency_ms"] for row in variant_rows], 95)),
            "avg_context_chars": statistics.fmean(row["context_chars"] for row in variant_rows),
            "metrics": metrics,
        }
    return summary


def _summarize_halueval(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_stratum[row["stratum"]].append(row)

    for stratum, stratum_rows in by_stratum.items():
        by_pair: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for row in stratum_rows:
            pair_id = row["sample_id"].rsplit("-", 1)[0]
            by_pair[pair_id][row["label"]] = row
        metric_summaries: dict[str, Any] = {}
        metric_names = ["groundedness_v2", *[f"fusion::{name}" for name in FUSION_GRID]]
        for metric_name in metric_names:
            wins = 0
            ties = 0
            count = 0
            scored_rows = stratum_rows
            if metric_name.startswith("fusion::"):
                weights = FUSION_GRID[metric_name.split("::", 1)[1]]
                scored_rows = _fused_metric_rows(stratum_rows, weights, metric_name)
                scored_by_id = {row["sample_id"]: row for row in scored_rows}
                pair_map: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
                for row in scored_by_id.values():
                    pair_map[row["sample_id"].rsplit("-", 1)[0]][row["label"]] = row
            else:
                pair_map = by_pair
            for pair in pair_map.values():
                pos = pair.get("faithful")
                neg = pair.get("hallucinated")
                if pos is None or neg is None:
                    continue
                pos_score = pos["scores"].get(metric_name)
                neg_score = neg["scores"].get(metric_name)
                if pos_score is None or neg_score is None:
                    continue
                count += 1
                if pos_score > neg_score:
                    wins += 1
                elif pos_score == neg_score:
                    ties += 1
            metric_summaries[metric_name] = {
                "n_pairs": count,
                "wins_right_higher": wins,
                "ties": ties,
                "paired_accuracy": (wins + ties / 2.0) / float(count) if count else 0.0,
            }
        out[stratum] = metric_summaries
    return out


def _worst_errors(rows: Sequence[dict[str, Any]], metric: str, *, limit: int = 8) -> list[dict[str, Any]]:
    labels, scores = _labels_and_scores(rows, metric)
    threshold = _best_threshold(scores, labels)
    errors: list[tuple[float, dict[str, Any]]] = []
    for row in rows:
        score = row["scores"].get(metric)
        if score is None:
            continue
        predicted = score >= threshold
        actual = bool(row["is_supported"])
        if predicted != actual:
            confidence = abs(float(score) - threshold)
            errors.append((confidence, row))
    errors.sort(key=lambda item: item[0], reverse=True)
    return [
        {
            "sample_id": row["sample_id"],
            "label": row["label"],
            "score": row["scores"].get(metric),
            "threshold": threshold,
            "topic": row.get("topic"),
            "claim": row.get("claim"),
            "variant": row.get("variant"),
        }
        for _confidence, row in errors[:limit]
    ]


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    fact = payload["factscore_summary"]
    lines = [
        "# FActScore Quality Experiments",
        "",
        "## Setup",
        "",
        f"- Timestamp: `{payload['timestamp']}`",
        f"- FactScore claims: `{payload['inputs']['factscore_claims']}`",
        f"- HaluEval pairs per stratum: `{payload['inputs']['halueval_pairs_per_stratum']}`",
        f"- Encoder model: `{payload['inputs']['encoder_model']}`",
        f"- NLI model: `{payload['inputs']['nli_model']}`",
        f"- Reranker enabled: `{payload['inputs']['reranker_enabled']}`",
        "",
        "## FactScore Results",
        "",
        "| Variant | Metric | Precision | Recall | F1 | Margin | Avg Context Chars | p95 Latency ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant, summary in fact.items():
        # Show the most important metrics to keep the report readable.
        for metric_name in (
            "groundedness_v2",
            "triangular",
            "triangular_blend_0.7v2_0.3tri",
            "fusion::nli_heavy_0.35_0.15_0.50",
            "fusion::nli_light_0.65_0.20_0.15",
        ):
            metric = summary["metrics"].get(metric_name)
            if not metric:
                continue
            margin = metric.get("mean_margin")
            lines.append(
                "| {variant} | `{metric_name}` | {precision:.3f} | {recall:.3f} | {f1:.3f} | {margin} | {ctx:.0f} | {lat:.1f} |".format(
                    variant=variant,
                    metric_name=metric_name,
                    precision=metric["precision"],
                    recall=metric["recall"],
                    f1=metric["f1"],
                    margin="-" if margin is None else f"{margin:.3f}",
                    ctx=summary["avg_context_chars"],
                    lat=summary["p95_latency_ms"],
                )
            )
    lines.extend(
        [
            "",
            "## HaluEval QA/Summarization English NLI Sweep",
            "",
            "| Stratum | Metric | Paired Accuracy | Pairs |",
            "|---|---:|---:|---:|",
        ]
    )
    for stratum, metrics in payload["halueval_summary"].items():
        for metric_name, metric in metrics.items():
            lines.append(
                f"| {stratum} | `{metric_name}` | {metric['paired_accuracy']:.3f} | {metric['n_pairs']} |"
            )
    lines.extend(
        [
            "",
            "## Current Recommendation",
            "",
            payload["recommendation"],
            "",
            "## Report Artefacts",
            "",
            f"- JSON: `{payload['json_path']}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factscore-per-label", type=int, default=20)
    parser.add_argument("--halueval-pairs-per-stratum", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260427)
    parser.add_argument("--chunk-token-budget", type=int, default=256)
    parser.add_argument("--encoder-model", default="lightonai/GTE-ModernColBERT-v1")
    parser.add_argument("--nli-model", default="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
    parser.add_argument("--no-nli", action="store_true")
    parser.add_argument("--enable-reranker", action="store_true")
    parser.add_argument("--out-dir", default="reports")
    args = parser.parse_args()

    os.environ.setdefault("VOYAGER_GROUNDEDNESS_FACTSCORE_DIR", "/workspace/datasets/factscore")
    os.environ.setdefault("VOYAGER_GROUNDEDNESS_HALUEVAL_DIR", "/workspace/datasets/halueval")
    os.environ.setdefault("VOYAGER_GROUNDEDNESS_TORCH_DTYPE", "bfloat16")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"factscore_quality_experiments_{timestamp}.json"
    md_path = out_dir / f"factscore_quality_experiments_{timestamp}.md"

    print(f"=> loading encoder {args.encoder_model}", flush=True)
    provider = _load_provider(args.encoder_model)
    print(f"=> provider={type(provider).__name__}", flush=True)
    null_bank_embeddings = build_null_bank_embeddings(provider)
    print(f"=> null_bank={len(null_bank_embeddings)}", flush=True)

    nli_provider = None
    nli_reranker = None
    if not args.no_nli:
        print(f"=> loading NLI {args.nli_model}", flush=True)
        nli_provider = HuggingFaceNLIProvider(args.nli_model)
        if args.enable_reranker:
            print("=> enabling cross-encoder premise reranker", flush=True)
            nli_reranker = CrossEncoderPremiseReranker()

    factscore_all = load_factscore() or []
    factscore_samples = _select_factscore_samples(
        factscore_all,
        per_label=args.factscore_per_label,
        seed=args.seed,
    )
    halueval_samples = _select_halueval_samples(
        load_halueval(max_samples_per_stratum=args.halueval_pairs_per_stratum) or [],
        pairs_per_stratum=args.halueval_pairs_per_stratum,
    )

    variants = [
        Variant("baseline_full_context"),
        Variant("query_triangular_full_context", query_metric=True),
        Variant("paragraph_lex_top1", paragraph_top_k=1),
        Variant("paragraph_lex_top3", paragraph_top_k=3),
        Variant("paragraph_lex_top5", paragraph_top_k=5),
    ]

    factscore_rows: list[dict[str, Any]] = []
    for variant in variants:
        print(f"=> FactScore variant {variant.name} ({len(factscore_samples)} claims)", flush=True)
        for idx, sample in enumerate(factscore_samples, start=1):
            if idx % 10 == 0:
                print(f"   scored {idx}/{len(factscore_samples)}", flush=True)
            factscore_rows.append(
                _score_sample(
                    provider,
                    sample,
                    variant=variant,
                    null_bank_embeddings=null_bank_embeddings,
                    nli_provider=nli_provider,
                    nli_reranker=nli_reranker,
                    chunk_token_budget=args.chunk_token_budget,
                )
            )

    print(f"=> HaluEval QA/Summ ({len(halueval_samples)} samples)", flush=True)
    halueval_rows = [
        _score_sample(
            provider,
            sample,
            variant=Variant("halueval_full_context"),
            null_bank_embeddings=null_bank_embeddings,
            nli_provider=nli_provider,
            nli_reranker=nli_reranker,
            chunk_token_budget=args.chunk_token_budget,
        )
        for sample in halueval_samples
    ]

    factscore_summary = _summarize_factscore(factscore_rows)
    halueval_summary = _summarize_halueval(halueval_rows)

    baseline_f1 = factscore_summary["baseline_full_context"]["metrics"]["groundedness_v2"]["f1"]
    best_variant = None
    best_f1 = -1.0
    for variant, summary in factscore_summary.items():
        for metric_name, metric in summary["metrics"].items():
            if metric["f1"] > best_f1:
                best_f1 = metric["f1"]
                best_variant = (variant, metric_name)
    if best_variant and best_f1 > baseline_f1 + 0.02:
        recommendation = (
            "Bounded pilot found a measurable FActScore lift. Promote the winning "
            f"configuration `{best_variant[0]}` / `{best_variant[1]}` into a larger n>=748 "
            "validation before changing production."
        )
    else:
        recommendation = (
            "Bounded pilot did not show enough evidence to change production defaults. "
            "Keep v1 as-is and use the report to target a larger follow-up sweep."
        )

    payload = {
        "timestamp": timestamp,
        "json_path": str(json_path),
        "markdown_path": str(md_path),
        "inputs": {
            "factscore_total_available": len(factscore_all),
            "factscore_claims": len(factscore_samples),
            "factscore_per_label": args.factscore_per_label,
            "halueval_samples": len(halueval_samples),
            "halueval_pairs_per_stratum": args.halueval_pairs_per_stratum,
            "seed": args.seed,
            "encoder_model": args.encoder_model,
            "provider_class": type(provider).__name__,
            "nli_model": None if args.no_nli else args.nli_model,
            "reranker_enabled": bool(nli_reranker is not None),
            "fusion_grid": FUSION_GRID,
        },
        "factscore_summary": factscore_summary,
        "halueval_summary": halueval_summary,
        "recommendation": recommendation,
        "factscore_worst_errors_baseline": _worst_errors(
            [row for row in factscore_rows if row["variant"] == "baseline_full_context"],
            "groundedness_v2",
        ),
        "factscore_rows": factscore_rows,
        "halueval_rows": halueval_rows,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_markdown(md_path, payload)

    print(json.dumps({
        "json": str(json_path),
        "markdown": str(md_path),
        "baseline_f1": baseline_f1,
        "best_variant": best_variant,
        "best_f1": best_f1,
        "recommendation": recommendation,
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
