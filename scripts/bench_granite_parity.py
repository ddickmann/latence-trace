"""Benchmark Latence Trace on Granite Guardian parity suites.

This is an evaluation-only harness. It intentionally does not feed any rows
into training, calibration, or student-head artefact generation.

Suites:

* ``lm-aggrefact``: grounded factuality benchmark used by Granite Guardian's
  RAG hallucination table. Metric: balanced accuracy.
* ``fc-reward``: function-calling hallucination benchmark used by Granite
  Guardian's FC Reward Bench table. Metric: balanced accuracy over paired
  chosen/rejected calls.

Examples:

    # Parser/metric smoke test, no endpoint required.
    python scripts/bench_granite_parity.py --suite all --fixture-smoke --mock-score oracle

    # Real TRACE endpoint run against RunPod /runsync.
    python scripts/bench_granite_parity.py \
      --suite lm-aggrefact --split test --url https://api.runpod.ai/v2/.../runsync
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence


_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


GRANITE_41_RAG_THINK = {
    "AggreFact-CNN": 0.606,
    "AggreFact-XSum": 0.765,
    "ClaimVerify": 0.773,
    "ExpertQA": 0.605,
    "FactCheck-GPT": 0.752,
    "LFQA": 0.885,
    "RAGTruth": 0.841,
    "Reveal": 0.888,
    "TofuEval-MediaS": 0.730,
    "TofuEval-MeetB": 0.795,
    "Wice": 0.767,
}
GRANITE_41_RAG_NON_THINK = {
    "AggreFact-CNN": 0.598,
    "AggreFact-XSum": 0.763,
    "ClaimVerify": 0.757,
    "ExpertQA": 0.603,
    "FactCheck-GPT": 0.749,
    "LFQA": 0.883,
    "RAGTruth": 0.834,
    "Reveal": 0.889,
    "TofuEval-MediaS": 0.737,
    "TofuEval-MeetB": 0.767,
    "Wice": 0.783,
}
GRANITE_41_FC_NON_THINK = 0.79

RAG_ORDER = (
    "AggreFact-CNN",
    "AggreFact-XSum",
    "ClaimVerify",
    "ExpertQA",
    "FactCheck-GPT",
    "LFQA",
    "RAGTruth",
    "Reveal",
    "TofuEval-MediaS",
    "TofuEval-MeetB",
    "Wice",
)

_DATASET_ALIASES = {
    "aggrefact-cnn": "AggreFact-CNN",
    "cnn": "AggreFact-CNN",
    "aggrefact-xsum": "AggreFact-XSum",
    "xsum": "AggreFact-XSum",
    "claim verify": "ClaimVerify",
    "claimverify": "ClaimVerify",
    "claim_verify": "ClaimVerify",
    "expert qa": "ExpertQA",
    "expertqa": "ExpertQA",
    "expert_qa": "ExpertQA",
    "fact check": "FactCheck-GPT",
    "factcheck": "FactCheck-GPT",
    "factcheck-gpt": "FactCheck-GPT",
    "fact_check_gpt": "FactCheck-GPT",
    "lfqa": "LFQA",
    "ragtruth": "RAGTruth",
    "rag truth": "RAGTruth",
    "rag_truth": "RAGTruth",
    "reveal": "Reveal",
    "tofueval-medias": "TofuEval-MediaS",
    "tofueval-mediasum": "TofuEval-MediaS",
    "tofueval_mediasum": "TofuEval-MediaS",
    "medias": "TofuEval-MediaS",
    "mediasum": "TofuEval-MediaS",
    "tofueval-meetb": "TofuEval-MeetB",
    "tofueval-meetingbank": "TofuEval-MeetB",
    "tofueval_meetingbank": "TofuEval-MeetB",
    "meetb": "TofuEval-MeetB",
    "meetingbank": "TofuEval-MeetB",
    "wice": "Wice",
    "wice/": "Wice",
}


@dataclass(frozen=True)
class BenchRow:
    suite: str
    dataset: str
    row_id: str
    query: str
    context: str
    response: str
    supported: bool
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoredRow:
    row: BenchRow
    score: Optional[float]
    predicted_supported: Optional[bool]
    latency_ms: Optional[float]
    error: Optional[str] = None
    output: dict[str, Any] = field(default_factory=dict)


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _norm_dataset_name(value: Any) -> str:
    raw = str(value or "").strip()
    key = raw.lower().replace("_", "-")
    key = " ".join(key.split())
    key = key.replace(" ", "-")
    return _DATASET_ALIASES.get(key, _DATASET_ALIASES.get(raw.lower(), raw))


def _jsonish(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _conversation_user_text(conversation: Any) -> str:
    if isinstance(conversation, str):
        return conversation
    if not isinstance(conversation, Sequence):
        return _jsonish(conversation)
    chunks: list[str] = []
    for turn in conversation:
        if not isinstance(turn, Mapping):
            continue
        role = str(turn.get("role") or turn.get("from") or "").lower()
        content = turn.get("content") or turn.get("value") or turn.get("text")
        if role in {"user", "human"} or not chunks:
            chunks.append(str(content or ""))
    return "\n".join(c for c in chunks if c).strip()


def _import_load_dataset():
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on local extras
        raise RuntimeError(
            "The `datasets` package is required for public HF suites. "
            "Install it or use --fixture-smoke for parser-only validation."
        ) from exc
    return load_dataset


def _load_lm_aggrefact(
    *,
    split: str,
    limit_per_dataset: Optional[int],
    datasets_filter: Optional[set[str]],
) -> list[BenchRow]:
    load_dataset = _import_load_dataset()
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    load_kwargs = {"split": split}
    if hf_token:
        load_kwargs["token"] = hf_token
    try:
        ds = load_dataset("lytang/LLM-AggreFact", **load_kwargs)
    except Exception as exc:
        raise RuntimeError(
            "Unable to load `lytang/LLM-AggreFact`. The dataset is gated on "
            "Hugging Face; accept its terms and authenticate with `huggingface-cli "
            "login` or provide `HF_TOKEN` before running the full Granite RAG parity "
            "benchmark."
        ) from exc
    counts: dict[str, int] = {}
    rows: list[BenchRow] = []
    for idx, rec in enumerate(ds):
        dataset = _norm_dataset_name(rec.get("dataset"))
        if datasets_filter and dataset not in datasets_filter:
            continue
        emitted = counts.get(dataset, 0)
        if limit_per_dataset is not None and emitted >= limit_per_dataset:
            continue
        label = rec.get("label")
        supported = bool(int(label)) if str(label).strip() in {"0", "1"} else bool(label)
        rows.append(
            BenchRow(
                suite="lm-aggrefact",
                dataset=dataset,
                row_id=str(rec.get("contamination_identifier") or f"{split}-{idx}"),
                query=str(rec.get("claim") or ""),
                context=str(rec.get("doc") or ""),
                response=str(rec.get("claim") or ""),
                supported=supported,
                metadata={
                    "split": split,
                    "source_dataset": rec.get("dataset"),
                    "contamination_identifier": rec.get("contamination_identifier"),
                },
            )
        )
        counts[dataset] = emitted + 1
    return rows


def _load_fc_reward(
    *,
    split: str,
    limit_per_dataset: Optional[int],
) -> list[BenchRow]:
    load_dataset = _import_load_dataset()
    try:
        loaded = load_dataset("ibm-research/fc-reward-bench")
    except Exception as exc:
        raise RuntimeError(
            "Unable to load `ibm-research/fc-reward-bench` from Hugging Face."
        ) from exc
    if hasattr(loaded, "keys"):
        split_name = split if split in loaded else next(iter(loaded.keys()))
        ds = loaded[split_name]
    else:
        split_name = split
        ds = loaded
    rows: list[BenchRow] = []
    count = 0
    for idx, rec in enumerate(ds):
        if limit_per_dataset is not None and count >= limit_per_dataset:
            break
        tools = rec.get("tools")
        query = _conversation_user_text(rec.get("conversation"))
        context = json.dumps({"tools": tools}, ensure_ascii=False, sort_keys=True)
        base_meta = {
            "split": split_name,
            "error_type": rec.get("error_type"),
            "model_name": rec.get("model_name"),
            "test_category": rec.get("test_category"),
            "test_id": rec.get("test_id"),
            "fc_mapping": "experimental_code_agentic_trace",
        }
        rows.append(
            BenchRow(
                suite="fc-reward",
                dataset="FC Reward Bench",
                row_id=f"{rec.get('test_id') or idx}:chosen",
                query=query,
                context=context,
                response=_jsonish(rec.get("chosen_output")),
                supported=True,
                metadata={**base_meta, "pair_side": "chosen"},
            )
        )
        rows.append(
            BenchRow(
                suite="fc-reward",
                dataset="FC Reward Bench",
                row_id=f"{rec.get('test_id') or idx}:rejected",
                query=query,
                context=context,
                response=_jsonish(rec.get("rejected_output")),
                supported=False,
                metadata={**base_meta, "pair_side": "rejected"},
            )
        )
        count += 1
    return _sample_rows_stratified(
        rows,
        sample_per_dataset=args.sample_per_dataset,
        seed=args.sample_seed,
    )


def _sample_rows_stratified(
    rows: Sequence[BenchRow],
    *,
    sample_per_dataset: Optional[int],
    seed: int,
) -> list[BenchRow]:
    if sample_per_dataset is None:
        return list(rows)
    rng = random.Random(seed)
    by_dataset: dict[str, list[BenchRow]] = {}
    for row in rows:
        by_dataset.setdefault(row.dataset, []).append(row)
    sampled: list[BenchRow] = []
    for dataset in sorted(by_dataset):
        dataset_rows = list(by_dataset[dataset])
        if len(dataset_rows) <= sample_per_dataset:
            sampled.extend(dataset_rows)
            continue
        positives = [row for row in dataset_rows if row.supported]
        negatives = [row for row in dataset_rows if not row.supported]
        rng.shuffle(positives)
        rng.shuffle(negatives)
        if positives and negatives:
            half = sample_per_dataset // 2
            pos_take = min(len(positives), half)
            neg_take = min(len(negatives), sample_per_dataset - pos_take)
            if pos_take + neg_take < sample_per_dataset:
                remaining_pos = len(positives) - pos_take
                remaining_neg = len(negatives) - neg_take
                extra = sample_per_dataset - pos_take - neg_take
                add_pos = min(remaining_pos, extra)
                pos_take += add_pos
                neg_take += min(remaining_neg, extra - add_pos)
            chosen = positives[:pos_take] + negatives[:neg_take]
        else:
            chosen = dataset_rows[:]
            rng.shuffle(chosen)
            chosen = chosen[:sample_per_dataset]
        chosen.sort(key=lambda row: row.row_id)
        sampled.extend(chosen)
    return sampled


def _fixture_rows() -> list[BenchRow]:
    return [
        BenchRow(
            suite="lm-aggrefact",
            dataset="AggreFact-CNN",
            row_id="fixture-rag-supported",
            query="The Eiffel Tower is in Paris.",
            context="The Eiffel Tower is a landmark in Paris, France.",
            response="The Eiffel Tower is in Paris.",
            supported=True,
            metadata={"fixture": True},
        ),
        BenchRow(
            suite="lm-aggrefact",
            dataset="AggreFact-CNN",
            row_id="fixture-rag-unsupported",
            query="The Eiffel Tower is in Berlin.",
            context="The Eiffel Tower is a landmark in Paris, France.",
            response="The Eiffel Tower is in Berlin.",
            supported=False,
            metadata={"fixture": True},
        ),
        BenchRow(
            suite="fc-reward",
            dataset="simple",
            row_id="fixture-fc-chosen",
            query="Fetch the first 15 comments for video 456789123.",
            context=json.dumps(
                {
                    "tools": [
                        {
                            "name": "comment_list",
                            "parameters": {"aweme_id": {"type": "int"}, "count": {"type": "int"}},
                        }
                    ]
                },
                sort_keys=True,
            ),
            response=json.dumps(
                [{"name": "comment_list", "arguments": {"aweme_id": 456789123, "count": 15}}],
                sort_keys=True,
            ),
            supported=True,
            metadata={"fixture": True},
        ),
        BenchRow(
            suite="fc-reward",
            dataset="Unexpected parameter",
            row_id="fixture-fc-rejected",
            query="Fetch the first 15 comments for video 456789123.",
            context=json.dumps(
                {
                    "tools": [
                        {
                            "name": "comment_list",
                            "parameters": {"aweme_id": {"type": "int"}, "count": {"type": "int"}},
                        }
                    ]
                },
                sort_keys=True,
            ),
            response=json.dumps(
                [{"name": "comment_list", "arguments": {"video_id": 456789123, "count": 15}}],
                sort_keys=True,
            ),
            supported=False,
            metadata={"fixture": True},
        ),
    ]


def _payload_for_row(row: BenchRow, *, profile: str, fc_mode: str) -> dict[str, Any]:
    if row.suite == "fc-reward":
        if fc_mode == "rag_structured":
            scoring_mode = "rag"
            corpus_type = "rag.structured"
        else:
            scoring_mode = "code"
            corpus_type = "code.agentic_trace"
        return {
            "action": "score",
            "scoring_mode": scoring_mode,
            "corpus_type": corpus_type,
            "query_text": row.query,
            "raw_context": row.context,
            "response_text": row.response,
            "profile": profile,
            "primary_metric": "reverse_context",
            "response_language_hint": "json",
            "session_id": f"granite-parity:{row.row_id}",
            "include_triangular_diagnostics": False,
            "auto_decide": True,
        }
    return {
        "action": "score",
        "scoring_mode": "rag",
        "query_text": row.query,
        "raw_context": row.context,
        "response_text": row.response,
        "profile": profile,
        "primary_metric": "reverse_context",
        "include_triangular_diagnostics": False,
        "auto_decide": True,
    }


def _extract_score(output: Mapping[str, Any], suite: str) -> Optional[float]:
    candidates = [
        output.get("groundedness_v2"),
        (output.get("score_channels") or {}).get("groundedness_v2")
        if isinstance(output.get("score_channels"), Mapping)
        else None,
        (output.get("score_channels") or {}).get("primary")
        if isinstance(output.get("score_channels"), Mapping)
        else None,
        output.get("score"),
    ]
    if suite == "fc-reward" and isinstance(output.get("code_lane"), Mapping):
        candidates.insert(0, output["code_lane"].get("composite_score"))
    for value in candidates:
        if isinstance(value, (int, float)) and not math.isnan(float(value)):
            return float(value)
    return None


def _prediction_from_output(
    output: Mapping[str, Any],
    *,
    score: Optional[float],
    threshold: float,
    decision_source: str,
) -> Optional[bool]:
    if decision_source == "runtime":
        runtime = output.get("runtime_decision")
        if isinstance(runtime, Mapping):
            action = str(runtime.get("action") or "").lower()
            if action == "allow":
                return True
            if action in {"block", "auto_repair", "auto-repair"}:
                return False
    if decision_source == "band":
        band = str(output.get("band") or output.get("risk_band") or "").lower()
        if band in {"green", "allow", "allowed"}:
            return True
        if band in {"red", "block", "blocked", "amber", "yellow", "auto_repair", "auto-repair"}:
            return False
    return None if score is None else bool(score >= threshold)


def _score_http(
    row: BenchRow,
    *,
    url: str,
    profile: str,
    threshold: float,
    timeout_s: float,
    fc_mode: str,
    api_key: str,
    poll_timeout_s: float,
    poll_initial_s: float,
    poll_max_s: float,
    decision_source: str,
) -> ScoredRow:
    payload = {"input": _payload_for_row(row, profile=profile, fc_mode=fc_mode)}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "content-type": "application/json",
            "accept": "application/json",
            "x-latence-tenant-id": "granite-parity",
            **({"Authorization": f"Bearer {api_key}"} if api_key else {}),
        },
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        if (
            isinstance(body, Mapping)
            and body.get("status") not in {None, "COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}
            and body.get("id")
            and "/runsync" in url
        ):
            body = _poll_runpod_status(
                url=url,
                job_id=str(body["id"]),
                api_key=api_key,
                timeout_s=poll_timeout_s,
                poll_initial_s=poll_initial_s,
                poll_max_s=poll_max_s,
            )
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        return ScoredRow(
            row=row,
            score=None,
            predicted_supported=None,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            error=str(exc),
        )
    dt = (time.perf_counter() - t0) * 1000.0
    output = body.get("output") if isinstance(body, Mapping) else None
    if not isinstance(output, Mapping):
        output = body if isinstance(body, Mapping) else {}
    score = _extract_score(output, row.suite)
    pred = _prediction_from_output(
        output,
        score=score,
        threshold=threshold,
        decision_source=decision_source,
    )
    status = body.get("status") if isinstance(body, Mapping) else None
    error = None
    if status and status != "COMPLETED":
        error = str(output.get("error") or body.get("error") or status)
    if output.get("success") is False:
        error = str(output.get("error") or "success=false")
    return ScoredRow(
        row=row,
        score=score,
        predicted_supported=pred,
        latency_ms=dt,
        error=error,
        output=dict(output),
    )


def _poll_runpod_status(
    *,
    url: str,
    job_id: str,
    api_key: str,
    timeout_s: float,
    poll_initial_s: float,
    poll_max_s: float,
) -> dict[str, Any]:
    if "/runsync" not in url:
        raise RuntimeError("can only poll RunPod jobs for /runsync URLs")
    status_url = url.replace("/runsync", f"/status/{job_id}")
    headers = {
        "accept": "application/json",
        **({"Authorization": f"Bearer {api_key}"} if api_key else {}),
    }
    deadline = time.perf_counter() + timeout_s
    delay = max(0.5, poll_initial_s)
    while True:
        if time.perf_counter() > deadline:
            return {"status": "TIMED_OUT", "id": job_id, "error": "poll timeout"}
        time.sleep(delay)
        delay = min(delay * 1.5, max(delay, poll_max_s))
        req = urllib.request.Request(status_url, headers=headers)
        with urllib.request.urlopen(req, timeout=min(30.0, timeout_s)) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        if body.get("status") in {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}:
            return body


def _score_mock(row: BenchRow, *, mode: str, threshold: float) -> ScoredRow:
    if mode == "oracle":
        score = 0.9 if row.supported else 0.1
    elif mode == "inverted":
        score = 0.1 if row.supported else 0.9
    else:
        score = 0.5
    return ScoredRow(
        row=row,
        score=score,
        predicted_supported=bool(score >= threshold),
        latency_ms=0.0,
        output={"mock_score": mode},
    )


def _balanced_accuracy(rows: Sequence[ScoredRow]) -> dict[str, Any]:
    supported_total = unsupported_total = 0
    supported_correct = unsupported_correct = 0
    errors = 0
    for item in rows:
        if item.predicted_supported is None or item.error:
            errors += 1
            continue
        if item.row.supported:
            supported_total += 1
            if item.predicted_supported:
                supported_correct += 1
        else:
            unsupported_total += 1
            if not item.predicted_supported:
                unsupported_correct += 1
    supported_recall = (
        supported_correct / supported_total if supported_total > 0 else None
    )
    unsupported_recall = (
        unsupported_correct / unsupported_total if unsupported_total > 0 else None
    )
    bacc = (
        (supported_recall + unsupported_recall) / 2.0
        if supported_recall is not None and unsupported_recall is not None
        else None
    )
    return {
        "n": len(rows),
        "errors": errors,
        "supported_total": supported_total,
        "unsupported_total": unsupported_total,
        "supported_recall": supported_recall,
        "unsupported_recall": unsupported_recall,
        "balanced_accuracy": bacc,
        "counts": {
            "supported_correct": supported_correct,
            "supported_total": supported_total,
            "unsupported_correct": unsupported_correct,
            "unsupported_total": unsupported_total,
        },
    }


def _best_threshold_bacc(rows: Sequence[ScoredRow]) -> dict[str, Any]:
    usable = [r for r in rows if r.score is not None and not r.error]
    if not usable:
        return {"threshold": None, "balanced_accuracy": None}
    thresholds = sorted({float(r.score) for r in usable})
    thresholds.append(max(thresholds) + 1e-9)
    best: Optional[dict[str, Any]] = None
    for threshold in thresholds:
        remapped = [
            ScoredRow(
                row=r.row,
                score=r.score,
                predicted_supported=bool(float(r.score or 0.0) >= threshold),
                latency_ms=r.latency_ms,
                error=r.error,
                output=r.output,
            )
            for r in usable
        ]
        metric = _balanced_accuracy(remapped)
        candidate = {
            "threshold": threshold,
            "balanced_accuracy": metric["balanced_accuracy"],
            "supported_recall": metric["supported_recall"],
            "unsupported_recall": metric["unsupported_recall"],
        }
        if (
            best is None
            or (candidate["balanced_accuracy"] or -1.0)
            > (best["balanced_accuracy"] or -1.0)
        ):
            best = candidate
    return best or {"threshold": None, "balanced_accuracy": None}


def _latency_summary(rows: Sequence[ScoredRow]) -> dict[str, Any]:
    vals = sorted(
        float(r.latency_ms)
        for r in rows
        if isinstance(r.latency_ms, (int, float)) and not math.isnan(float(r.latency_ms))
    )
    if not vals:
        return {"p50_ms": None, "p95_ms": None, "mean_ms": None}
    p95_idx = min(len(vals) - 1, int(math.ceil(0.95 * len(vals))) - 1)
    return {
        "p50_ms": vals[len(vals) // 2],
        "p95_ms": vals[p95_idx],
        "mean_ms": statistics.fmean(vals),
    }


def _aggregate(scored: Sequence[ScoredRow], *, threshold: float) -> dict[str, Any]:
    by_dataset: dict[str, list[ScoredRow]] = {}
    for row in scored:
        by_dataset.setdefault(row.row.dataset, []).append(row)
    datasets: dict[str, Any] = {}
    for dataset, rows in sorted(by_dataset.items()):
        fixed = _balanced_accuracy(rows)
        datasets[dataset] = {
            **fixed,
            "best_threshold": _best_threshold_bacc(rows),
            "latency": _latency_summary(rows),
        }
    macro_values = [
        payload["balanced_accuracy"]
        for payload in datasets.values()
        if payload.get("balanced_accuracy") is not None
    ]
    return {
        "threshold": threshold,
        "total_rows": len(scored),
        "overall": _balanced_accuracy(scored),
        "macro_balanced_accuracy": (
            statistics.fmean(macro_values) if macro_values else None
        ),
        "datasets": datasets,
        "latency": _latency_summary(scored),
    }


def _round(value: Any, digits: int = 4) -> Any:
    if isinstance(value, float):
        return round(value, digits)
    return value


def _scored_row_to_record(item: ScoredRow) -> dict[str, Any]:
    route = item.output.get("corpus_route") if isinstance(item.output, Mapping) else None
    runtime = item.output.get("runtime_decision") if isinstance(item.output, Mapping) else None
    profile_diag = item.output.get("profile_diagnostics") if isinstance(item.output, Mapping) else None
    scores = item.output.get("scores") if isinstance(item.output, Mapping) else None
    return {
        "suite": item.row.suite,
        "dataset": item.row.dataset,
        "row_id": item.row.row_id,
        "supported": item.row.supported,
        "score": item.score,
        "predicted_supported": item.predicted_supported,
        "latency_ms": item.latency_ms,
        "error": item.error,
        "metadata": item.row.metadata,
        "audit": {
            "band": item.output.get("band") or item.output.get("risk_band"),
            "corpus_route": route if isinstance(route, Mapping) else None,
            "profile_diagnostics": (
                profile_diag if isinstance(profile_diag, Mapping) else None
            ),
            "runtime_decision": runtime if isinstance(runtime, Mapping) else None,
            "score_channels": (
                item.output.get("score_channels")
                if isinstance(item.output.get("score_channels"), Mapping)
                else None
            ),
            "scores": scores if isinstance(scores, Mapping) else None,
            "warnings": item.output.get("warnings"),
        },
    }


def _record_to_scored_row(record: Mapping[str, Any]) -> ScoredRow:
    audit = record.get("audit") if isinstance(record.get("audit"), Mapping) else {}
    return ScoredRow(
        row=BenchRow(
            suite=str(record.get("suite") or ""),
            dataset=str(record.get("dataset") or ""),
            row_id=str(record.get("row_id") or ""),
            query="",
            context="",
            response="",
            supported=bool(record.get("supported")),
            metadata=dict(record.get("metadata") or {}),
        ),
        score=record.get("score") if isinstance(record.get("score"), (int, float)) else None,
        predicted_supported=(
            bool(record["predicted_supported"])
            if isinstance(record.get("predicted_supported"), bool)
            else None
        ),
        latency_ms=(
            float(record["latency_ms"])
            if isinstance(record.get("latency_ms"), (int, float))
            else None
        ),
        error=str(record.get("error")) if record.get("error") else None,
        output={
            "band": audit.get("band"),
            "corpus_route": audit.get("corpus_route"),
            "profile_diagnostics": audit.get("profile_diagnostics"),
            "runtime_decision": audit.get("runtime_decision"),
            "score_channels": audit.get("score_channels"),
            "scores": audit.get("scores"),
            "warnings": audit.get("warnings"),
        },
    )


def _load_streamed_records(path: Path) -> list[ScoredRow]:
    if not path.exists():
        return []
    rows: list[ScoredRow] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(_record_to_scored_row(json.loads(line)))
            except json.JSONDecodeError:
                continue
    return rows


def _target_fingerprint(args: argparse.Namespace, rows: Sequence[BenchRow]) -> dict[str, Any]:
    keys = [f"{row.dataset}\t{row.row_id}" for row in rows]
    digest = hashlib.sha256("\n".join(keys).encode("utf-8")).hexdigest()
    return {
        "suite": args.suite,
        "split": args.split,
        "sample_per_dataset": args.sample_per_dataset,
        "sample_seed": args.sample_seed,
        "profile": args.profile,
        "decision_source": args.decision_source,
        "row_count": len(rows),
        "row_sha256": digest,
    }


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _acquire_lock(lock_path: Path) -> None:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if lock_path.exists():
        try:
            payload = json.loads(lock_path.read_text(encoding="utf-8"))
            pid = int(payload.get("pid") or 0)
        except Exception:
            pid = 0
        if pid > 0 and _pid_alive(pid):
            raise SystemExit(f"active benchmark lock exists: {lock_path} (pid={pid})")
        lock_path.unlink(missing_ok=True)
    lock_path.write_text(
        json.dumps({"pid": os.getpid(), "created_at": datetime.now(timezone.utc).isoformat()}),
        encoding="utf-8",
    )


def _release_lock(lock_path: Optional[Path]) -> None:
    if lock_path is not None:
        lock_path.unlink(missing_ok=True)


def _stream_meta_path(stream_path: Path) -> Path:
    return stream_path.with_suffix(stream_path.suffix + ".meta.json")


def _write_stream_meta(stream_path: Path, meta: Mapping[str, Any]) -> None:
    _stream_meta_path(stream_path).write_text(
        json.dumps(dict(meta), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _check_stream_meta(stream_path: Path, meta: Mapping[str, Any]) -> None:
    meta_path = _stream_meta_path(stream_path)
    if not meta_path.exists():
        return
    existing = json.loads(meta_path.read_text(encoding="utf-8"))
    comparable = {k: existing.get(k) for k in meta}
    if comparable != dict(meta):
        raise SystemExit(
            f"stream metadata mismatch for {stream_path}; use a new --stream-rows path"
        )


def _filter_records_to_targets(
    records: Sequence[ScoredRow],
    rows: Sequence[BenchRow],
) -> list[ScoredRow]:
    by_key: dict[tuple[str, str], ScoredRow] = {}
    for item in records:
        by_key[(item.row.dataset, item.row.row_id)] = item
    return [
        by_key[(row.dataset, row.row_id)]
        for row in rows
        if (row.dataset, row.row_id) in by_key
    ]


def _nested_get(payload: Mapping[str, Any], path: str) -> Any:
    current: Any = payload
    for part in path.split("."):
        if not isinstance(current, Mapping):
            return None
        current = current.get(part)
    return current


def _calibrator_value(item: ScoredRow, field: str) -> Optional[float]:
    if field == "score":
        return item.score
    value = _nested_get(item.output, field)
    if isinstance(value, (int, float)) and not math.isnan(float(value)):
        return float(value)
    return None


def _calibrator_group(item: ScoredRow, group_field: Optional[str]) -> str:
    if not group_field:
        return "global"
    if group_field == "dataset":
        return item.row.dataset
    value = _nested_get(item.output, group_field)
    return str(value or "missing")


def _apply_calibrator(scored: Sequence[ScoredRow], path: str) -> list[ScoredRow]:
    if not path:
        return list(scored)
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    selected = artifact.get("selected_candidate") or artifact.get("candidate") or {}
    score_field = str(selected.get("score_field") or "score")
    group_field = selected.get("group_field")
    thresholds = selected.get("thresholds") if isinstance(selected.get("thresholds"), Mapping) else {}
    fallback = float(selected.get("fallback_threshold", selected.get("threshold", 0.5)))
    remapped: list[ScoredRow] = []
    for item in scored:
        value = _calibrator_value(item, score_field)
        if value is None:
            remapped.append(item)
            continue
        group = _calibrator_group(item, str(group_field) if group_field else None)
        threshold = float(thresholds.get(group, fallback))
        remapped.append(
            ScoredRow(
                row=item.row,
                score=value,
                predicted_supported=bool(value >= threshold),
                latency_ms=item.latency_ms,
                error=item.error,
                output=item.output,
            )
        )
    return remapped


def _write_reports(
    *,
    scored: Sequence[ScoredRow],
    aggregate: Mapping[str, Any],
    args: argparse.Namespace,
    caveats: Sequence[str],
) -> tuple[Path, Path, Path]:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"granite_parity_{args.suite}_{_now_stamp()}"
    report_path = out_dir / f"{stem}.json"
    rows_path = Path(args.stream_rows) if args.stream_rows else out_dir / f"{stem}.rows.jsonl"
    md_path = out_dir / f"{stem}.md"

    report = {
        "suite": args.suite,
        "split": args.split,
        "score_source": "http" if args.url else f"mock:{args.mock_score}",
        "url_used": bool(args.url),
        "profile": args.profile,
        "threshold": args.threshold,
        "decision_source": args.decision_source,
        "fc_mode": args.fc_mode,
        "sample_per_dataset": args.sample_per_dataset,
        "sample_seed": args.sample_seed,
        "calibrator_path": args.calibrator_path,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "granite_reference": {
            "rag_4_1_think": GRANITE_41_RAG_THINK,
            "rag_4_1_non_think": GRANITE_41_RAG_NON_THINK,
            "fc_4_1_non_think": GRANITE_41_FC_NON_THINK,
        },
        "aggregate": aggregate,
        "caveats": list(caveats),
    }
    report_path.write_text(json.dumps(report, indent=2, default=_round), encoding="utf-8")
    if not args.stream_rows:
        with rows_path.open("w", encoding="utf-8") as handle:
            for item in scored:
                handle.write(json.dumps(_scored_row_to_record(item), ensure_ascii=False, sort_keys=True) + "\n")
    md_path.write_text(_markdown_report(report), encoding="utf-8")
    return report_path, rows_path, md_path


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _markdown_report(report: Mapping[str, Any]) -> str:
    agg = report["aggregate"]
    datasets = agg.get("datasets") or {}
    lines = [
        "# Granite Guardian Benchmark Parity",
        "",
        f"- Suite: `{report['suite']}`",
        f"- Split: `{report['split']}`",
        f"- Score source: `{report['score_source']}`",
        f"- Decision source: `{report.get('decision_source')}`",
        f"- Fixed threshold: `{report['threshold']}`",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Macro balanced accuracy | {_fmt(agg.get('macro_balanced_accuracy'))} |",
        f"| Overall balanced accuracy | {_fmt((agg.get('overall') or {}).get('balanced_accuracy'))} |",
        f"| Rows | {agg.get('total_rows')} |",
        "",
        "## Per-Dataset Balanced Accuracy",
        "",
        "| Dataset | Latence fixed | Latence best threshold | Granite 4.1 non-think | Granite 4.1 think | n |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for dataset in list(RAG_ORDER) + sorted(set(datasets) - set(RAG_ORDER)):
        if dataset not in datasets:
            continue
        payload = datasets[dataset]
        best = payload.get("best_threshold") or {}
        granite_non_think = GRANITE_41_RAG_NON_THINK.get(dataset)
        if dataset == "FC Reward Bench":
            granite_non_think = GRANITE_41_FC_NON_THINK
        lines.append(
            "| {dataset} | {fixed} | {best} | {non_think} | {think} | {n} |".format(
                dataset=dataset,
                fixed=_fmt(payload.get("balanced_accuracy")),
                best=_fmt(best.get("balanced_accuracy")),
                non_think=_fmt(granite_non_think),
                think=_fmt(GRANITE_41_RAG_THINK.get(dataset)),
                n=payload.get("n"),
            )
        )
    if report.get("caveats"):
        lines.extend(["", "## Caveats", ""])
        lines.extend(f"- {item}" for item in report["caveats"])
    return "\n".join(lines) + "\n"


def _load_rows(args: argparse.Namespace) -> list[BenchRow]:
    if args.fixture_smoke:
        rows = _fixture_rows()
        if args.suite != "all":
            rows = [row for row in rows if row.suite == args.suite]
        return rows
    datasets_filter = (
        {_norm_dataset_name(name) for name in args.datasets.split(",") if name.strip()}
        if args.datasets
        else None
    )
    rows: list[BenchRow] = []
    if args.suite in {"all", "lm-aggrefact"}:
        rows.extend(
            _load_lm_aggrefact(
                split=args.split,
                limit_per_dataset=args.limit_per_dataset,
                datasets_filter=datasets_filter,
            )
        )
    if args.suite in {"all", "fc-reward"}:
        rows.extend(
            _load_fc_reward(
                split=args.split,
                limit_per_dataset=args.limit_per_dataset,
            )
        )
    return _sample_rows_stratified(
        rows,
        sample_per_dataset=args.sample_per_dataset,
        seed=args.sample_seed,
    )


def _score_rows(args: argparse.Namespace, rows: Sequence[BenchRow]) -> list[ScoredRow]:
    if args.offline_rows:
        offline = _load_streamed_records(Path(args.offline_rows))
        return _filter_records_to_targets(offline, rows)

    api_key = os.environ.get(args.api_key_env, "")
    if args.endpoint_id and not args.url:
        args.url = f"https://api.runpod.ai/v2/{args.endpoint_id}/runsync"
    if args.url and "api.runpod.ai" in args.url and not api_key:
        raise SystemExit(
            f"RunPod URL requires an API key. Set {args.api_key_env} or use --api-key-env."
        )

    def score_one(row: BenchRow) -> ScoredRow:
        if args.url:
            return _score_http(
                row,
                url=args.url,
                profile=args.profile,
                threshold=args.threshold,
                timeout_s=args.timeout_s,
                fc_mode=args.fc_mode,
                api_key=api_key,
                poll_timeout_s=args.poll_timeout_s,
                poll_initial_s=args.poll_initial_s,
                poll_max_s=args.poll_max_s,
                decision_source=args.decision_source,
            )
        else:
            if not args.mock_score:
                raise SystemExit(
                    "--url is required for real TRACE scoring. Use --mock-score oracle "
                    "only for parser/metric smoke tests."
                )
            return _score_mock(row, mode=args.mock_score, threshold=args.threshold)

    stream_path = Path(args.stream_rows) if args.stream_rows else None
    lock_path = Path(args.lock_file) if args.lock_file else (
        stream_path.with_suffix(stream_path.suffix + ".lock") if stream_path else None
    )
    if lock_path is not None:
        _acquire_lock(lock_path)
    completed: dict[tuple[str, str], ScoredRow] = {}
    if stream_path is not None:
        stream_path.parent.mkdir(parents=True, exist_ok=True)
        stream_meta = _target_fingerprint(args, rows)
        if args.resume:
            _check_stream_meta(stream_path, stream_meta)
            for item in _load_streamed_records(stream_path):
                completed[(item.row.dataset, item.row.row_id)] = item
            if completed:
                print(f"resuming from {len(completed)} streamed rows: {stream_path}", flush=True)
        elif stream_path.exists():
            stream_path.unlink()
        if not args.resume:
            _write_stream_meta(stream_path, stream_meta)
    pending_rows = [
        row for row in rows if (row.dataset, row.row_id) not in completed
    ]
    scored: list[ScoredRow] = list(completed.values())

    def record_progress(done: int) -> None:
        aggregate = _aggregate(scored, threshold=args.threshold)
        overall = (aggregate.get("overall") or {}).get("balanced_accuracy")
        macro = aggregate.get("macro_balanced_accuracy")
        print(
            "scored {done}/{total} rows | overall_bacc={overall} | macro_bacc={macro}".format(
                done=done,
                total=len(rows),
                overall=_fmt(overall),
                macro=_fmt(macro),
            ),
            flush=True,
        )

    def append_stream(item: ScoredRow) -> None:
        if stream_path is None:
            return
        with stream_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(_scored_row_to_record(item), ensure_ascii=False, sort_keys=True)
                + "\n"
            )

    try:
        if not pending_rows:
            return scored
        if args.concurrency <= 1:
            for row in pending_rows:
                item = score_one(row)
                scored.append(item)
                append_stream(item)
                if args.progress_every and len(scored) % args.progress_every == 0:
                    record_progress(len(scored))
            return scored

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [pool.submit(score_one, row) for row in pending_rows]
            for fut in concurrent.futures.as_completed(futures):
                item = fut.result()
                scored.append(item)
                append_stream(item)
                if args.progress_every and len(scored) % args.progress_every == 0:
                    record_progress(len(scored))
        return scored
    finally:
        _release_lock(lock_path)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("all", "lm-aggrefact", "fc-reward"), default="all")
    parser.add_argument("--split", default="test")
    parser.add_argument("--datasets", default="", help="Comma-separated LM-AggreFact dataset filter.")
    parser.add_argument("--limit-per-dataset", type=int, default=None)
    parser.add_argument(
        "--sample-per-dataset",
        type=int,
        default=None,
        help="Stratified balanced sample size per dataset after loading the full split.",
    )
    parser.add_argument("--sample-seed", type=int, default=13)
    parser.add_argument("--fixture-smoke", action="store_true")
    parser.add_argument("--url", default="")
    parser.add_argument("--endpoint-id", default=os.environ.get("RUNPOD_ENDPOINT_ID", ""))
    parser.add_argument("--api-key-env", default="RUNPOD_API_KEY")
    parser.add_argument("--profile", default="quality")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--decision-source",
        choices=("band", "runtime", "score"),
        default="band",
        help="Fixed prediction source. Best-threshold diagnostics always use score.",
    )
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--poll-timeout-s", type=float, default=600.0)
    parser.add_argument("--poll-initial-s", type=float, default=10.0)
    parser.add_argument("--poll-max-s", type=float, default=30.0)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--mock-score",
        choices=("oracle", "inverted", "constant"),
        default="",
        help="Parser/metric validation only; never use for Latence-vs-Granite claims.",
    )
    parser.add_argument(
        "--fc-mode",
        choices=("code_agentic", "rag_structured"),
        default="code_agentic",
        help="Experimental mapping for FC Reward Bench rows.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(_REPO / "research/triangular_maxsim/reports"),
    )
    parser.add_argument(
        "--stream-rows",
        default="",
        help="Append each completed row immediately to this JSONL checkpoint.",
    )
    parser.add_argument("--offline-rows", default="", help="Evaluate an existing streamed rows JSONL without scoring live requests.")
    parser.add_argument("--calibrator-path", default="", help="Apply a frozen Granite parity calibrator artifact before metrics.")
    parser.add_argument("--lock-file", default="", help="Override the default stream lock path.")
    parser.add_argument("--dedupe-stream-only", action="store_true", help="Filter --stream-rows to the intended target rows and exit.")
    parser.add_argument("--resume", action="store_true", help="Skip row ids already present in --stream-rows.")
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    caveats: list[str] = []
    if args.mock_score:
        caveats.append(
            "This run used mock scoring. It validates loader and metric plumbing only; "
            "it is not a Latence Trace benchmark result."
        )
    if args.sample_per_dataset is not None:
        caveats.append(
            "This run used a stratified balanced sample per dataset to estimate complete-set "
            "balanced accuracy with lower live-scoring cost. It is representative for BAcc "
            "diagnostics, not a row-exhaustive public split score."
        )
    if args.suite in {"all", "fc-reward"}:
        caveats.append(
            "FC Reward Bench mapping is experimental: Latence Trace currently has no "
            "first-class available_tools input equivalent to Granite Guardian's chat "
            "template. Results should be interpreted as tool-call text validation unless "
            "that API contract is added."
        )
    try:
        rows = _load_rows(args)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not rows:
        raise SystemExit("No benchmark rows loaded. Check dataset access or use --fixture-smoke.")
    if args.dedupe_stream_only:
        if not args.stream_rows:
            raise SystemExit("--dedupe-stream-only requires --stream-rows")
        stream_path = Path(args.stream_rows)
        clean = _filter_records_to_targets(_load_streamed_records(stream_path), rows)
        out = stream_path.with_name(stream_path.stem + ".clean.jsonl")
        with out.open("w", encoding="utf-8") as handle:
            for item in clean:
                handle.write(json.dumps(_scored_row_to_record(item), ensure_ascii=False, sort_keys=True) + "\n")
        print(f"wrote clean rows: {out} ({len(clean)}/{len(rows)})", flush=True)
        return 0
    print(f"loaded {len(rows)} rows", flush=True)
    scored = _score_rows(args, rows)
    scored = _apply_calibrator(scored, args.calibrator_path)
    aggregate = _aggregate(scored, threshold=args.threshold)
    report_path, rows_path, md_path = _write_reports(
        scored=scored,
        aggregate=aggregate,
        args=args,
        caveats=caveats,
    )
    print(f"macro balanced accuracy: {_fmt(aggregate.get('macro_balanced_accuracy'))}")
    print(f"overall balanced accuracy: {_fmt((aggregate.get('overall') or {}).get('balanced_accuracy'))}")
    print(f"report: {report_path}")
    print(f"rows:   {rows_path}")
    print(f"md:     {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
