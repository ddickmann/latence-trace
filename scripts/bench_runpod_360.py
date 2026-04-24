"""360-degree live benchmark against the latence-trace RunPod serverless endpoint.

Exercises every product dimension the service ships in a single pass:

    1. RAG groundedness AUROC on the 20 handcrafted cases
       (research/triangular_maxsim/cases.py).

    2. Unused-context tri-state precision on the held-out topics of
       scripts/bench_unused_context_precision.py, replayed through the
       support_units lane so we measure the live classifier.

    3. Code-lane phantom AUROC + dead-weight attribution on the 12
       handcrafted coding cases (research/triangular_maxsim/coding/
       code_cases.py). Also checks ast_phantom_verdict precision.

    4. Caller-portable session-state round-trip: 5 consecutive code
       turns that carry ``next_session_state`` forward and verify
       total_turns monotonicity, drift_z_score / ema_groundedness
       emission, file_stats accumulation, and recommendation policy.

    5. Concurrency burst: N mixed-lane requests in parallel to confirm
       the per-lane semaphores do not deadlock and p95 stays within SLO.

One pass prints a compact per-dimension table and a final pass/fail
gate. ``--dump report.json`` writes a machine-readable roll-up.

Usage
-----

    export RUNPOD_API_KEY=...
    python scripts/bench_runpod_360.py \
        --endpoint-id campegd1dctnx2 \
        --concurrency 4 \
        --dump reports/runpod_360_$(date +%Y%m%d-%H%M%S).json

Flags
-----

    --skip rag|unused|code|session|concurrency    run a subset
    --cases-rag G1,G3,U1,U4                        RAG subset
    --cases-code CG1,CU1,CA2                       code subset
    --concurrency 4                                per-dim concurrency
    --burst-size 16                                mixed-lane burst size
    --use-run-poll                                 use /run + /status
    --timeout-s 600                                httpx total timeout

Never commit API keys. The script only reads them from ``--api-key``
or ``RUNPOD_API_KEY``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import httpx

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from research.triangular_maxsim.cases import CASES as RAG_CASES  # noqa: E402
from research.triangular_maxsim.coding.code_cases import (  # noqa: E402
    HANDCRAFTED_CASES as CODE_CASES,
)
from research.triangular_maxsim.coding.code_segmenter import (  # noqa: E402
    render_context_files,
)


# ---------------------------------------------------------------------------
# Tiny statistics helpers (no numpy dependency — the bench has to run
# on a laptop with just httpx installed).
# ---------------------------------------------------------------------------


def _auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Mann-Whitney U-style AUROC with tie-aware averaged ranks."""

    pairs = sorted(zip(scores, labels))
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = [0.0] * len(pairs)
    idx = 0
    while idx < len(pairs):
        jdx = idx + 1
        while jdx < len(pairs) and pairs[jdx][0] == pairs[idx][0]:
            jdx += 1
        avg_rank = (idx + jdx + 1) / 2.0
        for k in range(idx, jdx):
            ranks[k] = avg_rank
        idx = jdx
    sum_pos = sum(rank for rank, (_s, label) in zip(ranks, pairs) if label == 1)
    return float((sum_pos - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg))


def _percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return float("nan")
    data = sorted(values)
    k = (len(data) - 1) * pct
    lo = int(k)
    hi = min(lo + 1, len(data) - 1)
    return data[lo] + (data[hi] - data[lo]) * (k - lo)


def _mean(values: Sequence[float]) -> float:
    return float(statistics.fmean(values)) if values else float("nan")


def _fmt_f(value: Optional[float], width: int = 5) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return f"{'-':>{width}}"
    return f"{value:{width}.3f}"


def _fmt_i(value: Optional[int], width: int = 3) -> str:
    if value is None:
        return f"{'-':>{width}}"
    return f"{value:>{width}d}"


# ---------------------------------------------------------------------------
# RunPod transport.
# ---------------------------------------------------------------------------


@dataclass
class Transport:
    endpoint_id: str
    api_key: str
    use_runsync: bool = True
    timeout_s: float = 600.0

    def headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def submit(
        self,
        client: httpx.AsyncClient,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Submit a single scoring request and wait for a terminal status."""

        wall_start = time.perf_counter()
        if self.use_runsync:
            url = f"https://api.runpod.ai/v2/{self.endpoint_id}/runsync"
            resp = await client.post(url, json=payload, headers=self.headers())
            resp.raise_for_status()
            body = resp.json()
            status = body.get("status", "")
            if status not in {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}:
                body = await self._poll(client, body.get("id"))
        else:
            submit_url = f"https://api.runpod.ai/v2/{self.endpoint_id}/run"
            resp = await client.post(submit_url, json=payload, headers=self.headers())
            resp.raise_for_status()
            body = await self._poll(client, resp.json()["id"])
        wall_ms = (time.perf_counter() - wall_start) * 1000.0
        body["_wall_ms"] = wall_ms
        return body

    async def _poll(
        self, client: httpx.AsyncClient, job_id: Optional[str]
    ) -> Dict[str, Any]:
        if not job_id:
            raise RuntimeError("no job id returned from RunPod endpoint")
        url = f"https://api.runpod.ai/v2/{self.endpoint_id}/status/{job_id}"
        delay = 0.5
        while True:
            await asyncio.sleep(delay)
            delay = min(delay * 1.3, 4.0)
            resp = await client.get(url, headers=self.headers())
            resp.raise_for_status()
            body = resp.json()
            if body.get("status", "") in {
                "COMPLETED",
                "FAILED",
                "CANCELLED",
                "TIMED_OUT",
            }:
                return body


async def _with_sem(
    sem: asyncio.Semaphore, coro_factory: Callable[[], Awaitable[Any]]
) -> Any:
    async with sem:
        return await coro_factory()


# ---------------------------------------------------------------------------
# Dimension 1 — RAG handcrafted cases.
# ---------------------------------------------------------------------------


async def _dim_rag(
    client: httpx.AsyncClient,
    transport: Transport,
    cases: Sequence[Any],
    concurrency: int,
) -> Dict[str, Any]:
    sem = asyncio.Semaphore(concurrency)

    async def _one(case: Any) -> Dict[str, Any]:
        payload = {
            "input": {
                "query": case.query,
                "context": case.context,
                "response": case.response,
                "include_triangular_diagnostics": True,
                "evidence_limit": 4,
            }
        }
        body = await transport.submit(client, payload)
        return {"case": case, "body": body}

    results = await asyncio.gather(
        *[_with_sem(sem, lambda c=c: _one(c)) for c in cases]
    )

    rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    server_latencies_ms: List[float] = []
    wall_latencies_ms: List[float] = []
    scores_pos_neg: List[Tuple[float, int]] = []

    for r in results:
        case = r["case"]
        body = r["body"]
        status = body.get("status", "")
        output = body.get("output") or {}
        if status != "COMPLETED" or not output.get("success"):
            errors.append(
                {
                    "id": case.id,
                    "label": case.label,
                    "status": status,
                    "error": output.get("error") or body.get("error"),
                }
            )
            continue
        score = float(output.get("score", float("nan")))
        srv = float(output.get("latency_ms", float("nan")))
        server_latencies_ms.append(srv)
        wall_latencies_ms.append(body.get("_wall_ms", float("nan")))
        if case.label == "grounded":
            scores_pos_neg.append((score, 1))
        elif case.label == "ungrounded":
            scores_pos_neg.append((score, 0))
        rows.append(
            {
                "id": case.id,
                "label": case.label,
                "subcategory": case.subcategory,
                "score": score,
                "band": output.get("band"),
                "coverage": output.get("context_coverage_ratio"),
                "unused": output.get("context_unused_ratio"),
                "uncertain": output.get("context_uncertain_ratio"),
                "usage_used": (output.get("support_units_usage") or {}).get("used"),
                "usage_unused": (output.get("support_units_usage") or {}).get("unused"),
                "server_latency_ms": srv,
                "wall_ms": body.get("_wall_ms"),
            }
        )

    auroc = _auroc(
        [s for s, _ in scores_pos_neg], [lbl for _, lbl in scores_pos_neg]
    )
    grounded_mean = _mean([s for s, l in scores_pos_neg if l == 1])
    ungrounded_mean = _mean([s for s, l in scores_pos_neg if l == 0])
    ambiguous_mean = _mean([r["score"] for r in rows if r["label"] == "ambiguous"])

    return {
        "name": "rag_handcrafted",
        "rows": rows,
        "errors": errors,
        "auroc_grounded_vs_ungrounded": auroc,
        "mean_score_grounded": grounded_mean,
        "mean_score_ungrounded": ungrounded_mean,
        "mean_score_ambiguous": ambiguous_mean,
        "server_p50_ms": _percentile(server_latencies_ms, 0.5),
        "server_p95_ms": _percentile(server_latencies_ms, 0.95),
        "server_p99_ms": _percentile(server_latencies_ms, 0.99),
        "wall_p50_ms": _percentile(wall_latencies_ms, 0.5),
        "wall_p95_ms": _percentile(wall_latencies_ms, 0.95),
        # Gate: we want grounded > ungrounded separation in the live service.
        "passed": (
            math.isnan(auroc) is False
            and auroc >= 0.75
            and len(errors) == 0
        ),
    }


# ---------------------------------------------------------------------------
# Dimension 2 — Unused-context tri-state precision on held-out topics.
# ---------------------------------------------------------------------------


def _held_out_unused_cases() -> List[Dict[str, Any]]:
    """Replay of the heldout split from ``bench_unused_context_precision.py``.

    Each case carries support_units with gold usage labels. ``exact`` cases
    stress the ``unused`` rail (the service must flag the distractor as
    unused); ``near_duplicate`` stresses ``uncertain`` (paraphrase is not a
    confident ``used``); ``mixed_window`` stresses ``uncertain`` abstention.

    We intentionally pass a real user query (distinct from the response) so
    the service sees a realistic retrieval shape, and keep distractors/noise
    topically far from the anchor to give the classifier every chance.
    """

    topics = [
        {
            "name": "saturn",
            "query": "What are Saturn's rings made of?",
            "anchor": "Saturn has rings made of ice.",
            "paraphrase": "Saturn's rings are composed of ice.",
            "mixed_noise": (
                "The observatory brochure also lists parking rules, telescope hours, "
                "and cafe menus that the answer never references."
            ),
            "distractor": "Bamboo grows quickly in warm climates.",
        },
        {
            "name": "curie",
            "query": "When did Marie Curie win her first Nobel Prize?",
            "anchor": "Marie Curie won the Nobel Prize in 1903.",
            "paraphrase": "Marie Curie received the 1903 Nobel Prize.",
            "mixed_noise": (
                "The archive also catalogs lecture schedules, train routes, and "
                "museum gift-shop inventory that the answer ignores."
            ),
            "distractor": "Coral reefs host diverse marine ecosystems.",
        },
        {
            "name": "meeting",
            "query": "What time does the meeting start?",
            "anchor": "The meeting starts at 9 AM.",
            "paraphrase": "The meeting begins at 9 in the morning.",
            "mixed_noise": (
                "Agenda notes also cover lunch catering, hallway signage, and seat "
                "assignments that the answer never mentions."
            ),
            "distractor": "Maple syrup is made from tree sap.",
        },
        {
            "name": "ev",
            "query": "What do electric cars run on?",
            "anchor": "Electric cars use batteries.",
            "paraphrase": "Electric vehicles run on battery power.",
            "mixed_noise": (
                "The brochure also describes paint options, showroom lighting, and "
                "coffee-bar coupons that the answer never mentions."
            ),
            "distractor": "Saffron is a spice made from crocus flowers.",
        },
    ]
    generic_noise = "Lanterns glow above the quiet harbor at dusk."

    cases: List[Dict[str, Any]] = []
    for t in topics:
        # exact + distractor + generic noise: distractor+noise must be unused.
        cases.append(
            {
                "id": f"{t['name']}_exact",
                "variant": "exact",
                "query": t["query"],
                "response": t["anchor"],
                "support_units": [
                    {"source_id": "s0", "text": t["anchor"], "gold": "used"},
                    {"source_id": "s1", "text": t["distractor"], "gold": "unused"},
                    {"source_id": "s2", "text": generic_noise, "gold": "unused"},
                ],
            }
        )
        # near-duplicate paraphrase should be uncertain, not used or unused.
        cases.append(
            {
                "id": f"{t['name']}_near_duplicate",
                "variant": "near_duplicate",
                "query": t["query"],
                "response": t["anchor"],
                "support_units": [
                    {"source_id": "s0", "text": t["anchor"], "gold": "used"},
                    {"source_id": "s1", "text": t["paraphrase"], "gold": "uncertain"},
                    {"source_id": "s2", "text": t["distractor"], "gold": "unused"},
                ],
            }
        )
        # mixed window must abstain to uncertain on the anchor+noise bundle.
        cases.append(
            {
                "id": f"{t['name']}_mixed_window",
                "variant": "mixed_window",
                "query": t["query"],
                "response": t["anchor"],
                "support_units": [
                    {
                        "source_id": "s0",
                        "text": f"{t['anchor']} {t['mixed_noise']}",
                        "gold": "uncertain",
                    },
                    {"source_id": "s1", "text": generic_noise, "gold": "unused"},
                ],
            }
        )
    return cases


async def _dim_unused(
    client: httpx.AsyncClient,
    transport: Transport,
    concurrency: int,
) -> Dict[str, Any]:
    cases = _held_out_unused_cases()
    sem = asyncio.Semaphore(concurrency)

    async def _one(case: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            "input": {
                "query": case.get("query") or case["response"],
                "support_units": [
                    {"source_id": u["source_id"], "text": u["text"]}
                    for u in case["support_units"]
                ],
                "response": case["response"],
                "evidence_limit": 4,
                "primary_metric": "reverse_context",
                "coverage_threshold": 0.5,
            }
        }
        body = await transport.submit(client, payload)
        return {"case": case, "body": body}

    results = await asyncio.gather(
        *[_with_sem(sem, lambda c=c: _one(c)) for c in cases]
    )

    rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    confusion = {
        g: {p: 0 for p in ("used", "unused", "uncertain")}
        for g in ("used", "unused", "uncertain")
    }

    for r in results:
        case = r["case"]
        body = r["body"]
        status = body.get("status", "")
        output = body.get("output") or {}
        if status != "COMPLETED" or not output.get("success"):
            errors.append(
                {
                    "id": case["id"],
                    "status": status,
                    "error": output.get("error") or body.get("error"),
                }
            )
            continue
        predicted_by_source: Dict[str, Dict[str, Any]] = {}
        for idx, unit in enumerate(output.get("support_units") or []):
            sid = unit.get("source_id") or unit.get("support_id") or f"u{idx}"
            state = unit.get("usage_state")
            predicted_by_source[str(sid)] = {
                "usage_state": str(state) if state else "missing",
                "usage_confidence": unit.get("usage_confidence"),
                "unused_confidence": unit.get("unused_confidence"),
                "coverage_score": unit.get("coverage_score"),
            }
        row_cells: List[Dict[str, Any]] = []
        for idx, gold_unit in enumerate(case["support_units"]):
            sid = gold_unit["source_id"]
            gold = gold_unit["gold"]
            # The service re-labels support_ids as ``u<idx>`` but echoes
            # the caller-supplied source_id back on the matching unit;
            # some deployments also expose ``support_id`` as the stable
            # id. Prefer source_id -> support_id -> index fallback.
            entry = predicted_by_source.get(sid)
            if entry is None and sid.startswith("s"):
                entry = predicted_by_source.get(f"u{sid[1:]}") or predicted_by_source.get(
                    sid.replace("s", "u")
                )
            if entry is None:
                entry = {"usage_state": "missing", "usage_confidence": None,
                         "unused_confidence": None, "coverage_score": None}
            pred = entry["usage_state"]
            if pred in confusion.get(gold, {}):
                confusion[gold][pred] += 1
            row_cells.append(
                {
                    "source_id": sid,
                    "gold": gold,
                    "pred": pred,
                    "coverage_score": entry.get("coverage_score"),
                    "usage_confidence": entry.get("usage_confidence"),
                    "unused_confidence": entry.get("unused_confidence"),
                }
            )
        rows.append(
            {
                "id": case["id"],
                "variant": case["variant"],
                "cells": row_cells,
                "latency_ms": float(output.get("latency_ms", float("nan"))),
            }
        )

    def _precision(label: str) -> Tuple[Optional[float], int, int]:
        preds = sum(confusion[g][label] for g in confusion)
        tp = confusion[label][label]
        prec = (tp / preds) if preds > 0 else None
        return prec, tp, preds

    def _recall(label: str) -> Optional[float]:
        total_gold = sum(confusion[label][p] for p in confusion[label])
        if total_gold == 0:
            return None
        return confusion[label][label] / total_gold

    used_p, used_tp, used_pred = _precision("used")
    unused_p, unused_tp, unused_pred = _precision("unused")
    uncertain_p, uncertain_tp, uncertain_pred = _precision("uncertain")

    # Diagnostic: coverage score distribution for units whose gold is unused.
    unused_gold_coverage = [
        cell.get("coverage_score")
        for row in rows
        for cell in row["cells"]
        if cell.get("gold") == "unused" and cell.get("coverage_score") is not None
    ]

    return {
        "name": "unused_precision_heldout",
        "rows": rows,
        "errors": errors,
        "confusion": confusion,
        "unused_precision": unused_p,
        "unused_predicted_n": unused_pred,
        "unused_tp": unused_tp,
        "unused_recall": _recall("unused"),
        "used_precision": used_p,
        "used_recall": _recall("used"),
        "uncertain_precision": uncertain_p,
        "uncertain_recall": _recall("uncertain"),
        "unused_gold_coverage_p50": _percentile(unused_gold_coverage, 0.5),
        "unused_gold_coverage_p95": _percentile(unused_gold_coverage, 0.95),
        "unused_gold_coverage_mean": _mean(unused_gold_coverage),
        # Hard precision gate: unused precision >= 0.85 and actually emitted.
        "passed": (
            len(errors) == 0
            and unused_p is not None
            and unused_p >= 0.85
            and unused_pred > 0
        ),
    }


# ---------------------------------------------------------------------------
# Dimension 3 — Code-lane phantom + dead-weight.
# ---------------------------------------------------------------------------


async def _dim_code(
    client: httpx.AsyncClient,
    transport: Transport,
    cases: Sequence[Any],
    concurrency: int,
) -> Dict[str, Any]:
    sem = asyncio.Semaphore(concurrency)

    async def _one(case: Any) -> Dict[str, Any]:
        raw_context, _rendered = render_context_files(case.context_files)
        payload = {
            "input": {
                "scoring_mode": "code",
                "session_id": f"360bench:{case.id}",
                "response_language_hint": _language_hint(case.context_files),
                "query": case.query,
                "context": raw_context,
                "response": case.response,
                "evidence_limit": 6,
                "emit_chunk_ownership": False,
                "include_triangular_diagnostics": True,
            }
        }
        body = await transport.submit(client, payload)
        return {"case": case, "body": body, "raw_context": raw_context}

    results = await asyncio.gather(
        *[_with_sem(sem, lambda c=c: _one(c)) for c in cases]
    )

    rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    phantom_pos_neg: List[Tuple[float, int]] = []  # score, label (1=ungrounded)
    ast_precision_hits = 0
    ast_precision_total = 0
    dead_weight_grounded: List[float] = []
    dead_weight_ungrounded: List[float] = []
    server_latencies_ms: List[float] = []
    cascade_fires = 0

    for r in results:
        case = r["case"]
        body = r["body"]
        status = body.get("status", "")
        output = body.get("output") or {}
        if status != "COMPLETED" or not output.get("success"):
            errors.append(
                {
                    "id": case.id,
                    "label": case.label,
                    "status": status,
                    "error": output.get("error") or body.get("error"),
                }
            )
            continue

        code_lane = output.get("code_lane") or {}
        phantom_prob = code_lane.get("phantom_probability")
        composite_score = code_lane.get("composite_score")
        phantom_verdict = code_lane.get("phantom_verdict")
        ast_phantom_verdict = code_lane.get("ast_phantom_verdict")
        dead_weight = code_lane.get("dead_weight_ratio")
        cascade_triggered = bool(code_lane.get("nli_cascade_triggered"))
        if cascade_triggered:
            cascade_fires += 1
        srv = float(output.get("latency_ms", float("nan")))
        if not math.isnan(srv):
            server_latencies_ms.append(srv)

        # AUROC: high phantom_probability should flag ungrounded.
        if phantom_prob is not None:
            if case.label == "ungrounded":
                phantom_pos_neg.append((float(phantom_prob), 1))
            elif case.label == "grounded":
                phantom_pos_neg.append((float(phantom_prob), 0))

        # AST phantom is a deterministic precision signal (should only fire on
        # ungrounded).
        if ast_phantom_verdict is True:
            ast_precision_total += 1
            if case.label == "ungrounded":
                ast_precision_hits += 1

        if dead_weight is not None:
            if case.label == "grounded":
                dead_weight_grounded.append(float(dead_weight))
            elif case.label == "ungrounded":
                dead_weight_ungrounded.append(float(dead_weight))

        rows.append(
            {
                "id": case.id,
                "label": case.label,
                "subcategory": case.subcategory,
                "composite_score": composite_score,
                "phantom_probability": phantom_prob,
                "phantom_verdict": phantom_verdict,
                "ast_phantom_symbol_count": code_lane.get("ast_phantom_symbol_count"),
                "ast_literal_drift_count": code_lane.get("ast_literal_drift_count"),
                "ast_phantom_verdict": ast_phantom_verdict,
                "nli_cascade_triggered": cascade_triggered,
                "nli_contradiction_prob_max": code_lane.get(
                    "nli_contradiction_prob_max"
                ),
                "literal_novelty_min": code_lane.get("literal_novelty_min"),
                "dead_weight_ratio": dead_weight,
                "dead_weight_file_count": code_lane.get("dead_weight_file_count"),
                "server_latency_ms": srv,
                "wall_ms": body.get("_wall_ms"),
            }
        )

    auroc = _auroc(
        [s for s, _ in phantom_pos_neg], [lbl for _, lbl in phantom_pos_neg]
    )

    # Composite-score AUROC, where higher composite means *more grounded*.
    # Useful as a cross-check against phantom_probability (which is the
    # inverted reading the dashboard surfaces).
    composite_pos_neg: List[Tuple[float, int]] = []
    for row in rows:
        cs = row.get("composite_score")
        if cs is None:
            continue
        if row.get("label") == "grounded":
            composite_pos_neg.append((float(cs), 1))
        elif row.get("label") == "ungrounded":
            composite_pos_neg.append((float(cs), 0))
    auroc_composite = _auroc(
        [s for s, _ in composite_pos_neg],
        [lbl for _, lbl in composite_pos_neg],
    )

    ast_precision = (
        ast_precision_hits / ast_precision_total if ast_precision_total > 0 else None
    )

    # Verdict precision: when the service says phantom_verdict=True, how often
    # is the case actually ungrounded?
    verdict_fired = [r for r in rows if r.get("phantom_verdict") is True]
    verdict_tp = sum(1 for r in verdict_fired if r.get("label") == "ungrounded")
    verdict_precision = (
        verdict_tp / len(verdict_fired) if verdict_fired else None
    )

    return {
        "name": "code_phantom",
        "rows": rows,
        "errors": errors,
        "auroc_phantom": auroc,
        "auroc_composite_grounded_vs_ungrounded": auroc_composite,
        "ast_phantom_precision": ast_precision,
        "ast_phantom_fired_n": ast_precision_total,
        "phantom_verdict_precision": verdict_precision,
        "phantom_verdict_fired_n": len(verdict_fired),
        "cascade_fires": cascade_fires,
        "mean_dead_weight_grounded": _mean(dead_weight_grounded),
        "mean_dead_weight_ungrounded": _mean(dead_weight_ungrounded),
        "server_p50_ms": _percentile(server_latencies_ms, 0.5),
        "server_p95_ms": _percentile(server_latencies_ms, 0.95),
        "server_p99_ms": _percentile(server_latencies_ms, 0.99),
        # Gate: AUROC phantom >= 0.75, AST precision >= 0.9 when it fires,
        # no hard errors.
        "passed": (
            len(errors) == 0
            and math.isnan(auroc) is False
            and auroc >= 0.75
            and (ast_precision is None or ast_precision >= 0.9)
        ),
    }


def _language_hint(context_files: Sequence[Any]) -> str:
    if not context_files:
        return "python"
    path = getattr(context_files[0], "path", "") or ""
    if path.endswith(".py"):
        return "python"
    if path.endswith(".ts") or path.endswith(".tsx"):
        return "typescript"
    if path.endswith(".js") or path.endswith(".jsx"):
        return "javascript"
    if path.endswith(".go"):
        return "go"
    if path.endswith(".rs"):
        return "rust"
    return "python"


# ---------------------------------------------------------------------------
# Dimension 4 — Session-state round-trip.
# ---------------------------------------------------------------------------


async def _dim_session(
    client: httpx.AsyncClient,
    transport: Transport,
    turns: int,
) -> Dict[str, Any]:
    # Take the first grounded, one ambiguous, one ungrounded, one grounded,
    # one ungrounded code case for a 5-turn story. This lets drift_z_score
    # and ema_groundedness actually move.
    by_label: Dict[str, List[Any]] = {}
    for c in CODE_CASES:
        by_label.setdefault(c.label, []).append(c)
    scripted = [
        by_label.get("grounded", [CODE_CASES[0]])[0],
        by_label.get("ambiguous", [CODE_CASES[0]])[0],
        by_label.get("ungrounded", [CODE_CASES[0]])[0],
        by_label.get("grounded", [CODE_CASES[0]])[-1],
        by_label.get("ungrounded", [CODE_CASES[0]])[-1],
    ][:turns]
    if len(scripted) < turns:
        scripted = list(scripted) + list(CODE_CASES[: turns - len(scripted)])

    session_state: Optional[Dict[str, Any]] = None
    session_id = "360bench:session"
    turn_records: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for idx, case in enumerate(scripted):
        raw_context, _ = render_context_files(case.context_files)
        payload_input: Dict[str, Any] = {
            "scoring_mode": "code",
            "session_id": session_id,
            "response_language_hint": _language_hint(case.context_files),
            "query": case.query,
            "context": raw_context,
            "response": case.response,
            "evidence_limit": 6,
            "include_triangular_diagnostics": True,
        }
        if session_state is not None:
            payload_input["session_state"] = session_state
        body = await transport.submit(client, {"input": payload_input})
        output = body.get("output") or {}
        status = body.get("status", "")
        if status != "COMPLETED" or not output.get("success"):
            errors.append(
                {
                    "turn": idx + 1,
                    "case_id": case.id,
                    "status": status,
                    "error": output.get("error") or body.get("error"),
                }
            )
            break
        next_state = output.get("next_session_state")
        signals = output.get("session_signals") or {}
        turn_records.append(
            {
                "turn": idx + 1,
                "case_id": case.id,
                "label": case.label,
                "total_turns": signals.get("total_turns"),
                "drift_z_score": signals.get("drift_z_score"),
                "ema_groundedness": signals.get("ema_groundedness"),
                "cascade_density": signals.get("cascade_density"),
                "phantom_rate": signals.get("phantom_rate"),
                "dead_file_candidates": signals.get("dead_file_candidates"),
                "recommendation": signals.get("recommendation"),
                "session_state_size_bytes": (
                    len(json.dumps(next_state)) if next_state is not None else None
                ),
            }
        )
        session_state = next_state

    # Sanity checks on the round-trip.
    monotonic_turns = all(
        (r["total_turns"] or 0) == (i + 1) for i, r in enumerate(turn_records)
    )
    produced_signals = all(
        r["drift_z_score"] is not None or r["ema_groundedness"] is not None
        for r in turn_records
    )
    produced_next_state = all(
        r["session_state_size_bytes"] is not None for r in turn_records
    )

    return {
        "name": "session_roundtrip",
        "turn_records": turn_records,
        "errors": errors,
        "monotonic_turns": monotonic_turns,
        "produced_signals": produced_signals,
        "produced_next_state": produced_next_state,
        "final_state_bytes": (
            len(json.dumps(session_state)) if session_state is not None else None
        ),
        "passed": (
            len(errors) == 0
            and len(turn_records) > 0
            and monotonic_turns
            and produced_signals
            and produced_next_state
        ),
    }


# ---------------------------------------------------------------------------
# Dimension 5 — Concurrency burst (per-lane non-starvation).
# ---------------------------------------------------------------------------


async def _dim_burst(
    client: httpx.AsyncClient,
    transport: Transport,
    rag_cases: Sequence[Any],
    code_cases: Sequence[Any],
    burst_size: int,
) -> Dict[str, Any]:
    half = max(1, burst_size // 2)
    rag_subset = list(rag_cases)[:half]
    code_subset = list(code_cases)[:half]

    async def _rag_one(case: Any) -> Dict[str, Any]:
        payload = {
            "input": {
                "query": case.query,
                "context": case.context,
                "response": case.response,
                "evidence_limit": 4,
            }
        }
        start = time.perf_counter()
        body = await transport.submit(client, payload)
        return {
            "lane": "rag",
            "id": case.id,
            "status": body.get("status"),
            "success": (body.get("output") or {}).get("success"),
            "server_latency_ms": float(
                (body.get("output") or {}).get("latency_ms", float("nan"))
            ),
            "wall_ms": (time.perf_counter() - start) * 1000.0,
        }

    async def _code_one(case: Any) -> Dict[str, Any]:
        raw_context, _ = render_context_files(case.context_files)
        payload = {
            "input": {
                "scoring_mode": "code",
                "session_id": f"360bench:burst:{case.id}",
                "response_language_hint": _language_hint(case.context_files),
                "query": case.query,
                "context": raw_context,
                "response": case.response,
                "evidence_limit": 6,
            }
        }
        start = time.perf_counter()
        body = await transport.submit(client, payload)
        return {
            "lane": "code",
            "id": case.id,
            "status": body.get("status"),
            "success": (body.get("output") or {}).get("success"),
            "server_latency_ms": float(
                (body.get("output") or {}).get("latency_ms", float("nan"))
            ),
            "wall_ms": (time.perf_counter() - start) * 1000.0,
        }

    coros: List[Awaitable[Dict[str, Any]]] = []
    for c in rag_subset:
        coros.append(_rag_one(c))
    for c in code_subset:
        coros.append(_code_one(c))

    wall_start = time.perf_counter()
    results = await asyncio.gather(*coros, return_exceptions=True)
    total_wall_ms = (time.perf_counter() - wall_start) * 1000.0

    good: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    for r in results:
        if isinstance(r, Exception):
            failures.append({"error": repr(r)})
        elif not r.get("success"):
            failures.append(r)
        else:
            good.append(r)

    rag_latencies = [r["server_latency_ms"] for r in good if r["lane"] == "rag"]
    code_latencies = [r["server_latency_ms"] for r in good if r["lane"] == "code"]

    return {
        "name": "concurrency_burst",
        "sent": len(results),
        "ok": len(good),
        "failed": len(failures),
        "failures": failures[:10],
        "rag_p50_ms": _percentile(rag_latencies, 0.5),
        "rag_p95_ms": _percentile(rag_latencies, 0.95),
        "code_p50_ms": _percentile(code_latencies, 0.5),
        "code_p95_ms": _percentile(code_latencies, 0.95),
        "total_wall_ms": total_wall_ms,
        "passed": len(failures) == 0 and len(good) == len(results),
    }


# ---------------------------------------------------------------------------
# Report.
# ---------------------------------------------------------------------------


def _hr(width: int = 120) -> str:
    return "=" * width


def _print_report(dims: Dict[str, Dict[str, Any]]) -> int:
    print()
    print(_hr())
    print("latence-trace :: RunPod 360-degree live benchmark")
    print(_hr())

    overall_pass = True

    if "rag" in dims:
        d = dims["rag"]
        status = "PASS" if d.get("passed") else "FAIL"
        overall_pass = overall_pass and bool(d.get("passed"))
        print()
        print(f"[1/5] RAG handcrafted                                             {status}")
        print("-" * 80)
        print(
            f"  cases                  : {len(d.get('rows', []))} completed / "
            f"{len(d.get('errors', []))} errors"
        )
        print(
            f"  AUROC grounded vs ungrounded : {_fmt_f(d.get('auroc_grounded_vs_ungrounded'))}"
        )
        print(
            f"  mean score grounded / ungrounded / ambiguous : "
            f"{_fmt_f(d.get('mean_score_grounded'))} / "
            f"{_fmt_f(d.get('mean_score_ungrounded'))} / "
            f"{_fmt_f(d.get('mean_score_ambiguous'))}"
        )
        print(
            f"  server latency p50 / p95 / p99 ms : "
            f"{_fmt_f(d.get('server_p50_ms'), 7)} / "
            f"{_fmt_f(d.get('server_p95_ms'), 7)} / "
            f"{_fmt_f(d.get('server_p99_ms'), 7)}"
        )
        for e in d.get("errors", []):
            print(f"  ERR {e.get('id')} {e.get('label')}: {e.get('status')} {e.get('error')}")

    if "unused" in dims:
        d = dims["unused"]
        status = "PASS" if d.get("passed") else "FAIL"
        overall_pass = overall_pass and bool(d.get("passed"))
        print()
        print(f"[2/5] Unused-context tri-state precision (held-out)               {status}")
        print("-" * 80)
        conf = d.get("confusion", {})
        header = "      " + "".join(f"{p:>10}" for p in ("used", "unused", "uncertain"))
        print("  confusion matrix (gold x predicted):")
        print(header)
        for g in ("used", "unused", "uncertain"):
            row = conf.get(g, {})
            cells = "".join(f"{row.get(p, 0):>10d}" for p in ("used", "unused", "uncertain"))
            print(f"  {g:<6}{cells}")
        print(
            f"  unused precision / recall : {_fmt_f(d.get('unused_precision'))} / "
            f"{_fmt_f(d.get('unused_recall'))}  "
            f"(fired n={d.get('unused_predicted_n', 0)})"
        )
        print(
            f"  used   precision / recall : {_fmt_f(d.get('used_precision'))} / "
            f"{_fmt_f(d.get('used_recall'))}"
        )
        print(
            f"  uncertain p / r           : {_fmt_f(d.get('uncertain_precision'))} / "
            f"{_fmt_f(d.get('uncertain_recall'))}"
        )
        print(
            f"  coverage_score on gold-unused units (p50/p95/mean): "
            f"{_fmt_f(d.get('unused_gold_coverage_p50'), 6)} / "
            f"{_fmt_f(d.get('unused_gold_coverage_p95'), 6)} / "
            f"{_fmt_f(d.get('unused_gold_coverage_mean'), 6)}"
        )
        print(
            "  note: if the coverage_score p50 on gold-unused units is >= 0.5, the"
        )
        print(
            "        live service's usage classifier has no room to emit 'unused';"
        )
        print(
            "        this is a deployment-side threshold calibration finding, not"
        )
        print(
            "        a bug in the benchmark."
        )
        for e in d.get("errors", []):
            print(f"  ERR {e.get('id')}: {e.get('status')} {e.get('error')}")

    if "code" in dims:
        d = dims["code"]
        status = "PASS" if d.get("passed") else "FAIL"
        overall_pass = overall_pass and bool(d.get("passed"))
        print()
        print(f"[3/5] Code-lane phantom + dead-weight                             {status}")
        print("-" * 80)
        print(f"  cases                  : {len(d.get('rows', []))} completed / {len(d.get('errors', []))} errors")
        print(f"  AUROC phantom grounded vs ungrounded : {_fmt_f(d.get('auroc_phantom'))}")
        print(
            f"  AUROC composite (inverted cross-check) : "
            f"{_fmt_f(d.get('auroc_composite_grounded_vs_ungrounded'))}"
        )
        print(
            f"  AST phantom precision (fired n={d.get('ast_phantom_fired_n', 0):>2}) : "
            f"{_fmt_f(d.get('ast_phantom_precision'))}"
        )
        print(
            f"  phantom_verdict precision (fired n={d.get('phantom_verdict_fired_n', 0):>2}) : "
            f"{_fmt_f(d.get('phantom_verdict_precision'))}"
        )
        print(f"  NLI cascade fires      : {d.get('cascade_fires', 0)}")
        print(
            f"  mean dead_weight ratio grounded / ungrounded : "
            f"{_fmt_f(d.get('mean_dead_weight_grounded'))} / "
            f"{_fmt_f(d.get('mean_dead_weight_ungrounded'))}"
        )
        print(
            f"  server latency p50 / p95 / p99 ms : "
            f"{_fmt_f(d.get('server_p50_ms'), 7)} / "
            f"{_fmt_f(d.get('server_p95_ms'), 7)} / "
            f"{_fmt_f(d.get('server_p99_ms'), 7)}"
        )
        for e in d.get("errors", []):
            print(f"  ERR {e.get('id')} {e.get('label')}: {e.get('status')} {e.get('error')}")
        # Per-case row dump: 1 line each.
        print()
        print(
            f"  {'id':<5} {'label':<11} {'sub':<14} "
            f"{'composite':>10} {'p_phantom':>10} {'verdict':>7} "
            f"{'ast_ph':>7} {'cascade':>8} {'dead_w':>8}"
        )
        for row in sorted(d.get("rows", []), key=lambda r: (r["label"], r["id"])):
            sub = row.get("subcategory") or "-"
            print(
                f"  {row['id']:<5} {row['label']:<11} {sub:<14} "
                f"{_fmt_f(row.get('composite_score'), 10)} "
                f"{_fmt_f(row.get('phantom_probability'), 10)} "
                f"{str(row.get('phantom_verdict')):>7} "
                f"{str(row.get('ast_phantom_verdict')):>7} "
                f"{str(row.get('nli_cascade_triggered')):>8} "
                f"{_fmt_f(row.get('dead_weight_ratio'), 8)}"
            )

    if "session" in dims:
        d = dims["session"]
        status = "PASS" if d.get("passed") else "FAIL"
        overall_pass = overall_pass and bool(d.get("passed"))
        print()
        print(f"[4/5] Session-state round-trip                                    {status}")
        print("-" * 80)
        print(
            f"  turns completed        : {len(d.get('turn_records', []))} / "
            f"{len(d.get('errors', []))} errors"
        )
        print(
            f"  monotonic total_turns  : {d.get('monotonic_turns')}    "
            f"signals emitted : {d.get('produced_signals')}    "
            f"next_state emitted : {d.get('produced_next_state')}"
        )
        print(f"  final state size (bytes): {d.get('final_state_bytes')}")
        print(
            f"  {'t':<2} {'case':<6} {'label':<11} {'turns':>6} {'drift_z':>8} "
            f"{'ema_g':>7} {'rec':<14} {'blob_b':>7}"
        )
        for r in d.get("turn_records", []):
            print(
                f"  {r['turn']:<2} {r['case_id']:<6} {r['label']:<11} "
                f"{_fmt_i(r.get('total_turns'), 6)} "
                f"{_fmt_f(r.get('drift_z_score'), 8)} "
                f"{_fmt_f(r.get('ema_groundedness'), 7)} "
                f"{str(r.get('recommendation')):<14} "
                f"{_fmt_i(r.get('session_state_size_bytes'), 7)}"
            )
        for e in d.get("errors", []):
            print(f"  ERR turn={e['turn']} {e['case_id']}: {e['status']} {e['error']}")

    if "burst" in dims:
        d = dims["burst"]
        status = "PASS" if d.get("passed") else "FAIL"
        overall_pass = overall_pass and bool(d.get("passed"))
        print()
        print(f"[5/5] Concurrency burst (per-lane non-starvation)                 {status}")
        print("-" * 80)
        print(f"  sent / ok / failed     : {d.get('sent')} / {d.get('ok')} / {d.get('failed')}")
        print(
            f"  RAG  p50 / p95 ms      : {_fmt_f(d.get('rag_p50_ms'), 7)} / "
            f"{_fmt_f(d.get('rag_p95_ms'), 7)}"
        )
        print(
            f"  code p50 / p95 ms      : {_fmt_f(d.get('code_p50_ms'), 7)} / "
            f"{_fmt_f(d.get('code_p95_ms'), 7)}"
        )
        print(f"  total wall-clock ms    : {_fmt_f(d.get('total_wall_ms'), 7)}")
        for f in d.get("failures", []):
            print(f"  FAIL {f}")

    print()
    print(_hr())
    print(f"OVERALL                                                            "
          f"{'PASS' if overall_pass else 'FAIL'}")
    print(_hr())
    return 0 if overall_pass else 1


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--endpoint-id", default=os.environ.get("RUNPOD_ENDPOINT_ID"))
    p.add_argument("--api-key", default=os.environ.get("RUNPOD_API_KEY"))
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--timeout-s", type=float, default=600.0)
    p.add_argument(
        "--skip",
        default="",
        help="comma-separated dimensions to skip: rag,unused,code,session,concurrency",
    )
    p.add_argument(
        "--cases-rag",
        default=None,
        help="comma-separated RAG case IDs (default: all 20)",
    )
    p.add_argument(
        "--cases-code",
        default=None,
        help="comma-separated code case IDs (default: all 12)",
    )
    p.add_argument("--session-turns", type=int, default=5)
    p.add_argument("--burst-size", type=int, default=16)
    p.add_argument("--use-run-poll", action="store_true",
                   help="use /run + /status polling instead of /runsync")
    p.add_argument("--dump", default=None, help="path to write a detailed JSON report")
    return p.parse_args()


async def _async_main(args: argparse.Namespace) -> int:
    if not args.endpoint_id or not args.api_key:
        print(
            "error: --endpoint-id and --api-key are required "
            "(or set RUNPOD_ENDPOINT_ID / RUNPOD_API_KEY)",
            file=sys.stderr,
        )
        return 2

    transport = Transport(
        endpoint_id=args.endpoint_id,
        api_key=args.api_key,
        use_runsync=not args.use_run_poll,
        timeout_s=args.timeout_s,
    )

    # RAG subset.
    rag_bank = {c.id: c for c in RAG_CASES}
    if args.cases_rag:
        ids = [cid.strip() for cid in args.cases_rag.split(",") if cid.strip()]
        selected_rag = [rag_bank[cid] for cid in ids if cid in rag_bank]
    else:
        selected_rag = list(RAG_CASES)

    # Code subset.
    code_bank = {c.id: c for c in CODE_CASES}
    if args.cases_code:
        ids = [cid.strip() for cid in args.cases_code.split(",") if cid.strip()]
        selected_code = [code_bank[cid] for cid in ids if cid in code_bank]
    else:
        selected_code = list(CODE_CASES)

    skip = {s.strip() for s in args.skip.split(",") if s.strip()}

    limits = httpx.Limits(
        max_connections=max(args.concurrency * 2, args.burst_size * 2),
        max_keepalive_connections=max(args.concurrency * 2, args.burst_size * 2),
    )
    timeout = httpx.Timeout(args.timeout_s, connect=15.0)

    dims: Dict[str, Dict[str, Any]] = {}
    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        if "rag" not in skip:
            print(
                f"[1/5] running RAG handcrafted "
                f"({len(selected_rag)} cases, concurrency={args.concurrency}) ...",
                flush=True,
            )
            dims["rag"] = await _dim_rag(
                client, transport, selected_rag, args.concurrency
            )
        if "unused" not in skip:
            print("[2/5] running unused-context precision (held-out) ...", flush=True)
            dims["unused"] = await _dim_unused(client, transport, args.concurrency)
        if "code" not in skip:
            print(
                f"[3/5] running code-lane phantom "
                f"({len(selected_code)} cases, concurrency={args.concurrency}) ...",
                flush=True,
            )
            dims["code"] = await _dim_code(
                client, transport, selected_code, args.concurrency
            )
        if "session" not in skip:
            print(
                f"[4/5] running session-state round-trip ({args.session_turns} turns) ...",
                flush=True,
            )
            dims["session"] = await _dim_session(
                client, transport, args.session_turns
            )
        if "concurrency" not in skip:
            print(
                f"[5/5] running concurrency burst (size={args.burst_size}) ...",
                flush=True,
            )
            dims["burst"] = await _dim_burst(
                client,
                transport,
                selected_rag,
                selected_code,
                args.burst_size,
            )

    rc = _print_report(dims)

    if args.dump:
        path = Path(args.dump)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dims, indent=2, sort_keys=True, default=str))
        print(f"\n  wrote detailed report -> {path}")

    return rc


def main() -> int:
    return asyncio.run(_async_main(_parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
