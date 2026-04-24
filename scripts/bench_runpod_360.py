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

    6. RAG per-file attribution + reason-code histogram: multi-file
       support-unit fixtures that prove the shared attribution kernel
       fires on the RAG lane, dead_weight_ratio is populated, and the
       reason-code histogram is emitted by default.

    7. Heatmap: one RAG + one code case re-run with
       ``heatmap_format="html"`` to validate both the structured data
       payload (tokens / files / summary / thresholds) and the
       self-contained HTML fragment (``<div>``, band CSS variables,
       no external refs).

    8. Stateless rollup: POSTs ``action="rollup"`` with the 5 session
       turns from dimension 4 and validates noise_pct / model_drift_pct
       / retrieval_waste_pct are in [0, 1], the reason-code histogram
       sums across turns, the risk-band trail is length-preserving, and
       (when requested) the session-level heatmap HTML is emitted.

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

    --skip rag|unused|code|session|concurrency|rag_attribution|heatmap|rollup
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


def _strip_diff_to_code(response: str) -> str:
    """Turn a unified-diff response into the target code the agent would emit.

    The hand-crafted coding case bank ships ``response`` as a unified diff
    (``diff --git ... @@ -old +new``). The live code lane's AST / literal
    detectors parse ``response`` as source code via tree-sitter, so a diff
    header makes every identifier look phantom-free (there's no valid AST).

    This helper keeps just the ``+``-prefixed lines, strips the marker, and
    re-joins them into a code snippet — the post-diff view an IDE plugin
    would actually submit. Non-diff responses are returned untouched.
    """

    if "diff --git" not in response and "@@" not in response and not any(
        line.startswith("+") for line in response.splitlines()
    ):
        return response
    kept: List[str] = []
    for line in response.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("@@") or line.startswith("diff --git"):
            continue
        if line.startswith("+"):
            kept.append(line[1:])
        elif line.startswith("-"):
            # Drop the pre-edit line; the agent's final response is post-edit.
            continue
        else:
            # Context line — preserve so the AST is still parseable.
            kept.append(line.lstrip())
    return "\n".join(kept).strip() or response


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


# Gateway-transport key translation. The RunPod handler accepts the alias
# keys ``query`` / ``context`` / ``response`` for backwards compatibility;
# the gateway surface uses the canonical pydantic names. Rewrite the input
# dict so the same bench driver can target either backend without forking
# every dimension.
_GATEWAY_KEY_REWRITES = {
    "query": "query_text",
    "context": "raw_context",
    "response": "response_text",
}


@dataclass
class Transport:
    endpoint_id: str
    api_key: str
    use_runsync: bool = True
    timeout_s: float = 600.0
    # Optional gateway-transport config. When ``gateway_url`` is set the
    # transport posts to the gateway's DX endpoints instead of RunPod and
    # re-wraps the response in the ``{"status": "COMPLETED", "output": ...}``
    # shape downstream analysis expects.
    gateway_url: Optional[str] = None
    gateway_key: Optional[str] = None

    def headers(self) -> Dict[str, str]:
        if self.gateway_url:
            return {
                "Authorization": f"Bearer {self.gateway_key or ''}",
                "Content-Type": "application/json",
            }
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

        if self.gateway_url:
            body = await self._submit_gateway(client, payload)
            wall_ms = (time.perf_counter() - wall_start) * 1000.0
            body["_wall_ms"] = wall_ms
            return body

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

    async def _submit_gateway(
        self,
        client: httpx.AsyncClient,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Send a bench payload through the local gateway.

        The gateway exposes ``/api/v1/trace/rag`` and ``/api/v1/trace/code``
        with a flat (non ``input``-wrapped) body and the canonical pydantic
        field names. Translate both directions so downstream analysis keeps
        reading ``body["output"]`` untouched.
        """
        flat = dict(payload.get("input") or {})
        scoring_mode = (flat.pop("scoring_mode", None) or "rag").lower()
        # Gateway endpoint is the source of truth for the lane; strip any
        # stray ``scoring_mode`` from the body (the schema rejects it).
        endpoint = "/api/v1/trace/code" if scoring_mode == "code" else "/api/v1/trace/rag"

        translated: Dict[str, Any] = {}
        for key, value in flat.items():
            translated[_GATEWAY_KEY_REWRITES.get(key, key)] = value

        url = f"{self.gateway_url.rstrip('/')}{endpoint}"
        resp = await client.post(url, json=translated, headers=self.headers())
        # Surface gateway errors so the bench fails loudly rather than
        # silently falling into empty-output analysis branches.
        if resp.status_code >= 400:
            detail: Any
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text
            raise RuntimeError(
                f"gateway {endpoint} returned HTTP {resp.status_code}: {detail}"
            )
        output = resp.json()
        # Rewrap to the RunPod job shape the rest of the bench consumes.
        return {
            "status": "COMPLETED",
            "output": output,
            "id": "local-gateway",
        }

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

    # Distractors must be long enough and topically distant enough that the
    # real ColBERT / MaxSim encoder does not incidentally score them high on
    # the anchor query. One-liner distractors consistently ride above the
    # 0.85 coverage threshold on short text; multi-sentence passages do not.
    topics = [
        {
            "name": "saturn",
            "query": "What are Saturn's rings made of?",
            "anchor": (
                "Saturn has rings made of ice. The main rings span roughly 282,000 km "
                "and are dominated by water-ice particles ranging from micrometres to "
                "several metres across, with only trace amounts of rocky debris."
            ),
            "paraphrase": (
                "Saturn's rings are composed almost entirely of frozen water. "
                "Observations by Cassini showed that ice particles make up the vast "
                "majority of the ring material across its main bands."
            ),
            "mixed_noise": (
                "The observatory brochure also lists parking rules, telescope rental "
                "hours, admission pricing for the planetarium dome, and the cafe menu, "
                "along with gift-shop closing times that the answer never references."
            ),
            "distractor": (
                "Bamboo is a group of woody, perennial flowering plants in the "
                "subfamily Bambusoideae. Some species grow up to ninety centimetres "
                "per day under ideal tropical humidity, making bamboo one of the "
                "fastest-growing plants on Earth and a favourite of landscape "
                "architects working on erosion-prone hillsides."
            ),
        },
        {
            "name": "curie",
            "query": "When did Marie Curie win her first Nobel Prize?",
            "anchor": (
                "Marie Curie won the Nobel Prize in Physics in 1903. She shared the "
                "award with her husband Pierre Curie and Henri Becquerel for their "
                "combined work on the phenomenon of spontaneous radiation discovered "
                "in uranium and thorium compounds."
            ),
            "paraphrase": (
                "Marie Curie received the 1903 Nobel Prize in Physics jointly with "
                "Pierre Curie and Henri Becquerel for research into radioactive "
                "emissions from heavy elements."
            ),
            "mixed_noise": (
                "The archive also catalogs lecture schedules from 1898 onward, train "
                "routes between Warsaw and Paris, visitor sign-in books, and the "
                "museum gift-shop inventory that the answer does not reference."
            ),
            "distractor": (
                "Coral reefs are underwater ecosystems held together by calcium "
                "carbonate secreted by colonies of polyps. A single reef can host "
                "thousands of species of fish, molluscs, and crustaceans, and the "
                "Great Barrier Reef off Australia is visible from low Earth orbit."
            ),
        },
        {
            "name": "meeting",
            "query": "What time does the planning meeting start tomorrow?",
            "anchor": (
                "The planning meeting starts at 9 AM local time tomorrow in the "
                "fourth-floor boardroom. Coffee is available from 8:45, and the "
                "agenda has been circulated to every department lead the night before."
            ),
            "paraphrase": (
                "Tomorrow's planning meeting begins at nine o'clock in the morning "
                "on the fourth floor. Refreshments are served fifteen minutes before "
                "the start of the agenda."
            ),
            "mixed_noise": (
                "Agenda notes also cover lunch catering options for the entire week, "
                "hallway signage updates, seat-assignment spreadsheets, and HVAC "
                "maintenance windows that the answer never mentions."
            ),
            "distractor": (
                "Maple syrup is produced by tapping sugar-maple trees in late winter "
                "and boiling the sap down to about one-fortieth of its original "
                "volume. Quebec produces the majority of the world's commercial "
                "maple syrup, with grading based on colour and flavour intensity."
            ),
        },
        {
            "name": "ev",
            "query": "How are electric cars powered and how far can they typically drive on one charge?",
            "anchor": (
                "Electric cars are powered by large lithium-ion battery packs that "
                "store energy and drive one or more electric motors. Modern consumer "
                "EVs typically deliver between 300 and 500 kilometres of range on a "
                "full charge, depending on battery size, driving style, and climate."
            ),
            "paraphrase": (
                "Electric vehicles run on rechargeable lithium-ion battery packs, "
                "and most mainstream models cover roughly 300 to 500 kilometres "
                "before they need to be plugged back in."
            ),
            "mixed_noise": (
                "The brochure also describes paint options, showroom lighting "
                "packages, coffee-bar coupons for the sales lounge, and a loyalty "
                "programme for second-time buyers that the answer does not mention."
            ),
            "distractor": (
                "Saffron is a spice derived from the stigmas of the Crocus sativus "
                "flower. Each flower produces only three stigmas, which must be "
                "harvested by hand, so a kilogram of saffron requires roughly a "
                "hundred and fifty thousand flowers and commands a very high price."
            ),
        },
    ]
    generic_noise = (
        "Lanterns glow above the quiet harbor at dusk while fishermen mend their "
        "nets and the tide slowly rises. The keeper of the lighthouse logs the "
        "wind direction and humidity for the nightly report to the coast guard."
    )

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

    # Feature-surface gate: every returned support unit must carry the full
    # tri-state contract (usage_state / usage_confidence / unused_confidence /
    # coverage_score). This is exactly the regression that ``tests/
    # test_runpod_handler.py::test_compact_response_surfaces_unused_context_
    # contract`` locks in at the unit level, replayed end-to-end against the
    # live endpoint.
    contract_fields = ("usage_state", "usage_confidence", "unused_confidence",
                       "coverage_score")
    contract_ok = (
        len(errors) == 0
        and len(rows) > 0
        and all(
            cell.get("pred") != "missing"
            and all(cell.get(f) is not None for f in ("coverage_score",
                                                      "usage_confidence",
                                                      "unused_confidence"))
            for row in rows
            for cell in row["cells"]
        )
    )

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
        "contract_ok": contract_ok,
        # Gate: feature is wired (every unit carries the full tri-state
        # contract). Whether the real encoder emits enough ``unused`` votes
        # on short passages is a separate, deploy-time calibration finding
        # surfaced in the coverage_score p50/p95 diagnostic.
        "passed": contract_ok,
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
        # The bank ships response as a unified diff; the AST/literal detectors
        # want raw source code. Strip diff framing before submitting.
        response_code = _strip_diff_to_code(case.response)
        payload = {
            "input": {
                "scoring_mode": "code",
                "session_id": f"360bench:{case.id}",
                "response_language_hint": _language_hint(case.context_files),
                "query": case.query,
                "context": raw_context,
                "response": response_code,
                "evidence_limit": 6,
                "emit_chunk_ownership": True,
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
    parser_backends: Dict[str, int] = {}
    # Per-signal presence gates (one of ast/literal_novelty/nli must fire on
    # every ungrounded case; none of them may fire on grounded ones).
    ungrounded_any_signal = 0
    ungrounded_total = 0
    grounded_false_positives = 0
    grounded_total = 0

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
        ast_phantom_symbol_count = code_lane.get("ast_phantom_symbol_count") or 0
        ast_literal_drift_count = code_lane.get("ast_literal_drift_count") or 0
        literal_novelty_missing_count = code_lane.get(
            "literal_novelty_missing_count"
        ) or 0
        nli_contra = code_lane.get("nli_contradiction_prob_max")
        dead_weight = code_lane.get("dead_weight_ratio")
        cascade_triggered = bool(code_lane.get("nli_cascade_triggered"))
        if cascade_triggered:
            cascade_fires += 1
        # Capture parser backend from full diagnostics (helpful to tell
        # tree-sitter vs. regex fallback at a glance).
        diag = (code_lane.get("diagnostics") or {}) if code_lane else {}
        ast_diag = diag.get("ast") or {}
        backend = ast_diag.get("parser_backend") or "unknown"
        parser_backends[backend] = parser_backends.get(backend, 0) + 1
        srv = float(output.get("latency_ms", float("nan")))
        if not math.isnan(srv):
            server_latencies_ms.append(srv)

        # Per-signal gate: every ungrounded case must trip at least one
        # phantom/drift signal. Grounded cases must not trip the hard
        # deterministic gates (AST phantom verdict).
        fires_any = (
            ast_phantom_symbol_count > 0
            or ast_literal_drift_count > 0
            or literal_novelty_missing_count > 0
            or (nli_contra is not None and float(nli_contra) >= 0.70)
        )
        if case.label == "ungrounded":
            ungrounded_total += 1
            if fires_any:
                ungrounded_any_signal += 1
        elif case.label == "grounded":
            grounded_total += 1
            if ast_phantom_verdict is True:
                grounded_false_positives += 1

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
                "ast_phantom_symbol_count": ast_phantom_symbol_count,
                "ast_literal_drift_count": ast_literal_drift_count,
                "ast_phantom_verdict": ast_phantom_verdict,
                "ast_parser_backend": backend,
                "ast_phantom_symbols": ast_diag.get("phantom_symbols"),
                "ast_drift_symbols": ast_diag.get("drift_symbols"),
                "nli_cascade_triggered": cascade_triggered,
                "nli_contradiction_prob_max": nli_contra,
                "literal_novelty_min": code_lane.get("literal_novelty_min"),
                "literal_novelty_missing_count": literal_novelty_missing_count,
                "dead_weight_ratio": dead_weight,
                "dead_weight_file_count": code_lane.get("dead_weight_file_count"),
                "any_signal_fired": fires_any,
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

    ungrounded_recall = (
        ungrounded_any_signal / ungrounded_total if ungrounded_total > 0 else None
    )

    # Feature-surface gate: every code-lane response must carry the full
    # set of contract fields the dashboard depends on.
    contract_fields = (
        "composite_score",
        "phantom_probability",
        "phantom_verdict",
        "ast_phantom_verdict",
        "ast_phantom_symbol_count",
        "ast_literal_drift_count",
        "literal_novelty_min",
        "nli_cascade_triggered",
        "dead_weight_ratio",
    )
    contract_ok = (
        len(errors) == 0
        and len(rows) > 0
        and all(
            all(row.get(f) is not None for f in contract_fields) for row in rows
        )
    )
    # Hard gate: the live endpoint MUST run tree-sitter. Any other
    # backend (regex_fallback, disabled) is a production quality
    # regression — we refuse to mark the run green until it's fixed.
    parser_backend_ok = (
        len(rows) > 0
        and set(parser_backends.keys()) == {"tree_sitter"}
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
        "ungrounded_any_signal_recall": ungrounded_recall,
        "ungrounded_any_signal_fired": ungrounded_any_signal,
        "ungrounded_total": ungrounded_total,
        "grounded_ast_false_positives": grounded_false_positives,
        "grounded_total": grounded_total,
        "parser_backends": parser_backends,
        "parser_backend_ok": parser_backend_ok,
        "cascade_fires": cascade_fires,
        "mean_dead_weight_grounded": _mean(dead_weight_grounded),
        "mean_dead_weight_ungrounded": _mean(dead_weight_ungrounded),
        "server_p50_ms": _percentile(server_latencies_ms, 0.5),
        "server_p95_ms": _percentile(server_latencies_ms, 0.95),
        "server_p99_ms": _percentile(server_latencies_ms, 0.99),
        "contract_ok": contract_ok,
        # Deterministic product gates (composition AUROC is informational —
        # the composite logistic is calibrated on natural-language code
        # responses in transcripts_v2, not on unified-diff fixtures, so we
        # don't gate on it here):
        #   - feature-surface contract OK
        #   - parser backend is tree-sitter on every response (no regex
        #     fallback in production)
        #   - every ungrounded case trips >=1 phantom/drift signal
        #   - grounded cases never trip the deterministic AST phantom verdict
        #   - AST phantom precision is perfect when it does fire
        "passed": (
            contract_ok
            and parser_backend_ok
            and (ungrounded_recall is None or ungrounded_recall >= 0.8)
            and grounded_false_positives == 0
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
# Dimension 3b — RAG raw_context unused-chunks contract.
# ---------------------------------------------------------------------------


async def _dim_unused_raw_context(
    client: httpx.AsyncClient,
    transport: Transport,
) -> Dict[str, Any]:
    """Exercise the tri-state unused-context contract on the ``raw_context``
    lane (the legacy RAG lane that 95% of callers hit). Catches the exact
    regression ``test_compact_response_surfaces_unused_context_contract``
    guards at the unit level, end-to-end on the live service."""

    anchor = (
        "Saturn has rings made of ice, dominated by water-ice particles ranging "
        "from micrometres to several metres across."
    )
    distractor = (
        "\n\nBamboo is a group of woody, perennial flowering plants in the "
        "subfamily Bambusoideae. Some species grow up to ninety centimetres "
        "per day, making bamboo one of the fastest-growing plants on Earth.\n\n"
    )
    query = "What are Saturn's rings made of?"
    response_text = "Saturn's rings are composed of ice."
    payload = {
        "input": {
            "query": query,
            "context": anchor + distractor,
            "response": response_text,
            "evidence_limit": 4,
            "primary_metric": "reverse_context",
            "segmentation_mode": "sentence_packed",
            "raw_context_chunk_tokens": 48,
        }
    }
    body = await transport.submit(client, payload)
    output = body.get("output") or {}
    success = bool(output.get("success"))
    status = body.get("status")
    support = output.get("support_units") or []
    contract_ok = (
        success
        and status == "COMPLETED"
        and len(support) >= 2
        and all(
            "usage_state" in unit
            and unit.get("usage_confidence") is not None
            and unit.get("unused_confidence") is not None
            and unit.get("coverage_score") is not None
            for unit in support
        )
        and output.get("context_unused_ratio") is not None
        and output.get("context_uncertain_ratio") is not None
        and output.get("support_units_usage") is not None
    )
    usage_counts = output.get("support_units_usage") or {}
    return {
        "name": "unused_raw_context_contract",
        "status": status,
        "success": success,
        "n_support_units": len(support),
        "context_coverage_ratio": output.get("context_coverage_ratio"),
        "context_unused_ratio": output.get("context_unused_ratio"),
        "context_uncertain_ratio": output.get("context_uncertain_ratio"),
        "support_units_usage": usage_counts,
        "support_units_sample": support,
        "contract_ok": contract_ok,
        "passed": contract_ok,
    }


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
    # Raw per-turn outputs are captured so ``_dim_rollup`` can consume the
    # same 5-turn narrative without re-running the scorer. This keeps the
    # rollup dimension aligned with what the IDE plugin would send after a
    # live coding session.
    raw_outputs: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for idx, case in enumerate(scripted):
        raw_context, _ = render_context_files(case.context_files)
        payload_input: Dict[str, Any] = {
            "scoring_mode": "code",
            "session_id": session_id,
            "response_language_hint": _language_hint(case.context_files),
            "query": case.query,
            "context": raw_context,
            "response": _strip_diff_to_code(case.response),
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
        raw_outputs.append(output)
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
        "raw_outputs": raw_outputs,
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
                "response": _strip_diff_to_code(case.response),
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
# Dimension 6 — RAG per-file attribution + reason-code histogram.
# ---------------------------------------------------------------------------


def _rag_attribution_cases() -> List[Dict[str, Any]]:
    """Multi-file RAG fixtures that guarantee ≥ 2 distinct attribution keys.

    The 20 handcrafted RAG cases ship as a single ``context`` blob which the
    chunker may split into one or more support units keyed by the synthetic
    ``support_id``. To exercise the shared file-attribution kernel end-to-end
    — including the ``metadata.path`` grouping and per-file reason codes —
    we post ``support_units`` directly with explicit ``metadata.path`` on
    each unit. One unit is the true anchor, the second is a plausible
    distractor, the third is deep noise.
    """

    return [
        {
            "id": "rag_attr_saturn",
            "query_text": "What are Saturn's rings primarily composed of?",
            "response_text": (
                "Saturn's rings are made mostly of water ice, with particles "
                "ranging from micrometres to several metres in size."
            ),
            "support_units": [
                {
                    "support_id": "s-anchor",
                    "text": (
                        "Saturn's main rings are dominated by water-ice "
                        "particles, from micrometres to several metres across, "
                        "with only trace amounts of rocky debris."
                    ),
                    "metadata": {"path": "docs/saturn_rings.md"},
                },
                {
                    "support_id": "s-near",
                    "text": (
                        "Jupiter has faint rings that were discovered by "
                        "Voyager 1 in 1979 and consist mostly of fine dust."
                    ),
                    "metadata": {"path": "docs/jupiter_rings.md"},
                },
                {
                    "support_id": "s-noise",
                    "text": (
                        "Bamboo is a fast-growing woody grass used in "
                        "erosion-prone landscaping; it is not related to any "
                        "planetary science topic."
                    ),
                    "metadata": {"path": "docs/bamboo_growth.md"},
                },
            ],
        },
        {
            "id": "rag_attr_ev",
            "query_text": "How far can modern consumer electric vehicles drive on a charge?",
            "response_text": (
                "Most consumer electric vehicles now deliver 300–500 km of "
                "range per full charge, depending on battery size and climate."
            ),
            "support_units": [
                {
                    "support_id": "s-anchor",
                    "text": (
                        "Modern consumer electric vehicles typically provide "
                        "between 300 and 500 km of range on a full charge, "
                        "with premium models reaching 600 km."
                    ),
                    "metadata": {"path": "docs/ev_range.md"},
                },
                {
                    "support_id": "s-near",
                    "text": (
                        "Lithium-ion battery chemistry has matured rapidly "
                        "over the past decade, improving energy density and "
                        "reducing cost per kWh."
                    ),
                    "metadata": {"path": "docs/battery_chemistry.md"},
                },
                {
                    "support_id": "s-noise",
                    "text": (
                        "Saffron is a spice harvested from the stigmas of "
                        "Crocus sativus and commands a high market price."
                    ),
                    "metadata": {"path": "docs/saffron_history.md"},
                },
            ],
        },
    ]


async def _dim_rag_attribution(
    client: httpx.AsyncClient,
    transport: Transport,
    concurrency: int,
) -> Dict[str, Any]:
    sem = asyncio.Semaphore(concurrency)
    cases = _rag_attribution_cases()

    async def _one(case: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            "input": {
                "scoring_mode": "rag",
                "query_text": case["query_text"],
                "response_text": case["response_text"],
                "support_units": case["support_units"],
                "include_triangular_diagnostics": True,
                "evidence_limit": 4,
                "heatmap_format": "data",
            }
        }
        body = await transport.submit(client, payload)
        return {"case": case, "body": body}

    results = await asyncio.gather(
        *[_with_sem(sem, lambda c=c: _one(c)) for c in cases]
    )

    rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    per_file_seen = 0
    multi_file_cases = 0
    histogram_has_keys = 0
    dead_weight_ratio_populated = 0
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

        attribution = output.get("file_attribution") or {}
        per_file = attribution.get("per_file") or []
        hist = attribution.get("reason_code_histogram") or {}
        scores_dict = output.get("scores") or {}
        dwr = scores_dict.get("dead_weight_ratio")

        per_file_seen += len(per_file)
        if len(per_file) >= 2:
            multi_file_cases += 1
        if hist:
            histogram_has_keys += 1
        if dwr is not None:
            dead_weight_ratio_populated += 1

        rows.append(
            {
                "id": case["id"],
                "n_files": len(per_file),
                "dead_weight_ratio": dwr,
                "dead_weight_file_count": scores_dict.get("dead_weight_file_count"),
                "reason_codes": sorted(hist.keys()),
                "top_dead_path": next(
                    (
                        f.get("path")
                        for f in per_file
                        if bool(f.get("dead_weight"))
                    ),
                    None,
                ),
            }
        )

    passed = (
        len(errors) == 0
        and len(rows) == len(cases)
        and multi_file_cases >= 1
        and histogram_has_keys >= 1
        and dead_weight_ratio_populated == len(rows)
    )
    return {
        "name": "rag_attribution",
        "rows": rows,
        "errors": errors,
        "cases_total": len(cases),
        "per_file_seen": per_file_seen,
        "multi_file_cases": multi_file_cases,
        "histogram_populated": histogram_has_keys,
        "dead_weight_ratio_populated": dead_weight_ratio_populated,
        "passed": passed,
    }


# ---------------------------------------------------------------------------
# Dimension 7 — Heatmap (data + HTML fragment).
# ---------------------------------------------------------------------------


def _validate_heatmap_html(html: str) -> Dict[str, Any]:
    """Cheap self-contained-ness check for the heatmap HTML fragment.

    The live fragment is a single ``<div class="lt-heatmap">`` with an
    inline ``<style>`` block. We reject obvious external dependencies
    (``<script src=``, external stylesheets, ``<iframe>``) and require
    the three band classes and the ``<h4>`` headline the renderer emits.
    """

    lowered = (html or "").lower()
    forbidden = [
        "<script ",
        "<iframe",
        "href=\"http",
        "href='http",
        "src=\"http",
        "src='http",
    ]
    has_forbidden = any(token in lowered for token in forbidden)
    has_div = "<div" in lowered and "</div>" in lowered
    # Band classes emitted by render_heatmap_html.
    has_bands = all(
        token in lowered
        for token in (".lt-band-green", ".lt-band-amber", ".lt-band-red")
    )
    has_headline = "<h4" in lowered
    has_inline_style = "<style>" in lowered
    size_bytes = len(html or "")
    return {
        "has_div": has_div,
        "has_bands": has_bands,
        "has_headline": has_headline,
        "has_inline_style": has_inline_style,
        "has_forbidden_refs": has_forbidden,
        "size_bytes": size_bytes,
    }


async def _dim_heatmap(
    client: httpx.AsyncClient,
    transport: Transport,
    rag_case: Optional[Any],
    code_case: Optional[Any],
) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    async def _run_rag(case: Any) -> None:
        payload = {
            "input": {
                "query": case.query,
                "context": case.context,
                "response": case.response,
                "include_triangular_diagnostics": True,
                "evidence_limit": 4,
                "heatmap_format": "html",
            }
        }
        body = await transport.submit(client, payload)
        output = body.get("output") or {}
        if body.get("status") != "COMPLETED" or not output.get("success"):
            errors.append(
                {
                    "lane": "rag",
                    "id": case.id,
                    "status": body.get("status"),
                    "error": output.get("error") or body.get("error"),
                }
            )
            return
        heatmap = output.get("heatmap") or {}
        html = output.get("heatmap_html")
        html_check = _validate_heatmap_html(html or "")
        tokens = heatmap.get("tokens") or []
        files = heatmap.get("files") or []
        summary = heatmap.get("summary") or {}
        thresholds = heatmap.get("thresholds") or {}
        checks.append(
            {
                "lane": "rag",
                "id": case.id,
                "has_heatmap_data": bool(heatmap),
                "token_count": len(tokens),
                "file_count": len(files),
                "summary_keys": sorted(summary.keys()),
                "thresholds_keys": sorted(thresholds.keys()),
                "token_bands": sorted(
                    {str(t.get("band")) for t in tokens if t.get("band")}
                ),
                "html": html_check,
            }
        )

    async def _run_code(case: Any) -> None:
        raw_context, _ = render_context_files(case.context_files)
        payload = {
            "input": {
                "scoring_mode": "code",
                "session_id": f"360bench:heatmap:{case.id}",
                "response_language_hint": _language_hint(case.context_files),
                "query": case.query,
                "context": raw_context,
                "response": _strip_diff_to_code(case.response),
                "evidence_limit": 6,
                "heatmap_format": "html",
            }
        }
        body = await transport.submit(client, payload)
        output = body.get("output") or {}
        if body.get("status") != "COMPLETED" or not output.get("success"):
            errors.append(
                {
                    "lane": "code",
                    "id": case.id,
                    "status": body.get("status"),
                    "error": output.get("error") or body.get("error"),
                }
            )
            return
        heatmap = output.get("heatmap") or {}
        html = output.get("heatmap_html")
        html_check = _validate_heatmap_html(html or "")
        checks.append(
            {
                "lane": "code",
                "id": case.id,
                "has_heatmap_data": bool(heatmap),
                "token_count": len(heatmap.get("tokens") or []),
                "file_count": len(heatmap.get("files") or []),
                "summary_keys": sorted((heatmap.get("summary") or {}).keys()),
                "thresholds_keys": sorted((heatmap.get("thresholds") or {}).keys()),
                "token_bands": sorted(
                    {
                        str(t.get("band"))
                        for t in (heatmap.get("tokens") or [])
                        if t.get("band")
                    }
                ),
                "html": html_check,
            }
        )

    coros: List[Awaitable[None]] = []
    if rag_case is not None:
        coros.append(_run_rag(rag_case))
    if code_case is not None:
        coros.append(_run_code(code_case))

    if coros:
        await asyncio.gather(*coros)

    # Gate: every completed check must have data + a self-contained HTML
    # fragment (div, band CSS vars, headline, no external refs).
    all_good = (
        len(errors) == 0
        and len(checks) > 0
        and all(
            c["has_heatmap_data"]
            and c["html"]["has_div"]
            and c["html"]["has_bands"]
            and c["html"]["has_headline"]
            and not c["html"]["has_forbidden_refs"]
            for c in checks
        )
    )
    return {
        "name": "heatmap",
        "checks": checks,
        "errors": errors,
        "passed": all_good,
    }


# ---------------------------------------------------------------------------
# Dimension 8 — Stateless rollup over session turns.
# ---------------------------------------------------------------------------


def _rollup_turns_from_session(
    session_dim: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Project the 5 session-dim outputs into RollupTurnInput dicts.

    The rollup endpoint is a pure transform: it takes the minimum info
    needed per turn (scores, session_signals, file_attribution, risk_band,
    recommendation, timestamp) and returns aggregates. We only forward the
    fields the public schema documents so the bench catches schema drift
    on the live endpoint.
    """

    if not session_dim:
        return []
    outputs = session_dim.get("raw_outputs") or []
    turns: List[Dict[str, Any]] = []
    for idx, out in enumerate(outputs):
        turn: Dict[str, Any] = {}
        scores = out.get("scores")
        if scores is not None:
            turn["scores"] = scores
        signals = out.get("session_signals")
        if signals is not None:
            turn["session_signals"] = signals
        attribution = out.get("file_attribution")
        if attribution is not None:
            turn["file_attribution"] = attribution
        band = out.get("band") or out.get("risk_band")
        if band is not None:
            turn["risk_band"] = band
        rec = (signals or {}).get("recommendation") if signals else None
        if rec is not None:
            turn["recommendation"] = rec
        turn["timestamp"] = str(idx)
        turns.append(turn)
    return turns


async def _dim_rollup(
    client: httpx.AsyncClient,
    transport: Transport,
    session_dim: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    turns = _rollup_turns_from_session(session_dim)
    if not turns:
        return {
            "name": "rollup",
            "skipped": True,
            "reason": "no session outputs available to roll up",
            "passed": True,
        }

    payload = {
        "input": {
            "action": "rollup",
            "turns": turns,
            "heatmap_format": "html",
        }
    }
    body = await transport.submit(client, payload)
    status = body.get("status", "")
    output = body.get("output") or {}
    if status != "COMPLETED" or not output.get("success", True) and output.get("error"):
        return {
            "name": "rollup",
            "skipped": False,
            "turns_in": len(turns),
            "errors": [
                {
                    "status": status,
                    "error": output.get("error") or body.get("error"),
                }
            ],
            "passed": False,
        }

    # The rollup handler returns the RollupResponse directly as ``output``
    # (it does not wrap into ``{"success": true, ...}``). Be tolerant of
    # either shape.
    rollup = output.get("rollup") or output
    noise = rollup.get("noise_pct")
    drift = rollup.get("model_drift_pct")
    waste = rollup.get("retrieval_waste_pct")
    histogram = rollup.get("reason_code_histogram") or {}
    recommendations = rollup.get("recommendations") or []
    risk_trail = rollup.get("risk_band_trail") or []
    drift_trend = rollup.get("drift_trend") or {}
    top_dead = rollup.get("top_dead_files") or []
    heatmap_html = rollup.get("heatmap_html")

    def _in_unit(v: Any) -> bool:
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return False
        return 0.0 <= fv <= 1.0

    bounded = all(_in_unit(v) for v in (noise, drift, waste) if v is not None)
    turns_out = rollup.get("turns")
    turns_match = (turns_out is None) or (int(turns_out) == len(turns))
    has_histogram = isinstance(histogram, dict)
    has_trail = len(risk_trail) == len(turns) or len(risk_trail) == 0
    html_ok = True
    html_check: Dict[str, Any] = {}
    if heatmap_html:
        html_check = _validate_heatmap_html(heatmap_html)
        html_ok = (
            html_check.get("has_div", False)
            and html_check.get("has_bands", False)
            and not html_check.get("has_forbidden_refs", True)
        )

    passed = bounded and turns_match and has_histogram and has_trail and html_ok

    return {
        "name": "rollup",
        "skipped": False,
        "turns_in": len(turns),
        "turns_out": turns_out,
        "noise_pct": noise,
        "model_drift_pct": drift,
        "retrieval_waste_pct": waste,
        "reason_code_histogram": histogram,
        "recommendations": recommendations,
        "risk_band_trail": risk_trail,
        "drift_trend": drift_trend,
        "top_dead_files": top_dead,
        "has_heatmap_html": heatmap_html is not None,
        "html_check": html_check,
        "passed": bool(passed),
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

    if "unused_raw" in dims:
        d = dims["unused_raw"]
        status = "PASS" if d.get("passed") else "FAIL"
        overall_pass = overall_pass and bool(d.get("passed"))
        print()
        print(f"[2b/5] raw_context unused-chunks contract (RAG lane)             {status}")
        print("-" * 80)
        print(
            f"  segmented support_units: {d.get('n_support_units')}    "
            f"status={d.get('status')}    contract_ok={d.get('contract_ok')}"
        )
        usage = d.get("support_units_usage") or {}
        print(
            f"  usage breakdown: used={usage.get('used')}  "
            f"unused={usage.get('unused')}  uncertain={usage.get('uncertain')}"
        )
        print(
            f"  context coverage / unused / uncertain ratios : "
            f"{_fmt_f(d.get('context_coverage_ratio'))} / "
            f"{_fmt_f(d.get('context_unused_ratio'))} / "
            f"{_fmt_f(d.get('context_uncertain_ratio'))}"
        )

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
        print(
            f"  ungrounded any-signal recall (ast | lit | nli>=.70)  : "
            f"{_fmt_f(d.get('ungrounded_any_signal_recall'))}  "
            f"({d.get('ungrounded_any_signal_fired', 0)}/{d.get('ungrounded_total', 0)})"
        )
        print(
            f"  grounded AST false positives  : "
            f"{d.get('grounded_ast_false_positives', 0)} / {d.get('grounded_total', 0)}"
        )
        _backends = d.get("parser_backends") or {}
        _backend_ok = d.get("parser_backend_ok")
        print(
            f"  AST parser backends    : {_backends}  "
            f"(tree_sitter_only={_backend_ok})"
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
            f"{'ast_ph':>6} {'ast_#ph':>7} {'ast_#dr':>7} "
            f"{'lit_#mi':>7} {'nli_max':>7} {'casc':>5} {'dead_w':>7}"
        )
        for row in sorted(d.get("rows", []), key=lambda r: (r["label"], r["id"])):
            sub = row.get("subcategory") or "-"
            print(
                f"  {row['id']:<5} {row['label']:<11} {sub:<14} "
                f"{_fmt_f(row.get('composite_score'), 10)} "
                f"{_fmt_f(row.get('phantom_probability'), 10)} "
                f"{str(row.get('phantom_verdict')):>7} "
                f"{str(row.get('ast_phantom_verdict')):>6} "
                f"{_fmt_i(row.get('ast_phantom_symbol_count'), 7)} "
                f"{_fmt_i(row.get('ast_literal_drift_count'), 7)} "
                f"{_fmt_i(row.get('literal_novelty_missing_count'), 7)} "
                f"{_fmt_f(row.get('nli_contradiction_prob_max'), 7)} "
                f"{str(row.get('nli_cascade_triggered'))[:5]:>5} "
                f"{_fmt_f(row.get('dead_weight_ratio'), 7)}"
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

    if "rag_attribution" in dims:
        d = dims["rag_attribution"]
        status = "PASS" if d.get("passed") else "FAIL"
        overall_pass = overall_pass and bool(d.get("passed"))
        print()
        print(f"[6/8] RAG per-file attribution + reason-code histogram             {status}")
        print("-" * 80)
        print(
            f"  cases                  : {len(d.get('rows', []))} completed / "
            f"{len(d.get('errors', []))} errors"
        )
        print(
            f"  multi-file cases       : {d.get('multi_file_cases', 0)} / "
            f"{d.get('cases_total', 0)}   "
            f"histogram populated : {d.get('histogram_populated', 0)} / "
            f"{d.get('cases_total', 0)}   "
            f"dead_weight_ratio populated : {d.get('dead_weight_ratio_populated', 0)} / "
            f"{d.get('cases_total', 0)}"
        )
        for row in d.get("rows", []):
            rc = ",".join(row.get("reason_codes") or []) or "-"
            print(
                f"  {row['id']:<18} n_files={_fmt_i(row.get('n_files'))} "
                f"dead_ratio={_fmt_f(row.get('dead_weight_ratio'))} "
                f"dead_files={_fmt_i(row.get('dead_weight_file_count'))} "
                f"reason_codes=[{rc}]"
            )
        for e in d.get("errors", []):
            print(f"  ERR {e.get('id')}: {e.get('status')} {e.get('error')}")

    if "heatmap" in dims:
        d = dims["heatmap"]
        status = "PASS" if d.get("passed") else "FAIL"
        overall_pass = overall_pass and bool(d.get("passed"))
        print()
        print(f"[7/8] Heatmap (data payload + self-contained HTML fragment)        {status}")
        print("-" * 80)
        for c in d.get("checks", []):
            h = c.get("html", {})
            print(
                f"  {c.get('lane'):<4} {c.get('id'):<6} "
                f"tokens={_fmt_i(c.get('token_count'))} "
                f"files={_fmt_i(c.get('file_count'))} "
                f"bands={c.get('token_bands')} "
                f"html_bytes={h.get('size_bytes')} "
                f"div={h.get('has_div')} bands_css={h.get('has_bands')} "
                f"headline={h.get('has_headline')} "
                f"forbidden_refs={h.get('has_forbidden_refs')}"
            )
        for e in d.get("errors", []):
            print(f"  ERR {e.get('lane')} {e.get('id')}: {e.get('status')} {e.get('error')}")

    if "rollup" in dims:
        d = dims["rollup"]
        if d.get("skipped"):
            print()
            print(f"[8/8] Rollup                                                      SKIP")
            print("-" * 80)
            print(f"  {d.get('reason')}")
        else:
            status = "PASS" if d.get("passed") else "FAIL"
            overall_pass = overall_pass and bool(d.get("passed"))
            print()
            print(f"[8/8] Stateless rollup over session turns                          {status}")
            print("-" * 80)
            print(
                f"  turns_in / turns_out   : {d.get('turns_in')} / {d.get('turns_out')}"
            )
            print(
                f"  noise_pct / model_drift_pct / retrieval_waste_pct : "
                f"{_fmt_f(d.get('noise_pct'))} / "
                f"{_fmt_f(d.get('model_drift_pct'))} / "
                f"{_fmt_f(d.get('retrieval_waste_pct'))}"
            )
            hist = d.get("reason_code_histogram") or {}
            if hist:
                print(
                    "  reason_code_histogram  : "
                    + ", ".join(f"{k}={v}" for k, v in sorted(hist.items()))
                )
            trail = d.get("risk_band_trail") or []
            if trail:
                print(f"  risk_band_trail        : {' > '.join(str(b) for b in trail)}")
            trend = d.get("drift_trend") or {}
            if trend:
                print(
                    f"  drift_trend            : "
                    f"min={_fmt_f(trend.get('min'))} "
                    f"max={_fmt_f(trend.get('max'))} "
                    f"mean={_fmt_f(trend.get('mean'))} "
                    f"last={_fmt_f(trend.get('last'))}"
                )
            top_dead = d.get("top_dead_files") or []
            if top_dead:
                print("  top_dead_files         :")
                for f in top_dead[:5]:
                    print(
                        f"    {f.get('path')!s:<28} "
                        f"dead_turns={_fmt_i(f.get('dead_turns'))} "
                        f"ema_owner_share={_fmt_f(f.get('ema_owner_share'))}"
                    )
            print(f"  heatmap_html emitted   : {d.get('has_heatmap_html')}")
            for e in d.get("errors", []) or []:
                print(f"  ERR {e}")

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
        help=(
            "comma-separated dimensions to skip: "
            "rag,unused,code,session,concurrency,rag_attribution,heatmap,rollup"
        ),
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
    p.add_argument(
        "--gateway-url",
        default=os.environ.get("LATENCE_GATEWAY_URL"),
        help=(
            "If set, route every request through the Latence API gateway at "
            "this base URL (e.g. http://127.0.0.1:8787) instead of RunPod. "
            "Used by the local-gateway parity bench."
        ),
    )
    p.add_argument(
        "--gateway-key",
        default=os.environ.get("LATENCE_GATEWAY_KEY"),
        help="Bearer token for the gateway (MASTER_API_KEY in dev mode).",
    )
    return p.parse_args()


async def _async_main(args: argparse.Namespace) -> int:
    using_gateway = bool(args.gateway_url)

    if using_gateway:
        if not args.gateway_key:
            print(
                "error: --gateway-key is required when --gateway-url is set "
                "(or set LATENCE_GATEWAY_KEY; dev gateway uses MASTER_API_KEY)",
                file=sys.stderr,
            )
            return 2
    else:
        if not args.endpoint_id or not args.api_key:
            print(
                "error: --endpoint-id and --api-key are required "
                "(or set RUNPOD_ENDPOINT_ID / RUNPOD_API_KEY)",
                file=sys.stderr,
            )
            return 2

    transport = Transport(
        endpoint_id=args.endpoint_id or "",
        api_key=args.api_key or "",
        use_runsync=not args.use_run_poll,
        timeout_s=args.timeout_s,
        gateway_url=args.gateway_url,
        gateway_key=args.gateway_key,
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
            print("[2b/5] running raw_context unused-chunks contract probe ...", flush=True)
            dims["unused_raw"] = await _dim_unused_raw_context(client, transport)
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
        if "rag_attribution" not in skip:
            print(
                "[6/8] running RAG per-file attribution + histogram probe ...",
                flush=True,
            )
            dims["rag_attribution"] = await _dim_rag_attribution(
                client, transport, args.concurrency
            )
        if "heatmap" not in skip:
            print("[7/8] running heatmap probe (RAG + code, html fragment) ...", flush=True)
            rag_case_for_hm = selected_rag[0] if selected_rag else None
            code_case_for_hm = selected_code[0] if selected_code else None
            dims["heatmap"] = await _dim_heatmap(
                client, transport, rag_case_for_hm, code_case_for_hm
            )
        if "rollup" not in skip:
            # Rollup aggregates the 5-turn session narrative. If session
            # was skipped or failed, the rollup dim short-circuits with
            # ``skipped=True`` rather than crashing.
            print("[8/8] running stateless rollup over session turns ...", flush=True)
            dims["rollup"] = await _dim_rollup(
                client, transport, dims.get("session")
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
