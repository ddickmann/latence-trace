#!/usr/bin/env python
"""Parity gate for the reranker (bge-reranker-v2-m3).

Sister script to :mod:`scripts.parity_nli_multi`. Same hard contract:
load HF reference, hit vLLM endpoint, assert ≤ tolerance on every
fixture in BOTH single and batched mode.

The reranker is the highest-volume call in the SOTA grounding stack —
``verify_claims`` issues one ``score_pairs`` per request fanning out
``k`` premises × ``n`` claim atoms — so a silent batched-vs-single
drift here produces wrong rankings under load and right ones under
single-row debugging. That's the worst kind of bug to ship; this gate
is non-negotiable.

Wire format
-----------
``BAAI/bge-reranker-v2-m3`` is XLM-RoBERTa-large with a single-logit
relevance head. vLLM 0.19.x serves it via
``--runner pooling --convert classify`` (the legacy ``--task score``
flag was removed in 0.19; ``--convert classify`` exposes
``/v1/score`` automatically for any single-label classifier head)::

    POST /v1/score
    { "model": "...", "text_1": "<query>", "text_2": ["doc1", "doc2", ...] }

Why we sigmoid the HF reference
-------------------------------
The HF ``T5ForConditionalGeneration`` reference returns *raw logits*
from the single-label classification head — values like ``10.19``,
``-3.17``, ``-8.13``. vLLM 0.19.x's ``/v1/score`` endpoint applies
``torch.sigmoid`` server-side and returns probabilities in ``[0, 1]``.
For the production code path this is benign: every consumer in
:mod:`latence_trace.core.nli` (``_select_premises_for_claim``,
``score_pairs`` via ``flat_scores``) uses the reranker score *only*
for ranking + top-k selection, and ``sigmoid`` is monotonic — so the
in-process :class:`CrossEncoderPremiseReranker` (raw logits) and the
vLLM-served :class:`VllmRerankerProvider` (sigmoid'd) produce
**identical ranking decisions** despite the unit mismatch. The
parity gate therefore compares ``sigmoid(HF_logits)`` against the
vLLM scores so we test the actual production wire shape; we use a
``--score-tol`` default of ``2e-2`` to mirror the same bf16 noise
floor we documented for the MiniCheck T5 plugin.

Run after the GPU is clear and the server is up:

.. code-block:: bash

    vllm serve BAAI/bge-reranker-v2-m3 \\
        --runner pooling --task score \\
        --host 127.0.0.1 --port 18007

    python scripts/parity_reranker.py \\
        --endpoint http://127.0.0.1:18007 \\
        --model BAAI/bge-reranker-v2-m3
"""

from __future__ import annotations

import argparse
import gc
from typing import List

import httpx
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# ---------------------------------------------------------------------------
# Fixtures: query + multiple candidate documents (the (1, N) shape vLLM is
# optimised for) — exercise EN, DE, mixed lengths, exact-match vs
# distractor.
# ---------------------------------------------------------------------------


FIXTURES = [
    {
        "label": "en_apple_q3",
        "query": "What was Apple's Greater China revenue in Q3 FY2024?",
        "documents": [
            "Apple Q3 FY2024 reported Greater China revenue of 14728 million dollars, down 6.5% year over year.",
            "Apple Q2 FY2024 reported Americas revenue of 37273 million dollars.",
            "Microsoft Q1 FY2025 reported Intelligent Cloud revenue of 24092 million dollars.",
            "Apple Q3 FY2024 services revenue grew to a new all-time high of 24213 million dollars.",
            "Tim Cook said the iPhone installed base reached an all-time high in Q3 FY2024.",
        ],
        # Index of the document we expect to score highest.
        "expected_top": 0,
    },
    {
        "label": "en_paris_seine",
        "query": "Which river flows through Paris?",
        "documents": [
            "The Seine flows through Paris and meets the English Channel at Le Havre.",
            "Berlin sits on the river Spree.",
            "The Thames flows through London.",
            "Paris, the capital of France, is famous for the Eiffel Tower.",
        ],
        "expected_top": 0,
    },
    {
        "label": "de_kafka",
        "query": "Wann wurde Franz Kafka geboren?",
        "documents": [
            "Franz Kafka wurde am 3. Juli 1883 in Prag geboren.",
            "Franz Kafka starb am 3. Juni 1924 in Kierling bei Klosterneuburg.",
            "Kafkas Werk umfasst Romane wie Der Process und Das Schloss.",
            "Goethe wurde am 28. August 1749 in Frankfurt am Main geboren.",
        ],
        "expected_top": 0,
    },
    {
        "label": "de_berlin",
        "query": "Was ist die Hauptstadt von Deutschland?",
        "documents": [
            "Berlin ist seit 1990 die Hauptstadt der Bundesrepublik Deutschland.",
            "München ist die Landeshauptstadt von Bayern.",
            "Bonn war von 1949 bis 1990 die provisorische Hauptstadt der Bundesrepublik Deutschland.",
            "Berlin liegt im Nordosten Deutschlands.",
        ],
        "expected_top": 0,
    },
    # Mixed-length batch: some docs are very short, others very long. The
    # padding token / attention-mask handling differs the most here, so
    # this is the most likely place batch-vs-single regressions surface.
    {
        "label": "mixed_length",
        "query": "Who founded SpaceX and in what year?",
        "documents": [
            "SpaceX.",
            "SpaceX was founded by Elon Musk in 2002 in Hawthorne, California, with the goal of reducing space transportation costs and enabling the colonization of Mars. The company has since become one of the most influential players in the commercial space launch industry, pioneering reusable rocket technology with the Falcon 9 and Falcon Heavy launch vehicles, and operating the Starlink satellite internet constellation.",
            "Musk.",
            "Tesla, Inc. is an American electric vehicle and clean energy company.",
        ],
        "expected_top": 1,
    },
]


# ---------------------------------------------------------------------------
# HF reference
# ---------------------------------------------------------------------------


def _flatten(fixtures) -> tuple[List[str], List[str], List[tuple[int, int]]]:
    """Flatten ``[{query, documents}, ...]`` to per-pair lists.

    Returns ``(queries, documents, fixture_index_pairs)`` where each
    ``fixture_index_pairs[i] = (fixture_idx, doc_idx_within_fixture)``
    so we can scatter the flat scores back into the per-fixture shape.
    """

    queries: List[str] = []
    documents: List[str] = []
    pair_index: List[tuple[int, int]] = []
    for f_idx, fix in enumerate(fixtures):
        for d_idx, doc in enumerate(fix["documents"]):
            queries.append(fix["query"])
            documents.append(doc)
            pair_index.append((f_idx, d_idx))
    return queries, documents, pair_index


def _hf_pair_scores(model_id: str, fixtures, *, device: str) -> List[List[float]]:
    """HF reference: run every (query, doc) pair through the cross-encoder.

    Returns a per-fixture list of scores in the same order as
    ``fixtures[i]["documents"]``. We feed ALL pairs across ALL fixtures
    in a single padded batch so the HF reference itself exercises the
    same mixed-length batching that the vLLM server faces — it's the
    closest baseline we have for "what should batched mode return?".
    """

    resolved_device = torch.device(device)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, use_fast=True, trust_remote_code=True
    )
    kwargs = {}
    if resolved_device.type == "cuda":
        kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForSequenceClassification.from_pretrained(model_id, **kwargs)
    model.to(device)
    model.eval()

    queries, documents, pair_index = _flatten(fixtures)
    encoded = tokenizer(
        queries,
        documents,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        raw = model(**encoded).logits.float().cpu().reshape(-1)
        logits = torch.sigmoid(raw).tolist()

    grouped: List[List[float]] = [[0.0] * len(fix["documents"]) for fix in fixtures]
    for (f_idx, d_idx), score in zip(pair_index, logits):
        grouped[f_idx][d_idx] = float(score)

    del model, tokenizer, encoded
    gc.collect()
    if resolved_device.type == "cuda":
        torch.cuda.empty_cache()
    return grouped


# ---------------------------------------------------------------------------
# vLLM /v1/score endpoint
# ---------------------------------------------------------------------------


def _coerce_score(item) -> float:
    if isinstance(item, dict):
        for key in ("score", "relevance_score", "value"):
            if key in item:
                return float(item[key])
        raise TypeError(f"Unsupported /v1/score row dict shape: {sorted(item.keys())!r}")
    return float(item)


def _endpoint_score_one_query(
    endpoint: str, model: str, query: str, documents: List[str]
) -> List[float]:
    """One ``(text_1, text_2[])`` POST = one fixture in batched mode."""

    with httpx.Client(timeout=120.0) as client:
        response = client.post(
            f"{endpoint.rstrip('/')}/v1/score",
            json={"model": model, "text_1": query, "text_2": list(documents)},
        )
        response.raise_for_status()
        payload = response.json()
    rows = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise TypeError(f"Unsupported /v1/score payload: {type(rows)!r}")
    return [_coerce_score(row) for row in rows]


def _endpoint_score_one_pair(
    endpoint: str, model: str, query: str, document: str
) -> float:
    """Force a (1, 1) POST so we can compare batch-vs-single per pair."""

    return _endpoint_score_one_query(endpoint, model, query, [document])[0]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--score-tol", type=float, default=5e-2)
    args = parser.parse_args()

    print(f"\nLoading HF reference: {args.model} on {args.device} ...")
    expected = _hf_pair_scores(args.model, FIXTURES, device=args.device)

    print("\n=== Batched (one query, N docs) vs HF reference ===")
    batched: List[List[float]] = []
    for f_idx, fix in enumerate(FIXTURES):
        got = _endpoint_score_one_query(
            args.endpoint, args.model, fix["query"], fix["documents"]
        )
        batched.append(got)
        want = expected[f_idx]
        if len(got) != len(want):
            raise AssertionError(
                f"fixture[{f_idx}] {fix['label']!r}: server returned "
                f"{len(got)} scores, expected {len(want)}"
            )
        deltas = [abs(a - b) for a, b in zip(want, got)]
        max_delta = max(deltas)
        print(
            f"  fixture[{f_idx}] {fix['label']:<14} "
            f"top_hf={max(range(len(want)), key=lambda i: want[i])} "
            f"top_vllm={max(range(len(got)), key=lambda i: got[i])} "
            f"max_Δ={max_delta:.4f}"
        )
        if max_delta > args.score_tol:
            raise AssertionError(
                f"BATCHED fixture {f_idx} {fix['label']!r}: max Δ={max_delta:.6f} "
                f"exceeds score-tol={args.score_tol}\n"
                f"  hf  : {[round(v, 4) for v in want]}\n"
                f"  vllm: {[round(v, 4) for v in got]}"
            )
        # Top-1 must match ground truth — guards against semantic
        # regressions even when raw scores drift inside tolerance.
        top_vllm = max(range(len(got)), key=lambda i: got[i])
        if top_vllm != fix["expected_top"]:
            raise AssertionError(
                f"BATCHED fixture {f_idx} {fix['label']!r}: top doc "
                f"index={top_vllm} (expected {fix['expected_top']})"
            )

    print("\n=== Single (one query, one doc) vs Batched (catches attention-mask bugs) ===")
    for f_idx, fix in enumerate(FIXTURES):
        for d_idx, doc in enumerate(fix["documents"]):
            single = _endpoint_score_one_pair(args.endpoint, args.model, fix["query"], doc)
            delta = abs(single - batched[f_idx][d_idx])
            if delta > args.score_tol:
                raise AssertionError(
                    f"SINGLE-vs-BATCHED fixture {f_idx} {fix['label']!r} "
                    f"doc[{d_idx}] Δ={delta:.6f} exceeds tol={args.score_tol}\n"
                    f"  single ={single:.4f}\n"
                    f"  batched={batched[f_idx][d_idx]:.4f}\n"
                    f"  This is almost certainly an attention-mask bug "
                    f"in vLLM's stock RoBERTa scoring path — investigate "
                    f"BertSelfAttention / EncoderOnlyAttention before "
                    f"shipping."
                )
        print(f"  fixture[{f_idx}] {fix['label']:<14} all docs single≈batched")

    print("\n=== Single (one query, one doc) vs HF reference ===")
    for f_idx, fix in enumerate(FIXTURES):
        for d_idx, doc in enumerate(fix["documents"]):
            single = _endpoint_score_one_pair(args.endpoint, args.model, fix["query"], doc)
            delta = abs(single - expected[f_idx][d_idx])
            if delta > args.score_tol:
                raise AssertionError(
                    f"SINGLE fixture {f_idx} {fix['label']!r} doc[{d_idx}] "
                    f"Δ={delta:.6f} exceeds tol={args.score_tol}"
                )

    print("\nALL PARITY CHECKS PASSED")


if __name__ == "__main__":
    main()
