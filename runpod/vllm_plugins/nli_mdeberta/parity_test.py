#!/usr/bin/env python
"""Parity harness for the live NLI vLLM plugin."""

from __future__ import annotations

import argparse
from typing import Iterable

import httpx
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


FIXTURES = [
    {
        "premise": "Apple Q3 FY2024 Greater China revenue was 14728 million dollars.",
        "hypothesis": "Apple Greater China revenue was 14728 million dollars in Q3 FY2024.",
    },
    {
        "premise": "Apple Q3 FY2024 Greater China revenue was 14728 million dollars.",
        "hypothesis": "Apple Greater China revenue was 12748 million dollars in Q3 FY2024.",
    },
    {
        "premise": "Berlin ist die Hauptstadt Deutschlands seit 1990.",
        "hypothesis": "Berlin ist die Hauptstadt von Deutschland.",
    },
    {
        "premise": "Berlin ist die Hauptstadt Deutschlands seit 1990.",
        "hypothesis": "Muenchen ist die Hauptstadt von Deutschland.",
    },
]


def _resolve_label_indices(id2label: dict) -> tuple[int, int, int]:
    normalized = {str(idx): str(label).lower() for idx, label in (id2label or {}).items()}
    contradiction_idx = neutral_idx = entailment_idx = -1
    for idx_str, label in normalized.items():
        idx = int(idx_str)
        if "contradiction" in label:
            contradiction_idx = idx
        elif "neutral" in label:
            neutral_idx = idx
        elif "entailment" in label:
            entailment_idx = idx
    if contradiction_idx < 0 or neutral_idx < 0 or entailment_idx < 0:
        return 0, 1, 2
    return contradiction_idx, neutral_idx, entailment_idx


def _hf_scores(model_id: str, pairs: Iterable[dict[str, str]], *, device: str) -> list[tuple[float, float, float]]:
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, trust_remote_code=True)
    model.to(device)
    model.eval()
    pairs = list(pairs)
    encoded = tokenizer(
        [pair["premise"] for pair in pairs],
        [pair["hypothesis"] for pair in pairs],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        probs = torch.softmax(model(**encoded).logits, dim=-1).cpu().numpy()
    c_idx, n_idx, e_idx = _resolve_label_indices(getattr(model.config, "id2label", None) or {})
    return [
        (float(row[e_idx]), float(row[n_idx]), float(row[c_idx]))
        for row in probs
    ]


def _endpoint_scores(endpoint: str, model: str, pairs: list[dict[str, str]]) -> list[tuple[float, float, float]]:
    with httpx.Client(timeout=60.0) as client:
        response = client.post(
            f"{endpoint.rstrip('/')}/pooling",
            json={
                "model": model,
                "task": "plugin",
                "data": {
                    "premise": [pair["premise"] for pair in pairs],
                    "hypothesis": [pair["hypothesis"] for pair in pairs],
                },
            },
        )
        response.raise_for_status()
        body = response.json()
        rows = body.get("data", body) if isinstance(body, dict) else body
        return [
            (
                float(row["entail"]),
                float(row["neutral"]),
                float(row["contradict"]),
            )
            for row in rows
        ]


def _endpoint_single(endpoint: str, model: str, pair: dict[str, str]) -> tuple[float, float, float]:
    with httpx.Client(timeout=60.0) as client:
        response = client.post(
            f"{endpoint.rstrip('/')}/pooling",
            json={
                "model": model,
                "task": "plugin",
                "data": pair,
            },
        )
        response.raise_for_status()
        body = response.json()
        row = body["data"][0]
        return (
            float(row["entail"]),
            float(row["neutral"]),
            float(row["contradict"]),
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--prob-tol", type=float, default=5e-4)
    args = parser.parse_args()

    expected = _hf_scores(args.model, FIXTURES, device=args.device)
    batched = _endpoint_scores(args.endpoint, args.model, FIXTURES)

    for idx, (want, got) in enumerate(zip(expected, batched)):
        delta = max(abs(a - b) for a, b in zip(want, got))
        if delta > args.prob_tol:
            raise AssertionError(f"batched fixture {idx} delta {delta:.6f} exceeds {args.prob_tol}")

    for idx, pair in enumerate(FIXTURES):
        want = expected[idx]
        got = _endpoint_single(args.endpoint, args.model, pair)
        delta = max(abs(a - b) for a, b in zip(want, got))
        if delta > args.prob_tol:
            raise AssertionError(f"single fixture {idx} delta {delta:.6f} exceeds {args.prob_tol}")

    for idx, pair in enumerate(FIXTURES):
        single = _endpoint_single(args.endpoint, args.model, pair)
        delta = max(abs(a - b) for a, b in zip(single, batched[idx]))
        if delta > args.prob_tol:
            raise AssertionError(
                f"single-vs-batch fixture {idx} delta {delta:.6f} exceeds {args.prob_tol}"
            )


if __name__ == "__main__":
    main()
