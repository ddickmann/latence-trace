#!/usr/bin/env python
"""Parity harness for the live NLI vLLM plugin."""

from __future__ import annotations

import argparse
import gc
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


def _unwrap_data(payload):
    current = payload
    while isinstance(current, dict) and "data" in current:
        current = current["data"]
    return current


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
    resolved_device = torch.device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
    model_kwargs = {"trust_remote_code": True}
    if resolved_device.type != "cpu":
        device_name = (
            f"{resolved_device.type}:{resolved_device.index}"
            if resolved_device.index is not None
            else f"{resolved_device.type}:0"
        )
        model_kwargs["device_map"] = {"": device_name}
        model_kwargs["low_cpu_mem_usage"] = True
        if resolved_device.type == "cuda":
            model_kwargs["dtype"] = torch.bfloat16
    model = AutoModelForSequenceClassification.from_pretrained(model_id, **model_kwargs)
    if resolved_device.type == "cpu":
        model.to(device)
    model.eval()
    pairs = list(pairs)
    target_device = next(model.parameters()).device
    c_idx, n_idx, e_idx = _resolve_label_indices(getattr(model.config, "id2label", None) or {})
    results = []
    for pair in pairs:
        encoded = tokenizer(
            pair["premise"],
            pair["hypothesis"],
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        encoded = {key: value.to(target_device) for key, value in encoded.items()}
        with torch.no_grad():
            probs = torch.softmax(model(**encoded).logits.float(), dim=-1).cpu().numpy()[0]
        results.append((float(probs[e_idx]), float(probs[n_idx]), float(probs[c_idx])))
        del encoded, probs
    del model, tokenizer
    gc.collect()
    if target_device.type == "cuda":
        torch.cuda.empty_cache()
    return results


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
        rows = _unwrap_data(response.json())
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
        rows = _unwrap_data(response.json())
        row = rows[0]
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
    parser.add_argument("--prob-tol", type=float, default=1e-2)
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
