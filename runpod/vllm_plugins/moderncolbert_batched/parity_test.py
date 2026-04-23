#!/usr/bin/env python
"""Reference/verify parity harness for the batched ModernColBERT IO path."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Any

import httpx
import numpy as np


FIXTURES = [
    {"text": "What was Apple Q3 FY24 Greater China revenue?", "is_query": True},
    {
        "text": (
            "Apple Q3 FY2024 segment revenue in millions: Americas 37678, Europe 21883, "
            "Greater China 14728, Japan 5099, Rest of Asia Pacific 5455."
        ),
        "is_query": False,
    },
    {"text": "Wann wurde Teardrops in den USA veroeffentlicht?", "is_query": True},
    {
        "text": "Teardrops was released in the United States on 20 July 1981.",
        "is_query": False,
    },
]


def _decode_matrix(payload: Any, *, colbert_dim: int = 128) -> np.ndarray:
    data = payload.get("data", payload) if isinstance(payload, dict) else payload
    if isinstance(data, str):
        raw = np.frombuffer(base64.b64decode(data.encode("ascii")), dtype=np.float32)
        if raw.size == 0:
            return np.zeros((0, colbert_dim), dtype=np.float32)
        return raw.reshape(-1, colbert_dim)
    array = np.asarray(data, dtype=np.float32)
    if array.ndim == 1:
        return array.reshape(-1, colbert_dim)
    return array


def _encode_reference(array: np.ndarray) -> dict[str, Any]:
    raw = np.asarray(array, dtype=np.float32)
    return {
        "shape": list(raw.shape),
        "data": base64.b64encode(raw.tobytes()).decode("ascii"),
    }


def _decode_reference(payload: dict[str, Any]) -> np.ndarray:
    shape = tuple(int(item) for item in payload["shape"])
    raw = np.frombuffer(base64.b64decode(payload["data"].encode("ascii")), dtype=np.float32)
    return raw.reshape(shape)


def _single_request(client: httpx.Client, endpoint: str, model: str, text: str, is_query: bool) -> np.ndarray:
    response = client.post(
        f"{endpoint.rstrip('/')}/pooling",
        json={
            "model": model,
            "task": "token_embed",
            "data": {"text": text, "is_query": is_query},
        },
    )
    response.raise_for_status()
    return _decode_matrix(response.json())


def _batched_request(client: httpx.Client, endpoint: str, model: str, fixtures: list[dict[str, Any]]) -> list[np.ndarray]:
    response = client.post(
        f"{endpoint.rstrip('/')}/pooling",
        json={
            "model": model,
            "task": "token_embed",
            "data": {
                "text": [item["text"] for item in fixtures],
                "is_query": [bool(item["is_query"]) for item in fixtures],
            },
        },
    )
    response.raise_for_status()
    body = response.json()
    rows = body.get("data", body) if isinstance(body, dict) else body
    return [_decode_matrix(row) for row in rows]


def record_reference(endpoint: str, model: str, output_path: Path) -> None:
    with httpx.Client(timeout=60.0) as client:
        rows = []
        for fixture in FIXTURES:
            matrix = _single_request(client, endpoint, model, fixture["text"], bool(fixture["is_query"]))
            rows.append({**fixture, "embedding": _encode_reference(matrix)})
    output_path.write_text(json.dumps({"model": model, "fixtures": rows}, indent=2), encoding="utf-8")


def verify_reference(endpoint: str, model: str, reference_path: Path, *, vector_tol: float) -> None:
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    rows = reference["fixtures"]
    with httpx.Client(timeout=60.0) as client:
        batched = _batched_request(client, endpoint, model, rows)
        for idx, fixture in enumerate(rows):
            expected = _decode_reference(fixture["embedding"])
            single = _single_request(client, endpoint, model, fixture["text"], bool(fixture["is_query"]))
            if single.shape != expected.shape:
                raise AssertionError(f"single shape mismatch for fixture {idx}: {single.shape} != {expected.shape}")
            single_delta = float(np.max(np.abs(single - expected))) if expected.size else 0.0
            if single_delta > vector_tol:
                raise AssertionError(f"single delta {single_delta:.6f} exceeds {vector_tol} for fixture {idx}")

            batch_value = batched[idx]
            if batch_value.shape != expected.shape:
                raise AssertionError(f"batch shape mismatch for fixture {idx}: {batch_value.shape} != {expected.shape}")
            batch_delta = float(np.max(np.abs(batch_value - expected))) if expected.size else 0.0
            if batch_delta > vector_tol:
                raise AssertionError(f"batch delta {batch_delta:.6f} exceeds {vector_tol} for fixture {idx}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--vector-tol", type=float, default=1e-3)
    args = parser.parse_args()

    reference_path = Path(args.reference)
    if args.record:
        record_reference(args.endpoint, args.model, reference_path)
        return
    verify_reference(args.endpoint, args.model, reference_path, vector_tol=float(args.vector_tol))


if __name__ == "__main__":
    main()
