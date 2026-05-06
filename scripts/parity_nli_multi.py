#!/usr/bin/env python
"""Parity gate for the multilingual NLI server (bge-m3-zeroshot-v2.0).

Mirrors :mod:`runpod.vllm_plugins.nli_mdeberta.parity_test` and
:mod:`runpod.vllm_plugins.minicheck_t5.parity_test` so all three SOTA
NLI servers ship behind the same uniform "load HF reference, hit vLLM
endpoint, assert ≤ tol on every fixture" gate.

This server is **vLLM-native** — bge-m3-zeroshot-v2.0 is XLM-RoBERTa-large
with a binary classification head, served by ``vllm serve … --task
classify`` with no custom plugin. The audit on
``/usr/local/lib/python3.11/dist-packages/vllm/model_executor/models/
roberta.py`` confirmed FlashAttention + fused QKVParallelLinear; the
only un-fused ops are LayerNorm + GeLU (~15-20 % left on the table for
a future vllm-factory Triton port). For now, parity is the hard gate.

Why batched mode is non-negotiable:
vLLM packs multiple prompts into one flat ``input_ids`` tensor and
relies on the encoder to honour per-row attention masks. Subtle
attention-mask bugs (off-by-one in segment IDs, wrong padding token,
missing ``</s></s>`` separator handling) **only surface in batched
mode**. Every fixture below is run single AND in a single mixed-length
batch; the script asserts BOTH paths agree with the HF reference AND
agree with each other.

Run after the GPU is clear and the new server is up:

.. code-block:: bash

    vllm serve MoritzLaurer/bge-m3-zeroshot-v2.0 \\
        --runner pooling --task classify \\
        --host 127.0.0.1 --port 18006

    python scripts/parity_nli_multi.py \\
        --endpoint http://127.0.0.1:18006 \\
        --model MoritzLaurer/bge-m3-zeroshot-v2.0
"""

from __future__ import annotations

import argparse
import gc
from typing import List, Tuple

import httpx
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# ---------------------------------------------------------------------------
# Fixtures — EN + DE, supported / contradicted / paraphrase, mixed lengths
# ---------------------------------------------------------------------------


FIXTURES = [
    {
        "premise": (
            "Apple Q3 FY2024 reported Greater China revenue of 14728 million "
            "dollars, down 6.5% year over year."
        ),
        "hypothesis": (
            "Apple's Greater China revenue was 14728 million dollars in Q3 "
            "FY2024."
        ),
        "expected_band": "supported",
    },
    {
        "premise": (
            "Apple Q3 FY2024 reported Greater China revenue of 14728 million "
            "dollars, down 6.5% year over year."
        ),
        "hypothesis": (
            "Apple's Greater China revenue was 12748 million dollars in Q3 "
            "FY2024."
        ),
        "expected_band": "contradicted",
    },
    {
        "premise": "The capital of France is Paris and it sits on the Seine.",
        "hypothesis": "Paris, the French capital, lies on the Seine.",
        "expected_band": "supported",
    },
    {
        "premise": (
            "Berlin ist seit 1990 die Hauptstadt der Bundesrepublik "
            "Deutschland."
        ),
        "hypothesis": "Berlin ist die Hauptstadt von Deutschland.",
        "expected_band": "supported",
    },
    {
        "premise": (
            "Berlin ist seit 1990 die Hauptstadt der Bundesrepublik "
            "Deutschland."
        ),
        "hypothesis": "München ist die Hauptstadt von Deutschland.",
        "expected_band": "contradicted",
    },
    # ── Long-context fixture (stresses attention masking on the
    # padded vs unpadded boundary in the batched path).
    {
        "premise": (
            "Franz Kafka wurde am 3. Juli 1883 in Prag geboren und starb am 3. "
            "Juni 1924 in Kierling bei Klosterneuburg. Er gilt als einer der "
            "wichtigsten Schriftsteller der deutschsprachigen Literatur des 20. "
            "Jahrhunderts. Sein Werk umfasst Romane wie Der Process, Das Schloss "
            "und Der Verschollene, daneben zahlreiche Erzählungen, etwa Die "
            "Verwandlung und In der Strafkolonie."
        ),
        "hypothesis": "Kafka wurde 1883 in Prag geboren.",
        "expected_band": "supported",
    },
]


# ---------------------------------------------------------------------------
# HF reference
# ---------------------------------------------------------------------------


def _resolve_label_indices(model) -> Tuple[int, int]:
    """Return ``(entail_idx, not_entail_idx)`` from the model's id2label."""

    id2label = getattr(model.config, "id2label", None) or {}
    entail = notentail = -1
    for raw_idx, label in id2label.items():
        try:
            idx = int(raw_idx)
        except (TypeError, ValueError):
            continue
        normalized = str(label).lower().replace("-", "_")
        if "not_entail" in normalized or normalized == "contradiction":
            notentail = idx
        elif "entail" in normalized:
            entail = idx
    if entail < 0 or notentail < 0:
        return 0, 1
    return entail, notentail


def _hf_pair_probs(
    model_id: str, pairs, *, device: str
) -> List[Tuple[float, float]]:
    """HF reference: tokenize each (premise, hypothesis) pair, softmax logits."""

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

    entail_idx, notentail_idx = _resolve_label_indices(model)

    results: List[Tuple[float, float]] = []
    pairs = list(pairs)
    encoded = tokenizer(
        [p["premise"] for p in pairs],
        [p["hypothesis"] for p in pairs],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        logits = model(**encoded).logits.float().cpu()
    probs = torch.softmax(logits, dim=-1).tolist()
    for row in probs:
        results.append((float(row[entail_idx]), float(row[notentail_idx])))

    del model, tokenizer, encoded, logits
    gc.collect()
    if resolved_device.type == "cuda":
        torch.cuda.empty_cache()
    return results


# ---------------------------------------------------------------------------
# vLLM /v1/classify endpoint
# ---------------------------------------------------------------------------


def _xlm_r_pair(premise: str, hypothesis: str) -> str:
    """Same pair format the in-process / HTTP providers use.

    XLM-R's tokenizer produces ``premise </s></s> hypothesis`` when
    called with ``(text, text_pair)``. vLLM's ``/v1/classify`` accepts
    a flat string per row, so we reproduce that wire format here.
    Keeping the literal ``</s></s>`` separator is critical — every
    XLM-R-based zero-shot NLI head was trained with it and dropping
    it costs ~15 balanced-accuracy points.
    """

    return f"{premise} </s></s> {hypothesis}"


def _coerce_row(row) -> List[float]:
    """Normalise the three vLLM ``/v1/classify`` row shapes to floats."""

    if isinstance(row, dict):
        scores = row.get("scores") or row.get("data") or row.get("probs")
        if scores is None:
            return [float(row.get("score", 0.0))]
        return _coerce_row(scores)
    if isinstance(row, list):
        if not row:
            return []
        if isinstance(row[0], dict):
            return [float(item.get("score", 0.0)) for item in row]
        return [float(value) for value in row]
    if isinstance(row, (int, float)):
        return [float(row)]
    raise TypeError(f"Unsupported /v1/classify row shape: {type(row)!r}")


def _endpoint_classify(
    endpoint: str, model: str, pairs
) -> List[Tuple[float, float]]:
    """Single batched POST — what production traffic actually hits.

    vLLM 0.19.0 exposes the classify head under ``/classify`` (no
    ``/v1`` prefix); the OpenAI-compatible ``/v1/classify`` alias was
    only added in 0.20+. We hit ``/classify`` directly to match the
    runtime our handler boots; production
    :class:`VllmClassifyNLIProvider` does the same.
    """

    inputs = [_xlm_r_pair(p["premise"], p["hypothesis"]) for p in pairs]
    with httpx.Client(timeout=120.0) as client:
        response = client.post(
            f"{endpoint.rstrip('/')}/classify",
            json={"model": model, "input": inputs},
        )
        response.raise_for_status()
        payload = response.json()
    rows = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise TypeError(f"Unsupported /v1/classify payload: {type(rows)!r}")
    out: List[Tuple[float, float]] = []
    for row in rows:
        probs = _coerce_row(row)
        if len(probs) < 2:
            raise ValueError(f"Expected 2-class output, got {len(probs)}")
        out.append((float(probs[0]), float(probs[1])))
    return out


def _endpoint_classify_one(
    endpoint: str, model: str, pair
) -> Tuple[float, float]:
    """One-row POST — exposes any batch-vs-single divergence."""

    return _endpoint_classify(endpoint, model, [pair])[0]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", default="MoritzLaurer/bge-m3-zeroshot-v2.0")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--prob-tol", type=float, default=1e-2)
    args = parser.parse_args()

    print(f"\nLoading HF reference: {args.model} on {args.device} ...")
    expected = _hf_pair_probs(args.model, FIXTURES, device=args.device)

    print("\n=== Batched vs HF reference ===")
    batched = _endpoint_classify(args.endpoint, args.model, FIXTURES)
    for idx, (want, got) in enumerate(zip(expected, batched)):
        delta = max(abs(a - b) for a, b in zip(want, got))
        print(
            f"  fixture[{idx}] band={FIXTURES[idx]['expected_band']:<13} "
            f"hf=(e={want[0]:.4f}, n={want[1]:.4f}) "
            f"vllm=(e={got[0]:.4f}, n={got[1]:.4f}) Δ={delta:.4f}"
        )
        if delta > args.prob_tol:
            raise AssertionError(
                f"BATCHED fixture {idx} Δ={delta:.6f} exceeds tol={args.prob_tol}"
            )

    print("\n=== Single vs HF reference ===")
    single = []
    for idx, pair in enumerate(FIXTURES):
        got = _endpoint_classify_one(args.endpoint, args.model, pair)
        single.append(got)
        delta = max(abs(a - b) for a, b in zip(expected[idx], got))
        print(f"  fixture[{idx}] Δ={delta:.4f}")
        if delta > args.prob_tol:
            raise AssertionError(
                f"SINGLE fixture {idx} Δ={delta:.6f} exceeds tol={args.prob_tol}"
            )

    print("\n=== Single vs Batched (catches attention-mask bugs) ===")
    for idx in range(len(FIXTURES)):
        delta = max(abs(a - b) for a, b in zip(single[idx], batched[idx]))
        print(f"  fixture[{idx}] Δ={delta:.4f}")
        if delta > args.prob_tol:
            raise AssertionError(
                f"SINGLE-vs-BATCHED fixture {idx} Δ={delta:.6f} "
                f"exceeds tol={args.prob_tol} — this is almost always an "
                f"attention-mask bug in the server, not a HF-vs-vLLM drift"
            )

    print("\nALL PARITY CHECKS PASSED")


if __name__ == "__main__":
    main()
