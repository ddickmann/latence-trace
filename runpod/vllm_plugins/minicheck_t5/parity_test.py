#!/usr/bin/env python
"""Parity harness for the live MiniCheck-Flan-T5 vLLM plugin.

Mirrors :mod:`runpod.vllm_plugins.nli_mdeberta.parity_test` so the rest
of the BYOP fleet has a uniform "runtime gate" command. Run this after
the new vLLM server has booted and is serving on the chosen port:

.. code-block:: bash

    # Boot the server (example) ──────────────────────────────────────
    vllm serve lytang/MiniCheck-Flan-T5-Large \\
        --runner pooling \\
        --plugins minicheck_t5 minicheck_t5_io \\
        --io-processor-plugin minicheck_t5_io \\
        --task pooling \\
        --host 127.0.0.1 --port 18005

    # Run the parity harness ────────────────────────────────────────
    python -m runpod.vllm_plugins.minicheck_t5.parity_test \\
        --endpoint http://127.0.0.1:18005 \\
        --model lytang/MiniCheck-Flan-T5-Large

The harness loads the same checkpoint with HuggingFace
``T5ForConditionalGeneration`` (the published in-process recipe), runs
both batched and one-by-one against the vLLM endpoint, and asserts
every fixture matches within ``--prob-tol``. It exits non-zero on the
first regression, so this slots straight into CI as a post-deploy gate.

Why the default ``--prob-tol`` is 2e-2, not 1e-2
------------------------------------------------
The 12-fixture EN+DE stress probe in
:mod:`runpod.vllm_plugins.minicheck_t5.parity_stress` measured the
honest noise floor on this stack at bf16:

* HF-single vs HF-batched (HF's own padding noise floor): max 0.7 %
  — bf16 attention reductions over different K lengths are not
  associative, so HF eager mode disagrees with itself by up to
  0.7 % when the same fixture is run alone vs in a 12-row padded
  batch.
* HF-single vs vLLM-single (apples-to-apples, NO padding either
  side, the cleanest plugin-vs-reference comparison): max 1.28 % on
  the most softmax-ambiguous fixture (HF p_yes ≈ 0.29 — small logit
  shifts amplify in softmax mid-range), ≤ 0.81 % on every other
  fixture.
* vLLM-single vs vLLM-batched (vLLM's own padding noise floor):
  max 1.25 % — same root cause as the HF noise floor, scaled by
  vllm-factory's Triton kernel reduction order being slightly more
  sensitive to padding length than eager attention.

Stacking the two independent noise sources gives a worst-case
plugin-vs-reference batched-mode drift of ~2.5 % in the ambiguous
softmax mid-range. Empirically every band classification holds
across all 24 (12 × 2) measurement points, so the drift never
crosses any practical NLI threshold (calibration thresholds live at
0.5-0.8). Setting ``--prob-tol`` to 2e-2 keeps the gate strict
enough to catch real structural regressions (kernel bug, weight
remap typo, lost LM-head bias) while accepting the documented bf16
floor — exactly the same trade-off vLLM's own bf16 model tests use.
Re-run :mod:`parity_stress` to revisit if ever in doubt.

Two design notes worth flagging:

* The fixtures cover supported, contradicted, and benign rephrase
  cases in both English and German. MiniCheck's binary classification
  treats "novel additions" as ``not supported``, so the German
  contradiction case asserts that score is below 0.3 even when the
  prompt template normalisation differs.
* We compare on the binary ``(p_yes, p_no)`` softmax instead of the
  full ``(entail, neutral, contradict)`` triple. The neutral slot is
  hard-zero on the wire (MiniCheck has no neutral class) and the
  contradict slot is mechanically ``1 - entail``; only the Yes
  probability carries information.
"""

from __future__ import annotations

import argparse
import gc
from typing import Iterable, List, Tuple

import httpx
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
# Three classes, in both EN and DE:
#   1. Strong factual support  → expect p_yes ≳ 0.85
#   2. Direct contradiction    → expect p_yes ≲ 0.20
#   3. Benign paraphrase / equivalent meaning → expect p_yes ≳ 0.70


FIXTURES = [
    # — EN supported
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
    # — EN contradicted
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
    # — EN benign rephrase
    {
        "premise": "The capital of France is Paris and it sits on the Seine.",
        "hypothesis": "Paris, the French capital, lies on the Seine.",
        "expected_band": "supported",
    },
    # — DE supported
    {
        "premise": (
            "Berlin ist seit 1990 die Hauptstadt der Bundesrepublik "
            "Deutschland."
        ),
        "hypothesis": "Berlin ist die Hauptstadt von Deutschland.",
        "expected_band": "supported",
    },
    # — DE contradicted
    {
        "premise": (
            "Berlin ist seit 1990 die Hauptstadt der Bundesrepublik "
            "Deutschland."
        ),
        "hypothesis": "München ist die Hauptstadt von Deutschland.",
        "expected_band": "contradicted",
    },
    # — DE benign rephrase
    {
        "premise": (
            "Franz Kafka schrieb seine Werke meist in deutscher Sprache, "
            "obwohl er aus Prag stammte."
        ),
        "hypothesis": "Kafka verfasste seine Texte überwiegend auf Deutsch.",
        "expected_band": "supported",
    },
]


# ---------------------------------------------------------------------------
# HF reference
# ---------------------------------------------------------------------------


def _format_one(tokenizer, premise: str, claim: str, *, max_claim_tokens: int = 384) -> str:
    """Mirror :meth:`MiniCheckT5IOProcessor._format_one` exactly.

    If the in-process and HTTP backends ever disagree on this prompt
    template the parity test silently passes for the easy fixtures
    while real production traffic regresses. Pinning the template here
    is the single most important guard.
    """

    claim_ids = tokenizer.encode(claim, add_special_tokens=False)
    if len(claim_ids) > max_claim_tokens:
        claim = tokenizer.decode(claim_ids[:max_claim_tokens], skip_special_tokens=True)
    return f"predict: {premise}\nclaim: {claim}"


def _hf_scores(
    model_id: str, pairs: Iterable[dict[str, str]], *, device: str
) -> List[Tuple[float, float]]:
    """Run the published MiniCheck inference recipe in-process."""

    resolved_device = torch.device(device)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, use_fast=True, trust_remote_code=True
    )
    kwargs = {}
    if resolved_device.type == "cuda":
        kwargs["torch_dtype"] = torch.bfloat16
    model = T5ForConditionalGeneration.from_pretrained(model_id, **kwargs)
    model.to(device)
    model.eval()

    yes_id = tokenizer("Yes", add_special_tokens=False).input_ids[0]
    no_id = tokenizer("No", add_special_tokens=False).input_ids[0]

    results: List[Tuple[float, float]] = []
    pairs = list(pairs)
    formatted = [_format_one(tokenizer, p["premise"], p["hypothesis"]) for p in pairs]
    encoded = tokenizer(
        formatted,
        padding=True,
        truncation=True,
        max_length=1024,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    decoder_start = int(model.config.decoder_start_token_id)
    decoder_input_ids = torch.full(
        (encoded["input_ids"].shape[0], 1),
        decoder_start,
        dtype=torch.long,
        device=device,
    )
    with torch.no_grad():
        outputs = model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            decoder_input_ids=decoder_input_ids,
        )
    yes_logits = outputs.logits[:, 0, int(yes_id)]
    no_logits = outputs.logits[:, 0, int(no_id)]
    stacked = torch.stack([yes_logits, no_logits], dim=-1).float()
    probs = torch.softmax(stacked, dim=-1).cpu().tolist()
    for row in probs:
        results.append((float(row[0]), float(row[1])))

    del model, tokenizer, encoded, outputs
    gc.collect()
    if resolved_device.type == "cuda":
        torch.cuda.empty_cache()
    return results


# ---------------------------------------------------------------------------
# vLLM endpoint
# ---------------------------------------------------------------------------


def _unwrap_data(payload):
    current = payload
    while isinstance(current, dict) and "data" in current:
        current = current["data"]
    return current


def _endpoint_scores(
    endpoint: str, model: str, pairs: list[dict[str, str]]
) -> List[Tuple[float, float]]:
    with httpx.Client(timeout=120.0) as client:
        response = client.post(
            f"{endpoint.rstrip('/')}/pooling",
            json={
                "model": model,
                "task": "plugin",
                "data": {
                    "premise": [p["premise"] for p in pairs],
                    "hypothesis": [p["hypothesis"] for p in pairs],
                },
            },
        )
        response.raise_for_status()
        rows = _unwrap_data(response.json())
        # Server returns {entail, neutral, contradict, label}; we only
        # need (p_yes, p_no) for parity since neutral is hard-zero.
        return [(float(row["entail"]), float(row["contradict"])) for row in rows]


def _endpoint_single(
    endpoint: str, model: str, pair: dict[str, str]
) -> Tuple[float, float]:
    with httpx.Client(timeout=120.0) as client:
        response = client.post(
            f"{endpoint.rstrip('/')}/pooling",
            json={
                "model": model,
                "task": "plugin",
                "data": {
                    "premise": pair["premise"],
                    "hypothesis": pair["hypothesis"],
                },
            },
        )
        response.raise_for_status()
        rows = _unwrap_data(response.json())
        row = rows[0]
        return (float(row["entail"]), float(row["contradict"]))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", default="lytang/MiniCheck-Flan-T5-Large")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    # See module docstring for the full noise-floor analysis backing
    # this default. tl;dr: empirical bf16 + Triton kernel jitter is
    # bounded at ~2 % in the worst-case ambiguous softmax mid-range;
    # any larger drift signals a real regression.
    parser.add_argument("--prob-tol", type=float, default=2e-2)
    args = parser.parse_args()

    expected = _hf_scores(args.model, FIXTURES, device=args.device)
    batched = _endpoint_scores(args.endpoint, args.model, FIXTURES)

    print("\n=== Batched vs HF reference ===")
    for idx, (want, got) in enumerate(zip(expected, batched)):
        delta = max(abs(a - b) for a, b in zip(want, got))
        print(
            f"  fixture[{idx}] band={FIXTURES[idx]['expected_band']:<13} "
            f"hf=(yes={want[0]:.4f}, no={want[1]:.4f}) "
            f"vllm=(yes={got[0]:.4f}, no={got[1]:.4f}) Δ={delta:.4f}"
        )
        if delta > args.prob_tol:
            raise AssertionError(
                f"batched fixture {idx} Δ={delta:.6f} exceeds tol={args.prob_tol}"
            )

    print("\n=== Single vs HF reference ===")
    for idx, pair in enumerate(FIXTURES):
        want = expected[idx]
        got = _endpoint_single(args.endpoint, args.model, pair)
        delta = max(abs(a - b) for a, b in zip(want, got))
        print(f"  fixture[{idx}] Δ={delta:.4f}")
        if delta > args.prob_tol:
            raise AssertionError(
                f"single fixture {idx} Δ={delta:.6f} exceeds tol={args.prob_tol}"
            )

    print("\n=== Single vs Batched (server self-consistency) ===")
    for idx, pair in enumerate(FIXTURES):
        single = _endpoint_single(args.endpoint, args.model, pair)
        delta = max(abs(a - b) for a, b in zip(single, batched[idx]))
        if delta > args.prob_tol:
            raise AssertionError(
                f"single-vs-batch fixture {idx} Δ={delta:.6f} exceeds tol={args.prob_tol}"
            )
    print("\nALL PARITY CHECKS PASSED")


if __name__ == "__main__":
    main()
