"""Held-out precision gate for the unused-context tri-state classifier.

This script builds a deterministic support-unit benchmark with four case
families per topic:

1. exact used unit + unrelated distractors
2. near-duplicate sibling that must stay ``uncertain``
3. mixed packed window that must abstain to ``uncertain``
4. low-overlap paraphrase rescued by NLI + reranker

It grid-searches the tri-state usage-classifier thresholds on a calibration
split, then evaluates the selected thresholds on a held-out split.

It is a hard precision gate:

* Precision is never synthesised from ``predicted_unused == 0``; an empty
  prediction is reported as precision 0.0 and ``unused_precision_defined=False``.
* ``select_thresholds`` only accepts candidates that actually emit ``unused``
  labels AND hit the ``target_unused_precision`` floor.
* The held-out split must satisfy ``unused_precision >= target_unused_precision``
  AND ``predicted_unused > 0`` or the script exits non-zero with a clear
  ``FAILED`` banner and records a ``gate_status: "failed"`` reason in the JSON
  report. On success the report carries ``gate_status: "passed"``.

Usage:
    python scripts/bench_unused_context_precision.py
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))

from latence_trace.core.groundedness import (  # noqa: E402
    SupportUnitInput,
    UsageClassifierThresholds,
    _build_response_chunks,
    apply_support_unit_usage_classification,
    encode_texts,
    partition_support_units,
    score_groundedness_response_chunked,
    tokenize_text,
)


_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    split: str
    response_text: str
    support_texts: List[str]
    gold_states: List[str]
    nli_entailed_pairs: List[tuple[str, str]] = field(default_factory=list)
    reranker_pairs: List[tuple[str, str]] = field(default_factory=list)


@dataclass
class PreparedCase:
    case: BenchmarkCase
    support_units: List[SupportUnitInput]
    result: Dict[str, Any]


class _HashEncoder:
    model_name = "unused-context-bench-hash"

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim
        self._seed = 31

    def tokenize(self, text: str) -> List[str]:
        return [token for token in _TOKEN_RE.findall(text) if token.strip()]

    def _vec(self, token: str) -> np.ndarray:
        key = token.lower()
        digest = hashlib.sha1(f"{self._seed}:{key}".encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "big", signed=False)
        rng = np.random.default_rng(seed)
        vec = np.zeros((self.dim,), dtype=np.float32)
        first = int(rng.integers(0, self.dim))
        second = int(rng.integers(0, self.dim))
        vec[first] = 1.0
        vec[second] = max(vec[second], 0.5)
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 1e-9 else vec

    def encode(self, texts, **_kwargs):
        texts = [texts] if isinstance(texts, str) else list(texts)
        out: List[np.ndarray] = []
        for text in texts:
            tokens = self.tokenize(text) or ["<empty>"]
            out.append(
                np.stack([self._vec(token) for token in tokens], axis=0).astype(np.float32)
            )
        return out


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", (text or "").lower())).strip()


class _MappingNLIProvider:
    def __init__(self, entailed_pairs: Sequence[tuple[str, str]]) -> None:
        self._entailed_pairs = {
            (_normalize_text(premise), _normalize_text(hypothesis))
            for premise, hypothesis in entailed_pairs
        }

    def entail(self, premises, hypotheses):
        out = []
        for premise, hypothesis in zip(premises, hypotheses):
            key = (_normalize_text(premise), _normalize_text(hypothesis))
            if key in self._entailed_pairs:
                out.append((0.94, 0.05, 0.01))
            else:
                out.append((0.08, 0.62, 0.30))
        return out


class _MappingReranker:
    def __init__(self, relevant_pairs: Sequence[tuple[str, str]]) -> None:
        self._relevant_pairs = {
            (_normalize_text(claim), _normalize_text(premise))
            for claim, premise in relevant_pairs
        }

    def score(self, claim: str, candidate_premises):
        claim_key = _normalize_text(claim)
        return [
            1.0 if (claim_key, _normalize_text(candidate)) in self._relevant_pairs else 0.0
            for candidate in candidate_premises
        ]


def _make_support_units_from_texts(
    provider: _HashEncoder,
    texts: Sequence[str],
) -> List[SupportUnitInput]:
    embeddings = encode_texts(provider, list(texts), is_query=False)
    units: List[SupportUnitInput] = []
    for idx, (text, embedding) in enumerate(zip(texts, embeddings)):
        tokens = tokenize_text(provider, text, expected_len=int(embedding.shape[0]))
        units.append(
            SupportUnitInput(
                support_id=f"unit-{idx}",
                text=text,
                embeddings=embedding,
                tokens=tokens,
                source_mode="raw_context",
                offset_start=0,
                offset_end=len(text),
            )
        )
    return units


def build_benchmark_cases() -> List[BenchmarkCase]:
    generic_noise_a = "Lanterns glow above the quiet harbor at dusk."
    generic_noise_b = "Ancient stone bridges span the river near the village square."
    topics = [
        {
            "name": "berlin",
            "split": "calibration",
            "anchor": "Berlin is the capital of Germany.",
            "paraphrase": "Germany's capital city is Berlin.",
            "mixed_noise": (
                "The museum district has winding streets, seasonal river cruises, "
                "restored observatories, and many travel details the response never mentions."
            ),
            "nli_support": "Germany's administrative center is Berlin.",
            "nli_response": "Berlin serves as the nation's seat of government.",
            "distractor": "Penguins are flightless aquatic birds native to Antarctica.",
        },
        {
            "name": "coffee",
            "split": "calibration",
            "anchor": "Coffee contains caffeine.",
            "paraphrase": "Caffeine is present in coffee.",
            "mixed_noise": (
                "Roasting notes mention caramel aromas, citrus brightness, and a "
                "long tasting finish that the answer never discusses."
            ),
            "nli_support": "Coffee includes caffeine as a stimulant compound.",
            "nli_response": "The drink carries a stimulant called caffeine.",
            "distractor": "Mars has two small moons named Phobos and Deimos.",
        },
        {
            "name": "python",
            "split": "calibration",
            "anchor": "Python is dynamically typed.",
            "paraphrase": "Python uses dynamic typing.",
            "mixed_noise": (
                "The release notes also mention packaging tweaks, CI jobs, and "
                "editor integrations that the answer leaves untouched."
            ),
            "nli_support": "Python determines types at runtime.",
            "nli_response": "The language resolves variable types while the program runs.",
            "distractor": "Oak trees can live for several centuries.",
        },
        {
            "name": "vaccine",
            "split": "calibration",
            "anchor": "The vaccine requires two doses.",
            "paraphrase": "Two shots are needed for the vaccine.",
            "mixed_noise": (
                "Clinic logistics include appointment reminders, parking maps, and "
                "waiting-room signage that the answer does not mention."
            ),
            "nli_support": "Immunization is completed only after a pair of doses.",
            "nli_response": "Full protection comes after receiving the shot twice.",
            "distractor": "Whales communicate with low-frequency calls.",
        },
        {
            "name": "saturn",
            "split": "heldout",
            "anchor": "Saturn has rings made of ice.",
            "paraphrase": "Saturn's rings are composed of ice.",
            "mixed_noise": (
                "The observatory brochure also lists parking rules, telescope hours, "
                "and cafe menus that the answer never references."
            ),
            "nli_support": "Ice particles make up Saturn's ring system.",
            "nli_response": "The planet's ring bands consist largely of frozen material.",
            "distractor": "Bamboo grows quickly in warm climates.",
        },
        {
            "name": "curie",
            "split": "heldout",
            "anchor": "Marie Curie won the Nobel Prize in 1903.",
            "paraphrase": "Marie Curie received the 1903 Nobel Prize.",
            "mixed_noise": (
                "The archive also catalogs lecture schedules, train routes, and "
                "museum gift-shop inventory that the answer ignores."
            ),
            "nli_support": "Marie Curie was awarded a Nobel Prize in 1903.",
            "nli_response": "Curie earned Nobel recognition in 1903.",
            "distractor": "Coral reefs host diverse marine ecosystems.",
        },
        {
            "name": "meeting",
            "split": "heldout",
            "anchor": "The meeting starts at 9 AM.",
            "paraphrase": "The meeting begins at 9 in the morning.",
            "mixed_noise": (
                "Agenda notes also cover lunch catering, hallway signage, and seat "
                "assignments that the answer never mentions."
            ),
            "nli_support": "The session begins at 9 AM.",
            "nli_response": "The session kicks off at nine in the morning.",
            "distractor": "Maple syrup is made from tree sap.",
        },
        {
            "name": "ev",
            "split": "heldout",
            "anchor": "Electric cars use batteries.",
            "paraphrase": "Electric vehicles run on battery power.",
            "mixed_noise": (
                "The brochure also describes paint options, showroom lighting, and "
                "coffee-bar coupons that the answer never mentions."
            ),
            "nli_support": "Battery packs power electric vehicles.",
            "nli_response": "These cars are driven by stored electrical energy.",
            "distractor": "Saffron is a spice made from crocus flowers.",
        },
    ]

    cases: List[BenchmarkCase] = []
    for topic in topics:
        cases.append(
            BenchmarkCase(
                name=f"{topic['name']}_exact",
                split=topic["split"],
                response_text=topic["anchor"],
                support_texts=[topic["anchor"], topic["distractor"], generic_noise_a],
                gold_states=["used", "unused", "unused"],
            )
        )
        cases.append(
            BenchmarkCase(
                name=f"{topic['name']}_near_duplicate",
                split=topic["split"],
                response_text=topic["anchor"],
                support_texts=[topic["anchor"], topic["paraphrase"], topic["distractor"]],
                gold_states=["used", "uncertain", "unused"],
            )
        )
        cases.append(
            BenchmarkCase(
                name=f"{topic['name']}_mixed_window",
                split=topic["split"],
                response_text=topic["anchor"],
                support_texts=[f"{topic['anchor']} {topic['mixed_noise']}", generic_noise_b],
                gold_states=["uncertain", "unused"],
            )
        )
        cases.append(
            BenchmarkCase(
                name=f"{topic['name']}_nli_rescue",
                split=topic["split"],
                response_text=topic["nli_response"],
                support_texts=[topic["nli_support"], topic["distractor"]],
                gold_states=["used", "unused"],
                nli_entailed_pairs=[(topic["nli_support"], topic["nli_response"])],
                reranker_pairs=[(topic["nli_response"], topic["nli_support"])],
            )
        )
    return cases


def _prepare_case(case: BenchmarkCase, provider: _HashEncoder) -> PreparedCase:
    support_units = _make_support_units_from_texts(provider, case.support_texts)
    support_batches = partition_support_units(
        support_units,
        batch_size=max(1, len(support_units)),
    )
    response_chunks = _build_response_chunks(
        case.response_text,
        provider=provider,
        chunk_token_budget=32,
        encode_fn=encode_texts,
    )
    nli_provider = (
        _MappingNLIProvider(case.nli_entailed_pairs)
        if case.nli_entailed_pairs
        else None
    )
    reranker = (
        _MappingReranker(case.reranker_pairs)
        if case.reranker_pairs
        else None
    )
    result = score_groundedness_response_chunked(
        response_chunks=response_chunks,
        support_batches=support_batches,
        response_text=case.response_text,
        evidence_limit=8,
        primary_metric="reverse_context",
        coverage_threshold=0.5,
        nli_provider=nli_provider,
        nli_reranker=reranker,
        nli_top_k_premises=1 if nli_provider is not None else None,
        nli_concat_premises=False if nli_provider is not None else None,
        nli_use_atomic_claims=False if nli_provider is not None else None,
    )
    return PreparedCase(case=case, support_units=support_units, result=result)


def _evaluate_cases(
    prepared_cases: Sequence[PreparedCase],
    thresholds: UsageClassifierThresholds,
) -> Dict[str, Any]:
    total = 0
    correct = 0
    predicted_unused = 0
    gold_unused = 0
    true_unused = 0
    confusion: Dict[str, Dict[str, int]] = {}
    emitted_unused_examples: List[Dict[str, Any]] = []

    for prepared in prepared_cases:
        support_units_payload = copy.deepcopy(prepared.result["support_units"])
        apply_support_unit_usage_classification(
            support_units_payload=support_units_payload,
            support_inputs=prepared.support_units,
            coverage_threshold=0.5,
            claim_records=(
                prepared.result.get("nli_diagnostics", {}).get("claims")
                if prepared.result.get("nli_diagnostics")
                else None
            ),
            thresholds=thresholds,
        )
        predicted = [payload["usage_state"] for payload in support_units_payload]
        for unit_index, (gold, pred) in enumerate(
            zip(prepared.case.gold_states, predicted)
        ):
            total += 1
            if gold == pred:
                correct += 1
            confusion.setdefault(gold, {})
            confusion[gold][pred] = confusion[gold].get(pred, 0) + 1
            if gold == "unused":
                gold_unused += 1
            if pred == "unused":
                predicted_unused += 1
                if gold == "unused":
                    true_unused += 1
                emitted_unused_examples.append(
                    {
                        "case": prepared.case.name,
                        "unit_index": unit_index,
                        "gold": gold,
                        "pred": pred,
                        "text": prepared.case.support_texts[unit_index],
                    }
                )

    unused_precision = (
        float(true_unused) / float(predicted_unused)
        if predicted_unused > 0
        else 0.0
    )
    unused_recall = (
        float(true_unused) / float(gold_unused)
        if gold_unused > 0
        else 0.0
    )
    return {
        "total_units": total,
        "accuracy": float(correct) / float(total) if total > 0 else 0.0,
        "unused_precision": unused_precision,
        "unused_precision_defined": bool(predicted_unused > 0),
        "unused_recall": unused_recall,
        "unused_recall_defined": bool(gold_unused > 0),
        "predicted_unused": predicted_unused,
        "gold_unused": gold_unused,
        "true_unused": true_unused,
        "confusion": confusion,
        "emitted_unused_examples": emitted_unused_examples[:12],
    }


def _threshold_grid() -> Sequence[UsageClassifierThresholds]:
    grid = []
    for strong_coverage_min, unused_coverage_max, support_token_ratio_min, redundancy_overlap_min in itertools.product(
        [0.58, 0.62, 0.66],
        [0.16, 0.18, 0.22, 0.26],
        [0.18, 0.20, 0.24],
        [0.70, 0.75, 0.80],
    ):
        grid.append(
            UsageClassifierThresholds(
                strong_coverage_min=strong_coverage_min,
                unused_coverage_max=unused_coverage_max,
                support_token_ratio_min=support_token_ratio_min,
                redundancy_overlap_min=redundancy_overlap_min,
            )
        )
    return grid


def select_thresholds(
    prepared_cases: Sequence[PreparedCase],
    *,
    target_unused_precision: float,
) -> tuple[UsageClassifierThresholds, Dict[str, Any]]:
    baseline = UsageClassifierThresholds()
    best_thresholds: UsageClassifierThresholds | None = None
    best_metrics: Dict[str, Any] | None = None
    best_key: tuple[Any, ...] | None = None
    for thresholds in _threshold_grid():
        metrics = _evaluate_cases(prepared_cases, thresholds)
        meets_target = (
            bool(metrics["unused_precision_defined"])
            and metrics["predicted_unused"] > 0
            and metrics["unused_precision"] >= target_unused_precision
        )
        distance_to_baseline = (
            abs(thresholds.strong_coverage_min - baseline.strong_coverage_min)
            + abs(thresholds.unused_coverage_max - baseline.unused_coverage_max)
            + abs(thresholds.support_token_ratio_min - baseline.support_token_ratio_min)
            + abs(thresholds.redundancy_overlap_min - baseline.redundancy_overlap_min)
        )
        key = (
            1 if meets_target else 0,
            1 if metrics["predicted_unused"] > 0 else 0,
            metrics["unused_precision"],
            metrics["accuracy"],
            metrics["unused_recall"],
            -distance_to_baseline,
        )
        if best_key is None or key > best_key:
            best_key = key
            best_thresholds = thresholds
            best_metrics = metrics
    assert best_thresholds is not None
    assert best_metrics is not None
    best_metrics = dict(best_metrics)
    best_metrics["meets_target"] = (
        bool(best_metrics.get("unused_precision_defined"))
        and int(best_metrics.get("predicted_unused", 0)) > 0
        and float(best_metrics.get("unused_precision", 0.0))
        >= float(target_unused_precision)
    )
    return best_thresholds, best_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(
            _HERE.parent.parent
            / "research"
            / "triangular_maxsim"
            / "reports"
            / "unused_context_precision.json"
        ),
    )
    parser.add_argument("--target-unused-precision", type=float, default=0.90)
    args = parser.parse_args()
    target_precision = float(args.target_unused_precision)

    provider = _HashEncoder()
    cases = build_benchmark_cases()
    prepared = [_prepare_case(case, provider) for case in cases]
    calibration = [case for case in prepared if case.case.split == "calibration"]
    heldout = [case for case in prepared if case.case.split == "heldout"]

    selected_thresholds, calibration_metrics = select_thresholds(
        calibration,
        target_unused_precision=target_precision,
    )
    heldout_metrics = _evaluate_cases(heldout, selected_thresholds)
    default_metrics = _evaluate_cases(heldout, UsageClassifierThresholds())

    calibration_meets_target = bool(calibration_metrics.get("meets_target", False))

    gate_reasons: List[str] = []
    if not bool(heldout_metrics.get("unused_precision_defined")):
        gate_reasons.append(
            "held-out produced zero `unused` predictions; precision undefined"
        )
    if int(heldout_metrics.get("predicted_unused", 0)) == 0:
        gate_reasons.append("held-out predicted_unused == 0")
    if float(heldout_metrics.get("unused_precision", 0.0)) < target_precision:
        gate_reasons.append(
            "held-out unused_precision={:.3f} < target {:.3f}".format(
                float(heldout_metrics.get("unused_precision", 0.0)),
                target_precision,
            )
        )
    gate_passed = len(gate_reasons) == 0

    report = {
        "target_unused_precision": target_precision,
        "gate_status": "passed" if gate_passed else "failed",
        "gate_reasons": gate_reasons,
        "calibration_cases": len(calibration),
        "heldout_cases": len(heldout),
        "selected_thresholds": selected_thresholds.__dict__,
        "selected_thresholds_meets_target_on_calibration": calibration_meets_target,
        "calibration": calibration_metrics,
        "heldout": heldout_metrics,
        "heldout_default_thresholds": default_metrics,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("== Unused-context precision benchmark ==")
    print(f"  Target unused precision : {target_precision:.3f}")
    print(f"  Calibration cases       : {len(calibration)}")
    print(f"  Held-out cases          : {len(heldout)}")
    print("  Selected thresholds:")
    for key, value in selected_thresholds.__dict__.items():
        print(f"    {key}: {value}")
    print()
    calibration_banner = "MEETS target" if calibration_meets_target else "MISSES target"
    print(
        "  Calibration {banner}: unused precision={prec:.3f} (defined={defined}) "
        "recall={rec:.3f} accuracy={acc:.3f} predicted_unused={pu}".format(
            banner=calibration_banner,
            prec=calibration_metrics["unused_precision"],
            defined=bool(calibration_metrics.get("unused_precision_defined")),
            rec=calibration_metrics["unused_recall"],
            acc=calibration_metrics["accuracy"],
            pu=calibration_metrics["predicted_unused"],
        )
    )
    print(
        "  Held-out    : unused precision={prec:.3f} (defined={defined}) "
        "recall={rec:.3f} accuracy={acc:.3f} predicted_unused={pu}".format(
            prec=heldout_metrics["unused_precision"],
            defined=bool(heldout_metrics.get("unused_precision_defined")),
            rec=heldout_metrics["unused_recall"],
            acc=heldout_metrics["accuracy"],
            pu=heldout_metrics["predicted_unused"],
        )
    )
    print(
        "  Held-out default thresholds: unused precision={prec:.3f} (defined={defined}) "
        "recall={rec:.3f} accuracy={acc:.3f} predicted_unused={pu}".format(
            prec=default_metrics["unused_precision"],
            defined=bool(default_metrics.get("unused_precision_defined")),
            rec=default_metrics["unused_recall"],
            acc=default_metrics["accuracy"],
            pu=default_metrics["predicted_unused"],
        )
    )
    print(f"  Report written to {out_path}")
    print()
    if gate_passed:
        print(
            "  GATE PASSED: held-out unused precision {:.3f} >= target {:.3f}.".format(
                float(heldout_metrics["unused_precision"]),
                target_precision,
            )
        )
        return

    banner = "=" * 72
    print(banner)
    print("  GATE FAILED: unused-context precision target not met on held-out split.")
    for reason in gate_reasons:
        print(f"    - {reason}")
    print(banner)
    sys.exit(1)


if __name__ == "__main__":
    main()
