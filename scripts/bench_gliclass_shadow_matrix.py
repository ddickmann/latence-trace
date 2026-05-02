"""Offline GLiClass shadow matrix harness for TRACE customer gates.

The harness is deliberately shadow-only: it never changes runtime decisions.
It scores response chunks against context chunks, reduces the matrix into
TRACE-compatible diagnostic features, and reports whether simple candidate
guards would improve customer-adversarial gates without regressions.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from latence_trace.core.groundedness import extract_literals  # noqa: E402
from scripts.customer_breaker_smoke import BreakerCase, CASES, SUITES  # noqa: E402


PAIR_LABELS = (
    "fully_supported",
    "partially_supported",
    "contradicts_evidence",
    "adds_unsupported_fact",
    "numeric_or_unit_error",
    "unrelated",
)
PAIR_LABEL_DESCRIPTIONS = {
    "fully_supported": "the claim is fully supported by the evidence",
    "partially_supported": "the claim is only partially supported by the evidence",
    "contradicts_evidence": "the claim contradicts the evidence",
    "adds_unsupported_fact": "the claim adds an unsupported fact",
    "numeric_or_unit_error": "the claim contains a wrong number or unit",
    "unrelated": "the claim is unrelated to the evidence",
}
RULE_LABELS = (
    "follows_evidence",
    "violates_evidence",
    "insufficient_evidence",
)
RULE_LABEL_DESCRIPTIONS = {
    "follows_evidence": "the claim follows the evidence and policy rule",
    "violates_evidence": "the claim violates the evidence or policy rule",
    "insufficient_evidence": "there is insufficient evidence for the claim",
}
DEFAULT_MATRIX_MODES = ("nli-single", "rerank-single")
SUPPORTED_MATRIX_MODES = DEFAULT_MATRIX_MODES + (
    "pair-classification",
    "label-as-claim",
    "rule-following",
)

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_TOKEN_RE = re.compile(r"[\w\u00C0-\u017F]{2,}", re.UNICODE)
_NEGATION_RE = re.compile(r"\b(?:not|no|never|cannot|can't|nicht|kein|keine|without)\b", re.I)
_PASS_RE = re.compile(r"\b(?:passed|all tests passed|0 failed|improved|faster|deployed)\b", re.I)
_FAIL_RE = re.compile(r"\b(?:failed|failure|regressed|slower|not deployed|deployment skipped)\b", re.I)


@dataclass(frozen=True)
class ShadowChunk:
    index: int
    text: str
    offset_start: int | None = None
    offset_end: int | None = None


@dataclass(frozen=True)
class ThresholdCandidate:
    support_min: float
    error_max: float
    unsupported_rate: float


class ShadowScorer(Protocol):
    def classify_pairs(self, texts: Sequence[str], labels: Sequence[str]) -> list[dict[str, float]]:
        """Return one label-score map per text."""

    def score_labels(self, texts: Sequence[str], labels: Sequence[str]) -> list[dict[str, float]]:
        """Return one candidate-label score map per text."""

    def score_single_labels(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        """Return one score per ``(text, single_label)`` pair."""


class GliclassShadowScorer:
    """Thin wrapper around GLiClass' zero-shot pipeline."""

    def __init__(self, *, model_id: str, device: str, batch_size: int) -> None:
        try:
            from gliclass import GLiClassModel, ZeroShotClassificationPipeline
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on optional package
            raise SystemExit(
                "GLiClass is not installed. Install `gliclass` or rerun with "
                "`--scorer lexical-smoke` to test the harness logic."
            ) from exc

        model = GLiClassModel.from_pretrained(model_id)
        _repair_t5_encoder_embeddings(model)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._pipeline = ZeroShotClassificationPipeline(
            model,
            tokenizer,
            classification_type="multi-label",
            device=device,
        )
        self._batch_size = int(batch_size)

    def classify_pairs(self, texts: Sequence[str], labels: Sequence[str]) -> list[dict[str, float]]:
        descriptions = (
            PAIR_LABEL_DESCRIPTIONS
            if tuple(labels) == PAIR_LABELS
            else RULE_LABEL_DESCRIPTIONS if tuple(labels) == RULE_LABELS else None
        )
        prompt = (
            "Classify whether the claim is supported by, contradicted by, or unsupported by the evidence."
            if tuple(labels) == PAIR_LABELS
            else "Classify whether the claim follows the evidence and policy rule."
        )
        return self._run(texts, labels, descriptions=descriptions, prompt=prompt)

    def score_labels(self, texts: Sequence[str], labels: Sequence[str]) -> list[dict[str, float]]:
        return self._run(texts, labels)

    def score_single_labels(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        scores: list[float] = []
        # GLiClass' NLI/reranking-style use works best with one hypothesis at a
        # time, so do not batch labels together here.
        for text, label in pairs:
            row = self._pipeline([text], [label], threshold=0.0)[0]
            scores.append(float(row[0]["score"]) if row else 0.0)
        return scores

    def _run(
        self,
        texts: Sequence[str],
        labels: Sequence[str],
        *,
        descriptions: Mapping[str, str] | None = None,
        prompt: str | None = None,
    ) -> list[dict[str, float]]:
        model_labels = [descriptions.get(label, label) for label in labels] if descriptions else list(labels)
        inverse_labels = {
            descriptions.get(label, label): label for label in labels
        } if descriptions else {label: label for label in labels}
        outputs: list[dict[str, float]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            rows = self._pipeline(batch, model_labels, threshold=0.0, prompt=prompt)
            for row in rows:
                scores = {
                    inverse_labels.get(str(item["label"]), str(item["label"])): float(item["score"])
                    for item in row
                }
                outputs.append({label: float(scores.get(label, 0.0)) for label in labels})
        return outputs


def _repair_t5_encoder_embeddings(model: Any) -> bool:
    """Tie T5/MT5 encoder embeddings when GLiClass checkpoints ship shared only.

    Some GLiClass MT5 checkpoints store ``encoder_model.shared.weight`` but
    omit ``encoder_model.encoder.embed_tokens.weight``. Transformers then
    leaves ``encoder.embed_tokens`` randomly initialized, which makes every
    zero-shot score collapse around 0.5. T5-style encoders are supposed to use
    the shared embedding table, so tying the module back to ``shared.weight``
    restores discriminative scoring without changing model semantics.
    """

    encoder_model = getattr(getattr(model, "model", None), "encoder_model", None)
    shared = getattr(encoder_model, "shared", None)
    encoder = getattr(encoder_model, "encoder", None)
    embed_tokens = getattr(encoder, "embed_tokens", None)
    if shared is None or embed_tokens is None:
        return False
    shared_weight = getattr(shared, "weight", None)
    embed_weight = getattr(embed_tokens, "weight", None)
    if shared_weight is None or embed_weight is None:
        return False
    if embed_weight.data_ptr() == shared_weight.data_ptr():
        return False
    if tuple(embed_weight.shape) != tuple(shared_weight.shape):
        return False
    embed_tokens.weight = shared_weight
    return True


class LexicalSmokeScorer:
    """Deterministic local scorer used to test matrix/reduction logic.

    This is not a GLiClass substitute. It is intentionally simple and exists
    so CI can validate the shadow harness without downloading a model.
    """

    def classify_pairs(self, texts: Sequence[str], labels: Sequence[str]) -> list[dict[str, float]]:
        return [self._classify_pair(text, labels) for text in texts]

    def score_labels(self, texts: Sequence[str], labels: Sequence[str]) -> list[dict[str, float]]:
        rows: list[dict[str, float]] = []
        for text in texts:
            evidence_tokens = _tokens(text)
            label_scores: dict[str, float] = {}
            for label in labels:
                claim_tokens = _tokens(label)
                if not claim_tokens:
                    label_scores[label] = 0.0
                    continue
                label_scores[label] = len(evidence_tokens & claim_tokens) / float(len(claim_tokens))
            rows.append(label_scores)
        return rows

    def score_single_labels(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        rows = self.score_labels([text for text, _ in pairs], [label for _, label in pairs])
        scores: list[float] = []
        for row, (_, label) in zip(rows, pairs):
            scores.append(float(row.get(label, 0.0)))
        return scores

    def _classify_pair(self, text: str, labels: Sequence[str]) -> dict[str, float]:
        evidence, claim = _split_pair_text(text)
        evidence_tokens = _tokens(evidence)
        claim_tokens = _tokens(claim)
        precision = len(evidence_tokens & claim_tokens) / float(max(1, len(claim_tokens)))
        numeric_error = 1.0 if _has_numeric_or_unit_error(evidence, claim) else 0.0
        contradiction = 1.0 if _has_contradiction_cue(evidence, claim) else 0.0
        support = max(0.0, precision - 0.5 * max(numeric_error, contradiction))
        partial = min(1.0, precision + 0.15) if support < 0.72 else 0.25
        extra = max(0.0, 1.0 - precision)
        unrelated = 1.0 if precision < 0.15 else max(0.0, 0.4 - precision)
        values = {
            "fully_supported": support,
            "partially_supported": partial,
            "contradicts_evidence": contradiction,
            "adds_unsupported_fact": extra,
            "numeric_or_unit_error": numeric_error,
            "unrelated": unrelated,
            "follows_evidence": support,
            "violates_evidence": max(numeric_error, contradiction),
            "insufficient_evidence": max(extra, unrelated),
        }
        return {label: float(values.get(label, 0.0)) for label in labels}


FRESH_HARD_WAVE: tuple[BreakerCase, ...] = (
    BreakerCase(
        case_id="fresh_latency_unit_swap",
        customer_attempt="Performance unit swap 120 ms to 120 seconds",
        expected_actions=("auto_repair", "block"),
        expected_class="rag.prose.enterprise",
        payload={
            "query_text": "What is the p95 latency?",
            "raw_context": "SLO report: checkout p95 latency is 120 ms after the cache patch.",
            "response_text": "Checkout p95 latency is 120 seconds after the cache patch.",
            "corpus_type": "rag.prose.enterprise",
        },
    ),
    BreakerCase(
        case_id="fresh_pharmacy_supported_side_effect",
        customer_attempt="Supported pharmacy chatbot safety answer",
        expected_actions=("allow",),
        expected_class="rag.prose.enterprise",
        require_value_signal=False,
        payload={
            "query_text": "Can cetirizine cause drowsiness?",
            "raw_context": (
                "Pharmacy FAQ: Cetirizine can cause drowsiness in some customers. "
                "Customers should ask a pharmacist before combining it with alcohol."
            ),
            "response_text": "Cetirizine can cause drowsiness in some customers.",
            "corpus_type": "rag.prose.enterprise",
        },
    ),
    BreakerCase(
        case_id="fresh_unsupported_refund_extra_claim",
        customer_attempt="Support answer adds an unsupported manager approval claim",
        expected_actions=("auto_repair", "block"),
        expected_class="rag.prose.enterprise",
        payload={
            "query_text": "Can this opened accessory be refunded?",
            "raw_context": "Support policy: opened accessories are exchange-only.",
            "response_text": "The opened accessory can be refunded if a manager approves it.",
            "corpus_type": "rag.prose.enterprise",
        },
    ),
)

SHADOW_SUITES: dict[str, tuple[BreakerCase, ...]] = {
    **SUITES,
    "fresh_hard_wave": FRESH_HARD_WAVE,
    "gliclass_all": SUITES["all"] + FRESH_HARD_WAVE,
}


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in _TOKEN_RE.findall(text or "")}


def _split_pair_text(text: str) -> tuple[str, str]:
    evidence_match = re.search(r"Evidence:\s*(?P<evidence>.*?)(?:\nClaim:|$)", text, re.S)
    claim_match = re.search(r"Claim:\s*(?P<claim>.*)$", text, re.S)
    return (
        evidence_match.group("evidence").strip() if evidence_match else text,
        claim_match.group("claim").strip() if claim_match else "",
    )


def _normalised_literal_values(text: str) -> dict[str, set[str]]:
    buckets: dict[str, set[str]] = {}
    for literal in extract_literals(text or ""):
        buckets.setdefault(str(literal["kind"]), set()).add(str(literal["normalized"]))
    return buckets


def _has_numeric_or_unit_error(evidence: str, claim: str) -> bool:
    evidence_literals = _normalised_literal_values(evidence)
    for kind, values in _normalised_literal_values(claim).items():
        if kind not in {"number", "currency", "percent", "measurement"}:
            continue
        evidence_values = evidence_literals.get(kind, set())
        for value in values:
            if value not in evidence_values:
                return True
    return False


def _has_contradiction_cue(evidence: str, claim: str) -> bool:
    evidence_neg = bool(_NEGATION_RE.search(evidence or ""))
    claim_neg = bool(_NEGATION_RE.search(claim or ""))
    if evidence_neg != claim_neg:
        return True
    evidence_fail = bool(_FAIL_RE.search(evidence or ""))
    claim_pass = bool(_PASS_RE.search(claim or ""))
    evidence_pass = bool(_PASS_RE.search(evidence or ""))
    claim_fail = bool(_FAIL_RE.search(claim or ""))
    return (evidence_fail and claim_pass) or (evidence_pass and claim_fail)


def split_chunks(text: str, *, max_chunks: int, prefer_sentences: bool) -> list[ShadowChunk]:
    if not (text or "").strip():
        return []
    raw_parts = _SENTENCE_RE.split(text.strip()) if prefer_sentences else re.split(r"\n\s*\n", text)
    parts = [part.strip() for part in raw_parts if part and part.strip()]
    if not parts:
        parts = [text.strip()]
    chunks: list[ShadowChunk] = []
    cursor = 0
    for part in parts[:max_chunks]:
        start = text.find(part, cursor)
        if start < 0:
            start = None
            end = None
        else:
            end = start + len(part)
            cursor = end
        chunks.append(ShadowChunk(len(chunks), part, start, end))
    return chunks


def _case_payload(case: BreakerCase) -> Mapping[str, Any]:
    return case.payload


def _case_context(case: BreakerCase) -> str:
    return str(_case_payload(case).get("raw_context") or "")


def _case_response(case: BreakerCase) -> str:
    return str(_case_payload(case).get("response_text") or "")


def _case_query(case: BreakerCase) -> str:
    return str(_case_payload(case).get("query_text") or "")


def score_case(
    case: BreakerCase,
    *,
    scorer: ShadowScorer,
    matrix_modes: Sequence[str],
    max_chunks: int,
    max_sentences: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    context_chunks = split_chunks(_case_context(case), max_chunks=max_chunks, prefer_sentences=False)
    response_chunks = split_chunks(
        _case_response(case),
        max_chunks=max_sentences,
        prefer_sentences=True,
    )
    pair_matrix: list[list[dict[str, float]]] = []
    label_as_claim_matrix: list[list[float]] = []
    nli_single_matrix: list[list[float]] = []
    rerank_single_matrix: list[list[float]] = []
    rule_matrix: list[list[dict[str, float]]] = []

    if "pair-classification" in matrix_modes:
        pair_texts = [
            f"Evidence: {context.text}\nClaim: {response.text}"
            for response in response_chunks
            for context in context_chunks
        ]
        scored_pairs = scorer.classify_pairs(pair_texts, PAIR_LABELS)
        pair_matrix = _reshape(scored_pairs, rows=len(response_chunks), cols=len(context_chunks))

    if "label-as-claim" in matrix_modes and response_chunks:
        labels = [chunk.text for chunk in response_chunks]
        label_rows = scorer.score_labels([chunk.text for chunk in context_chunks], labels)
        label_as_claim_matrix = [
            [float(label_rows[col].get(response.text, 0.0)) for col in range(len(context_chunks))]
            for response in response_chunks
        ]

    if "nli-single" in matrix_modes:
        pairs = [
            (context.text, response.text)
            for response in response_chunks
            for context in context_chunks
        ]
        nli_single_matrix = _reshape(
            scorer.score_single_labels(pairs),
            rows=len(response_chunks),
            cols=len(context_chunks),
        )

    if "rerank-single" in matrix_modes:
        pairs = [
            (context.text, response.text)
            for response in response_chunks
            for context in context_chunks
        ]
        rerank_single_matrix = _reshape(
            scorer.score_single_labels(pairs),
            rows=len(response_chunks),
            cols=len(context_chunks),
        )

    if "rule-following" in matrix_modes:
        rule = _case_query(case)
        rule_texts = [
            f"Evidence: {context.text}\nPolicyRule: {rule}\nClaim: {response.text}"
            for response in response_chunks
            for context in context_chunks
        ]
        scored_rules = scorer.classify_pairs(rule_texts, RULE_LABELS)
        rule_matrix = _reshape(scored_rules, rows=len(response_chunks), cols=len(context_chunks))

    features = reduce_shadow_features(
        pair_matrix=pair_matrix,
        label_as_claim_matrix=_merge_support_matrices(
            label_as_claim_matrix,
            nli_single_matrix,
            rerank_single_matrix,
        ),
        rule_matrix=rule_matrix,
    )
    latency_ms = (time.perf_counter() - started) * 1000.0
    return {
        "case_id": case.case_id,
        "customer_attempt": case.customer_attempt,
        "expected_actions": list(case.expected_actions),
        "expected_class": case.expected_class,
        "payload": dict(case.payload),
        "context_chunks": [chunk.__dict__ for chunk in context_chunks],
        "response_chunks": [chunk.__dict__ for chunk in response_chunks],
        "matrices": {
            "pair_classification": pair_matrix,
            "label_as_claim_support": label_as_claim_matrix,
            "nli_single_support": nli_single_matrix,
            "rerank_single_support": rerank_single_matrix,
            "rule_following": rule_matrix,
        },
        "features": features,
        "latency_ms": latency_ms,
        "diagnostics": build_diagnostics(response_chunks, context_chunks, features),
    }


def _reshape(items: Sequence[Any], *, rows: int, cols: int) -> list[list[Any]]:
    matrix: list[list[Any]] = []
    cursor = 0
    for _ in range(rows):
        matrix.append(list(items[cursor : cursor + cols]))
        cursor += cols
    return matrix


def _merge_support_matrices(*matrices: Sequence[Sequence[float]]) -> list[list[float]]:
    usable = [matrix for matrix in matrices if matrix]
    if not usable:
        return []
    rows = max(len(matrix) for matrix in usable)
    cols = max((len(row) for matrix in usable for row in matrix), default=0)
    merged: list[list[float]] = []
    for row_idx in range(rows):
        row: list[float] = []
        for col_idx in range(cols):
            row.append(
                max(
                    (
                        float(matrix[row_idx][col_idx])
                        for matrix in usable
                        if row_idx < len(matrix) and col_idx < len(matrix[row_idx])
                    ),
                    default=0.0,
                )
            )
        merged.append(row)
    return merged


def reduce_shadow_features(
    *,
    pair_matrix: Sequence[Sequence[Mapping[str, float]]],
    label_as_claim_matrix: Sequence[Sequence[float]],
    rule_matrix: Sequence[Sequence[Mapping[str, float]]] | None = None,
    support_threshold: float = 0.55,
    error_relevance_floor: float = 0.25,
) -> dict[str, Any]:
    row_count = max(len(pair_matrix), len(label_as_claim_matrix), len(rule_matrix or []))
    per_response: list[dict[str, Any]] = []
    for row_idx in range(row_count):
        pair_row = list(pair_matrix[row_idx]) if row_idx < len(pair_matrix) else []
        label_row = list(label_as_claim_matrix[row_idx]) if row_idx < len(label_as_claim_matrix) else []
        rule_row = list(rule_matrix[row_idx]) if rule_matrix and row_idx < len(rule_matrix) else []
        cells = pair_row or [{} for _ in label_row or rule_row]
        support_scores: list[float] = []
        partial_scores: list[float] = []
        contradiction_scores: list[float] = []
        numeric_scores: list[float] = []
        extra_scores: list[float] = []
        for col, cell in enumerate(cells):
            pair_support = float(cell.get("fully_supported", 0.0))
            pair_partial = float(cell.get("partially_supported", 0.0))
            label_support = float(label_row[col]) if col < len(label_row) else 0.0
            rule_support = (
                float(rule_row[col].get("follows_evidence", 0.0)) if col < len(rule_row) else 0.0
            )
            relevance = max(pair_support, pair_partial, label_support, rule_support)
            support_scores.append(max(pair_support, label_support, rule_support))
            partial_scores.append(pair_partial)
            # Error labels on unrelated/distractor chunks are not actionable.
            # Gate them by relevance so a clean supported answer is not repaired
            # just because an irrelevant chunk lacks the same number or entity.
            if relevance >= error_relevance_floor:
                contradiction_scores.append(
                    max(
                        float(cell.get("contradicts_evidence", 0.0)),
                        (
                            float(rule_row[col].get("violates_evidence", 0.0))
                            if col < len(rule_row)
                            else 0.0
                        ),
                    )
                )
                numeric_scores.append(float(cell.get("numeric_or_unit_error", 0.0)))
                extra_scores.append(float(cell.get("adds_unsupported_fact", 0.0)))
        support_max = max(support_scores, default=0.0)
        top_context = _argmax(support_scores)
        per_response.append(
            {
                "response_index": row_idx,
                "gliclass_support_max": support_max,
                "gliclass_partial_max": max(partial_scores, default=0.0),
                "gliclass_contradiction_max": max(contradiction_scores, default=0.0),
                "gliclass_numeric_unit_error_max": max(numeric_scores, default=0.0),
                "gliclass_extra_claim_max": max(extra_scores, default=0.0),
                "gliclass_top_context_index": top_context,
                "unsupported": support_max < support_threshold,
            }
        )

    support_values = [float(row["gliclass_support_max"]) for row in per_response]
    unsupported_count = sum(1 for row in per_response if row["unsupported"])
    return {
        "per_response": per_response,
        "gliclass_support_min": min(support_values, default=0.0),
        "gliclass_support_mean": (
            sum(support_values) / float(len(support_values)) if support_values else 0.0
        ),
        "gliclass_unsupported_sentence_rate": (
            unsupported_count / float(len(per_response)) if per_response else 1.0
        ),
        "gliclass_contradiction_max": max(
            (float(row["gliclass_contradiction_max"]) for row in per_response),
            default=0.0,
        ),
        "gliclass_numeric_unit_error_max": max(
            (float(row["gliclass_numeric_unit_error_max"]) for row in per_response),
            default=0.0,
        ),
        "gliclass_extra_claim_max": max(
            (float(row["gliclass_extra_claim_max"]) for row in per_response),
            default=0.0,
        ),
    }


def _argmax(values: Sequence[float]) -> int | None:
    if not values:
        return None
    return max(range(len(values)), key=lambda index: values[index])


def build_diagnostics(
    response_chunks: Sequence[ShadowChunk],
    context_chunks: Sequence[ShadowChunk],
    features: Mapping[str, Any],
) -> list[dict[str, Any]]:
    diagnostics = []
    for row in features.get("per_response", []):
        response_index = int(row["response_index"])
        context_index = row.get("gliclass_top_context_index")
        diagnostics.append(
            {
                "response_index": response_index,
                "response_text": (
                    response_chunks[response_index].text
                    if response_index < len(response_chunks)
                    else ""
                ),
                "top_context_index": context_index,
                "top_context_text": (
                    context_chunks[context_index].text
                    if isinstance(context_index, int) and context_index < len(context_chunks)
                    else ""
                ),
                "support": row["gliclass_support_max"],
                "contradiction": row["gliclass_contradiction_max"],
                "numeric_or_unit_error": row["gliclass_numeric_unit_error_max"],
                "extra_claim": row["gliclass_extra_claim_max"],
            }
        )
    return diagnostics


def candidate_action(features: Mapping[str, Any], threshold: ThresholdCandidate) -> str:
    error = max(
        float(features.get("gliclass_contradiction_max", 0.0)),
        float(features.get("gliclass_numeric_unit_error_max", 0.0)),
        float(features.get("gliclass_extra_claim_max", 0.0)),
    )
    if error >= threshold.error_max:
        return "auto_repair"
    if float(features.get("gliclass_unsupported_sentence_rate", 1.0)) >= threshold.unsupported_rate:
        return "auto_repair"
    if float(features.get("gliclass_support_min", 0.0)) < threshold.support_min:
        return "auto_repair"
    return "allow"


def sweep_thresholds(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    candidates = [
        ThresholdCandidate(support, error, unsupported)
        for support in (0.35, 0.45, 0.49, 0.50, 0.51, 0.52, 0.53, 0.55, 0.65)
        for error in (0.50, 0.65, 0.80)
        for unsupported in (0.50, 0.75, 1.01)
    ]
    scored: list[dict[str, Any]] = []
    for candidate in candidates:
        decisions = []
        passed = 0
        false_blocks = 0
        false_allows = 0
        for row in rows:
            action = candidate_action(row["features"], candidate)
            ok = action in set(row["expected_actions"])
            passed += int(ok)
            if "allow" in row["expected_actions"] and action != "allow":
                false_blocks += 1
            if "allow" not in row["expected_actions"] and action == "allow":
                false_allows += 1
            decisions.append({"case_id": row["case_id"], "action": action, "passed": ok})
        scored.append(
            {
                "thresholds": candidate.__dict__,
                "passed": passed,
                "total": len(rows),
                "false_blocks": false_blocks,
                "false_allows": false_allows,
                "decisions": decisions,
            }
        )
    scored.sort(key=lambda item: (-item["passed"], item["false_allows"], item["false_blocks"]))
    return {"best": scored[0] if scored else None, "candidates": scored[:10]}


def summarize_report(rows: Sequence[Mapping[str, Any]], sweep: Mapping[str, Any] | None) -> dict[str, Any]:
    best = (sweep or {}).get("best") if sweep else None
    latency_values = [float(row.get("latency_ms", 0.0)) for row in rows]
    supported_rows = [row for row in rows if "allow" in row["expected_actions"]]
    unsafe_rows = [row for row in rows if "allow" not in row["expected_actions"]]
    decision = "keep_shadow_diagnostics_only"
    if best and best["false_allows"] == 0 and best["false_blocks"] == 0:
        decision = "promote_to_narrow_guard_candidate_after_heldout_validation"
    elif best and best["false_allows"] == 0:
        decision = "keep_shadow_and_collect_more_supported_cases"
    return {
        "case_count": len(rows),
        "supported_case_count": len(supported_rows),
        "unsafe_case_count": len(unsafe_rows),
        "latency_ms_mean": sum(latency_values) / float(len(latency_values)) if latency_values else 0.0,
        "latency_ms_p95": _percentile(latency_values, 0.95),
        "integration_decision": decision,
        "production_wiring_allowed": False,
        "production_wiring_blocker": (
            "Shadow results must pass no-regression gates and held-out customer-style validation first."
        ),
    }


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = min(len(sorted_values) - 1, int(math.ceil(q * len(sorted_values)) - 1))
    return float(sorted_values[index])


def _parse_csv(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in (raw or "").split(",") if part.strip())


def _select_cases(suites: Sequence[str]) -> tuple[BreakerCase, ...]:
    cases: list[BreakerCase] = []
    for suite in suites:
        if suite not in SHADOW_SUITES:
            raise SystemExit(
                "unknown suite {suite!r}; choices: {choices}".format(
                    suite=suite,
                    choices=", ".join(sorted(SHADOW_SUITES)),
                )
            )
        cases.extend(SHADOW_SUITES[suite])
    return tuple(cases)


def build_scorer(args: argparse.Namespace) -> ShadowScorer:
    if args.scorer == "lexical-smoke":
        return LexicalSmokeScorer()
    return GliclassShadowScorer(
        model_id=args.model,
        device=args.device,
        batch_size=args.batch_size,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    matrix_modes = _parse_csv(args.matrix_mode)
    invalid_modes = set(matrix_modes) - set(SUPPORTED_MATRIX_MODES)
    if invalid_modes:
        raise SystemExit(f"unsupported matrix mode(s): {', '.join(sorted(invalid_modes))}")
    scorer = build_scorer(args)
    cases = _select_cases(_parse_csv(args.suite))
    rows = [
        score_case(
            case,
            scorer=scorer,
            matrix_modes=matrix_modes,
            max_chunks=args.max_chunks,
            max_sentences=args.max_sentences,
        )
        for case in cases
    ]
    sweep = sweep_thresholds(rows) if args.threshold_sweep else None
    report = {
        "purpose": "gliclass_shadow_matrix_no_production_wiring",
        "model": args.model if args.scorer == "gliclass" else "lexical-smoke",
        "scorer": args.scorer,
        "matrix_modes": list(matrix_modes),
        "suites": list(_parse_csv(args.suite)),
        "rows": rows,
        "threshold_sweep": sweep,
        "summary": summarize_report(rows, sweep),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="knowledgator/gliclass-multilang-ultra")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--suite", default="customer_breaker,vertical_pilot")
    parser.add_argument("--matrix-mode", default=",".join(DEFAULT_MATRIX_MODES))
    parser.add_argument("--max-chunks", type=int, default=16)
    parser.add_argument("--max-sentences", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threshold-sweep", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--scorer",
        choices=("gliclass", "lexical-smoke"),
        default="gliclass",
        help="Use lexical-smoke only to validate harness logic without GLiClass installed.",
    )
    args = parser.parse_args()
    report = run(args)
    summary = report["summary"]
    print(
        "scored {count} cases; p95_latency_ms={latency:.2f}; decision={decision}".format(
            count=summary["case_count"],
            latency=summary["latency_ms_p95"],
            decision=summary["integration_decision"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
