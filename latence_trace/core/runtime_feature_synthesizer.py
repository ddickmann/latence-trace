"""Server-side runtime-head feature synthesis for bare TRACE requests.

The promoted runtime heads still prefer caller-supplied feature maps. This
module fills the gap for product surfaces where a user sends only text by
deriving conservative, auditable features from the scored response.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class SynthesizedRuntimeFeatures:
    features: Optional[Dict[str, float]]
    source: str
    missing_groups: list[str] = field(default_factory=list)


_SENTENCE_RE = re.compile(r"[.!?]+\s+|[.!?]+$")
_IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
_CODE_CUE_RE = re.compile(
    r"\b(?:def|class|function|method|pytest|test_|import|sdk|api|client|patch|"
    r"diff|timeout|retry|benchmark|p95|latency|before|after|improved|"
    r"regressed|slower|faster|performance)\b|[A-Za-z_][A-Za-z0-9_]*\([^)]*\)",
    re.IGNORECASE,
)
_COMMAND_CUE_RE = re.compile(
    r"\b(?:pytest|npm test|go test|cargo test|last command|verified|benchmark)\b",
    re.IGNORECASE,
)
_PERFORMANCE_CUE_RE = re.compile(
    r"\b(?:benchmark|p95(?:_ms)?|latency|before\s*[=:]|after\s*[=:]|ms|"
    r"improved|regressed|slower|faster|performance)\b",
    re.IGNORECASE,
)
_PERFORMANCE_VALUE_RE = re.compile(
    r"\b(?P<label>before|after|p95(?:_ms)?|latency)\s*[=:]\s*"
    r"(?P<value>\d+(?:\.\d+)?)\s*(?:ms|milliseconds?)?\b",
    re.IGNORECASE,
)
_PERFORMANCE_TO_VALUE_RE = re.compile(
    r"\b(?:to|at|down\s+to|up\s+to)\s+(?P<value>\d+(?:\.\d+)?)\s*(?:ms|milliseconds?)\b",
    re.IGNORECASE,
)
_IMPROVEMENT_CUE_RE = re.compile(
    r"\b(?:improved|improvement|faster|reduced|lower(?:ed)?|down\s+to)\b",
    re.IGNORECASE,
)
_REGRESSION_CUE_RE = re.compile(
    r"\b(?:regressed|regression|slower|increased|higher|worse|up\s+to)\b",
    re.IGNORECASE,
)
_PASS_CUE_RE = re.compile(
    r"\b(?:passed|pass(?:es|ed)?|succeeded|success(?:ful|fully)?|0\s+failed|all\s+tests\s+passed|verified)\b",
    re.IGNORECASE,
)
_FAIL_CUE_RE = re.compile(
    r"\b(?:failed|failing|failure|errored|error|timed\s*out|non[-\s]?zero|[1-9]\d*\s+failed)\b",
    re.IGNORECASE,
)
_BENIGN_FAILURE_CUE_RE = re.compile(
    r"\b(?:0\s+failed|0\s+failures?|no\s+failures?|no\s+errors?|without\s+errors?)\b",
    re.IGNORECASE,
)
_DEPLOYED_CUE_RE = re.compile(r"\b(?:deployed|released|rolled\s*out|shipped)\b", re.IGNORECASE)
_DEPLOY_SKIPPED_CUE_RE = re.compile(
    r"\b(?:deployment\s+(?:skipped|not\s+attempted)|deploy(?:ment)?\s+skipped|not\s+deployed|no\s+deployment)\b",
    re.IGNORECASE,
)


def synthesize_runtime_features(request: Any, response: Any) -> SynthesizedRuntimeFeatures:
    """Return a feature map for promoted heads when it is safe to synthesize one."""

    class_key = _class_key(response)
    if not class_key or class_key == "rag.prose.enterprise":
        return SynthesizedRuntimeFeatures(None, "missing", ["head_does_not_require_synthesized_features"])
    query = _text(request, "query_text")
    context = _text(request, "raw_context")
    response_text = _text(request, "response_text")
    if class_key == "code.agentic_trace":
        return _synthesize_trajectory(query=query, context=context, response=response_text, scored=response)
    if class_key == "rag.structured":
        return _synthesize_structured(query=query, context=context, response=response_text, scored=response)
    if class_key == "rag.code_in_context":
        return _synthesize_code_context(query=query, context=context, response=response_text, scored=response)
    if class_key in {"rag.prose.short_factoid", "rag.prose.multi_claim"}:
        return SynthesizedRuntimeFeatures(_generic_trace_features(query, context, response_text, response), "synthesized", [])
    return SynthesizedRuntimeFeatures(None, "missing", [f"unsupported_class:{class_key}"])


def _class_key(response: Any) -> Optional[str]:
    route = getattr(response, "corpus_route", None)
    value = getattr(route, "corpus_type", None)
    return str(value) if value else None


def _text(obj: Any, name: str) -> str:
    value = getattr(obj, name, None)
    return "" if value is None else str(value)


def _score(response: Any, name: str, default: float = 0.0) -> float:
    scores = getattr(response, "scores", None)
    value = getattr(scores, name, None) if scores is not None else None
    if value is None and name == "groundedness_v2":
        value = getattr(scores, "primary_score", None) if scores is not None else None
    try:
        return float(default if value is None else value)
    except (TypeError, ValueError):
        return float(default)


def _bounded(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _sentence_count(text: str) -> int:
    parts = [part for part in _SENTENCE_RE.split(text or "") if part.strip()]
    return max(1, len(parts)) if text.strip() else 0


def _identifier_overlap(context: str, response: str) -> float:
    ctx = {tok for tok in _IDENT_RE.findall(context or "") if len(tok) > 2}
    rsp = {tok for tok in _IDENT_RE.findall(response or "") if len(tok) > 2}
    if not rsp:
        return 1.0
    return len(ctx & rsp) / max(1, len(rsp))


def _coverage_ratio(response: Any) -> float:
    return _bounded(_score(response, "context_coverage_ratio", _score(response, "groundedness_v2", 0.0)))


def _generic_trace_features(query: str, context: str, response_text: str, response: Any) -> Dict[str, float]:
    grounded = _bounded(_score(response, "groundedness_v2", _score(response, "primary_score", 0.0)))
    nli = _bounded(_score(response, "nli_aggregate", grounded))
    reverse = _bounded(_score(response, "reverse_context", _score(response, "primary_score", grounded)))
    literal = _bounded(_score(response, "literal_guarded", min(grounded, reverse)))
    coverage = _coverage_ratio(response)
    unused = _bounded(_score(response, "context_unused_ratio", 1.0 - coverage))
    dead = _bounded(_score(response, "dead_weight_ratio", unused))
    token_mean = _bounded(max(grounded, nli, reverse))
    token_bottom = _bounded(min(token_mean, literal, coverage))
    unsupported = _bounded(1.0 - max(nli, literal, coverage))
    claim_count = float(max(1, _sentence_count(response_text)))
    identifier = _identifier_overlap(context, response_text)
    return {
        "v1_score": grounded,
        "v1_nli_aggregate": nli,
        "reverse_context": reverse,
        "groundedness_v2": grounded,
        "literal_guarded": literal,
        "literal_mismatch_count": float(max(0, round((1.0 - literal) * 10))),
        "context_coverage_ratio": coverage,
        "context_unused_ratio": unused,
        "dead_weight_ratio": dead,
        "token_mean": token_mean,
        "token_bottom10": token_bottom,
        "token_saturation_rate": 0.0 if token_bottom > 0.25 else 1.0,
        "calibrated_mean": grounded,
        "nli_token_mean": nli,
        "literal_coverage": literal,
        "atom_match": _bounded((literal + nli + coverage) / 3.0),
        "claim_count": claim_count,
        "unsupported_claim_fraction": unsupported,
        "schema_match": 1.0 if _looks_structured(context) else 0.0,
        "api_symbol_match": identifier,
        "trajectory_order_match": 1.0 if not query else identifier,
        "test_outcome_match": 1.0 if _COMMAND_CUE_RE.search(f"{context}\n{response_text}") else 0.0,
        "identifier_coverage": identifier,
        "identifier_count": float(len(_IDENT_RE.findall(response_text or ""))),
        "numeric_coverage": _numeric_overlap(context, response_text),
        "coverage_label_mean": coverage,
        "support_unit_label_mean": coverage,
        "min_claim_coverage": _bounded(1.0 - unsupported),
    }


def _looks_structured(context: str) -> bool:
    stripped = (context or "").lstrip()
    return stripped.startswith(("{", "[")) or "|" in context


def _numeric_overlap(context: str, response: str) -> float:
    numbers = re.findall(r"\d+(?:[,.]\d+)*", response or "")
    if not numbers:
        return 1.0
    normalised_context = (context or "").replace(",", "")
    hits = sum(1 for number in numbers if number.replace(",", "") in normalised_context)
    return hits / max(1, len(numbers))


def _normalise_metric_number(value: str) -> str:
    number = value.replace(",", "")
    if "." not in number:
        return number
    return number.rstrip("0").rstrip(".")


def _performance_label_values(text: str) -> dict[str, list[str]]:
    values: dict[str, list[str]] = {}
    for match in _PERFORMANCE_VALUE_RE.finditer(text or ""):
        label = match.group("label").lower()
        if label == "p95":
            label = "p95_ms"
        values.setdefault(label, []).append(_normalise_metric_number(match.group("value")))
    return values


def _context_performance_values(text: str) -> set[str]:
    values = set()
    for items in _performance_label_values(text).values():
        values.update(items)
    return values


def _response_performance_target_values(text: str) -> set[str]:
    values = set()
    labelled = _performance_label_values(text)
    for label in ("after", "p95_ms", "latency"):
        values.update(labelled.get(label, []))
    for match in _PERFORMANCE_TO_VALUE_RE.finditer(text or ""):
        values.add(_normalise_metric_number(match.group("value")))
    return values


def _performance_outcome_alignment(context: str, response: str) -> Optional[float]:
    if not (_PERFORMANCE_CUE_RE.search(context or "") or _PERFORMANCE_CUE_RE.search(response or "")):
        return None
    context_values = _performance_label_values(context)
    before_values = context_values.get("before", [])
    after_values = context_values.get("after", []) or context_values.get("p95_ms", [])
    response_text = response or ""
    response_claims_improvement = bool(_IMPROVEMENT_CUE_RE.search(response_text))
    response_claims_regression = bool(_REGRESSION_CUE_RE.search(response_text))

    if before_values and after_values:
        try:
            before = float(before_values[-1])
            after = float(after_values[-1])
            actually_improved = after < before
            actually_regressed = after > before
        except ValueError:
            actually_improved = False
            actually_regressed = False
        if response_claims_improvement and not actually_improved:
            return 0.0
        if response_claims_regression and not actually_regressed:
            return 0.0

    context_numeric_values = _context_performance_values(context)
    response_targets = _response_performance_target_values(response)
    if context_numeric_values and response_targets:
        if not response_targets <= context_numeric_values:
            return 0.0
    return 1.0


def _performance_value_alignment(context: str, response: str) -> float:
    if _performance_outcome_alignment(context, response) != 1.0:
        return 0.0
    context_values = _performance_label_values(context)
    before_values = set(context_values.get("before", []))
    after_values = set(context_values.get("after", []) or context_values.get("p95_ms", []))
    if not before_values or not after_values:
        return 0.0
    response_targets = _response_performance_target_values(response)
    if not response_targets:
        return 0.0
    context_numeric_values = _context_performance_values(context)
    if not response_targets <= context_numeric_values:
        return 0.0
    if not response_targets & after_values:
        return 0.0
    return 1.0


def _agentic_outcome_alignment(context: str, response: str) -> float:
    """Return 0 when the response contradicts explicit trace outcomes."""

    context_text = _BENIGN_FAILURE_CUE_RE.sub("", context or "")
    response_text = _BENIGN_FAILURE_CUE_RE.sub("", response or "")
    context_failed = bool(_FAIL_CUE_RE.search(context_text))
    context_passed = bool(_PASS_CUE_RE.search(context or "")) and not context_failed
    response_claims_passed = bool(_PASS_CUE_RE.search(response or ""))
    response_claims_failed = bool(_FAIL_CUE_RE.search(response_text))
    if context_failed and response_claims_passed:
        return 0.0
    if context_passed and response_claims_failed:
        return 0.0
    if _DEPLOY_SKIPPED_CUE_RE.search(context or "") and _DEPLOYED_CUE_RE.search(response or ""):
        return 0.0
    performance_alignment = _performance_outcome_alignment(context, response)
    if performance_alignment is not None:
        return performance_alignment
    return 1.0


def _synthesize_structured(query: str, context: str, response: str, scored: Any) -> SynthesizedRuntimeFeatures:
    if not _looks_structured(context):
        return SynthesizedRuntimeFeatures(None, "missing", ["structured_source_missing"])
    numeric = _numeric_overlap(context, response)
    identifier = _identifier_overlap(context, response)
    if numeric < 0.5 and identifier < 0.5:
        return SynthesizedRuntimeFeatures(None, "partial", ["structured_value_alignment_missing"])
    features = _generic_trace_features(query, context, response, scored)
    # The promoted structured head was calibrated on typed-cell features where
    # safe table answers often have weak prose scores but strong schema/value
    # provenance. Preserve that shape instead of over-trusting narrative NLI.
    features.update(
        {
            "v1_score": 0.0,
            "reverse_context": 0.0,
            "groundedness_v2": 0.0,
            "literal_guarded": 0.0,
            "claim_count": float(max(2, _sentence_count(response))),
            "schema_match": identifier,
            "atom_match": max(numeric, identifier),
            "numeric_coverage": numeric,
            "row_alignment": identifier,
            "column_alignment": identifier,
            "value_alignment": numeric,
            "unit_alignment": 1.0,
            "date_alignment": 1.0,
            "numeric_tolerance_match": numeric,
            "schema_alias_match": identifier,
            "cell_provenance_match": max(numeric, identifier),
        }
    )
    return SynthesizedRuntimeFeatures(features, "synthesized", [])


def _synthesize_code_context(query: str, context: str, response: str, scored: Any) -> SynthesizedRuntimeFeatures:
    code_cues = len(_CODE_CUE_RE.findall(f"{query}\n{context}\n{response}"))
    if code_cues < 2:
        return SynthesizedRuntimeFeatures(None, "missing", ["code_symbol_evidence_missing"])
    features = _generic_trace_features(query, context, response, scored)
    identifier = _identifier_overlap(context, response)
    features.update(
        {
            "identifier_coverage": identifier,
            "api_symbol_match": identifier,
            "identifier_count": float(len(_IDENT_RE.findall(response or ""))),
            "trajectory_order_match": 1.0,
            "test_outcome_match": 1.0 if _COMMAND_CUE_RE.search(f"{context}\n{response}") else 0.0,
        }
    )
    return SynthesizedRuntimeFeatures(features, "synthesized", [])


def _synthesize_trajectory(query: str, context: str, response: str, scored: Any) -> SynthesizedRuntimeFeatures:
    joined = f"{query}\n{context}\n{response}"
    performance_evidence = bool(_PERFORMANCE_CUE_RE.search(joined))
    if len(_CODE_CUE_RE.findall(joined)) < 3:
        return SynthesizedRuntimeFeatures(None, "missing", ["trajectory_code_evidence_missing"])
    command_match = 1.0 if (_COMMAND_CUE_RE.search(joined) or performance_evidence) else 0.0
    outcome_alignment = _agentic_outcome_alignment(context, response)
    performance_value_alignment = _performance_value_alignment(context, response)
    identifier = _identifier_overlap(context, response)
    evidence_alignment = max(identifier, 0.95 if performance_value_alignment >= 1.0 else 0.0)
    grounded = _bounded(_score(scored, "groundedness_v2", _score(scored, "primary_score", 0.0)))
    reverse = _bounded(_score(scored, "reverse_context", grounded))
    coverage = _coverage_ratio(scored)
    dead = _bounded(_score(scored, "dead_weight_ratio", 1.0 - coverage))
    uncertain = _bounded(_score(scored, "context_uncertain_ratio", 0.0))
    semantic_evidence = max(grounded, reverse) if outcome_alignment >= 1.0 else min(grounded, reverse)
    if performance_value_alignment >= 1.0:
        semantic_evidence = max(semantic_evidence, 0.95)
        reverse = max(reverse, 0.95)
        coverage = max(coverage, 0.95)
        dead = min(dead, 0.05)
    query_alignment = _identifier_overlap(query, response)
    if performance_value_alignment >= 1.0:
        query_alignment = max(query_alignment, 0.85)
    good = _bounded((evidence_alignment + coverage + command_match + outcome_alignment + semantic_evidence) / 5.0)
    features = {
        "file_alignment": evidence_alignment,
        "symbol_alignment": evidence_alignment,
        "test_outcome_alignment": outcome_alignment if command_match else 0.0,
        "patch_alignment": max(evidence_alignment, command_match) if outcome_alignment >= 1.0 else min(evidence_alignment, command_match),
        "temporal_order_alignment": command_match if outcome_alignment >= 1.0 else 0.0,
        "claim_atom_coverage": good,
        "unsupported_atom_rate": _bounded(1.0 - good),
        "phantom_symbol_rate": _bounded(1.0 - evidence_alignment),
        "missing_command_evidence": 1.0 - command_match,
        "literal_match_rate": evidence_alignment,
        "literal_mismatch_rate": _bounded(1.0 - evidence_alignment),
        "identifier_query_overlap": query_alignment,
        "identifier_query_absent_rate": _bounded(1.0 - query_alignment),
        "warning_identifier_rate": _bounded(1.0 - evidence_alignment),
        "reverse_context": reverse,
        "consensus_hardened": semantic_evidence,
        "groundedness_v2": semantic_evidence,
        "triangular": semantic_evidence,
        "context_attribution_ratio": coverage,
        "context_uncertain_ratio": uncertain,
        "dead_weight_ratio": dead,
        "support_usage_rate": coverage,
        "context_token_log": math.log1p(max(0, len((context or "").split()))),
        "multi_cell_reverse_min": semantic_evidence,
        "multi_cell_reverse_max": semantic_evidence,
        "multi_cell_reverse_std": 0.0 if outcome_alignment >= 1.0 else abs(reverse - grounded),
    }
    missing = [] if command_match else ["trajectory_command_or_test_evidence_missing"]
    source = "synthesized" if not missing else "partial"
    return SynthesizedRuntimeFeatures(features if source == "synthesized" else None, source, missing)

