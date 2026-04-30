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
    r"diff|timeout|retry)\b|[A-Za-z_][A-Za-z0-9_]*\([^)]*\)",
    re.IGNORECASE,
)
_COMMAND_CUE_RE = re.compile(r"\b(?:pytest|npm test|go test|cargo test|last command|verified)\b", re.IGNORECASE)


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
    if len(_CODE_CUE_RE.findall(joined)) < 3:
        return SynthesizedRuntimeFeatures(None, "missing", ["trajectory_code_evidence_missing"])
    command_match = 1.0 if _COMMAND_CUE_RE.search(joined) else 0.0
    identifier = _identifier_overlap(context, response)
    grounded = _bounded(_score(scored, "groundedness_v2", _score(scored, "primary_score", 0.0)))
    reverse = _bounded(_score(scored, "reverse_context", grounded))
    coverage = _coverage_ratio(scored)
    dead = _bounded(_score(scored, "dead_weight_ratio", 1.0 - coverage))
    uncertain = _bounded(_score(scored, "context_uncertain_ratio", 0.0))
    good = _bounded((identifier + coverage + command_match + max(grounded, reverse)) / 4.0)
    features = {
        "file_alignment": identifier,
        "symbol_alignment": identifier,
        "test_outcome_alignment": command_match,
        "patch_alignment": max(identifier, command_match),
        "temporal_order_alignment": command_match,
        "claim_atom_coverage": good,
        "unsupported_atom_rate": _bounded(1.0 - good),
        "phantom_symbol_rate": _bounded(1.0 - identifier),
        "missing_command_evidence": 1.0 - command_match,
        "literal_match_rate": identifier,
        "literal_mismatch_rate": _bounded(1.0 - identifier),
        "identifier_query_overlap": _identifier_overlap(query, response),
        "identifier_query_absent_rate": _bounded(1.0 - _identifier_overlap(query, response)),
        "warning_identifier_rate": _bounded(1.0 - identifier),
        "reverse_context": reverse,
        "consensus_hardened": grounded,
        "groundedness_v2": grounded,
        "triangular": grounded,
        "context_attribution_ratio": coverage,
        "context_uncertain_ratio": uncertain,
        "dead_weight_ratio": dead,
        "support_usage_rate": coverage,
        "context_token_log": math.log1p(max(0, len((context or "").split()))),
        "multi_cell_reverse_min": min(reverse, grounded),
        "multi_cell_reverse_max": max(reverse, grounded),
        "multi_cell_reverse_std": abs(reverse - grounded),
    }
    missing = [] if command_match else ["trajectory_command_or_test_evidence_missing"]
    source = "synthesized" if not missing else "partial"
    return SynthesizedRuntimeFeatures(features if source == "synthesized" else None, source, missing)

