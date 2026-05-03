"""Span extraction for TRACE Memory."""

from __future__ import annotations

import json
import re

from latence_trace.memory.models import SpanRecord, SpanType
from latence_trace.memory.signature import (
    build_signature,
    extract_exact_critical_terms,
    span_id_for,
)
from latence_trace.memory.types import classify_span

_CODE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d .()/-]{6,}\d)(?!\d)")
_EMAIL_RE = re.compile(r"\b([A-Za-z0-9._%+-]{3,120})@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_ISO_DATE_FULL_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
_LOW_AUTHORITY_MARKERS = (
    "generated noise",
    "noise:",
    "noisy transcript",
    "old draft",
    "superseded",
    "marked superseded",
    "preliminary spreadsheet",
    "unrelated stack",
)


def _token_count(text: str) -> int:
    return len(text.split())


def _sentence_spans(text: str) -> list[str]:
    spans = [item.strip() for item in _SENTENCE_SPLIT_RE.split(text) if item.strip()]
    return spans or ([text.strip()] if text.strip() else [])


def extract_spans(
    *,
    turn_text: str = "",
    response_text: str | None = None,
    raw_context: str | None = None,
    query_text: str | None = None,
    turn_index: int = 0,
    memory_domain: str | None = None,
    source_pointer: dict | None = None,
) -> list[SpanRecord]:
    records: list[SpanRecord] = []
    domain = (memory_domain or "").strip().lower()
    sources = [
        ("query", query_text or ""),
        ("turn", turn_text or ""),
        ("response", response_text or ""),
        ("raw_context", raw_context or ""),
    ]
    for source, text in sources:
        original_text = text
        sanitized_text = _sanitize_sensitive_text(text)
        if not sanitized_text.strip():
            continue
        should_index = source in {"raw_context", "response"} or (source == "query" and domain == "code")
        if (
            should_index
            and not (source == "raw_context" and domain == "chat")
            and not _is_low_authority_exact_source(sanitized_text)
        ):
            index_text = original_text if domain == "tool" else sanitized_text
            for fact in _structured_exact_spans(index_text, domain=domain):
                records.append(
                    _make_record(
                        fact,
                        source=f"exact_index_{source}",
                        turn_index=turn_index,
                        span_type=_index_span_type(domain),
                        source_pointer=source_pointer,
                    )
                )
        consumed: set[str] = set()
        for block in _CODE_BLOCK_RE.findall(sanitized_text):
            consumed.add(block)
            records.append(
                _make_record(block, source=source, turn_index=turn_index, source_pointer=source_pointer)
            )
        remainder = sanitized_text
        for block in consumed:
            remainder = remainder.replace(block, "\n")
        for sentence in _sentence_spans(remainder):
            if _is_noise_sentence(sentence):
                continue
            if _token_count(sentence) < 3 and source != "response":
                continue
            records.append(
                _make_record(
                    sentence,
                    source=source,
                    turn_index=turn_index,
                    memory_domain=domain,
                    source_pointer=source_pointer,
                )
            )
    return records


def _make_record(
    text: str,
    *,
    source: str,
    turn_index: int,
    span_type: SpanType | None = None,
    memory_domain: str | None = None,
    source_pointer: dict | None = None,
) -> SpanRecord:
    span_type = span_type or _classify_span(text, source=source, memory_domain=memory_domain)
    signature = build_signature(text, span_type)
    provenance = {"source": source, "turn_index": turn_index}
    if source_pointer:
        provenance["source_pointer"] = {**source_pointer, "field": source}
    return SpanRecord(
        id=span_id_for(text, span_type),
        text=text.strip(),
        span_type=span_type,
        signature=signature,
        token_count=_token_count(text),
        created_turn=turn_index,
        last_seen_turn=turn_index,
        source=source,
        provenance=provenance,
    )


def _classify_span(text: str, *, source: str, memory_domain: str | None) -> SpanType:
    if memory_domain == "tool" and source == "raw_context":
        return "tool_result"
    return classify_span(text, source=source)


def _structured_exact_spans(text: str, *, domain: str) -> list[str]:
    if domain == "tool":
        return _structured_tool_facts(text)
    if domain in {"code", "rag", "grounding", "search", "chat"}:
        return _domain_exact_indexes(text, domain=domain)
    return []


def _sanitize_sensitive_text(text: str) -> str:
    """Mask obvious direct identifiers before memory spans are persisted."""

    text = _EMAIL_RE.sub("[email]", text)
    return _PHONE_RE.sub(_mask_phone_match, text)


def _mask_phone_match(match: re.Match[str]) -> str:
    value = match.group(0)
    if _ISO_DATE_FULL_RE.fullmatch(value):
        return value
    return "[phone]"


def _is_low_authority_exact_source(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _LOW_AUTHORITY_MARKERS)


def _is_noise_sentence(text: str) -> bool:
    lowered = text.lower()
    if "do not" in lowered or "never" in lowered:
        return False
    return any(marker in lowered for marker in _LOW_AUTHORITY_MARKERS)


def _domain_exact_indexes(text: str, *, domain: str, max_terms: int = 120) -> list[str]:
    if domain == "code":
        max_terms = 40
    elif domain in {"rag", "search"}:
        max_terms = 24
    elif domain == "grounding":
        max_terms = 40
    elif domain == "chat":
        max_terms = 16
    terms = extract_exact_critical_terms(text, domain=domain)
    if domain == "chat":
        clean = text.strip().strip("\"'")
        if 1 <= len(clean.split()) <= 18 and len(clean) <= 160:
            terms = sorted({clean, *terms})
    if not terms:
        return []
    prefix = f"{domain}_exact_index"
    indexes = []
    for start in range(0, len(terms), max_terms):
        chunk = terms[start : start + max_terms]
        indexes.append(prefix + " " + " ".join(chunk))
    return indexes


def _index_span_type(domain: str) -> SpanType:
    if domain == "tool":
        return "tool_result"
    if domain == "code":
        return "code_symbol"
    if domain in {"rag", "grounding", "search"}:
        return "retrieval_chunk"
    return "evidence"


def _structured_tool_facts(text: str) -> list[str]:
    payloads = _json_payloads(text)
    facts: list[str] = []
    facts.extend(_tool_exact_indexes(text))
    for payload in payloads:
        facts.extend(_flatten_tool_payload(payload))
    facts.extend(_regex_tool_facts(text))
    return sorted({fact for fact in facts if fact})


def _tool_exact_indexes(text: str, *, max_terms: int = 80) -> list[str]:
    terms = extract_exact_critical_terms(text, domain="tool")
    if not terms:
        return []
    indexes = []
    for start in range(0, len(terms), max_terms):
        chunk = terms[start : start + max_terms]
        indexes.append("tool_exact_index " + " ".join(chunk))
    return indexes


def _json_payloads(text: str) -> list[object]:
    stripped = text.strip()
    if not stripped:
        return []
    payloads: list[object] = []
    candidates = [stripped, *[line.strip() for line in stripped.splitlines() if line.strip()]]
    for candidate in candidates:
        if not candidate.startswith(("{", "[")):
            continue
        try:
            payloads.append(json.loads(candidate))
        except json.JSONDecodeError:
            continue
    return payloads


def _flatten_tool_payload(payload: object, path: str = "") -> list[str]:
    facts: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            child_path = f"{path}.{key}" if path else str(key)
            facts.extend(_flatten_tool_payload(value, child_path))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            child_path = f"{path}[{index}]" if path else f"item[{index}]"
            facts.extend(_flatten_tool_payload(value, child_path))
    elif "email" in path.lower() and isinstance(payload, str):
        facts.extend(f"tool_fact email_localpart={match.group(1)}" for match in _EMAIL_RE.finditer(payload))
    elif _is_tool_fact(path, payload):
        facts.append(f"tool_fact {path}={_clean_scalar(payload)}")
    return facts


def _regex_tool_facts(text: str) -> list[str]:
    facts = []
    for token in re.split(r"\s+", text.replace(",", " ")):
        if "=" not in token:
            continue
        key, value = token.strip(".;").split("=", 1)
        if _is_tool_fact(key, value):
            facts.append(f"tool_fact {key}={value}")
    for key, value in re.findall(r'"?([A-Za-z_][A-Za-z0-9_.-]{1,80})"?\s*[:=]\s*"?([A-Za-z0-9_./@:-]{2,160})"?', text):
        if _is_tool_fact(key, value):
            facts.append(f"tool_fact {key}={value}")
    return facts


def _is_tool_fact(path: str, value: object) -> bool:
    if value is None or isinstance(value, bool):
        return False
    scalar = _clean_scalar(value)
    if len(scalar) < 2 or len(scalar) > 120:
        return False
    lowered = path.lower()
    important_keys = (
        "id",
        "reservation",
        "booking",
        "order",
        "payment",
        "shipment",
        "status",
        "policy",
        "amount",
        "currency",
        "flight",
        "origin",
        "destination",
        "cabin",
        "seat",
        "tracking",
        "method",
    )
    if any(key in lowered for key in important_keys):
        return True
    return bool(re.fullmatch(r"[A-Z0-9]{5,12}", scalar))


def _clean_scalar(value: object) -> str:
    return str(value).strip().replace(" ", "_")
