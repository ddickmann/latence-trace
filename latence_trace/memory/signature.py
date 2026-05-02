"""Span signature helpers for TRACE Memory."""

from __future__ import annotations

import hashlib
import re

from latence_trace.memory.models import SpanSignature, SpanType

_FILE_RE = re.compile(r"\b(?:[\w.-]+/)+[\w.-]+\b")
_SYMBOL_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?\b")
_NUMBER_RE = re.compile(r"\b\d+(?:[.,:]\d+)*(?:\s?(?:ms|s|%|eur|usd|gb|mb|days?|hours?))?\b", re.I)
_CODE_MEASURE_RE = re.compile(r"\b\d+(?:[.,:]\d+)*(?:\s?(?:ms|s|%|gb|mb|kb|x))\b", re.I)
_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")
_CURRENCY_RE = re.compile(r"(?:[$€£]\s?\d[\d,]*(?:\.\d+)?|\b\d[\d,]*(?:\.\d+)?\s?(?:USD|EUR|GBP)\b)")
_JSON_FIELD_RE = re.compile(r'"([A-Za-z_][A-Za-z0-9_.-]{1,80})"\s*:\s*("[^"]{1,160}"|\d[\d.,]*|true|false|null)')
_KEY_VALUE_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_.-]{1,60})\s*[=:]\s*([A-Za-z0-9_.:/@-]{2,160})")
_UUID_RE = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I)
_ISSUE_RE = re.compile(r"\b(?:issue|ticket|order|booking|reservation|payment|shipment|tool_call|call|task)[_-]?(?:id)?[:=#\s-]*([A-Za-z0-9_.-]{3,80})\b", re.I)
_TEST_RE = re.compile(r"\b(?:test_[A-Za-z0-9_]+|[A-Za-z0-9_]+Test|pytest\s+[A-Za-z0-9_./:-]+)\b")
_PYTEST_NODE_RE = re.compile(r"\b[\w./-]+\.py::[A-Za-z0-9_:.:-]+")
_RUST_SYMBOL_RE = re.compile(r"\b[A-Z][A-Za-z0-9_]*::[A-Za-z_][A-Za-z0-9_]*\b")
_STACK_RE = re.compile(r"\b(?:AssertionError|TypeError|ValueError|RuntimeError|KeyError|ImportError|Traceback|panic|segfault)\b[^\n]{0,160}")
_QUOTED_RE = re.compile(r'"([^"\n]{3,160})"|`([^`\n]{3,160})`')
_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "then",
    "than",
    "when",
    "where",
    "there",
    "their",
    "agent",
    "trace",
    "if",
    "it",
    "in",
    "consider",
    "suddenly",
    "true",
    "false",
}


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def span_id_for(text: str, span_type: SpanType) -> str:
    digest = hashlib.sha1(f"{span_type}:{normalize_text(text)}".encode()).hexdigest()
    return f"mem_{digest[:16]}"


def build_signature(text: str, span_type: SpanType) -> SpanSignature:
    normalized = normalize_text(text)
    identifiers = sorted({item for item in _SYMBOL_RE.findall(text) if "_" in item or "." in item})
    file_paths = sorted(set(_FILE_RE.findall(text)))
    symbols = sorted(
        {
            item
            for item in _SYMBOL_RE.findall(text)
            if item[:1].isupper() or "_" in item or item.endswith(("Error", "Exception"))
        }
    )
    numbers = sorted(set(_NUMBER_RE.findall(text)))
    dates = sorted(set(_DATE_RE.findall(text)))
    words = [
        word.lower()
        for word in re.findall(r"\b[A-Za-z][A-Za-z0-9_-]{5,}\b", text)
        if word.lower() not in _STOPWORDS
    ]
    rare_terms = sorted(set(words))[:24]
    hash_source = normalized
    typed_key = "|".join([span_type, *file_paths[:4], *symbols[:4], *numbers[:4]]) or span_type
    return SpanSignature(
        normalized_hash=hashlib.sha1(hash_source.encode("utf-8")).hexdigest(),
        typed_key=typed_key,
        identifiers=identifiers,
        file_paths=file_paths,
        symbols=symbols,
        numbers=numbers,
        dates=dates,
        rare_terms=rare_terms,
    )


def extract_exact_critical_terms(text: str, *, domain: str | None = None) -> list[str]:
    """Extract exact strings whose loss should count against memory quality.

    This intentionally favors precision-heavy artifacts: code paths/symbols,
    structured tool identifiers, cited RAG facts, dates/numbers/currency, and
    quoted policy or answer spans. Domain hints broaden the extractor without
    changing the low-level span signature.
    """

    domain_key = (domain or "").strip().lower()
    terms: set[str] = set()
    if domain_key in {"tool", "workflow", "tau", "tau-bench"}:
        base_regexes = [_CURRENCY_RE, _UUID_RE]
    elif domain_key in {"code", "coding", "agentic_code"}:
        base_regexes = [_FILE_RE, _UUID_RE, _CODE_MEASURE_RE]
    else:
        base_regexes = [_FILE_RE, _DATE_RE, _CURRENCY_RE, _UUID_RE]
        base_regexes.append(_NUMBER_RE)
    for regex in base_regexes:
        terms.update(match.group(0).strip() for match in regex.finditer(text))
    terms.update(_TEST_RE.findall(text))
    terms.update(match.group(0).strip() for match in _STACK_RE.finditer(text))
    for key, value in _JSON_FIELD_RE.findall(text):
        clean_value = value.strip('"')
        if _is_exact_key(key, domain_key):
            terms.add(key)
        if _is_exact_key(key, domain_key) and clean_value not in {"true", "false", "null"}:
            terms.add(clean_value)
    for key, value in _KEY_VALUE_RE.findall(text):
        if _is_exact_key(key, domain_key):
            terms.add(f"{key}={value}")
            terms.add(value)
    for match in _QUOTED_RE.finditer(text):
        quoted = (match.group(1) or match.group(2) or "").strip()
        if _is_useful_exact_phrase(quoted, domain_key):
            terms.add(quoted)
    if domain_key in {"code", "coding", "agentic_code"}:
        terms.update(_code_terms(text))
    if domain_key in {"rag", "search", "grounding", "agentic_rag"}:
        terms.update(_rag_terms(text))
    if domain_key in {"tool", "workflow", "tau", "tau-bench"}:
        terms.update(_tool_terms(text))
    return sorted(term for term in terms if _valid_term(term) and not _is_noisy_code_term(term, domain_key))


def _code_terms(text: str) -> set[str]:
    terms = set(_FILE_RE.findall(text))
    terms.update(_TEST_RE.findall(text))
    terms.update(match.group(0).split("[", 1)[0] for match in _PYTEST_NODE_RE.finditer(text))
    terms.update(_RUST_SYMBOL_RE.findall(text))
    for symbol in _SYMBOL_RE.findall(text):
        lowered = symbol.lower()
        if lowered in _STOPWORDS or len(symbol) <= 2:
            continue
        if _is_code_symbol(symbol):
            terms.add(symbol)
    return terms


def _is_code_symbol(symbol: str) -> bool:
    if symbol.endswith(("Error", "Exception")) or _is_camel_case_symbol(symbol):
        return True
    if "_" in symbol and not symbol.startswith(("_", "__")):
        return True
    if "." not in symbol:
        return False
    parts = symbol.split(".")
    if any(part[:1].isupper() for part in parts):
        return True
    if parts[0] in {"django", "astropy", "numpy", "pandas", "torch", "latence_trace", "pyflakes", "reproman"}:
        return len(parts) <= 3
    return False


def _is_camel_case_symbol(symbol: str) -> bool:
    return bool(re.match(r"[A-Z][A-Za-z0-9]*[a-z][A-Z][A-Za-z0-9]*$", symbol))


def _rag_terms(text: str) -> set[str]:
    terms: set[str] = set()
    citation_re = re.compile(r"\b(?:cite|citation|source|passage|doc|evidence)[_-]?\d{0,3}\b", re.I)
    terms.update(match.group(0) for match in citation_re.finditer(text))
    # Preserve concise policy clauses and answer-bearing spans with exact values.
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if len(sentence.split()) <= 24 and (_NUMBER_RE.search(sentence) or _DATE_RE.search(sentence)):
            terms.add(sentence.strip())
    return terms


def _tool_terms(text: str) -> set[str]:
    terms: set[str] = set()
    api_re = re.compile(r"\b(?:GET|POST|PUT|PATCH|DELETE)\s+/[A-Za-z0-9_./{}:-]+|\b[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*\b")
    terms.update(match.group(0) for match in api_re.finditer(text))
    for match in _ISSUE_RE.finditer(text):
        terms.add(match.group(1))
    return terms


def _is_exact_key(key: str, domain: str) -> bool:
    lowered = key.lower()
    tool_fragments = (
        "id",
        "uuid",
        "order",
        "booking",
        "reservation",
        "payment",
        "shipment",
        "status",
        "policy",
        "amount",
        "currency",
        "departure",
        "arrival",
        "flight",
        "api",
        "tool",
    )
    general_fragments = (
        *tool_fragments,
        "date",
        "repo",
        "file",
        "test",
        "error",
    )
    fragments = tool_fragments if domain in {"tool", "workflow", "tau", "tau-bench"} else general_fragments
    return any(fragment in lowered for fragment in fragments)


def _is_useful_exact_phrase(phrase: str, domain: str) -> bool:
    if len(phrase.split()) > 18:
        return False
    if domain in {"rag", "search", "grounding", "agentic_rag"}:
        return bool(_NUMBER_RE.search(phrase) or _DATE_RE.search(phrase) or _CURRENCY_RE.search(phrase))
    if domain in {"tool", "workflow", "tau", "tau-bench"}:
        return any(token in phrase.lower() for token in ("must", "policy", "order", "booking", "payment", "status"))
    return bool(_FILE_RE.search(phrase) or "." in phrase or "_" in phrase)


def _valid_term(term: str) -> bool:
    clean = term.strip()
    if len(clean) < 2 or len(clean) > 220:
        return False
    return clean.lower() not in _STOPWORDS


def _is_noisy_code_term(term: str, domain: str) -> bool:
    if domain not in {"code", "coding", "agentic_code"}:
        return False
    lowered = term.lower()
    return any(
        fragment in lowered
        for fragment in (
            "/noise/",
            "/generated_",
            "generated_",
            "generated ",
            "tmp_",
            "unrelated_",
            "irrelevant ",
            "site-packages",
            "/venv/",
            "/usr/local/",
            "/.heroku/",
        )
    )
