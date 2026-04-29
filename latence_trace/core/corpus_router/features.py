"""Handcrafted feature extractor for the corpus-type classifier.

All features are cheap, regex / string operations so the full vector is
produced in well under 3 ms on CPU (the router's latency budget).

Every feature is deterministic and depends only on the request payload.
The classifier training pipeline (``scripts/train_corpus_classifier.py``)
uses the same function so training and serving see identical vectors.

Feature order is locked: changing the length or meaning of ``feature_names``
requires retraining and a version bump on the persisted joblib bundle
(see :mod:`latence_trace.core.corpus_router.classifier`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Schema version for the serialized classifier bundle. Bump if feature
# semantics change so older joblib artefacts refuse to load.
FEATURE_SCHEMA_VERSION = 1

# Pre-compiled regexes - avoid per-call compilation overhead.
_FENCE_RE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)
_JSON_LINE_RE = re.compile(r'^\s*[\[{]')
_JSON_HINT_RE = re.compile(r'"[^"]+"\s*:\s*(?:"[^"]*"|-?\d+(?:\.\d+)?|true|false|null)')
_YAML_LINE_RE = re.compile(r"^\s*[A-Za-z_][A-Za-z0-9_\-]*:\s")
_TABLE_ROW_RE = re.compile(r"^\s*\|[^\n]+\|\s*$", re.MULTILINE)
_FILE_HEADER_RE = re.compile(r"^===\s+.+?\s+===$", re.MULTILINE)
_CODE_FILE_EXT_RE = re.compile(
    r"\.(?:py|ts|tsx|js|jsx|go|rs|java|kt|swift|rb|cpp|cc|h|hpp|cs|sh|yml|yaml|toml|json|sql|md|txt)\b",
    re.IGNORECASE,
)
_DEF_RE = re.compile(r"\b(?:def|class|function|fn|impl|interface|trait|struct|enum|pub\s+fn)\b")
_DIFF_HEADER_RE = re.compile(r"^(?:diff --git|@@|\+\+\+|---)\s", re.MULTILINE)
_SENTENCE_RE = re.compile(r"[.!?]+\s+|[.!?]+$")

# Very small English stopword set used for the ASCII / language heuristic.
_EN_STOP = frozenset(
    {
        "the", "a", "an", "and", "or", "but", "of", "in", "on", "at", "to", "for",
        "from", "with", "by", "as", "is", "are", "was", "were", "be", "been", "being",
        "has", "have", "had", "do", "does", "did", "this", "that", "these", "those",
        "it", "its", "they", "them", "we", "you", "i", "he", "she",
    }
)

FEATURE_NAMES: Tuple[str, ...] = (
    "response_len_tokens_log",           # log1p of whitespace token count
    "response_sentence_count_log",
    "response_punct_density",
    "response_fenced_code_blocks",
    "response_fenced_max_nonblank_lines",
    "response_has_diff_headers",
    "response_def_class_hits",
    "response_brace_density",
    "response_backtick_density",
    "context_len_chars_log",
    "context_json_hint_density",
    "context_yaml_hint_density",
    "context_table_row_hits",
    "context_file_header_count",          # === path === markers
    "context_code_ext_hits",              # .py/.ts/... in context
    "context_has_multiple_chunks",
    "query_len_tokens_log",
    "response_ascii_ratio",
    "response_en_stopword_ratio",
    "response_numeric_token_ratio",
)


@dataclass(frozen=True)
class FeatureVector:
    names: Sequence[str]
    values: Sequence[float]

    def as_dict(self) -> Dict[str, float]:
        return {k: float(v) for k, v in zip(self.names, self.values)}


def _log1p_int(n: int) -> float:
    import math
    return float(math.log1p(max(0, n)))


def _token_count(text: str) -> int:
    return len(text.split()) if text else 0


def _fenced_max_nonblank(response: str) -> Tuple[int, int]:
    n_blocks = 0
    best = 0
    for m in _FENCE_RE.finditer(response or ""):
        n_blocks += 1
        body = m.group(1)
        nonblank = sum(1 for ln in body.splitlines() if ln.strip())
        if nonblank > best:
            best = nonblank
    return n_blocks, best


def _sentence_count(text: str) -> int:
    if not text:
        return 0
    parts = [p for p in _SENTENCE_RE.split(text) if p.strip()]
    return max(1, len(parts))


def _punct_density(text: str) -> float:
    if not text:
        return 0.0
    punct = sum(1 for ch in text if ch in ",;:!?.()[]{}-—")
    return float(punct) / max(1, len(text))


def _char_ratio(text: str, predicate) -> float:
    if not text:
        return 0.0
    hits = sum(1 for ch in text if predicate(ch))
    return float(hits) / max(1, len(text))


def _stopword_ratio(text: str) -> float:
    if not text:
        return 0.0
    toks = [t.lower() for t in re.findall(r"[A-Za-z]+", text)]
    if not toks:
        return 0.0
    hits = sum(1 for t in toks if t in _EN_STOP)
    return float(hits) / len(toks)


def _numeric_token_ratio(text: str) -> float:
    if not text:
        return 0.0
    toks = text.split()
    if not toks:
        return 0.0
    n = sum(1 for t in toks if any(ch.isdigit() for ch in t))
    return float(n) / len(toks)


def _json_hint_density(text: str) -> float:
    if not text:
        return 0.0
    # Weight: number of "key": value hits per 1KB of context. Saturates
    # at ~10.0 to avoid overflow on very large JSON blobs.
    hits = len(_JSON_HINT_RE.findall(text))
    if hits == 0:
        return 0.0
    density = hits / max(1, len(text) / 1024.0)
    return float(min(density, 10.0))


def _yaml_hint_density(text: str) -> float:
    if not text:
        return 0.0
    lines = text.splitlines()
    if not lines:
        return 0.0
    hits = sum(1 for ln in lines if _YAML_LINE_RE.match(ln))
    return float(hits) / len(lines)


def featurize(
    *,
    query: str,
    response: str,
    raw_context: str,
) -> FeatureVector:
    """Compute the locked feature vector for a request.

    Deterministic, pure, <3 ms on CPU for responses up to ~50 KB.
    """
    r = response or ""
    c = raw_context or ""
    q = query or ""

    n_blocks, fenced_max_nonblank = _fenced_max_nonblank(r)
    file_headers = len(_FILE_HEADER_RE.findall(c))
    code_ext_hits = len(_CODE_FILE_EXT_RE.findall(c))

    values: List[float] = [
        _log1p_int(_token_count(r)),
        _log1p_int(_sentence_count(r)),
        _punct_density(r),
        float(n_blocks),
        float(fenced_max_nonblank),
        float(1 if _DIFF_HEADER_RE.search(r) else 0),
        float(len(_DEF_RE.findall(r))),
        _char_ratio(r, lambda ch: ch in "{}"),
        _char_ratio(r, lambda ch: ch == "`"),
        _log1p_int(len(c)),
        _json_hint_density(c),
        _yaml_hint_density(c),
        float(min(len(_TABLE_ROW_RE.findall(c)), 50)),
        float(min(file_headers, 50)),
        float(min(code_ext_hits, 50)),
        float(1 if file_headers >= 2 else 0),
        _log1p_int(_token_count(q)),
        _char_ratio(r, lambda ch: ord(ch) < 128),
        _stopword_ratio(r),
        _numeric_token_ratio(r),
    ]
    assert len(values) == len(FEATURE_NAMES), "feature vector length drift"
    return FeatureVector(names=FEATURE_NAMES, values=values)


def featurize_batch(
    rows: Iterable[Dict[str, str]],
    *,
    query_key: str = "query",
    response_key: str = "response",
    context_key: str = "raw_context",
) -> Tuple[List[List[float]], Tuple[str, ...]]:
    """Vectorised wrapper that preserves input row order."""
    matrix: List[List[float]] = []
    for row in rows:
        vec = featurize(
            query=str(row.get(query_key) or ""),
            response=str(row.get(response_key) or ""),
            raw_context=str(row.get(context_key) or ""),
        )
        matrix.append(list(vec.values))
    return matrix, FEATURE_NAMES
