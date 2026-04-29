"""Deterministic high-precision corpus-type rules.

These rules run before the learned classifier and only fire when the
structural evidence is unambiguous. Two goals:

1. **Generalization**: the LR classifier trained on a curated mixture
   can overfit to dataset-specific surface features (metadata headers,
   specific JSON shapes). Rules encode the structural priors directly
   so out-of-distribution inputs with clean structural evidence get
   routed correctly regardless of what the LR learnt.
2. **Auditability**: when a rule fires, the routing decision carries a
   human-readable reason string (``rule:json_structured``,
   ``rule:fenced_code_multi_file``, ...) so customers can reproduce
   why TRACE picked a given bundle.

A rule only fires when its confidence is >= ``MIN_RULE_CONFIDENCE``
(default 0.9). Below that, the router defers to the LR classifier.

All rules are pure, deterministic, <0.5 ms CPU for typical payloads.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


MIN_RULE_CONFIDENCE = 0.9


# Pre-compiled patterns. Kept tight to avoid false positives.
_JSON_ROOT_RE = re.compile(r"^\s*[\[{]")
_JSON_KV_RE = re.compile(
    r'"[^"]+"\s*:\s*(?:"[^"]*"|-?\d+(?:\.\d+)?|true|false|null|\{|\[)'
)
_MD_TABLE_SEP_RE = re.compile(r"^\s*\|\s*:?-{2,}.*\|", re.MULTILINE)
_MD_TABLE_ROW_RE = re.compile(r"^\s*\|[^\n]+\|\s*$", re.MULTILINE)
_FENCE_RE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)

# Broader file-header detection than the training featurizer — matches
# ``=== path ===``, ``# File: path``, ``File: path``, ``[path.ext]`` so
# agentic traces with any common file-list convention are recognised.
_SENTENCE_RE = re.compile(r"[.!?]+\s+|[.!?]+$")


def _count_sentences(text: str) -> int:
    if not text:
        return 0
    parts = [p for p in _SENTENCE_RE.split(text) if p.strip()]
    return max(1, len(parts))


def _count_tokens(text: str) -> int:
    return len(text.split()) if text else 0


_FILE_HEADER_PATTERNS = [
    re.compile(r"^===\s+\S+?\s+===\s*$", re.MULTILINE),
    re.compile(r"^\s*#\s*File:\s*\S+", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\s*File:\s*\S+", re.MULTILINE),
    re.compile(r"^\s*\[\S+\.(?:py|ts|tsx|js|jsx|go|rs|java|kt|swift|rb|cpp|cc|h|hpp|cs|sh|yml|yaml|toml|json|sql|md)\]\s*$", re.MULTILINE | re.IGNORECASE),
    re.compile(r"^\s*---\s+\S+\s+---\s*$", re.MULTILINE),
]


@dataclass(frozen=True)
class RuleDecision:
    """Outcome of the rule overlay.

    ``corpus_type`` is ``None`` when no rule fired with high enough
    confidence; the router then defers to the LR classifier.
    """

    corpus_type: Optional[str]
    confidence: float
    reason: Optional[str]


def _count_file_headers(context: str) -> int:
    if not context:
        return 0
    seen: set[tuple[int, int]] = set()
    total = 0
    for pat in _FILE_HEADER_PATTERNS:
        for m in pat.finditer(context):
            key = (m.start(), m.end())
            if key not in seen:
                seen.add(key)
                total += 1
    return total


def _has_json_root_structure(context: str) -> bool:
    if not context:
        return False
    stripped = context.lstrip()
    if not stripped.startswith(("{", "[")):
        return False
    # Require at least 2 key-value pairs so a string starting with
    # ``[`` that is not actually JSON (e.g. ``[US-01:DOC-xxx]``
    # Veracier header) does not trigger.
    return len(_JSON_KV_RE.findall(context)) >= 2


def _has_markdown_table(context: str) -> bool:
    if not context:
        return False
    # A markdown table must have a separator row (``|---|---|``) plus
    # at least two body rows.
    return bool(_MD_TABLE_SEP_RE.search(context)) and len(
        _MD_TABLE_ROW_RE.findall(context)
    ) >= 3


def _fenced_code_blocks(response: str) -> int:
    return len(_FENCE_RE.findall(response or ""))


def _fenced_max_nonblank_lines(response: str) -> int:
    best = 0
    for m in _FENCE_RE.finditer(response or ""):
        lines = [ln for ln in m.group(1).splitlines() if ln.strip()]
        if len(lines) > best:
            best = len(lines)
    return best


def apply_rules(
    *,
    query: str,
    response: str,
    raw_context: str,
) -> RuleDecision:
    """Run the rule overlay over a request payload.

    Returns the first rule that fires at >= ``MIN_RULE_CONFIDENCE``.
    Order of rules is significant: earlier rules are more specific.
    """

    ctx = raw_context or ""
    rsp = response or ""

    # ------------------------------------------------------------------
    # Rule 1: structured source detection.
    # Triggers when the context is obviously JSON-rooted or a markdown
    # table. This is the highest-confidence rule because JSON/table is
    # a universal format that doesn't depend on dataset conventions.
    # ------------------------------------------------------------------
    file_headers = _count_file_headers(ctx)
    n_fences = _fenced_code_blocks(rsp)
    max_fence_lines = _fenced_max_nonblank_lines(rsp)

    # Pre-compute "looks like a code bundle" so the structured rules
    # below can defer to the code rules when the context is clearly a
    # multi-file plan / transcript / source-bundle. Plan files often
    # contain markdown tables and JSON snippets in their prose; those
    # are not the same as a query against a structured source.
    is_code_bundle_context = file_headers >= 3

    # ------------------------------------------------------------------
    # Rule 1: structured source detection.
    # Triggers when the context is obviously JSON-rooted or a markdown
    # table AND the context does NOT also look like a multi-file code
    # bundle (which would mean the JSON / table is incidental to a
    # plan or trace, not the actual structured source).
    # ------------------------------------------------------------------
    if not is_code_bundle_context and _has_json_root_structure(ctx):
        return RuleDecision(
            corpus_type="rag.structured",
            confidence=0.98,
            reason="rule:json_rooted_context",
        )
    if not is_code_bundle_context and _has_markdown_table(ctx):
        return RuleDecision(
            corpus_type="rag.structured",
            confidence=0.95,
            reason="rule:markdown_table_context",
        )

    # ------------------------------------------------------------------
    # Rule 2: agentic code trace.
    # Multi-file context (3+ file headers) + fenced code block whose
    # largest block has >= 5 non-blank lines. The 5-line cut matches
    # the training-time splitter for ``code.agentic_trace`` vs
    # ``rag.code_in_context`` (see research/corpus_classifier dataset
    # builder), so the rule mirrors the gold definition rather than
    # learning it from features.
    # ------------------------------------------------------------------
    if file_headers >= 3 and n_fences >= 1 and max_fence_lines >= 5:
        return RuleDecision(
            corpus_type="code.agentic_trace",
            confidence=0.95,
            reason=(
                f"rule:multi_file_trace(headers={file_headers},"
                f"fences={n_fences},max_lines={max_fence_lines})"
            ),
        )

    # ------------------------------------------------------------------
    # Rule 3: code in RAG context.
    # Two shapes:
    #   a) at most one file header + at least one fenced code block
    #      with one non-blank line (RAG-style code-as-answer);
    #   b) multi-file context with a SHORT fenced response (< 5 non-
    #      blank lines), which by the training splitter falls in
    #      ``rag.code_in_context`` rather than the agentic trace class.
    # ------------------------------------------------------------------
    if n_fences >= 1 and file_headers <= 1 and max_fence_lines >= 1:
        return RuleDecision(
            corpus_type="rag.code_in_context",
            confidence=0.92,
            reason=f"rule:code_in_context(fences={n_fences},max_lines={max_fence_lines})",
        )
    if file_headers >= 3 and n_fences >= 1 and max_fence_lines < 5:
        return RuleDecision(
            corpus_type="rag.code_in_context",
            confidence=0.9,
            reason=(
                f"rule:code_in_context_short(headers={file_headers},"
                f"fences={n_fences},max_lines={max_fence_lines})"
            ),
        )

    # ------------------------------------------------------------------
    # Rule 4: prose length-based tiebreakers.
    # Short factoids and multi-claim paragraph summaries have very
    # different fusion profiles (pure-NLI vs literal-heavy) so getting
    # the wrong bundle materially changes the score. These two
    # heuristics are dataset-agnostic — they key on response shape
    # (sentence count, token count) rather than source-specific
    # surface features.
    # ------------------------------------------------------------------
    if n_fences == 0 and not file_headers:
        sentence_count = _count_sentences(rsp)
        token_count = _count_tokens(rsp)
        ctx_tokens = _count_tokens(ctx)
        # Short factoid: one concise answer sentence, short context.
        # Minimum response / context lengths guard against trivial
        # pytest fixtures and Veracier metadata snippets that happen to
        # be short but are NOT factoid prose.
        if (
            sentence_count <= 1
            and 4 <= token_count <= 25
            and len(ctx) <= 800
            and ctx_tokens >= 5
            and not ctx.lstrip().startswith("[")
        ):
            return RuleDecision(
                corpus_type="rag.prose.short_factoid",
                confidence=0.91,
                reason=f"rule:short_factoid(sentences={sentence_count},tokens={token_count})",
            )
        # Multi-claim summary: paragraph response with several claims.
        # Picks up summary-style answers with >= 3 sentences AND >= 60
        # tokens; those are reliably paragraph-length answers.
        if sentence_count >= 3 and token_count >= 60:
            return RuleDecision(
                corpus_type="rag.prose.multi_claim",
                confidence=0.9,
                reason=f"rule:multi_claim(sentences={sentence_count},tokens={token_count})",
            )

    # ------------------------------------------------------------------
    # No rule fired with enough confidence; defer to LR.
    # (rag.prose.enterprise has no reliable structural signature vs
    # multi_claim or short_factoid in pure text — callers that know
    # they are serving enterprise content should pass ``corpus_type``
    # explicitly on the request.)
    # ------------------------------------------------------------------
    return RuleDecision(corpus_type=None, confidence=0.0, reason=None)
