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


def _fenced_total_nonblank_lines(response: str) -> int:
    """Sum of non-blank lines across every fenced block.

    Real agentic traces often contain several short diffs / patches
    rather than one long listing. Counting total lines captures that
    shape where ``max_nonblank_lines`` would miss it.
    """
    total = 0
    for m in _FENCE_RE.finditer(response or ""):
        total += sum(1 for ln in m.group(1).splitlines() if ln.strip())
    return total


# Regex matches a line that starts with a `+` or `-` marker followed by
# whitespace — the standard unified-diff / Cursor diff shape. Requires
# the marker to sit on its own line so prose bullets like "- item" inside
# a list don't false-positive.
_DIFF_LINE_RE = re.compile(r"^\s*[+\-]\s+\S", re.MULTILINE)

# Counts distinct numeric tokens in a response. Used to reject dense
# enterprise summaries (which routinely cite 3+ numbers in a single
# sentence) from the ``short_factoid`` rule, without hurting HaluEval
# QA answers that typically cite at most 1–2 numbers per answer.
_NUMBER_RE = re.compile(r"(?<![A-Za-z])\d+(?:[.,]\d+)*(?![A-Za-z])")
_POLICY_CUE_RE = re.compile(
    r"\b(?:policy|policies|manual|sop|procedure|compliance|approval|approve|requires?|"
    r"must|shall|cannot|prohibited|exception|quarantine|qa|regulation|wire transfer)\b",
    re.IGNORECASE,
)
_MULTI_CLAIM_CUE_RE = re.compile(
    r"\b(?:and|or|also|plus|should|must|may|cannot|while|until)\b|[,;]",
    re.IGNORECASE,
)
_CODE_EXT_RE = re.compile(
    r"\.(?:py|ts|tsx|js|jsx|go|rs|java|kt|swift|rb|cpp|cc|h|hpp|cs|sh|yml|yaml|toml|json|sql)\b",
    re.IGNORECASE,
)
_CODE_SYMBOL_RE = re.compile(
    r"\b(?:def|class|function|method|pytest|test_|import|SDK|client|API|"
    r"[A-Za-z_][A-Za-z0-9_]*\([^)]*\)|[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*)",
)


def _count_numbers(text: str) -> int:
    if not text:
        return 0
    return len(_NUMBER_RE.findall(text))


def _count_policy_cues(text: str) -> int:
    return len(_POLICY_CUE_RE.findall(text or ""))


def _count_multi_claim_cues(text: str) -> int:
    return len(_MULTI_CLAIM_CUE_RE.findall(text or ""))


def _count_code_cues(text: str) -> int:
    return len(_CODE_EXT_RE.findall(text or "")) + len(_CODE_SYMBOL_RE.findall(text or ""))


def _has_diff_markers(response: str) -> bool:
    if not response:
        return False
    # A single stray ``- ...`` line is usually a bullet, not a diff.
    # Require at least two diff-shaped lines for confidence.
    return len(_DIFF_LINE_RE.findall(response)) >= 2


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
    total_fence_lines = _fenced_total_nonblank_lines(rsp)
    diff_markers = _has_diff_markers(rsp)

    # Pre-compute "looks like a code bundle" so the structured rules
    # below can defer to the code rules when the context is clearly a
    # multi-file plan / transcript / source-bundle. Plan files often
    # contain markdown tables and JSON snippets in their prose; those
    # are not the same as a query against a structured source.
    is_code_bundle_context = file_headers >= 3

    policy_cues = _count_policy_cues(f"{query}\n{ctx}\n{rsp}")
    code_cues = _count_code_cues(f"{query}\n{ctx}\n{rsp}")

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
    # Multi-file context (3+ file headers) plus one of three agentic
    # shapes on the response side:
    #   a) one long fenced block (>= 5 non-blank lines) — mirrors the
    #      training-time splitter for transcripts_v2 gold labels
    #   b) multiple fenced blocks totalling >= 6 non-blank lines —
    #      captures real agentic diffs where the model ships 2-3 short
    #      hunks instead of one long listing
    #   c) diff-marker lines (``+`` / ``-``) plus at least one fenced
    #      block — Cursor / Aider style patches are unambiguously
    #      agentic regardless of block length
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
    if file_headers >= 3 and n_fences >= 2 and total_fence_lines >= 6:
        return RuleDecision(
            corpus_type="code.agentic_trace",
            confidence=0.93,
            reason=(
                f"rule:multi_file_trace_multi_block(headers={file_headers},"
                f"fences={n_fences},total_lines={total_fence_lines})"
            ),
        )
    if file_headers >= 3 and n_fences >= 1 and diff_markers:
        return RuleDecision(
            corpus_type="code.agentic_trace",
            confidence=0.93,
            reason=(
                f"rule:multi_file_trace_with_diff(headers={file_headers},"
                f"fences={n_fences},max_lines={max_fence_lines})"
            ),
        )

    # Agentic traces often arrive as prose summaries from IDE plugins,
    # without markdown fences. A code-scoring caller plus file/test/command
    # evidence is enough to route into the trajectory bundle.
    if code_cues >= 4 and re.search(r"\b(?:pytest|test_|last command|verified|patch|diff)\b", f"{ctx}\n{rsp}", re.IGNORECASE):
        return RuleDecision(
            corpus_type="code.agentic_trace",
            confidence=0.92,
            reason=f"rule:agentic_trace_prose(code_cues={code_cues})",
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
    if code_cues >= 3 and re.search(r"\b(?:sdk|api|function|method|client|import|class)\b", f"{query}\n{ctx}", re.IGNORECASE):
        return RuleDecision(
            corpus_type="rag.code_in_context",
            confidence=0.91,
            reason=f"rule:code_symbol_context(code_cues={code_cues})",
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
        rsp_numbers = _count_numbers(rsp)
        multi_claim_cues = _count_multi_claim_cues(rsp)
        if sentence_count >= 3 and token_count >= 60:
            return RuleDecision(
                corpus_type="rag.prose.multi_claim",
                confidence=0.9,
                reason=f"rule:multi_claim(sentences={sentence_count},tokens={token_count})",
            )
        if policy_cues >= 2 or (policy_cues >= 1 and token_count >= 20):
            return RuleDecision(
                corpus_type="rag.prose.enterprise",
                confidence=0.93,
                reason=f"rule:enterprise_policy_cues(cues={policy_cues})",
            )
        if sentence_count >= 2 or (token_count >= 14 and multi_claim_cues >= 3):
            return RuleDecision(
                corpus_type="rag.prose.multi_claim",
                confidence=0.91,
                reason=(
                    f"rule:multi_claim_compact(sentences={sentence_count},"
                    f"tokens={token_count},cues={multi_claim_cues})"
                ),
            )
        # Short factoid: one concise answer sentence, short context,
        # low numeric density. HaluEval QA answers are typically
        # single-claim one-liners (0-2 numbers). Dense enterprise
        # summaries pack 3+ numbers into a single sentence (revenue,
        # margin, EPS in one go) and must defer to the LR classifier
        # so they route to ``rag.prose.enterprise``.
        #
        # Minimum response / context lengths guard against trivial
        # pytest fixtures and Veracier metadata snippets that happen
        # to be short but are NOT factoid prose. The context-token
        # ceiling (<= 60) prevents production RAG contexts from
        # vector DBs (typically >= 256 tokens) from ever being
        # misrouted here — that's the path that burns enterprise
        # customers with false reds.
        if (
            sentence_count <= 1
            and 4 <= token_count <= 25
            and rsp_numbers <= 2
            and len(ctx) <= 500
            and ctx_tokens >= 5
            and ctx_tokens <= 60
            and not ctx.lstrip().startswith("[")
            and policy_cues == 0
            and multi_claim_cues <= 2
            and code_cues == 0
        ):
            return RuleDecision(
                corpus_type="rag.prose.short_factoid",
                confidence=0.91,
                reason=(
                    f"rule:short_factoid(sentences={sentence_count},"
                    f"tokens={token_count},numbers={rsp_numbers},"
                    f"ctx_tokens={ctx_tokens})"
                ),
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
