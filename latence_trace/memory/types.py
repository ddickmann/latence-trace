"""Rule-based span typing for InfiniMem v1."""

from __future__ import annotations

import re

from latence_trace.memory.models import SpanType

_CONSTRAINT_RE = re.compile(r"\b(must|never|do not|don't|required|constraint|preserve|always)\b", re.I)
_GOAL_RE = re.compile(r"\b(goal|objective|task|implement|fix|add|build|prove|ship)\b", re.I)
_DECISION_RE = re.compile(r"\b(decided|decision|therefore|we will|approved|rejected|supersedes)\b", re.I)
_ERROR_RE = re.compile(r"\b(error|exception|traceback|failed|failure|panic|segfault|pytest|assertion)\b", re.I)
_TOOL_RE = re.compile(r"\b(command|stdout|stderr|exit code|tool|shell|runpod|deploy)\b", re.I)
_FILE_RE = re.compile(r"\b(?:[\w.-]+/)+[\w.-]+\b")
_CODE_RE = re.compile(r"```|class\s+\w+|def\s+\w+|function\s+\w+|const\s+\w+|fn\s+\w+", re.I)
_RETRIEVAL_RE = re.compile(r"\b(source|citation|retrieved|document|chunk|support unit)\b", re.I)


def classify_span(text: str, *, source: str = "turn") -> SpanType:
    stripped = text.strip()
    if not stripped:
        return "filler"
    if _CODE_RE.search(stripped):
        return "code_fragment"
    if _ERROR_RE.search(stripped):
        return "error"
    if _CONSTRAINT_RE.search(stripped):
        return "constraint"
    if _DECISION_RE.search(stripped):
        return "decision"
    if _FILE_RE.search(stripped):
        return "code_symbol"
    if source == "raw_context" or _RETRIEVAL_RE.search(stripped):
        return "retrieval_chunk"
    if _TOOL_RE.search(stripped):
        return "tool_result"
    if _GOAL_RE.search(stripped):
        return "goal"
    if len(stripped.split()) <= 4:
        return "filler"
    return "evidence"
