"""Deterministic corruption strategies for coding-agent responses."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence


_IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
_CALL_RE = re.compile(r"(?P<name>\b[A-Za-z_][A-Za-z0-9_\.]*)\((?P<args>[^()\n]*?,[^()\n]*?)\)")
_IMPORT_LINE_RE = re.compile(
    r"(?m)^(?P<prefix>[+\- ]?)(?P<body>(?:from\s+\S+\s+import\s+[^\n]+|import\s+[^\n]+|use\s+[^\n]+|#include\s+[^\n]+))$"
)
_DIFF_MARKER_RE = re.compile(r"(?m)^(?:diff --git|index |@@ |--- |\+\+\+ ).*$")
_KEYWORDS = {
    "and",
    "as",
    "break",
    "class",
    "const",
    "continue",
    "def",
    "elif",
    "else",
    "except",
    "export",
    "false",
    "finally",
    "for",
    "from",
    "function",
    "if",
    "import",
    "in",
    "let",
    "new",
    "none",
    "null",
    "pass",
    "raise",
    "return",
    "self",
    "static",
    "super",
    "this",
    "throw",
    "true",
    "try",
    "use",
    "var",
    "while",
}


@dataclass(frozen=True)
class CorruptionResult:
    strategy: str
    subcategory: str
    text: str
    note: str


def _response_body(text: str) -> str:
    body = _DIFF_MARKER_RE.sub("", text)
    body = re.sub(r"(?m)^[+\- ]", "", body)
    return body


def _identifier_candidates(text: str) -> List[str]:
    seen = set()
    candidates: List[str] = []
    for token in _IDENTIFIER_RE.findall(_response_body(text)):
        if token.lower() in _KEYWORDS:
            continue
        if len(token) <= 2:
            continue
        if token not in seen:
            seen.add(token)
            candidates.append(token)
    return candidates


def _replace_once(pattern: str, replacement: str, text: str) -> Optional[str]:
    updated, count = re.subn(pattern, replacement, text, count=1)
    return updated if count else None


def rename_symbol(text: str) -> Optional[CorruptionResult]:
    for identifier in _identifier_candidates(text):
        replacement = f"{identifier}_shadow"
        updated = _replace_once(rf"\b{re.escape(identifier)}\b", replacement, text)
        if updated and updated != text:
            return CorruptionResult(
                strategy="rename_symbol",
                subcategory="entity_swap",
                text=updated,
                note=f"Renamed grounded symbol '{identifier}' to unsupported near-miss '{replacement}'.",
            )
    return None


def _split_args(argument_text: str) -> List[str]:
    parts: List[str] = []
    current: List[str] = []
    depth = 0
    for char in argument_text:
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)
        if char == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
            continue
        current.append(char)
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def swap_args(text: str) -> Optional[CorruptionResult]:
    for match in _CALL_RE.finditer(_response_body(text)):
        call_name = match.group("name")
        args = _split_args(match.group("args"))
        if len(args) < 2:
            continue
        swapped_args = list(args)
        swapped_args[0], swapped_args[1] = swapped_args[1], swapped_args[0]
        original_call = f"{call_name}({match.group('args')})"
        swapped_call = f"{call_name}({', '.join(swapped_args)})"
        updated = text.replace(original_call, swapped_call, 1)
        if updated != text:
            return CorruptionResult(
                strategy="swap_args",
                subcategory="entity_swap",
                text=updated,
                note=f"Swapped the first two arguments in call '{call_name}(...)'.",
            )
    return None


def drop_import(text: str) -> Optional[CorruptionResult]:
    for match in _IMPORT_LINE_RE.finditer(text):
        line = match.group("body")
        prefix = match.group("prefix")
        if line.startswith("from ") and " import " in line:
            left, right = line.split(" import ", 1)
            symbols = [symbol.strip() for symbol in right.split(",") if symbol.strip()]
            if len(symbols) >= 2:
                updated_line = f"{prefix}{left} import {', '.join(symbols[1:])}"
            else:
                updated_line = ""
        elif line.startswith("import "):
            updated_line = ""
        elif line.startswith("use "):
            updated_line = ""
        elif line.startswith("#include "):
            updated_line = ""
        else:
            continue
        updated = text[: match.start()] + updated_line + text[match.end() :]
        if updated != text:
            return CorruptionResult(
                strategy="drop_import",
                subcategory="partial",
                text=updated,
                note="Dropped an import/include line so the response now references symbols without their declared dependency.",
            )
    return None


def phantom_call(text: str) -> Optional[CorruptionResult]:
    for match in _CALL_RE.finditer(_response_body(text)):
        call_name = match.group("name")
        replacement = "phantom_helper"
        if call_name.endswith(replacement):
            continue
        original_call = f"{call_name}({match.group('args')})"
        phantom_name = replacement if "." not in call_name else f"{call_name.rsplit('.', 1)[0]}.{replacement}"
        updated = text.replace(original_call, f"{phantom_name}({match.group('args')})", 1)
        if updated != text:
            return CorruptionResult(
                strategy="phantom_call",
                subcategory="phantom_api",
                text=updated,
                note=f"Replaced grounded call '{call_name}' with phantom API '{phantom_name}'.",
            )
    for identifier in _identifier_candidates(text):
        replacement = "phantom_helper"
        updated = _replace_once(rf"\b{re.escape(identifier)}\b", replacement, text)
        if updated and updated != text:
            return CorruptionResult(
                strategy="phantom_call",
                subcategory="phantom_api",
                text=updated,
                note=f"Injected phantom API name '{replacement}' in place of '{identifier}'.",
            )
    return None


def deprecated_api(text: str) -> Optional[CorruptionResult]:
    for match in _CALL_RE.finditer(_response_body(text)):
        call_name = match.group("name")
        tail = call_name.rsplit(".", 1)[-1]
        replacement_tail = f"legacy_{tail}"
        replacement = replacement_tail if "." not in call_name else f"{call_name.rsplit('.', 1)[0]}.{replacement_tail}"
        original_call = f"{call_name}({match.group('args')})"
        updated = text.replace(original_call, f"{replacement}({match.group('args')})", 1)
        if updated != text:
            return CorruptionResult(
                strategy="deprecated_api",
                subcategory="negation_flip",
                text=updated,
                note=f"Replaced supported call '{call_name}' with deprecated-looking variant '{replacement}'.",
            )
    for identifier in _identifier_candidates(text):
        replacement = f"legacy_{identifier}"
        updated = _replace_once(rf"\b{re.escape(identifier)}\b", replacement, text)
        if updated and updated != text:
            return CorruptionResult(
                strategy="deprecated_api",
                subcategory="negation_flip",
                text=updated,
                note=f"Renamed '{identifier}' to deprecated-looking variant '{replacement}'.",
            )
    return None


_STRATEGIES: Sequence[Callable[[str], Optional[CorruptionResult]]] = (
    rename_symbol,
    swap_args,
    drop_import,
    phantom_call,
    deprecated_api,
)


def available_strategies() -> List[str]:
    return [strategy.__name__ for strategy in _STRATEGIES]


def make_corrupted_variant(
    text: str,
    *,
    case_id: str = "",
    preferred_strategy: Optional[str] = None,
) -> CorruptionResult:
    if preferred_strategy:
        for strategy in _STRATEGIES:
            if strategy.__name__ == preferred_strategy:
                result = strategy(text)
                if result is not None:
                    return result
                break

    rotation = 0
    if case_id:
        rotation = int(hashlib.sha1(case_id.encode("utf-8")).hexdigest(), 16) % len(_STRATEGIES)
    ordered = list(_STRATEGIES[rotation:]) + list(_STRATEGIES[:rotation])
    for strategy in ordered:
        result = strategy(text)
        if result is not None:
            return result

    fallback = phantom_call(text)
    if fallback is not None:
        return fallback
    raise ValueError("Unable to corrupt response text with the available strategies")
