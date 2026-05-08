#!/usr/bin/env python
"""Build multi-turn A/B eval cases from real chat transcripts.

Extracts realistic multi-turn conversation sequences from the workspace's
agent transcripts, with varying context patterns:
  - Initial context (first user turn with code/docs attached)
  - Follow-up without new context (pure question)
  - New context arrives (user pastes new code, docs, error logs)
  - No-context follow-up (clarification, refinement)
  - New context again (updated code, new retrieval)

Outputs a JSONL file where each line is one eval case containing
multiple turns, each turn carrying its own context/query pair.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
TRANSCRIPT_ROOT = Path("/root/.cursor/projects/workspace/agent-transcripts")
OUTPUT_DIR = REPO_ROOT / "research" / "trace_memory" / "multiturn_eval"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Turn:
    """One turn in a multi-turn eval case."""
    turn_index: int
    user_query: str
    assistant_response: str
    raw_context: str | None  # context block if user provided docs/code/files
    context_pattern: str     # "initial_context", "follow_up", "new_context", "no_context"
    token_count_query: int
    token_count_response: int
    token_count_context: int


@dataclass
class EvalCase:
    """One multi-turn eval case."""
    case_id: str
    case_type: str           # "rag" or "code"
    language: str            # "en" or "de"
    source_transcript: str
    title: str
    turns: list[Turn]
    total_tokens: int
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Transcript parsing
# ---------------------------------------------------------------------------

def load_transcript_messages(path: Path) -> list[dict[str, Any]]:
    """Load user/assistant message pairs from a JSONL transcript."""
    messages: list[dict[str, Any]] = []
    with path.open("r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            role = obj.get("role")
            if role not in ("user", "assistant"):
                continue
            content = _extract_content(obj)
            if not content.strip():
                continue
            messages.append({
                "role": role,
                "content": content,
                "raw": obj,
            })
    return messages


def _extract_content(obj: dict[str, Any]) -> str:
    msg = obj.get("message", {})
    content = msg.get("content", "")
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content)


# ---------------------------------------------------------------------------
# Context extraction from user messages
# ---------------------------------------------------------------------------

_CONTEXT_PATTERNS = [
    re.compile(r"<START_CONTEXT>(.*?)</?\s*END_CONTEXT\s*>", re.DOTALL | re.IGNORECASE),
    re.compile(r"<CTX>(.*?)</CTX>", re.DOTALL | re.IGNORECASE),
    re.compile(r"\[CONTEXT\](.*?)\[/CONTEXT\]", re.DOTALL | re.IGNORECASE),
    re.compile(r"```[\w]*\n(.*?)```", re.DOTALL),
]

_FILE_HEADER_RE = re.compile(r"(?:^|\n)(?:#\s*file:|---\npath:|File:\s)", re.IGNORECASE)
_ATTACHED_FILES_RE = re.compile(
    r"<attached_files>(.*?)</attached_files>", re.DOTALL | re.IGNORECASE
)
_UPLOADED_DOCS_RE = re.compile(
    r"<uploaded_documents>(.*?)</uploaded_documents>", re.DOTALL | re.IGNORECASE
)
_USER_QUERY_RE = re.compile(
    r"<user_query>(.*?)</user_query>", re.DOTALL | re.IGNORECASE
)
_OPEN_FILES_RE = re.compile(
    r"<open_and_recently_viewed_files>(.*?)</open_and_recently_viewed_files>",
    re.DOTALL | re.IGNORECASE,
)


def extract_context_and_query(user_content: str) -> tuple[str, str | None]:
    """Split a user message into (query, context_block_or_None).

    Context is extracted from:
    - <START_CONTEXT>...</END_CONTEXT> markers
    - <attached_files> / <uploaded_documents> blocks
    - Large code blocks (```...```)
    - File header patterns (# file: ...)
    """
    context_parts: list[str] = []
    query = user_content

    uq_match = _USER_QUERY_RE.search(user_content)
    if uq_match:
        query = uq_match.group(1).strip()

    for pattern in [_ATTACHED_FILES_RE, _UPLOADED_DOCS_RE, _OPEN_FILES_RE]:
        for m in pattern.finditer(user_content):
            context_parts.append(m.group(1).strip())

    for pattern in _CONTEXT_PATTERNS:
        for m in pattern.finditer(user_content):
            block = m.group(1).strip()
            if _tok(block) >= 20:
                context_parts.append(block)

    if _FILE_HEADER_RE.search(user_content) and _tok(user_content) > 200:
        file_blocks = re.split(r"\n(?=#\s*file:)", user_content)
        for block in file_blocks:
            if _tok(block) >= 50 and block.strip() != query.strip():
                context_parts.append(block.strip())

    if not context_parts and _tok(user_content) > 500:
        lines = user_content.split("\n")
        code_lines = [l for l in lines if l.startswith("  ") or l.startswith("\t") or "def " in l or "class " in l or "import " in l]
        if len(code_lines) > 10:
            context_parts.append("\n".join(code_lines))

    context = "\n\n".join(context_parts).strip() if context_parts else None
    if context and _tok(context) < 15:
        context = None

    return query, context


# ---------------------------------------------------------------------------
# Turn sequence builder — creates realistic multi-turn patterns
# ---------------------------------------------------------------------------

def build_turn_sequence(
    messages: list[dict[str, Any]],
    *,
    min_turns: int = 4,
    max_turns: int = 8,
    start_index: int = 0,
) -> list[Turn] | None:
    """Extract a sequence of turns from consecutive user/assistant pairs.

    Returns None if insufficient quality turns are found.
    """
    turns: list[Turn] = []
    i = start_index
    turn_idx = 0

    while i < len(messages) and len(turns) < max_turns:
        if messages[i]["role"] != "user":
            i += 1
            continue

        user_content = messages[i]["content"]
        query, context = extract_context_and_query(user_content)

        if _tok(query) < 5:
            i += 1
            continue

        asst_content = ""
        j = i + 1
        while j < len(messages) and messages[j]["role"] == "assistant":
            asst_content += messages[j]["content"] + "\n"
            j += 1

        if _tok(asst_content) < 20:
            i = j
            continue

        asst_content = _truncate(asst_content, 4000)

        if turn_idx == 0:
            pattern = "initial_context" if context else "no_context"
        elif context:
            pattern = "new_context"
        else:
            pattern = "follow_up" if turn_idx % 3 != 0 else "no_context"

        turns.append(Turn(
            turn_index=turn_idx,
            user_query=_truncate(query, 2000),
            assistant_response=asst_content,
            raw_context=_truncate(context, 8000) if context else None,
            context_pattern=pattern,
            token_count_query=_tok(query),
            token_count_response=_tok(asst_content),
            token_count_context=_tok(context) if context else 0,
        ))
        turn_idx += 1
        i = j

    if len(turns) < min_turns:
        return None

    patterns = {t.context_pattern for t in turns}
    has_context = any(t.raw_context for t in turns)
    has_no_context = any(t.raw_context is None for t in turns)
    if not has_context or not has_no_context:
        return None

    return turns


# ---------------------------------------------------------------------------
# Case builder — extracts multiple cases from each transcript
# ---------------------------------------------------------------------------

def extract_cases_from_transcript(
    path: Path,
    case_type: str,
    *,
    max_cases: int = 4,
    language: str = "en",
) -> list[EvalCase]:
    """Extract multi-turn eval cases from a transcript."""
    messages = load_transcript_messages(path)
    uuid = path.stem

    user_indices = [i for i, m in enumerate(messages) if m["role"] == "user"]
    if len(user_indices) < 6:
        return []

    cases: list[EvalCase] = []
    positions = _spread_positions(user_indices, max_cases)

    for pos_idx, start in enumerate(positions):
        turns = build_turn_sequence(messages, start_index=start, min_turns=4, max_turns=8)
        if turns is None:
            turns = build_turn_sequence(messages, start_index=start, min_turns=3, max_turns=6)
        if turns is None:
            continue

        title = _make_title(turns[0].user_query)
        total_tokens = sum(
            t.token_count_query + t.token_count_response + t.token_count_context
            for t in turns
        )

        case_id = f"{case_type}_{language}_{_short_hash(uuid)}_{pos_idx:02d}"
        cases.append(EvalCase(
            case_id=case_id,
            case_type=case_type,
            language=language,
            source_transcript=uuid,
            title=title,
            turns=turns,
            total_tokens=total_tokens,
            metadata={
                "transcript_size": path.stat().st_size,
                "start_message_index": start,
                "turn_count": len(turns),
                "context_turns": sum(1 for t in turns if t.raw_context),
                "no_context_turns": sum(1 for t in turns if not t.raw_context),
                "patterns": [t.context_pattern for t in turns],
            },
        ))

        if len(cases) >= max_cases:
            break

    return cases


def _spread_positions(user_indices: list[int], count: int) -> list[int]:
    """Pick spread-out starting positions from user turn indices."""
    if len(user_indices) <= count:
        return user_indices[:count]
    step = max(1, len(user_indices) // (count + 1))
    positions = []
    for i in range(1, count + 1):
        idx = min(i * step, len(user_indices) - 6)
        positions.append(user_indices[max(0, idx)])
    return positions


# ---------------------------------------------------------------------------
# German detection
# ---------------------------------------------------------------------------

_GERMAN_MARKERS = re.compile(
    r"\b(?:und|oder|aber|nicht|haben|werden|können|müssen|sollen|"
    r"dass|wenn|weil|nach|über|zwischen|während|durch|gegen|unter|"
    r"Bitte|Danke|Entschuldigung|bereits|allerdings|eigentlich|"
    r"deshalb|trotzdem|außerdem|deswegen|nämlich|doch)\b",
    re.IGNORECASE,
)


def is_german(text: str) -> bool:
    words = text.split()
    if len(words) < 10:
        return False
    matches = len(_GERMAN_MARKERS.findall(text))
    return matches / max(1, len(words)) > 0.04


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _tok(text: str | None) -> int:
    if not text:
        return 0
    return len(text.split())


def _truncate(text: str | None, max_tokens: int) -> str:
    if not text:
        return ""
    words = text.split()
    if len(words) <= max_tokens:
        return text
    return " ".join(words[:max_tokens])


def _short_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:8]


def _make_title(query: str) -> str:
    clean = re.sub(r"<[^>]+>", " ", query)
    clean = re.sub(r"\s+", " ", clean).strip()
    return " ".join(clean.split()[:10])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

CODE_TRANSCRIPTS = [
    "91c1ac40-89c8-490f-a0e5-c5be3f429b27",  # Happy path user journey
    "8a0f6f68-933e-4731-888b-ad08a2b77bf0",  # vllm-factory plugin
    "f0afeaf2-c826-473b-a113-71948a444cf0",  # Triton MaxSim kernel
    "a4d14a56-9888-49df-b195-f272a3c9fca9",  # Architecture roadmap
    "126e38fc-d0f7-4817-aa65-c6498ce4b07c",  # next-plaid audit
    "d7fd7127-e0e7-48e9-944b-5ce3549e61ee",  # roq triton kernel
    "c36242f5-da80-4b5f-8d00-f541be9e08f8",  # Redaction model
    "fa052757-4c42-4a56-9108-47327151950a",  # ColGREP
    "ed69b0cd-1a1c-4e8c-8907-d785c13a1153",  # GEM graph retrieval
    "7b7fb621-592a-4fb9-9faa-999a5e204ab4",  # Tabu search solver
    "37db4b43-4b3b-4015-ac07-5f733da33cc8",  # rroq158/rroq4 quant
    "1f9e43fc-b886-452b-a976-a2d466f0e493",  # Code quality refactor
]

RAG_TRANSCRIPTS = [
    "f6a09f28-aa6d-44b7-ab59-ba7f19b82707",  # TRACE analytics (this chat!)
    "5e875b7e-fe30-49eb-83f4-6063ba3b7275",  # Compliance engine integration
    "aaf8cb18-62ce-4942-b7ba-6f213c0aed47",  # SDK documentation
    "91cf0ec9-d516-4c0e-83c6-96531d46a205",  # External research
    "3c8ee794-d9e1-4486-9ca0-1c6858861bd0",  # Portal integration
    "0b1422bb-a3d6-4b57-a251-68650a5b7073",  # Control plane buildout
    "d6519b80-c039-485f-bc3e-05e31444f803",  # TRACE production push
    "ee0ad405-c3dd-4122-abdf-8fbb8d082111",  # LLM optimization docs
    "ca28ed49-4aaf-4bb8-a527-1785fda99c7f",  # External research 2
    "f08909cd-d853-49e6-b418-4d29ea6f1642",  # Website features/benefits
]


def main() -> int:
    args = _parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_cases: list[EvalCase] = []

    # --- Code cases: 20 English ---
    print("=== Building CODE cases ===")
    code_cases: list[EvalCase] = []
    for uuid in CODE_TRANSCRIPTS:
        path = TRANSCRIPT_ROOT / uuid / f"{uuid}.jsonl"
        if not path.exists():
            print(f"  SKIP {uuid}: not found")
            continue
        cases = extract_cases_from_transcript(
            path, "code", max_cases=3, language="en"
        )
        code_cases.extend(cases)
        print(f"  {uuid[:8]}: {len(cases)} cases")

    code_cases.sort(key=lambda c: c.total_tokens, reverse=True)
    code_cases = code_cases[:20]
    print(f"  Total code cases: {len(code_cases)}")

    # --- RAG cases: 5 German + 15 English ---
    print("\n=== Building RAG cases ===")
    rag_cases: list[EvalCase] = []
    for uuid in RAG_TRANSCRIPTS:
        path = TRANSCRIPT_ROOT / uuid / f"{uuid}.jsonl"
        if not path.exists():
            print(f"  SKIP {uuid}: not found")
            continue
        cases = extract_cases_from_transcript(
            path, "rag", max_cases=4, language="en"
        )
        for case in cases:
            if any(is_german(t.user_query) or is_german(t.assistant_response) for t in case.turns):
                case.language = "de"
                case.case_id = case.case_id.replace("_en_", "_de_")
        rag_cases.extend(cases)
        print(f"  {uuid[:8]}: {len(cases)} cases ({sum(1 for c in cases if c.language == 'de')} de)")

    german_cases = [c for c in rag_cases if c.language == "de"]
    english_cases = [c for c in rag_cases if c.language == "en"]
    german_cases.sort(key=lambda c: c.total_tokens, reverse=True)
    english_cases.sort(key=lambda c: c.total_tokens, reverse=True)

    final_rag = german_cases[:5] + english_cases[:15]
    if len(german_cases) < 5:
        deficit = 5 - len(german_cases)
        final_rag = german_cases + english_cases[:15 + deficit]
    final_rag = final_rag[:20]
    print(f"  Total RAG cases: {len(final_rag)} ({sum(1 for c in final_rag if c.language == 'de')} de)")

    all_cases = code_cases + final_rag

    # --- Write output ---
    output_path = OUTPUT_DIR / "cases.jsonl"
    with output_path.open("w", encoding="utf-8") as f:
        for case in all_cases:
            row = asdict(case)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # --- Summary ---
    summary_path = OUTPUT_DIR / "summary.md"
    summary = _build_summary(all_cases)
    summary_path.write_text(summary, encoding="utf-8")

    print(f"\n=== Done ===")
    print(f"  Cases written: {len(all_cases)}")
    print(f"  Output: {output_path}")
    print(f"  Summary: {summary_path}")

    if args.verbose:
        for case in all_cases:
            patterns = [t.context_pattern for t in case.turns]
            print(f"  {case.case_id}: {case.case_type}/{case.language} "
                  f"{len(case.turns)}T {case.total_tokens}tok "
                  f"patterns={patterns}")

    return 0


def _build_summary(cases: list[EvalCase]) -> str:
    code_cases = [c for c in cases if c.case_type == "code"]
    rag_cases = [c for c in cases if c.case_type == "rag"]

    lines = [
        "# Multi-Turn InfiniMem Eval Cases",
        "",
        f"Total cases: {len(cases)}",
        f"- Code: {len(code_cases)} (all English)",
        f"- RAG: {len(rag_cases)} ({sum(1 for c in rag_cases if c.language == 'de')} German, "
        f"{sum(1 for c in rag_cases if c.language == 'en')} English)",
        "",
        "## Context Pattern Distribution",
        "",
    ]

    all_patterns: dict[str, int] = {}
    for case in cases:
        for turn in case.turns:
            all_patterns[turn.context_pattern] = all_patterns.get(turn.context_pattern, 0) + 1
    for pattern, count in sorted(all_patterns.items()):
        lines.append(f"- {pattern}: {count}")

    lines.extend(["", "## Cases", ""])
    for case in cases:
        patterns = [t.context_pattern for t in case.turns]
        ctx_turns = sum(1 for t in case.turns if t.raw_context)
        lines.append(
            f"### {case.case_id}\n"
            f"- Type: `{case.case_type}` | Lang: `{case.language}` | "
            f"Turns: `{len(case.turns)}` | Tokens: `{case.total_tokens:,}`\n"
            f"- Context turns: `{ctx_turns}/{len(case.turns)}`\n"
            f"- Patterns: `{patterns}`\n"
            f"- Title: {case.title}\n"
            f"- Source: `{case.source_transcript[:8]}...`\n"
        )

    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--transcript-root", type=Path, default=TRANSCRIPT_ROOT)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
