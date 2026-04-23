"""Agent-transcript-sourced coding-agent groundedness fixtures.

Each scenario corresponds to a real Cursor agent session. We read the session's
accumulated user-turn context (up to a configurable cutoff) and turn it into a
heterogeneous list of ``CodeContextFile`` objects, then pair that context with
three tiers of response derived from the real assistant turn at a fixed fence:

- ``correct``    — real assistant prose + unchanged fenced block.
- ``wrong``      — same prose + a heavily corrupted fenced block (phantom
  APIs, invented classes, cross-version drift, imports from absent packages).
- ``ambiguous``  — same prose + a minimal, subtle corruption (single phantom
  helper, kwarg drift, or off-by-one signature change).

Edits live in ``cases_transcripts_v1.yaml`` so they are explicit and auditable.
The loader dynamically reconstructs the ``correct`` variant from the referenced
transcript, while the YAML embeds the literal ``wrong`` / ``ambiguous``
responses verbatim.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import unquote

import yaml

from research.triangular_maxsim.coding.code_cases import CodeContextFile, CodingCase


logger = logging.getLogger(__name__)

_AGENT_TRANSCRIPT_ROOT = Path("/root/.cursor/projects/workspace/agent-transcripts")

_DEFAULT_YAML = Path(__file__).resolve().parent / "cases_transcripts_v1.yaml"

_CODE_FENCE_RE = re.compile(r"```([\w+-]*)\n(.*?)\n```", re.DOTALL)

_CODE_SELECTION_OPEN_RE = re.compile(
    r'<code_selection\s+path="([^"]+)"\s+lines="([^"]+)"\s*>', re.IGNORECASE
)
_CODE_SELECTION_BLOCK_RE = re.compile(
    r'<code_selection\s+path="([^"]+)"\s+lines="([^"]+)"\s*>(.*?)</code_selection>',
    re.IGNORECASE | re.DOTALL,
)
_VSCODE_REMOTE_PREFIX_RE = re.compile(r"vscode-remote://ssh-remote\+[^/]+/")
_LINE_NUM_PREFIX_RE = re.compile(r"^\s{0,5}\d+\|")

_CURSOR_WRAPPER_TAGS = (
    "attached_files",
    "code_selection",
    "system_reminder",
    "system_notification",
    "open_and_recently_viewed_files",
    "user_info",
    "agent_transcripts",
    "agent_skills",
    "available_skills",
    "user_query",
    "terminal_output",
    "terminal_snapshot",
    "uploaded_documents",
)
_ANY_XML_TAG_RE = re.compile(r"</?[A-Za-z_][\w\-]*(?:\s[^>]*)?>")


@dataclass(frozen=True)
class _Scenario:
    """One base scenario spec parsed from the YAML curation file."""

    id: str
    transcript_id: str
    response_turn: int
    fence_idx: int
    description: str
    variants: List[Dict[str, Any]]


def _load_turns(transcript_path: Path) -> List[Dict[str, Any]]:
    turns: List[Dict[str, Any]] = []
    with transcript_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            role = event.get("role")
            content = event["message"]["content"]
            if isinstance(content, list):
                text = "\n".join(
                    item.get("text", "") for item in content if isinstance(item, dict)
                )
            else:
                text = content or ""
            turns.append({"role": role, "text": text})
    return turns


def _locate_transcript(transcript_id: str) -> Path:
    for directory in _AGENT_TRANSCRIPT_ROOT.iterdir():
        if not directory.is_dir():
            continue
        if directory.name.startswith(transcript_id):
            candidate = directory / f"{directory.name}.jsonl"
            if candidate.exists():
                return candidate
    raise FileNotFoundError(
        f"Transcript with prefix {transcript_id!r} not found under {_AGENT_TRANSCRIPT_ROOT}"
    )


def _clean_path(raw_path: str) -> str:
    decoded = unquote(raw_path)
    stripped = _VSCODE_REMOTE_PREFIX_RE.sub("", decoded)
    stripped = stripped.lstrip("/")
    if stripped.startswith("workspace/"):
        stripped = stripped[len("workspace/"):]
    if stripped.startswith("root/.cursor/plans/"):
        stripped = stripped.replace("root/.cursor/plans/", "plans/", 1)
    return stripped or "unknown_path"


def _strip_line_numbers(block: str) -> str:
    cleaned: List[str] = []
    for line in block.splitlines():
        cleaned.append(_LINE_NUM_PREFIX_RE.sub("", line))
    result = "\n".join(cleaned)
    return result.strip("\n") + "\n"


def _extract_code_selections(
    turns: List[Dict[str, Any]], cutoff_turn: int
) -> List[Tuple[str, str, int]]:
    """Return (path, content, turn_index) tuples from <code_selection> blocks."""

    selections: List[Tuple[str, str, int]] = []
    for turn_index, turn in enumerate(turns[:cutoff_turn]):
        if turn["role"] != "user":
            continue
        for match in _CODE_SELECTION_BLOCK_RE.finditer(turn["text"]):
            raw_path = match.group(1).strip()
            body = match.group(3)
            path = _clean_path(raw_path)
            content = _strip_line_numbers(body)
            if not content.strip():
                continue
            selections.append((path, content, turn_index))
    return selections


def _strip_cursor_wrappers(text: str) -> str:
    cleaned = text
    for tag in _CURSOR_WRAPPER_TAGS:
        cleaned = re.sub(
            rf"<{tag}\b[^>]*>(.*?)</{tag}>",
            r"\1",
            cleaned,
            flags=re.IGNORECASE | re.DOTALL,
        )
    cleaned = _ANY_XML_TAG_RE.sub("", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _collect_prose_file(
    turns: List[Dict[str, Any]], cutoff_turn: int, min_chars: int = 200
) -> List[CodeContextFile]:
    """Package each substantive user turn (minus code_selection blocks) into a markdown file."""

    prose_files: List[CodeContextFile] = []
    for turn_index, turn in enumerate(turns[:cutoff_turn]):
        if turn["role"] != "user":
            continue
        text = _CODE_SELECTION_BLOCK_RE.sub("", turn["text"])
        text = _strip_cursor_wrappers(text)
        if len(text) < min_chars:
            continue
        prose_files.append(
            CodeContextFile(
                path=f"session_notes/turn_{turn_index:04d}.md",
                content=text + "\n",
            )
        )
    return prose_files


def _dedupe_files_by_path(files: List[CodeContextFile]) -> List[CodeContextFile]:
    """Keep the latest occurrence per path (latest reflects post-edit state)."""

    last_by_path: Dict[str, CodeContextFile] = {}
    order: List[str] = []
    for f in files:
        if f.path not in last_by_path:
            order.append(f.path)
        last_by_path[f.path] = f
    return [last_by_path[path] for path in order]


def _cap_files(
    files: List[CodeContextFile],
    *,
    max_files: int,
    token_cap_chars: int,
) -> List[CodeContextFile]:
    """Trim to ``max_files`` and an aggregate character budget.

    Prefers the largest files first (they tend to carry the most signal).
    """

    sized = sorted(files, key=lambda f: len(f.content), reverse=True)
    kept: List[CodeContextFile] = []
    total_chars = 0
    for f in sized:
        if len(kept) >= max_files:
            break
        if total_chars + len(f.content) > token_cap_chars and kept:
            break
        kept.append(f)
        total_chars += len(f.content)
    original_order = {id(f): i for i, f in enumerate(files)}
    kept.sort(key=lambda f: original_order.get(id(f), 10**9))
    return kept


def _build_context_files(
    turns: List[Dict[str, Any]],
    *,
    cutoff_turn: int,
    max_files: int,
    token_cap_chars: int,
) -> List[CodeContextFile]:
    selections = _extract_code_selections(turns, cutoff_turn)
    code_files = [CodeContextFile(path=p, content=c) for p, c, _ in selections]
    code_files = _dedupe_files_by_path(code_files)

    prose_files = _collect_prose_file(turns, cutoff_turn)

    merged = code_files + prose_files
    capped = _cap_files(
        merged, max_files=max_files, token_cap_chars=token_cap_chars
    )
    if not capped:
        raise ValueError(
            f"Cutoff turn {cutoff_turn} produced no context files; widen cutoff or check transcript."
        )
    return capped


def _extract_correct_response(
    turns: List[Dict[str, Any]],
    *,
    response_turn: int,
    fence_idx: int,
    before_chars: int,
    after_chars: int,
) -> str:
    turn = turns[response_turn]
    if turn["role"] != "assistant":
        raise ValueError(
            f"Response turn {response_turn} is role={turn['role']!r}, expected assistant"
        )
    text = turn["text"]
    fences = list(_CODE_FENCE_RE.finditer(text))
    if fence_idx >= len(fences):
        raise ValueError(
            f"fence_idx={fence_idx} out of range for turn {response_turn} "
            f"(only {len(fences)} fences)"
        )
    fence = fences[fence_idx]
    lo = max(0, fence.start() - before_chars)
    hi = min(len(text), fence.end() + after_chars)

    prose_before = text[lo:fence.start()]
    sentence_break = max(
        prose_before.rfind(". "),
        prose_before.rfind(".\n"),
        prose_before.rfind("! "),
        prose_before.rfind("? "),
    )
    if sentence_break > 0:
        prose_before = prose_before[sentence_break + 1:].lstrip()

    prose_after = text[fence.end():hi]
    next_break = prose_after.find("\n\n")
    if next_break > 30:
        prose_after = prose_after[:next_break]
    else:
        cut = max(prose_after.rfind(". "), prose_after.rfind(".\n"))
        if cut > 30:
            prose_after = prose_after[: cut + 1]

    return (prose_before.strip() + "\n\n" + text[fence.start():fence.end()] + "\n\n" + prose_after.strip()).strip()


def _tier_to_label(tier: str) -> str:
    if tier == "correct":
        return "grounded"
    if tier == "ambiguous":
        return "ambiguous"
    return "ungrounded"


def load_transcript_cases(
    yaml_path: Optional[Path] = None,
    *,
    transcript_root: Optional[Path] = None,
) -> List[CodingCase]:
    yaml_file = Path(yaml_path) if yaml_path is not None else _DEFAULT_YAML
    if transcript_root is not None:
        global _AGENT_TRANSCRIPT_ROOT
        _AGENT_TRANSCRIPT_ROOT = Path(transcript_root)

    with yaml_file.open("r", encoding="utf-8") as handle:
        spec = yaml.safe_load(handle)

    context_spec = dict(spec.get("context") or {})
    max_files = int(context_spec.get("max_files", 40))
    token_cap_chars = int(context_spec.get("context_token_cap_chars", 160_000))
    scope_before = int(context_spec.get("scope_before_chars", 500))
    scope_after = int(context_spec.get("scope_after_chars", 250))

    scenarios: List[_Scenario] = []
    for raw in spec.get("scenarios") or []:
        scenarios.append(
            _Scenario(
                id=str(raw["id"]),
                transcript_id=str(raw["transcript_id"]),
                response_turn=int(raw["response_turn"]),
                fence_idx=int(raw.get("fence_idx", 0)),
                description=str(raw.get("description") or ""),
                variants=list(raw.get("variants") or []),
            )
        )

    cases: List[CodingCase] = []
    for scenario in scenarios:
        transcript_path = _locate_transcript(scenario.transcript_id)
        turns = _load_turns(transcript_path)

        context_files = _build_context_files(
            turns,
            cutoff_turn=scenario.response_turn,
            max_files=max_files,
            token_cap_chars=token_cap_chars,
        )
        correct_response = _extract_correct_response(
            turns,
            response_turn=scenario.response_turn,
            fence_idx=scenario.fence_idx,
            before_chars=scope_before,
            after_chars=scope_after,
        )
        query_text = f"{scenario.description.strip()} (transcript {scenario.transcript_id}, turn {scenario.response_turn})"

        base_metadata: Dict[str, Any] = {
            "base_scenario_id": scenario.id,
            "transcript_id": scenario.transcript_id,
            "response_turn": scenario.response_turn,
            "fence_idx": scenario.fence_idx,
            "context_file_count": len(context_files),
            "context_char_count": sum(len(f.content) for f in context_files),
        }

        cases.append(
            CodingCase(
                id=f"{scenario.id}__correct",
                source=f"transcript_v1:{scenario.id}",
                query=query_text,
                response=correct_response,
                label="grounded",
                subcategory="correct",
                notes=f"Correct (tier=correct). {scenario.description}",
                context_files=context_files,
                metadata={**base_metadata, "tier": "correct"},
            )
        )

        for variant in scenario.variants:
            tier = str(variant.get("tier") or "").strip()
            if tier not in {"wrong", "ambiguous"}:
                logger.warning(
                    "Skipping unknown tier %r for scenario %s", tier, scenario.id
                )
                continue
            variant_id = str(variant.get("id") or f"{scenario.id}__{tier}")
            response_text = str(variant.get("response") or "").strip()
            if not response_text:
                logger.warning(
                    "Variant %s for scenario %s has empty response; skipping",
                    tier,
                    scenario.id,
                )
                continue
            label = _tier_to_label(tier)
            subcategory = tier
            cases.append(
                CodingCase(
                    id=variant_id,
                    source=f"transcript_v1:{scenario.id}",
                    query=query_text,
                    response=response_text,
                    label=label,
                    subcategory=subcategory,
                    notes=str(variant.get("notes") or ""),
                    context_files=context_files,
                    metadata={**base_metadata, "tier": tier},
                )
            )

    return cases


def summarize_loaded_cases(cases: List[CodingCase]) -> Dict[str, Any]:
    by_tier: Dict[str, int] = {}
    per_base: Dict[str, Dict[str, int]] = {}
    total_context_chars = 0
    total_files = 0
    for case in cases:
        tier = str(case.metadata.get("tier") or case.subcategory or "unknown")
        by_tier[tier] = by_tier.get(tier, 0) + 1
        base = str(case.metadata.get("base_scenario_id") or "unknown")
        per_base.setdefault(base, {})[tier] = per_base.get(base, {}).get(tier, 0) + 1
        total_context_chars += sum(len(f.content) for f in case.context_files)
        total_files += len(case.context_files)
    return {
        "case_count": len(cases),
        "by_tier": by_tier,
        "per_base": per_base,
        "mean_context_files": total_files / max(1, len(cases)),
        "mean_context_chars": total_context_chars / max(1, len(cases)),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cases = load_transcript_cases()
    summary = summarize_loaded_cases(cases)
    print(json.dumps(summary, indent=2))
    for case in cases:
        print(
            f"{case.id:<55} tier={case.metadata['tier']:<10} "
            f"label={case.label:<10} files={len(case.context_files):<3} "
            f"response_chars={len(case.response):<5}"
        )
