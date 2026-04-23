"""Code-aware support segmentation for the coding-agent benchmark."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from latence_trace.core.groundedness import count_text_tokens, segment_text


_DEFAULT_CHUNK_TOKEN_BUDGET = 256


@dataclass(frozen=True)
class RenderedContextFile:
    path: str
    content: str
    render_offset_start: int
    code_offset_start: int


def _coerce_context_file(context_file: Any) -> Tuple[str, str]:
    if isinstance(context_file, Mapping):
        path = str(context_file.get("path") or "").strip()
        content = str(context_file.get("content") or "")
    else:
        path = str(getattr(context_file, "path", "") or "").strip()
        content = str(getattr(context_file, "content", "") or "")
    if not path:
        raise ValueError("Each context file must include a non-empty path")
    return path, content


def render_context_files(context_files: Sequence[Any]) -> Tuple[str, List[RenderedContextFile]]:
    """Render a multi-file coding context into the raw string baseline format."""

    rendered_parts: List[str] = []
    entries: List[RenderedContextFile] = []
    cursor = 0

    for index, context_file in enumerate(context_files):
        path, content = _coerce_context_file(context_file)
        normalized = content.rstrip()
        if index > 0:
            rendered_parts.append("\n\n")
            cursor += 2
        header = f"# File: {path}\n"
        render_offset_start = cursor
        code_offset_start = render_offset_start + len(header)
        rendered_parts.append(header)
        rendered_parts.append(normalized)
        cursor = code_offset_start + len(normalized)
        entries.append(
            RenderedContextFile(
                path=path,
                content=normalized,
                render_offset_start=render_offset_start,
                code_offset_start=code_offset_start,
            )
        )

    return "".join(rendered_parts), entries


def _load_colgrep_parser() -> tuple[Optional[Any], str]:
    try:
        module = importlib.import_module("colgrep_parser")
        return module, "python_sdk"
    except Exception:
        return None, "sentence_packed_fallback"


def describe_segmenter_backend() -> Dict[str, str]:
    _module, backend = _load_colgrep_parser()
    if backend == "python_sdk":
        return {
            "name": "colgrep",
            "backend": backend,
            "status": "ready",
        }
    return {
        "name": "sentence_packed_fallback",
        "backend": backend,
        "status": "missing_colgrep_parser",
    }


def _line_char_bounds(text: str, start_line: int, end_line: int) -> tuple[int, int]:
    if not text:
        return 0, 0
    lines = text.splitlines(keepends=True)
    if not lines:
        return 0, len(text)
    line_starts: List[int] = []
    cursor = 0
    for line in lines:
        line_starts.append(cursor)
        cursor += len(line)
    safe_start = max(1, min(int(start_line or 1), len(lines)))
    safe_end = max(safe_start, min(int(end_line or safe_start), len(lines)))
    start_offset = line_starts[safe_start - 1]
    end_offset = line_starts[safe_end - 1] + len(lines[safe_end - 1])
    return start_offset, end_offset


def _split_long_line(
    line: str,
    *,
    provider: Any | None,
    chunk_token_budget: int,
) -> List[str]:
    stripped = line.strip()
    if not stripped:
        return []
    if provider is None or chunk_token_budget <= 0:
        return [stripped]
    if count_text_tokens(provider, stripped) <= chunk_token_budget:
        return [stripped]

    words = stripped.split()
    if len(words) > 1:
        chunks: List[str] = []
        current: List[str] = []
        for word in words:
            candidate = " ".join(current + [word]).strip()
            if current and count_text_tokens(provider, candidate) > chunk_token_budget:
                chunks.append(" ".join(current).strip())
                current = [word]
            else:
                current.append(word)
        if current:
            chunks.append(" ".join(current).strip())
        return [chunk for chunk in chunks if chunk]

    hard_chunks: List[str] = []
    step = 200
    for start in range(0, len(stripped), step):
        hard_chunks.append(stripped[start : start + step].strip())
    return [chunk for chunk in hard_chunks if chunk]


def pack_text_by_lines(
    text: str,
    *,
    provider: Any | None,
    chunk_token_budget: int,
) -> List[str]:
    stripped = text.strip()
    if not stripped:
        return []
    if provider is None or chunk_token_budget <= 0:
        return [stripped]
    if count_text_tokens(provider, stripped) <= chunk_token_budget:
        return [stripped]

    packed: List[str] = []
    current: List[str] = []
    for raw_line in stripped.splitlines():
        line = raw_line.rstrip()
        if provider is not None and count_text_tokens(provider, line) > chunk_token_budget:
            if current:
                packed.append("\n".join(current).strip())
                current = []
            packed.extend(
                _split_long_line(
                    line,
                    provider=provider,
                    chunk_token_budget=chunk_token_budget,
                )
            )
            continue
        candidate_lines = current + [line]
        candidate_text = "\n".join(candidate_lines).strip()
        if current and count_text_tokens(provider, candidate_text) > chunk_token_budget:
            packed.append("\n".join(current).strip())
            current = [line]
        else:
            current = candidate_lines
    if current:
        packed.append("\n".join(current).strip())
    return [chunk for chunk in packed if chunk]


def segment_code_files(
    context_files: Sequence[Any],
    *,
    provider: Any | None = None,
    chunk_token_budget: int = _DEFAULT_CHUNK_TOKEN_BUDGET,
) -> List[Dict[str, Any]]:
    """Segment code context with colgrep's code-unit descriptions when available."""

    rendered_context, rendered_files = render_context_files(context_files)
    parser, backend = _load_colgrep_parser()
    if parser is None:
        baseline = segment_text(
            rendered_context,
            "sentence_packed",
            provider=provider,
            chunk_token_budget=chunk_token_budget,
        )
        for segment in baseline:
            metadata = dict(segment.get("metadata") or {})
            metadata["segmenter_backend"] = backend
            segment["metadata"] = metadata
        return baseline

    segments: List[Dict[str, Any]] = []
    for context_file in rendered_files:
        try:
            units = parser.parse_code(context_file.content, context_file.path)
        except Exception:
            units = []

        if not units:
            fallback_text = f"File: {context_file.path}\n{context_file.content}".strip()
            for segment in pack_text_by_lines(
                fallback_text,
                provider=provider,
                chunk_token_budget=chunk_token_budget,
            ):
                segments.append(
                    {
                        "text": segment,
                        "offset_start": context_file.render_offset_start,
                        "offset_end": context_file.code_offset_start + len(context_file.content),
                        "metadata": {
                            "path": context_file.path,
                            "segmenter": "file_fallback",
                            "segmenter_backend": backend,
                        },
                    }
                )
            continue

        for unit_index, unit in enumerate(units):
            unit_text = (
                getattr(unit, "embedding_text", None)
                or getattr(unit, "description", lambda: "")()
                or ""
            ).strip()
            if not unit_text:
                continue
            unit_start, unit_end = _line_char_bounds(
                context_file.content,
                getattr(unit, "line", 1),
                getattr(unit, "end_line", getattr(unit, "line", 1)),
            )
            base_start = context_file.code_offset_start + unit_start
            base_end = context_file.code_offset_start + unit_end
            chunks = pack_text_by_lines(
                unit_text,
                provider=provider,
                chunk_token_budget=chunk_token_budget,
            )
            for chunk_index, chunk in enumerate(chunks):
                segments.append(
                    {
                        "text": chunk,
                        "offset_start": base_start,
                        "offset_end": base_end,
                        "metadata": {
                            "path": context_file.path,
                            "unit_name": getattr(unit, "name", None),
                            "unit_type": getattr(unit, "unit_type", None),
                            "language": getattr(unit, "language", None),
                            "unit_index": int(unit_index),
                            "chunk_index": int(chunk_index),
                            "segmenter": "colgrep",
                            "segmenter_backend": backend,
                        },
                    }
                )

    if segments:
        return segments

    baseline = segment_text(
        rendered_context,
        "sentence_packed",
        provider=provider,
        chunk_token_budget=chunk_token_budget,
    )
    for segment in baseline:
        metadata = dict(segment.get("metadata") or {})
        metadata["segmenter_backend"] = "empty_colgrep_result_fallback"
        segment["metadata"] = metadata
    return baseline
