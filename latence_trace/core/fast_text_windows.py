"""Fast text windowing shared by compression and context-trust runtimes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FastTextWindow:
    text: str
    start: int
    end: int
    window_index: int
    total_windows: int
    source: str
    token_count: int | None = None


def fast_text_windows(
    text: str,
    *,
    chunk_size: int,
    text_processing: Any | None = None,
) -> list[FastTextWindow]:
    """Split text with the Rust FastPreprocessor when available.

    The Rust wheel is the production path used by TRACE compression. The Python
    fallback only keeps local/dev and test environments usable when that wheel is
    not installed.
    """

    if not (text or "").strip():
        return []
    safe_chunk_size = min(max(512, int(chunk_size)), 7500)
    runtime = text_processing
    if runtime is None:
        try:
            import text_processing as runtime  # type: ignore[no-redef]
        except Exception:
            runtime = None

    if runtime is not None and hasattr(runtime, "FastPreprocessor"):
        try:
            preprocessor = runtime.FastPreprocessor(safe_chunk_size)
            raw_windows = preprocessor.process_single_text(text)
            windows = _windows_from_rust_chunks(
                text,
                raw_windows,
                source="rust_fast_preprocessor",
            )
            if windows:
                return windows
        except Exception:
            pass

    return _fallback_windows(text, max_chars=safe_chunk_size)


def _windows_from_rust_chunks(
    text: str,
    raw_windows: Any,
    *,
    source: str,
) -> list[FastTextWindow]:
    windows: list[FastTextWindow] = []
    text_len = len(text)
    for raw in list(raw_windows or []):
        start = max(0, min(text_len, int(getattr(raw, "start", 0))))
        end = max(start, min(text_len, int(getattr(raw, "end", start))))
        chunk_text = text[start:end]
        if not chunk_text.strip():
            continue
        token_count = getattr(raw, "token_count", None)
        windows.append(
            FastTextWindow(
                text=chunk_text,
                start=start,
                end=end,
                window_index=len(windows),
                total_windows=0,
                source=source,
                token_count=int(token_count) if token_count is not None else None,
            )
        )
    return _with_totals(windows)


def _fallback_windows(text: str, *, max_chars: int) -> list[FastTextWindow]:
    windows: list[FastTextWindow] = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(text_len, start + max_chars)
        if end < text_len:
            for sep in (". ", "! ", "? ", "\n\n", "\n", " "):
                split_at = text.rfind(sep, start + 1, end + 1)
                if split_at > start:
                    end = split_at + len(sep)
                    break
        chunk_text = text[start:end]
        if chunk_text.strip():
            windows.append(
                FastTextWindow(
                    text=chunk_text,
                    start=start,
                    end=end,
                    window_index=len(windows),
                    total_windows=0,
                    source="python_fallback",
                    token_count=max(1, len(chunk_text.split())),
                )
            )
        if end <= start:
            break
        start = end
    return _with_totals(windows)


def _with_totals(windows: list[FastTextWindow]) -> list[FastTextWindow]:
    total = len(windows)
    return [
        FastTextWindow(
            text=window.text,
            start=window.start,
            end=window.end,
            window_index=idx,
            total_windows=total,
            source=window.source,
            token_count=window.token_count,
        )
        for idx, window in enumerate(windows)
    ]
