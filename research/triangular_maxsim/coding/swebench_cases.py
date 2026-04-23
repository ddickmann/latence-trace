"""SWE-bench Lite adapter for the coding-agent groundedness benchmark."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import PurePosixPath
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from research.triangular_maxsim.coding.code_cases import CodeContextFile, CodingCase
from research.triangular_maxsim.coding.corruptions import CorruptionResult, make_corrupted_variant


_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_DIFF_HEADER_RE = re.compile(r"^diff --git a/(?P<old>.*?) b/(?P<new>.*?)$")
_HUNK_RE = re.compile(r"^@@ -(?P<old_start>\d+)(?:,(?P<old_count>\d+))? \+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@")
_PY_IMPORT_RE = re.compile(r"(?m)^(?:from\s+(?P<from>[.\w]+)\s+import\s+.+|import\s+(?P<import>[A-Za-z_][\w\.]*))$")
_JS_IMPORT_RE = re.compile(
    r"""(?m)^(?:import\s+.+?\s+from\s+["'](?P<from>[^"']+)["']|export\s+.+?\s+from\s+["'](?P<export>[^"']+)["'])"""
)
_CODE_EXTENSIONS = {
    ".c",
    ".cc",
    ".cpp",
    ".cs",
    ".go",
    ".h",
    ".hpp",
    ".java",
    ".js",
    ".jsx",
    ".kt",
    ".php",
    ".py",
    ".rb",
    ".rs",
    ".sql",
    ".swift",
    ".ts",
    ".tsx",
}


@dataclass(frozen=True)
class PatchFile:
    old_path: str
    new_path: str
    old_hunks: List[Tuple[int, int]]


def _estimate_tokens(text: str) -> int:
    return len(_TOKEN_RE.findall(text))


def _is_code_path(path: str) -> bool:
    suffix = PurePosixPath(path).suffix.lower()
    return suffix in _CODE_EXTENSIONS


def _parse_patch_files(patch: str) -> List[PatchFile]:
    files: List[PatchFile] = []
    current: Optional[PatchFile] = None
    for raw_line in patch.splitlines():
        header = _DIFF_HEADER_RE.match(raw_line)
        if header:
            if current is not None:
                files.append(current)
            current = PatchFile(
                old_path=header.group("old"),
                new_path=header.group("new"),
                old_hunks=[],
            )
            continue
        if current is None:
            continue
        hunk = _HUNK_RE.match(raw_line)
        if hunk:
            old_start = int(hunk.group("old_start"))
            old_count = int(hunk.group("old_count") or "1")
            current.old_hunks.append((old_start, max(1, old_count)))
    if current is not None:
        files.append(current)
    return files


def _merge_windows(windows: Sequence[Tuple[int, int]], *, line_count: int) -> List[Tuple[int, int]]:
    merged: List[Tuple[int, int]] = []
    for start, end in sorted(windows):
        safe_start = max(1, min(start, line_count))
        safe_end = max(safe_start, min(end, line_count))
        if not merged or safe_start > merged[-1][1] + 1:
            merged.append((safe_start, safe_end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], safe_end))
    return merged


def _trim_file_to_hunks(content: str, hunks: Sequence[Tuple[int, int]], *, padding: int = 40) -> str:
    lines = content.splitlines()
    if not lines:
        return content
    if not hunks:
        return "\n".join(lines[: min(len(lines), 160)])
    windows = []
    for old_start, old_count in hunks:
        start = max(1, old_start - padding)
        end = min(len(lines), old_start + max(1, old_count) - 1 + padding)
        windows.append((start, end))
    merged = _merge_windows(windows, line_count=len(lines))
    snippets: List[str] = []
    for start, end in merged:
        block = "\n".join(lines[start - 1 : end]).rstrip()
        snippets.append(f"# Lines {start}-{end}\n{block}")
    return "\n\n".join(snippets).strip() + "\n"


@lru_cache(maxsize=512)
def _fetch_repo_file(repo: str, commit: str, path: str) -> str:
    url = f"https://raw.githubusercontent.com/{repo}/{commit}/{path}"
    with urlopen(url, timeout=30) as response:
        return response.read().decode("utf-8")


def _resolve_python_import(base_path: str, import_target: str) -> Optional[str]:
    current = PurePosixPath(base_path)
    if import_target.startswith("."):
        dot_count = len(import_target) - len(import_target.lstrip("."))
        base_dir = current.parent
        for _ in range(max(0, dot_count - 1)):
            base_dir = base_dir.parent
        remainder = import_target[dot_count:].replace(".", "/")
        target = base_dir / remainder if remainder else base_dir
        return str(target.with_suffix(".py"))
    return str(PurePosixPath(import_target.replace(".", "/")).with_suffix(".py"))


def _resolve_js_import(base_path: str, import_target: str) -> List[str]:
    if not import_target.startswith("."):
        return []
    base_dir = PurePosixPath(base_path).parent
    target = (base_dir / import_target).as_posix()
    path = PurePosixPath(target)
    candidates = [path]
    if path.suffix:
        candidates.append(path.with_suffix(".ts"))
        candidates.append(path.with_suffix(".tsx"))
        candidates.append(path.with_suffix(".js"))
    else:
        candidates.extend(
            [
                PurePosixPath(f"{target}.ts"),
                PurePosixPath(f"{target}.tsx"),
                PurePosixPath(f"{target}.js"),
                PurePosixPath(f"{target}.jsx"),
                PurePosixPath(f"{target}/index.ts"),
                PurePosixPath(f"{target}/index.tsx"),
                PurePosixPath(f"{target}/index.js"),
            ]
        )
    return [candidate.as_posix() for candidate in candidates]


def _relative_import_candidates(path: str, content: str) -> List[str]:
    suffix = PurePosixPath(path).suffix.lower()
    candidates: List[str] = []
    seen = set()
    if suffix == ".py":
        for match in _PY_IMPORT_RE.finditer(content):
            target = match.group("from") or match.group("import")
            if not target:
                continue
            resolved = _resolve_python_import(path, target)
            if resolved and resolved not in seen and _is_code_path(resolved):
                seen.add(resolved)
                candidates.append(resolved)
    elif suffix in {".ts", ".tsx", ".js", ".jsx"}:
        for match in _JS_IMPORT_RE.finditer(content):
            target = match.group("from") or match.group("export")
            if not target:
                continue
            for resolved in _resolve_js_import(path, target):
                if resolved not in seen and _is_code_path(resolved):
                    seen.add(resolved)
                    candidates.append(resolved)
    return candidates


def _build_context_files(
    row: Dict[str, object],
    patch_files: Sequence[PatchFile],
    *,
    max_context_tokens: int,
    max_extra_imports: int,
) -> List[CodeContextFile]:
    repo = str(row["repo"])
    commit = str(row["base_commit"])
    context_files: List[CodeContextFile] = []
    used_paths = set()
    token_budget_used = 0

    for patch_file in patch_files:
        old_path = patch_file.old_path
        if old_path == "/dev/null" or not _is_code_path(old_path):
            continue
        try:
            full_content = _fetch_repo_file(repo, commit, old_path)
        except (HTTPError, URLError, TimeoutError, ValueError):
            continue
        trimmed = _trim_file_to_hunks(full_content, patch_file.old_hunks)
        estimated = _estimate_tokens(trimmed)
        if context_files and token_budget_used + estimated > max_context_tokens:
            break
        context_files.append(CodeContextFile(path=old_path, content=trimmed))
        used_paths.add(old_path)
        token_budget_used += estimated

        if max_extra_imports <= 0 or token_budget_used >= max_context_tokens:
            continue
        extra_added = 0
        for import_path in _relative_import_candidates(old_path, full_content):
            if import_path in used_paths:
                continue
            try:
                extra_content = _fetch_repo_file(repo, commit, import_path)
            except (HTTPError, URLError, TimeoutError, ValueError):
                continue
            trimmed_extra = _trim_file_to_hunks(extra_content, [], padding=0)
            extra_estimated = _estimate_tokens(trimmed_extra)
            if token_budget_used + extra_estimated > max_context_tokens:
                break
            context_files.append(CodeContextFile(path=import_path, content=trimmed_extra))
            used_paths.add(import_path)
            token_budget_used += extra_estimated
            extra_added += 1
            if extra_added >= max_extra_imports:
                break

    return context_files


def _load_dataset_rows(split: str) -> Iterable[Dict[str, object]]:
    try:
        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover - dependency error path
        raise RuntimeError(
            "datasets is required for SWE-bench loading. Install it with `pip install datasets`."
        ) from exc
    dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split=split)
    for row in dataset:
        yield dict(row)


def _build_pair(
    *,
    index: int,
    row: Dict[str, object],
    context_files: Sequence[CodeContextFile],
    corruption: CorruptionResult,
) -> List[CodingCase]:
    instance_id = str(row["instance_id"])
    source = f"SWE:{instance_id}"
    metadata = {
        "repo": str(row["repo"]),
        "instance_id": instance_id,
        "base_commit": str(row["base_commit"]),
        "hints_text": str(row.get("hints_text") or ""),
    }
    grounded = CodingCase(
        id=f"SWG{index}",
        source=source,
        query=str(row["problem_statement"]).strip(),
        response=str(row["patch"]).strip(),
        label="grounded",
        subcategory=None,
        notes="Gold SWE-bench Lite patch scored against the pre-change code windows for the touched files.",
        context_files=list(context_files),
        metadata=metadata,
    )
    ungrounded = replace(
        grounded,
        id=f"SWU{index}",
        response=corruption.text,
        label="ungrounded",
        subcategory=corruption.subcategory,
        notes=f"Programmatically corrupted gold patch via {corruption.strategy}. {corruption.note}",
        metadata={**metadata, "corruption_strategy": corruption.strategy},
    )
    return [grounded, ungrounded]


def load_swebench_cases(
    *,
    split: str = "dev",
    max_instances: int = 15,
    max_context_tokens: int = 8000,
    max_extra_imports: int = 2,
) -> List[CodingCase]:
    """Return grounded and ungrounded coding cases derived from SWE-bench Lite."""

    selected_pairs: List[CodingCase] = []
    pair_index = 0
    for row in _load_dataset_rows(split):
        patch = str(row.get("patch") or "")
        patch_files = [patch_file for patch_file in _parse_patch_files(patch) if _is_code_path(patch_file.old_path)]
        if not patch_files:
            continue
        try:
            context_files = _build_context_files(
                row,
                patch_files,
                max_context_tokens=max_context_tokens,
                max_extra_imports=max_extra_imports,
            )
        except RuntimeError:
            raise
        except Exception:
            continue
        if not context_files:
            continue
        try:
            corruption = make_corrupted_variant(patch, case_id=str(row["instance_id"]))
        except ValueError:
            continue
        pair_index += 1
        selected_pairs.extend(
            _build_pair(
                index=pair_index,
                row=row,
                context_files=context_files,
                corruption=corruption,
            )
        )
        if pair_index >= max_instances:
            break
    return selected_pairs
