"""Mine agent-transcripts for ~60 v2 base-scenario candidates.

For each ``*.jsonl`` transcript under
``/root/.cursor/projects/workspace/agent-transcripts/``, we scan every
assistant turn, check whether it contains at least one fenced code block
with enough substance, and emit a JSON shortlist sorted by a simple
"signal" score. The shortlist is the input for manual curation
in ``cases_transcripts_v2.yaml``.

Usage::

    python -m research.triangular_maxsim.coding.experiments.mine_v2_candidates \\
        --max-per-transcript 8 --top 60

Writes ``artifacts/v2_candidates.{json,md}``.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, "/workspace/latence-trace")

from research.triangular_maxsim.coding.transcript_cases import (  # noqa: E402
    _build_context_files,
    _CODE_FENCE_RE,
    _load_turns,
    _locate_transcript,
)

_TRANSCRIPT_ROOT = Path("/root/.cursor/projects/workspace/agent-transcripts")
_ARTIFACTS = Path("/workspace/latence-trace/research/triangular_maxsim/coding/artifacts")
_OUT_JSON = _ARTIFACTS / "v2_candidates.json"
_OUT_MD = _ARTIFACTS / "v2_candidates.md"

_MIN_FENCE_CHARS = 80
_MIN_CONTEXT_CHARS = 10_000
_MAX_CONTEXT_CHARS = 160_000
_MAX_FILES = 40
_MIN_PROSE_CHARS = 40


@dataclass
class Candidate:
    transcript_id: str
    response_turn: int
    fence_idx: int
    fence_lang: str
    fence_chars: int
    prose_chars: int
    context_files: int
    context_chars: int
    score: float
    fence_head: str
    prose_head: str


def _score(fence_chars: int, prose_chars: int, files: int, ctx_chars: int) -> float:
    """Heuristic: prefer turns with both substantive code *and* prose, rich
    context (>= 10k chars), and diverse files. Penalise huge transcripts that
    likely have thin per-turn context density."""

    if ctx_chars < _MIN_CONTEXT_CHARS:
        return 0.0
    if fence_chars < _MIN_FENCE_CHARS:
        return 0.0
    if prose_chars < _MIN_PROSE_CHARS:
        return 0.0
    fence_term = min(1.0, fence_chars / 1500.0)
    prose_term = min(1.0, prose_chars / 400.0)
    file_term = min(1.0, files / 20.0)
    ctx_term = min(1.0, ctx_chars / 80_000.0)
    return 0.35 * fence_term + 0.20 * prose_term + 0.20 * file_term + 0.25 * ctx_term


def _trim(text: str, limit: int) -> str:
    flat = re.sub(r"\s+", " ", text.strip())
    if len(flat) <= limit:
        return flat
    return flat[: limit - 1] + "..."


def _surrounding_prose(text: str, fence_start: int, fence_end: int) -> str:
    before_cut = max(0, fence_start - 600)
    after_cut = min(len(text), fence_end + 200)
    return text[before_cut:fence_start] + " / " + text[fence_end:after_cut]


def mine_transcript(
    transcript_id: str,
    *,
    max_candidates: int,
) -> List[Candidate]:
    path = _locate_transcript(transcript_id)
    turns = _load_turns(path)
    candidates: List[Candidate] = []

    for turn_index, turn in enumerate(turns):
        if turn["role"] != "assistant":
            continue
        text = turn["text"]
        fences = list(_CODE_FENCE_RE.finditer(text))
        if not fences:
            continue

        # Build the context files once per (transcript, response_turn).
        try:
            context_files = _build_context_files(
                turns,
                cutoff_turn=turn_index,
                max_files=_MAX_FILES,
                token_cap_chars=_MAX_CONTEXT_CHARS,
            )
        except ValueError:
            continue
        ctx_chars = sum(len(f.content) for f in context_files)
        if ctx_chars < _MIN_CONTEXT_CHARS:
            continue

        for fence_idx, fence in enumerate(fences):
            body = fence.group(2)
            lang = fence.group(1) or ""
            fence_chars = len(body)
            if fence_chars < _MIN_FENCE_CHARS:
                continue
            prose = _surrounding_prose(text, fence.start(), fence.end())
            prose_chars = len(prose.strip())
            s = _score(fence_chars, prose_chars, len(context_files), ctx_chars)
            if s <= 0:
                continue
            candidates.append(
                Candidate(
                    transcript_id=transcript_id,
                    response_turn=turn_index,
                    fence_idx=fence_idx,
                    fence_lang=lang,
                    fence_chars=fence_chars,
                    prose_chars=prose_chars,
                    context_files=len(context_files),
                    context_chars=ctx_chars,
                    score=s,
                    fence_head=_trim(body, 200),
                    prose_head=_trim(prose, 200),
                )
            )
    candidates.sort(key=lambda c: c.score, reverse=True)
    # Enforce a per-(transcript, response_turn) diversity: keep the top fence
    # per turn only, to spread bases across sessions.
    seen: set = set()
    unique: List[Candidate] = []
    for c in candidates:
        key = (c.transcript_id, c.response_turn)
        if key in seen:
            continue
        seen.add(key)
        unique.append(c)
        if len(unique) >= max_candidates:
            break
    return unique


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-per-transcript", type=int, default=6)
    parser.add_argument("--top", type=int, default=80)
    args = parser.parse_args()

    all_candidates: List[Candidate] = []
    for directory in sorted(_TRANSCRIPT_ROOT.iterdir()):
        if not directory.is_dir():
            continue
        tid = directory.name.split("-")[0]
        try:
            cs = mine_transcript(tid, max_candidates=args.max_per_transcript)
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] {tid}: {type(exc).__name__}: {exc}")
            continue
        print(f"[ok] {tid}: {len(cs)} candidates (best score={cs[0].score:.3f} if any)" if cs else f"[ok] {tid}: 0")
        all_candidates.extend(cs)

    all_candidates.sort(key=lambda c: c.score, reverse=True)
    all_candidates = all_candidates[: args.top]

    _ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with _OUT_JSON.open("w") as handle:
        json.dump([asdict(c) for c in all_candidates], handle, indent=2)

    lines: List[str] = ["# v2 candidate shortlist", ""]
    lines.append(f"- total kept: **{len(all_candidates)}**")
    lines.append("")
    lines.append("| # | transcript | turn | fence | lang | fence chars | prose | files | ctx chars | score | fence head |")
    lines.append("|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---|")
    for idx, c in enumerate(all_candidates, 1):
        lines.append(
            "| {i} | {t} | {rt} | {fi} | {lang} | {fc} | {pc} | {files} | {ctx} | {s:.3f} | `{head}` |".format(
                i=idx,
                t=c.transcript_id,
                rt=c.response_turn,
                fi=c.fence_idx,
                lang=c.fence_lang or "-",
                fc=c.fence_chars,
                pc=c.prose_chars,
                files=c.context_files,
                ctx=c.context_chars,
                s=c.score,
                head=c.fence_head[:90].replace("|", "/"),
            )
        )
    _OUT_MD.write_text("\n".join(lines))
    print(f"Wrote {_OUT_JSON}")
    print(f"Wrote {_OUT_MD}")


if __name__ == "__main__":
    main()
