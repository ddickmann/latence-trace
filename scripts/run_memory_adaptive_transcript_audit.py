#!/usr/bin/env python
"""Audit adaptive TRACE Memory budgets on Cursor transcript histories."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

from latence_trace.memory.extract import extract_spans
from latence_trace.memory.models import MemoryPolicy, SpanRecord
from latence_trace.memory.select import assign_layers, hot_context
from latence_trace.memory.signature import extract_exact_critical_terms
from latence_trace.memory.survival import update_survival


DEFAULT_TRANSCRIPT_ROOT = Path("/root/.cursor/projects/workspace/agent-transcripts")


def main() -> None:
    args = _parse_args()
    start = time.perf_counter()
    transcript_paths = _select_transcripts(args)
    rows = []
    failures = []
    for idx, path in enumerate(transcript_paths, start=1):
        try:
            row = _evaluate_transcript(path, args)
            rows.append(row)
        except Exception as exc:  # noqa: BLE001 - audit must report all failures
            failures.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
        if args.progress_every > 0 and idx % args.progress_every == 0:
            print(
                json.dumps(
                    {
                        "event": "progress",
                        "processed": idx,
                        "rows": len(rows),
                        "failures": len(failures),
                        "elapsed_s": round(time.perf_counter() - start, 2),
                    }
                ),
                flush=True,
            )
    report = {
        "summary": _summary(rows, failures, started_at=start),
        "rows": rows,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "event": "done",
                "rows": len(rows),
                "failures": len(failures),
                "output": str(args.output),
                "elapsed_s": round(time.perf_counter() - start, 2),
            }
        )
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transcript-root", type=Path, default=DEFAULT_TRANSCRIPT_ROOT)
    parser.add_argument("--output", type=Path, default=Path("/tmp/memory_adaptive_transcript_audit.json"))
    parser.add_argument("--include-subagents", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--chunk-tokens", type=int, default=6_000)
    parser.add_argument("--fixed-budgets", default="800,1200,2700,5400,13500,27000")
    parser.add_argument("--context-window-tokens", type=int, default=270_000)
    parser.add_argument("--memory-context-ratio", type=float, default=0.10)
    parser.add_argument("--target-token-reduction", type=float, default=0.90)
    parser.add_argument("--min-exact-critical-recall", type=float, default=0.98)
    parser.add_argument("--min-survival-mass", type=float, default=0.95)
    parser.add_argument("--recent-tail-token-budget", type=int, default=4_096)
    parser.add_argument("--stress-long-traces", type=int, default=0)
    parser.add_argument("--min-stress-tokens", type=int, default=200_000)
    parser.add_argument("--progress-every", type=int, default=10)
    return parser.parse_args()


def _select_transcripts(args: argparse.Namespace) -> list[Path]:
    paths = sorted(args.transcript_root.glob("**/*.jsonl"))
    if not args.include_subagents:
        paths = [path for path in paths if "subagents" not in path.parts]
    if args.stress_long_traces > 0:
        paths = sorted(paths, key=_token_estimate, reverse=True)[: args.stress_long_traces]
    elif args.limit > 0:
        paths = paths[: args.limit]
    return paths


def _token_estimate(path: Path) -> int:
    try:
        return len(path.read_text(errors="ignore").split())
    except OSError:
        return 0


def _evaluate_transcript(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    timings: dict[str, float] = {}
    load_start = time.perf_counter()
    messages = _load_messages(path)
    full_text = "\n".join(text for _, text in messages)
    if args.stress_long_traces > 0 and len(full_text.split()) < args.min_stress_tokens:
        full_text = _extend_trace(full_text, args.min_stress_tokens)
    full_tokens = len(full_text.split())
    user_texts = [text for role, text in messages if role == "user"]
    timings["load"] = _elapsed_ms(load_start)

    terms_start = time.perf_counter()
    all_user_terms = _critical_terms("\n".join(user_texts))
    last_user_terms = _critical_terms(user_texts[-1] if user_texts else "")
    timings["term_extraction"] = _elapsed_ms(terms_start)

    extract_start = time.perf_counter()
    spans = _extract_chunked_spans(full_text, args.chunk_tokens)
    timings["span_extraction"] = _elapsed_ms(extract_start)

    dedupe_start = time.perf_counter()
    spans = _dedupe_by_id(spans)
    timings["dedupe"] = _elapsed_ms(dedupe_start)

    survival_start = time.perf_counter()
    scored, _ = update_survival(
        spans,
        turn_index=max(1, full_tokens // max(1, args.chunk_tokens)),
        policy=MemoryPolicy(),
        trace_signals={"trajectory_domain": "code"},
    )
    timings["survival"] = _elapsed_ms(survival_start)

    selection_start = time.perf_counter()
    fixed = [
        _evaluate_policy(
            scored,
            full_text,
            all_user_terms,
            last_user_terms,
            MemoryPolicy(
                hot_token_budget=budget,
                warm_token_budget=max(4_000, budget * 2),
                max_spans=max(256, len(scored)),
                memory_budget_mode="fixed",
            ),
        )
        for budget in _fixed_budgets(args.fixed_budgets)
    ]
    adaptive_policy = MemoryPolicy(
        hot_token_budget=min(_fixed_budgets(args.fixed_budgets)),
        warm_token_budget=4_000,
        max_spans=max(256, len(scored)),
        memory_budget_mode="adaptive",
        context_window_tokens=args.context_window_tokens,
        memory_context_ratio=args.memory_context_ratio,
        target_token_reduction=args.target_token_reduction,
        min_exact_critical_recall=args.min_exact_critical_recall,
        min_survival_mass=args.min_survival_mass,
        recent_tail_token_budget=args.recent_tail_token_budget,
    )
    adaptive = _evaluate_policy(scored, full_text, all_user_terms, last_user_terms, adaptive_policy)
    timings["selection_and_metrics"] = _elapsed_ms(selection_start)
    timings["total"] = round(sum(timings.values()), 3)

    status = _status(adaptive, args.min_exact_critical_recall)
    return {
        "path": str(path),
        "title": _title(messages),
        "messages": len(messages),
        "full_tokens": full_tokens,
        "span_count": len(scored),
        "all_user_term_count": len(all_user_terms),
        "last_user_term_count": len(last_user_terms),
        "status": status,
        "adaptive": adaptive,
        "fixed": fixed,
        "timings_ms": timings,
    }


def _load_messages(path: Path) -> list[tuple[str, str]]:
    messages = []
    with path.open(errors="ignore") as handle:
        for line in handle:
            if not line.strip():
                continue
            obj = json.loads(line)
            role = str(obj.get("role") or "")
            content = (obj.get("message") or {}).get("content", "")
            text = _flatten_text(content).strip()
            if text:
                messages.append((role, text))
    return messages


def _flatten_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(_flatten_text(item) for item in value)
    if isinstance(value, dict):
        if value.get("type") == "text" and isinstance(value.get("text"), str):
            return value["text"]
        return "\n".join(
            _flatten_text(item)
            for key, item in value.items()
            if key in {"content", "message", "result", "summary", "text"} or isinstance(item, str | list)
        )
    return ""


def _extend_trace(text: str, min_tokens: int) -> str:
    tokens = text.split()
    if not tokens:
        return text
    repeats = max(1, (min_tokens + len(tokens) - 1) // len(tokens))
    return "\n".join(f"stress_replay_{idx}\n{text}" for idx in range(repeats))


def _critical_terms(text: str) -> list[str]:
    terms = set(extract_exact_critical_terms(text, domain="code"))
    terms.update(extract_exact_critical_terms(text, domain="tool"))
    return sorted(term for term in terms if "[REDACTED]" not in term and len(term) <= 180)


def _extract_chunked_spans(text: str, chunk_tokens: int) -> list[SpanRecord]:
    spans: list[SpanRecord] = []
    tokens = text.split()
    for idx in range(0, len(tokens), chunk_tokens):
        chunk = " ".join(tokens[idx : idx + chunk_tokens])
        spans.extend(extract_spans(raw_context=chunk, turn_index=idx // chunk_tokens + 1, memory_domain="code"))
    return spans


def _dedupe_by_id(spans: list[SpanRecord]) -> list[SpanRecord]:
    by_id: dict[str, SpanRecord] = {}
    for span in spans:
        existing = by_id.get(span.id)
        if existing is None:
            by_id[span.id] = span
        else:
            existing.last_seen_turn = max(existing.last_seen_turn, span.last_seen_turn)
            existing.scores.redundancy = max(existing.scores.redundancy, 0.7)
    return list(by_id.values())


def _evaluate_policy(
    spans: list[SpanRecord],
    full_text: str,
    all_user_terms: list[str],
    last_user_terms: list[str],
    policy: MemoryPolicy,
) -> dict[str, Any]:
    selected, diagnostics = assign_layers(spans, policy=policy)
    hot = hot_context(selected)
    tail = _tail_text(full_text, diagnostics.effective_hot_token_budget)
    return {
        "mode": diagnostics.budget_mode_used,
        "hot_tokens": len(hot.split()),
        "effective_hot_token_budget": diagnostics.effective_hot_token_budget,
        "target_token_reduction": diagnostics.target_token_reduction,
        "actual_token_reduction": diagnostics.actual_token_reduction,
        "estimated_exact_critical_recall": diagnostics.estimated_exact_critical_recall,
        "survival_mass_retained": diagnostics.survival_mass_retained,
        "memory_underbudgeted": diagnostics.memory_underbudgeted,
        "recommended_hot_token_budget": diagnostics.recommended_hot_token_budget,
        "recent_tail_required": diagnostics.recent_tail_required,
        "last_user_recall": _recall(last_user_terms, hot),
        "all_user_recall": _recall(all_user_terms, hot),
        "tail_last_user_recall": _recall(last_user_terms, tail),
    }


def _status(adaptive: dict[str, Any], min_recall: float) -> str:
    if not adaptive["memory_underbudgeted"] and (adaptive["last_user_recall"] or 1.0) >= min_recall:
        return "safe"
    if (adaptive["tail_last_user_recall"] or 0.0) >= min_recall:
        return "needs_recent_tail"
    return "needs_more_memory"


def _recall(terms: list[str], text: str) -> float | None:
    if not terms:
        return None
    return round(sum(1 for term in terms if term in text) / len(terms), 4)


def _tail_text(text: str, budget: int) -> str:
    tokens = text.split()
    return " ".join(tokens[-budget:])


def _fixed_budgets(raw: str) -> list[int]:
    return sorted({int(item) for item in raw.split(",") if item.strip()})


def _title(messages: list[tuple[str, str]]) -> str:
    for role, text in messages:
        if role == "user":
            clean = re.sub(r"<[^>]+>", " ", text)
            return " ".join(clean.split()[:8])
    return "untitled"


def _summary(rows: list[dict[str, Any]], failures: list[dict[str, Any]], *, started_at: float) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    for row in rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
    return {
        "case_count": len(rows),
        "failure_count": len(failures),
        "status_counts": status_counts,
        "avg_adaptive_hot_tokens": _mean(row["adaptive"]["hot_tokens"] for row in rows),
        "avg_adaptive_last_user_recall": _mean(
            row["adaptive"]["last_user_recall"]
            for row in rows
            if row["adaptive"]["last_user_recall"] is not None
        ),
        "avg_total_ms": _mean(row["timings_ms"]["total"] for row in rows),
        "elapsed_s": round(time.perf_counter() - started_at, 2),
    }


def _mean(values: Any) -> float:
    items = list(values)
    if not items:
        return 0.0
    return round(sum(float(item) for item in items) / len(items), 4)


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000.0, 3)


if __name__ == "__main__":
    main()
