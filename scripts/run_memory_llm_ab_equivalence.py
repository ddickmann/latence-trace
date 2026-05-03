#!/usr/bin/env python
"""Run an LLM A/B equivalence gate for full vs TRACE-compressed context."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
TRANSCRIPT_ROOT = Path("/root/.cursor/projects/workspace/agent-transcripts")
EXTERNAL_DATA = REPO_ROOT / "research" / "triangular_maxsim" / "external_data"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "research" / "trace_memory" / "llm_ab_equivalence"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from latence_trace.memory.extract import extract_spans  # noqa: E402
from latence_trace.memory.models import MemoryDiagnostics, MemoryPolicy, SpanRecord  # noqa: E402
from latence_trace.memory.select import assign_layers, hot_context  # noqa: E402
from latence_trace.memory.signature import extract_exact_critical_terms  # noqa: E402
from latence_trace.memory.survival import update_survival  # noqa: E402


@dataclass
class EvalCase:
    case_id: str
    case_type: str
    source: str
    title: str
    task: str
    full_context: str
    compressed_context: str
    required_terms: list[str]
    diagnostics: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnswerPair:
    full_answer: str
    compressed_answer: str
    full_seconds: float
    compressed_seconds: float
    full_first: bool


def main() -> int:
    args = _parse_args()
    rng = random.Random(args.seed)
    output_dir = _output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = build_cases(args, rng)
    case_rows = [_case_summary(case) for case in cases]
    (output_dir / "cases.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in case_rows) + "\n",
        encoding="utf-8",
    )
    if args.dry_run:
        report = _dry_run_report(case_rows, args)
        (output_dir / "report.md").write_text(report, encoding="utf-8")
        print(json.dumps({"event": "dry_run_done", "cases": len(cases), "output_dir": str(output_dir)}))
        return 0

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required for live A/B calls.")

    client = _openai_client()
    results = []
    with (output_dir / "results.jsonl").open("w", encoding="utf-8") as handle:
        for index, case in enumerate(cases, start=1):
            print(json.dumps({"event": "case_start", "index": index, "case_id": case.case_id}), flush=True)
            try:
                result = run_case(case, args, rng, client)
            except Exception as exc:  # noqa: BLE001 - complete the full gate and report failures
                result = {
                    "case": _case_summary(case),
                    "classification": "api_failure",
                    "error": f"{type(exc).__name__}: {exc}",
                    "judge": {},
                    "side_checks": {},
                    "repair": {"triggered": False},
                    "answers": {},
                }
            results.append(result)
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                json.dumps(
                    {
                        "event": "case_done",
                        "index": index,
                        "case_id": case.case_id,
                        "classification": result["classification"],
                    }
                ),
                flush=True,
            )

    report = _final_report(results, args)
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"event": "done", "cases": len(results), "output_dir": str(output_dir)}))
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=20260503)
    parser.add_argument("--case-count", type=int, default=20)
    parser.add_argument("--trace-cases", type=int, default=12)
    parser.add_argument("--rag-cases", type=int, default=8)
    parser.add_argument(
        "--case-ids",
        default="",
        help="Comma-separated case ids to run after deterministic case construction.",
    )
    parser.add_argument("--transcript-root", type=Path, default=TRANSCRIPT_ROOT)
    parser.add_argument("--external-data-root", type=Path, default=EXTERNAL_DATA)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--verbosity", default="medium")
    parser.add_argument("--max-output-tokens", type=int, default=1600)
    parser.add_argument("--chunk-tokens", type=int, default=6000)
    parser.add_argument("--context-window-tokens", type=int, default=270_000)
    parser.add_argument("--memory-context-ratio", type=float, default=0.40)
    parser.add_argument("--target-token-reduction", type=float, default=0.90)
    parser.add_argument("--recent-tail-token-budget", type=int, default=4096)
    parser.add_argument("--max-full-context-tokens", type=int, default=0)
    parser.add_argument("--max-judge-context-tokens", type=int, default=60_000)
    parser.add_argument("--min-transcript-tokens", type=int, default=25_000)
    parser.add_argument("--enable-repair-loop", action="store_true")
    parser.add_argument("--repair-max-tokens", type=int, default=6_000)
    parser.add_argument("--repair-max-excerpts", type=int, default=8)
    parser.add_argument("--repair-min-context-recall", type=float, default=0.98)
    parser.add_argument("--progress-every", type=int, default=1)
    return parser.parse_args()


def _output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    suffix = "dry_run" if args.dry_run else "live"
    return DEFAULT_OUTPUT_ROOT / f"{stamp}_{suffix}"


def build_cases(args: argparse.Namespace, rng: random.Random) -> list[EvalCase]:
    trace_count = min(args.trace_cases, args.case_count)
    rag_count = min(args.rag_cases, args.case_count - trace_count)
    cases = _build_trace_cases(args, rng, trace_count)
    cases.extend(_build_rag_cases(args, rng, rag_count))
    target_ids = _target_case_ids(args)
    if target_ids:
        by_id = {case.case_id: case for case in cases}
        missing = sorted(target_ids - set(by_id))
        if missing:
            raise RuntimeError(f"requested case ids were not built: {missing}")
        return [by_id[case_id] for case_id in sorted(target_ids)]
    if len(cases) < args.case_count:
        raise RuntimeError(f"built {len(cases)} cases, expected {args.case_count}")
    rng.shuffle(cases)
    return cases[: args.case_count]


def _target_case_ids(args: argparse.Namespace) -> set[str]:
    raw = str(getattr(args, "case_ids", "") or "")
    return {item.strip() for item in raw.split(",") if item.strip()}


def _build_trace_cases(args: argparse.Namespace, rng: random.Random, count: int) -> list[EvalCase]:
    paths = [
        path
        for path in sorted(args.transcript_root.glob("**/*.jsonl"))
        if "subagents" not in path.parts
    ]
    scored_paths = []
    for path in paths:
        estimate = _path_token_estimate(path)
        if estimate >= args.min_transcript_tokens:
            scored_paths.append((estimate, path))
    scored_paths.sort(reverse=True, key=lambda item: item[0])
    if not scored_paths:
        raise RuntimeError(f"no long parent transcripts found under {args.transcript_root}")

    cases: list[EvalCase] = []
    positions = ["early", "middle", "late"]
    transcript_pool = [path for _, path in scored_paths[: max(8, count)]]
    combos = [(path, position) for path in transcript_pool for position in positions]
    rng.shuffle(combos)
    for path, position in combos:
        if len(cases) >= count:
            break
        messages = _load_messages(path)
        user_indices = [idx for idx, (role, text) in enumerate(messages) if role == "user" and len(text.split()) >= 8]
        if len(user_indices) < 3:
            continue
        target_index = _positioned_index(user_indices, position)
        target_text = messages[target_index][1]
        full_context = _format_messages(messages)
        full_context = _maybe_bound_context(full_context, args.max_full_context_tokens, anchor=target_text)
        compressed_context, diagnostics = _compress_context(
            full_context,
            domain="code",
            query_text=target_text,
            args=args,
        )
        task = (
            "You are auditing a long coding-agent trace. Using only the provided context, "
            "answer the target user turn below. State the concrete request, constraints, "
            "implementation state or next step, and any exact files, symbols, commands, IDs, "
            "or tests that matter. Do not invent facts.\n\n"
            f"TARGET_USER_TURN ({position}):\n{target_text}"
        )
        required_terms = _required_terms(target_text, "code")
        case_id = f"trace_{len(cases):02d}_{position}_{_hash_text(str(path))[:8]}"
        cases.append(
            EvalCase(
                case_id=case_id,
                case_type="coding_trace",
                source=str(path),
                title=_title(messages),
                task=task,
                full_context=full_context,
                compressed_context=compressed_context,
                required_terms=required_terms,
                diagnostics=diagnostics,
                metadata={
                    "position": position,
                    "target_message_index": target_index,
                    "message_count": len(messages),
                    "transcript_tokens": _path_token_estimate(path),
                },
            )
        )
    if len(cases) < count:
        raise RuntimeError(f"built {len(cases)} trace cases, expected {count}")
    return cases


def _build_rag_cases(args: argparse.Namespace, rng: random.Random, count: int) -> list[EvalCase]:
    rows = _load_rag_rows(args.external_data_root, rng)
    if len(rows) < count:
        raise RuntimeError(f"built {len(rows)} RAG rows, expected at least {count}")

    cases: list[EvalCase] = []
    by_bench: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_bench.setdefault(row["bench"], []).append(row)
    bench_cycle = ["halueval_qa", "halueval_summ", "ragtruth_qa", "ragtruth_summary"]
    for bench in bench_cycle:
        rng.shuffle(by_bench.get(bench, []))

    cursor = dict.fromkeys(bench_cycle, 0)
    while len(cases) < count:
        bench = bench_cycle[len(cases) % len(bench_cycle)]
        bench_rows = by_bench.get(bench) or []
        if not bench_rows:
            bench = next((item for item in bench_cycle if by_bench.get(item)), "")
            bench_rows = by_bench.get(bench) or []
        if not bench_rows:
            break
        row = bench_rows[cursor[bench] % len(bench_rows)]
        cursor[bench] += 1
        distractors = _rag_distractors(rows, row, rng, count=3)
        full_context = _rag_context_with_distractors(row, distractors)
        full_context = _maybe_bound_context(full_context, args.max_full_context_tokens, anchor=row["question"])
        query_text = f"{row['question']}\nCandidate answer: {row['response_text']}"
        compressed_context, diagnostics = _compress_context(
            full_context,
            domain="rag",
            query_text=query_text,
            args=args,
        )
        task = (
            "You are verifying a RAG answer. Using only the provided context, decide whether "
            "the candidate answer is fully supported. If supported, answer 'supported' and cite "
            "the key facts. If not supported, answer 'unsupported' and list the unsupported or "
            "contradicted claims. Do not use outside knowledge.\n\n"
            f"QUESTION:\n{row['question']}\n\n"
            f"CANDIDATE_ANSWER:\n{row['response_text']}"
        )
        required_terms = _required_terms(row["question"] + "\n" + row["response_text"], "rag")
        case_id = f"rag_{len(cases):02d}_{row['bench']}_{row['row_id']}"
        cases.append(
            EvalCase(
                case_id=case_id,
                case_type="rag_verification",
                source=row["bench"],
                title=row["row_id"],
                task=task,
                full_context=full_context,
                compressed_context=compressed_context,
                required_terms=required_terms,
                diagnostics=diagnostics,
                metadata={
                    "bench": row["bench"],
                    "row_id": row["row_id"],
                    "expected_label": row.get("expected_label"),
                    "expected_band": row.get("expected_band"),
                    "distractor_count": len(distractors),
                },
            )
        )
    if len(cases) < count:
        raise RuntimeError(f"built {len(cases)} RAG cases, expected {count}")
    return cases


def _compress_context(
    text: str,
    *,
    domain: str,
    query_text: str,
    args: argparse.Namespace,
) -> tuple[str, dict[str, Any]]:
    spans = _extract_chunked_spans(text, args.chunk_tokens, domain=domain)
    spans = _dedupe_spans(spans)
    policy = MemoryPolicy(
        hot_token_budget=800,
        warm_token_budget=4000,
        max_spans=max(256, len(spans)),
        memory_budget_mode="adaptive",
        context_window_tokens=args.context_window_tokens,
        memory_context_ratio=args.memory_context_ratio,
        target_token_reduction=args.target_token_reduction,
        min_exact_critical_recall=0.98,
        min_survival_mass=0.95,
        recent_tail_token_budget=args.recent_tail_token_budget,
    )
    scored, _ = update_survival(
        spans,
        turn_index=max(1, _token_count(text) // max(1, args.chunk_tokens)),
        policy=policy,
        trace_signals={"trajectory_domain": domain, "query_text": query_text},
    )
    selected, diagnostics = assign_layers(scored, policy=policy)
    compressed = hot_context(selected)
    if diagnostics.recent_tail_required:
        compressed = f"{compressed}\n\n[RECENT_TAIL]\n{_tail_text(text, args.recent_tail_token_budget)}"
    diag = diagnostics.model_dump(mode="json") if isinstance(diagnostics, MemoryDiagnostics) else {}
    diag["span_count"] = len(scored)
    diag["used_recent_tail"] = diagnostics.recent_tail_required
    return compressed, diag


def _extract_chunked_spans(text: str, chunk_tokens: int, *, domain: str) -> list[SpanRecord]:
    spans: list[SpanRecord] = []
    tokens = text.split()
    for idx in range(0, len(tokens), chunk_tokens):
        chunk = " ".join(tokens[idx : idx + chunk_tokens])
        spans.extend(extract_spans(raw_context=chunk, turn_index=idx // chunk_tokens + 1, memory_domain=domain))
    return spans


def _dedupe_spans(spans: list[SpanRecord]) -> list[SpanRecord]:
    by_id: dict[str, SpanRecord] = {}
    for span in spans:
        existing = by_id.get(span.id)
        if existing is None:
            by_id[span.id] = span
            continue
        existing.last_seen_turn = max(existing.last_seen_turn, span.last_seen_turn)
        existing.scores.redundancy = max(existing.scores.redundancy, 0.7)
    return list(by_id.values())


def run_case(case: EvalCase, args: argparse.Namespace, rng: random.Random, client: Any) -> dict[str, Any]:
    full_first = rng.choice([True, False])
    if full_first:
        full_answer, full_seconds = _call_answer_model(client, args, case, "full", case.full_context)
        compressed_answer, compressed_seconds = _call_answer_model(
            client,
            args,
            case,
            "compressed",
            case.compressed_context,
        )
    else:
        compressed_answer, compressed_seconds = _call_answer_model(
            client,
            args,
            case,
            "compressed",
            case.compressed_context,
        )
        full_answer, full_seconds = _call_answer_model(client, args, case, "full", case.full_context)

    pair = AnswerPair(
        full_answer=full_answer,
        compressed_answer=compressed_answer,
        full_seconds=full_seconds,
        compressed_seconds=compressed_seconds,
        full_first=full_first,
    )
    repair_info: dict[str, Any] = {"triggered": False}
    initial_side_checks = _side_checks(case, pair)
    repair_triggers = _repair_trigger_reasons(case, pair, initial_side_checks, args)
    if args.enable_repair_loop and repair_triggers:
        repair_context, repair_excerpts = _build_repair_context(case, pair, initial_side_checks, args)
        repaired_answer, repaired_seconds = _call_answer_model(
            client,
            args,
            case,
            "compressed_repaired",
            f"{case.compressed_context}\n\n[SOURCE_VAULT_REPAIR_EXCERPTS]\n{repair_context}",
        )
        repair_info = {
            "triggered": True,
            "triggers": repair_triggers,
            "excerpt_count": len(repair_excerpts),
            "repair_tokens": _token_count(repair_context),
            "compressed_initial": compressed_answer,
            "compressed_initial_seconds": round(compressed_seconds, 3),
            "compressed_repaired_seconds": round(repaired_seconds, 3),
            "excerpts": repair_excerpts,
        }
        compressed_answer = repaired_answer
        compressed_seconds += repaired_seconds
        pair = AnswerPair(
            full_answer=full_answer,
            compressed_answer=compressed_answer,
            full_seconds=full_seconds,
            compressed_seconds=compressed_seconds,
            full_first=full_first,
        )
    judge = _judge_pair(client, args, case, pair, rng)
    side_checks = _side_checks(case, pair)
    classification = _classification_from_judge(judge, side_checks)
    return {
        "case": _case_summary(case),
        "classification": classification,
        "judge": judge,
        "side_checks": side_checks,
        "repair": repair_info,
        "answers": {
            "full": full_answer,
            "compressed": compressed_answer,
            "full_seconds": round(full_seconds, 3),
            "compressed_seconds": round(compressed_seconds, 3),
            "full_first": full_first,
        },
    }


def _call_answer_model(
    client: Any,
    args: argparse.Namespace,
    case: EvalCase,
    arm: str,
    context: str,
) -> tuple[str, float]:
    prompt = (
        "You are a careful production assistant. Answer the task using only the context. "
        "Prefer concise, factual answers with exact identifiers when present.\n\n"
        f"TASK:\n{case.task}\n\n"
        "CRITICAL_EXACT_TERMS_TO_PRESERVE_IF_SUPPORTED:\n"
        f"{json.dumps(case.required_terms, ensure_ascii=False)}\n\n"
        f"CONTEXT_ARM: {arm}\n"
        f"CONTEXT:\n{context}"
    )
    if "repaired" in arm:
        prompt = (
            "You are in a source-vault repair pass. The compressed memory may have omitted "
            "or blurred details. Treat SOURCE_VAULT_REPAIR_EXCERPTS as higher-priority "
            "evidence. If the excerpts contain later implementation state, run results, "
            "artifacts, or exact paths/symbols, include them. Do not invent facts.\n\n"
            + prompt
        )
    start = time.perf_counter()
    response = _responses_create(client, args, prompt)
    return _response_text(response), time.perf_counter() - start


def _judge_pair(
    client: Any,
    args: argparse.Namespace,
    case: EvalCase,
    pair: AnswerPair,
    rng: random.Random,
) -> dict[str, Any]:
    compressed_is_a = rng.choice([True, False])
    answer_a = pair.compressed_answer if compressed_is_a else pair.full_answer
    answer_b = pair.full_answer if compressed_is_a else pair.compressed_answer
    source = _maybe_bound_context(
        case.full_context,
        args.max_judge_context_tokens,
        anchor=case.task,
    )
    prompt = (
        "You are a strict blinded evaluator for a context-compression A/B test. "
        "Compare Answer A and Answer B against the source evidence and task. "
        "Do not reward verbosity. Penalize missing exact IDs, file paths, symbols, numbers, "
        "dates, citations, constraints, or unsupported claims.\n\n"
        "Return JSON only with keys: equivalence, better_answer, a_missing_critical_facts, "
        "b_missing_critical_facts, a_hallucinated_facts, b_hallucinated_facts, "
        "coding_equivalence, rationale. equivalence must be one of "
        "equivalent, minor_delta, material_delta. better_answer must be A, B, or tie.\n\n"
        f"TASK:\n{case.task}\n\n"
        f"REQUIRED_TERMS:\n{json.dumps(case.required_terms, ensure_ascii=False)}\n\n"
        f"SOURCE_EVIDENCE:\n{source}\n\n"
        f"ANSWER_A:\n{answer_a}\n\n"
        f"ANSWER_B:\n{answer_b}\n"
    )
    response = _responses_create(client, args, prompt)
    text = _response_text(response)
    parsed = _parse_json_object(text)
    parsed["_raw_judge_text"] = text if not parsed else ""
    parsed["_compressed_is_a"] = compressed_is_a
    return parsed


def _responses_create(client: Any, args: argparse.Namespace, prompt: str) -> Any:
    kwargs = {
        "model": args.model,
        "input": [{"role": "user", "content": prompt}],
        "text": {"format": {"type": "text"}, "verbosity": args.verbosity},
        "reasoning": {"effort": args.reasoning_effort, "summary": "detailed"},
        "tools": [],
        "store": False,
        "include": ["reasoning.encrypted_content", "web_search_call.action.sources"],
        "max_output_tokens": args.max_output_tokens,
    }
    try:
        return client.responses.create(**kwargs)
    except Exception as exc:  # noqa: BLE001 - retry only for optional include incompatibilities
        message = str(exc).lower()
        if "include" not in message and "web_search_call" not in message:
            raise
        kwargs.pop("include", None)
        return client.responses.create(**kwargs)


def _openai_client() -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("The openai package is required. Install it or run in an environment that provides it.") from exc
    return OpenAI()


def _response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text).strip()
    chunks: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                chunks.append(str(text))
    return "\n".join(chunks).strip()


def _classification_from_judge(judge: dict[str, Any], side_checks: dict[str, Any]) -> str:
    equivalence = str(judge.get("equivalence") or "").strip().lower()
    better = str(judge.get("better_answer") or "tie").strip().upper()
    compressed_is_a = bool(judge.get("_compressed_is_a"))
    compressed_label = "A" if compressed_is_a else "B"
    full_label = "B" if compressed_is_a else "A"
    missing = side_checks.get("compressed_missing_required_terms") or []
    if equivalence == "equivalent" and not missing:
        return "equivalent"
    if better == compressed_label and equivalence == "material_delta":
        return "compressed_better"
    if better == full_label and equivalence == "material_delta":
        return "regression"
    if missing and better == full_label:
        return "regression"
    if equivalence in {"minor_delta", "material_delta"} or missing:
        return "minor_delta"
    return "minor_delta"


def _side_checks(case: EvalCase, pair: AnswerPair) -> dict[str, Any]:
    required = case.required_terms
    full_has = [term for term in required if _contains_term(pair.full_answer, term)]
    compressed_has = [term for term in required if _contains_term(pair.compressed_answer, term)]
    compressed_missing = [
        term for term in required if term in full_has and term not in compressed_has
    ]
    return {
        "required_term_count": len(required),
        "full_required_terms_present": full_has,
        "compressed_required_terms_present": compressed_has,
        "compressed_missing_required_terms": compressed_missing,
        "compressed_context_required_recall": _recall(required, case.compressed_context),
        "full_answer_required_recall": _recall(required, pair.full_answer),
        "compressed_answer_required_recall": _recall(required, pair.compressed_answer),
    }


def _repair_trigger_reasons(
    case: EvalCase,
    pair: AnswerPair,
    side_checks: dict[str, Any],
    args: argparse.Namespace,
) -> list[str]:
    reasons: list[str] = []
    compressed_answer = pair.compressed_answer.strip()
    if not compressed_answer:
        reasons.append("empty_compressed_answer")
    missing = side_checks.get("compressed_missing_required_terms") or []
    if missing:
        reasons.append("missing_exact_terms_in_compressed_answer")
    context_recall = side_checks.get("compressed_context_required_recall")
    if context_recall is not None and float(context_recall) < args.repair_min_context_recall:
        reasons.append("compressed_context_low_required_recall")
    if _looks_low_confidence(compressed_answer):
        reasons.append("compressed_answer_low_confidence")
    diagnostics = case.diagnostics or {}
    if diagnostics.get("memory_underbudgeted"):
        reasons.append("trace_memory_underbudgeted")
    if diagnostics.get("recent_tail_required"):
        reasons.append("recent_tail_required")
    return reasons


def _looks_low_confidence(text: str) -> bool:
    if not text.strip():
        return True
    lowered = text.lower()
    patterns = (
        "not enough context",
        "insufficient context",
        "i can't determine",
        "cannot determine",
        "i do not know",
        "unclear from the context",
        "not specified in the context",
    )
    return any(pattern in lowered for pattern in patterns)


def _build_repair_context(
    case: EvalCase,
    pair: AnswerPair,
    side_checks: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[str, list[dict[str, Any]]]:
    terms = _repair_terms(case, pair, side_checks)
    excerpts: list[dict[str, Any]] = []
    used: set[str] = set()
    remaining = args.repair_max_tokens

    target_excerpt = _target_turn_excerpt(case, max_tokens=min(900, remaining))
    if target_excerpt:
        digest = _hash_text(target_excerpt)
        excerpts.append({"reason": "target_turn_neighborhood", "matched_terms": [], "text": target_excerpt})
        used.add(digest)
        remaining -= _token_count(target_excerpt)

    for term in terms:
        if remaining <= 0 or len(excerpts) >= args.repair_max_excerpts:
            break
        excerpt = _bounded_excerpt(case.full_context, [term], max_tokens=min(500, remaining))
        if not excerpt:
            continue
        digest = _hash_text(excerpt)
        if digest in used:
            continue
        excerpts.append({"reason": "matched_exact_term", "matched_terms": [term], "text": excerpt})
        used.add(digest)
        remaining -= _token_count(excerpt)

    if remaining > 0 and len(excerpts) < args.repair_max_excerpts:
        tail_excerpt = _tail_text(case.full_context, min(1_500, remaining))
        digest = _hash_text(tail_excerpt)
        if tail_excerpt and digest not in used:
            excerpts.append({"reason": "latest_session_tail", "matched_terms": [], "text": tail_excerpt})
            used.add(digest)
            remaining -= _token_count(tail_excerpt)

    if remaining > 0 and len(excerpts) < args.repair_max_excerpts:
        overlap_excerpt = _overlap_excerpt(case.full_context, case.task, max_tokens=min(500, remaining))
        digest = _hash_text(overlap_excerpt)
        if overlap_excerpt and digest not in used:
            excerpts.append({"reason": "query_overlap", "matched_terms": [], "text": overlap_excerpt})

    rendered = "\n\n".join(
        f"[REPAIR_EXCERPT_{idx} reason={excerpt['reason']} terms={json.dumps(excerpt['matched_terms'], ensure_ascii=False)}]\n"
        f"{excerpt['text']}"
        for idx, excerpt in enumerate(excerpts, start=1)
    )
    return rendered, excerpts


def _repair_terms(
    case: EvalCase,
    pair: AnswerPair,
    side_checks: dict[str, Any],
) -> list[str]:
    terms = list(side_checks.get("compressed_missing_required_terms") or [])
    terms.extend(term for term in case.required_terms if term not in terms)
    full_terms = _required_terms(pair.full_answer, "code" if case.case_type == "coding_trace" else "rag")
    for term in full_terms:
        if term not in terms and not _contains_term(pair.compressed_answer, term):
            terms.append(term)
    return terms[:64]


def _target_turn_excerpt(case: EvalCase, *, max_tokens: int) -> str:
    marker = "TARGET_USER_TURN"
    if marker not in case.task:
        return _bounded_excerpt(case.full_context, case.required_terms, max_tokens=max_tokens)
    target = case.task.split(marker, 1)[1]
    target = target.split("):", 1)[-1].strip() if "):" in target else target.strip()
    if not target:
        return ""
    context = case.full_context
    position = context.find(target[: min(300, len(target))])
    if position < 0:
        return _overlap_excerpt(context, target, max_tokens=max_tokens)
    before = context[:position]
    start_word = max(0, len(before.split()) - max_tokens // 5)
    words = context.split()
    return " ".join(words[start_word : start_word + max_tokens])


def _bounded_excerpt(text: str, terms: list[str], *, max_tokens: int) -> str:
    words = text.split()
    if not words:
        return ""
    lower = text.lower()
    positions = [lower.find(term.lower()) for term in terms if term and lower.find(term.lower()) >= 0]
    if not positions:
        return " ".join(words[:max_tokens])
    char_pos = min(positions)
    start_word = max(0, len(text[:char_pos].split()) - max_tokens // 4)
    return " ".join(words[start_word : start_word + max_tokens])


def _overlap_excerpt(text: str, query: str, *, max_tokens: int) -> str:
    query_words = {
        word
        for word in re.findall(r"[A-Za-z0-9_./-]{5,}", query.lower())
        if word not in {"context", "target_user_turn", "provided", "answer"}
    }
    if not query_words:
        return " ".join(text.split()[:max_tokens])
    words = text.split()
    best_start = 0
    best_score = -1
    window = max_tokens
    step = max(1, window // 2)
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + window]).lower()
        score = sum(1 for word in query_words if word in chunk)
        if score > best_score:
            best_score = score
            best_start = start
    return " ".join(words[best_start : best_start + window])


def _case_summary(case: EvalCase) -> dict[str, Any]:
    full_tokens = _token_count(case.full_context)
    compressed_tokens = _token_count(case.compressed_context)
    reduction = 1.0 - compressed_tokens / max(1, full_tokens)
    return {
        "case_id": case.case_id,
        "case_type": case.case_type,
        "source": case.source,
        "title": case.title,
        "full_tokens": full_tokens,
        "compressed_tokens": compressed_tokens,
        "token_reduction": round(reduction, 4),
        "required_term_count": len(case.required_terms),
        "diagnostics": case.diagnostics,
        "metadata": case.metadata,
        "task_preview": _preview(case.task, 700),
        "full_context_sha256": _hash_text(case.full_context),
        "compressed_context_sha256": _hash_text(case.compressed_context),
    }


def _final_report(results: list[dict[str, Any]], args: argparse.Namespace) -> str:
    counts: dict[str, int] = {}
    reductions = []
    for result in results:
        counts[result["classification"]] = counts.get(result["classification"], 0) + 1
        reductions.append(float(result["case"]["token_reduction"]))
    regressions = [row for row in results if row["classification"] == "regression"]
    minor = [row for row in results if row["classification"] == "minor_delta"]
    failures = [row for row in results if row["classification"] == "api_failure"]
    pass_gate = not regressions and not failures and len(minor) <= 2
    lines = [
        "# LLM A/B Equivalence Report",
        "",
        f"- Model: `{args.model}`",
        f"- Cases: `{len(results)}`",
        f"- Pass gate: `{'PASS' if pass_gate else 'FAIL'}`",
        f"- Classification counts: `{json.dumps(counts, sort_keys=True)}`",
        f"- Mean token reduction: `{_mean(reductions):.4f}`",
        f"- Median token reduction: `{_median(reductions):.4f}`",
        "",
        "## Cases",
    ]
    for result in results:
        case = result["case"]
        judge = result["judge"]
        if result["classification"] == "api_failure":
            lines.extend(
                [
                    "",
                    f"### {case['case_id']}",
                    f"- Type: `{case['case_type']}`",
                    "- Classification: `api_failure`",
                    f"- Error: `{result.get('error', '')}`",
                ]
            )
            continue
        lines.extend(
            [
                "",
                f"### {case['case_id']}",
                f"- Type: `{case['case_type']}`",
                f"- Classification: `{result['classification']}`",
                f"- Tokens: `{case['full_tokens']}` full -> `{case['compressed_tokens']}` compressed "
                f"({case['token_reduction']:.2%} reduction)",
                f"- Repair triggered: `{bool((result.get('repair') or {}).get('triggered'))}`",
                f"- Judge equivalence: `{judge.get('equivalence')}`; better: `{judge.get('better_answer')}`",
                f"- Rationale: {str(judge.get('rationale') or '').strip()}",
            ]
        )
        repair = result.get("repair") or {}
        if repair.get("triggered"):
            lines.append(f"- Repair triggers: `{json.dumps(repair.get('triggers', []), sort_keys=True)}`")
            lines.append(
                f"- Repair packet: `{repair.get('excerpt_count', 0)}` excerpts, "
                f"`{repair.get('repair_tokens', 0)}` tokens"
            )
    return "\n".join(lines) + "\n"


def _dry_run_report(case_rows: list[dict[str, Any]], args: argparse.Namespace) -> str:
    reductions = [float(row["token_reduction"]) for row in case_rows]
    by_type: dict[str, int] = {}
    for row in case_rows:
        by_type[row["case_type"]] = by_type.get(row["case_type"], 0) + 1
    lines = [
        "# LLM A/B Equivalence Dry Run",
        "",
        f"- Cases: `{len(case_rows)}`",
        f"- Case types: `{json.dumps(by_type, sort_keys=True)}`",
        f"- Mean token reduction: `{_mean(reductions):.4f}`",
        f"- Median token reduction: `{_median(reductions):.4f}`",
        f"- Memory context ratio: `{args.memory_context_ratio}`",
        "",
        "## Selected Cases",
    ]
    for row in case_rows:
        lines.extend(
            [
                "",
                f"### {row['case_id']}",
                f"- Type: `{row['case_type']}`",
                f"- Source: `{row['source']}`",
                f"- Tokens: `{row['full_tokens']}` full -> `{row['compressed_tokens']}` compressed "
                f"({row['token_reduction']:.2%} reduction)",
                f"- Required terms: `{row['required_term_count']}`",
                f"- Task: {row['task_preview']}",
            ]
        )
    return "\n".join(lines) + "\n"


def _load_messages(path: Path) -> list[tuple[str, str]]:
    messages: list[tuple[str, str]] = []
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


def _format_messages(messages: list[tuple[str, str]]) -> str:
    return "\n\n".join(
        f"[{idx:04d} {role.upper()}]\n{text}"
        for idx, (role, text) in enumerate(messages, start=1)
    )


def _positioned_index(indices: list[int], position: str) -> int:
    if position == "early":
        return indices[max(0, min(len(indices) - 1, len(indices) // 10))]
    if position == "middle":
        return indices[len(indices) // 2]
    return indices[-1]


def _title(messages: list[tuple[str, str]]) -> str:
    for role, text in messages:
        if role == "user":
            return " ".join(re.sub(r"<[^>]+>", " ", text).split()[:8])
    return "untitled"


def _load_rag_rows(root: Path, rng: random.Random) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows.extend(_iter_halueval_qa(root, limit=80))
    rows.extend(_iter_halueval_summ(root, limit=80))
    rows.extend(_iter_ragtruth(root, task="qa", limit=80))
    rows.extend(_iter_ragtruth(root, task="summary", limit=80))
    rows = [
        row
        for row in rows
        if row.get("question") and row.get("response_text") and row.get("raw_context")
    ]
    rng.shuffle(rows)
    return rows


def _iter_halueval_qa(root: Path, *, limit: int) -> list[dict[str, Any]]:
    path = root / "HaluEval" / "data" / "qa_data.jsonl"
    rows = _read_jsonl(path, limit=limit)
    out = []
    for idx, row in enumerate(rows):
        question = row.get("question", "")
        knowledge = row.get("knowledge", "")
        if row.get("right_answer"):
            out.append(
                _rag_row("halueval_qa", f"halueval_qa_{idx}", question, row["right_answer"], knowledge, "green", "faithful")
            )
        if row.get("hallucinated_answer"):
            out.append(
                _rag_row(
                    "halueval_qa",
                    f"halueval_qa_{idx}_halluc",
                    question,
                    row["hallucinated_answer"],
                    knowledge,
                    "red",
                    "hallucinated",
                )
            )
    return out


def _iter_halueval_summ(root: Path, *, limit: int) -> list[dict[str, Any]]:
    path = root / "HaluEval" / "data" / "summarization_data.jsonl"
    rows = _read_jsonl(path, limit=limit)
    out = []
    for idx, row in enumerate(rows):
        document = row.get("document", "")
        question = "Summarise the article."
        right = row.get("right_summary") or row.get("summary") or ""
        wrong = row.get("hallucinated_summary") or ""
        if right:
            out.append(_rag_row("halueval_summ", f"halueval_summ_{idx}", question, right, document, "green", "faithful"))
        if wrong:
            out.append(_rag_row("halueval_summ", f"halueval_summ_{idx}_halluc", question, wrong, document, "red", "hallucinated"))
    return out


def _iter_ragtruth(root: Path, *, task: str, limit: int) -> list[dict[str, Any]]:
    task_dir = {"qa": "qa", "summary": "summarization"}.get(task, task)
    path = root / "RAGTruth" / "voyager_layout" / task_dir / "test.jsonl"
    rows = _read_jsonl(path, limit=limit)
    out = []
    for idx, row in enumerate(rows):
        labels = row.get("labels") or []
        source_info = row.get("source_info") or ""
        if isinstance(source_info, dict):
            raw_context = json.dumps(source_info, ensure_ascii=False)
            question = row.get("query") or source_info.get("question") or ""
        else:
            raw_context = str(source_info)
            question = row.get("query") or ""
        out.append(
            _rag_row(
                f"ragtruth_{task}",
                f"ragtruth_{task}_{idx}",
                question,
                row.get("response", ""),
                raw_context,
                "red" if labels else "green",
                "hallucinated" if labels else "faithful",
            )
        )
    return out


def _rag_row(
    bench: str,
    row_id: str,
    question: str,
    response_text: str,
    raw_context: str,
    expected_band: str,
    expected_label: str,
) -> dict[str, Any]:
    return {
        "bench": bench,
        "row_id": row_id,
        "question": question,
        "response_text": response_text,
        "raw_context": raw_context,
        "expected_band": expected_band,
        "expected_label": expected_label,
    }


def _read_jsonl(path: Path, *, limit: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if len(rows) >= limit:
                break
    return rows


def _rag_distractors(
    rows: list[dict[str, Any]],
    primary: dict[str, Any],
    rng: random.Random,
    *,
    count: int,
) -> list[dict[str, Any]]:
    candidates = [row for row in rows if row["row_id"] != primary["row_id"] and row.get("raw_context")]
    rng.shuffle(candidates)
    return candidates[:count]


def _rag_context_with_distractors(primary: dict[str, Any], distractors: list[dict[str, Any]]) -> str:
    chunks = [("[PRIMARY_SOURCE]", primary["raw_context"])]
    chunks.extend((f"[DISTRACTOR_SOURCE_{idx}]", row["raw_context"]) for idx, row in enumerate(distractors, start=1))
    return "\n\n".join(f"{label}\n{text}" for label, text in chunks)


def _required_terms(text: str, domain: str) -> list[str]:
    terms = set(extract_exact_critical_terms(text, domain=domain))
    terms.update(extract_exact_critical_terms(text, domain="tool"))
    terms.update(_PATH_OR_SYMBOL_RE.findall(text))
    clean = []
    for term in sorted(terms, key=lambda item: (len(item), item), reverse=True):
        if "[REDACTED]" in term or len(term) > 180:
            continue
        if term not in clean:
            clean.append(term)
    return clean[:40]


_PATH_OR_SYMBOL_RE = re.compile(
    r"\b(?:[A-Za-z0-9_.-]+/[A-Za-z0-9_./-]+|[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_.]*)\b"
)


def _maybe_bound_context(text: str, max_tokens: int, *, anchor: str = "") -> str:
    if max_tokens <= 0 or _token_count(text) <= max_tokens:
        return text
    tokens = text.split()
    anchor_tokens = anchor.split()
    anchor_idx = -1
    if anchor_tokens:
        needle = " ".join(anchor_tokens[: min(12, len(anchor_tokens))])
        joined = " ".join(tokens)
        char_idx = joined.find(needle)
        if char_idx >= 0:
            anchor_idx = max(0, len(joined[:char_idx].split()))
    head_budget = max_tokens // 4
    tail_budget = max_tokens // 4
    middle_budget = max_tokens - head_budget - tail_budget
    head = tokens[:head_budget]
    tail = tokens[-tail_budget:] if tail_budget else []
    if anchor_idx >= 0:
        start = max(0, anchor_idx - middle_budget // 2)
        middle = tokens[start : start + middle_budget]
    else:
        center = len(tokens) // 2
        start = max(0, center - middle_budget // 2)
        middle = tokens[start : start + middle_budget]
    return (
        " ".join(head)
        + "\n\n[... CONTEXT TRUNCATED FOR MODEL LIMIT ...]\n\n"
        + " ".join(middle)
        + "\n\n[... CONTEXT TRUNCATED FOR MODEL LIMIT ...]\n\n"
        + " ".join(tail)
    )


def _tail_text(text: str, budget: int) -> str:
    tokens = text.split()
    return " ".join(tokens[-budget:])


def _path_token_estimate(path: Path) -> int:
    try:
        return len(path.read_text(errors="ignore").split())
    except OSError:
        return 0


def _token_count(text: str) -> int:
    return len(text.split())


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def _preview(text: str, chars: int) -> str:
    return " ".join(text.split())[:chars]


def _contains_term(text: str, term: str) -> bool:
    return term.lower() in text.lower()


def _recall(terms: list[str], text: str) -> float | None:
    if not terms:
        return None
    return round(sum(1 for term in terms if _contains_term(text, term)) / len(terms), 4)


def _parse_json_object(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


if __name__ == "__main__":
    raise SystemExit(main())
