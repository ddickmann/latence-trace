#!/usr/bin/env python
"""Multi-turn A/B equivalence eval for InfiniMem context compression.

For each eval case, simulates a multi-turn conversation where:
  ARM A (full): The LLM receives the full accumulated context at every turn
  ARM B (infinimem): The LLM receives InfiniMem's hot_context (compressed,
    survival-scored, budget-limited) — the exact path that production uses

At each turn, both arms answer the same user query. A blinded judge then
compares the answers against the full context ground truth.

Key measurements:
  - Per-turn quality: equivalent / minor_delta / regression
  - Quality degradation over turns: does compressed arm get worse on later turns?
  - Required term recall: do critical exact terms survive compression?
  - Token reduction: how much context was compressed?

Usage:
  python scripts/run_multiturn_ab_eval.py --dry-run          # validate cases
  python scripts/run_multiturn_ab_eval.py                    # full A/B eval
  python scripts/run_multiturn_ab_eval.py --case-type code   # code only
  python scripts/run_multiturn_ab_eval.py --case-type rag    # rag only
"""

from __future__ import annotations

import argparse
import copy
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
CASES_PATH = REPO_ROOT / "research" / "trace_memory" / "multiturn_eval" / "cases.jsonl"
OUTPUT_ROOT = REPO_ROOT / "research" / "trace_memory" / "multiturn_eval" / "results"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from latence_trace.memory.extract import extract_spans
from latence_trace.memory.models import MemoryDiagnostics, MemoryPolicy, MemoryState, SpanRecord
from latence_trace.memory.select import assign_layers, hot_context
from latence_trace.memory.survival import update_survival
from latence_trace.memory.dedup import merge_spans
from latence_trace.memory.signature import extract_exact_critical_terms


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TurnResult:
    turn_index: int
    context_pattern: str
    full_context_tokens: int
    compressed_context_tokens: int
    token_reduction: float
    full_answer: str
    compressed_answer: str
    classification: str      # equivalent, minor_delta, regression, compressed_better
    judge: dict[str, Any]
    side_checks: dict[str, Any]
    span_count: int
    hot_span_count: int
    warm_span_count: int
    diagnostics: dict[str, Any]


@dataclass
class CaseResult:
    case_id: str
    case_type: str
    language: str
    turn_count: int
    overall_classification: str
    turn_results: list[TurnResult]
    degradation_slope: float | None
    mean_token_reduction: float
    regression_count: int
    equivalent_count: int


# ---------------------------------------------------------------------------
# InfiniMem simulation — mirrors production exactly
# ---------------------------------------------------------------------------

class InfiniMemSimulator:
    """Simulates the production InfiniMem pipeline across turns."""

    def __init__(self, policy: MemoryPolicy | None = None):
        self.policy = policy or MemoryPolicy(
            hot_token_budget=8_000,
            warm_token_budget=32_000,
            max_spans=1024,
            memory_context_ratio=0.25,
            context_window_tokens=131_072,
        )
        self.spans: list[SpanRecord] = []
        self.turn_index: int = 0

    def ingest_turn(
        self,
        *,
        user_query: str,
        assistant_response: str,
        raw_context: str | None,
        domain: str = "rag",
        grounded: bool | None = None,
    ) -> str:
        """Ingest one turn and return the hot_context for the NEXT turn.

        This mirrors the production flow:
        1. Extract spans from the new turn's content
        2. Merge with existing spans (dedup)
        3. Score survival
        4. Assign layers (hot/warm/cold)
        5. Return hot_context (concatenated hot-layer span texts)
        """
        new_spans = extract_spans(
            turn_text=f"{user_query}\n{assistant_response}",
            response_text=assistant_response,
            raw_context=raw_context,
            query_text=user_query,
            turn_index=self.turn_index,
            memory_domain=domain,
        )

        if self.spans:
            merged, _ = merge_spans(self.spans, new_spans)
        else:
            merged = new_spans

        signals: dict[str, Any] = {
            "trajectory_domain": domain,
            "query_text": user_query,
        }
        if grounded is not None:
            signals["scores"] = {"grounded": grounded}

        scored, _ = update_survival(
            merged,
            turn_index=self.turn_index,
            policy=self.policy,
            trace_signals=signals,
        )

        selected, diagnostics = assign_layers(scored, policy=self.policy)
        self.spans = selected
        self.turn_index += 1

        hot = hot_context(selected)
        hot_context_only = "\n".join(
            s.text for s in selected
            if s.layer == "hot"
            and (s.source.startswith("raw_context") or s.source.startswith("exact_index_raw"))
        )

        self._last_hot_all = hot
        self._last_hot_context_only = hot_context_only

        self._last_diagnostics = {
            "total_spans": len(selected),
            "hot_spans": sum(1 for s in selected if s.layer == "hot"),
            "warm_spans": sum(1 for s in selected if s.layer == "warm"),
            "cold_spans": sum(1 for s in selected if s.layer == "cold"),
            "hot_tokens": sum(s.token_count for s in selected if s.layer == "hot"),
            "hot_context_tokens": sum(
                s.token_count for s in selected
                if s.layer == "hot"
                and (s.source.startswith("raw_context") or s.source.startswith("exact_index_raw"))
            ),
            "hot_conversation_tokens": sum(
                s.token_count for s in selected
                if s.layer == "hot"
                and not (s.source.startswith("raw_context") or s.source.startswith("exact_index_raw"))
            ),
            "warm_tokens": sum(s.token_count for s in selected if s.layer == "warm"),
        }
        if isinstance(diagnostics, MemoryDiagnostics):
            self._last_diagnostics.update(diagnostics.model_dump(mode="json"))

        return hot

    @property
    def diagnostics(self) -> dict[str, Any]:
        return getattr(self, "_last_diagnostics", {})


# ---------------------------------------------------------------------------
# Full context accumulator — the baseline (ARM A)
# ---------------------------------------------------------------------------

class FullContextAccumulator:
    """Accumulates all context across turns — the "full" baseline."""

    def __init__(self, max_tokens: int = 131_072):
        self.history: list[str] = []
        self.max_tokens = max_tokens

    def add_turn(
        self,
        *,
        user_query: str,
        assistant_response: str,
        raw_context: str | None,
    ) -> str:
        """Add a turn and return the full accumulated context."""
        parts = []
        if raw_context:
            parts.append(f"[CONTEXT]\n{raw_context}")
        parts.append(f"[USER]\n{user_query}")
        parts.append(f"[ASSISTANT]\n{assistant_response}")
        self.history.append("\n".join(parts))

        full = "\n\n---\n\n".join(self.history)
        tokens = full.split()
        if len(tokens) > self.max_tokens:
            full = " ".join(tokens[-self.max_tokens:])
        return full


# ---------------------------------------------------------------------------
# LLM caller
# ---------------------------------------------------------------------------

def call_llm(
    client: Any,
    args: argparse.Namespace,
    *,
    system: str,
    context: str,
    query: str,
) -> tuple[str, float]:
    """Call the LLM with context + query, return (answer, seconds)."""
    prompt = (
        f"{system}\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"USER_QUERY:\n{query}"
    )
    start = time.perf_counter()
    response = _responses_create(client, args, prompt)
    return _response_text(response), time.perf_counter() - start


def judge_pair(
    client: Any,
    args: argparse.Namespace,
    *,
    query: str,
    full_answer: str,
    compressed_answer: str,
    source_context: str,
    required_terms: list[str],
    rng: random.Random,
) -> dict[str, Any]:
    """Blinded judge compares full vs compressed answers."""
    compressed_is_a = rng.choice([True, False])
    answer_a = compressed_answer if compressed_is_a else full_answer
    answer_b = full_answer if compressed_is_a else compressed_answer

    source_bounded = _bound_text(source_context, args.max_judge_tokens)

    prompt = (
        "You are a strict blinded evaluator for a context-compression A/B test. "
        "Compare Answer A and Answer B against the source evidence and query. "
        "Do not reward verbosity. Penalize missing exact IDs, file paths, symbols, numbers, "
        "dates, citations, constraints, or unsupported claims.\n\n"
        "Return JSON only with keys: equivalence, better_answer, a_missing_critical_facts, "
        "b_missing_critical_facts, a_hallucinated_facts, b_hallucinated_facts, "
        "rationale. equivalence must be one of "
        "equivalent, minor_delta, material_delta. better_answer must be A, B, or tie.\n\n"
        f"QUERY:\n{query}\n\n"
        f"REQUIRED_TERMS:\n{json.dumps(required_terms, ensure_ascii=False)}\n\n"
        f"SOURCE_EVIDENCE:\n{source_bounded}\n\n"
        f"ANSWER_A:\n{answer_a}\n\n"
        f"ANSWER_B:\n{answer_b}\n"
    )
    response = _responses_create(client, args, prompt)
    text = _response_text(response)
    parsed = _parse_json(text)
    parsed["_compressed_is_a"] = compressed_is_a
    return parsed


# ---------------------------------------------------------------------------
# Eval loop
# ---------------------------------------------------------------------------

def run_case(
    case: dict[str, Any],
    args: argparse.Namespace,
    rng: random.Random,
    client: Any | None,
) -> CaseResult:
    """Run one multi-turn eval case."""
    domain = case["case_type"]
    turns = case["turns"]

    mem = InfiniMemSimulator()
    full = FullContextAccumulator()

    turn_results: list[TurnResult] = []

    for turn in turns:
        query = turn["user_query"]
        response = turn["assistant_response"]
        context = turn.get("raw_context")

        full_ctx = full.add_turn(
            user_query=query,
            assistant_response=response,
            raw_context=context,
        )

        hot_ctx = mem.ingest_turn(
            user_query=query,
            assistant_response=response,
            raw_context=context,
            domain=domain,
        )

        full_tokens = _tok(full_ctx)
        compressed_tokens = _tok(hot_ctx)
        reduction = 1.0 - compressed_tokens / max(1, full_tokens)

        if client is None:
            turn_results.append(TurnResult(
                turn_index=turn["turn_index"],
                context_pattern=turn["context_pattern"],
                full_context_tokens=full_tokens,
                compressed_context_tokens=compressed_tokens,
                token_reduction=round(reduction, 4),
                full_answer="(dry-run)",
                compressed_answer="(dry-run)",
                classification="(dry-run)",
                judge={},
                side_checks={},
                span_count=mem.diagnostics.get("total_spans", 0),
                hot_span_count=mem.diagnostics.get("hot_spans", 0),
                warm_span_count=mem.diagnostics.get("warm_spans", 0),
                diagnostics=mem.diagnostics,
            ))
            continue

        system = (
            f"You are a helpful {'coding' if domain == 'code' else 'knowledge'} assistant. "
            "Answer the user's query using only the provided context. "
            "Be precise and cite exact identifiers, paths, and values when relevant."
        )

        full_answer, full_secs = call_llm(
            client, args, system=system, context=full_ctx, query=query
        )
        compressed_answer, comp_secs = call_llm(
            client, args, system=system, context=hot_ctx or "(no context available)", query=query
        )

        required_terms = _extract_required_terms(query + "\n" + response, domain)
        judge_result = judge_pair(
            client, args,
            query=query,
            full_answer=full_answer,
            compressed_answer=compressed_answer,
            source_context=full_ctx,
            required_terms=required_terms,
            rng=rng,
        )
        side = _side_checks(required_terms, full_answer, compressed_answer)
        classification = _classify(judge_result, side)

        turn_results.append(TurnResult(
            turn_index=turn["turn_index"],
            context_pattern=turn["context_pattern"],
            full_context_tokens=full_tokens,
            compressed_context_tokens=compressed_tokens,
            token_reduction=round(reduction, 4),
            full_answer=full_answer,
            compressed_answer=compressed_answer,
            classification=classification,
            judge=judge_result,
            side_checks=side,
            span_count=mem.diagnostics.get("total_spans", 0),
            hot_span_count=mem.diagnostics.get("hot_spans", 0),
            warm_span_count=mem.diagnostics.get("warm_spans", 0),
            diagnostics=mem.diagnostics,
        ))

    overall = _overall_classification(turn_results)
    degradation = _degradation_slope(turn_results)
    reductions = [tr.token_reduction for tr in turn_results]
    regressions = sum(1 for tr in turn_results if tr.classification == "regression")
    equivalents = sum(1 for tr in turn_results if tr.classification == "equivalent")

    return CaseResult(
        case_id=case["case_id"],
        case_type=case["case_type"],
        language=case["language"],
        turn_count=len(turns),
        overall_classification=overall,
        turn_results=turn_results,
        degradation_slope=degradation,
        mean_token_reduction=round(sum(reductions) / max(1, len(reductions)), 4),
        regression_count=regressions,
        equivalent_count=equivalents,
    )


def _overall_classification(turn_results: list[TurnResult]) -> str:
    if not turn_results or turn_results[0].classification == "(dry-run)":
        return "(dry-run)"
    regressions = sum(1 for tr in turn_results if tr.classification == "regression")
    if regressions >= 2:
        return "FAIL"
    if regressions == 1:
        return "MARGINAL"
    equivalents = sum(1 for tr in turn_results if tr.classification == "equivalent")
    if equivalents >= len(turn_results) * 0.6:
        return "PASS"
    return "ACCEPTABLE"


def _degradation_slope(turn_results: list[TurnResult]) -> float | None:
    """Compute quality degradation over turns (negative = getting worse)."""
    reductions = [tr.token_reduction for tr in turn_results]
    if len(reductions) < 3:
        return None
    n = len(reductions)
    x_mean = (n - 1) / 2
    y_mean = sum(reductions) / n
    numerator = sum((i - x_mean) * (r - y_mean) for i, r in enumerate(reductions))
    denominator = sum((i - x_mean) ** 2 for i in range(n))
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 6)


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------

def _classify(judge: dict[str, Any], side: dict[str, Any]) -> str:
    equivalence = str(judge.get("equivalence", "")).strip().lower()
    better = str(judge.get("better_answer", "tie")).strip().upper()
    compressed_is_a = bool(judge.get("_compressed_is_a"))
    compressed_label = "A" if compressed_is_a else "B"
    full_label = "B" if compressed_is_a else "A"
    missing = side.get("compressed_missing_required_terms") or []

    if equivalence == "equivalent" and not missing:
        return "equivalent"
    if better == compressed_label and equivalence == "material_delta":
        return "compressed_better"
    if better == full_label and equivalence == "material_delta":
        return "regression"
    if missing and better == full_label:
        return "regression"
    return "minor_delta"


def _side_checks(required: list[str], full_answer: str, compressed_answer: str) -> dict[str, Any]:
    full_has = [t for t in required if t.lower() in full_answer.lower()]
    comp_has = [t for t in required if t.lower() in compressed_answer.lower()]
    missing = [t for t in required if t in full_has and t not in comp_has]
    return {
        "required_term_count": len(required),
        "full_required_terms_present": full_has,
        "compressed_required_terms_present": comp_has,
        "compressed_missing_required_terms": missing,
    }


def _extract_required_terms(text: str, domain: str) -> list[str]:
    terms = set(extract_exact_critical_terms(text, domain=domain))
    terms.update(extract_exact_critical_terms(text, domain="tool"))
    path_re = re.compile(r"\b(?:[A-Za-z0-9_.-]+/[A-Za-z0-9_./-]+)\b")
    terms.update(path_re.findall(text))
    clean = sorted(t for t in terms if len(t) <= 120 and "[REDACTED]" not in t)
    return clean[:30]


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def _responses_create(client: Any, args: argparse.Namespace, prompt: str) -> Any:
    kwargs = {
        "model": args.model,
        "input": [{"role": "user", "content": prompt}],
        "text": {"format": {"type": "text"}},
        "reasoning": {"effort": args.reasoning_effort},
        "tools": [],
        "store": False,
        "max_output_tokens": args.max_output_tokens,
    }
    try:
        return client.responses.create(**kwargs)
    except Exception:
        kwargs.pop("reasoning", None)
        return client.responses.create(**kwargs)


def _response_text(response: Any) -> str:
    text = getattr(response, "output_text", None)
    if text:
        return str(text).strip()
    chunks = []
    for item in getattr(response, "output", []) or []:
        for c in getattr(item, "content", []) or []:
            t = getattr(c, "text", None)
            if t:
                chunks.append(str(t))
    return "\n".join(chunks).strip()


def _parse_json(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _tok(text: str | None) -> int:
    return len(text.split()) if text else 0


def _bound_text(text: str, max_tokens: int) -> str:
    words = text.split()
    if len(words) <= max_tokens:
        return text
    head = max_tokens // 3
    tail = max_tokens // 3
    mid = max_tokens - head - tail
    center = len(words) // 2
    return (
        " ".join(words[:head])
        + "\n\n[...TRUNCATED...]\n\n"
        + " ".join(words[max(0, center - mid // 2):center + mid // 2])
        + "\n\n[...TRUNCATED...]\n\n"
        + " ".join(words[-tail:])
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def build_report(results: list[CaseResult], args: argparse.Namespace) -> str:
    is_dry = results and results[0].overall_classification == "(dry-run)"

    code_results = [r for r in results if r.case_type == "code"]
    rag_results = [r for r in results if r.case_type == "rag"]

    lines = [
        "# Multi-Turn InfiniMem A/B Eval Report",
        "",
        f"- Mode: `{'dry-run' if is_dry else 'live'}`",
        f"- Model: `{args.model}`",
        f"- Cases: `{len(results)}` ({len(code_results)} code, {len(rag_results)} rag)",
        "",
    ]

    if not is_dry:
        pass_count = sum(1 for r in results if r.overall_classification == "PASS")
        acceptable = sum(1 for r in results if r.overall_classification == "ACCEPTABLE")
        marginal = sum(1 for r in results if r.overall_classification == "MARGINAL")
        fail_count = sum(1 for r in results if r.overall_classification == "FAIL")
        gate = "PASS" if fail_count == 0 and marginal <= 2 else "FAIL"
        lines.extend([
            f"- Gate: **{gate}**",
            f"- PASS: {pass_count} | ACCEPTABLE: {acceptable} | MARGINAL: {marginal} | FAIL: {fail_count}",
        ])

    all_turns = [tr for r in results for tr in r.turn_results]
    reductions = [tr.token_reduction for tr in all_turns if tr.token_reduction != 0]
    lines.extend([
        "",
        "## Token Reduction Stats",
        f"- Mean: `{_mean(reductions):.2%}`",
        f"- Median: `{_median(reductions):.2%}`",
        f"- Min: `{min(reductions):.2%}`" if reductions else "",
        f"- Max: `{max(reductions):.2%}`" if reductions else "",
        "",
        "## Per-Case Results",
    ])

    for result in results:
        lines.append(f"\n### {result.case_id}")
        lines.append(f"- Type: `{result.case_type}` | Lang: `{result.language}` | Turns: `{result.turn_count}`")
        lines.append(f"- Overall: `{result.overall_classification}`")
        lines.append(f"- Mean reduction: `{result.mean_token_reduction:.2%}`")
        if result.degradation_slope is not None:
            lines.append(f"- Degradation slope: `{result.degradation_slope:.4f}`")

        lines.append("\n| Turn | Pattern | Full Tok | Compressed Tok | Reduction | Spans | Hot | Classification |")
        lines.append("|------|---------|----------|----------------|-----------|-------|-----|----------------|")
        for tr in result.turn_results:
            lines.append(
                f"| {tr.turn_index} | {tr.context_pattern} | "
                f"{tr.full_context_tokens:,} | {tr.compressed_context_tokens:,} | "
                f"{tr.token_reduction:.1%} | {tr.span_count} | {tr.hot_span_count} | "
                f"{tr.classification} |"
            )

    return "\n".join(lines) + "\n"


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = _parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    cases = _load_cases(args)
    print(f"Loaded {len(cases)} cases")

    client = None
    if not args.dry_run:
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY required for live eval. Use --dry-run for validation.")
        try:
            from openai import OpenAI
            client = OpenAI()
        except ImportError:
            raise SystemExit("openai package required")

    results: list[CaseResult] = []
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    suffix = "dry_run" if args.dry_run else "live"
    out_dir = OUTPUT_ROOT / f"{stamp}_{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "results.jsonl").open("w", encoding="utf-8") as f:
        for idx, case in enumerate(cases, 1):
            print(f"[{idx}/{len(cases)}] {case['case_id']} ({case['case_type']}/{case['language']}, "
                  f"{len(case['turns'])} turns)...")
            result = run_case(case, args, rng, client)
            results.append(result)
            row = {
                "case_id": result.case_id,
                "case_type": result.case_type,
                "language": result.language,
                "turn_count": result.turn_count,
                "overall_classification": result.overall_classification,
                "mean_token_reduction": result.mean_token_reduction,
                "degradation_slope": result.degradation_slope,
                "regression_count": result.regression_count,
                "equivalent_count": result.equivalent_count,
                "turns": [
                    {
                        "turn_index": tr.turn_index,
                        "context_pattern": tr.context_pattern,
                        "full_tokens": tr.full_context_tokens,
                        "compressed_tokens": tr.compressed_context_tokens,
                        "reduction": tr.token_reduction,
                        "spans": tr.span_count,
                        "hot_spans": tr.hot_span_count,
                        "warm_spans": tr.warm_span_count,
                        "classification": tr.classification,
                    }
                    for tr in result.turn_results
                ],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()

    report = build_report(results, args)
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    print(f"\nDone. Report: {out_dir / 'report.md'}")

    return 0


def _load_cases(args: argparse.Namespace) -> list[dict[str, Any]]:
    cases_path = args.cases_path or CASES_PATH
    cases = []
    with cases_path.open("r") as f:
        for line in f:
            if line.strip():
                case = json.loads(line)
                if args.case_type and case["case_type"] != args.case_type:
                    continue
                if args.language and case["language"] != args.language:
                    continue
                cases.append(case)
    if args.max_cases:
        cases = cases[:args.max_cases]
    return cases


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cases-path", type=Path, default=None)
    parser.add_argument("--case-type", choices=["code", "rag"], default=None)
    parser.add_argument("--language", choices=["en", "de"], default=None)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--max-output-tokens", type=int, default=1600)
    parser.add_argument("--max-judge-tokens", type=int, default=60_000)
    parser.add_argument("--seed", type=int, default=20260508)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
