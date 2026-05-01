"""Higher-resolution Granite Guardian runtime benchmark.

The smoke benchmark checks whether Granite can answer one yes/no question. This
benchmark asks the production question: what happens when we preprocess inputs
like TRACE does, splitting context and responses into smaller units so a runtime
harness can expose heatmap-like evidence resolution?
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))

from latence_trace.core.groundedness import segment_text  # noqa: E402
from scripts.bench_granite_guardian_vllm import (  # noqa: E402
    FC_CRITERIA,
    RAG_CRITERIA,
    build_guardian_block,
    parse_output,
)


MODEL_PATH = "ibm-granite/granite-guardian-4.1-8b"
RAG_FIXTURE = _HERE.parent.parent / "tests" / "fixtures" / "rag_prose_cases.jsonl"
AGENTIC_TRACE_SLICE = (
    _HERE.parent.parent
    / "research"
    / "triangular_maxsim"
    / "student_v2"
    / "root_cause_solution_runs"
    / "latest"
    / "root_cause_slices"
    / "code.agentic_trace.jsonl"
)

CODE_CRITERIA = (
    "Agentic trace hallucination occurs when the assistant claims that code was "
    "edited, commands succeeded, tests passed, symbols exist, or files changed "
    "without support from the supplied execution trace."
)


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    lane: str
    expected_score: Optional[str]
    expected_band: str
    query_text: str
    context_text: str
    response_text: str
    source: str
    tools: Optional[list[dict[str, Any]]] = None


@dataclass(frozen=True)
class Unit:
    idx: int
    text: str
    offset_start: Optional[int] = None
    offset_end: Optional[int] = None


@dataclass(frozen=True)
class PromptTask:
    task_id: str
    strategy: str
    case: EvalCase
    prompt: str
    response_idx: Optional[int] = None
    context_idx: Optional[int] = None


class TokenCountProvider:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def encoded_token_count(self, text: str, is_query: bool = False) -> int:
        return len(self.tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"])


def _score_from_band(band: str) -> Optional[str]:
    lowered = (band or "").lower()
    if lowered == "green":
        return "no"
    if lowered == "red":
        return "yes"
    return None


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _sample_by_key(rows: Sequence[EvalCase], *, key: str, per_key: int, seed: int) -> list[EvalCase]:
    rng = random.Random(seed)
    buckets: dict[str, list[EvalCase]] = {}
    for row in rows:
        value = getattr(row, key)
        buckets.setdefault(str(value), []).append(row)
    sampled: list[EvalCase] = []
    for bucket_key in sorted(buckets):
        bucket = buckets[bucket_key][:]
        rng.shuffle(bucket)
        sampled.extend(bucket[:per_key])
    return sorted(sampled, key=lambda row: row.case_id)


def load_rag_cases(path: Path, *, per_band: int, seed: int) -> list[EvalCase]:
    rows: list[EvalCase] = []
    for raw in _read_jsonl(path):
        band = str(raw.get("expected_band") or raw.get("label") or "").lower()
        context = str(raw.get("raw_context") or raw.get("support_text") or "")
        response = str(raw.get("response_text") or "")
        if not band or not context.strip() or not response.strip():
            continue
        rows.append(
            EvalCase(
                case_id=f"rag::{raw.get('id')}",
                lane="rag",
                expected_score=_score_from_band(band),
                expected_band=band,
                query_text=str(raw.get("query_text") or "Assess whether the answer is supported."),
                context_text=context,
                response_text=response,
                source=str(path.relative_to(_HERE.parent.parent)),
            )
        )
    return _sample_by_key(rows, key="expected_band", per_key=per_band, seed=seed)


def load_agentic_trace_cases(path: Path, *, limit: int, seed: int) -> list[EvalCase]:
    rows: list[EvalCase] = []
    for raw in _read_jsonl(path):
        claim = str(raw.get("claim") or "").replace("\u0120", " ").strip()
        evidence = str(raw.get("evidence") or "").strip()
        binary = ((raw.get("gold") or {}).get("binary") if isinstance(raw.get("gold"), dict) else None)
        if not claim or not evidence or binary not in {0, 1}:
            continue
        band = "green" if binary == 1 else "red"
        rows.append(
            EvalCase(
                case_id=f"agentic::{raw.get('row_id')}",
                lane="agentic_trace",
                expected_score=_score_from_band(band),
                expected_band=band,
                query_text="Assess whether the assistant's trace claim is supported by the execution evidence.",
                context_text=evidence,
                response_text=claim,
                source=str(path.relative_to(_HERE.parent.parent)),
            )
        )
    return _sample_by_key(rows, key="expected_band", per_key=max(1, limit // 2), seed=seed)[:limit]


def _tool_catalog() -> list[dict[str, Any]]:
    return [
        {
            "name": "comment_list",
            "description": "Fetch comments for a video.",
            "parameters": {
                "aweme_id": {"type": "int", "description": "Video identifier."},
                "count": {"type": "int", "description": "Number of comments, max 30."},
                "cursor": {"type": "int, optional", "description": "Pagination cursor."},
            },
        },
        {
            "name": "transfer_funds",
            "description": "Schedule a bank transfer.",
            "parameters": {
                "from_account": {"type": "string"},
                "to_account": {"type": "string"},
                "amount": {"type": "number"},
                "currency": {"type": "string", "enum": ["EUR", "USD", "GBP"]},
            },
        },
        {
            "name": "create_calendar_event",
            "description": "Create a calendar event.",
            "parameters": {
                "title": {"type": "string"},
                "start_time": {"type": "string"},
                "end_time": {"type": "string"},
                "attendees": {"type": "array[string], optional"},
            },
        },
    ]


def generated_function_call_cases() -> list[EvalCase]:
    tools = _tool_catalog()
    scenarios = [
        (
            "comments",
            "Fetch 15 comments for video 456789123.",
            [{"name": "comment_list", "arguments": {"aweme_id": 456789123, "count": 15}}],
            [
                ("wrong_arg", [{"name": "comment_list", "arguments": {"video_id": 456789123, "count": 15}}]),
                ("wrong_type", [{"name": "comment_list", "arguments": {"aweme_id": "456789123", "count": 15}}]),
                ("wrong_fn", [{"name": "video_details", "arguments": {"aweme_id": 456789123, "count": 15}}]),
            ],
        ),
        (
            "transfer",
            "Transfer 125.50 EUR from acct_1 to acct_9.",
            [
                {
                    "name": "transfer_funds",
                    "arguments": {
                        "from_account": "acct_1",
                        "to_account": "acct_9",
                        "amount": 125.50,
                        "currency": "EUR",
                    },
                }
            ],
            [
                (
                    "wrong_currency",
                    [
                        {
                            "name": "transfer_funds",
                            "arguments": {
                                "from_account": "acct_1",
                                "to_account": "acct_9",
                                "amount": 125.50,
                                "currency": "JPY",
                            },
                        }
                    ],
                ),
                (
                    "wrong_amount",
                    [
                        {
                            "name": "transfer_funds",
                            "arguments": {
                                "from_account": "acct_1",
                                "to_account": "acct_9",
                                "amount": 1250.50,
                                "currency": "EUR",
                            },
                        }
                    ],
                ),
                ("missing_required", [{"name": "transfer_funds", "arguments": {"to_account": "acct_9", "amount": 125.50}}]),
            ],
        ),
        (
            "calendar",
            "Create a design review from 2026-05-02T10:00Z to 2026-05-02T10:30Z.",
            [
                {
                    "name": "create_calendar_event",
                    "arguments": {
                        "title": "design review",
                        "start_time": "2026-05-02T10:00Z",
                        "end_time": "2026-05-02T10:30Z",
                    },
                }
            ],
            [
                (
                    "wrong_time",
                    [
                        {
                            "name": "create_calendar_event",
                            "arguments": {
                                "title": "design review",
                                "start_time": "2026-05-02T11:00Z",
                                "end_time": "2026-05-02T11:30Z",
                            },
                        }
                    ],
                ),
                ("wrong_arg", [{"name": "create_calendar_event", "arguments": {"subject": "design review"}}]),
                ("wrong_fn", [{"name": "send_email", "arguments": {"title": "design review"}}]),
            ],
        ),
    ]
    rows: list[EvalCase] = []
    context = json.dumps({"tools": tools}, indent=2, sort_keys=True)
    for prefix, query, correct, wrongs in scenarios:
        rows.append(
            EvalCase(
                case_id=f"fc::{prefix}::correct",
                lane="function_call",
                expected_score="no",
                expected_band="green",
                query_text=query,
                context_text=context,
                response_text=json.dumps(correct, sort_keys=True),
                source="generated_internal_function_call",
                tools=tools,
            )
        )
        for label, response in wrongs:
            rows.append(
                EvalCase(
                    case_id=f"fc::{prefix}::{label}",
                    lane="function_call",
                    expected_score="yes",
                    expected_band="red",
                    query_text=query,
                    context_text=context,
                    response_text=json.dumps(response, sort_keys=True),
                    source="generated_internal_function_call",
                    tools=tools,
                )
            )
    return rows


def criteria_for_lane(lane: str) -> str:
    if lane == "function_call":
        return FC_CRITERIA
    if lane == "agentic_trace":
        return CODE_CRITERIA
    return RAG_CRITERIA


def build_prompt(tokenizer: Any, case: EvalCase, *, context_text: str, response_text: str, suffix: str = "") -> str:
    context_label = "Available tools" if case.lane == "function_call" else "Context"
    user_content = (
        f"Task/query:\n{case.query_text}\n\n"
        f"{context_label}:\n{context_text}\n\n"
        f"{suffix}".strip()
    )
    messages = [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": response_text},
        {"role": "user", "content": build_guardian_block(criteria_for_lane(case.lane))},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _segments(text: str, *, provider: TokenCountProvider, budget: int) -> list[Unit]:
    spans = segment_text(text, "sentence_packed", provider=provider, chunk_token_budget=budget)
    return [
        Unit(
            idx=idx,
            text=str(span["text"]),
            offset_start=int(span.get("offset_start", 0)),
            offset_end=int(span.get("offset_end", 0)),
        )
        for idx, span in enumerate(spans)
        if str(span.get("text") or "").strip()
    ]


def response_units(case: EvalCase, *, provider: TokenCountProvider, budget: int) -> list[Unit]:
    if case.lane == "function_call":
        try:
            parsed = json.loads(case.response_text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list) and parsed:
            return [
                Unit(idx=idx, text=json.dumps(item, sort_keys=True))
                for idx, item in enumerate(parsed)
            ]
        if isinstance(parsed, dict):
            return [Unit(idx=0, text=json.dumps(parsed, sort_keys=True))]
    return _segments(case.response_text, provider=provider, budget=budget) or [Unit(idx=0, text=case.response_text)]


def context_units(case: EvalCase, *, provider: TokenCountProvider, budget: int) -> list[Unit]:
    if case.lane == "function_call" and case.tools:
        return [
            Unit(idx=idx, text=json.dumps(tool, sort_keys=True))
            for idx, tool in enumerate(case.tools)
        ]
    return _segments(case.context_text, provider=provider, budget=budget) or [Unit(idx=0, text=case.context_text)]


def build_tasks(
    tokenizer: Any,
    cases: Sequence[EvalCase],
    *,
    strategies: set[str],
    response_budget: int,
    context_budget: int,
) -> tuple[list[PromptTask], dict[str, Any]]:
    provider = TokenCountProvider(tokenizer)
    tasks: list[PromptTask] = []
    resolution_rows: dict[str, dict[str, Any]] = {}
    for case in cases:
        r_units = response_units(case, provider=provider, budget=response_budget)
        c_units = context_units(case, provider=provider, budget=context_budget)
        resolution_rows[case.case_id] = {
            "lane": case.lane,
            "expected_band": case.expected_band,
            "expected_score": case.expected_score,
            "response_unit_count": len(r_units),
            "context_unit_count": len(c_units),
            "matrix_cell_count": len(r_units) * len(c_units),
        }
        if "row" in strategies:
            tasks.append(
                PromptTask(
                    task_id=f"row::{case.case_id}",
                    strategy="row",
                    case=case,
                    prompt=build_prompt(
                        tokenizer,
                        case,
                        context_text=case.context_text,
                        response_text=case.response_text,
                        suffix="Assess the entire assistant response against the full context.",
                    ),
                )
            )
        if "response" in strategies:
            for r_unit in r_units:
                tasks.append(
                    PromptTask(
                        task_id=f"response::{case.case_id}::{r_unit.idx}",
                        strategy="response",
                        case=case,
                        response_idx=r_unit.idx,
                        prompt=build_prompt(
                            tokenizer,
                            case,
                            context_text=case.context_text,
                            response_text=r_unit.text,
                            suffix="Assess only this assistant response unit against the full context.",
                        ),
                    )
                )
        if "matrix" in strategies:
            for r_unit in r_units:
                for c_unit in c_units:
                    tasks.append(
                        PromptTask(
                            task_id=f"matrix::{case.case_id}::{r_unit.idx}::{c_unit.idx}",
                            strategy="matrix",
                            case=case,
                            response_idx=r_unit.idx,
                            context_idx=c_unit.idx,
                            prompt=build_prompt(
                                tokenizer,
                                case,
                                context_text=c_unit.text,
                                response_text=r_unit.text,
                                suffix=(
                                    "Assess only this assistant response unit against only this context/tool unit. "
                                    "This prompt is one cell of a heatmap-style matrix."
                                ),
                            ),
                        )
                    )
    stats = {
        "case_count": len(cases),
        "avg_response_units": statistics.fmean(row["response_unit_count"] for row in resolution_rows.values()),
        "avg_context_units": statistics.fmean(row["context_unit_count"] for row in resolution_rows.values()),
        "avg_matrix_cells": statistics.fmean(row["matrix_cell_count"] for row in resolution_rows.values()),
        "max_matrix_cells": max(row["matrix_cell_count"] for row in resolution_rows.values()),
        "rows": resolution_rows,
    }
    return tasks, stats


def _latency_summary(values: Sequence[float]) -> dict[str, Optional[float]]:
    if not values:
        return {"p50_ms": None, "p95_ms": None, "mean_ms": None}
    sorted_values = sorted(values)
    return {
        "p50_ms": statistics.median(sorted_values),
        "p95_ms": sorted_values[min(len(sorted_values) - 1, max(0, math.ceil(0.95 * len(sorted_values)) - 1))],
        "mean_ms": statistics.fmean(sorted_values),
    }


def _batched(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for idx in range(0, len(items), size):
        yield items[idx : idx + size]


def run_tasks(llm: Any, sampling_params: Any, tasks: Sequence[PromptTask], *, batch_size: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    batch_latencies: list[float] = []
    prompt_latencies: list[float] = []
    generated_tokens: list[int] = []
    started = time.perf_counter()
    for batch in _batched(list(tasks), batch_size):
        prompts = [task.prompt for task in batch]
        t0 = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        batch_ms = (time.perf_counter() - t0) * 1000.0
        batch_latencies.append(batch_ms)
        per_prompt_ms = batch_ms / max(1, len(batch))
        prompt_latencies.extend([per_prompt_ms] * len(batch))
        for task, output in zip(batch, outputs):
            text = output.outputs[0].text.strip()
            token_count = len(output.outputs[0].token_ids)
            generated_tokens.append(token_count)
            score = parse_output(text)
            rows.append(
                {
                    "task_id": task.task_id,
                    "strategy": task.strategy,
                    "case_id": task.case.case_id,
                    "lane": task.case.lane,
                    "expected_score": task.case.expected_score,
                    "expected_band": task.case.expected_band,
                    "score": score,
                    "parse_ok": score in {"yes", "no"},
                    "correct": score == task.case.expected_score if task.case.expected_score else None,
                    "response_idx": task.response_idx,
                    "context_idx": task.context_idx,
                    "raw_output": text,
                    "generated_tokens": token_count,
                    "batch_size": len(batch),
                    "batch_latency_ms": batch_ms,
                    "amortized_prompt_latency_ms": per_prompt_ms,
                }
            )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return rows, {
        "prompt_count": len(tasks),
        "batch_size": batch_size,
        "elapsed_ms": elapsed_ms,
        "throughput_prompts_per_s": len(tasks) / max(elapsed_ms / 1000.0, 1e-9),
        "batch_latency": _latency_summary(batch_latencies),
        "amortized_prompt_latency": _latency_summary(prompt_latencies),
        "generated_tokens": _latency_summary(generated_tokens),
    }


def aggregate_strategy(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    binary = [row for row in rows if row.get("expected_score") in {"yes", "no"}]
    parse_rate = sum(1 for row in rows if row.get("parse_ok")) / max(1, len(rows))
    accuracy = sum(1 for row in binary if row.get("correct")) / max(1, len(binary)) if binary else None
    by_lane: dict[str, dict[str, Any]] = {}
    for lane in sorted({str(row["lane"]) for row in rows}):
        lane_rows = [row for row in rows if row["lane"] == lane]
        lane_binary = [row for row in lane_rows if row.get("expected_score") in {"yes", "no"}]
        by_lane[lane] = {
            "n": len(lane_rows),
            "binary_n": len(lane_binary),
            "accuracy": (
                sum(1 for row in lane_binary if row.get("correct")) / max(1, len(lane_binary))
                if lane_binary
                else None
            ),
            "parse_rate": sum(1 for row in lane_rows if row.get("parse_ok")) / max(1, len(lane_rows)),
            "yes_rate": sum(1 for row in lane_rows if row.get("score") == "yes") / max(1, len(lane_rows)),
        }
    return {
        "n": len(rows),
        "binary_n": len(binary),
        "accuracy": accuracy,
        "parse_rate": parse_rate,
        "yes_rate": sum(1 for row in rows if row.get("score") == "yes") / max(1, len(rows)),
        "by_lane": by_lane,
    }


def aggregate_case_predictions(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_strategy: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in rows:
        by_strategy.setdefault(str(row["strategy"]), {}).setdefault(str(row["case_id"]), []).append(row)
    out: dict[str, Any] = {}
    for strategy, cases in by_strategy.items():
        predicted: list[dict[str, Any]] = []
        for case_id, case_rows in cases.items():
            expected = case_rows[0].get("expected_score")
            if strategy == "row":
                score = case_rows[0].get("score")
            elif strategy == "response":
                # For response and matrix resolution, any unsupported response
                # unit should make the whole case a hallucination.
                score = "yes" if any(row.get("score") == "yes" for row in case_rows) else "no"
            else:
                # Matrix cells are response_unit x context_unit. A response
                # unit is supported if any context/tool cell returns "no"
                # (not hallucinated); it is unsupported only when every
                # available context/tool cell returns "yes".
                by_response: dict[int, list[dict[str, Any]]] = {}
                for row in case_rows:
                    idx = int(row.get("response_idx") or 0)
                    by_response.setdefault(idx, []).append(row)
                unsupported_response = False
                for response_rows in by_response.values():
                    has_supporting_cell = any(row.get("score") == "no" for row in response_rows)
                    if not has_supporting_cell:
                        unsupported_response = True
                        break
                score = "yes" if unsupported_response else "no"
            parse_ok = all(bool(row.get("parse_ok")) for row in case_rows)
            predicted.append(
                {
                    "case_id": case_id,
                    "lane": case_rows[0].get("lane"),
                    "expected_score": expected,
                    "score": score,
                    "parse_ok": parse_ok,
                    "correct": score == expected if expected in {"yes", "no"} else None,
                    "expected_band": case_rows[0].get("expected_band"),
                    "task_count": len(case_rows),
                }
            )
        out[strategy] = aggregate_strategy(predicted)
        out[strategy]["case_rows"] = predicted
    return out


def run_latency_sweep(llm: Any, sampling_params: Any, tasks: Sequence[PromptTask], *, batch_sizes: Sequence[int], sample: int) -> list[dict[str, Any]]:
    sample_tasks = list(tasks[:sample])
    results: list[dict[str, Any]] = []
    for batch_size in batch_sizes:
        _, metrics = run_tasks(llm, sampling_params, sample_tasks, batch_size=batch_size)
        results.append(metrics)
    return results


def _format_pct(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def write_markdown(report: dict[str, Any], path: Path) -> None:
    gate = report["decision_gate"]
    lines = [
        "# Granite Guardian Runtime Harness Decision",
        "",
        f"Created: `{report['created_at']}`",
        "",
        "## Recommendation",
        "",
        gate["recommendation"],
        "",
        "## Benchmark Summary",
        "",
        f"- Cases: `{report['dataset']['case_count']}`",
        f"- Quality prompts: `{report['quality_latency']['prompt_count']}`",
        f"- Average response units per case: `{report['resolution']['avg_response_units']:.2f}`",
        f"- Average context units per case: `{report['resolution']['avg_context_units']:.2f}`",
        f"- Average matrix cells per case: `{report['resolution']['avg_matrix_cells']:.2f}`",
        f"- Prompt multiplier for full row+response+matrix run: `{report['quality_latency']['prompt_count'] / max(1, report['dataset']['case_count']):.2f}x`",
        "",
        "## Case-Level Accuracy",
        "",
        "| Strategy | Accuracy | Parse Rate | Prompt/Case Shape |",
        "| --- | ---: | ---: | --- |",
    ]
    case_metrics = report["case_level"]
    for strategy in ("row", "response", "matrix"):
        metric = case_metrics.get(strategy)
        if not metric:
            continue
        lines.append(
            f"| `{strategy}` | {_format_pct(metric['accuracy'])} | {_format_pct(metric['parse_rate'])} | "
            f"`{metric['n']}` cases |"
        )
    lines.extend(
        [
            "",
            "## Latency",
            "",
            f"- Cold load: `{report['cold_load_ms']:.1f} ms`",
            f"- Quality throughput: `{report['quality_latency']['throughput_prompts_per_s']:.2f}` prompts/s",
            f"- Amortized prompt p50/p95: "
            f"`{report['quality_latency']['amortized_prompt_latency']['p50_ms']:.1f}` / "
            f"`{report['quality_latency']['amortized_prompt_latency']['p95_ms']:.1f}` ms",
            "",
            "| Batch Size | Prompts/s | Batch p50 ms | Batch p95 ms | Amortized Prompt p95 ms |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in report["latency_sweep"]:
        lines.append(
            f"| {item['batch_size']} | {item['throughput_prompts_per_s']:.2f} | "
            f"{item['batch_latency']['p50_ms']:.1f} | {item['batch_latency']['p95_ms']:.1f} | "
            f"{item['amortized_prompt_latency']['p95_ms']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Gate Result",
            "",
            f"- Replace TRACE: `{gate['replace_trace']}`",
            f"- Build Granite production replacement now: `{gate.get('build_production_replacement_now', False)}`",
            f"- Build Granite shadow harness now: `{gate.get('build_shadow_harness_now', gate.get('build_production_harness_now', False))}`",
            f"- Reasons: {gate['reasons']}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_PATH)
    parser.add_argument("--quantization", default="", help="Empty string disables quantization.")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--latency-sweep-batches", default="1,2,4,8,16")
    parser.add_argument("--latency-sample", type=int, default=64)
    parser.add_argument("--rag-per-band", type=int, default=12)
    parser.add_argument("--agentic-limit", type=int, default=24)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--response-budget", type=int, default=48)
    parser.add_argument("--context-budget", type=int, default=96)
    parser.add_argument("--strategies", default="row,response,matrix")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("research/triangular_maxsim/reports/granite_guardian_resolution_bench.json"),
    )
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    strategies = {part.strip() for part in args.strategies.split(",") if part.strip()}
    cases = [
        *load_rag_cases(RAG_FIXTURE, per_band=args.rag_per_band, seed=args.seed),
        *load_agentic_trace_cases(AGENTIC_TRACE_SLICE, limit=args.agentic_limit, seed=args.seed),
        *generated_function_call_cases(),
    ]
    cases = sorted(cases, key=lambda case: (case.lane, case.case_id))

    cold_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm_kwargs: dict[str, Any] = {
        "model": args.model,
        "max_model_len": args.max_model_len,
        "trust_remote_code": True,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": args.enforce_eager,
    }
    if args.quantization:
        llm_kwargs["quantization"] = args.quantization
    llm = LLM(**llm_kwargs)
    sampling_params = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)
    cold_load_ms = (time.perf_counter() - cold_start) * 1000.0

    tasks, resolution = build_tasks(
        tokenizer,
        cases,
        strategies=strategies,
        response_budget=args.response_budget,
        context_budget=args.context_budget,
    )
    # Warmup with one real prompt so the quality run does not include first-token setup.
    llm.generate([tasks[0].prompt], sampling_params, use_tqdm=False)
    rows, quality_latency = run_tasks(llm, sampling_params, tasks, batch_size=args.batch_size)
    latency_sweep = run_latency_sweep(
        llm,
        sampling_params,
        tasks,
        batch_sizes=[int(part) for part in args.latency_sweep_batches.split(",") if part.strip()],
        sample=min(args.latency_sample, len(tasks)),
    )

    strategy_metrics = {
        strategy: aggregate_strategy([row for row in rows if row["strategy"] == strategy])
        for strategy in sorted(strategies)
    }
    case_level = aggregate_case_predictions(rows)
    matrix_accuracy = (case_level.get("matrix") or {}).get("accuracy")
    row_accuracy = (case_level.get("row") or {}).get("accuracy")
    parse_rate = sum(1 for row in rows if row["parse_ok"]) / max(1, len(rows))
    prompt_multiplier = len(tasks) / max(1, len(cases))
    reasons: list[str] = []
    replace_trace = False
    build_harness = False
    if parse_rate < 0.99:
        reasons.append(f"parse rate {parse_rate:.3f} is below production gate 0.99")
    if row_accuracy is not None and row_accuracy < 0.95:
        reasons.append(f"row accuracy {row_accuracy:.3f} is below replacement gate 0.95")
    if matrix_accuracy is not None and matrix_accuracy < 0.90:
        reasons.append(f"matrix accuracy {matrix_accuracy:.3f} is below heatmap gate 0.90")
    matrix_by_lane = ((case_level.get("matrix") or {}).get("by_lane") or {})
    for lane, lane_metric in matrix_by_lane.items():
        lane_accuracy = lane_metric.get("accuracy")
        if lane_accuracy is not None and lane_accuracy < 0.90:
            reasons.append(f"matrix accuracy for {lane} is {lane_accuracy:.3f}, below per-lane gate 0.90")
    if prompt_multiplier > 4.0:
        reasons.append(f"heatmap-style preprocessing expands calls by {prompt_multiplier:.2f}x")
    if not reasons:
        build_harness = True
        reasons.append("all benchmark gates passed; proceed to shadow integration before replacement")
    recommendation = (
        "Do not replace TRACE with a Granite runtime harness from this benchmark alone. "
        "Granite can be considered as an optional quality-mode judge only if the larger metrics clear the gates."
    )
    if build_harness:
        recommendation = (
            "Build a shadow-mode Granite harness, but keep TRACE as the primary runtime until live production traffic "
            "proves equivalent coverage, calibration, and latency."
        )

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "quantization": args.quantization or None,
        "max_model_len": args.max_model_len,
        "sampling": {"temperature": args.temperature, "max_tokens": args.max_tokens},
        "cold_load_ms": cold_load_ms,
        "dataset": {
            "case_count": len(cases),
            "by_lane": {
                lane: sum(1 for case in cases if case.lane == lane)
                for lane in sorted({case.lane for case in cases})
            },
            "by_band": {
                band: sum(1 for case in cases if case.expected_band == band)
                for band in sorted({case.expected_band for case in cases})
            },
        },
        "resolution": resolution,
        "strategy_metrics": strategy_metrics,
        "case_level": case_level,
        "quality_latency": quality_latency,
        "latency_sweep": latency_sweep,
        "decision_gate": {
            "replace_trace": replace_trace,
            "build_production_replacement_now": False,
            "build_production_harness_now": False,
            "build_shadow_harness_now": build_harness,
            "recommendation": recommendation,
            "reasons": "; ".join(reasons),
        },
        "rows": rows,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(report, args.out.with_suffix(".md"))

    print(f"cases: {len(cases)}")
    print(f"quality_prompts: {len(tasks)}")
    print(f"row_accuracy: {row_accuracy}")
    print(f"matrix_accuracy: {matrix_accuracy}")
    print(f"parse_rate: {parse_rate:.3f}")
    print(f"prompt_multiplier: {prompt_multiplier:.2f}")
    print(f"throughput_prompts_per_s: {quality_latency['throughput_prompts_per_s']:.2f}")
    print(f"report: {args.out}")
    print(f"proposal: {args.out.with_suffix('.md')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
