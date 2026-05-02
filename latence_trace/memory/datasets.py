"""Dataset adapters for the InfiniMem serious-v1 stack."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from latence_trace.memory.signature import extract_exact_critical_terms
from latence_trace.memory.trajectory import CanonicalSpan, CanonicalTrajectory, CanonicalTurn

Adapter = Callable[[dict[str, Any]], CanonicalTrajectory]


def load_dataset(path: Path, *, dataset: str) -> list[CanonicalTrajectory]:
    rows = _read_rows(path)
    adapter = adapter_for(dataset)
    return [adapter(row) for row in rows]


def adapter_for(dataset: str) -> Adapter:
    key = dataset.strip().lower().replace("_", "-")
    adapters: dict[str, Adapter] = {
        "longmemeval": adapt_longmemeval,
        "long-mem-eval": adapt_longmemeval,
        "mtrag": adapt_mtrag,
        "mtrag-mtrag": adapt_mtrag,
        "ragtruth": adapt_ragtruth,
        "garage": adapt_garage,
        "garag": adapt_garage,
        "swe-agent": adapt_swe_agent,
        "swe-agent-trajectories": adapt_swe_agent,
        "swe-bench-verified": adapt_swe_bench_verified,
        "swebench-verified": adapt_swe_bench_verified,
        "tau-bench": adapt_tau_bench,
        "taubench": adapt_tau_bench,
    }
    if key not in adapters:
        raise ValueError(f"Unsupported TRACE Memory dataset adapter: {dataset}")
    return adapters[key]


def adapt_longmemeval(row: dict[str, Any]) -> CanonicalTrajectory:
    messages = _messages(row)
    haystack = _spans(row.get("haystack_sessions") or [], 1, "retrieval_chunk", domain="chat")
    turns = [
        CanonicalTurn(
            turn_index=idx,
            query_text=_get(row, "question", "query") if idx == len(messages) else None,
            turn_text=_message_text(message),
            response_text=_get(row, "answer", "gold") if idx == len(messages) else _get(message, "answer", "response"),
            raw_context="\n".join(span.text for span in haystack) if idx == len(messages) and haystack else None,
            retrieved_context=haystack if idx == len(messages) else [],
            gold={"critical_terms": _critical_terms(row, domain="chat")},
        )
        for idx, message in enumerate(messages, start=1)
    ] or [
        CanonicalTurn(
            turn_index=1,
            query_text=_get(row, "question", "query"),
            turn_text=_get(row, "question", "query", default=""),
            response_text=_get(row, "answer", "response"),
            raw_context="\n".join(span.text for span in haystack) or None,
            retrieved_context=haystack,
            gold={"critical_terms": _critical_terms(row, domain="chat")},
        )
    ]
    return CanonicalTrajectory(
        dataset="LongMemEval",
        case_id=_case_id(row),
        domain="chat",
        turns=turns,
        retrieved_context=haystack,
        outcome=_outcome(row),
        gold={"critical_terms": _critical_terms(row, domain="chat"), "answer": _get(row, "answer", "gold")},
        metadata=_metadata(row),
    )


def adapt_mtrag(row: dict[str, Any]) -> CanonicalTrajectory:
    turns = []
    for idx, turn in enumerate(_turns(row), start=1):
        retrieved = _spans(_get(turn, "retrieved_context", "contexts", "passages") or [], idx, "retrieval_chunk", domain="rag")
        turns.append(
            CanonicalTurn(
                turn_index=idx,
                query_text=_get(turn, "question", "query"),
                turn_text=_get(turn, "question", "query", default=""),
                response_text=_get(turn, "answer", "response"),
                raw_context="\n".join(span.text for span in retrieved) or None,
                retrieved_context=retrieved,
                gold={"critical_terms": _critical_terms(row, turn, domain="rag")},
            )
        )
    return CanonicalTrajectory(
        dataset="MTRAG",
        case_id=_case_id(row),
        domain="rag",
        turns=turns or [_single_turn(row, domain="rag")],
        retrieved_context=[span for turn in turns for span in turn.retrieved_context],
        outcome=_outcome(row),
        gold={"critical_terms": _critical_terms(row, domain="rag")},
        metadata=_metadata(row),
    )


def adapt_ragtruth(row: dict[str, Any]) -> CanonicalTrajectory:
    context = _get(row, "source", "context", "passage", default="")
    hallucinations = _get(row, "hallucinations", "labels", default=[]) or []
    critical = _critical_terms(row, domain="grounding") + [
        str(item.get("text", "")) for item in hallucinations if isinstance(item, dict)
    ]
    anchor_index = _critical_index("grounding", [term for term in critical if term])
    return CanonicalTrajectory(
        dataset="RAGTruth",
        case_id=_case_id(row),
        domain="grounding",
        turns=[
            CanonicalTurn(
                turn_index=1,
                query_text=_get(row, "prompt", "question", "query"),
                turn_text=_get(row, "prompt", "question", "query", default=""),
                response_text=_get(row, "response", "answer", "output"),
                raw_context="\n".join(part for part in [anchor_index, context] if part),
                retrieved_context=_spans([context], 1, "retrieval_chunk", domain="grounding") if context else [],
                gold={"critical_terms": [term for term in critical if term]},
            )
        ],
        gold={"critical_terms": [term for term in critical if term]},
        outcome=_outcome(row),
        metadata=_metadata(row),
    )


def adapt_garage(row: dict[str, Any]) -> CanonicalTrajectory:
    passages = _get(row, "grounding_passages", "passages", "contexts", "grounding", default=[]) or []
    retrieved = _spans(passages, 1, "retrieval_chunk", domain="grounding")
    critical = _garage_critical_terms(row, retrieved)
    anchor_index = _critical_index("grounding", critical)
    return CanonicalTrajectory(
        dataset="GaRAGe",
        case_id=_case_id(row),
        domain="grounding",
        turns=[
            CanonicalTurn(
                turn_index=1,
                query_text=_get(row, "question", "query"),
                turn_text=_get(row, "question", "query", default=""),
                response_text=_get(row, "answer", "response", "answer_generate"),
                raw_context="\n".join(part for part in [anchor_index, *(span.text for span in retrieved)] if part) or None,
                retrieved_context=retrieved,
                gold={"critical_terms": critical},
            )
        ],
        retrieved_context=retrieved,
        gold={"critical_terms": critical},
        outcome=_outcome(row),
        metadata=_metadata(row),
    )


def adapt_swe_agent(row: dict[str, Any], *, max_steps: int | None = None) -> CanonicalTrajectory:
    steps = _get(row, "trajectory", "steps", "messages", default=[]) or []
    if max_steps is not None:
        steps = steps[:max_steps]
    turns = []
    for idx, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            step = {"text": str(step)}
        raw_text = _message_text(step) or _get(step, "thought", "action", default="")
        tool_outputs = _spans(
            _get(step, "observation", "tool_output", "stdout", default=[]),
            idx,
            "tool_result",
            domain="code",
        )
        if not tool_outputs and str(step.get("role", "")).lower() in {"tool", "environment", "system"}:
            tool_outputs = _spans([raw_text], idx, "tool_result", domain="code")
        turns.append(
            CanonicalTurn(
                turn_index=idx,
                query_text=_get(row, "problem_statement", "issue", "query"),
                turn_text=raw_text,
                response_text=_get(step, "response", "assistant", "action"),
                raw_context=_get(step, "observation", "tool_output", "stdout") or raw_text,
                tool_outputs=tool_outputs,
                gold={"critical_terms": _swe_agent_critical_terms(row)},
                metadata={
                    "step_index": idx,
                    "role": step.get("role"),
                    "branch_id": _get(step, "branch_id", "trajectory_id"),
                    "failed_branch": _is_failed_step(step),
                    "resolved_at": _get(row, "exit_status", "target"),
                },
            )
        )
    generated_patch = _get(row, "generated_patch", "patch", default="")
    if turns and generated_patch:
        last = turns[-1]
        response_text = "\n".join(part for part in [last.response_text or "", str(generated_patch)] if part.strip())
        raw_context = "\n".join(
            part
            for part in [_critical_index("code", _swe_agent_critical_terms(row)), last.raw_context or ""]
            if part.strip()
        )
        turns[-1] = last.model_copy(update={"response_text": response_text, "raw_context": raw_context})
    return CanonicalTrajectory(
        dataset="SWE-agent trajectories",
        case_id=_case_id(row),
        domain="code",
        turns=turns or [_single_turn(row, domain="code")],
        tool_outputs=[span for turn in turns for span in turn.tool_outputs],
        outcome=_outcome(row),
        gold={"critical_terms": _swe_agent_critical_terms(row)},
        metadata=_metadata(row),
    )


def stitch_long_code_trajectory(
    rows: list[dict[str, Any]],
    *,
    target_steps: int,
    case_id: str = "long_code_stress",
) -> CanonicalTrajectory:
    turns: list[CanonicalTurn] = []
    critical_terms: list[str] = []
    for row in rows:
        trajectory = adapt_swe_agent(row)
        critical_terms.extend(trajectory.critical_terms)
        for turn in trajectory.turns:
            turns.append(
                turn.model_copy(
                    update={
                        "turn_index": len(turns) + 1,
                        "metadata": {
                            **turn.metadata,
                            "stitched_source_case_id": trajectory.case_id,
                        },
                    }
                )
            )
            if len(turns) >= target_steps:
                break
        if len(turns) >= target_steps:
            break
    if not turns:
        raise ValueError("Cannot stitch long code trajectory without source turns")
    return CanonicalTrajectory(
        dataset="SWE-agent long-horizon stress",
        case_id=case_id,
        domain="code",
        turns=turns,
        outcome={"status": "unknown"},
        gold={"critical_terms": sorted(set(critical_terms))},
        metadata={"target_steps": target_steps, "stitched_cases": len(rows)},
    )


def adapt_swe_bench_verified(row: dict[str, Any]) -> CanonicalTrajectory:
    issue = _get(row, "problem_statement", "issue", "text", default="")
    patch = _get(row, "patch", "gold_patch", default="")
    tests = _get(row, "test_patch", "tests", default="")
    critical = _swe_bench_critical_terms(row, issue=issue, patch=patch, tests=tests)
    fail_to_pass = "\n".join(_fail_to_pass_tests(row))
    anchor_index = _critical_index("code", critical)
    return CanonicalTrajectory(
        dataset="SWE-bench Verified",
        case_id=_case_id(row),
        domain="code",
        turns=[
            CanonicalTurn(
                turn_index=1,
                query_text=issue,
                turn_text=issue,
                response_text=patch or tests or issue,
                raw_context="\n".join(part for part in [anchor_index, patch, tests, fail_to_pass] if part),
                gold={"critical_terms": critical},
            )
        ],
        outcome=_outcome(row),
        gold={"critical_terms": critical},
        metadata=_metadata(row),
    )


def adapt_tau_bench(row: dict[str, Any]) -> CanonicalTrajectory:
    turns = []
    for idx, turn in enumerate(_turns(row), start=1):
        tool_payload = _get(turn, "tool_outputs", "api_results", "observations", default=[])
        if not tool_payload and str(turn.get("role", "")).lower() in {"tool", "function"}:
            tool_payload = [turn]
        tool_outputs = _spans(tool_payload, idx, "tool_result")
        turns.append(
            CanonicalTurn(
                turn_index=idx,
                query_text=_get(turn, "user", "query", "instruction"),
                turn_text=_message_text(turn) or _get(turn, "user", "instruction", default=""),
                response_text=_get(turn, "assistant", "response"),
                raw_context="\n".join(span.text for span in tool_outputs) or None,
                tool_outputs=tool_outputs,
                gold={"critical_terms": _critical_terms(row, turn, domain="tool")},
            )
        )
    return CanonicalTrajectory(
        dataset="tau-bench",
        case_id=_case_id(row),
        domain="tool",
        turns=turns or [_single_turn(row, domain="tool")],
        tool_outputs=[span for turn in turns for span in turn.tool_outputs],
        outcome=_outcome(row),
        gold={"critical_terms": _critical_terms(row, domain="tool")},
        metadata=_metadata(row),
    )


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "rows", "examples"):
            if isinstance(data.get(key), list):
                return list(data[key])
        return [data]
    raise ValueError(f"Unsupported dataset file shape: {path}")


def _turns(row: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("turns", "conversation", "messages", "dialogue", "trajectory", "steps"):
        value = row.get(key)
        if isinstance(value, list):
            return [item if isinstance(item, dict) else {"text": str(item)} for item in value]
    return []


def _messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    return _turns(row)


def _single_turn(row: dict[str, Any], *, domain: str = "chat") -> CanonicalTurn:
    return CanonicalTurn(
        turn_index=1,
        query_text=_get(row, "question", "query", "problem_statement"),
        turn_text=_get(row, "question", "query", "instruction", "problem_statement", default=""),
        response_text=_get(row, "answer", "response", "output", "patch"),
        raw_context=_get(row, "context", "source", "passage"),
        gold={"critical_terms": _critical_terms(row), "domain": domain},
    )


def _spans(values: Any, turn_index: int, span_type: str, *, domain: str | None = None) -> list[CanonicalSpan]:
    if isinstance(values, str):
        values = [values]
    if isinstance(values, dict):
        values = [values]
    if not isinstance(values, Iterable):
        return []
    spans = []
    for idx, value in enumerate(values):
        text = _message_text(value) if isinstance(value, dict) else str(value)
        if not text.strip():
            continue
        for chunk_idx, chunk in enumerate(_chunk_text(text)):
            spans.append(
                CanonicalSpan(
                    id=f"span_{turn_index}_{idx}_{chunk_idx}",
                    text=chunk,
                    source=span_type,
                    turn_index=turn_index,
                    span_type=span_type,  # type: ignore[arg-type]
                    exact_critical_terms=_extract_code_terms(chunk, domain=domain or _span_domain(span_type)),
                    provenance=value if isinstance(value, dict) else {},
                    metadata={"chunk_index": chunk_idx},
                )
            )
    return spans


def _get(row: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            return row[key]
    return default


def _message_text(row: Any) -> str:
    if not isinstance(row, dict):
        return str(row)
    for key in ("content", "text", "message", "observation", "tool_output"):
        if row.get(key):
            return str(row[key])
    citation_parts = [str(value) for key, value in row.items() if key.startswith("cite_") and value]
    if citation_parts:
        return "\n".join(citation_parts)
    role = row.get("role") or row.get("speaker")
    content = row.get("assistant") or row.get("user") or row.get("response")
    return f"{role}: {content}" if role and content else str(content or "")


def _case_id(row: dict[str, Any]) -> str:
    return str(_get(row, "id", "case_id", "instance_id", "task_id", "conversation_id", default="case"))


def _critical_terms(*rows: dict[str, Any], domain: str | None = None) -> list[str]:
    terms: list[str] = []
    for row in rows:
        keys = ["critical_terms", "gold_terms", "evidence_terms"]
        if domain not in {"tool", "tau", "tau-bench", "code"}:
            keys.append("entities")
        for key in keys:
            value = row.get(key)
            if isinstance(value, list):
                terms.extend(str(item) for item in value if item)
        answer = _get(row, "answer", "gold", "expected_answer")
        if isinstance(answer, str):
            terms.extend(_short_answer_terms(answer, domain=domain))
            terms.extend(_extract_code_terms(answer, domain=domain))
        for key in (
            "output",
            "response",
            "answer_generate",
            "answer_related_info",
            "generated_patch",
            "eval_logs",
            "problem_statement",
            "test_patch",
        ):
            value = row.get(key)
            if isinstance(value, str):
                if domain == "code" and key == "eval_logs":
                    continue
                if key in {"output", "response", "answer_generate", "answer_related_info"}:
                    terms.extend(_short_answer_terms(value, domain=domain))
                terms.extend(extract_exact_critical_terms(value, domain=domain))
    return sorted({term for term in terms if term})


def _short_answer_terms(text: str, *, domain: str | None = None) -> list[str]:
    if domain == "code":
        return []
    clean = text.strip().strip("\"'")
    if not clean:
        return []
    words = clean.split()
    if 1 <= len(words) <= 18 and len(clean) <= 160:
        return [clean]
    return []


def _garage_critical_terms(row: dict[str, Any], retrieved: list[CanonicalSpan]) -> list[str]:
    terms = _critical_terms(row, domain="grounding")
    cited = row.get("evidence_cited") or []
    for idx, _span in enumerate(retrieved):
        cited_ok = idx < len(cited) and str(cited[idx]).upper() == "YES"
        if cited_ok:
            terms.append(f"cite_{idx + 1}")
    return sorted({term for term in terms if term})


def _swe_agent_critical_terms(row: dict[str, Any]) -> list[str]:
    terms: list[str] = []
    for key in ("problem_statement", "issue", "generated_patch", "patch"):
        value = row.get(key)
        if isinstance(value, str):
            terms.extend(_code_anchor_terms(value))
    return sorted({term for term in terms if term})


def _swe_bench_critical_terms(row: dict[str, Any], *, issue: str, patch: str, tests: str) -> list[str]:
    terms: list[str] = []
    terms.extend(_code_anchor_terms(issue))
    terms.extend(_code_anchor_terms(patch))
    terms.extend(_code_anchor_terms(tests))
    terms.extend(_normalize_test_id(item) for item in _fail_to_pass_tests(row))
    return sorted({term for term in terms if term})


def _fail_to_pass_tests(row: dict[str, Any], *, limit: int = 24) -> list[str]:
    value = row.get("FAIL_TO_PASS")
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = [value]
    elif isinstance(value, list):
        parsed = value
    else:
        parsed = []
    return [str(item) for item in parsed if item][:limit]


def _code_anchor_terms(text: str, *, include_public_symbols: bool = False) -> list[str]:
    anchors = []
    for term in extract_exact_critical_terms(text, domain="code"):
        if _is_code_anchor_term(term, include_public_symbols=include_public_symbols):
            normalized = _normalize_code_anchor(term)
            if normalized:
                anchors.append(normalized)
    return sorted(set(anchors))


def _critical_index(domain: str, terms: list[str], *, max_terms: int = 40) -> str:
    chunks = []
    unique_terms = sorted({term for term in terms if term})
    for start in range(0, len(unique_terms), max_terms):
        chunks.append(f"{domain}_exact_index " + " ".join(unique_terms[start : start + max_terms]))
    return "\n".join(chunks)


def _is_code_anchor_term(term: str, *, include_public_symbols: bool = False) -> bool:
    lowered = term.lower()
    if lowered in {"dev/null", "/dev/null", "none", "exception"}:
        return False
    if lowered.startswith(("github.com/", "http://", "https://")) or "docs/" in lowered:
        return False
    if any(fragment in lowered for fragment in ("site-packages", "/venv/", "/usr/local/", "/.heroku/")):
        return False
    if "/" in term or lowered.endswith((".py", ".ts", ".tsx", ".js", ".java", ".go", ".rs")):
        return True
    if "::" in term:
        return True
    if term.endswith(("Error", "Exception", "Warning")):
        return True
    return bool(include_public_symbols and ("." in term or "_" in term or any(char.isupper() for char in term[1:])))


def _normalize_code_anchor(term: str) -> str:
    clean = term.strip()
    if clean.startswith(("a/", "b/")):
        clean = clean[2:]
    if "::" in clean:
        clean = clean.rsplit("::", 1)[-1]
    if clean in {"dev/null", "/dev/null"}:
        return ""
    return clean


def _normalize_test_id(test_id: str) -> str:
    clean = _normalize_code_anchor(test_id)
    if " (" in clean:
        clean = clean.split(" (", 1)[0]
    if "[" in clean:
        clean = clean.split("[", 1)[0]
    return clean


def _extract_code_terms(text: str, *, domain: str | None = None) -> list[str]:
    terms = extract_exact_critical_terms(text, domain=domain or "code")
    if domain in {"tool", "tau", "tau-bench"}:
        return _tool_eval_terms(text, terms)
    return terms


def _tool_eval_terms(text: str, terms: list[str]) -> list[str]:
    lowered = text.lower()
    is_ephemeral_search = "available_seats" in lowered or '"status": "available"' in lowered
    has_many_flight_options = len(re.findall(r"\bHAT\d{3}\b", text)) > 1
    return [
        term
        for term in terms
        if not _is_noncritical_tool_term(term)
        and not ((is_ephemeral_search or has_many_flight_options) and _is_ephemeral_tool_search_term(term))
    ]


def _is_noncritical_tool_term(term: str) -> bool:
    lowered = term.lower()
    return (
        lowered == "example.com"
        or ".example.com" in lowered
        or "@" in lowered
        or bool(re.fullmatch(r"\d+,?", term))
    )


def _is_ephemeral_tool_search_term(term: str) -> bool:
    if re.fullmatch(r"\d{2}:\d{2}:\d{2}(?:\+1)?", term):
        return True
    if re.fullmatch(r"HAT\d{3}", term):
        return True
    return bool(re.fullmatch(r"\d+,?", term))


def _span_domain(span_type: str) -> str:
    if span_type == "tool_result":
        return "tool"
    if span_type == "retrieval_chunk":
        return "rag"
    if span_type.startswith("code"):
        return "code"
    return span_type


def _outcome(row: dict[str, Any]) -> dict[str, Any]:
    status = _get(row, "status", "outcome", default="unknown")
    if isinstance(status, bool):
        status = "success" if status else "failure"
    if status not in {"success", "failure", "unknown", "partial"}:
        lowered = str(status).lower()
        status = "success" if lowered in {"pass", "passed", "resolved", "correct"} else "unknown"
    return {"status": status, "metadata": {"raw_outcome": row.get("outcome")}}


def _metadata(row: dict[str, Any]) -> dict[str, Any]:
    excluded = {"turns", "conversation", "messages", "dialogue", "trajectory", "steps"}
    return {key: value for key, value in row.items() if key not in excluded}


def _chunk_text(text: str, *, max_words: int = 220) -> list[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text]
    chunks = []
    for start in range(0, len(words), max_words):
        chunks.append(" ".join(words[start : start + max_words]))
    return chunks


def _is_failed_step(step: dict[str, Any]) -> bool:
    text = _message_text(step).lower()
    return any(marker in text for marker in ("failed", "error", "traceback", "exception", "pytest failed"))
