"""Mem0 benchmark compatibility adapter backed by TRACE InfiniMem.

The upstream ``memory-benchmarks`` suite expects a small async client surface:
``add()``, ``search()``, and ``delete_user()``.  This adapter keeps that contract
while routing ingestion through ``TraceSessionService`` and search through the
current TRACE memory state plus source-vault repair excerpts.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from latence_trace.memory.models import MemoryPolicy, SpanRecord
from latence_trace.memory.signature import extract_exact_critical_terms
from latence_trace.sessions.models import (
    TraceSessionCreateRequest,
    TraceSessionEvent,
    TraceSessionEventRequest,
    TraceSessionRepairRequest,
)
from latence_trace.sessions.service import TraceSessionService

_TOKEN_RE = re.compile(r"\b[A-Za-z0-9][A-Za-z0-9_.:/@-]{2,}\b")


@dataclass
class TraceMem0AdapterConfig:
    """Configuration shared by benchmark-created adapter instances."""

    kind: str = "rag"
    memory_domain: str = "rag"
    top_k_context_result: bool = False
    repair_max_excerpts: int = 8
    repair_max_tokens: int = 2048
    memory_policy: MemoryPolicy = field(default_factory=MemoryPolicy)


_GLOBAL_CONFIG = TraceMem0AdapterConfig()
_CLIENTS: list[TraceMemoryClient] = []


def configure_trace_mem0_adapter(config: TraceMem0AdapterConfig) -> None:
    """Set process-wide defaults used when Mem0 benchmark code constructs a client."""

    global _GLOBAL_CONFIG
    _GLOBAL_CONFIG = config


def trace_mem0_diagnostics() -> dict[str, Any]:
    """Return aggregate diagnostics for all adapter instances in this process."""

    users = {}
    for client in _CLIENTS:
        users.update(client.diagnostics_snapshot())
    return {
        "users": users,
        "client_count": len(_CLIENTS),
        "created_at": datetime.now(UTC).isoformat(),
    }


class TraceMemoryClient:
    """Async Mem0-compatible client backed by process-local TRACE sessions."""

    def __init__(
        self,
        mode: str = "oss",
        host: str | None = None,
        api_key: str | None = None,
        organization_id: str | None = None,
        project_id: str | None = None,
        max_retries: int = 5,
        retry_delay: float = 5.0,
        rpm: int = 60,
        timeout: float = 300.0,
        event_poll_interval: float = 0.5,
        event_poll_timeout: float = 300.0,
        *,
        config: TraceMem0AdapterConfig | None = None,
    ) -> None:
        del mode, host, api_key, organization_id, project_id
        del max_retries, retry_delay, rpm, timeout, event_poll_interval, event_poll_timeout
        self.config = config or _GLOBAL_CONFIG
        self._service = TraceSessionService(groundedness_service=object())
        self._user_sessions: dict[str, str] = {}
        self._user_events: dict[str, int] = {}
        self._last_search: dict[str, dict[str, Any]] = {}
        self._source_created_at: dict[str, str] = {}
        _CLIENTS.append(self)

    async def __aenter__(self) -> TraceMemoryClient:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    async def close(self) -> None:
        return None

    async def add(
        self,
        messages: list[dict[str, str]],
        user_id: str,
        observation_date: str | None = None,
        timestamp: int | None = None,
        custom_instructions: str | None = None,
        metadata: dict | None = None,
    ) -> dict | None:
        """Add benchmark conversation messages to a TRACE session."""

        session_id = self._ensure_session(user_id)
        content = _format_messages(messages)
        if custom_instructions:
            content = f"Custom instructions: {custom_instructions}\n{content}"
        event_metadata = dict(metadata or {})
        if observation_date:
            event_metadata["observation_date"] = observation_date
        if timestamp is not None:
            event_metadata["timestamp_epoch"] = timestamp
        event_timestamp = _timestamp_to_iso(timestamp) or observation_date
        response = self._service.append_event(
            session_id,
            TraceSessionEventRequest(
                memory_domain=self.config.memory_domain,
                event=TraceSessionEvent(
                    event_type="observation",
                    content=content,
                    raw_context=content,
                    query_text=_latest_role_content(messages, "user"),
                    response_text=_latest_role_content(messages, "assistant"),
                    metadata=event_metadata,
                    timestamp=event_timestamp,
                ),
            ),
        )
        self._user_events[user_id] = self._user_events.get(user_id, 0) + 1
        source_id = response.source_pointer.source_id if response.source_pointer else ""
        extracted = []
        memory_state = response.session.memory_state
        if memory_state is not None:
            for span in memory_state.spans:
                pointer = span.provenance.get("source_pointer", {})
                if pointer.get("source_id") == source_id:
                    if source_id and event_timestamp:
                        self._source_created_at[source_id] = event_timestamp
                    extracted.append(
                        {
                            "id": span.id,
                            "event": "ADD",
                            "memory": span.text,
                            "score": _span_static_score(span),
                        }
                    )
        if not extracted and content:
            extracted.append({"id": source_id or "", "event": "ADD", "memory": content, "score": 1.0})
        return {"results": extracted}

    async def search(
        self,
        query: str,
        user_id: str,
        top_k: int = 200,
        rerank: bool = False,
        score_debug: bool = False,
    ) -> list[dict]:
        """Search TRACE memory and return Mem0-shaped ranked memories."""

        del rerank
        session_id = self._user_sessions.get(user_id)
        if session_id is None:
            return []
        session = self._service.get(session_id).session
        if session.memory_state is None:
            return []

        query_terms = extract_exact_critical_terms(query, domain=self.config.memory_domain)
        query_tokens = _tokens(query)
        rows = []
        for span in session.memory_state.spans:
            if span.compression_level == "tombstone" or not span.text.strip():
                continue
            rows.append(
                _span_to_result(
                    span,
                    query_tokens,
                    query_terms,
                    created_at=self._created_at_for_span(span),
                    score_debug=score_debug,
                )
            )

        context = self._service.context(session_id)
        repair_rows = self._repair_rows(
            session_id=session_id,
            query=query,
            query_terms=query_terms,
            hot_context=context.hot_context,
            score_debug=score_debug,
        )
        rows.extend(repair_rows)

        if self.config.top_k_context_result and context.context_for_generation.strip():
            rows.append(
                {
                    "memory": context.context_for_generation,
                    "score": 1.0,
                    "id": f"{session_id}:context_for_generation",
                    "score_debug": {
                        "combined_score": 1.0,
                        "source": "trace_context_for_generation",
                    },
                }
            )

        rows = _dedupe_results(rows)
        rows.sort(key=lambda item: item.get("score", 0.0), reverse=True)
        limited = rows[: max(0, top_k)]
        self._last_search[user_id] = {
            "query": query,
            "top_k": top_k,
            "returned": len(limited),
            "repair_results": len(repair_rows),
            "hot_tokens": context.hot_tokens,
            "warm_tokens": context.warm_tokens,
            "cold_tokens": context.cold_tokens,
            "diagnostics": context.diagnostics,
        }
        if not score_debug:
            for row in limited:
                row.pop("score_debug", None)
        return limited

    async def delete_user(self, user_id: str) -> bool:
        """Reset the adapter mapping for a benchmark user."""

        self._user_sessions.pop(user_id, None)
        self._user_events.pop(user_id, None)
        self._last_search.pop(user_id, None)
        return True

    async def get_user_profile(self, user_id: str) -> dict[str, Any] | None:
        """Mem0 benchmark optional hook; TRACE does not synthesize profiles."""

        del user_id
        return None

    def diagnostics_snapshot(self) -> dict[str, Any]:
        """Return per-user TRACE memory diagnostics for reporting."""

        snapshot = {}
        for user_id, session_id in self._user_sessions.items():
            context = self._service.context(session_id)
            snapshot[user_id] = {
                "session_id": session_id,
                "events": self._user_events.get(user_id, 0),
                "hot_tokens": context.hot_tokens,
                "warm_tokens": context.warm_tokens,
                "cold_tokens": context.cold_tokens,
                "diagnostics": context.diagnostics,
                "last_search": self._last_search.get(user_id, {}),
            }
        return snapshot

    def _ensure_session(self, user_id: str) -> str:
        existing = self._user_sessions.get(user_id)
        if existing is not None:
            return existing
        session_id = f"trace_mem0_{abs(hash(user_id)) % 10**16:x}"
        created = self._service.create(
            TraceSessionCreateRequest(
                session_id=session_id,
                kind=self.config.kind,  # type: ignore[arg-type]
                memory_policy=self.config.memory_policy,
                metadata={"mem0_benchmark_user_id": user_id},
            )
        )
        self._user_sessions[user_id] = created.session.session_id
        self._user_events[user_id] = 0
        return created.session.session_id

    def _repair_rows(
        self,
        *,
        session_id: str,
        query: str,
        query_terms: list[str],
        hot_context: str,
        score_debug: bool,
    ) -> list[dict[str, Any]]:
        missing_terms = [
            term for term in query_terms if term.lower() not in hot_context.lower()
        ]
        if not missing_terms:
            return []
        packet = self._service.repair(
            session_id,
            TraceSessionRepairRequest(
                query_text=query,
                missing_terms=missing_terms,
                reason="mem0 benchmark query-specific source-vault repair",
                max_excerpts=self.config.repair_max_excerpts,
                max_tokens=self.config.repair_max_tokens,
            ),
        ).repair_packet
        rows = []
        for idx, excerpt in enumerate(packet.excerpts):
            overlap = _term_overlap(excerpt.text, query_terms)
            score = min(1.0, 0.92 + 0.02 * overlap)
            row = {
                "memory": excerpt.text,
                "score": score,
                "id": f"repair:{excerpt.source_id}:{idx}",
            }
            created_at = self._source_created_at.get(excerpt.source_id)
            if created_at:
                row["created_at"] = created_at
            if score_debug:
                row["score_debug"] = {
                    "combined_score": score,
                    "source": "trace_source_vault_repair",
                    "matched_terms": excerpt.matched_terms,
                }
            rows.append(row)
        return rows

    def _created_at_for_span(self, span: SpanRecord) -> str:
        pointer = span.provenance.get("source_pointer", {})
        source_id = str(pointer.get("source_id") or "")
        return self._source_created_at.get(source_id, "")


def _format_messages(messages: list[dict[str, str]]) -> str:
    parts = []
    for message in messages:
        role = str(message.get("role") or "message").strip() or "message"
        content = str(message.get("content") or "").strip()
        if content:
            parts.append(f"{role}: {content}")
    return "\n".join(parts)


def _latest_role_content(messages: list[dict[str, str]], role: str) -> str | None:
    for message in reversed(messages):
        if message.get("role") == role and message.get("content"):
            return str(message["content"])
    return None


def _timestamp_to_iso(timestamp: int | None) -> str | None:
    if timestamp is None:
        return None
    try:
        return datetime.fromtimestamp(timestamp, tz=UTC).isoformat()
    except (OSError, OverflowError, ValueError):
        return None


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in _TOKEN_RE.findall(text)}


def _term_overlap(text: str, terms: list[str]) -> int:
    lowered = text.lower()
    return sum(1 for term in terms if term.lower() in lowered)


def _span_to_result(
    span: SpanRecord,
    query_tokens: set[str],
    query_terms: list[str],
    *,
    created_at: str,
    score_debug: bool,
) -> dict[str, Any]:
    text_tokens = _tokens(span.text)
    lexical = len(query_tokens & text_tokens) / max(1.0, math.sqrt(len(query_tokens) * len(text_tokens)))
    exact_overlap = _term_overlap(span.text, query_terms) / max(1, len(query_terms))
    static = _span_static_score(span)
    layer_boost = {"hot": 0.18, "warm": 0.08, "cold": 0.0, "tombstone": -1.0}.get(span.layer, 0.0)
    score = min(1.0, 0.48 * lexical + 0.24 * exact_overlap + 0.22 * static + layer_boost)
    result: dict[str, Any] = {
        "memory": span.text,
        "score": round(score, 6),
        "id": span.id,
    }
    if created_at:
        result["created_at"] = created_at
    if score_debug:
        result["score_debug"] = {
            "combined_score": result["score"],
            "lexical_score": round(lexical, 6),
            "exact_overlap": round(exact_overlap, 6),
            "trace_static_score": round(static, 6),
            "layer": span.layer,
            "span_type": span.span_type,
        }
    return result


def _span_static_score(span: SpanRecord) -> float:
    return max(
        0.0,
        min(
            1.0,
            0.40 * span.scores.survival_value
            + 0.25 * span.scores.exact_critical
            + 0.20 * span.scores.learned_survival
            + 0.15 * span.scores.relevance,
        ),
    )


def _dedupe_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_text: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = re.sub(r"\s+", " ", str(row.get("memory", "")).strip().lower())
        if not key:
            continue
        previous = by_text.get(key)
        if previous is None or row.get("score", 0.0) > previous.get("score", 0.0):
            by_text[key] = row
    return list(by_text.values())


def write_trace_mem0_diagnostics(path: str) -> None:
    """Write aggregate adapter diagnostics as JSON."""

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(trace_mem0_diagnostics(), handle, indent=2, sort_keys=True)
