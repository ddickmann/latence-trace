"""Stateful TRACE session service."""

from __future__ import annotations

import threading
from copy import deepcopy
from datetime import UTC, datetime
from typing import Any, Protocol
from uuid import uuid4

from latence_trace.api.models import GroundednessRequest, RollupRequest
from latence_trace.api.service import GroundednessService, ValidationError
from latence_trace.memory.models import MemoryPolicy, MemoryState, MemoryUpdateRequest
from latence_trace.memory.select import hot_context
from latence_trace.memory.service import update_memory
from latence_trace.sessions.models import (
    TraceSessionCloseResponse,
    TraceSessionContextResponse,
    TraceSessionCreateRequest,
    TraceSessionCreateResponse,
    TraceSessionEvent,
    TraceSessionEventRequest,
    TraceSessionEventResponse,
    TraceSessionGetResponse,
    TraceSessionRollupRequest,
    TraceSessionRollupResponse,
    TraceSessionScoreRequest,
    TraceSessionScoreResponse,
    TraceSessionState,
)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


class SessionStore(Protocol):
    """Persistence boundary for TRACE sessions."""

    def create(self, session: TraceSessionState) -> TraceSessionState: ...

    def get(self, session_id: str) -> TraceSessionState | None: ...

    def save(self, session: TraceSessionState) -> TraceSessionState: ...

    def append_event(self, session_id: str, event: TraceSessionEvent) -> None: ...

    def events(self, session_id: str) -> list[TraceSessionEvent]: ...


class InMemorySessionStore:
    """Process-local store used for local tests and direct RunPod actions."""

    def __init__(self) -> None:
        self._sessions: dict[str, TraceSessionState] = {}
        self._events: dict[str, list[TraceSessionEvent]] = {}
        self._lock = threading.Lock()

    def create(self, session: TraceSessionState) -> TraceSessionState:
        with self._lock:
            if session.session_id in self._sessions:
                raise ValidationError(f"TRACE session already exists: {session.session_id}")
            self._sessions[session.session_id] = deepcopy(session)
            self._events.setdefault(session.session_id, [])
            return deepcopy(session)

    def get(self, session_id: str) -> TraceSessionState | None:
        with self._lock:
            session = self._sessions.get(session_id)
            return deepcopy(session) if session is not None else None

    def save(self, session: TraceSessionState) -> TraceSessionState:
        with self._lock:
            self._sessions[session.session_id] = deepcopy(session)
            self._events.setdefault(session.session_id, [])
            return deepcopy(session)

    def append_event(self, session_id: str, event: TraceSessionEvent) -> None:
        with self._lock:
            self._events.setdefault(session_id, []).append(deepcopy(event))

    def events(self, session_id: str) -> list[TraceSessionEvent]:
        with self._lock:
            return deepcopy(self._events.get(session_id, []))


class TraceSessionService:
    """Continuous stateful layer over Core TRACE and InfiniMem."""

    def __init__(
        self,
        *,
        groundedness_service: GroundednessService,
        store: SessionStore | None = None,
    ) -> None:
        self._groundedness = groundedness_service
        self._store = store or InMemorySessionStore()

    def create(self, request: TraceSessionCreateRequest) -> TraceSessionCreateResponse:
        now = _now_iso()
        session = TraceSessionState(
            session_id=request.session_id or _new_id("trcsess"),
            kind=request.kind,
            state_backend=request.state_backend,
            privacy_mode=request.privacy_mode,
            created_at=now,
            updated_at=now,
            memory_policy=request.memory_policy or MemoryPolicy(),
            metadata=dict(request.metadata),
            redaction_policy=dict(request.redaction_policy),
            retention_policy=dict(request.retention_policy),
        )
        return TraceSessionCreateResponse(session=self._store.create(session))

    def get(self, session_id: str) -> TraceSessionGetResponse:
        return TraceSessionGetResponse(session=self._require_session(session_id))

    def append_event(
        self,
        session_id: str,
        request: TraceSessionEventRequest,
    ) -> TraceSessionEventResponse:
        session = self._require_active_session(session_id)
        event = self._prepare_event(request.event)
        if self._is_duplicate(session, event):
            return TraceSessionEventResponse(
                session=session,
                event=event,
                hot_context=self._hot_context(session) if request.return_context else "",
            )
        result = update_memory(
            MemoryUpdateRequest(
                turn_text=event.content,
                query_text=event.query_text,
                response_text=event.response_text,
                raw_context=event.raw_context,
                prior_memory_state=session.memory_state,
                trace_signals=event.trace_signals,
                memory_domain=request.memory_domain or self._memory_domain(session),
                memory_policy=session.memory_policy,
            )
        )
        session.memory_state = result.next_memory_state
        self._record_event(session, event)
        saved = self._store.save(session)
        return TraceSessionEventResponse(
            session=saved,
            event=event,
            hot_context=result.hot_context if request.return_context else "",
            memory_diagnostics=result.diagnostics,
        )

    def score(
        self,
        session_id: str,
        request: TraceSessionScoreRequest,
    ) -> TraceSessionScoreResponse:
        session = self._require_active_session(session_id)
        if request.event is not None:
            event = self._prepare_event(request.event)
            if not self._is_duplicate(session, event):
                self._record_event(session, event)
        payload = dict(request.trace_request or {})
        if request.lane and "scoring_mode" not in payload:
            payload["scoring_mode"] = request.lane
        if session.kind in {"code", "rag"} and "scoring_mode" not in payload:
            payload["scoring_mode"] = session.kind
        if "session_id" not in payload:
            payload["session_id"] = session.session_id
        if session.code_session_state and "session_state" not in payload:
            payload["session_state"] = session.code_session_state
        payload.setdefault("enable_memory_shadow", True)
        payload.setdefault("memory_policy", session.memory_policy.model_dump(mode="json"))
        if session.memory_state and "memory_state" not in payload:
            payload["memory_state"] = session.memory_state.model_dump(mode="json")
        if not _has_premise(payload):
            context = self._hot_context(session)
            if context:
                payload["raw_context"] = context
        trace_request = GroundednessRequest.model_validate(payload)
        response = self._groundedness.groundedness(trace_request)
        response_payload = response.model_dump(mode="json", exclude_none=True)
        session.code_session_state = (
            response.next_session_state.model_dump(mode="json")
            if response.next_session_state is not None
            else session.code_session_state
        )
        if response.next_memory_state is not None:
            session.memory_state = response.next_memory_state
        session.rollup_turns.append(_rollup_turn_from_trace(response_payload))
        session.rollup_turns = session.rollup_turns[-1000:]
        session.updated_at = _now_iso()
        saved = self._store.save(session)
        return TraceSessionScoreResponse(
            session=saved,
            trace_response=response_payload,
            hot_context=self._hot_context(saved) if request.return_context else "",
            memory_diagnostics=response.memory_diagnostics,
        )

    def context(self, session_id: str) -> TraceSessionContextResponse:
        session = self._require_session(session_id)
        diagnostics = _memory_budget(session.memory_state)
        return TraceSessionContextResponse(
            session_id=session.session_id,
            kind=session.kind,
            status=session.status,
            hot_context=self._hot_context(session),
            hot_tokens=diagnostics["hot_tokens"],
            warm_tokens=diagnostics["warm_tokens"],
            cold_tokens=diagnostics["cold_tokens"],
            memory_state=session.memory_state,
            diagnostics=diagnostics,
        )

    def rollup(
        self,
        session_id: str,
        request: TraceSessionRollupRequest,
    ) -> TraceSessionRollupResponse:
        session = self._require_session(session_id)
        turns = [*session.rollup_turns, *request.extra_turns]
        rollup = self._groundedness.rollup(
            RollupRequest(
                turns=turns,
                session_id=session.session_id,
                heatmap_format=request.heatmap_format,
            )
        )
        return TraceSessionRollupResponse(
            session=session,
            rollup=rollup.model_dump(mode="json", exclude_none=True),
        )

    def close(self, session_id: str) -> TraceSessionCloseResponse:
        session = self._require_session(session_id)
        session.status = "closed"
        session.updated_at = _now_iso()
        return TraceSessionCloseResponse(session=self._store.save(session))

    def _require_session(self, session_id: str) -> TraceSessionState:
        session = self._store.get(session_id)
        if session is None:
            raise ValidationError(f"TRACE session not found: {session_id}")
        return session

    def _require_active_session(self, session_id: str) -> TraceSessionState:
        session = self._require_session(session_id)
        if session.status != "active":
            raise ValidationError(f"TRACE session is not active: {session_id}")
        return session

    def _record_event(self, session: TraceSessionState, event: TraceSessionEvent) -> None:
        self._store.append_event(session.session_id, event)
        session.last_event_id = event.event_id
        session.event_count += 1
        session.updated_at = _now_iso()
        seen = list(session.metadata.get("_idempotency_keys", []))
        if event.idempotency_key:
            seen.append(event.idempotency_key)
            session.metadata["_idempotency_keys"] = seen[-512:]

    def _prepare_event(self, event: TraceSessionEvent) -> TraceSessionEvent:
        if event.event_id is None:
            event = event.model_copy(update={"event_id": _new_id("evt")})
        if event.timestamp is None:
            event = event.model_copy(update={"timestamp": _now_iso()})
        return event

    def _is_duplicate(self, session: TraceSessionState, event: TraceSessionEvent) -> bool:
        return bool(
            event.idempotency_key
            and event.idempotency_key in set(session.metadata.get("_idempotency_keys", []))
        )

    def _memory_domain(self, session: TraceSessionState) -> str:
        if session.kind == "code":
            return "code"
        if session.kind == "rag":
            return "rag"
        return "chat"

    def _hot_context(self, session: TraceSessionState) -> str:
        if session.memory_state is None:
            return ""
        return hot_context(session.memory_state.spans)


def _has_premise(payload: dict[str, Any]) -> bool:
    return bool(payload.get("raw_context") or payload.get("chunk_ids") or payload.get("support_units"))


def _memory_budget(memory_state: MemoryState | None) -> dict[str, Any]:
    if memory_state is None:
        return {"hot_tokens": 0, "warm_tokens": 0, "cold_tokens": 0, "span_count": 0}
    hot = sum(span.token_count for span in memory_state.spans if span.layer == "hot")
    warm = sum(span.token_count for span in memory_state.spans if span.layer == "warm")
    cold = sum(span.token_count for span in memory_state.spans if span.layer == "cold")
    return {
        "hot_tokens": hot,
        "warm_tokens": warm,
        "cold_tokens": cold,
        "span_count": len(memory_state.spans),
        "turn_index": memory_state.turn_index,
    }


def _rollup_turn_from_trace(response: dict[str, Any]) -> dict[str, Any]:
    return {
        "scores": response.get("scores"),
        "risk_band": response.get("risk_band"),
        "file_attribution": response.get("file_attribution"),
        "session_signals": response.get("session_signals"),
        "recommendation": (
            response.get("session_signals", {}).get("recommendation")
            if isinstance(response.get("session_signals"), dict)
            else None
        ),
    }
