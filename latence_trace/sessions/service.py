"""Stateful TRACE session service."""

from __future__ import annotations

import hashlib
import re
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
from latence_trace.memory.signature import extract_exact_critical_terms
from latence_trace.sessions.models import (
    TraceRepairExcerpt,
    TraceRepairPacket,
    TraceRepairTrigger,
    TraceSessionCloseResponse,
    TraceSessionContextResponse,
    TraceSessionCreateRequest,
    TraceSessionCreateResponse,
    TraceSessionEvent,
    TraceSessionEventRequest,
    TraceSessionEventResponse,
    TraceSessionGetResponse,
    TraceSessionRepairRequest,
    TraceSessionRepairResponse,
    TraceSessionRollupRequest,
    TraceSessionRollupResponse,
    TraceSessionScoreRequest,
    TraceSessionScoreResponse,
    TraceSessionSourceResponse,
    TraceSessionState,
    TraceSourcePointer,
    TraceSourceRecord,
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

    def append_source(self, record: TraceSourceRecord) -> None: ...

    def source(self, session_id: str, source_id: str) -> TraceSourceRecord | None: ...

    def sources(self, session_id: str) -> list[TraceSourceRecord]: ...


class InMemorySessionStore:
    """Process-local store used for local tests and direct RunPod actions."""

    def __init__(self) -> None:
        self._sessions: dict[str, TraceSessionState] = {}
        self._events: dict[str, list[TraceSessionEvent]] = {}
        self._sources: dict[str, list[TraceSourceRecord]] = {}
        self._lock = threading.Lock()

    def create(self, session: TraceSessionState) -> TraceSessionState:
        with self._lock:
            if session.session_id in self._sessions:
                raise ValidationError(f"TRACE session already exists: {session.session_id}")
            self._sessions[session.session_id] = deepcopy(session)
            self._events.setdefault(session.session_id, [])
            self._sources.setdefault(session.session_id, [])
            return deepcopy(session)

    def get(self, session_id: str) -> TraceSessionState | None:
        with self._lock:
            session = self._sessions.get(session_id)
            return deepcopy(session) if session is not None else None

    def save(self, session: TraceSessionState) -> TraceSessionState:
        with self._lock:
            self._sessions[session.session_id] = deepcopy(session)
            self._events.setdefault(session.session_id, [])
            self._sources.setdefault(session.session_id, [])
            return deepcopy(session)

    def append_event(self, session_id: str, event: TraceSessionEvent) -> None:
        with self._lock:
            self._events.setdefault(session_id, []).append(deepcopy(event))

    def events(self, session_id: str) -> list[TraceSessionEvent]:
        with self._lock:
            return deepcopy(self._events.get(session_id, []))

    def append_source(self, record: TraceSourceRecord) -> None:
        with self._lock:
            records = self._sources.setdefault(record.session_id, [])
            existing = next((item for item in records if item.source_id == record.source_id), None)
            if existing is not None:
                if existing.content_hash != record.content_hash:
                    raise ValidationError(f"immutable TRACE source already exists: {record.source_id}")
                return
            records.append(deepcopy(record))

    def source(self, session_id: str, source_id: str) -> TraceSourceRecord | None:
        with self._lock:
            for record in self._sources.get(session_id, []):
                if record.source_id == source_id:
                    return deepcopy(record)
            return None

    def sources(self, session_id: str) -> list[TraceSourceRecord]:
        with self._lock:
            return deepcopy(self._sources.get(session_id, []))


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
        self._repair = TraceRepairService(self._store)

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
        source_record, source_pointer = _source_from_event(session, event)
        result = update_memory(
            MemoryUpdateRequest(
                turn_text=event.content,
                query_text=event.query_text,
                response_text=event.response_text,
                raw_context=event.raw_context,
                prior_memory_state=session.memory_state,
                trace_signals=event.trace_signals,
                source_pointer=source_pointer.model_dump(mode="json"),
                memory_domain=request.memory_domain or self._memory_domain(session),
                memory_policy=session.memory_policy,
            )
        )
        session.memory_state = result.next_memory_state
        _attach_source_pointer(session.memory_state, source_pointer)
        self._store.append_source(source_record)
        _index_source_terms(session, source_record, request.memory_domain or self._memory_domain(session))
        self._record_event(session, event)
        saved = self._store.save(session)
        return TraceSessionEventResponse(
            session=saved,
            event=event,
            hot_context=result.hot_context if request.return_context else "",
            memory_diagnostics=result.diagnostics,
            source_pointer=source_pointer,
        )

    def score(
        self,
        session_id: str,
        request: TraceSessionScoreRequest,
    ) -> TraceSessionScoreResponse:
        session = self._require_active_session(session_id)
        source_pointer: TraceSourcePointer | None = None
        if request.event is not None:
            event = self._prepare_event(request.event)
            if not self._is_duplicate(session, event):
                source_record, source_pointer = _source_from_event(session, event)
                self._store.append_source(source_record)
                _index_source_terms(session, source_record, request.memory_domain or self._memory_domain(session))
                self._record_event(session, event)
        payload = dict(request.trace_request or {})
        _normalize_trace_request_aliases(payload)
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
        if source_pointer is None:
            synthetic_event = self._prepare_event(_event_from_trace_payload(payload))
            source_record, source_pointer = _source_from_event(session, synthetic_event)
            if not self._is_duplicate(session, synthetic_event):
                self._store.append_source(source_record)
                _index_source_terms(session, source_record, request.memory_domain or self._memory_domain(session))
                self._record_event(session, synthetic_event)
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
            _attach_source_pointer(session.memory_state, source_pointer)
        session.rollup_turns.append(_rollup_turn_from_trace(response_payload))
        session.rollup_turns = session.rollup_turns[-1000:]
        session.updated_at = _now_iso()
        repair_packet = self._repair.evaluate(
            session=session,
            request=request,
            trace_payload=response_payload,
            hot_context=self._hot_context(session),
        )
        saved = self._store.save(session)
        return TraceSessionScoreResponse(
            session=saved,
            trace_response=response_payload,
            hot_context=self._hot_context(saved) if request.return_context else "",
            memory_diagnostics=response.memory_diagnostics,
            repair_packet=repair_packet if repair_packet.triggered else None,
        )

    def source(
        self,
        session_id: str,
        source_id: str,
        *,
        include_raw: bool = False,
    ) -> TraceSessionSourceResponse:
        self._require_session(session_id)
        source = self._store.source(session_id, source_id)
        if source is not None and not include_raw:
            source = _redacted_source(source)
        return TraceSessionSourceResponse(session_id=session_id, source=source)

    def repair(
        self,
        session_id: str,
        request: TraceSessionRepairRequest,
    ) -> TraceSessionRepairResponse:
        session = self._require_session(session_id)
        packet = self._repair.build_manual_packet(session, request)
        return TraceSessionRepairResponse(session_id=session_id, repair_packet=packet)

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


class TraceRepairService:
    """Build minimal repair packets from immutable session source records."""

    def __init__(self, store: SessionStore) -> None:
        self._store = store

    def evaluate(
        self,
        *,
        session: TraceSessionState,
        request: TraceSessionScoreRequest,
        trace_payload: dict[str, Any],
        hot_context: str,
    ) -> TraceRepairPacket:
        triggers = _repair_triggers(
            session=session,
            request=request,
            trace_payload=trace_payload,
            hot_context=hot_context,
        )
        if not triggers:
            return TraceRepairPacket(triggered=False)
        terms = _dedupe([term for trigger in triggers for term in trigger.terms])
        source_request = TraceSessionRepairRequest(
            query_text=_trace_text(request.trace_request, "query_text"),
            response_text=_trace_text(request.trace_request, "response_text"),
            missing_terms=terms,
            reason="; ".join(trigger.reason for trigger in triggers),
        )
        return self._packet_from_sources(session.session_id, source_request, triggers=triggers)

    def build_manual_packet(
        self,
        session: TraceSessionState,
        request: TraceSessionRepairRequest,
    ) -> TraceRepairPacket:
        terms = _terms_from_repair_request(request, domain=_session_domain(session))
        trigger = TraceRepairTrigger(
            trigger_type="manual_repair",
            severity="warn",
            reason=request.reason or "manual source-vault repair requested",
            terms=terms,
        )
        return self._packet_from_sources(session.session_id, request, triggers=[trigger])

    def _packet_from_sources(
        self,
        session_id: str,
        request: TraceSessionRepairRequest,
        *,
        triggers: list[TraceRepairTrigger],
    ) -> TraceRepairPacket:
        terms = _dedupe([*request.missing_terms, *_terms_from_repair_request(request)])
        excerpts: list[TraceRepairExcerpt] = []
        budget = request.max_tokens
        for record in self._store.sources(session_id):
            if len(excerpts) >= request.max_excerpts or budget <= 0:
                break
            for field, value in _source_fields(record).items():
                if len(excerpts) >= request.max_excerpts or budget <= 0:
                    break
                matched = _matched_terms(value, terms)
                if not matched and terms:
                    continue
                if not matched and not _text_overlap(value, request.query_text or request.response_text or ""):
                    continue
                excerpt_text = _bounded_excerpt(value, matched, max_tokens=min(120, budget))
                if not excerpt_text:
                    continue
                if not request.include_raw:
                    excerpt_text = _redact_text(excerpt_text)
                token_count = len(excerpt_text.split())
                budget -= token_count
                excerpts.append(
                    TraceRepairExcerpt(
                        source_id=record.source_id,
                        event_id=record.event_id,
                        field=field,
                        text=excerpt_text,
                        matched_terms=matched,
                        redacted=not request.include_raw,
                    )
                )
        repaired_terms = _dedupe(term for excerpt in excerpts for term in excerpt.matched_terms)
        action = _suggested_action(triggers, repaired_terms)
        return TraceRepairPacket(
            triggered=bool(triggers),
            triggers=triggers,
            excerpts=excerpts,
            suggested_action=action,
            token_count=sum(len(excerpt.text.split()) for excerpt in excerpts),
            repaired_terms=repaired_terms,
        )


def _has_premise(payload: dict[str, Any]) -> bool:
    return bool(payload.get("raw_context") or payload.get("chunk_ids") or payload.get("support_units"))


def _normalize_trace_request_aliases(payload: dict[str, Any]) -> None:
    if "query_text" not in payload and "query" in payload:
        payload["query_text"] = payload["query"]
    if "raw_context" not in payload and "context" in payload:
        payload["raw_context"] = payload["context"]
    if "response_text" not in payload and "response" in payload:
        payload["response_text"] = payload["response"]


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


_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d\s().-]{6,}\d)(?!\d)")


def _event_from_trace_payload(payload: dict[str, Any]) -> TraceSessionEvent:
    return TraceSessionEvent(
        event_type="observation",
        content="\n".join(
            str(item)
            for item in [payload.get("query_text") or "", payload.get("response_text") or ""]
            if item
        ),
        query_text=str(payload.get("query_text") or "") or None,
        response_text=str(payload.get("response_text") or "") or None,
        raw_context=str(payload.get("raw_context") or "") or None,
        metadata={"source": "trace_session_score"},
        idempotency_key=str(payload.get("request_id") or "") or None,
    )


def _source_from_event(
    session: TraceSessionState,
    event: TraceSessionEvent,
) -> tuple[TraceSourceRecord, TraceSourcePointer]:
    event_id = str(event.event_id or _new_id("evt"))
    content_hash = _source_hash(event)
    source_id = f"src_{content_hash[:24]}"
    record = TraceSourceRecord(
        source_id=source_id,
        session_id=session.session_id,
        event_id=event_id,
        event_type=event.event_type,
        created_at=event.timestamp or _now_iso(),
        content_hash=content_hash,
        content=event.content,
        raw_context=event.raw_context,
        query_text=event.query_text,
        response_text=event.response_text,
        metadata=dict(event.metadata),
        redacted=session.privacy_mode != "standard",
    )
    pointer = TraceSourcePointer(
        source_id=source_id,
        event_id=event_id,
        content_hash=content_hash,
        redacted=record.redacted,
    )
    return record, pointer


def _source_hash(event: TraceSessionEvent) -> str:
    payload = "|".join(
        str(item or "")
        for item in [
            event.event_id,
            event.event_type,
            event.content,
            event.raw_context,
            event.query_text,
            event.response_text,
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _index_source_terms(session: TraceSessionState, source: TraceSourceRecord, domain: str) -> None:
    text = "\n".join(_source_fields(source).values())
    terms = extract_exact_critical_terms(text, domain=domain)
    existing = list(session.metadata.get("_source_exact_terms", []))
    session.metadata["_source_exact_terms"] = _dedupe([*existing, *terms])[-2048:]


def _attach_source_pointer(memory_state: MemoryState | None, pointer: TraceSourcePointer) -> None:
    if memory_state is None:
        return
    for span in memory_state.spans:
        if span.created_turn == memory_state.turn_index and "source_pointer" not in span.provenance:
            span.provenance["source_pointer"] = pointer.model_dump(mode="json")


def _redacted_source(source: TraceSourceRecord) -> TraceSourceRecord:
    return source.model_copy(
        update={
            "content": _redact_text(source.content),
            "raw_context": _redact_text(source.raw_context or "") if source.raw_context is not None else None,
            "query_text": _redact_text(source.query_text or "") if source.query_text is not None else None,
            "response_text": _redact_text(source.response_text or "") if source.response_text is not None else None,
            "redacted": True,
        }
    )


def _redact_text(text: str) -> str:
    text = _EMAIL_RE.sub("[email]", text)
    return _PHONE_RE.sub("[phone]", text)


def _source_fields(record: TraceSourceRecord) -> dict[str, str]:
    return {
        key: value
        for key, value in {
            "content": record.content,
            "raw_context": record.raw_context or "",
            "query_text": record.query_text or "",
            "response_text": record.response_text or "",
        }.items()
        if value.strip()
    }


def _repair_triggers(
    *,
    session: TraceSessionState,
    request: TraceSessionScoreRequest,
    trace_payload: dict[str, Any],
    hot_context: str,
) -> list[TraceRepairTrigger]:
    triggers: list[TraceRepairTrigger] = []
    band = str(trace_payload.get("risk_band") or "").lower()
    runtime = trace_payload.get("runtime_decision") or {}
    action = str(runtime.get("action") or runtime.get("recommendation") or "").lower() if isinstance(runtime, dict) else ""
    terms = _terms_from_trace_request(request.trace_request, domain=_session_domain(session))
    missing = [term for term in terms if term.lower() not in hot_context.lower()]
    if request.force_original_on_trigger and terms:
        triggers.append(
            TraceRepairTrigger(
                trigger_type="forced_original",
                severity="warn",
                reason="caller requested original-source repair on trigger",
                terms=terms,
            )
        )
    if band in {"red", "amber", "unknown", "unsupported"}:
        triggers.append(
            TraceRepairTrigger(
                trigger_type="risk_band",
                severity="critical" if band == "red" else "warn",
                reason=f"TRACE score returned {band or 'unknown'} risk band",
                terms=terms[:16],
            )
        )
    if action in {"auto_repair", "block", "block_until_regrounded", "ask_tool_again"}:
        triggers.append(
            TraceRepairTrigger(
                trigger_type="runtime_decision",
                severity="critical" if "block" in action else "warn",
                reason=f"runtime decision requested {action}",
                terms=terms[:16],
            )
        )
    if missing and _terms_exist_in_vault(session, missing):
        triggers.append(
            TraceRepairTrigger(
                trigger_type="exact_critical_loss",
                severity="critical",
                reason="exact-critical terms exist in immutable history but are missing from hot context",
                terms=missing[:24],
            )
        )
    return _dedupe_triggers(triggers)


def _terms_exist_in_vault(session: TraceSessionState, terms: list[str]) -> bool:
    # The service-level caller fills this by searching all sources during packet
    # construction. The gate only needs to know whether a term is plausibly
    # repairable, and session metadata keeps this deterministic without raw logs.
    known = {str(term).lower() for term in session.metadata.get("_source_exact_terms", [])}
    return any(term.lower() in known for term in terms)


def _terms_from_trace_request(trace_request: dict[str, Any], *, domain: str) -> list[str]:
    text = "\n".join(
        str(trace_request.get(key) or "")
        for key in ("query_text", "response_text", "raw_context")
        if trace_request.get(key)
    )
    return extract_exact_critical_terms(text, domain=domain)[:64]


def _terms_from_repair_request(
    request: TraceSessionRepairRequest,
    *,
    domain: str | None = None,
) -> list[str]:
    text = "\n".join(item for item in [request.query_text or "", request.response_text or ""] if item)
    return _dedupe([*request.missing_terms, *extract_exact_critical_terms(text, domain=domain)[:64]])


def _trace_text(trace_request: dict[str, Any], key: str) -> str | None:
    value = trace_request.get(key)
    return str(value) if value is not None else None


def _matched_terms(text: str, terms: list[str]) -> list[str]:
    lower = text.lower()
    return [term for term in terms if term and term.lower() in lower]


def _text_overlap(text: str, query: str) -> bool:
    if not query.strip():
        return True
    words = set(re.findall(r"[A-Za-z0-9_./-]{4,}", query.lower()))
    if not words:
        return True
    lower = text.lower()
    return any(word in lower for word in words)


def _bounded_excerpt(text: str, terms: list[str], *, max_tokens: int) -> str:
    words = text.split()
    if not words:
        return ""
    if not terms:
        return " ".join(words[:max_tokens])
    lower = text.lower()
    positions = [lower.find(term.lower()) for term in terms if term and lower.find(term.lower()) >= 0]
    if not positions:
        return " ".join(words[:max_tokens])
    char_pos = min(positions)
    prefix = text[:char_pos]
    start_word = max(0, len(prefix.split()) - max_tokens // 3)
    return " ".join(words[start_word : start_word + max_tokens])


def _suggested_action(triggers: list[TraceRepairTrigger], repaired_terms: list[str]) -> str:
    if any(trigger.severity == "critical" for trigger in triggers):
        return "block_until_regrounded" if repaired_terms else "ask_tool_again"
    if any(trigger.trigger_type == "runtime_decision" for trigger in triggers):
        return "ask_tool_again" if not repaired_terms else "re_score"
    return "re_score" if repaired_terms else "none"


def _dedupe(items) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        value = str(item or "").strip()
        key = value.lower()
        if value and key not in seen:
            seen.add(key)
            output.append(value)
    return output


def _dedupe_triggers(triggers: list[TraceRepairTrigger]) -> list[TraceRepairTrigger]:
    seen: set[tuple[str, str]] = set()
    output: list[TraceRepairTrigger] = []
    for trigger in triggers:
        key = (trigger.trigger_type, trigger.reason)
        if key not in seen:
            seen.add(key)
            output.append(trigger)
    return output


def _session_domain(session: TraceSessionState) -> str:
    if session.kind == "code":
        return "code"
    if session.kind == "rag":
        return "rag"
    return "chat"
