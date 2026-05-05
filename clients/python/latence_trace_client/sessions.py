"""SDK-managed session state for stateless TRACE deployments."""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field


class TraceEvent(BaseModel):
    """Common event shape used by sessions and integration adapters."""

    model_config = ConfigDict(extra="allow")

    event_type: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    idempotency_key: str | None = None


class TraceSessionSnapshot(BaseModel):
    """Portable session snapshot; no server state is required."""

    model_config = ConfigDict(extra="allow")

    session_id: str | None = None
    memory_state: dict[str, Any] | None = None
    events: list[dict[str, Any]] = Field(default_factory=list)
    idempotency_keys: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SessionStorage(Protocol):
    """Minimal storage adapter contract for SDK-managed session state."""

    def load(self, session_id: str) -> TraceSessionSnapshot | None:
        ...

    def save(self, snapshot: TraceSessionSnapshot) -> None:
        ...

    def delete(self, session_id: str) -> None:
        ...


class InMemorySessionStorage:
    """Process-local storage adapter for tests, notebooks, and short jobs."""

    def __init__(self) -> None:
        self._snapshots: MutableMapping[str, TraceSessionSnapshot] = {}

    def load(self, session_id: str) -> TraceSessionSnapshot | None:
        snapshot = self._snapshots.get(session_id)
        return TraceSessionSnapshot.model_validate(snapshot.model_dump()) if snapshot else None

    def save(self, snapshot: TraceSessionSnapshot) -> None:
        if not snapshot.session_id:
            return
        self._snapshots[snapshot.session_id] = TraceSessionSnapshot.model_validate(
            snapshot.model_dump()
        )

    def delete(self, session_id: str) -> None:
        self._snapshots.pop(session_id, None)


class FileSessionStorage:
    """JSON-file storage adapter for durable local SDK sessions."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def load(self, session_id: str) -> TraceSessionSnapshot | None:
        path = self._path(session_id)
        if not path.exists():
            return None
        return TraceSessionSnapshot.model_validate_json(path.read_text(encoding="utf-8"))

    def save(self, snapshot: TraceSessionSnapshot) -> None:
        if not snapshot.session_id:
            return
        path = self._path(snapshot.session_id)
        path.write_text(snapshot.model_dump_json(indent=2), encoding="utf-8")

    def delete(self, session_id: str) -> None:
        self._path(session_id).unlink(missing_ok=True)

    def _path(self, session_id: str) -> Path:
        safe = "".join(ch for ch in session_id if ch.isalnum() or ch in {"-", "_"})
        return self.directory / f"{safe or 'session'}.json"


def new_idempotency_key(prefix: str = "trace") -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def normalize_events(events: Sequence[TraceEvent | Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        event.model_dump(mode="json", exclude_none=True)
        if isinstance(event, TraceEvent)
        else dict(event)
        for event in events
    ]


def canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
