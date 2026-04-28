"""Remote MCP endpoint over HTTP + SSE (Plan C6).

Wraps the existing stdio ``_dispatch`` router so the same JSON-RPC
methods (``initialize``, ``tools/list``, ``tools/call``, ``ping``)
are available at ``https://api.latence.ai/mcp``.

Two transports are supported simultaneously:

* **streamable-HTTP** - the client POSTs a JSON-RPC request and gets
  a single JSON response back.  Mirrors the stdio round-trip shape
  and is what Cursor + custom agent frameworks use today.
* **SSE** - the client ``GET``s ``/mcp/sse`` with a ``Last-Event-ID``
  header, keeps the connection open, and POSTs requests to
  ``/mcp/messages``.  The response is written as a server-sent event
  on the SSE stream.  Supports concurrent requests over a single
  connection (Claude Desktop uses this shape).

Tenant identity is expected from the Cloudflare gateway as
``X-Latence-Tenant-Id``; the gateway authenticated the API key before
routing here (see ``gateway/cloudflare/``).
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncIterator
from typing import Any, Callable

try:  # pragma: no cover - FastAPI is a soft dep for the remote server
    from fastapi import APIRouter, HTTPException, Request
    from fastapi.responses import JSONResponse, StreamingResponse
except ImportError:  # pragma: no cover
    APIRouter = None  # type: ignore[assignment]
    HTTPException = Exception  # type: ignore[assignment]
    Request = object  # type: ignore[assignment]
    JSONResponse = object  # type: ignore[assignment]
    StreamingResponse = object  # type: ignore[assignment]

from latence_trace.mcp.server import _default_service_factory, _dispatch

logger = logging.getLogger(__name__)


def _make_router(
    service_factory: Callable[[], Any] | None = None,
) -> Any:
    if APIRouter is None:
        raise RuntimeError(
            "fastapi is required for the remote MCP endpoint; install with "
            "'pip install latence-trace[server]'"
        )
    router = APIRouter(prefix="/mcp")
    factory = service_factory or _default_service_factory

    # Per-session SSE outbound queues keyed by session id.  A single
    # queue can interleave responses from concurrent requests.
    sessions: dict[str, asyncio.Queue[str]] = {}

    async def _call_sync(message: dict[str, Any]) -> dict[str, Any] | None:
        """Run the synchronous dispatch in a worker thread."""

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _dispatch, message, factory)

    @router.post("")
    @router.post("/")
    async def streamable_http(request: Request) -> Any:
        """Streamable-HTTP transport: one request, one JSON response."""

        try:
            message = await request.json()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"invalid json: {exc}")
        tenant_id = request.headers.get("x-latence-tenant-id") or "anonymous"
        logger.info(
            "mcp.streamable_http",
            extra={"tenant_id": tenant_id, "method": message.get("method")},
        )
        response = await _call_sync(message)
        if response is None:
            # Notification — no response body per JSON-RPC.
            return JSONResponse(status_code=204, content=None)
        return JSONResponse(content=response)

    @router.get("/sse")
    async def sse(request: Request) -> Any:
        """SSE transport: long-lived stream; responses push here."""

        session_id = request.headers.get("x-latence-session-id") or str(uuid.uuid4())
        queue: asyncio.Queue[str] = asyncio.Queue()
        sessions[session_id] = queue

        async def _event_source() -> AsyncIterator[bytes]:
            try:
                # Announce the session id so the client posts subsequent
                # messages with the right X-Latence-Session-Id.
                yield f"event: session\ndata: {session_id}\n\n".encode("utf-8")
                while True:
                    try:
                        payload = await asyncio.wait_for(queue.get(), timeout=15.0)
                    except asyncio.TimeoutError:
                        # Heartbeat every 15s so proxies don't kill idle
                        # connections.
                        yield b": ping\n\n"
                        continue
                    yield f"data: {payload}\n\n".encode("utf-8")
            finally:
                sessions.pop(session_id, None)

        return StreamingResponse(
            _event_source(),
            media_type="text/event-stream",
            headers={
                "cache-control": "no-cache",
                "connection": "keep-alive",
                "x-accel-buffering": "no",
            },
        )

    @router.post("/messages")
    async def sse_post(request: Request) -> Any:
        """POST a JSON-RPC request; response arrives on the SSE stream."""

        session_id = request.headers.get("x-latence-session-id")
        if not session_id or session_id not in sessions:
            raise HTTPException(
                status_code=400,
                detail="missing or unknown session; open GET /mcp/sse first",
            )
        try:
            message = await request.json()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"invalid json: {exc}")
        response = await _call_sync(message)
        queue = sessions.get(session_id)
        if response is not None and queue is not None:
            await queue.put(json.dumps(response))
        return JSONResponse(status_code=202, content={"accepted": True})

    @router.get("/health")
    async def health() -> Any:
        return JSONResponse(
            content={
                "status": "ok",
                "protocol_version": "2025-03-26",
                "server": "latence-trace",
                "active_sessions": len(sessions),
            }
        )

    return router


__all__ = ["_make_router"]
