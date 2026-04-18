"""Stdio MCP adapter for latence-trace (PA6).

The Model Context Protocol uses newline-delimited JSON-RPC 2.0 over
stdio so AI agent runtimes (Cursor, Claude Desktop, the OpenAI Agents
SDK, custom orchestrators) can register external tools without booting
a network server. This adapter exposes the FastAPI ``/groundedness``
endpoint as a single ``score_groundedness`` MCP tool plus the canonical
``initialize``, ``tools/list`` and ``tools/call`` handshake methods so
any compliant client can discover the tool, send a payload, and read
the calibrated response.

The implementation is deliberately dependency-free (stdlib only) so it
can run inside locked-down agent sandboxes that forbid additional pip
installs. We never touch the network from this module - the
groundedness call is in-process via :class:`GroundednessService`.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


_PROTOCOL_VERSION = "2025-03-26"
_SERVER_NAME = "latence-trace"
_SERVER_VERSION = "1.0.0"

# JSON-RPC 2.0 error codes (https://www.jsonrpc.org/specification#error_object).
_JSONRPC_PARSE_ERROR = -32700
_JSONRPC_INVALID_REQUEST = -32600
_JSONRPC_METHOD_NOT_FOUND = -32601
_JSONRPC_INVALID_PARAMS = -32602
_JSONRPC_INTERNAL_ERROR = -32603


def _success(req_id: Any, result: Dict[str, Any]) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _error(req_id: Any, code: int, message: str, data: Any = None) -> Dict[str, Any]:
    err: Dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": req_id, "error": err}


def _tool_descriptor() -> Dict[str, Any]:
    """Tool schema served on ``tools/list``.

    The JSON Schema mirrors :class:`GroundednessRequest` so MCP clients
    can validate before sending. We surface only the most useful fields
    here; the full superset is still accepted at the service layer
    because we forward the entire payload to ``GroundednessRequest``.
    """

    return {
        "name": "score_groundedness",
        "description": (
            "Score a generated response for groundedness against caller-supplied "
            "evidence. Returns calibrated probabilities, a risk band, NLI claim "
            "verdicts, and a token-level heatmap. Closed-book by default - empty "
            "premises return risk_band='unknown' instead of inventing a score."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query_text": {
                    "type": "string",
                    "description": "Optional user query that produced the response (required when primary_metric=triangular).",
                },
                "response_text": {
                    "type": "string",
                    "description": "The model-generated text to be scored for groundedness.",
                },
                "raw_context": {
                    "type": "string",
                    "description": "Premise text. Mutually exclusive with chunk_ids and support_units.",
                },
                "chunk_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Stable chunk identifiers resolved by the caller-provided chunk resolver.",
                },
                "support_units": {
                    "type": "array",
                    "description": "Structured premise units with per-unit attribution (source_id, speaker, timestamp, metadata).",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "source_id": {"type": "string"},
                            "speaker": {"type": "string"},
                            "timestamp": {"type": "string"},
                            "metadata": {"type": "object"},
                        },
                        "required": ["text"],
                    },
                },
                "attribution_mode": {
                    "type": "string",
                    "enum": ["closed_book", "open_domain"],
                    "description": "Evidence policy. 'closed_book' (default) refuses to score zero-evidence inputs.",
                },
                "primary_metric": {
                    "type": "string",
                    "enum": ["reverse_context", "triangular"],
                },
                "coverage_threshold": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": (
                        "Threshold on per-support-unit reverse-context similarity "
                        "used to flag a unit as 'used' in the retrieval-efficiency "
                        "observability signal. Default 0.5. Response carries "
                        "scores.context_coverage_ratio and per-unit coverage_score "
                        "/ used so callers can identify dead-weight retrieval."
                    ),
                },
            },
            "required": ["response_text"],
        },
    }


def _handle_initialize(req_id: Any, _params: Dict[str, Any]) -> Dict[str, Any]:
    return _success(
        req_id,
        {
            "protocolVersion": _PROTOCOL_VERSION,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": _SERVER_NAME, "version": _SERVER_VERSION},
        },
    )


def _handle_tools_list(req_id: Any, _params: Dict[str, Any]) -> Dict[str, Any]:
    return _success(req_id, {"tools": [_tool_descriptor()]})


def _handle_tools_call(
    req_id: Any,
    params: Dict[str, Any],
    service_factory: Callable[[], Any],
) -> Dict[str, Any]:
    name = (params or {}).get("name")
    if name != "score_groundedness":
        # Per JSON-RPC 2.0, "Method not found" is -32601. The MCP spec
        # treats the tool name as a parameter to ``tools/call``, not a
        # JSON-RPC method, so an unknown tool is "invalid params"
        # (-32602). Clients distinguish "the tool I asked for does not
        # exist" from "the dispatcher does not know about tools/call".
        return _error(req_id, _JSONRPC_INVALID_PARAMS, f"unknown tool: {name}")

    arguments = (params or {}).get("arguments") or {}
    try:
        from latence_trace.api.models import GroundednessRequest  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - import guard
        return _error(
            req_id,
            _JSONRPC_INTERNAL_ERROR,
            f"failed to import request schema: {exc}",
        )

    try:
        request = GroundednessRequest(**arguments)
    except Exception as exc:
        return _error(
            req_id,
            _JSONRPC_INVALID_PARAMS,
            "invalid arguments for score_groundedness",
            {"validation_error": str(exc)},
        )

    try:
        service = service_factory()
        response = service.groundedness(request)
        body = response.model_dump(mode="json")
    except Exception as exc:
        return _error(
            req_id,
            _JSONRPC_INTERNAL_ERROR,
            f"score_groundedness failed: {exc}",
        )

    return _success(
        req_id,
        {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(body, indent=2, default=str),
                }
            ],
            "structuredContent": body,
            "isError": False,
        },
    )


# Module-level singleton + lock so repeat tool calls reuse the same
# warmed encoder / NLI / reranker state. The previous per-call factory
# was correct but threw away the model caches every invocation, which
# defeated PA1 / PA2 / PA3 entirely for MCP clients (Cursor / Claude
# Desktop) that issue many short turns in sequence.
_DEFAULT_SERVICE_HOLDER: Dict[str, Any] = {"service": None}
_DEFAULT_SERVICE_LOCK = threading.Lock()


def _default_service_factory():
    cached = _DEFAULT_SERVICE_HOLDER.get("service")
    if cached is not None:
        return cached
    with _DEFAULT_SERVICE_LOCK:
        cached = _DEFAULT_SERVICE_HOLDER.get("service")
        if cached is not None:
            return cached
        # Lazy: the service pulls in pylate / vLLM clients which we do
        # not want to import unless a tool call actually arrives.
        from latence_trace.api.service import GroundednessService  # noqa: PLC0415

        service = GroundednessService(
            device=os.environ.get("LATENCE_TRACE_DEVICE", "cpu"),
        )
        _DEFAULT_SERVICE_HOLDER["service"] = service
        return service


def reset_default_service_factory_for_tests() -> None:
    """Test hook: drop the cached default ``GroundednessService``.

    Production code never calls this. The MCP test suite uses it to
    reset between test cases that swap the service factory.
    """

    with _DEFAULT_SERVICE_LOCK:
        _DEFAULT_SERVICE_HOLDER["service"] = None


def _dispatch(
    message: Dict[str, Any],
    service_factory: Callable[[], Any],
) -> Optional[Dict[str, Any]]:
    method = message.get("method")
    req_id = message.get("id")
    params = message.get("params") or {}

    if method == "initialize":
        return _handle_initialize(req_id, params)
    if method == "initialized":
        # Notification: clients send this after initialize, no reply
        # expected by the spec.
        return None
    if method == "tools/list":
        return _handle_tools_list(req_id, params)
    if method == "tools/call":
        return _handle_tools_call(req_id, params, service_factory)
    if method in {"ping"}:
        return _success(req_id, {})
    if req_id is None:
        # Unknown notification - per JSON-RPC, do not respond.
        return None
    return _error(req_id, _JSONRPC_METHOD_NOT_FOUND, f"method not found: {method}")


def run_stdio_loop(
    *,
    service_factory: Optional[Callable[[], Any]] = None,
    stdin=None,
    stdout=None,
) -> int:
    """Run the MCP stdio loop until EOF.

    ``service_factory`` defaults to a lazy :class:`GroundednessService`
    builder so import cost only kicks in when the first tool call
    arrives. Tests pass an in-process stub.
    """

    factory = service_factory or _default_service_factory
    in_stream = stdin or sys.stdin
    out_stream = stdout or sys.stdout

    while True:
        line = in_stream.readline()
        if not line:
            return 0
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError as exc:
            err = _error(None, _JSONRPC_PARSE_ERROR, f"parse error: {exc}")
            out_stream.write(json.dumps(err) + "\n")
            out_stream.flush()
            continue

        try:
            response = _dispatch(message, factory)
        except Exception as exc:  # pragma: no cover - last-resort guard
            response = _error(
                message.get("id"),
                _JSONRPC_INTERNAL_ERROR,
                f"internal error: {exc}",
            )

        if response is not None:
            out_stream.write(json.dumps(response, default=str) + "\n")
            out_stream.flush()
