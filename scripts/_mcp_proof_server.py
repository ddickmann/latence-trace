"""Minimal FastAPI app that mounts only the MCP remote router.

Used by scripts/prove_integrations.py to exercise the SSE + streamable-
HTTP transports without booting the full TRACE worker.  The service
factory proxies to the live sidecar at http://127.0.0.1:8092 so the
same GPU-backed scorer is behind every transport.
"""

from __future__ import annotations

import json
import urllib.request

from fastapi import FastAPI

from latence_trace.mcp.remote import _make_router


class _SidecarResp:
    def __init__(self, body: dict) -> None:
        self._body = body

    def model_dump(self, mode: str = "json") -> dict:
        return self._body


class _SidecarService:
    def __init__(self, base: str = "http://127.0.0.1:8092") -> None:
        self._base = base

    def groundedness(self, request):
        raw = getattr(request, "raw_context", None) or ""
        if isinstance(raw, list):
            raw = "\n\n".join(str(x) for x in raw)
        payload = {
            "question": getattr(request, "query", None) or "",
            "response_text": request.response_text,
            "raw_context": raw,
            "profile": "standard",
        }
        req = urllib.request.Request(
            f"{self._base}/v1/score/groundedness",
            data=json.dumps(payload).encode("utf-8"),
            headers={"content-type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return _SidecarResp(json.loads(resp.read().decode("utf-8")))


app = FastAPI(title="latence-trace MCP proof server")
app.include_router(_make_router(lambda: _SidecarService()))


@app.get("/health")
async def health() -> dict:
    return {"ok": True}
