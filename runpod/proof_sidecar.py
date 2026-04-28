"""Translator sidecar: /groundedness -> dev_app /runsync.

The RunPod dev_app exposes /runsync in the
``{"input": {"action": "score", ...}}`` shape.  The Python /
TypeScript / LangChain / LlamaIndex clients all expect the
production FastAPI shape at /groundedness (flat request body,
response wrapped in the agent-friendly envelope).

This sidecar bridges the two for local integration testing so every
adapter can be verified against a single endpoint without rebuilding
the production API.  It is NOT used in production - the production
path goes Cloudflare worker -> RunPod handler directly.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, Request

app = FastAPI(title="latence-trace integration proof sidecar")

UPSTREAM = "http://127.0.0.1:8091/runsync"


@app.get("/v1/health")
@app.get("/healthz")
async def health() -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=5.0) as client:
        resp = await client.get("http://127.0.0.1:8091/healthz")
    return resp.json()


def _reshape_response(upstream: dict[str, Any]) -> dict[str, Any]:
    inner = upstream.get("output") or upstream
    band = str(inner.get("band") or "amber").lower()
    score = inner.get("score") or inner.get("groundedness") or 0.0
    profile = inner.get("profile") or inner.get("effective_profile") or "standard"
    return {
        "band": band,
        "groundedness": float(score) if score is not None else 0.0,
        "groundedness_v2": float(score) if score is not None else 0.0,
        "risk_band": band,
        "profile": profile,
        "effective_profile": profile,
        "nli_aggregate": inner.get("nli_aggregate"),
        "context_coverage_ratio": inner.get("context_coverage_ratio"),
        "context_usage_ratio": inner.get("context_usage_ratio"),
        "support_units": inner.get("support_units", []),
        "version": inner.get("version"),
        "scoring_mode": inner.get("scoring_mode"),
        "request_id": upstream.get("request_id") or inner.get("request_id"),
        "raw": inner,
    }


async def _call_upstream(payload: dict[str, Any]) -> dict[str, Any]:
    body = {"input": {"action": "score", **payload}}
    async with httpx.AsyncClient(timeout=90.0) as client:
        resp = await client.post(UPSTREAM, json=body)
    resp.raise_for_status()
    return resp.json()


@app.post("/groundedness")
@app.post("/v1/score/groundedness")
async def groundedness(request: Request) -> dict[str, Any]:
    body = await request.json()
    payload = {
        "question": body.get("query") or body.get("question") or "",
        "response_text": body.get("response_text") or body.get("response") or "",
        "raw_context": _coerce_context(body.get("raw_context") or body.get("context") or ""),
        "profile": body.get("profile") or "standard",
    }
    upstream = await _call_upstream(payload)
    return _reshape_response(upstream)


def _coerce_context(val: Any) -> str:
    if isinstance(val, str):
        return val
    if isinstance(val, list):
        parts: list[str] = []
        for item in val:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(json.dumps(item, ensure_ascii=False))
        return "\n\n".join(parts)
    return str(val)


if __name__ == "__main__":
    uvicorn.run("runpod.proof_sidecar:app", host="127.0.0.1", port=8092, log_level="warning")
