"""Tiny local FastAPI wrapper around the RunPod handler."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
import uvicorn

from handler import _health_payload, handler as runpod_handler, initialize, shutdown


@asynccontextmanager
async def lifespan(_: FastAPI):
    initialize()
    try:
        yield
    finally:
        shutdown()


app = FastAPI(title="latence-trace RunPod dev wrapper", lifespan=lifespan)


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    return _health_payload()


@app.post("/run")
@app.post("/runsync")
async def run(request: Request) -> dict[str, Any]:
    body = await request.json()
    if isinstance(body, dict) and isinstance(body.get("input"), dict):
        job = body
    else:
        job = {"input": body}
    return await runpod_handler(job)


if __name__ == "__main__":  # pragma: no cover
    uvicorn.run("dev_app:app", host="127.0.0.1", port=8091, log_level="warning")
