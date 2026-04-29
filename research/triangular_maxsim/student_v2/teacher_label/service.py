"""HTTP client for the live ``dev_app`` teacher server.

Instead of spinning up a second :class:`GroundednessService` inside this
process (which contends for GPU memory with the already-running
``dev_app`` + vLLM containers), we reuse the live teacher via its
``POST /runsync`` endpoint.

That endpoint is already wired up with:

* ColBERT on vLLM  (port 18001, GPU-resident)
* mDeBERTa NLI on vLLM  (port 18002, GPU-resident)
* Quality profile defaults (atomic claims on, cascades on, etc.)

All the heavy models are warm and pooled, so labeling 15k rows turns
into ~15k cheap HTTP calls with per-request budgets in the 200-500 ms
range on A5000.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import httpx

logger = logging.getLogger("trace.v2.teacher_label.service")

_DEFAULT_ENDPOINT = os.environ.get("LATENCE_TRACE_TEACHER_URL", "http://127.0.0.1:8091")


@dataclass
class TeacherConfig:
    """Knobs for the HTTP-backed teacher client."""

    endpoint: str = _DEFAULT_ENDPOINT
    profile: str = "quality"
    raw_context_chunk_tokens: int = 256
    response_chunk_tokens: int = 256
    evidence_limit: int = 8
    timeout_s: float = 120.0
    retries: int = 2
    verbose: bool = True  # needed for per-token diagnostics

    # Optional: ignored here, but kept for backwards compat with the
    # old in-process wrapper's call sites.
    device: str = "cuda"

    # Kept for unit tests that want to stub out the HTTP client.
    client: Optional[httpx.Client] = field(default=None, repr=False)


class TeacherHttpClient:
    """Thin wrapper over :class:`httpx.Client` with retry + healthcheck."""

    def __init__(self, cfg: TeacherConfig) -> None:
        self.cfg = cfg
        self._client = cfg.client or httpx.Client(timeout=cfg.timeout_s)
        self._run_url = f"{cfg.endpoint.rstrip('/')}/runsync"
        self._health_url = f"{cfg.endpoint.rstrip('/')}/healthz"
        self._check_health()

    def _check_health(self) -> None:
        try:
            resp = self._client.get(self._health_url, timeout=10.0)
            resp.raise_for_status()
        except Exception as exc:
            raise RuntimeError(
                f"teacher healthcheck failed at {self._health_url}: {exc}. "
                f"Is dev_app.py running? (expected on port 8091)"
            ) from exc

    def score(self, row: Mapping[str, Any]) -> dict:
        """POST one groundedness request and return the full-response dict."""

        payload = {
            "input": {
                "response_text": row["response_text"],
                "query_text": row.get("query_text"),
                "raw_context": row["evidence_text"],
                "profile": self.cfg.profile,
                "verbose": self.cfg.verbose,
                "raw_context_chunk_tokens": self.cfg.raw_context_chunk_tokens,
                "response_chunk_tokens": self.cfg.response_chunk_tokens,
                "evidence_limit": self.cfg.evidence_limit,
            }
        }
        last_exc: Exception | None = None
        for attempt in range(1 + max(0, self.cfg.retries)):
            try:
                resp = self._client.post(self._run_url, json=payload)
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:
                last_exc = exc
                if attempt < self.cfg.retries:
                    sleep_s = 1.5 ** attempt
                    logger.debug(
                        "teacher call failed (attempt %d): %s; sleeping %.1fs",
                        attempt + 1,
                        exc,
                        sleep_s,
                    )
                    time.sleep(sleep_s)
                else:
                    raise
        raise RuntimeError(f"exhausted retries: {last_exc}")  # pragma: no cover


def build_teacher(cfg: Optional[TeacherConfig] = None) -> tuple[TeacherHttpClient, TeacherConfig]:
    cfg = cfg or TeacherConfig()
    client = TeacherHttpClient(cfg)
    logger.info(
        "teacher ready: endpoint=%s profile=%s verbose=%s",
        cfg.endpoint,
        cfg.profile,
        cfg.verbose,
    )
    return client, cfg


def score_row(
    client: TeacherHttpClient,
    cfg: TeacherConfig,
    row: Mapping[str, Any],
) -> dict:
    """Score one row and return the decoded JSON body (compact + 'full')."""

    return client.score(row)
