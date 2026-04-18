"""OpenAI ChatCompletion wrapper that scores the assistant turn.

Two helpers:

- :func:`wrap_openai_chat` -- decorate a `client.chat.completions.create`
  call so the assistant message is scored against any user-supplied
  context (passed as a kwarg).
- :func:`score_openai_response` -- explicit one-shot scoring on an
  already-produced response object.

Designed to be a drop-in retrofit: change one import line in the
existing app, get the groundedness band on every assistant turn.
"""

from __future__ import annotations

import logging
from functools import wraps
from typing import Any, Callable, List, Optional, Sequence

try:  # pragma: no cover - extras-only import
    from openai.types.chat import ChatCompletion
except ImportError as exc:  # pragma: no cover - extras-only import
    raise ImportError(
        "openai integration requires the OpenAI extras. Install with: "
        "pip install 'latence-trace-client[openai]'"
    ) from exc

from latence_trace_client.client import LatenceTraceClient
from latence_trace_client.errors import LatenceTraceAPIError
from latence_trace_client.models import AttributionMode, GroundednessResponse

logger = logging.getLogger(__name__)


def score_openai_response(
    completion: ChatCompletion,
    *,
    client: LatenceTraceClient,
    query: Optional[str] = None,
    raw_context: Optional[Sequence[str]] = None,
    attribution_mode: AttributionMode = AttributionMode.CLOSED_BOOK,
) -> Optional[GroundednessResponse]:
    """Score the *first* assistant message in ``completion``.

    Returns ``None`` (and logs a warning) on transport / server errors
    so the wrapping app can keep flowing -- a score outage must never
    break the chat. The latence-trace request_id is attached to the
    log line so the operator can correlate both sides.
    """

    if not completion.choices:
        return None
    response_text = completion.choices[0].message.content or ""
    if not response_text:
        return None
    try:
        return client.score_groundedness(
            query=query,
            response_text=response_text,
            raw_context=list(raw_context) if raw_context else None,
            attribution_mode=attribution_mode,
        )
    except LatenceTraceAPIError as exc:
        logger.warning(
            "latence_trace_openai_score_failed",
            extra={"code": exc.code, "status": exc.status},
        )
        return None


def wrap_openai_chat(
    create_fn: Callable[..., ChatCompletion],
    *,
    client: LatenceTraceClient,
    attribution_mode: AttributionMode = AttributionMode.CLOSED_BOOK,
) -> Callable[..., ChatCompletion]:
    """Wrap an OpenAI ``chat.completions.create`` callable.

    The wrapped callable accepts two extra keyword args:

    - ``latence_trace_query``: the user query (defaults to the last
      "user" message in ``messages``).
    - ``latence_trace_context``: list of raw context strings to score
      the assistant turn against. When omitted we extract any
      "system" messages that look like RAG context (heuristic).

    The score is attached to the returned object as
    ``response.latence_trace`` (Pydantic model). The original wire
    object is returned unchanged so existing call sites keep working.
    """

    @wraps(create_fn)
    def wrapped(*args, **kwargs):
        latence_query = kwargs.pop("latence_trace_query", None)
        latence_context = kwargs.pop("latence_trace_context", None)
        completion = create_fn(*args, **kwargs)

        if latence_query is None:
            latence_query = _extract_query(kwargs.get("messages") or [])
        if latence_context is None:
            latence_context = _extract_context(kwargs.get("messages") or [])

        score = score_openai_response(
            completion,
            client=client,
            query=latence_query,
            raw_context=latence_context,
            attribution_mode=attribution_mode,
        )
        try:
            object.__setattr__(completion, "latence_trace", score)
        except (AttributeError, TypeError):
            # ChatCompletion is a pydantic model; fall back to
            # model_extra so callers can still find the score.
            try:
                completion.model_extra["latence_trace"] = score
            except Exception:  # pragma: no cover - last-resort
                pass
        return completion

    return wrapped


def _extract_query(messages: Sequence[Any]) -> Optional[str]:
    for msg in reversed(messages):
        role = _role(msg)
        if role == "user":
            return _content(msg)
    return None


def _extract_context(messages: Sequence[Any]) -> Optional[List[str]]:
    chunks: List[str] = []
    for msg in messages:
        role = _role(msg)
        if role == "system":
            content = _content(msg)
            if content:
                chunks.append(content)
    return chunks or None


def _role(msg: Any) -> str:
    if hasattr(msg, "role"):
        return getattr(msg, "role") or ""
    if isinstance(msg, dict):
        return msg.get("role") or ""
    return ""


def _content(msg: Any) -> str:
    if hasattr(msg, "content"):
        value = getattr(msg, "content")
    elif isinstance(msg, dict):
        value = msg.get("content")
    else:
        value = None
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: List[str] = []
        for item in value:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(value)
