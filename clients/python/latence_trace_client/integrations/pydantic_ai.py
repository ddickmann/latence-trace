"""Pydantic AI adapter: agent result validator.

Pydantic AI exposes result validators that run after the agent
produces its final output. This adapter gives you a one-liner that
rejects ungrounded answers before they reach the caller.

Usage::

    from pydantic_ai import Agent
    from latence_trace_client.integrations.pydantic_ai import trace_result_validator

    agent = Agent("gpt-4o", output_type=str)
    agent.output_validator(trace_result_validator(
        client=LatenceTraceClient(...),
        context_getter=lambda ctx: ctx.deps.get("raw_context", ""),
        question_getter=lambda ctx: ctx.deps.get("question", ""),
    ))
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from latence_trace_client.client import LatenceTraceClient
from latence_trace_client.errors import LatenceTraceAPIError

logger = logging.getLogger(__name__)


def trace_result_validator(
    *,
    client: LatenceTraceClient,
    context_getter: Callable[[Any], str],
    question_getter: Callable[[Any], str] | None = None,
    profile: str = "standard",
    reject_bands: tuple[str, ...] = ("red",),
) -> Callable[[Any, str], str]:
    """Return a Pydantic AI result validator.

    Raises a ``ValueError`` (which Pydantic AI turns into a retry
    signal for the model) when the scored band is in ``reject_bands``.
    Defaults to rejecting only ``red`` so amber still propagates to
    the reviewer queue.
    """

    def _validator(ctx: Any, result: str) -> str:
        raw_context = context_getter(ctx) or ""
        if not raw_context or not result:
            return result
        question = question_getter(ctx) if question_getter else None
        try:
            response = client.score_groundedness(
                query=question,
                response_text=result,
                raw_context=raw_context,
                profile=profile,
            )
        except LatenceTraceAPIError as exc:
            logger.warning(
                "latence_trace.pydantic_ai.score_error",
                extra={"error": str(exc), "status": exc.status_code},
            )
            return result
        ctx.latest_trace = response
        if response.band in reject_bands:
            raise ValueError(
                f"Answer failed groundedness check (band={response.band}, "
                f"score={response.groundedness:.2f}). Retry with more evidence."
            )
        return result

    return _validator


__all__ = ["trace_result_validator"]
