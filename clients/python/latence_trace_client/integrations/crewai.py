"""CrewAI adapter: score the output of an agent task.

CrewAI models work as a ``Crew`` of ``Agent``s executing ``Task``s.
Every task has a ``callback`` hook that receives the final output.
This adapter wraps that hook so the output is scored against any
context the task carried.

Usage::

    from crewai import Agent, Task, Crew
    from latence_trace_client.integrations.crewai import LatenceTraceCallback

    trace = LatenceTraceCallback(
        client=LatenceTraceClient(...),
        context_getter=lambda task: task.context or "",
        question_getter=lambda task: task.description,
    )

    research_task = Task(
        description="Find 2023 ARR in the shareholder letter.",
        expected_output="A single sentence with the figure.",
        agent=analyst,
        callback=trace,
    )
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from latence_trace_client.client import LatenceTraceClient
from latence_trace_client.errors import LatenceTraceAPIError
from latence_trace_client.integrations import _band_utils

logger = logging.getLogger(__name__)


class LatenceTraceCallback:
    """Callable CrewAI task-callback that scores each task output.

    The CrewAI callback signature is ``Callable[[TaskOutput], Any]``.
    We intentionally stay duck-typed so this class works across
    several CrewAI versions.
    """

    def __init__(
        self,
        *,
        client: LatenceTraceClient,
        context_getter: Callable[[Any], str],
        question_getter: Callable[[Any], str] | None = None,
        profile: str = "standard",
        on_band: Callable[[str, Any, Any], None] | None = None,
    ) -> None:
        self._client = client
        self._context_getter = context_getter
        self._question_getter = question_getter
        self._profile = profile
        self._on_band = on_band

    def __call__(self, task_output: Any) -> Any:
        response_text = getattr(task_output, "raw", None) or str(task_output)
        task = getattr(task_output, "task", None) or task_output
        question = self._question_getter(task) if self._question_getter else None
        raw_context = self._context_getter(task) or ""
        if not raw_context:
            logger.debug("latence_trace.crewai.no_context")
            return task_output
        try:
            response = self._client.score_groundedness(
                query=question,
                response_text=response_text,
                raw_context=[raw_context] if isinstance(raw_context, str) else list(raw_context),
                extra={"profile": self._profile} if self._profile else None,
            )
        except LatenceTraceAPIError as exc:
            logger.warning(
                "latence_trace.crewai.score_error",
                extra={"error": str(exc), "status": exc.status_code},
            )
            return task_output
        # Stash on the task_output if mutable; callers can inspect
        # .trace_band / .trace_score downstream.
        try:
            task_output.trace_band = _band_utils.resolve_band(response)
            task_output.trace_score = _band_utils.resolve_score(response)
            task_output.trace_response = response
        except Exception:  # pragma: no cover - frozen models
            pass
        if self._on_band:
            self._on_band(_band_utils.resolve_band(response), task_output, response)
        return task_output


__all__ = ["LatenceTraceCallback"]
