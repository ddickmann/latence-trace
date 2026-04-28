"""AutoGen (Microsoft) adapter: reply hook that scores assistant turns.

AutoGen's ``ConversableAgent`` exposes a ``register_reply`` hook that
runs after the agent generates a reply but before it is sent back.
This adapter registers a hook which scores the reply against a
caller-supplied context and sets ``agent.latest_trace`` on success.

Usage::

    from autogen import AssistantAgent
    from latence_trace_client.integrations.autogen import register_trace_hook

    assistant = AssistantAgent("rag-assistant", llm_config={"model": "gpt-4o"})
    register_trace_hook(
        assistant,
        client=LatenceTraceClient(...),
        context_getter=lambda messages: extract_context(messages),
        question_getter=lambda messages: messages[-1]["content"],
    )
"""

from __future__ import annotations

import logging
from typing import Any, Callable, List, Optional

from latence_trace_client.client import LatenceTraceClient
from latence_trace_client.errors import LatenceTraceAPIError

logger = logging.getLogger(__name__)


def register_trace_hook(
    agent: Any,
    *,
    client: LatenceTraceClient,
    context_getter: Callable[[List[dict]], str],
    question_getter: Optional[Callable[[List[dict]], str]] = None,
    profile: str = "standard",
    block_on_red: bool = False,
) -> None:
    """Register a reply hook on an AutoGen ``ConversableAgent``.

    When ``block_on_red`` is True and TRACE returns a red band, the
    hook rewrites the assistant reply to a safe fallback string.
    Otherwise it leaves the reply alone and just records the band on
    ``agent.latest_trace``.
    """

    if not hasattr(agent, "register_reply"):
        raise TypeError(
            "register_trace_hook expects an AutoGen ConversableAgent-like "
            "object with a register_reply method"
        )

    def _score_reply(
        recipient: Any,
        messages: Optional[List[dict]] = None,
        sender: Optional[Any] = None,  # noqa: ARG001
        config: Optional[Any] = None,  # noqa: ARG001
    ) -> tuple[bool, Optional[str]]:
        messages = messages or []
        if not messages:
            return False, None
        assistant_msg = messages[-1].get("content", "") if messages else ""
        if not assistant_msg:
            return False, None
        raw_context = context_getter(messages) or ""
        question = question_getter(messages) if question_getter else None
        try:
            response = client.score_groundedness(
                query=question,
                response_text=assistant_msg,
                raw_context=raw_context,
                profile=profile,
            )
        except LatenceTraceAPIError as exc:
            logger.warning(
                "latence_trace.autogen.score_error",
                extra={"error": str(exc), "status": exc.status_code},
            )
            return False, None
        recipient.latest_trace = response
        if block_on_red and response.band == "red":
            return True, (
                "I don't have enough grounded evidence to answer. "
                "Please provide additional source material."
            )
        return False, None

    agent.register_reply([Any], _score_reply, position=0)


__all__ = ["register_trace_hook"]
