"""LangGraph adapter: score groundedness as a graph node.

LangGraph workflows compose state transitions as a graph of nodes; any
node can read from and write to a shared state dict. This adapter
gives you a ready-made ``score_groundedness_node`` that plugs into an
existing graph without modifying the rest of the flow.

Usage::

    from langgraph.graph import StateGraph
    from latence_trace_client.integrations.langgraph import score_groundedness_node
    from latence_trace_client import LatenceTraceClient

    trace = LatenceTraceClient(api_key=os.environ["LATENCE_API_KEY"])

    graph = StateGraph(dict)
    graph.add_node("generate", generate_answer)
    graph.add_node("score", score_groundedness_node(trace))
    graph.add_conditional_edges(
        "score",
        lambda s: s["trace_band"],
        {"green": END, "amber": "review", "red": "retry"},
    )

Expected state shape (what the node reads):

* ``state["question"]`` - the user question (str)
* ``state["answer"]`` or ``state["response_text"]`` - the model's
  candidate answer (str)
* ``state["raw_context"]`` - retrieved evidence as a string (str)

Written back:

* ``state["trace_band"]`` - ``green`` / ``amber`` / ``red``
* ``state["trace_score"]`` - float groundedness
* ``state["trace_response"]`` - full :class:`GroundednessResponse`
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

from latence_trace_client.client import LatenceTraceClient
from latence_trace_client.errors import LatenceTraceAPIError

logger = logging.getLogger(__name__)


def score_groundedness_node(
    client: LatenceTraceClient,
    *,
    profile: str = "standard",
    fail_band: str = "red",
    question_key: str = "question",
    answer_key: str = "answer",
    context_key: str = "raw_context",
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """Return a LangGraph-compatible node function that scores the answer.

    ``fail_band`` is the band assigned when the TRACE call raises so the
    downstream conditional edge still gets a sane value.  Defaults to
    ``red`` (fail-closed).
    """

    def _node(state: Dict[str, Any]) -> Dict[str, Any]:
        question = state.get(question_key) or ""
        answer = state.get(answer_key) or state.get("response_text") or ""
        raw_context = state.get(context_key) or state.get("context") or ""
        if not answer or not raw_context:
            logger.warning(
                "latence_trace.langgraph.missing_inputs",
                extra={"has_answer": bool(answer), "has_context": bool(raw_context)},
            )
            return {**state, "trace_band": fail_band, "trace_score": 0.0}
        try:
            response = client.score_groundedness(
                query=question,
                response_text=answer,
                raw_context=raw_context,
                profile=profile,
            )
        except LatenceTraceAPIError as exc:
            logger.warning(
                "latence_trace.langgraph.score_error",
                extra={"error": str(exc), "status": exc.status_code},
            )
            return {**state, "trace_band": fail_band, "trace_score": 0.0}
        return {
            **state,
            "trace_band": response.band,
            "trace_score": response.groundedness,
            "trace_response": response,
        }

    return _node


__all__ = ["score_groundedness_node"]
