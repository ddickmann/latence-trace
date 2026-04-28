"""Haystack 2.x adapter: scoring component for a RAG pipeline.

Drops into a Haystack ``Pipeline`` right after the ``Generator`` so
every generated answer is scored against the retrieved evidence.

Usage::

    from haystack import Pipeline
    from latence_trace_client.integrations.haystack import LatenceTraceScorer

    pipe = Pipeline()
    pipe.add_component("retriever", retriever)
    pipe.add_component("prompt_builder", prompt_builder)
    pipe.add_component("llm", generator)
    pipe.add_component("scorer", LatenceTraceScorer(client=trace_client))

    pipe.connect("retriever", "prompt_builder.documents")
    pipe.connect("prompt_builder", "llm.prompt")
    pipe.connect("llm.replies", "scorer.responses")
    pipe.connect("retriever.documents", "scorer.documents")
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

try:  # pragma: no cover - extras-only import
    from haystack import Document, component
except ImportError as exc:  # pragma: no cover - extras-only import
    raise ImportError(
        "haystack integration requires Haystack 2.x. Install with: "
        "pip install 'latence-trace-client[haystack]'"
    ) from exc

from latence_trace_client.client import LatenceTraceClient
from latence_trace_client.errors import LatenceTraceAPIError

logger = logging.getLogger(__name__)


@component
class LatenceTraceScorer:
    """Haystack component that scores LLM replies against documents."""

    def __init__(
        self,
        *,
        client: LatenceTraceClient,
        profile: str = "standard",
    ) -> None:
        self._client = client
        self._profile = profile

    @component.output_types(scored=List[Dict[str, Any]])
    def run(
        self,
        responses: List[str],
        documents: Optional[List[Document]] = None,
        question: Optional[str] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        raw_context = "\n\n".join(
            (doc.content or "") for doc in (documents or []) if doc.content
        )
        scored: List[Dict[str, Any]] = []
        for response_text in responses:
            payload: Dict[str, Any] = {"response_text": response_text}
            if not response_text or not raw_context:
                payload.update({"band": "red", "groundedness": 0.0, "reason": "empty_input"})
                scored.append(payload)
                continue
            try:
                res = self._client.score_groundedness(
                    query=question,
                    response_text=response_text,
                    raw_context=raw_context,
                    profile=self._profile,
                )
            except LatenceTraceAPIError as exc:
                logger.warning(
                    "latence_trace.haystack.score_error",
                    extra={"error": str(exc), "status": exc.status_code},
                )
                payload.update({"band": "amber", "groundedness": None, "error": str(exc)})
                scored.append(payload)
                continue
            payload.update(
                {
                    "band": res.band,
                    "groundedness": res.groundedness,
                    "response": res.model_dump() if hasattr(res, "model_dump") else dict(res),
                }
            )
            scored.append(payload)
        return {"scored": scored}


__all__ = ["LatenceTraceScorer"]
