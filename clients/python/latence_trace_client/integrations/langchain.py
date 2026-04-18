"""LangChain callback that scores every chain output for groundedness.

Usage::

    from langchain_core.runnables import RunnableConfig
    from latence_trace_client import LatenceTraceClient
    from latence_trace_client.integrations.langchain import LatenceTraceCallback

    client = LatenceTraceClient(base_url="...", api_key="...")
    callback = LatenceTraceCallback(client)
    chain.invoke({"question": "..."}, config={"callbacks": [callback]})

The callback expects the chain ``inputs`` to contain the user query
under ``question`` (configurable) and the retrieved context under
``context`` -- the same shape ``langchain`` retrievers + prompt
templates produce. The score and risk band are attached to the chain
output under ``metadata.latence_trace`` so the application can
inspect / log it.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

try:  # pragma: no cover - extras-only import
    from langchain_core.callbacks.base import BaseCallbackHandler
    from langchain_core.outputs import LLMResult
except ImportError as exc:  # pragma: no cover - extras-only import
    raise ImportError(
        "langchain integration requires the LangChain extras. Install with: "
        "pip install 'latence-trace-client[langchain]'"
    ) from exc

from latence_trace_client.client import LatenceTraceClient
from latence_trace_client.errors import LatenceTraceAPIError
from latence_trace_client.models import AttributionMode

logger = logging.getLogger(__name__)


class LatenceTraceCallback(BaseCallbackHandler):
    """Score every LLM output produced by the chain.

    Per the LangChain callback contract the handler is *fire-and-forget*
    on errors -- a score timeout never breaks the user's chain. The
    last-seen score is exposed via :attr:`last_result`.
    """

    def __init__(
        self,
        client: LatenceTraceClient,
        *,
        question_key: str = "question",
        context_key: str = "context",
        attribution_mode: AttributionMode = AttributionMode.CLOSED_BOOK,
    ) -> None:
        self._client = client
        self._question_key = question_key
        self._context_key = context_key
        self._attribution_mode = attribution_mode
        self.last_result: Optional[Dict[str, Any]] = None
        self._chain_inputs: Dict[Any, Dict[str, Any]] = {}

    # LangChain BaseCallbackHandler ----------------------------------------

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: Any,
        **kwargs: Any,
    ) -> None:
        self._chain_inputs[run_id] = dict(inputs)

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: Any,
        parent_run_id: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        inputs = self._chain_inputs.pop(run_id, {})
        self._score_outputs(inputs, outputs)

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: Any,
        parent_run_id: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        inputs = self._chain_inputs.get(parent_run_id) or self._chain_inputs.get(run_id) or {}
        text = ""
        for gen_list in response.generations:
            for gen in gen_list:
                text = gen.text
                break
            if text:
                break
        if not text:
            return
        self._score_outputs(inputs, {"output": text})

    # ----------------------------------------------------------------------

    def _score_outputs(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        response_text = self._extract_response_text(outputs)
        if not response_text:
            return
        query = self._coerce_str(inputs.get(self._question_key))
        context = self._extract_context(inputs.get(self._context_key))
        if not context and self._attribution_mode == AttributionMode.CLOSED_BOOK:
            # Closed-book + no premise = degenerate score; skip rather
            # than send a guaranteed unknown.
            return
        try:
            result = self._client.score_groundedness(
                query=query,
                response_text=response_text,
                raw_context=context,
                attribution_mode=self._attribution_mode,
            )
        except LatenceTraceAPIError as exc:
            logger.warning(
                "latence_trace_score_failed",
                extra={"code": exc.code, "status": exc.status},
            )
            return
        payload = {
            "risk_band": result.risk_band.value,
            "groundedness_v2": result.scores.groundedness_v2,
            "coverage_score_u": result.scores.coverage_score_u,
            "request_id": result.request_id,
        }
        self.last_result = payload
        outputs.setdefault("metadata", {})
        if isinstance(outputs["metadata"], dict):
            outputs["metadata"].setdefault("latence_trace", payload)

    @staticmethod
    def _extract_response_text(outputs: Dict[str, Any]) -> str:
        for key in ("output", "answer", "text", "result"):
            value = outputs.get(key)
            if isinstance(value, str):
                return value
        return ""

    @staticmethod
    def _coerce_str(value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        return str(value)

    @staticmethod
    def _extract_context(value: Any) -> Optional[List[str]]:
        if value is None:
            return None
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            out: List[str] = []
            for item in value:
                if isinstance(item, str):
                    out.append(item)
                elif hasattr(item, "page_content"):
                    out.append(getattr(item, "page_content"))
            return out or None
        return None
