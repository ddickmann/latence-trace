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
from typing import Any
from uuid import UUID

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
from latence_trace_client.models import (
    AttributionMode,
    ComplianceLabelMode,
    ComplianceRedactionMode,
)

logger = logging.getLogger(__name__)


class LatenceComplianceRedactor:
    """Small LangChain-friendly callable for pre-prompt PII redaction.

    Use it in a RunnableLambda or directly before prompt formatting:

        redactor = LatenceComplianceRedactor(client, labels=["email", "person"])
        safe_input = redactor({"question": user_question})
    """

    def __init__(
        self,
        client: LatenceTraceClient,
        *,
        text_key: str = "question",
        output_key: str = "redacted_question",
        labels: list[str] | None = None,
        categories: list[str] | None = None,
        redaction_mode: ComplianceRedactionMode = ComplianceRedactionMode.MASK,
    ) -> None:
        self._client = client
        self._text_key = text_key
        self._output_key = output_key
        self._labels = labels
        self._categories = categories
        self._redaction_mode = redaction_mode

    def __call__(self, inputs: dict[str, Any]) -> dict[str, Any]:
        text = inputs.get(self._text_key)
        if not isinstance(text, str) or not text:
            return inputs
        result = self._client.redact_compliance(
            text=text,
            labels=self._labels,
            categories=self._categories,
            mode=(
                ComplianceLabelMode.CATEGORY
                if self._labels or self._categories
                else ComplianceLabelMode.OPEN
            ),
            redact=True,
            redaction_mode=self._redaction_mode,
            include_original_text=False,
        )
        return {
            **inputs,
            self._output_key: result.redacted_text or text,
            "latence_compliance": result.model_dump(mode="json", exclude_none=True),
        }


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
        self.last_result: dict[str, Any] | None = None
        self._chain_inputs: dict[Any, dict[str, Any]] = {}

    # LangChain BaseCallbackHandler ----------------------------------------

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **_kwargs: Any,
    ) -> Any:
        del serialized, parent_run_id, tags, metadata
        self._chain_inputs[run_id] = dict(inputs)

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: Any,
        _parent_run_id: Any | None = None,
        **_kwargs: Any,
    ) -> None:
        inputs = self._chain_inputs.pop(run_id, {})
        self._score_outputs(inputs, outputs)

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: Any,
        parent_run_id: Any | None = None,
        **_kwargs: Any,
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

    def _score_outputs(self, inputs: dict[str, Any], outputs: dict[str, Any]) -> None:
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
    def _extract_response_text(outputs: dict[str, Any]) -> str:
        for key in ("output", "answer", "text", "result"):
            value = outputs.get(key)
            if isinstance(value, str):
                return value
        return ""

    @staticmethod
    def _coerce_str(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        return str(value)

    @staticmethod
    def _extract_context(value: Any) -> list[str] | None:
        if value is None:
            return None
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            out: list[str] = []
            for item in value:
                if isinstance(item, str):
                    out.append(item)
                elif hasattr(item, "page_content"):
                    out.append(item.page_content)
            return out or None
        return None
