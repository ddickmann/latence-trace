"""Synchronous SDK client.

Wraps an ``httpx.Client`` with the SDK's retry policy and typed
models. The class is reusable across many requests; use a single
instance per process whenever possible (httpx pools connections).
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

import httpx
from pydantic import BaseModel, ValidationError

from latence_trace_client._transport import (
    DEFAULT_TIMEOUT_SECONDS,
    RETRYABLE_STATUS,
    RetryPolicy,
    coerce_api_key,
    coerce_base_url,
    decode_error,
    default_headers,
    parse_retry_after,
)
from latence_trace_client.errors import (
    LatenceTraceAPIError,
    LatenceTraceRateLimited,
    LatenceTraceServerError,
    LatenceTraceTimeout,
    LatenceTraceValidationError,
)
from latence_trace_client.models import (
    AttributionMode,
    ComplianceCustomLabel,
    ComplianceLabelMode,
    ComplianceRedactionMode,
    ComplianceRedactionRequest,
    ComplianceRedactionResponse,
    GroundednessRequest,
    GroundednessResponse,
    SupportUnit,
)

PremiseSupportUnits = Sequence[SupportUnit | Mapping[str, Any]]
ComplianceCustomLabels = Sequence[ComplianceCustomLabel | Mapping[str, Any]]


class PrivacyClient:
    def __init__(self, owner: LatenceTraceClient) -> None:
        self._owner = owner

    def redact(self, **kwargs: Any) -> ComplianceRedactionResponse:
        return self._owner.redact_compliance(**kwargs)


class GroundingClient:
    def __init__(self, owner: LatenceTraceClient) -> None:
        self._owner = owner

    def rag(self, **kwargs: Any) -> GroundednessResponse:
        extra = dict(kwargs.pop("extra", {}) or {})
        extra.setdefault("scoring_mode", "rag")
        return self._owner.score_groundedness(extra=extra, **kwargs)

    def code(self, **kwargs: Any) -> GroundednessResponse:
        extra = dict(kwargs.pop("extra", {}) or {})
        extra.setdefault("scoring_mode", "code")
        return self._owner.score_groundedness(extra=extra, **kwargs)


class CompressionClient:
    def __init__(self, owner: LatenceTraceClient) -> None:
        self._owner = owner

    def text(self, text: str, **options: Any) -> Mapping[str, Any]:
        return self._owner._request(
            "POST",
            "/v1/compression",
            json={"text": text, **options},
            expected_model=None,
        )

    def messages(
        self,
        messages: Sequence[Mapping[str, Any]],
        **options: Any,
    ) -> Mapping[str, Any]:
        return self._owner._request(
            "POST",
            "/v1/compression",
            json={"action": "compress_messages", "messages": list(messages), **options},
            expected_model=None,
        )


class MemoryClient:
    def __init__(self, owner: LatenceTraceClient) -> None:
        self._owner = owner

    def step(
        self,
        *,
        prior_memory_state: Mapping[str, Any] | None = None,
        **payload: Any,
    ) -> Mapping[str, Any]:
        body = dict(payload)
        body["prior_memory_state"] = prior_memory_state
        return self._owner._request(
            "POST",
            "/v1/memory/update",
            json=body,
            expected_model=None,
        )


class TraceSession:
    """SDK-managed state facade for stateless TRACE deployments."""

    def __init__(
        self,
        owner: LatenceTraceClient,
        *,
        session_id: str | None = None,
        memory_state: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self._owner = owner
        self.session_id = session_id
        self.memory_state = dict(memory_state) if memory_state else None
        self.metadata = dict(metadata or {})
        self.events: list[Mapping[str, Any]] = []

    def event(self, event_type: str, content: str, **metadata: Any) -> Mapping[str, Any]:
        event = {
            "event_type": event_type,
            "content": content,
            "metadata": metadata,
        }
        self.events.append(event)
        return event

    def memory_step(self, **payload: Any) -> Mapping[str, Any]:
        result = self._owner.memory.step(
            prior_memory_state=self.memory_state,
            **payload,
        )
        next_state = result.get("next_memory_state")
        if isinstance(next_state, Mapping):
            self.memory_state = dict(next_state)
        return result

    def rag(self, **kwargs: Any) -> GroundednessResponse:
        extra = dict(kwargs.pop("extra", {}) or {})
        if self.session_id:
            extra.setdefault("session_id", self.session_id)
        if self.memory_state:
            extra.setdefault("memory_state", self.memory_state)
        return self._owner.grounding.rag(extra=extra, **kwargs)

    def code(self, **kwargs: Any) -> GroundednessResponse:
        extra = dict(kwargs.pop("extra", {}) or {})
        if self.session_id:
            extra.setdefault("session_id", self.session_id)
        return self._owner.grounding.code(extra=extra, **kwargs)


class LatenceTraceClient:
    """Sync client. ``with LatenceTraceClient(...) as c:`` closes the pool."""

    def __init__(
        self,
        base_url: str | None = None,
        *,
        api_key: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        retry_policy: RetryPolicy | None = None,
        transport: httpx.BaseTransport | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self._base_url = coerce_base_url(base_url)
        self._api_key = coerce_api_key(api_key)
        self._retry = retry_policy or RetryPolicy()
        self._headers = default_headers(self._api_key, headers)
        self._client = httpx.Client(
            base_url=self._base_url,
            timeout=timeout,
            transport=transport,
            headers=self._headers,
        )
        self.privacy = PrivacyClient(self)
        self.grounding = GroundingClient(self)
        self.compression = CompressionClient(self)
        self.memory = MemoryClient(self)

    # context manager support ---------------------------------------------

    def __enter__(self) -> LatenceTraceClient:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    # --- public surface --------------------------------------------------

    def health(self) -> Mapping[str, Any]:
        return self._request("GET", "/healthz", expected_model=None)

    def ready(self) -> Mapping[str, Any]:
        return self._request("GET", "/readyz", expected_model=None)

    def agent_help(self) -> Mapping[str, Any]:
        return self._request("GET", "/agent-help", expected_model=None)

    def score_groundedness(
        self,
        *,
        response_text: str,
        query: str | None = None,
        chunk_ids: Sequence[str] | None = None,
        raw_context: Sequence[str] | None = None,
        support_units: PremiseSupportUnits | None = None,
        attribution_mode: AttributionMode = AttributionMode.CLOSED_BOOK,
        primary_metric: str | None = None,
        coverage_threshold: float | None = None,
        chunk_token_budget: int | None = None,
        chunk_token_overlap: int | None = None,
        locale: str | None = None,
        runtime_head_features: Mapping[str, float] | None = None,
        trajectory_features: Mapping[str, float] | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> GroundednessResponse:
        """Score a response for groundedness against the supplied evidence.

        At least one of ``chunk_ids``, ``raw_context``, or
        ``support_units`` must be provided unless the deployment is
        configured for open-domain attribution. The server returns a
        calibrated risk band (`green`/`amber`/`red`/`unknown`) along
        with the per-token signals used to draw a heatmap.
        """

        payload = self._build_payload(
            response_text=response_text,
            query=query,
            chunk_ids=chunk_ids,
            raw_context=raw_context,
            support_units=support_units,
            attribution_mode=attribution_mode,
            primary_metric=primary_metric,
            coverage_threshold=coverage_threshold,
            chunk_token_budget=chunk_token_budget,
            chunk_token_overlap=chunk_token_overlap,
            locale=locale,
            runtime_head_features=runtime_head_features,
            trajectory_features=trajectory_features,
            extra=extra,
        )
        return self._request(
            "POST",
            "/groundedness",
            json=payload,
            expected_model=GroundednessResponse,
        )

    def redact_compliance(
        self,
        *,
        text: str,
        mode: ComplianceLabelMode = ComplianceLabelMode.OPEN,
        categories: Sequence[str] | None = None,
        labels: Sequence[str] | None = None,
        threshold: float = 0.5,
        redact: bool = True,
        redaction_mode: ComplianceRedactionMode = ComplianceRedactionMode.MASK,
        custom_labels: ComplianceCustomLabels | None = None,
        country: str | None = None,
        flat_ner: bool = True,
        multi_label: bool = False,
        include_original_text: bool = False,
        extra: Mapping[str, Any] | None = None,
    ) -> ComplianceRedactionResponse:
        """Detect and redact PII through the TRACE compliance runtime."""

        normalised_custom = [
            c.model_dump(exclude_none=True) if isinstance(c, ComplianceCustomLabel) else dict(c)
            for c in (custom_labels or [])
        ]
        try:
            req = ComplianceRedactionRequest(
                text=text,
                mode=mode,
                categories=list(categories or []),
                labels=list(labels) if labels else None,
                threshold=threshold,
                redact=redact,
                redaction_mode=redaction_mode,
                custom_labels=[ComplianceCustomLabel(**c) for c in normalised_custom],
                country=country,
                flat_ner=flat_ner,
                multi_label=multi_label,
                include_original_text=include_original_text,
            )
        except ValidationError as exc:
            raise LatenceTraceValidationError(
                f"client-side compliance request validation failed: {exc.errors()[:3]}",
                status=422,
            ) from exc
        body = req.model_dump(mode="json", exclude_none=True)
        if extra:
            body.update(dict(extra))
        return self._request(
            "POST",
            "/v1/compliance/redact",
            json=body,
            expected_model=ComplianceRedactionResponse,
        )

    def rollup(self, turns: Sequence[Mapping[str, Any]], **options: Any) -> Mapping[str, Any]:
        return self._request(
            "POST",
            "/groundedness/rollup",
            json={"turns": list(turns), **options},
            expected_model=None,
        )

    def session(
        self,
        *,
        session_id: str | None = None,
        memory_state: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> TraceSession:
        return TraceSession(
            self,
            session_id=session_id,
            memory_state=memory_state,
            metadata=metadata,
        )

    # --- internal --------------------------------------------------------

    def _build_payload(
        self,
        *,
        response_text: str,
        query: str | None,
        chunk_ids: Sequence[str] | None,
        raw_context: Sequence[str] | None,
        support_units: PremiseSupportUnits | None,
        attribution_mode: AttributionMode,
        primary_metric: str | None,
        coverage_threshold: float | None,
        chunk_token_budget: int | None,
        chunk_token_overlap: int | None,
        locale: str | None,
        runtime_head_features: Mapping[str, float] | None,
        trajectory_features: Mapping[str, float] | None,
        extra: Mapping[str, Any] | None,
    ) -> dict:
        normalised_units: list[dict] | None = None
        if support_units:
            normalised_units = [
                u.model_dump(exclude_none=True) if isinstance(u, SupportUnit) else dict(u)
                for u in support_units
            ]
        try:
            req = GroundednessRequest(
                query=query,
                response_text=response_text,
                chunk_ids=list(chunk_ids) if chunk_ids else None,
                raw_context=list(raw_context) if raw_context else None,
                support_units=[SupportUnit(**u) for u in normalised_units] if normalised_units else None,
                attribution_mode=attribution_mode,
                primary_metric=primary_metric,
                coverage_threshold=coverage_threshold,
                chunk_token_budget=chunk_token_budget,
                chunk_token_overlap=chunk_token_overlap,
                locale=locale,
                runtime_head_features=runtime_head_features,
                trajectory_features=trajectory_features,
            )
        except ValidationError as exc:
            raise LatenceTraceValidationError(
                f"client-side request validation failed: {exc.errors()[:3]}",
                status=422,
            ) from exc
        body = req.model_dump(mode="json", exclude_none=True)
        if extra:
            body.update(dict(extra))
        return body

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict | None = None,
        expected_model: type[BaseModel] | None = None,
    ) -> Any:
        attempt = 0
        last_error: Exception | None = None
        while True:
            try:
                response = self._client.request(method, path, json=json)
            except httpx.TimeoutException as exc:
                last_error = LatenceTraceTimeout(str(exc))
                if attempt >= self._retry.max_retries:
                    raise last_error from exc
                time.sleep(self._retry.sleep_for(attempt, None))
                attempt += 1
                continue
            except httpx.HTTPError as exc:
                raise LatenceTraceAPIError(str(exc), status=0) from exc

            if response.status_code < 400:
                return self._parse_success(response, expected_model)

            if response.status_code in RETRYABLE_STATUS and attempt < self._retry.max_retries:
                retry_after = parse_retry_after(response.headers.get("Retry-After"))
                time.sleep(self._retry.sleep_for(attempt, retry_after))
                attempt += 1
                continue

            raise self._error_from_response(response)

    @staticmethod
    def _parse_success(
        response: httpx.Response,
        expected_model: type[BaseModel] | None,
    ) -> Any:
        request_id = response.headers.get("x-request-id")
        body = response.json()
        if expected_model is None:
            if isinstance(body, dict) and request_id and "request_id" not in body:
                body = {**body, "request_id": request_id}
            return body
        try:
            instance = expected_model.model_validate({**body, "raw": body})
        except ValidationError as exc:
            raise LatenceTraceServerError(
                f"server response did not match expected schema: {exc.errors()[:3]}",
                status=200,
                request_id=request_id,
            ) from exc
        if request_id and getattr(instance, "request_id", None) is None:
            object.__setattr__(instance, "request_id", request_id)
        return instance

    @staticmethod
    def _error_from_response(response: httpx.Response) -> LatenceTraceAPIError:
        try:
            body = response.json()
        except ValueError:
            body = response.text
        request_id = response.headers.get("x-request-id")
        err = decode_error(response.status_code, body, request_id)
        if isinstance(err, LatenceTraceRateLimited):
            err.retry_after = parse_retry_after(response.headers.get("Retry-After"))
        return err
