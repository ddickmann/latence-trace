"""Asyncio version of :class:`LatenceTraceClient`.

Behavioural parity with the sync client; share the same retry policy,
error decoding, and pydantic models.
"""

from __future__ import annotations

import asyncio
from typing import Any, List, Mapping, Optional, Sequence, Union

import httpx
from pydantic import ValidationError

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
    GroundednessRequest,
    GroundednessResponse,
    SupportUnit,
)

PremiseSupportUnits = Sequence[Union[SupportUnit, Mapping[str, Any]]]


class AsyncLatenceTraceClient:
    """Async client. ``async with AsyncLatenceTraceClient(...) as c:`` closes the pool."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        *,
        api_key: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        retry_policy: Optional[RetryPolicy] = None,
        transport: Optional[httpx.AsyncBaseTransport] = None,
        headers: Optional[Mapping[str, str]] = None,
    ) -> None:
        self._base_url = coerce_base_url(base_url)
        self._api_key = coerce_api_key(api_key)
        self._retry = retry_policy or RetryPolicy()
        self._headers = default_headers(self._api_key, headers)
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=timeout,
            transport=transport,
            headers=self._headers,
        )

    async def __aenter__(self) -> "AsyncLatenceTraceClient":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    # --- public surface --------------------------------------------------

    async def health(self) -> Mapping[str, Any]:
        return await self._request("GET", "/healthz", expected_model=None)

    async def ready(self) -> Mapping[str, Any]:
        return await self._request("GET", "/readyz", expected_model=None)

    async def agent_help(self) -> Mapping[str, Any]:
        return await self._request("GET", "/agent-help", expected_model=None)

    async def score_groundedness(
        self,
        *,
        response_text: str,
        query: Optional[str] = None,
        chunk_ids: Optional[Sequence[str]] = None,
        raw_context: Optional[Sequence[str]] = None,
        support_units: Optional[PremiseSupportUnits] = None,
        attribution_mode: AttributionMode = AttributionMode.CLOSED_BOOK,
        primary_metric: Optional[str] = None,
        coverage_threshold: Optional[float] = None,
        chunk_token_budget: Optional[int] = None,
        chunk_token_overlap: Optional[int] = None,
        locale: Optional[str] = None,
        runtime_head_features: Optional[Mapping[str, float]] = None,
        trajectory_features: Optional[Mapping[str, float]] = None,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> GroundednessResponse:
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
        return await self._request(
            "POST",
            "/groundedness",
            json=payload,
            expected_model=GroundednessResponse,
        )

    # --- internal --------------------------------------------------------

    def _build_payload(
        self,
        *,
        response_text: str,
        query: Optional[str],
        chunk_ids: Optional[Sequence[str]],
        raw_context: Optional[Sequence[str]],
        support_units: Optional[PremiseSupportUnits],
        attribution_mode: AttributionMode,
        primary_metric: Optional[str],
        coverage_threshold: Optional[float],
        chunk_token_budget: Optional[int],
        chunk_token_overlap: Optional[int],
        locale: Optional[str],
        runtime_head_features: Optional[Mapping[str, float]],
        trajectory_features: Optional[Mapping[str, float]],
        extra: Optional[Mapping[str, Any]],
    ) -> dict:
        normalised_units: Optional[List[dict]] = None
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

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: Optional[dict] = None,
        expected_model: Optional[type] = None,
    ) -> Any:
        attempt = 0
        while True:
            try:
                response = await self._client.request(method, path, json=json)
            except httpx.TimeoutException as exc:
                if attempt >= self._retry.max_retries:
                    raise LatenceTraceTimeout(str(exc)) from exc
                await asyncio.sleep(self._retry.sleep_for(attempt, None))
                attempt += 1
                continue
            except httpx.HTTPError as exc:
                raise LatenceTraceAPIError(str(exc), status=0) from exc

            if response.status_code < 400:
                return self._parse_success(response, expected_model)

            if response.status_code in RETRYABLE_STATUS and attempt < self._retry.max_retries:
                retry_after = parse_retry_after(response.headers.get("Retry-After"))
                await asyncio.sleep(self._retry.sleep_for(attempt, retry_after))
                attempt += 1
                continue

            raise self._error_from_response(response)

    @staticmethod
    def _parse_success(response: httpx.Response, expected_model: Optional[type]) -> Any:
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
