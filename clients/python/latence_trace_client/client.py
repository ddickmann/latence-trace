"""Synchronous SDK client.

Wraps an ``httpx.Client`` with the SDK's retry policy and typed
models. The class is reusable across many requests; use a single
instance per process whenever possible (httpx pools connections).
"""

from __future__ import annotations

import time
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


class LatenceTraceClient:
    """Sync client. ``with LatenceTraceClient(...) as c:`` closes the pool."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        *,
        api_key: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        retry_policy: Optional[RetryPolicy] = None,
        transport: Optional[httpx.BaseTransport] = None,
        headers: Optional[Mapping[str, str]] = None,
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

    # context manager support ---------------------------------------------

    def __enter__(self) -> "LatenceTraceClient":
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
        extra: Optional[Mapping[str, Any]] = None,
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
            extra=extra,
        )
        return self._request(
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
        json: Optional[dict] = None,
        expected_model: Optional[type] = None,
    ) -> Any:
        attempt = 0
        last_error: Optional[Exception] = None
        while True:
            try:
                response = self._client.request(method, path, json=json)
            except httpx.TimeoutException as exc:
                last_error = LatenceTraceTimeout(str(exc))
                if attempt >= self._retry.max_retries:
                    raise last_error
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
