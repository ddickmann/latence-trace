#!/usr/bin/env python3
"""Canary the Latence TRACE compliance runtime through public paths.

Set either or both:

  LATENCE_TRACE_COMPLIANCE_RUNPOD_URL=https://api.runpod.ai/v2/<endpoint>/runsync
  RUNPOD_ENDPOINT_ID=<endpoint-id>
  LATENCE_TRACE_RUNPOD_ENDPOINT_NAME=latence-trace
  LATENCE_TRACE_COMPLIANCE_GATEWAY_URL=https://api.latence.ai/v1/compliance/redact

For gateway checks, also set LATENCE_TRACE_API_KEY. For RunPod checks, set
LATENCE_TRACE_RUNPOD_API_KEY or RUNPOD_API_KEY.
The script validates behavior without printing raw PII from the fixture.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any

import httpx

FIXTURE_TEXT = "Contact Jane Doe at jane@example.com before sending the prompt."
EXPECTED_LABELS = {"person", "email"}
RUNPOD_GRAPHQL_URL = "https://api.runpod.io/graphql"
DEFAULT_GATEWAY_URL = "https://api.latence.ai/v1/compliance/redact"


@dataclass(frozen=True)
class CanaryTarget:
    name: str
    url: str
    headers: dict[str, str]
    payload: dict[str, Any]


def _gateway_target(url: str) -> CanaryTarget:
    api_key = os.environ.get("LATENCE_TRACE_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("LATENCE_TRACE_API_KEY is required for gateway canary")
    return CanaryTarget(
        name="gateway",
        url=url,
        headers={"authorization": f"Bearer {api_key}", "content-type": "application/json"},
        payload={
            "text": FIXTURE_TEXT,
            "mode": "category",
            "labels": sorted(EXPECTED_LABELS),
            "redact": True,
            "redaction_mode": "mask",
            "include_original_text": False,
        },
    )


def _runpod_target(url: str) -> CanaryTarget:
    token = (
        os.environ.get("LATENCE_TRACE_RUNPOD_API_KEY")
        or os.environ.get("RUNPOD_API_KEY")
        or ""
    ).strip()
    headers = {"content-type": "application/json"}
    if token:
        headers["authorization"] = f"Bearer {token}"
    return CanaryTarget(
        name="runpod",
        url=url,
        headers=headers,
        payload={
            "input": {
                "action": "redact",
                "text": FIXTURE_TEXT,
                "labels": sorted(EXPECTED_LABELS),
                "redact": True,
                "redaction_mode": "mask",
                "include_original_text": False,
            }
        },
    )


def _runpod_api_key() -> str:
    return (
        os.environ.get("LATENCE_TRACE_RUNPOD_API_KEY")
        or os.environ.get("RUNPOD_API_KEY")
        or ""
    ).strip()


def _runpod_graphql(client: httpx.Client, query: str) -> dict[str, Any]:
    api_key = _runpod_api_key()
    if not api_key:
        raise SystemExit(
            "RUNPOD_API_KEY or LATENCE_TRACE_RUNPOD_API_KEY is required to discover endpoints"
        )
    response = client.post(
        RUNPOD_GRAPHQL_URL,
        params={"api_key": api_key},
        json={"query": query},
        headers={"content-type": "application/json"},
    )
    if response.status_code >= 400:
        raise RuntimeError(f"RunPod GraphQL endpoint discovery failed with HTTP {response.status_code}")
    payload = response.json()
    if payload.get("errors"):
        messages = "; ".join(str(err.get("message", err)) for err in payload["errors"])
        raise RuntimeError(f"RunPod GraphQL endpoint discovery failed: {messages}")
    return payload


def _discover_runpod_endpoint_ids(client: httpx.Client) -> list[str]:
    query = """
    query ComplianceCanaryEndpoints {
      myself {
        endpoints {
          id
          name
        }
      }
    }
    """
    payload = _runpod_graphql(client, query)
    endpoints = payload.get("data", {}).get("myself", {}).get("endpoints") or []
    endpoints = [item for item in endpoints if isinstance(item, dict) and item.get("id")]
    if not endpoints:
        return []

    requested_name = (
        os.environ.get("LATENCE_TRACE_RUNPOD_ENDPOINT_NAME")
        or os.environ.get("RUNPOD_ENDPOINT_NAME")
        or ""
    ).strip().lower()
    if requested_name:
        matches = [
            item
            for item in endpoints
            if str(item.get("name", "")).strip().lower() == requested_name
        ]
        if len(matches) == 1:
            return [str(matches[0]["id"])]
        raise RuntimeError(f"No unique RunPod endpoint matched name {requested_name!r}")

    candidates = [
        item
        for item in endpoints
        if any(
            token in str(item.get("name", "")).lower()
            for token in ("trace", "latence", "compliance", "redaction")
        )
    ]
    if candidates:
        return [str(item["id"]) for item in candidates]
    if len(endpoints) == 1:
        return [str(endpoints[0]["id"])]

    names = ", ".join(str(item.get("name") or item.get("id")) for item in endpoints[:8])
    raise RuntimeError(
        "Multiple RunPod endpoints found; set RUNPOD_ENDPOINT_ID or "
        f"LATENCE_TRACE_RUNPOD_ENDPOINT_NAME. Candidates: {names}"
    )


def _extract_response(target: CanaryTarget, raw: dict[str, Any]) -> dict[str, Any]:
    if target.name == "runpod":
        local_result = raw.get("result")
        if isinstance(local_result, dict):
            return {
                **local_result,
                "_runpod_status": raw.get("status"),
                "_runpod_error": raw.get("error"),
            }
        output = raw.get("output")
        if isinstance(output, dict):
            result = output.get("result")
            if isinstance(result, dict):
                return {
                    **result,
                    "_runpod_status": raw.get("status"),
                    "_runpod_error": raw.get("error"),
                }
            return {
                **output,
                "_runpod_status": raw.get("status"),
                "_runpod_error": raw.get("error"),
            }
        return raw
    return raw


def _runpod_status_url(run_url: str, job_id: str) -> str:
    base = run_url.rsplit("/", 1)[0]
    return f"{base}/status/{job_id}"


def _wait_for_runpod_completion(
    client: httpx.Client,
    target: CanaryTarget,
    raw: dict[str, Any],
    started: float,
) -> dict[str, Any]:
    status = str(raw.get("status") or "").upper()
    job_id = raw.get("id")
    terminal_statuses = {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}
    if target.name != "runpod" or status in terminal_statuses or not isinstance(job_id, str):
        return raw

    max_ms = float(os.environ.get("LATENCE_TRACE_COMPLIANCE_CANARY_MAX_MS", "5000"))
    poll_seconds = float(os.environ.get("LATENCE_TRACE_COMPLIANCE_CANARY_POLL_S", "2"))
    status_url = _runpod_status_url(target.url, job_id)
    while (time.perf_counter() - started) * 1000.0 < max_ms:
        time.sleep(max(0.1, poll_seconds))
        response = client.get(status_url, headers=target.headers)
        if response.status_code >= 400:
            raise RuntimeError(f"RunPod status polling failed with HTTP {response.status_code}")
        raw = response.json()
        status = str(raw.get("status") or "").upper()
        if status in terminal_statuses:
            return raw
    return raw


def _validate_response(target: CanaryTarget, response: dict[str, Any], elapsed_ms: float) -> None:
    if response.get("success") is not True:
        detail = response.get("hint") or response.get("error") or response.get("error_code")
        status = response.get("_runpod_status") or response.get("status")
        suffix = f": {detail}" if detail else ""
        if status:
            suffix = f" ({status}){suffix}"
        raise AssertionError(f"{target.name}: response success was not true{suffix}")
    if response.get("original_text") is not None:
        raise AssertionError(f"{target.name}: original_text leaked despite include_original_text=false")
    redacted_text = response.get("redacted_text")
    if not isinstance(redacted_text, str) or "[EMAIL]" not in redacted_text:
        raise AssertionError(f"{target.name}: expected email mask in redacted_text")
    labels = set(response.get("unique_labels") or [])
    if not labels.intersection(EXPECTED_LABELS):
        raise AssertionError(f"{target.name}: expected at least one compliance label")
    if int(response.get("entity_count") or 0) < 1:
        raise AssertionError(f"{target.name}: expected at least one detected entity")
    if elapsed_ms > float(os.environ.get("LATENCE_TRACE_COMPLIANCE_CANARY_MAX_MS", "5000")):
        raise AssertionError(f"{target.name}: canary exceeded latency budget ({elapsed_ms:.1f} ms)")


def _run_target(client: httpx.Client, target: CanaryTarget) -> dict[str, Any]:
    started = time.perf_counter()
    res = client.post(target.url, headers=target.headers, json=target.payload)
    res.raise_for_status()
    raw = _wait_for_runpod_completion(client, target, res.json(), started)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    data = _extract_response(target, raw)
    _validate_response(target, data, elapsed_ms)
    return {
        "target": target.name,
        "status": "ok",
        "elapsed_ms": round(elapsed_ms, 2),
        "entity_count": data.get("entity_count"),
        "unique_labels": data.get("unique_labels"),
        "chunks_processed": data.get("chunks_processed"),
    }


def _run_first_successful_runpod_target(
    client: httpx.Client,
    endpoint_ids: list[str],
) -> dict[str, Any]:
    failures: list[str] = []
    for endpoint_id in endpoint_ids:
        target = _runpod_target(f"https://api.runpod.ai/v2/{endpoint_id}/runsync")
        try:
            return _run_target(client, target)
        except Exception as exc:  # noqa: BLE001 - aggregate candidate failures
            failures.append(str(exc))
    raise RuntimeError("No discovered RunPod endpoint passed compliance canary: " + " | ".join(failures))


def main() -> int:
    targets: list[CanaryTarget] = []
    timeout = float(os.environ.get("LATENCE_TRACE_COMPLIANCE_CANARY_TIMEOUT_S", "30"))
    results = []
    with httpx.Client(timeout=timeout) as client:
        runpod_url = os.environ.get("LATENCE_TRACE_COMPLIANCE_RUNPOD_URL", "").strip()
        endpoint_id = os.environ.get("RUNPOD_ENDPOINT_ID", "").strip()
        if not runpod_url and endpoint_id:
            runpod_url = f"https://api.runpod.ai/v2/{endpoint_id}/runsync"
        if runpod_url:
            targets.append(_runpod_target(runpod_url))
        elif _runpod_api_key():
            endpoint_ids = _discover_runpod_endpoint_ids(client)
            if endpoint_ids:
                results.append(_run_first_successful_runpod_target(client, endpoint_ids))

        gateway_url = os.environ.get("LATENCE_TRACE_COMPLIANCE_GATEWAY_URL", "").strip()
        if not gateway_url and os.environ.get("LATENCE_TRACE_API_KEY"):
            gateway_url = DEFAULT_GATEWAY_URL
        if gateway_url:
            targets.append(_gateway_target(gateway_url))

        if not targets and not results:
            raise SystemExit(
                "Set RUNPOD_ENDPOINT_ID, LATENCE_TRACE_COMPLIANCE_RUNPOD_URL, "
                "or LATENCE_TRACE_API_KEY for the default gateway canary"
            )
        for target in targets:
            results.append(_run_target(client, target))
    print(json.dumps({"success": True, "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"success": False, "error": str(exc)}), file=sys.stderr)
        raise SystemExit(1) from exc
