"""SDK contract helpers used by tests and release gates."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def collect_sync_sdk_methods(client: Any) -> set[str]:
    return _collect_client_methods(client, trace_session_name="TraceSession")


def collect_async_sdk_methods(client: Any) -> set[str]:
    return _collect_client_methods(client, trace_session_name="AsyncTraceSession")


def manifest_sdk_methods(manifest: dict[str, Any], *, async_mode: bool) -> set[str]:
    key = "async" if async_mode else "sync"
    return {
        str(method)
        for product in manifest.get("product_paths", [])
        for method in product.get("sdk", {}).get(key, [])
    }


def missing_methods(required: Iterable[str], actual: Iterable[str]) -> list[str]:
    return sorted(set(required) - set(actual))


def _collect_client_methods(client: Any, *, trace_session_name: str) -> set[str]:
    methods = set()
    for namespace in ("privacy", "grounding", "compression", "memory"):
        namespace_client = getattr(client, namespace)
        for name in dir(namespace_client):
            if not name.startswith("_") and callable(getattr(namespace_client, name)):
                methods.add(f"{namespace}.{name}")
    for name in dir(client):
        if not name.startswith("_") and callable(getattr(client, name)):
            methods.add(name)
    session = client.session(session_id="contract")
    for name in dir(session):
        if not name.startswith("_") and callable(getattr(session, name)):
            methods.add(f"{trace_session_name}.{name}")
    return methods
