"""Verify the TRACE product API surface against the freeze manifest.

This is the executable source of truth for phases 1 and 2 of the core
freeze: capability/API mapping and product API packaging. It intentionally
checks the full FastAPI app, `/agent-help`, RunPod actions, SDK namespaces,
golden examples, and benchmark gate names.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "docs" / "core_freeze" / "api_surface_manifest.json"
EXAMPLES_DIR = ROOT / "docs" / "core_freeze" / "examples"


def _load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _full_app_openapi() -> dict[str, Any]:
    os.environ.setdefault("LATENCE_TRACE_DISABLE_WARMUP", "1")
    os.environ.setdefault("LATENCE_TRACE_LICENSE_REQUIRE", "false")
    os.environ.setdefault("VOYAGER_GROUNDEDNESS_NLI_ENABLED", "0")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from server.main import create_app  # noqa: PLC0415

    return create_app(profile="fast").openapi()


async def _agent_help() -> dict[str, Any]:
    os.environ.setdefault("LATENCE_TRACE_DISABLE_WARMUP", "1")
    os.environ.setdefault("LATENCE_TRACE_LICENSE_REQUIRE", "false")
    os.environ.setdefault("VOYAGER_GROUNDEDNESS_NLI_ENABLED", "0")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from server.main import create_app  # noqa: PLC0415

    app = create_app(profile="fast")
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://contract-check",
    ) as client:
        response = await client.get("/agent-help")
    response.raise_for_status()
    return response.json()


def _openapi_operations(openapi: Mapping[str, Any]) -> set[tuple[str, str, str]]:
    operations: set[tuple[str, str, str]] = set()
    for path, methods in (openapi.get("paths") or {}).items():
        if not isinstance(methods, Mapping):
            continue
        for method, payload in methods.items():
            if not isinstance(payload, Mapping):
                continue
            operation_id = payload.get("operationId")
            if isinstance(operation_id, str):
                operations.add((str(method).upper(), str(path), operation_id))
    return operations


def _check_fastapi(manifest: Mapping[str, Any], errors: list[str]) -> None:
    operations = _openapi_operations(_full_app_openapi())
    for product in manifest["product_paths"]:
        for route in product.get("fastapi", []):
            expected = (
                str(route["method"]).upper(),
                str(route["path"]),
                str(route["operation_id"]),
            )
            if expected not in operations:
                errors.append(f"{product['id']}: FastAPI route missing {expected}")

    operation_ids = {operation_id for _method, _path, operation_id in operations}
    for operation_id in manifest.get("discovery", {}).get("fastapi_operation_ids", []):
        if operation_id not in operation_ids:
            errors.append(f"discovery: FastAPI operation_id missing {operation_id}")


def _check_agent_help(manifest: Mapping[str, Any], errors: list[str]) -> None:
    body = asyncio.run(_agent_help())
    endpoints = body.get("endpoints") or {}
    state_model = body.get("state_model") or {}
    for product in manifest["product_paths"]:
        for key in product.get("agent_help_keys", []):
            if key not in endpoints:
                errors.append(f"{product['id']}: /agent-help missing endpoint key {key}")
    for key in manifest.get("discovery", {}).get("agent_help_keys", []):
        if key not in endpoints:
            errors.append(f"discovery: /agent-help missing endpoint key {key}")
    for bucket in ("stateless_compute", "caller_carried_state", "server_stateful"):
        if bucket not in state_model:
            errors.append(f"/agent-help state_model missing {bucket}")


def _check_runpod(manifest: Mapping[str, Any], errors: list[str]) -> None:
    tree = ast.parse((ROOT / "runpod" / "handler.py").read_text(encoding="utf-8"))
    supported: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "SUPPORTED_RUNPOD_ACTIONS" for target in node.targets):
            continue
        if (
            isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "frozenset"
            and node.value.args
        ):
            supported = set(ast.literal_eval(node.value.args[0]))
            break
    if not supported:
        errors.append("runpod: SUPPORTED_RUNPOD_ACTIONS constant missing or not statically readable")
        return
    for product in manifest["product_paths"]:
        for action in product.get("runpod_actions", []):
            if action not in supported:
                errors.append(f"{product['id']}: RunPod action missing {action}")


def _check_examples(manifest: Mapping[str, Any], errors: list[str]) -> None:
    for product in manifest["product_paths"]:
        for example in product.get("examples", []):
            path = EXAMPLES_DIR / example
            if not path.exists():
                errors.append(f"{product['id']}: example missing {example}")
                continue
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                errors.append(f"{product['id']}: example {example} is invalid JSON: {exc}")


def _check_gates(manifest: Mapping[str, Any], errors: list[str]) -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scripts.trace_core_freeze_gate import GATES  # noqa: PLC0415

    gate_names = {gate.name for gate in GATES}
    for product in manifest["product_paths"]:
        for gate in product.get("gates", []):
            if gate not in gate_names:
                errors.append(f"{product['id']}: gate missing {gate}")


def _has_sdk_path(root: Any, path: str) -> bool:
    target = root
    for part in path.split("."):
        if not hasattr(target, part):
            return False
        target = getattr(target, part)
    return True


def _check_sdk(manifest: Mapping[str, Any], errors: list[str]) -> None:
    if str(ROOT / "clients" / "python") not in sys.path:
        sys.path.insert(0, str(ROOT / "clients" / "python"))
    from latence_trace_client import (  # noqa: PLC0415
        AsyncLatenceTraceClient,
        AsyncTraceSession,
        LatenceTraceClient,
        TraceSession,
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    sync_client = LatenceTraceClient(
        base_url="http://contract-check",
        transport=httpx.MockTransport(handler),
    )
    async_client = AsyncLatenceTraceClient(
        base_url="http://contract-check",
        transport=httpx.MockTransport(handler),
    )
    roots = {
        "sync": {
            "client": sync_client,
            "TraceSession": TraceSession,
        },
        "async": {
            "client": async_client,
            "AsyncTraceSession": AsyncTraceSession,
        },
    }
    try:
        for product in manifest["product_paths"]:
            sdk = product.get("sdk", {})
            for mode, paths in sdk.items():
                for path in paths:
                    if "." in path and path.split(".", 1)[0] in roots[mode]:
                        root_name, rest = path.split(".", 1)
                        root = roots[mode][root_name]
                    else:
                        root = roots[mode]["client"]
                        rest = path
                    if not _has_sdk_path(root, rest):
                        errors.append(f"{product['id']}: SDK {mode} path missing {path}")
    finally:
        sync_client.close()
        asyncio.run(async_client.aclose())


def run_contract_check(manifest_path: Path = MANIFEST_PATH) -> list[str]:
    manifest = _load_manifest(manifest_path)
    errors: list[str] = []
    _check_fastapi(manifest, errors)
    _check_agent_help(manifest, errors)
    _check_runpod(manifest, errors)
    _check_examples(manifest, errors)
    _check_gates(manifest, errors)
    _check_sdk(manifest, errors)
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--json", action="store_true", help="Print machine-readable result.")
    args = parser.parse_args(argv)

    errors = run_contract_check(args.manifest)
    payload = {
        "success": not errors,
        "manifest": str(args.manifest),
        "errors": errors,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    elif errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
    else:
        print("TRACE core contract check passed")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
