"""Build and check the TRACE feature inventory.

The SDK/runtime freeze must not depend on manual memory of routes, flags, or
SDK methods. This script reads source files statically, emits a reproducible
inventory, and fails when the manifest and SDK/API surface drift apart.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SDK_REPO_ROOT = Path(os.environ.get("LATENCE_TRACE_SDK_REPO", ROOT.parent / "latence-trace-python")).resolve()
SDK_PACKAGE_ROOT = SDK_REPO_ROOT / "src" / "latence"
MANIFEST_PATH = ROOT / "docs/core_freeze/api_surface_manifest.json"
DEFAULT_OUTPUT = ROOT / "docs/core_freeze/trace_feature_inventory.json"
PY_FILE_GLOBS = (
    "latence_trace/**/*.py",
    "runpod/**/*.py",
    "scripts/*.py",
)
ENV_RE = re.compile(r"\b(?:LATENCE_TRACE|VOYAGER|RUNPOD)_[A-Z0-9_]+\b")
HTTP_METHODS = {"get", "post", "put", "patch", "delete"}


@dataclass(frozen=True)
class RouteRecord:
    file: str
    method: str
    path: str
    function: str
    operation_id: str | None


def _rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        try:
            return f"../latence-trace-python/{path.relative_to(SDK_REPO_ROOT).as_posix()}"
        except ValueError:
            return path.as_posix()


def _python_files() -> list[Path]:
    files: set[Path] = set()
    for pattern in PY_FILE_GLOBS:
        files.update(ROOT.glob(pattern))
    return sorted(path for path in files if path.is_file())


def _parse(path: Path) -> ast.Module | None:
    try:
        return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError):
        return None


def _literal(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except Exception:
        return None


def _decorator_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Call):
        node = node.func
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return None


def discover_routes() -> list[dict[str, Any]]:
    records: list[RouteRecord] = []
    for path in _python_files():
        tree = _parse(path)
        if tree is None:
            continue
        records.extend(_routes_in_body(path, tree.body, known_prefixes={}))
    return [record.__dict__ for record in sorted(records, key=lambda item: (item.path, item.method))]


def _routes_in_body(
    path: Path,
    body: list[ast.stmt],
    *,
    known_prefixes: dict[str, str],
    defaults: dict[str, Any] | None = None,
) -> list[RouteRecord]:
    records: list[RouteRecord] = []
    local_prefixes = dict(known_prefixes)
    defaults = defaults or {}

    for stmt in body:
        if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
            if _decorator_name(stmt.value.func) != "APIRouter":
                continue
            prefix = ""
            for keyword in stmt.value.keywords:
                if keyword.arg == "prefix":
                    prefix = _resolve_static_value(keyword.value, defaults) or ""
                    break
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    local_prefixes[target.id] = prefix

    for stmt in body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for decorator in stmt.decorator_list:
                if not isinstance(decorator, ast.Call):
                    continue
                method = _decorator_name(decorator)
                if method not in HTTP_METHODS:
                    continue
                router_name = _decorator_owner_name(decorator)
                route_path = _literal(decorator.args[0]) if decorator.args else None
                if not isinstance(route_path, str):
                    continue
                operation_id = None
                for keyword in decorator.keywords:
                    if keyword.arg == "operation_id":
                        value = _resolve_static_value(keyword.value, defaults)
                        operation_id = value if isinstance(value, str) else None
                records.append(
                    RouteRecord(
                        file=_rel(path),
                        method=method.upper(),
                        path=_join_route(local_prefixes.get(router_name, ""), route_path),
                        function=stmt.name,
                        operation_id=operation_id,
                    )
                )
            child_defaults = _function_defaults(stmt)
            records.extend(
                _routes_in_body(
                    path,
                    stmt.body,
                    known_prefixes=local_prefixes,
                    defaults=child_defaults,
                )
            )
    return records


def _function_defaults(node: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    positional = list(node.args.args)
    if node.args.defaults:
        defaulted_args = positional[-len(node.args.defaults) :]
        for arg, default_node in zip(defaulted_args, node.args.defaults, strict=True):
            defaults[arg.arg] = _literal(default_node)
    for arg, default_node in zip(node.args.kwonlyargs, node.args.kw_defaults, strict=True):
        defaults[arg.arg] = _literal(default_node) if default_node is not None else None
    return defaults


def _resolve_static_value(node: ast.AST, defaults: dict[str, Any]) -> Any:
    value = _literal(node)
    if value is not None:
        return value
    if isinstance(node, ast.Name):
        return defaults.get(node.id)
    if isinstance(node, ast.JoinedStr):
        pieces: list[str] = []
        for item in node.values:
            if isinstance(item, ast.Constant) and isinstance(item.value, str):
                pieces.append(item.value)
            elif isinstance(item, ast.FormattedValue):
                resolved = _resolve_static_value(item.value, defaults)
                if resolved is None:
                    return None
                pieces.append(str(resolved))
        return "".join(pieces)
    return None


def _decorator_owner_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        return func.value.id
    return None


def _join_route(prefix: str, route_path: str) -> str:
    if not prefix:
        return route_path or "/"
    if not route_path:
        return prefix
    if route_path == "/":
        return prefix
    return f"{prefix.rstrip('/')}/{route_path.lstrip('/')}"


def discover_runpod_actions() -> list[str]:
    handler_path = ROOT / "runpod/handler.py"
    tree = _parse(handler_path)
    if tree is None:
        return []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            names = [target.id for target in node.targets if isinstance(target, ast.Name)]
            if "SUPPORTED_RUNPOD_ACTIONS" not in names:
                continue
            value_node = node.value
            if isinstance(value_node, ast.Call) and value_node.args:
                value_node = value_node.args[0]
            value = _literal(value_node)
            if isinstance(value, (set, frozenset, tuple, list)):
                return sorted(str(item) for item in value)
    return []


def _annotation_to_string(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    try:
        return ast.unparse(node)
    except Exception:
        return None


def discover_pydantic_models() -> dict[str, Any]:
    models: dict[str, Any] = {}
    for path in sorted((ROOT / "latence_trace").glob("**/*models*.py")):
        tree = _parse(path)
        if tree is None:
            continue
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            base_names = {
                ast.unparse(base) for base in node.bases if isinstance(base, (ast.Name, ast.Attribute, ast.Subscript))
            }
            if "BaseModel" not in base_names and not any(name.endswith(".BaseModel") for name in base_names):
                continue
            fields: dict[str, Any] = {}
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    default = None
                    alias_choices: list[str] = []
                    if stmt.value is not None:
                        default = ast.unparse(stmt.value)
                        alias_choices = _field_alias_choices(stmt.value)
                    fields[stmt.target.id] = {
                        "annotation": _annotation_to_string(stmt.annotation),
                        "default": default,
                        "aliases": alias_choices,
                    }
            models[node.name] = {"file": _rel(path), "fields": fields}
    return dict(sorted(models.items()))


def _field_alias_choices(node: ast.AST) -> list[str]:
    aliases: list[str] = []
    if not isinstance(node, ast.Call):
        return aliases
    for keyword in node.keywords:
        if keyword.arg not in {"validation_alias", "alias"}:
            continue
        value = keyword.value
        if isinstance(value, ast.Call) and _decorator_name(value) == "AliasChoices":
            for arg in value.args:
                literal = _literal(arg)
                if isinstance(literal, str):
                    aliases.append(literal)
        else:
            literal = _literal(value)
            if isinstance(literal, str):
                aliases.append(literal)
    return aliases


def discover_env_settings() -> dict[str, list[str]]:
    settings: dict[str, set[str]] = {}
    for path in _python_files():
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for match in ENV_RE.finditer(text):
            settings.setdefault(match.group(0), set()).add(_rel(path))
    return {name: sorted(files) for name, files in sorted(settings.items())}


def _class_methods(path: Path) -> dict[str, list[str]]:
    tree = _parse(path)
    if tree is None:
        return {}
    classes: dict[str, list[str]] = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            methods = [
                stmt.name
                for stmt in node.body
                if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)) and not stmt.name.startswith("_")
            ]
            classes[node.name] = sorted(methods)
    return classes


def discover_sdk_methods() -> dict[str, Any]:
    sync_classes = _class_methods(SDK_PACKAGE_ROOT / "client.py")
    async_classes = _class_methods(SDK_PACKAGE_ROOT / "async_client.py")
    sync_methods = _sdk_surface(sync_classes, async_mode=False)
    async_methods = _sdk_surface(async_classes, async_mode=True)
    integrations = sorted(
        _display_path(path)
        for path in (SDK_PACKAGE_ROOT / "integrations").glob("*.py")
        if path.name != "__init__.py"
    )
    return {
        "repo": str(SDK_REPO_ROOT),
        "import_package": "latence",
        "sync_classes": sync_classes,
        "async_classes": async_classes,
        "sync_methods": sorted(sync_methods),
        "async_methods": sorted(async_methods),
        "integrations": integrations,
    }


def _sdk_surface(classes: dict[str, list[str]], *, async_mode: bool) -> set[str]:
    prefix_map = (
        {
            "AsyncPrivacyClient": "privacy",
            "AsyncGroundingClient": "grounding",
            "AsyncCompressionClient": "compression",
            "AsyncMemoryClient": "memory",
            "AsyncLatence": "",
            "AsyncTraceSession": "AsyncTraceSession",
        }
        if async_mode
        else {
            "PrivacyClient": "privacy",
            "GroundingClient": "grounding",
            "CompressionClient": "compression",
            "MemoryClient": "memory",
            "Latence": "",
            "TraceSession": "TraceSession",
        }
    )
    surface: set[str] = set()
    for cls_name, prefix in prefix_map.items():
        for method in classes.get(cls_name, []):
            surface.add(f"{prefix}.{method}" if prefix else method)
    return surface


def discover_examples_and_benchmarks() -> dict[str, Any]:
    examples_dir = ROOT / "docs/core_freeze/examples"
    return {
        "examples": sorted(path.name for path in examples_dir.glob("*.json")) if examples_dir.exists() else [],
        "benchmarks": sorted(
            _rel(path)
            for pattern in ("scripts/bench*.py", "scripts/run_*benchmark*.py", "scripts/*proof*.py")
            for path in ROOT.glob(pattern)
        ),
    }


def load_manifest() -> dict[str, Any]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def build_inventory() -> dict[str, Any]:
    manifest = load_manifest()
    return {
        "version": "trace-feature-inventory-v1",
        "manifest_version": manifest.get("version"),
        "fastapi_routes": discover_routes(),
        "runpod_actions": discover_runpod_actions(),
        "pydantic_models": discover_pydantic_models(),
        "env_settings": discover_env_settings(),
        "sdk": discover_sdk_methods(),
        **discover_examples_and_benchmarks(),
        "manifest_product_paths": manifest.get("product_paths", []),
        "discovery": manifest.get("discovery", {}),
        "checks": check_inventory(manifest=manifest, inventory=None),
    }


def check_inventory(*, manifest: dict[str, Any], inventory: dict[str, Any] | None) -> dict[str, list[str]]:
    data = inventory or {
        "fastapi_routes": discover_routes(),
        "runpod_actions": discover_runpod_actions(),
        "sdk": discover_sdk_methods(),
        "examples": discover_examples_and_benchmarks()["examples"],
        "env_settings": discover_env_settings(),
    }
    product_paths = manifest.get("product_paths", [])
    manifest_fastapi = {
        (entry.get("method"), entry.get("path"))
        for product in product_paths
        for entry in product.get("fastapi", [])
    }
    manifest_runpod = {
        str(action)
        for product in product_paths
        for action in product.get("runpod_actions", [])
    }
    manifest_examples = {
        str(example)
        for product in product_paths
        for example in product.get("examples", [])
    }
    manifest_sync = {
        str(method)
        for product in product_paths
        for method in product.get("sdk", {}).get("sync", [])
    }
    manifest_async = {
        str(method)
        for product in product_paths
        for method in product.get("sdk", {}).get("async", [])
    }
    non_canonical_paths = {
        path
        for surface in manifest.get("non_canonical_surfaces", [])
        for path in surface.get("paths", [])
    }
    discovery_paths = {
        "/agent-help",
        "/.well-known/ai-plugin.json",
        "/health",
        "/healthz",
        "/readyz",
        "/docs",
        "/openapi.json",
    }
    route_mismatches = []
    for route in data["fastapi_routes"]:
        route_key = (route["method"], route["path"])
        if route_key in manifest_fastapi:
            continue
        if _path_matches(route["path"], non_canonical_paths) or route["path"] in discovery_paths:
            continue
        if route["path"].startswith("/v1/amber/"):
            continue
        route_mismatches.append(f"{route['method']} {route['path']} ({route['file']})")

    runpod_unmapped = [
        action
        for action in data["runpod_actions"]
        if action and action not in manifest_runpod
    ]
    examples_missing = sorted(example for example in manifest_examples if example not in data["examples"])
    sync_missing = sorted(method for method in manifest_sync if method not in data["sdk"]["sync_methods"])
    async_missing = sorted(method for method in manifest_async if method not in data["sdk"]["async_methods"])
    sdk_extra_sync = sorted(
        method
        for method in data["sdk"]["sync_methods"]
        if _is_public_sdk_method(method) and method not in manifest_sync
    )
    sdk_extra_async = sorted(
        method
        for method in data["sdk"]["async_methods"]
        if _is_public_sdk_method(method) and method not in manifest_async
    )
    undocumented_env = [
        name
        for name in data["env_settings"]
        if name.startswith("LATENCE_TRACE_")
        and not _env_is_documented_enough(name, data["env_settings"][name])
    ]
    return {
        "route_manifest_mismatches": sorted(route_mismatches),
        "runpod_actions_unmapped": sorted(runpod_unmapped),
        "manifest_examples_missing": examples_missing,
        "sdk_sync_missing_from_manifest": sync_missing,
        "sdk_async_missing_from_manifest": async_missing,
        "sdk_extra_sync_methods": sdk_extra_sync,
        "sdk_extra_async_methods": sdk_extra_async,
        "env_settings_to_review": sorted(undocumented_env),
    }


def _is_public_sdk_method(method: str) -> bool:
    if method in {"close", "health", "ready", "agent_help", "session", "redact_compliance", "score_groundedness", "rollup"}:
        return method in {"session", "redact_compliance", "score_groundedness", "rollup"}
    return method.startswith(
        ("privacy.", "grounding.", "compression.", "memory.", "TraceSession.", "AsyncTraceSession.")
    )


def _path_matches(path: str, patterns: set[str]) -> bool:
    for pattern in patterns:
        if pattern.endswith("/*") and path.startswith(pattern[:-1]):
            return True
        if path == pattern:
            return True
    return False


def _env_is_documented_enough(name: str, files: list[str]) -> bool:
    if any(file.startswith("docs/") for file in files):
        return True
    # Runtime-only flags are allowed but still emitted for review in the
    # inventory; keep the check conservative enough to avoid blocking on every
    # operational tuning variable during the first freeze pass.
    return name in {
        "LATENCE_TRACE_CONTEXT_TRUST_ENABLED",
        "LATENCE_TRACE_CONTEXT_TRUST_PROVIDER",
        "LATENCE_TRACE_CONTEXT_TRUST_PROMPT_GUARD_COMPILE",
        "LATENCE_TRACE_CONTEXT_TRUST_PROMPT_GUARD_COMPILE_MODE",
        "LATENCE_TRACE_CONTEXT_TRUST_PROMPT_GUARD_MODEL",
        "LATENCE_TRACE_START_MANAGED_VLLM",
    }


def has_blocking_failures(checks: dict[str, list[str]]) -> bool:
    blocking = (
        "route_manifest_mismatches",
        "runpod_actions_unmapped",
        "manifest_examples_missing",
        "sdk_sync_missing_from_manifest",
        "sdk_async_missing_from_manifest",
        "sdk_extra_sync_methods",
        "sdk_extra_async_methods",
    )
    return any(checks.get(key) for key in blocking)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", type=Path, help="Write inventory JSON to this path.")
    parser.add_argument("--check", action="store_true", help="Fail if blocking drift is detected.")
    args = parser.parse_args(argv)

    inventory = build_inventory()
    inventory["checks"] = check_inventory(manifest=load_manifest(), inventory=inventory)
    if args.write:
        output = args.write if args.write.is_absolute() else ROOT / args.write
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {output}")
    if args.check:
        print(json.dumps(inventory["checks"], indent=2, sort_keys=True))
        if has_blocking_failures(inventory["checks"]):
            return 1
    if not args.write and not args.check:
        print(json.dumps(inventory, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
