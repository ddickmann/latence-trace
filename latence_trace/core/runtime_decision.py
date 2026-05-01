"""Rollback-safe automatic decision policy for TRACE runtime records.

The policy is intentionally small and data-driven. The production runtime emits
decision records by default; operators can roll back immediately with
``LATENCE_TRACE_RUNTIME_DECISION_ENABLED=0`` and may point
``LATENCE_TRACE_RUNTIME_POLICY_PATH`` at a versioned JSON policy.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Optional

from latence_trace.core import runtime_heads

_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
_DEFAULT_POLICY_PATH = _DATA_DIR / "runtime_policy.optimized_v1_plus_calibrator.json"
_DEFAULT_HEAD_REGISTRY_PATH = _DATA_DIR / "runtime_head_registry.root_cause_solution_v1.json"
_FALSE_VALUES = {"0", "false", "no", "off"}
_LOCK = threading.Lock()
_CACHE: tuple[Optional[Path], Optional[float], Optional[dict[str, Any]], Optional[str]] = (
    None,
    None,
    None,
    None,
)
_HEAD_REGISTRY_CACHE: tuple[
    Optional[Path],
    Optional[float],
    Optional[dict[str, Any]],
    Optional[str],
] = (None, None, None, None)
_CACHE_LAST_CHECK: dict[str, float] = {"policy": 0.0, "head_registry": 0.0}
_CACHE_TTL_SECONDS = 1.0


def enabled() -> bool:
    return os.environ.get("LATENCE_TRACE_RUNTIME_DECISION_ENABLED", "1").strip().lower() not in _FALSE_VALUES


def _policy_path() -> Path:
    raw = os.environ.get("LATENCE_TRACE_RUNTIME_POLICY_PATH", "").strip()
    return Path(raw) if raw else _DEFAULT_POLICY_PATH


def _head_registry_path() -> Path:
    raw = os.environ.get("LATENCE_TRACE_RUNTIME_HEAD_REGISTRY_PATH", "").strip()
    return Path(raw) if raw else _DEFAULT_HEAD_REGISTRY_PATH


def _load_json_cached(
    path: Path,
    cache_name: str,
) -> tuple[Optional[dict[str, Any]], Optional[str], Optional[str]]:
    now = time.monotonic()
    global _CACHE, _HEAD_REGISTRY_CACHE
    with _LOCK:
        cache = _CACHE if cache_name == "policy" else _HEAD_REGISTRY_CACHE
        cached_path, _cached_mtime, cached_payload, cached_sha = cache
        if (
            cached_path == path
            and cached_payload is not None
            and now - _CACHE_LAST_CHECK.get(cache_name, 0.0) < _CACHE_TTL_SECONDS
        ):
            return cached_payload, cached_sha, None
    if not path.exists():
        return None, None, f"missing_json:{path}"
    try:
        stat = path.stat()
    except OSError as exc:
        return None, None, f"stat_failed:{exc}"

    with _LOCK:
        cache = _CACHE if cache_name == "policy" else _HEAD_REGISTRY_CACHE
        cached_path, cached_mtime, cached_payload, cached_sha = cache
        if cached_path == path and cached_mtime == stat.st_mtime and cached_payload is not None:
            return cached_payload, cached_sha, None
        try:
            raw = path.read_bytes()
            payload = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            return None, None, f"load_failed:{exc}"
        sha = hashlib.sha256(raw).hexdigest()
        if cache_name == "policy":
            _CACHE = (path, stat.st_mtime, payload, sha)
        else:
            _HEAD_REGISTRY_CACHE = (path, stat.st_mtime, payload, sha)
        _CACHE_LAST_CHECK[cache_name] = now
        return payload, sha, None


def _load_policy() -> tuple[Optional[dict[str, Any]], Optional[str], Optional[str]]:
    policy, sha, error = _load_json_cached(_policy_path(), "policy")
    if error and error.startswith("missing_json:"):
        return None, None, error.replace("missing_json:", "missing_policy:", 1)
    return policy, sha, error


def _load_head_registry() -> tuple[Optional[dict[str, Any]], Optional[str]]:
    registry, sha, error = _load_json_cached(_head_registry_path(), "head_registry")
    if error:
        return None, None
    return registry, sha


def reset_policy_cache_for_tests() -> None:
    global _CACHE, _HEAD_REGISTRY_CACHE
    with _LOCK:
        _CACHE = (None, None, None, None)
        _HEAD_REGISTRY_CACHE = (None, None, None, None)
        _CACHE_LAST_CHECK["policy"] = 0.0
        _CACHE_LAST_CHECK["head_registry"] = 0.0
    runtime_heads.reset_head_cache_for_tests()


def _score_from_response(response: Any) -> tuple[float, str]:
    scores = response.scores
    if getattr(scores, "composite_phantom_score", None) is not None:
        return float(scores.composite_phantom_score), "composite_phantom_score"
    if getattr(scores, "groundedness_v2", None) is not None:
        return float(scores.groundedness_v2), "groundedness_v2"
    return float(scores.primary_score), str(scores.primary_name or "primary_score")


def _class_key(response: Any) -> str:
    route = getattr(response, "corpus_route", None)
    if route is not None and getattr(route, "corpus_type", None):
        return str(route.corpus_type)
    return "unknown"


def _class_policy(policy: Mapping[str, Any], class_key: str) -> Mapping[str, Any]:
    classes = policy.get("classes") if isinstance(policy.get("classes"), Mapping) else {}
    entry = classes.get(class_key)
    if isinstance(entry, Mapping):
        return entry
    default = policy.get("default")
    return default if isinstance(default, Mapping) else {}


def _head_registry_entry(registry: Optional[Mapping[str, Any]], class_key: str) -> Mapping[str, Any]:
    if not registry:
        return {}
    heads = registry.get("runtime_head_registry")
    if not isinstance(heads, Mapping):
        return {}
    entry = heads.get(class_key)
    return entry if isinstance(entry, Mapping) else {}


def decide_action(score: float, class_policy: Mapping[str, Any]) -> str:
    allow_disabled = bool(class_policy.get("allow_disabled", False))
    block_disabled = bool(class_policy.get("block_disabled", False))
    allow_threshold = float(class_policy.get("allow_threshold", 1.0))
    block_threshold = float(class_policy.get("block_threshold", 0.0))

    if not block_disabled and score <= block_threshold:
        return "block"
    if not allow_disabled and score >= allow_threshold:
        return "allow"
    return "auto_repair"


def _band_for_action(action: str) -> str:
    if action == "allow":
        return "green"
    if action == "block":
        return "red"
    return "amber"


def _support_evidence(response: Any, *, limit: int = 3) -> list[dict[str, Any]]:
    units = list(getattr(response, "support_units", []) or [])
    units.sort(
        key=lambda unit: (
            float(getattr(unit, "coverage_score", 0.0) or 0.0),
            float(getattr(unit, "usage_confidence", 0.0) or 0.0),
        ),
        reverse=True,
    )
    evidence: list[dict[str, Any]] = []
    for unit in units[:limit]:
        evidence.append(
            {
                "index": int(getattr(unit, "index", len(evidence))),
                "support_id": str(getattr(unit, "support_id", "")),
                "text": str(getattr(unit, "text", ""))[:800],
                "coverage_score": float(getattr(unit, "coverage_score", 0.0) or 0.0),
                "usage_state": str(getattr(getattr(unit, "usage_state", None), "value", getattr(unit, "usage_state", ""))),
            }
        )
    return evidence


def _unsupported_spans(response: Any, *, limit: int = 16) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    for token in list(getattr(response, "response_tokens", []) or []):
        token_text = str(getattr(token, "token", ""))
        if token_text in {"[CLS]", "[SEP]", "[PAD]"}:
            continue
        heatmap_score = float(getattr(token, "heatmap_score", 0.0) or 0.0)
        nli_score = getattr(token, "nli_score", None)
        calibrated = getattr(token, "reverse_context_calibrated", None)
        unsupported = heatmap_score < 0.50
        if nli_score is not None:
            unsupported = unsupported or float(nli_score) < -0.50
        if calibrated is not None:
            unsupported = unsupported or float(calibrated) < 0.50
        if not unsupported:
            continue
        spans.append(
            {
                "token_index": int(getattr(token, "index", len(spans))),
                "token": token_text,
                "heatmap_score": heatmap_score,
                "nli_score": None if nli_score is None else float(nli_score),
                "reverse_context_calibrated": None if calibrated is None else float(calibrated),
                "char_start": getattr(token, "char_start", None),
                "char_end": getattr(token, "char_end", None),
            }
        )
        if len(spans) >= limit:
            break
    return spans


def _reason_codes(response: Any) -> list[str]:
    codes: set[str] = set()
    if getattr(response, "reason", None):
        codes.add(str(response.reason))
    file_attr = getattr(response, "file_attribution", None)
    hist = getattr(file_attr, "reason_code_histogram", None)
    if isinstance(hist, Mapping):
        codes.update(str(key) for key, count in hist.items() if int(count or 0) > 0)
    return sorted(codes)


def build_runtime_decision(response: Any) -> Optional[dict[str, Any]]:
    if not enabled():
        return None
    policy, policy_sha, error = _load_policy()
    if policy is None:
        score, score_channel = _score_from_response(response)
        return {
            "policy_version": "unavailable",
            "policy_sha256": None,
            "class_key": _class_key(response),
            "score": score,
            "score_channel": score_channel,
            "band": str(response.scores.risk_band or "unknown"),
            "action": "auto_repair",
            "evidence": _support_evidence(response),
            "unsupported_spans": _unsupported_spans(response),
            "reason_codes": [error or "policy_unavailable"],
            "rollback_safe": True,
        }

    score, score_channel = _score_from_response(response)
    class_key = _class_key(response)
    class_policy = _class_policy(policy, class_key)
    head_registry, head_registry_sha = _load_head_registry()
    head_entry = _head_registry_entry(head_registry, class_key)
    head_eval = runtime_heads.evaluate_runtime_head(response, class_key)
    decision_score = score
    decision_channel = score_channel
    if head_eval.get("head_enabled") and head_eval.get("head_score") is not None:
        decision_score = float(head_eval["head_score"])
        decision_channel = f"head:{head_eval.get('head_id') or 'runtime_head'}"
    if bool(class_policy.get("requires_head_score")) and not head_eval.get("head_enabled"):
        action = "auto_repair"
    else:
        action = decide_action(decision_score, class_policy)
    decision_band = _band_for_action(action)
    return {
        "policy_version": str(policy.get("channel") or "runtime_decision"),
        "policy_sha256": policy_sha,
        "head_id": head_eval.get("head_id", head_entry.get("head_id")),
        "head_version": head_eval.get("head_version", head_entry.get("version")),
        "head_registry_sha256": head_eval.get("head_registry_sha256", head_registry_sha),
        "head_enabled": head_eval.get("head_enabled", head_entry.get("enabled")),
        "head_score": head_eval.get("head_score"),
        "head_features_used": head_eval.get("head_features_used", []),
        "head_reason_codes": head_eval.get("head_reason_codes", []),
        "class_key": class_key,
        "score": decision_score,
        "score_channel": decision_channel,
        "band": decision_band,
        "action": action,
        "evidence": _support_evidence(response),
        "unsupported_spans": _unsupported_spans(response),
        "reason_codes": _reason_codes(response),
        "allow_disabled": bool(class_policy.get("allow_disabled", False)),
        "block_disabled": bool(class_policy.get("block_disabled", False)),
        "allow_threshold": float(class_policy.get("allow_threshold", 1.0)),
        "block_threshold": float(class_policy.get("block_threshold", 0.0)),
        "rollback_safe": True,
    }
