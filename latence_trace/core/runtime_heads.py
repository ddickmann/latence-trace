"""Runtime loader for per-router-class TRACE head artifacts.

The loader is deliberately conservative: head artifacts are versioned JSON,
checksum-validated against the registry, cached by path/mtime, and optional.
Any missing, corrupt, disabled, or invalid head returns a non-fatal evaluation
record so the runtime decision layer can fall back to v1/router behavior.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Optional

_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIR = Path(__file__).resolve().parents[1] / "data"
_DEFAULT_REGISTRY_PATH = _DATA_DIR / "runtime_head_registry.root_cause_solution_v1.json"
_LOCK = threading.Lock()
_REGISTRY_CACHE: tuple[Optional[Path], Optional[float], Optional[dict[str, Any]], Optional[str]] = (
    None,
    None,
    None,
    None,
)
_HEAD_CACHE: dict[Path, tuple[float, dict[str, Any], str]] = {}
_CACHE_LAST_CHECK: dict[Any, float] = {}
_CACHE_TTL_SECONDS = 1.0


def reset_head_cache_for_tests() -> None:
    global _REGISTRY_CACHE, _HEAD_CACHE
    with _LOCK:
        _REGISTRY_CACHE = (None, None, None, None)
        _HEAD_CACHE = {}
        _CACHE_LAST_CHECK.clear()


def registry_path() -> Path:
    raw = os.environ.get("LATENCE_TRACE_RUNTIME_HEAD_REGISTRY_PATH", "").strip()
    return Path(raw) if raw else _DEFAULT_REGISTRY_PATH


def evaluate_runtime_head(response: Any, class_key: str) -> dict[str, Any]:
    registry, registry_sha, registry_error = _load_registry()
    if registry is None:
        return _fallback(
            class_key=class_key,
            reason=f"head_registry_unavailable:{registry_error}",
            registry_sha=registry_sha,
        )

    entry = _registry_entry(registry, class_key)
    if not entry:
        return _fallback(class_key=class_key, reason="head_registry_missing_class", registry_sha=registry_sha)

    head, head_sha, head_error = _load_head_artifact(entry)
    if head is None:
        return _fallback(
            class_key=class_key,
            reason=f"head_artifact_unavailable:{head_error}",
            registry_sha=registry_sha,
            entry=entry,
        )

    enabled = bool(entry.get("enabled")) and bool(head.get("enabled"))
    result = {
        "head_id": str(entry.get("head_id") or head.get("head_id") or ""),
        "head_version": str(entry.get("version") or head.get("version") or ""),
        "head_enabled": enabled,
        "head_registry_sha256": registry_sha,
        "head_artifact_sha256": head_sha,
        "head_score": None,
        "head_features_used": list(head.get("feature_names") or []),
        "head_reason_codes": [],
    }
    strategy = head.get("score_strategy") if isinstance(head.get("score_strategy"), Mapping) else {}
    result["head_reason_codes"].extend(str(code) for code in strategy.get("reason_codes") or [])
    if not enabled:
        result["head_reason_codes"].append("head_disabled_repair_only")
        return result

    strategy_type = str(strategy.get("type") or "")
    if strategy_type == "response_score_passthrough":
        score, channel = _score_from_response(response, strategy.get("score_channel_preference") or [])
        result["head_score"] = score
        result["head_features_used"] = [channel]
        result["head_reason_codes"].append(f"head_score_channel:{channel}")
        return result
    if strategy_type == "linear_feature_model":
        score, features_used, available = _linear_feature_score(response, strategy)
        if not available:
            result["head_enabled"] = False
            result["head_reason_codes"].append("trajectory_features_missing_repair_only")
            result["head_reason_codes"].append("head_features_missing_repair_only")
            return result
        result["head_score"] = score
        result["head_features_used"] = features_used
        result["head_reason_codes"].append("head_score_channel:linear_feature_model")
        return result
    if strategy_type == "tfidf_linear_model":
        score = _tfidf_linear_score(response, strategy)
        result["head_score"] = score
        result["head_features_used"] = ["response_text", "support_evidence_text"]
        result["head_reason_codes"].append("head_score_channel:tfidf_linear_model")
        return result
    if strategy_type == "rule_feature_model":
        score, features_used, available = _rule_feature_score(response, strategy)
        if not available:
            result["head_enabled"] = False
            result["head_reason_codes"].append("head_features_missing_repair_only")
            return result
        result["head_score"] = score
        result["head_features_used"] = features_used
        result["head_reason_codes"].append("head_score_channel:rule_feature_model")
        return result

    result["head_enabled"] = False
    result["head_reason_codes"].append(f"unsupported_head_strategy:{strategy_type}")
    return result


def _fallback(
    *,
    class_key: str,
    reason: str,
    registry_sha: Optional[str],
    entry: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    entry = entry or {}
    return {
        "head_id": entry.get("head_id"),
        "head_version": entry.get("version"),
        "head_enabled": False,
        "head_registry_sha256": registry_sha,
        "head_artifact_sha256": None,
        "head_score": None,
        "head_features_used": [],
        "head_reason_codes": [reason, f"class_key:{class_key}"],
    }


def _load_registry() -> tuple[Optional[dict[str, Any]], Optional[str], Optional[str]]:
    path = registry_path()
    return _load_json_cached(path, "registry")


def _registry_entry(registry: Mapping[str, Any], class_key: str) -> Mapping[str, Any]:
    entries = registry.get("runtime_head_registry")
    if not isinstance(entries, Mapping):
        return {}
    entry = entries.get(class_key)
    return entry if isinstance(entry, Mapping) else {}


def _load_head_artifact(entry: Mapping[str, Any]) -> tuple[Optional[dict[str, Any]], Optional[str], Optional[str]]:
    raw_path = entry.get("artifact_path")
    if not raw_path:
        return None, None, "missing_artifact_path"
    path = Path(str(raw_path))
    if not path.is_absolute():
        path = _ROOT / path
    artifact, sha, error = _load_json_cached(path, "head")
    if artifact is None:
        return None, sha, error
    expected_sha = entry.get("artifact_sha256")
    if expected_sha and sha != expected_sha:
        return None, sha, "artifact_checksum_mismatch"
    if artifact.get("class_key") != entry.get("class_key"):
        return None, sha, "artifact_class_mismatch"
    if artifact.get("head_id") != entry.get("head_id"):
        return None, sha, "artifact_head_mismatch"
    return artifact, sha, None


def _load_json_cached(
    path: Path,
    cache_name: str,
) -> tuple[Optional[dict[str, Any]], Optional[str], Optional[str]]:
    global _REGISTRY_CACHE
    now = time.monotonic()
    cache_key: Any = cache_name if cache_name == "registry" else path
    with _LOCK:
        if cache_name == "registry":
            cached_path, _cached_mtime, cached_payload, cached_sha = _REGISTRY_CACHE
            if (
                cached_path == path
                and cached_payload is not None
                and now - _CACHE_LAST_CHECK.get(cache_key, 0.0) < _CACHE_TTL_SECONDS
            ):
                return cached_payload, cached_sha, None
        else:
            cached = _HEAD_CACHE.get(path)
            if cached is not None and now - _CACHE_LAST_CHECK.get(cache_key, 0.0) < _CACHE_TTL_SECONDS:
                return cached[1], cached[2], None
    if not path.exists():
        return None, None, f"missing:{path}"
    try:
        stat = path.stat()
    except OSError as exc:
        return None, None, f"stat_failed:{exc}"
    with _LOCK:
        if cache_name == "registry":
            cached_path, cached_mtime, cached_payload, cached_sha = _REGISTRY_CACHE
            if cached_path == path and cached_mtime == stat.st_mtime and cached_payload is not None:
                return cached_payload, cached_sha, None
        else:
            cached = _HEAD_CACHE.get(path)
            if cached is not None and cached[0] == stat.st_mtime:
                return cached[1], cached[2], None
        try:
            raw = path.read_bytes()
            payload = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            return None, None, f"load_failed:{exc}"
        sha = hashlib.sha256(raw).hexdigest()
        if cache_name == "registry":
            _REGISTRY_CACHE = (path, stat.st_mtime, payload, sha)
        else:
            _HEAD_CACHE[path] = (stat.st_mtime, payload, sha)
        _CACHE_LAST_CHECK[cache_key] = now
        return payload, sha, None


def _score_from_response(response: Any, preferred_channels: list[Any]) -> tuple[float, str]:
    scores = response.scores
    for channel in preferred_channels:
        name = str(channel)
        if hasattr(scores, name) and getattr(scores, name) is not None:
            return float(getattr(scores, name)), name
    if getattr(scores, "groundedness_v2", None) is not None:
        return float(scores.groundedness_v2), "groundedness_v2"
    return float(scores.primary_score), str(scores.primary_name or "primary_score")


def _linear_feature_score(response: Any, strategy: Mapping[str, Any]) -> tuple[float, list[str], bool]:
    feature_names = [str(name) for name in strategy.get("feature_names") or []]
    means = _compiled_float_map(strategy, "means")
    scales = _compiled_float_map(strategy, "scales", default_for_zero=1.0)
    weights = _compiled_float_map(strategy, "weights")
    features, explicit = _features_from_response(response)
    if bool(strategy.get("requires_explicit_features")) and not explicit:
        return 0.0, [], False
    raw = float(strategy.get("intercept") or 0.0)
    used: list[str] = []
    for name in feature_names:
        value = float(features.get(name, 0.0) or 0.0)
        raw += weights.get(name, 0.0) * ((value - means.get(name, 0.0)) / scales.get(name, 1.0))
        used.append(name)
    if str(strategy.get("output") or "") == "sigmoid":
        raw = 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, raw))))
    return float(raw), used, True


def _features_from_response(response: Any) -> tuple[dict[str, float], bool]:
    for attr in ("runtime_head_features", "trajectory_features", "head_features"):
        payload = getattr(response, attr, None)
        if isinstance(payload, Mapping):
            return {str(k): float(v) for k, v in payload.items() if _is_number(v)}, True
    payload = getattr(response, "metadata", None)
    if isinstance(payload, Mapping):
        nested = payload.get("trajectory_features") or payload.get("runtime_head_features")
        if isinstance(nested, Mapping):
            return {str(k): float(v) for k, v in nested.items() if _is_number(v)}, True
    scores = getattr(response, "scores", None)
    if scores is None:
        return {}, False
    return {
        "reverse_context": float(getattr(scores, "primary_score", 0.0) or 0.0),
        "groundedness_v2": float(getattr(scores, "groundedness_v2", getattr(scores, "primary_score", 0.0)) or 0.0),
        "consensus_hardened": float(getattr(scores, "groundedness_v2", getattr(scores, "primary_score", 0.0)) or 0.0),
        "triangular": float(getattr(scores, "groundedness_v2", getattr(scores, "primary_score", 0.0)) or 0.0),
    }, False


def _is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _rule_feature_score(response: Any, strategy: Mapping[str, Any]) -> tuple[float, list[str], bool]:
    features, explicit = _features_from_response(response)
    if bool(strategy.get("requires_explicit_features")) and not explicit:
        return 0.0, [], False
    rule = str(strategy.get("rule") or "")
    if rule == "factoid_atom_rule_head":
        score = (
            0.55 * features.get("numeric_coverage", 1.0)
            + 0.30 * features.get("literal_coverage", 0.0)
            + 0.15 * features.get("token_bottom10", 0.0)
        )
    elif rule == "claim_decomposition_rule_head":
        score = (
            0.55 * (1.0 - features.get("unsupported_claim_fraction", 0.0))
            + 0.30 * features.get("min_claim_coverage", 1.0)
            + 0.15 * features.get("support_unit_label_mean", 1.0)
        )
    elif rule == "code_symbol_rule_head":
        identifier = features.get("identifier_coverage", 1.0)
        score = (
            0.40 * identifier
            + 0.30 * features.get("api_symbol_match", identifier)
            + 0.15 * features.get("trajectory_order_match", 1.0)
            + 0.15 * features.get("test_outcome_match", 1.0)
        )
    elif rule == "structured_cell_rule_head":
        schema = min(features.get("schema_match", 0.5), features.get("schema_alias_match", features.get("schema_match", 0.5)))
        numeric = min(
            features.get("numeric_coverage", 1.0),
            features.get("numeric_tolerance_match", features.get("numeric_coverage", 1.0)),
        )
        score = (
            0.18 * schema
            + 0.18 * numeric
            + 0.14 * features.get("coverage_label_mean", features.get("context_coverage_ratio", 0.0))
            + 0.14 * features.get("row_alignment", schema)
            + 0.14 * features.get("column_alignment", schema)
            + 0.12 * features.get("value_alignment", numeric)
            + 0.05 * features.get("unit_alignment", 1.0)
            + 0.03 * features.get("date_alignment", 1.0)
            + 0.02 * features.get("cell_provenance_match", 0.0)
        )
    else:
        score = sum(features.get(str(name), 0.0) for name in strategy.get("feature_names") or []) / max(
            1, len(strategy.get("feature_names") or [])
        )
    return max(0.0, min(1.0, float(score))), [str(name) for name in strategy.get("feature_names") or []], True


def _tfidf_linear_score(response: Any, strategy: Mapping[str, Any]) -> float:
    weights = _compiled_float_map(strategy, "weights")
    idf = _compiled_float_map(strategy, "idf")
    text = _head_text(response)
    terms = _tfidf_terms(text)
    if not terms:
        raw = float(strategy.get("intercept") or 0.0)
    else:
        counts: dict[str, float] = {}
        for term in terms:
            if term in weights:
                counts[term] = counts.get(term, 0.0) + 1.0
        for left, right in zip(terms, terms[1:]):
            term = f"{left} {right}"
            if term in weights:
                counts[term] = counts.get(term, 0.0) + 1.0
        tfidf = {term: count * idf.get(term, 1.0) for term, count in counts.items()}
        norm = math.sqrt(sum(value * value for value in tfidf.values())) or 1.0
        raw = float(strategy.get("intercept") or 0.0) + sum(
            (value / norm) * weights.get(term, 0.0) for term, value in tfidf.items()
        )
    if str(strategy.get("output") or "") == "sigmoid":
        return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, raw))))
    return raw


def _tfidf_terms(text: str) -> list[str]:
    return [term.lower() for term in re.findall(r"(?u)\b\w\w+\b", text)]


def _head_text(response: Any) -> str:
    response_text = getattr(response, "response_text", None)
    if response_text is None:
        response_text = " ".join(str(getattr(token, "token", "")) for token in getattr(response, "response_tokens", []) or [])
    support_text = "\n".join(str(getattr(unit, "text", "")) for unit in getattr(response, "support_units", []) or [])
    return f"{response_text}\n[TRACE_EVIDENCE]\n{support_text}"


def _compiled_float_map(
    strategy: Mapping[str, Any],
    key: str,
    *,
    default_for_zero: float | None = None,
) -> dict[str, float]:
    cache_key = f"_{key}_float"
    if isinstance(strategy, dict) and isinstance(strategy.get(cache_key), dict):
        return strategy[cache_key]
    values = {
        str(k): (default_for_zero if default_for_zero is not None and float(v) == 0.0 else float(v))
        for k, v in (strategy.get(key) or {}).items()
    }
    if isinstance(strategy, dict):
        strategy[cache_key] = values
    return values
