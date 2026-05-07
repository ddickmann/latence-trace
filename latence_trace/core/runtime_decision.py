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
from latence_trace.core.context_trust import (
    blocked_threshold as context_trust_blocked_threshold,
)
from latence_trace.core.context_trust import (
    context_trust_enabled,
)
from latence_trace.core.context_trust import (
    suspicious_threshold as context_trust_suspicious_threshold,
)

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


_CALIBRATION_BANDS = frozenset({"green", "amber", "red"})


def _calibration_band(response: Any) -> Optional[str]:
    """Return the user-visible calibration band attached to ``response``.

    The calibration band is the band the per-class calibration bundle
    assigned to the fused groundedness signal (``groundedness_v2`` /
    ``primary_score``). It is the verdict every consumer surface
    (heatmap header, event log, SDK callers) reads as the canonical
    risk band, so the runtime decision MUST agree with it whenever it
    is set; otherwise the same response carries two contradictory band
    labels (e.g. amber on the live overlay, red in the event log).
    """

    raw_band = getattr(getattr(response, "scores", None), "risk_band", None)
    if raw_band is None:
        return None
    band = str(raw_band).strip().lower()
    return band if band in _CALIBRATION_BANDS else None


def _coerce_action_to_calibration(
    action: str,
    calibration_band: Optional[str],
    class_policy: Mapping[str, Any],
) -> str:
    """Force ``action`` to ``block`` when calibration says ``red``.

    Calibration is the user-visible verdict that drives the heatmap
    header, the event log row, the SDK ``risk_band`` field, and the
    in-product red/amber/green pill. When calibration says ``red`` for
    a response, we never let a head-driven ``allow`` or ``auto_repair``
    survive — the live overlay would otherwise show a benign action on
    a response the user already sees as red, which is the exact
    contradiction the German Kafka case surfaced.

    The coercion is intentionally **one-way**: head/policy verdicts are
    untouched on green and amber so the existing safety overrides
    (structured literal-mismatch repair, head feature gating, etc.)
    keep firing as-designed. Green is left to the head precisely
    because head-driven logic — the structured guard, the missing-
    feature gate — exists to demote allow to auto_repair when the
    head detects risks calibration may have missed.

    When ``block_disabled=true`` for the class we fall back to
    ``auto_repair`` to honour the operator's explicit "do not block"
    contract rather than forcing an action they turned off.
    """

    if calibration_band != "red":
        return action
    if bool(class_policy.get("block_disabled", False)):
        return "auto_repair"
    return "block"


def _support_evidence(response: Any, *, limit: int = 3) -> list[dict[str, Any]]:
    units = list(getattr(response, "support_units", []) or [])
    units.sort(
        key=lambda unit: (
            float(getattr(unit, "context_trust_score", 0.0) or 0.0),
            float(getattr(unit, "coverage_score", 0.0) or 0.0),
            float(getattr(unit, "usage_confidence", 0.0) or 0.0),
        ),
        reverse=True,
    )
    evidence: list[dict[str, Any]] = []
    for unit in units[:limit]:
        labels = []
        for label in list(getattr(unit, "context_trust_labels", []) or []):
            labels.append(str(getattr(label, "label", label)))
        context_trust_score = getattr(unit, "context_trust_score", None)
        evidence.append(
            {
                "index": int(getattr(unit, "index", len(evidence))),
                "support_id": str(getattr(unit, "support_id", "")),
                "text": str(getattr(unit, "text", ""))[:800],
                "coverage_score": float(getattr(unit, "coverage_score", 0.0) or 0.0),
                "usage_state": str(getattr(getattr(unit, "usage_state", None), "value", getattr(unit, "usage_state", ""))),
                "context_trust_state": getattr(unit, "context_trust_state", None),
                "context_trust_score": (
                    None
                    if context_trust_score is None
                    else float(context_trust_score or 0.0)
                ),
                "context_trust_labels": labels,
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
    context_codes = _context_trust_reason_codes(response)
    codes.update(context_codes)
    return sorted(codes)


def _context_trust_decision_enabled() -> bool:
    raw = os.environ.get("LATENCE_TRACE_CONTEXT_TRUST_DECISION_ENABLED", "").strip().lower()
    if not raw:
        return context_trust_enabled()
    return raw not in _FALSE_VALUES


def _context_trust_max_risk(response: Any) -> float | None:
    scores = getattr(response, "scores", None)
    value = getattr(scores, "context_trust_max_risk", None)
    if value is None:
        diagnostics = getattr(response, "context_trust_diagnostics", None)
        value = getattr(diagnostics, "max_risk", None)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _context_trust_reason_codes(response: Any) -> list[str]:
    if not _context_trust_decision_enabled():
        return []
    max_risk = _context_trust_max_risk(response)
    if max_risk is None:
        return []
    if max_risk >= context_trust_blocked_threshold():
        return ["context_trust_blocked"]
    if max_risk >= context_trust_suspicious_threshold():
        return ["context_trust_suspicious"]
    return []


def _apply_context_trust_action(
    action: str,
    response: Any,
    *,
    block_disabled: bool,
) -> str:
    if not _context_trust_decision_enabled():
        return action
    max_risk = _context_trust_max_risk(response)
    if max_risk is None:
        return action
    if max_risk >= context_trust_blocked_threshold():
        return "auto_repair" if block_disabled else "block"
    if max_risk >= context_trust_suspicious_threshold() and action == "allow":
        return "auto_repair"
    return action


def _structured_literal_guard_reason(response: Any, class_key: str) -> Optional[str]:
    if class_key != "rag.structured":
        return None
    for warning in list(getattr(response, "warnings", []) or []):
        text = str(warning).strip().lower()
        if not text.startswith("literal_mismatch:"):
            continue
        if "measurement=" in text or "unit=" in text:
            return "structured_measurement_literal_mismatch_repair_only"
        if "currency=" in text:
            return "structured_currency_literal_mismatch_repair_only"
        if "number=" in text:
            return "structured_numeric_literal_mismatch_repair_only"
    return None


def build_runtime_decision(response: Any) -> Optional[dict[str, Any]]:
    if not enabled():
        return None
    policy, policy_sha, error = _load_policy()
    if policy is None:
        score, score_channel = _score_from_response(response)
        action = _apply_context_trust_action(
            "auto_repair",
            response,
            block_disabled=False,
        )
        # Even on the rollback-safe fallback path the band is the
        # canonical calibration band; coerce action so the live UI and
        # event log never disagree when the policy file is missing in
        # rare ops scenarios.
        calibration_band = _calibration_band(response)
        action = _coerce_action_to_calibration(action, calibration_band, {})
        return {
            "policy_version": "unavailable",
            "policy_sha256": None,
            "class_key": _class_key(response),
            "score": score,
            "score_channel": score_channel,
            "band": calibration_band if calibration_band is not None else str(response.scores.risk_band or "unknown"),
            "action": action,
            "evidence": _support_evidence(response),
            "unsupported_spans": _unsupported_spans(response),
            "reason_codes": sorted(
                {error or "policy_unavailable", *_context_trust_reason_codes(response)}
            ),
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
    reason_codes = _reason_codes(response)
    structured_guard_reason = _structured_literal_guard_reason(response, class_key)
    if action == "allow" and structured_guard_reason is not None:
        action = "auto_repair"
        reason_codes.append(structured_guard_reason)
    action = _apply_context_trust_action(
        action,
        response,
        block_disabled=bool(class_policy.get("block_disabled", False)),
    )
    # Final stage: align action+band with the per-class calibration band
    # so the user-visible verdict (event log, heatmap, SDK callers) and
    # the runtime decision (live overlay, action label) never disagree
    # on the same response. The head score and head decision still drive
    # the final verdict for green/amber, but a red calibration band is
    # treated as authoritative — calibration was trained on the
    # per-(class, language) ground-truth distribution and is the
    # canonical "is this answer ungrounded?" signal.
    calibration_band = _calibration_band(response)
    pre_calibration_action = action
    action = _coerce_action_to_calibration(action, calibration_band, class_policy)
    if calibration_band is not None and action != pre_calibration_action:
        reason_codes.append(f"calibration_band_coercion:{calibration_band}")
    decision_band = calibration_band if calibration_band is not None else _band_for_action(action)
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
        "reason_codes": sorted(set(reason_codes)),
        "allow_disabled": bool(class_policy.get("allow_disabled", False)),
        "block_disabled": bool(class_policy.get("block_disabled", False)),
        "allow_threshold": float(class_policy.get("allow_threshold", 1.0)),
        "block_threshold": float(class_policy.get("block_threshold", 0.0)),
        "rollback_safe": True,
    }
