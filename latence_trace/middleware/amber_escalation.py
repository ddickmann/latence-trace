"""Auto-decide amber escalation (Phase 2 / `plans/trace-sota-auto-verifier`).

When the TRACE RAG lane returns ``band == "amber"`` and the caller has opted
into ``auto_decide``, this middleware resolves the amber band to a binary
``green`` / ``red`` verdict by calling a pinned LLM judge over the atomic-claim
/ evidence pairs the NLI diagnostics already produced.

Design contract:

* **One** judge call per amber response. Budget capped per-tenant at the
  Cloudflare worker (``api.latence.ai``) so a misconfigured client cannot run
  away with costs. This middleware only *enforces* the per-request latency and
  cost budget; it does not meter rolling usage.
* **Structured prompt, structured output.** The judge receives a compact,
  deterministic prompt (system + user) and must return
  ``{"verdict": "green" | "red", "reasoning": "..."}``. Any other output is
  treated as a soft failure and leaves the band at ``amber``.
* **Evidence-only.** The judge sees *only* the claim/evidence pairs the
  atomic-claim decomposer already produced plus the original query. It does
  not see support IDs, tenant identifiers, or any PII beyond what the caller
  already provided. This is a deliberate minimum-surface decision for the
  enterprise / finance / legal lane.
* **Providers.** OpenAI (``gpt-4o``/``gpt-4o-mini``) and Anthropic
  (``claude-3-5-sonnet``/``claude-3-5-haiku``). Selection is env-driven:
  ``LATENCE_TRACE_AUTO_DECIDE_PROVIDER=openai|anthropic|offline``. ``offline``
  is a deterministic heuristic used in tests and when no API key is present;
  it applies a score-based fallback so unit tests can assert wiring without
  an outbound network call.
* **Default off.** Auto-decide is opt-in per request (``auto_decide=True``)
  and per tenant (``VOYAGER_TRACE_AUTO_DECIDE_DEFAULT=1`` to flip the default
  on the hosted worker for a pilot tenant).

The middleware is deliberately small and stateless: it is safe to wire into
both the RunPod handler hot path and the FastAPI ``/v1/score/groundedness``
endpoint.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Literal, Mapping, Optional, Sequence

logger = logging.getLogger(__name__)


_DEFAULT_MODEL = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-haiku-20241022",
}
_DEFAULT_TIMEOUT_S = 6.0
_DEFAULT_MAX_CLAIMS = 8
_DEFAULT_MAX_EVIDENCE_CHARS = 600
_MAX_REASONING_CHARS = 600


AmberVerdict = Literal["green", "red", "amber"]


@dataclass(frozen=True)
class AmberEscalationConfig:
    """Resolved per-tenant config. Built once by the service layer and threaded
    into :func:`escalate` so unit tests can stub provider + model cleanly."""

    enabled: bool = False
    provider: Literal["openai", "anthropic", "offline"] = "offline"
    model: Optional[str] = None
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    timeout_s: float = _DEFAULT_TIMEOUT_S
    max_claims: int = _DEFAULT_MAX_CLAIMS
    max_evidence_chars: int = _DEFAULT_MAX_EVIDENCE_CHARS

    @classmethod
    def from_env(cls, override: Optional[Mapping[str, str]] = None) -> "AmberEscalationConfig":
        """Build the config from process env with optional overrides.

        Boolean env toggles accept ``1/0``, ``true/false``, ``yes/no``.
        """
        env = dict(os.environ)
        if override:
            env.update(override)

        enabled = _coerce_bool(env.get("LATENCE_TRACE_AUTO_DECIDE_ENABLED"), default=False)
        provider = str(env.get("LATENCE_TRACE_AUTO_DECIDE_PROVIDER") or "offline").lower()
        if provider not in {"openai", "anthropic", "offline"}:
            logger.warning("unknown auto_decide provider %s; falling back to offline", provider)
            provider = "offline"
        model = env.get("LATENCE_TRACE_AUTO_DECIDE_MODEL") or _DEFAULT_MODEL.get(provider)
        api_key_env = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "offline": None,
        }[provider]
        api_key = env.get(api_key_env) if api_key_env else None
        api_base = env.get("LATENCE_TRACE_AUTO_DECIDE_API_BASE")
        timeout_s = float(env.get("LATENCE_TRACE_AUTO_DECIDE_TIMEOUT_S") or _DEFAULT_TIMEOUT_S)
        max_claims = int(env.get("LATENCE_TRACE_AUTO_DECIDE_MAX_CLAIMS") or _DEFAULT_MAX_CLAIMS)
        max_evidence_chars = int(
            env.get("LATENCE_TRACE_AUTO_DECIDE_MAX_EVIDENCE_CHARS") or _DEFAULT_MAX_EVIDENCE_CHARS
        )

        if provider in ("openai", "anthropic") and enabled and not api_key:
            logger.warning(
                "auto_decide enabled for provider=%s but %s is unset; forcing offline",
                provider,
                api_key_env,
            )
            provider = "offline"

        return cls(
            enabled=enabled,
            provider=provider,  # type: ignore[arg-type]
            model=model,
            api_key=api_key,
            api_base=api_base,
            timeout_s=timeout_s,
            max_claims=max_claims,
            max_evidence_chars=max_evidence_chars,
        )


@dataclass(frozen=True)
class AmberJudgeVerdict:
    """Result of a single judge call."""

    verdict: AmberVerdict
    reasoning: str
    judge_provider: str
    judge_model: Optional[str]
    judge_latency_ms: float
    judge_cost_usd: Optional[float]
    error: Optional[str] = None


@dataclass(frozen=True)
class AmberEscalationPayload:
    """Compact judge payload derived from a scored response."""

    query: str
    response_text: str
    claims: List[Mapping[str, Any]] = field(default_factory=list)


def build_payload(
    *,
    query_text: str,
    response_text: str,
    nli_claims: Sequence[Mapping[str, Any]],
    config: AmberEscalationConfig,
) -> AmberEscalationPayload:
    """Turn a NLI diagnostics claim list into a judge payload.

    ``nli_claims`` elements are dicts with at least ``text`` + ``score`` plus
    optional ``premises``/``support_ids``/``atoms`` (the structure produced by
    ``latence_trace/api/models.py::GroundednessNLIClaim``). We keep only the
    claim text, entailment score, and the shortest-evidence premise per claim
    so the judge's prompt is bounded.
    """
    selected: List[Mapping[str, Any]] = []
    for claim in nli_claims[: max(1, int(config.max_claims))]:
        text = str(claim.get("text") or "").strip()
        if not text:
            continue
        score = claim.get("score")
        premises = claim.get("premises") or []
        atom_texts: List[str] = []
        for atom in claim.get("atoms") or []:
            atext = str(atom.get("text") or "").strip()
            if atext:
                atom_texts.append(atext)
            if len(atom_texts) >= 3:
                break
        evidence = _pick_shortest_non_empty(premises, limit=config.max_evidence_chars)
        selected.append(
            {
                "claim": text,
                "score": _safe_float(score),
                "entailment": _safe_float(claim.get("entailment")),
                "contradiction": _safe_float(claim.get("contradiction")),
                "atoms": atom_texts,
                "evidence": evidence,
            }
        )
    return AmberEscalationPayload(
        query=_clip(query_text, 400),
        response_text=_clip(response_text, 2000),
        claims=selected,
    )


def escalate(
    payload: AmberEscalationPayload,
    config: AmberEscalationConfig,
    *,
    fallback_score: Optional[float] = None,
) -> AmberJudgeVerdict:
    """Run the judge. Returns ``verdict="amber"`` if the judge fails.

    ``fallback_score`` is the original groundedness score; used only by the
    offline provider so tests have a deterministic binary decision.
    """
    if not config.enabled:
        return AmberJudgeVerdict(
            verdict="amber",
            reasoning="auto_decide_disabled",
            judge_provider=config.provider,
            judge_model=None,
            judge_latency_ms=0.0,
            judge_cost_usd=None,
        )
    if config.provider == "offline":
        return _offline_escalate(payload, config, fallback_score=fallback_score)
    if config.provider == "openai":
        return _openai_escalate(payload, config)
    if config.provider == "anthropic":
        return _anthropic_escalate(payload, config)
    return AmberJudgeVerdict(
        verdict="amber",
        reasoning="unknown_provider",
        judge_provider=config.provider,
        judge_model=config.model,
        judge_latency_ms=0.0,
        judge_cost_usd=None,
        error=f"unknown provider: {config.provider}",
    )


# ----------------------------------------------------------------------
# Internals
# ----------------------------------------------------------------------


def _build_prompt(payload: AmberEscalationPayload) -> tuple[str, str]:
    system = (
        "You are a strict hallucination auditor for an enterprise RAG pipeline. "
        "You receive a user question, the assistant's response, and a list of atomic claims "
        "the response makes together with the supporting evidence excerpts and NLI scores. "
        "Decide whether the response is FAITHFUL to the evidence (verdict=green) or contains "
        "at least one unsupported or contradicted claim (verdict=red). "
        "Be strict: any claim not clearly entailed by the evidence is a hallucination. "
        "Respond with a JSON object: {\"verdict\": \"green\"|\"red\", \"reasoning\": \"...\"}. "
        "Keep reasoning under 80 words."
    )
    user = {
        "question": payload.query,
        "response": payload.response_text,
        "claims": payload.claims,
    }
    return system, json.dumps(user, ensure_ascii=False)


def _offline_escalate(
    payload: AmberEscalationPayload,
    config: AmberEscalationConfig,
    *,
    fallback_score: Optional[float],
) -> AmberJudgeVerdict:
    """Deterministic heuristic: weigh per-claim contradictions vs entailments.

    Used in tests + as the CI fallback when no API key is wired. Not the
    shipping production path.
    """
    t0 = time.perf_counter()
    worst_contradict = 0.0
    mean_entail = 0.0
    count = 0
    for claim in payload.claims:
        c = claim.get("contradiction")
        e = claim.get("entailment")
        if isinstance(c, (int, float)):
            worst_contradict = max(worst_contradict, float(c))
        if isinstance(e, (int, float)):
            mean_entail += float(e)
            count += 1
    mean_entail = mean_entail / count if count else (fallback_score or 0.5)
    verdict: AmberVerdict
    reasoning: str
    if worst_contradict >= 0.5:
        verdict = "red"
        reasoning = f"offline heuristic: max contradiction={worst_contradict:.2f} >= 0.5"
    elif mean_entail >= 0.6:
        verdict = "green"
        reasoning = f"offline heuristic: mean entailment={mean_entail:.2f} >= 0.6"
    elif fallback_score is not None and fallback_score >= 0.55:
        verdict = "green"
        reasoning = f"offline heuristic: fallback score={fallback_score:.2f} >= 0.55"
    else:
        verdict = "red"
        reasoning = (
            f"offline heuristic: entailment={mean_entail:.2f}, "
            f"contradiction={worst_contradict:.2f}, fallback={fallback_score}"
        )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return AmberJudgeVerdict(
        verdict=verdict,
        reasoning=_clip(reasoning, _MAX_REASONING_CHARS),
        judge_provider="offline",
        judge_model="heuristic",
        judge_latency_ms=round(elapsed_ms, 2),
        judge_cost_usd=0.0,
    )


def _openai_escalate(
    payload: AmberEscalationPayload,
    config: AmberEscalationConfig,
) -> AmberJudgeVerdict:
    try:
        import httpx
    except ImportError as exc:
        return _fail("openai", config.model, f"httpx missing: {exc}")
    system, user = _build_prompt(payload)
    base = (config.api_base or "https://api.openai.com").rstrip("/")
    url = f"{base}/v1/chat/completions"
    body = {
        "model": config.model or _DEFAULT_MODEL["openai"],
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
        "max_tokens": 256,
    }
    headers = {
        "authorization": f"Bearer {config.api_key}",
        "content-type": "application/json",
    }
    t0 = time.perf_counter()
    try:
        with httpx.Client(timeout=config.timeout_s) as client:
            resp = client.post(url, json=body, headers=headers)
        resp.raise_for_status()
        obj = resp.json()
    except Exception as exc:  # pragma: no cover - network path
        return _fail("openai", config.model, str(exc), elapsed_ms=(time.perf_counter() - t0) * 1000)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    content = _first_choice_text(obj)
    parsed = _parse_verdict_json(content)
    usage = obj.get("usage") or {}
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    cost_usd = _openai_cost(config.model or "gpt-4o-mini", prompt_tokens, completion_tokens)
    if parsed is None:
        return _fail(
            "openai",
            config.model,
            f"unparseable judge output: {_clip(content, 200)!r}",
            elapsed_ms=elapsed_ms,
            cost_usd=cost_usd,
        )
    verdict, reasoning = parsed
    return AmberJudgeVerdict(
        verdict=verdict,
        reasoning=_clip(reasoning, _MAX_REASONING_CHARS),
        judge_provider="openai",
        judge_model=config.model,
        judge_latency_ms=round(elapsed_ms, 2),
        judge_cost_usd=cost_usd,
    )


def _anthropic_escalate(
    payload: AmberEscalationPayload,
    config: AmberEscalationConfig,
) -> AmberJudgeVerdict:
    try:
        import httpx
    except ImportError as exc:
        return _fail("anthropic", config.model, f"httpx missing: {exc}")
    system, user = _build_prompt(payload)
    base = (config.api_base or "https://api.anthropic.com").rstrip("/")
    url = f"{base}/v1/messages"
    body = {
        "model": config.model or _DEFAULT_MODEL["anthropic"],
        "system": system,
        "messages": [{"role": "user", "content": user}],
        "temperature": 0.0,
        "max_tokens": 256,
    }
    headers = {
        "x-api-key": config.api_key or "",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    t0 = time.perf_counter()
    try:
        with httpx.Client(timeout=config.timeout_s) as client:
            resp = client.post(url, json=body, headers=headers)
        resp.raise_for_status()
        obj = resp.json()
    except Exception as exc:  # pragma: no cover - network path
        return _fail(
            "anthropic", config.model, str(exc), elapsed_ms=(time.perf_counter() - t0) * 1000
        )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    text = ""
    for block in obj.get("content") or []:
        if isinstance(block, dict) and block.get("type") == "text":
            text += str(block.get("text") or "")
    parsed = _parse_verdict_json(text)
    usage = obj.get("usage") or {}
    input_tokens = int(usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    cost_usd = _anthropic_cost(config.model or _DEFAULT_MODEL["anthropic"], input_tokens, output_tokens)
    if parsed is None:
        return _fail(
            "anthropic",
            config.model,
            f"unparseable judge output: {_clip(text, 200)!r}",
            elapsed_ms=elapsed_ms,
            cost_usd=cost_usd,
        )
    verdict, reasoning = parsed
    return AmberJudgeVerdict(
        verdict=verdict,
        reasoning=_clip(reasoning, _MAX_REASONING_CHARS),
        judge_provider="anthropic",
        judge_model=config.model,
        judge_latency_ms=round(elapsed_ms, 2),
        judge_cost_usd=cost_usd,
    )


def _first_choice_text(obj: Mapping[str, Any]) -> str:
    choices = obj.get("choices") or []
    if not choices:
        return ""
    msg = (choices[0] or {}).get("message") or {}
    return str(msg.get("content") or "")


def _parse_verdict_json(raw: str) -> Optional[tuple[AmberVerdict, str]]:
    if not raw:
        return None
    body = raw.strip()
    # strip fenced code blocks if present
    if body.startswith("```"):
        body = body.strip("`")
        if body.lower().startswith("json"):
            body = body[4:]
    try:
        obj = json.loads(body)
    except json.JSONDecodeError:
        return None
    verdict = str(obj.get("verdict") or "").lower()
    reasoning = str(obj.get("reasoning") or "")
    if verdict not in ("green", "red"):
        return None
    return verdict, reasoning  # type: ignore[return-value]


def _openai_cost(model: str, prompt_tokens: int, completion_tokens: int) -> Optional[float]:
    # Approximate public per-million prices in USD. Updated 2026-04.
    table = {
        "gpt-4o-mini": (0.15, 0.60),
        "gpt-4o": (2.50, 10.00),
        "gpt-4.1-mini": (0.40, 1.60),
        "gpt-4.1": (2.00, 8.00),
        "o4-mini": (1.10, 4.40),
    }
    model_key = model.split(":")[0]
    prices = table.get(model_key)
    if not prices:
        return None
    ppm_prompt, ppm_completion = prices
    return round(
        (prompt_tokens / 1_000_000) * ppm_prompt + (completion_tokens / 1_000_000) * ppm_completion,
        6,
    )


def _anthropic_cost(model: str, input_tokens: int, output_tokens: int) -> Optional[float]:
    # Approximate public per-million prices in USD. Updated 2026-04.
    table = {
        "claude-3-5-haiku-20241022": (0.80, 4.00),
        "claude-3-5-haiku": (0.80, 4.00),
        "claude-3-5-sonnet-20241022": (3.00, 15.00),
        "claude-3-5-sonnet": (3.00, 15.00),
    }
    prices = table.get(model)
    if not prices:
        return None
    ppm_prompt, ppm_completion = prices
    return round(
        (input_tokens / 1_000_000) * ppm_prompt + (output_tokens / 1_000_000) * ppm_completion, 6
    )


def _fail(
    provider: str,
    model: Optional[str],
    msg: str,
    *,
    elapsed_ms: float = 0.0,
    cost_usd: Optional[float] = None,
) -> AmberJudgeVerdict:
    return AmberJudgeVerdict(
        verdict="amber",
        reasoning=_clip(msg, _MAX_REASONING_CHARS),
        judge_provider=provider,
        judge_model=model,
        judge_latency_ms=round(elapsed_ms, 2),
        judge_cost_usd=cost_usd,
        error=msg,
    )


def _pick_shortest_non_empty(premises: Iterable[Any], *, limit: int) -> Optional[str]:
    best: Optional[str] = None
    for p in premises or []:
        text = str(p).strip()
        if not text:
            continue
        if best is None or len(text) < len(best):
            best = text
    if best is None:
        return None
    return _clip(best, limit)


def _clip(text: str, limit: int) -> str:
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "\u2026"


def _safe_float(val: Any) -> Optional[float]:
    try:
        if val is None:
            return None
        return float(val)
    except (TypeError, ValueError):
        return None


def _coerce_bool(val: Any, *, default: bool = False) -> bool:
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    if s in {"1", "true", "yes", "on", "y"}:
        return True
    if s in {"0", "false", "no", "off", "n"}:
        return False
    return default


__all__ = [
    "AmberEscalationConfig",
    "AmberEscalationPayload",
    "AmberJudgeVerdict",
    "AmberVerdict",
    "build_payload",
    "escalate",
]
