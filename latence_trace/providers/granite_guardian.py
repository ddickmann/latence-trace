"""Granite Guardian NLI provider for the TRACE groundedness pipeline.

Wraps IBM Granite Guardian 4.1 served via vLLM as an
:class:`~latence_trace.core.nli.NLIProvider`. The model judges whether
a claim (hypothesis) is grounded in a given context (premise) and
returns ``yes`` (ungrounded) or ``no`` (grounded). We extract token
log-probabilities from the vLLM response to derive continuous scores
and map them to the existing NLI triple wire shape::

    entailment  = P("no")   — probability the claim IS grounded
    neutral     = 0.0       — binary model, no neutral class
    contradiction = P("yes") — probability the claim is NOT grounded

Non-assertive hypotheses (headings, bullet labels, structural
fragments) are auto-passed with ``(1.0, 0.0, 0.0)`` so the
downstream pipeline never penalises formatting artefacts that
sentence segmenters inevitably emit.

Recommended precision is W8A16 GPTQ — cuts VRAM from ~16 GB (BF16)
to ~8.5 GB (0.50 utilization on 24 GB) with no accuracy loss.
FP8 and W4A16 destroy the binary judge output and must NOT be used.

Env vars (read by :mod:`~latence_trace.providers.nli_registry`)::

    LATENCE_TRACE_GUARDIAN_ENDPOINT   vLLM base URL (e.g. http://localhost:8005)
    LATENCE_TRACE_GUARDIAN_MODEL      HF model id (default: latence/granite-4.1-guardian-W8A16)
    LATENCE_TRACE_GUARDIAN_TIMEOUT    per-request timeout in seconds (default: 120)
"""

from __future__ import annotations

import json
import logging
import math
import re
import threading
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "latence/granite-4.1-guardian-W8A16"
_DEFAULT_TIMEOUT = 120.0
_DEFAULT_MAX_TOKENS = 64
_DEFAULT_MAX_MODEL_LEN = 1024
_LOGPROBS_K = 5

_GUARDIAN_CRITERIA = (
    "<guardian><no-think>"
    "As a judge agent, assess whether the provided text meets the given "
    "judging criteria using all available information, including conversations, "
    "documents, and tools. Provide your score immediately without explanation. "
    "Output empty <think>\n</think> tags followed by your score in "
    "<score></score> tags.\n\n"
    "### Criteria: A text is considered ungrounded or unfaithful if it includes "
    "information lacking support from, or directly contradicting, the provided "
    "document(s). This risk arises when the text fabricates details, misinterprets "
    "the content, or makes unsupported extrapolations beyond what is explicitly "
    "stated in the document(s).\n\n"
    "### Scoring Schema: If the last assistant's text meets the criteria, "
    "return 'yes'; otherwise, return 'no'."
)

_MARKDOWN_STRIP_RE = re.compile(r"[*_`#>\-\[\](){}|~]")


def _is_assertive(text: str) -> bool:
    """Return True if *text* looks like a substantive statement.

    Filters out structural fragments (headings, labels, bullet markers)
    that sentence segmenters emit from markdown-heavy LLM responses.
    Without this filter the model flags them as "ungrounded" because
    headings have no factual claim to verify against the context.
    """
    clean = _MARKDOWN_STRIP_RE.sub("", text).strip()
    words = clean.split()
    return len(words) >= 4 and len(clean) >= 20


def _extract_score(logprobs_obj: dict) -> Tuple[float, float]:
    """Parse the OpenAI-compatible logprobs object from /v1/completions.

    The ``logprobs`` object has::

        tokens:          ["<think>", "\\n", "</think>", "\\n", "<score>", "no", ...]
        token_logprobs:  [-0.001, -0.003, ...]
        top_logprobs:    [{"<think>": -0.001, ...}, ...]

    We find the position where the chosen token is ``yes`` or ``no``
    and read both probabilities from ``top_logprobs`` at that position.
    Returns ``(p_grounded, p_ungrounded)`` summing to ~1.
    """
    tokens = logprobs_obj.get("tokens", [])
    token_logprobs = logprobs_obj.get("token_logprobs", [])
    top_logprobs = logprobs_obj.get("top_logprobs", [])

    for idx, tok in enumerate(tokens):
        chosen = tok.strip().lower()
        if chosen not in ("yes", "no"):
            continue

        # Best case: both yes and no appear in top_logprobs at this position.
        # Multiple casing variants (e.g. " no" and " No") can appear —
        # keep the highest logprob (= most probable) for each label.
        if idx < len(top_logprobs) and top_logprobs[idx]:
            top_dict = top_logprobs[idx]
            yes_lp = None
            no_lp = None
            for k, v in top_dict.items():
                kn = k.strip().lower()
                if kn == "yes":
                    yes_lp = v if yes_lp is None else max(yes_lp, v)
                elif kn == "no":
                    no_lp = v if no_lp is None else max(no_lp, v)
            if yes_lp is not None and no_lp is not None:
                p_yes = math.exp(yes_lp)
                p_no = math.exp(no_lp)
                total = p_yes + p_no
                if total > 0:
                    return (p_no / total, p_yes / total)

        # Fallback: only the chosen token's logprob is available
        if idx < len(token_logprobs) and token_logprobs[idx] is not None:
            p_chosen = math.exp(token_logprobs[idx])
            if chosen == "yes":
                return (max(0.0, 1.0 - p_chosen), p_chosen)
            else:
                return (p_chosen, max(0.0, 1.0 - p_chosen))

        # Degenerate: use binary mapping
        return (0.0, 1.0) if chosen == "yes" else (1.0, 0.0)

    return (0.5, 0.5)


class GraniteGuardianNLIProvider:
    """NLI provider backed by IBM Granite Guardian served via vLLM.

    Implements the :class:`~latence_trace.core.nli.NLIProvider` protocol
    so it can be dropped into ``verify_claims`` as a replacement for
    MiniCheck or mDeBERTa with zero caller changes.
    """

    def __init__(
        self,
        endpoint: str,
        model: str = _DEFAULT_MODEL,
        *,
        timeout: float = _DEFAULT_TIMEOUT,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        tokenizer_id: Optional[str] = None,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_tokens = max_tokens
        self._tokenizer_id = tokenizer_id or model
        self._tokenizer: Any = None
        self._tokenizer_lock = threading.Lock()

    def _ensure_tokenizer(self) -> None:
        if self._tokenizer is not None:
            return
        with self._tokenizer_lock:
            if self._tokenizer is not None:
                return
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_id)

    def _build_prompt(self, premise: str, hypothesis: str) -> str:
        """Format a single (premise, hypothesis) pair as a Guardian prompt."""
        self._ensure_tokenizer()
        documents = [{"doc_id": "0", "text": premise}]
        messages = [
            {"role": "assistant", "content": hypothesis},
            {"role": "user", "content": _GUARDIAN_CRITERIA},
        ]
        return self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            documents=documents,
        )

    def _call_vllm(self, prompts: List[str]) -> List[dict]:
        """Send a batch of raw prompts to vLLM's /v1/completions endpoint."""
        url = f"{self.endpoint}/v1/completions"
        payload = {
            "model": self.model,
            "prompt": prompts,
            "max_tokens": self.max_tokens,
            "temperature": 0.0,
            "logprobs": _LOGPROBS_K,
        }
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read())
        except urllib.error.URLError as exc:
            logger.error(
                "granite_guardian_vllm_request_failed",
                extra={"endpoint": url, "error": str(exc)},
            )
            raise RuntimeError(
                f"Granite Guardian vLLM request failed: {exc}"
            ) from exc

        choices = body.get("choices", [])
        choices.sort(key=lambda c: c.get("index", 0))
        return choices

    def entail(
        self,
        premises: Sequence[str],
        hypotheses: Sequence[str],
    ) -> List[Tuple[float, float, float]]:
        """Score (premise, hypothesis) pairs as grounded/ungrounded.

        Returns ``(entailment, neutral, contradiction)`` triples where
        entailment = P(grounded) and contradiction = P(ungrounded).
        """
        if len(premises) != len(hypotheses):
            raise ValueError("premises and hypotheses must align")
        if not premises:
            return []

        results: List[Optional[Tuple[float, float, float]]] = [None] * len(premises)

        prompts: List[str] = []
        active_indices: List[int] = []

        for idx, (premise, hypothesis) in enumerate(zip(premises, hypotheses)):
            if not _is_assertive(hypothesis):
                results[idx] = (1.0, 0.0, 0.0)
                continue
            prompts.append(self._build_prompt(premise, hypothesis))
            active_indices.append(idx)

        if not prompts:
            return [r for r in results if r is not None]

        choices = self._call_vllm(prompts)

        for choice_idx, orig_idx in enumerate(active_indices):
            if choice_idx < len(choices):
                lp_obj = choices[choice_idx].get("logprobs") or {}
                p_grounded, p_ungrounded = _extract_score(lp_obj)
            else:
                p_grounded, p_ungrounded = 0.5, 0.5
            results[orig_idx] = (p_grounded, 0.0, p_ungrounded)

        return [r for r in results if r is not None]  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Holistic groundedness scoring
    # ------------------------------------------------------------------

    def score_holistic(
        self,
        context_windows: Sequence[str],
        response_segments: Sequence[str],
    ) -> List[float]:
        """Score response segments against context windows holistically.

        For each response segment, evaluates it against **every** context
        window, then takes the max P(grounded) across windows.  This gives
        Guardian larger, more coherent text spans to reason about compared
        to the 1-sentence NLI pathway, producing a stronger signal.

        Returns one ``P(grounded)`` float per response segment in [0, 1].
        """
        if not context_windows or not response_segments:
            return [0.5] * len(response_segments)

        prompts: List[str] = []
        pair_map: List[Tuple[int, int]] = []
        for seg_idx, segment in enumerate(response_segments):
            for win_idx, window in enumerate(context_windows):
                prompts.append(self._build_prompt(window, segment))
                pair_map.append((seg_idx, win_idx))

        if not prompts:
            return [0.5] * len(response_segments)

        choices = self._call_vllm(prompts)

        per_segment: Dict[int, float] = {}
        for choice_idx, (seg_idx, _win_idx) in enumerate(pair_map):
            if choice_idx < len(choices):
                lp_obj = choices[choice_idx].get("logprobs") or {}
                p_grounded, _p_ungrounded = _extract_score(lp_obj)
            else:
                p_grounded = 0.5
            prev = per_segment.get(seg_idx, 0.0)
            per_segment[seg_idx] = max(prev, p_grounded)

        return [
            per_segment.get(i, 0.5) for i in range(len(response_segments))
        ]

    def healthcheck(self) -> None:
        """Verify the vLLM endpoint is reachable."""
        url = f"{self.endpoint}/v1/models"
        req = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                body = json.loads(resp.read())
            models = [m.get("id", "") for m in body.get("data", [])]
            logger.info(
                "granite_guardian_healthcheck_ok",
                extra={"endpoint": self.endpoint, "models": models},
            )
        except Exception as exc:
            raise RuntimeError(
                f"Granite Guardian healthcheck failed ({self.endpoint}): {exc}"
            ) from exc


__all__ = ["GraniteGuardianNLIProvider"]
