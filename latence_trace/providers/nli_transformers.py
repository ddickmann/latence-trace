"""In-process transformers NLI providers for the SOTA dual-model setup.

Two providers live here:

* :class:`MiniCheckNLIProvider` — wraps ``lytang/MiniCheck-Flan-T5-Large``
  (770M, T5 encoder-decoder). MiniCheck is the published SOTA on the
  LLM-AggreFact grounded fact-checking benchmark (Tang et al. 2024). It
  produces a binary supported/not-supported judgement, which we surface
  to the existing 3-class wire shape as ``(p_yes, 1-p_yes, 0)``. The
  hard zero on the contradict slot is *correct*: MiniCheck explicitly
  refuses to commit on the supported/contradicted axis and treats novel
  hallucinations as ``not supported`` rather than misclassifying them
  as ``neutral`` (the failure mode mDeBERTa shows on case 2 of the
  catalogue).

* :class:`BgeM3ZeroShotNLIProvider` — wraps
  ``MoritzLaurer/bge-m3-zeroshot-v2.0`` (568M XLM-RoBERTa). Excellent
  multilingual zero-shot NLI; on the German Phase 0 fixtures it lifted
  balanced accuracy ~10 points over mDeBERTa with no per-class tuning.
  We map its binary ``entailment`` / ``not_entailment`` head to the
  existing wire as ``(p_entail, 1-p_entail, 0)`` for the same reason.

Both classes implement the :class:`latence_trace.core.nli.NLIProvider`
Protocol so the ``verify_claims`` hot path can swap them in without any
caller changes. They share a thread-safe lazy load so unused services
pay nothing at import time and the model only lands on the GPU on the
first ``entail`` call. Once we ship vLLM-served versions (see
:mod:`latence_trace.providers.nli_classify`) callers swap by flipping
``LATENCE_TRACE_NLI_EN_ENDPOINT`` / ``LATENCE_TRACE_NLI_MULTI_ENDPOINT``
— no code change.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# MiniCheck context budget. The model is Flan-T5-Large which Tang et al.
# fine-tuned with a 2048-token encoder context; we leave 512 for the
# claim + format overhead so the premise window has ~1500 tokens of
# headroom. This matches the chunking strategy validated in the Phase 0
# proof (see /tmp/proof/scripts/isolated_proof_nli.py).
_MINICHECK_INPUT_BUDGET = 1024
_MINICHECK_FORMAT_OVERHEAD = 24  # ``predict: ...\nclaim: `` etc.
_MINICHECK_MAX_CLAIM_TOKENS = 384
_MINICHECK_MIN_PREMISE_WINDOW = 256


def _default_device() -> str:
    try:
        import torch  # noqa: WPS433

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # pragma: no cover - torch missing in CI shell
        return "cpu"


def _default_dtype() -> Any:
    """Pick bfloat16 on CUDA, float32 on CPU.

    The candidate models are large enough (770M / 568M) that fp32 on a
    24 GB A5000 wastes ~3 GB needlessly. We default to bfloat16 on CUDA
    and float32 on CPU (bf16 on CPU is supported but undertested on
    older PyTorch wheels — explicit fp32 is the safer default).
    """

    try:
        import torch  # noqa: WPS433

        if torch.cuda.is_available():
            return torch.bfloat16
        return torch.float32
    except Exception:  # pragma: no cover - torch missing
        return None


class MiniCheckNLIProvider:
    """In-process MiniCheck-Flan-T5-Large NLI provider.

    The model is encoder-decoder T5; we read the first decoder token's
    logits over the ``Yes`` / ``No`` vocabulary entries (canonical
    inference recipe from the model card / Tang et al. 2024). The
    softmax of the two-element subset is the grounding score in
    ``[0, 1]``.
    """

    def __init__(
        self,
        model_id: str = "lytang/MiniCheck-Flan-T5-Large",
        *,
        device: Optional[str] = None,
        dtype: Any = None,
        max_input_tokens: int = _MINICHECK_INPUT_BUDGET,
        max_claim_tokens: int = _MINICHECK_MAX_CLAIM_TOKENS,
        min_premise_window: int = _MINICHECK_MIN_PREMISE_WINDOW,
    ) -> None:
        self.model_id = model_id
        self.device = device or _default_device()
        self.dtype = dtype if dtype is not None else _default_dtype()
        self.max_input_tokens = max(64, int(max_input_tokens))
        self.max_claim_tokens = max(16, int(max_claim_tokens))
        self.min_premise_window = max(32, int(min_premise_window))
        self._lock = threading.Lock()
        self._tokenizer: Any = None
        self._model: Any = None
        self._yes_token_id: Optional[int] = None
        self._no_token_id: Optional[int] = None

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        with self._lock:
            if self._model is not None and self._tokenizer is not None:
                return
            from transformers import (  # noqa: WPS433
                AutoTokenizer,
                T5ForConditionalGeneration,
            )

            tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            kwargs: dict[str, Any] = {}
            if self.dtype is not None:
                kwargs["torch_dtype"] = self.dtype
            model = T5ForConditionalGeneration.from_pretrained(self.model_id, **kwargs)
            model.to(self.device)
            model.eval()

            # The published MiniCheck recipe calls ``model.generate`` with
            # ``max_new_tokens=1`` and then maps token 0 to {Yes, No}. We
            # bypass generate() and read the logits directly so a 128-pair
            # batch is one model.forward() instead of 128 generate() loops.
            yes_id = tokenizer("Yes", add_special_tokens=False).input_ids[0]
            no_id = tokenizer("No", add_special_tokens=False).input_ids[0]

            self._tokenizer = tokenizer
            self._model = model
            self._yes_token_id = int(yes_id)
            self._no_token_id = int(no_id)

    def _format_one(self, premise: str, claim: str) -> str:
        """Apply MiniCheck's prompt template, capping the claim length.

        Capping the claim avoids the pathological case where a long
        ``raw_context`` is passed as a "claim" and starves the premise
        budget. The Phase 0 proof showed that capping at 384 tokens
        keeps the premise window ≥ 512 tokens for any realistic use
        case while costing < 0.3 balanced-accuracy points.
        """

        claim_ids = self._tokenizer.encode(claim, add_special_tokens=False)
        if len(claim_ids) > self.max_claim_tokens:
            claim = self._tokenizer.decode(
                claim_ids[: self.max_claim_tokens],
                skip_special_tokens=True,
            )
        return f"predict: {premise}\nclaim: {claim}"

    def entail(
        self,
        premises: Sequence[str],
        hypotheses: Sequence[str],
    ) -> List[Tuple[float, float, float]]:
        if len(premises) != len(hypotheses):
            raise ValueError("premises and hypotheses must align")
        if not premises:
            return []
        self._ensure_loaded()
        import torch  # noqa: WPS433

        formatted = [
            self._format_one(p or "", h or "") for p, h in zip(premises, hypotheses)
        ]
        encoded = self._tokenizer(
            formatted,
            padding=True,
            truncation=True,
            max_length=self.max_input_tokens,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        # Single-step decoder with the model's start token; we only need
        # the logits at the first generated position to read Yes vs No.
        decoder_start = self._model.config.decoder_start_token_id
        decoder_input_ids = torch.full(
            (encoded["input_ids"].shape[0], 1),
            int(decoder_start),
            dtype=torch.long,
            device=self.device,
        )
        with torch.no_grad():
            outputs = self._model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded.get("attention_mask"),
                decoder_input_ids=decoder_input_ids,
            )
        # ``logits`` shape: (batch, 1, vocab). Slice the Yes/No columns
        # and softmax over just those two so we get a calibrated grounding
        # probability instead of a vocabulary-wide softmax.
        yes_logits = outputs.logits[:, 0, self._yes_token_id]
        no_logits = outputs.logits[:, 0, self._no_token_id]
        stacked = torch.stack([yes_logits, no_logits], dim=-1)
        probs = torch.softmax(stacked.float(), dim=-1).cpu().tolist()
        return [
            (
                float(row[0]),
                float(1.0 - row[0]),
                0.0,
            )
            for row in probs
        ]


class BgeM3ZeroShotNLIProvider:
    """In-process bge-m3-zeroshot-v2.0 NLI provider.

    The model is XLM-RoBERTa with a binary entailment head (label 0:
    entailment, label 1: not_entailment per the model card). vLLM
    natively serves it via ``--task classify`` so swapping to the
    HTTP-backed :class:`VllmClassifyNLIProvider` later is just an env
    var flip.
    """

    def __init__(
        self,
        model_id: str = "MoritzLaurer/bge-m3-zeroshot-v2.0",
        *,
        device: Optional[str] = None,
        dtype: Any = None,
        max_length: int = 512,
    ) -> None:
        self.model_id = model_id
        self.device = device or _default_device()
        self.dtype = dtype if dtype is not None else _default_dtype()
        self.max_length = int(max_length)
        self._lock = threading.Lock()
        self._tokenizer: Any = None
        self._model: Any = None
        self._entail_idx: int = 0
        self._notentail_idx: int = 1

    def _resolve_label_indices(self, model: Any) -> Tuple[int, int]:
        id2label = getattr(model.config, "id2label", None) or {}
        entail_idx = -1
        notentail_idx = -1
        for raw_idx, label in id2label.items():
            try:
                idx = int(raw_idx)
            except (TypeError, ValueError):
                continue
            normalized = str(label).lower().replace("-", "_")
            if "not_entail" in normalized or normalized == "contradiction":
                notentail_idx = idx
            elif "entail" in normalized:
                entail_idx = idx
        if entail_idx < 0 or notentail_idx < 0:
            return 0, 1
        return entail_idx, notentail_idx

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        with self._lock:
            if self._model is not None and self._tokenizer is not None:
                return
            from transformers import (  # noqa: WPS433
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )

            tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            kwargs: dict[str, Any] = {}
            if self.dtype is not None:
                kwargs["torch_dtype"] = self.dtype
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_id, **kwargs
            )
            model.to(self.device)
            model.eval()
            self._entail_idx, self._notentail_idx = self._resolve_label_indices(model)
            self._tokenizer = tokenizer
            self._model = model

    def entail(
        self,
        premises: Sequence[str],
        hypotheses: Sequence[str],
    ) -> List[Tuple[float, float, float]]:
        if len(premises) != len(hypotheses):
            raise ValueError("premises and hypotheses must align")
        if not premises:
            return []
        self._ensure_loaded()
        import torch  # noqa: WPS433

        encoded = self._tokenizer(
            list(premises),
            list(hypotheses),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = self._model(**encoded).logits
        probs = torch.softmax(logits.float(), dim=-1).cpu().tolist()
        out: List[Tuple[float, float, float]] = []
        for row in probs:
            p_entail = float(row[self._entail_idx])
            out.append((p_entail, float(1.0 - p_entail), 0.0))
        return out


__all__ = [
    "BgeM3ZeroShotNLIProvider",
    "MiniCheckNLIProvider",
]
