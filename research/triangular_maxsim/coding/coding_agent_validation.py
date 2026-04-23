"""Scorer-stacking validation harness for coding-agent groundedness.

Scorer-stack x chunker matrix. The primary scorer is
``lightonai/GTE-ModernColBERT-v1``. ``lightonai/LateOn-Code-edge`` contributes a
second, orthogonal per-unit score that is fused with the GTE score per support
unit. Both encoders always see the **same** chunks, which are built once per
(case, chunker) at GTE's 256-token budget.

Scorer configurations emitted per case x chunker:

- ``gte_only`` — GTE reverse_context (the existing baseline).
- ``code_only`` — LateOn-Code-edge reverse_context (reference, not the target).
- ``fuse_mean`` / ``fuse_max`` / ``fuse_min`` — per-unit fusion of the two.

Held-out under fusion is precision-first: a unit is fused-held-out only if
**both** encoders independently marked it ``usage_state == "unused"``. No
local held-out classifier is reimplemented here; we read the upstream one.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from latence_trace.core.groundedness import (
    ResponseChunkInput,
    SupportUnitInput,
    count_text_tokens,
    encode_texts,
    partition_support_units,
    provider_token_limit,
    score_groundedness_response_chunked,
    tokenize_text,
    tokenize_with_offsets,
)
from research.triangular_maxsim.dataset import DEFAULT_MODEL
from research.triangular_maxsim.groundedness_service_validation import _load_provider
from research.triangular_maxsim.coding.code_cases import CodingCase, load_handcrafted_cases
from research.triangular_maxsim.coding.code_segmenter import (
    describe_segmenter_backend,
    render_context_files,
    segment_code_files,
)
from research.triangular_maxsim.coding.swebench_cases import load_swebench_cases
from research.triangular_maxsim.coding.transcript_cases import load_transcript_cases
from research.triangular_maxsim.coding.transcript_cases_v2 import load_transcript_cases_v2


logger = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent
_CACHE_DIR = _HERE / "cache"
_ARTIFACTS_DIR = _HERE / "artifacts"
_CACHE_SCHEMA_VERSION = 5

CASE_BANK_CHOICES: Tuple[str, ...] = ("v1_diff", "transcripts_v1", "transcripts_v2", "both")
DEFAULT_CASE_BANK = "transcripts_v1"

GO_NO_GO = {"min_anchor_auroc_reverse_context": 0.90}

PRIMARY_ENCODER = DEFAULT_MODEL  # "lightonai/GTE-ModernColBERT-v1"
CODE_ENCODER = "lightonai/LateOn-Code-edge"
PRIMARY_SLUG = "gte"
CODE_SLUG = "code"
SHARED_CHUNK_TOKENS = 256
CHUNKING_PROVIDER_CHOICES: Tuple[str, ...] = ("gte", "code")
DEFAULT_SWEEP_BUDGETS: Tuple[int, ...] = (64, 128, 256, 512, 1024)
SATURATION_THRESHOLD = 0.95

CHUNKERS: Tuple[str, ...] = ("sentence_packed", "colgrep")
SCORER_CONFIGS: Tuple[str, ...] = (
    "gte_only",
    "code_only",
    "fuse_mean",
    "fuse_max",
    "fuse_min",
)
FUSION_RULES: Tuple[str, ...] = ("mean", "max", "min")
_FLAGSHIP_CELL: Tuple[str, str] = ("colgrep", "fuse_mean")

_WORD_RE = re.compile(r"\S+\s*")


@dataclass
class SharedEncodedCase:
    """Per (case, chunker) bundle: shared segments + per-encoder encodings."""

    case: CodingCase
    chunker: str
    segmenter_backend: str
    raw_context: str
    support_ids: List[str]
    support_texts: List[str]
    support_metadata: List[Dict[str, Any]]
    support_units_by_encoder: Dict[str, List[SupportUnitInput]]
    response_chunks_by_encoder: Dict[str, List[ResponseChunkInput]]
    query_embeddings_by_encoder: Dict[str, Optional[torch.Tensor]] = field(default_factory=dict)
    query_tokens_by_encoder: Dict[str, Optional[List[str]]] = field(default_factory=dict)
    context_token_count_by_encoder: Dict[str, int] = field(default_factory=dict)


def _score_batch_units() -> int:
    return 64


def _latency_repeats() -> int:
    return 1


def auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    pairs = sorted(zip(scores, labels))
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0

    ranks = [0.0] * len(pairs)
    idx = 0
    while idx < len(pairs):
        jdx = idx + 1
        while jdx < len(pairs) and pairs[jdx][0] == pairs[idx][0]:
            jdx += 1
        avg_rank = (idx + jdx + 1) / 2.0
        for rank_idx in range(idx, jdx):
            ranks[rank_idx] = avg_rank
        idx = jdx

    sum_pos = sum(rank for rank, (_score, label) in zip(ranks, pairs) if label == 1)
    return float((sum_pos - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg))


def _maybe_auroc(scores: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    if len(labels) < 2 or len(set(labels)) < 2:
        return None
    return auroc(scores, labels)


def _fmt_metric(value: Optional[float], *, width: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{width}f}"


def _fmt_delta(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.4f}"


def _slugify(name: str) -> str:
    return name.replace("/", "__")


def cache_path_for(
    chunker: str,
    *,
    use_prompts: bool,
    max_swebench_instances: int,
    context_chunk_tokens: int,
    response_chunk_tokens: int,
    has_code_encoder: bool,
    chunking_provider: str,
    case_bank: str,
    max_cases: Optional[int] = None,
) -> Path:
    prompt_suffix = "__prompts" if use_prompts else ""
    encoder_suffix = "__dual" if has_code_encoder else "__gte"
    chunker_suffix = f"__cp_{chunking_provider}"
    bank_suffix = f"__bank_{case_bank}"
    pilot_suffix = f"__pilot{max_cases}" if max_cases else ""
    return _CACHE_DIR / (
        f"coding_shared_v{_CACHE_SCHEMA_VERSION}__{chunker}{encoder_suffix}{prompt_suffix}{chunker_suffix}{bank_suffix}"
        f"__sw{max_swebench_instances}__ctx{context_chunk_tokens}__resp{response_chunk_tokens}{pilot_suffix}.pt"
    )


def _build_case_bank(
    *,
    case_bank: str,
    max_swebench_instances: int,
    max_cases: Optional[int] = None,
) -> List[CodingCase]:
    if case_bank == "v1_diff":
        cases = load_handcrafted_cases() + load_swebench_cases(
            max_instances=max_swebench_instances
        )
    elif case_bank == "transcripts_v1":
        cases = load_transcript_cases()
    elif case_bank == "transcripts_v2":
        cases = load_transcript_cases_v2()
    elif case_bank == "both":
        cases = (
            load_transcript_cases()
            + load_handcrafted_cases()
            + load_swebench_cases(max_instances=max_swebench_instances)
        )
    else:
        raise ValueError(
            f"case_bank must be one of {CASE_BANK_CHOICES}, got {case_bank!r}"
        )

    if max_cases is not None and max_cases > 0:
        if case_bank.startswith("transcripts_"):
            keep_bases: List[str] = []
            seen_bases: set = set()
            for case in cases:
                base = str(case.metadata.get("base_scenario_id") or "")
                if base and base not in seen_bases:
                    seen_bases.add(base)
                    keep_bases.append(base)
                if len(keep_bases) >= max_cases:
                    break
            keep_set = set(keep_bases)
            cases = [c for c in cases if str(c.metadata.get("base_scenario_id") or "") in keep_set]
        else:
            cases = cases[: max_cases * 3]
    return cases


def _effective_doc_budget(provider: Any, requested_budget: int) -> int:
    limit = provider_token_limit(provider, is_query=False)
    if limit is None:
        return requested_budget
    return max(32, min(int(requested_budget), int(limit) - 16))


def _split_atomic_response_spans(
    text: str,
    *,
    provider: Any,
    chunk_token_budget: int,
) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    cursor = 0
    for raw_line in text.splitlines(keepends=True):
        line_start = cursor
        line_end = cursor + len(raw_line)
        cursor = line_end
        if count_text_tokens(provider, raw_line) <= chunk_token_budget:
            spans.append((line_start, line_end))
            continue

        matches = list(_WORD_RE.finditer(raw_line))
        if not matches:
            step = 200
            for start in range(line_start, line_end, step):
                spans.append((start, min(line_end, start + step)))
            continue

        current_start = line_start + matches[0].start()
        current_end = line_start + matches[0].end()
        for match in matches[1:]:
            candidate_end = line_start + match.end()
            candidate_text = text[current_start:candidate_end]
            if count_text_tokens(provider, candidate_text) > chunk_token_budget:
                spans.append((current_start, current_end))
                current_start = line_start + match.start()
                current_end = line_start + match.end()
            else:
                current_end = candidate_end
        spans.append((current_start, current_end))
    return spans


def _build_text_spans_by_lines(
    text: str,
    *,
    provider: Any,
    chunk_token_budget: int,
) -> List[Dict[str, Any]]:
    effective_budget = _effective_doc_budget(provider, chunk_token_budget)
    atomic_spans = _split_atomic_response_spans(
        text,
        provider=provider,
        chunk_token_budget=effective_budget,
    )
    merged_spans: List[Tuple[int, int]] = []
    current_start: Optional[int] = None
    current_end: Optional[int] = None
    for start, end in atomic_spans:
        if current_start is None:
            current_start = start
            current_end = end
            continue
        candidate = text[current_start:end]
        if count_text_tokens(provider, candidate) > effective_budget:
            merged_spans.append((current_start, current_end or current_start))
            current_start = start
            current_end = end
        else:
            current_end = end
    if current_start is not None and current_end is not None:
        merged_spans.append((current_start, current_end))

    return [
        {
            "text": text[start:end].strip(),
            "offset_start": int(start),
            "offset_end": int(end),
        }
        for start, end in merged_spans
        if text[start:end].strip()
    ]


def _segment_case_context(
    case: CodingCase,
    *,
    chunker: str,
    provider: Any,
    context_chunk_tokens: int,
) -> Tuple[str, List[Dict[str, Any]], str]:
    raw_context, _rendered = render_context_files(case.context_files)
    effective_budget = _effective_doc_budget(provider, context_chunk_tokens)
    if chunker == "sentence_packed":
        segments = _build_text_spans_by_lines(
            raw_context,
            provider=provider,
            chunk_token_budget=effective_budget,
        )
        for index, segment in enumerate(segments):
            metadata = dict(segment.get("metadata") or {})
            metadata["segmenter"] = "sentence_packed_line_fallback"
            metadata["chunk_index"] = int(index)
            segment["metadata"] = metadata
        return raw_context, segments, "sentence_packed"
    if chunker == "colgrep":
        segments = segment_code_files(
            case.context_files,
            provider=provider,
            chunk_token_budget=effective_budget,
        )
        backend = str(
            ((segments[0].get("metadata") or {}).get("segmenter_backend"))
            if segments
            else "colgrep"
        )
        return raw_context, segments, backend
    raise ValueError(f"Unknown chunker: {chunker}")


def _encode_support_units(
    *,
    support_ids: Sequence[str],
    support_texts: Sequence[str],
    support_metadata: Sequence[Dict[str, Any]],
    provider: Any,
    document_prompt_name: Optional[str],
) -> List[SupportUnitInput]:
    embeddings = encode_texts(
        provider,
        list(support_texts),
        is_query=False,
        prompt_name=document_prompt_name,
    )
    units: List[SupportUnitInput] = []
    for support_id, text, metadata, embedding in zip(
        support_ids, support_texts, support_metadata, embeddings
    ):
        path = metadata.get("path")
        offset_start = metadata.get("_offset_start")
        offset_end = metadata.get("_offset_end")
        units.append(
            SupportUnitInput(
                support_id=str(support_id),
                chunk_id=str(path or support_id),
                source_mode="support_units",
                text=str(text),
                embeddings=embedding,
                tokens=tokenize_text(
                    provider,
                    str(text),
                    expected_len=int(embedding.shape[0]),
                    is_query=False,
                ),
                offset_start=int(offset_start) if offset_start is not None else None,
                offset_end=int(offset_end) if offset_end is not None else None,
                source_id=str(path) if path else None,
                metadata={k: v for k, v in metadata.items() if not k.startswith("_")},
            )
        )
    return units


def _encode_response_chunks(
    response_text: str,
    *,
    provider: Any,
    chunk_token_budget: int,
    document_prompt_name: Optional[str],
    spans: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[ResponseChunkInput], List[Dict[str, Any]]]:
    resolved_spans = (
        spans
        if spans is not None
        else _build_text_spans_by_lines(
            response_text,
            provider=provider,
            chunk_token_budget=chunk_token_budget,
        )
    )
    chunk_texts = [span["text"] for span in resolved_spans]
    if not chunk_texts:
        return [], resolved_spans
    embeddings_list = encode_texts(
        provider,
        chunk_texts,
        is_query=False,
        prompt_name=document_prompt_name,
    )
    response_chunks: List[ResponseChunkInput] = []
    for span, chunk_text, chunk_embedding in zip(resolved_spans, chunk_texts, embeddings_list):
        start = int(span["offset_start"])
        end = int(span["offset_end"])
        tokens, offsets = tokenize_with_offsets(
            provider,
            chunk_text,
            expected_len=int(chunk_embedding.shape[0]),
            is_query=False,
        )
        shifted_offsets = [
            None if span_offset is None else (int(span_offset[0]) + start, int(span_offset[1]) + start)
            for span_offset in offsets
        ]
        response_chunks.append(
            ResponseChunkInput(
                text=chunk_text,
                embeddings=chunk_embedding,
                tokens=tokens,
                offset_start=int(start),
                offset_end=int(end),
                token_char_spans=shifted_offsets,
            )
        )
    return response_chunks, resolved_spans


def _encode_query(
    query_text: str,
    *,
    provider: Any,
    query_prompt_name: Optional[str],
) -> Tuple[Optional[torch.Tensor], Optional[List[str]]]:
    stripped = (query_text or "").strip()
    if not stripped:
        return None, None
    embedding = encode_texts(
        provider,
        [stripped],
        is_query=True,
        prompt_name=query_prompt_name,
    )[0]
    tokens = tokenize_text(
        provider,
        stripped,
        expected_len=int(embedding.shape[0]),
        is_query=True,
    )
    return embedding, tokens


def _build_shared_encoded_case(
    case: CodingCase,
    *,
    chunker: str,
    gte_provider: Any,
    gte_query_prompt: Optional[str],
    gte_doc_prompt: Optional[str],
    code_provider: Optional[Any],
    code_query_prompt: Optional[str],
    code_doc_prompt: Optional[str],
    context_chunk_tokens: int,
    response_chunk_tokens: int,
    chunking_provider: Any,
) -> SharedEncodedCase:
    raw_context, segments, backend = _segment_case_context(
        case,
        chunker=chunker,
        provider=chunking_provider,
        context_chunk_tokens=context_chunk_tokens,
    )
    if not segments:
        raise ValueError(f"{case.id} produced no support segments for chunker={chunker}")

    support_ids: List[str] = []
    support_texts: List[str] = []
    support_metadata: List[Dict[str, Any]] = []
    for index, segment in enumerate(segments):
        support_id = f"{case.id}-{chunker}-{index}"
        metadata = dict(segment.get("metadata") or {})
        offset_start = segment.get("offset_start")
        offset_end = segment.get("offset_end")
        if offset_start is not None:
            metadata["_offset_start"] = int(offset_start)
        if offset_end is not None:
            metadata["_offset_end"] = int(offset_end)
        support_ids.append(support_id)
        support_texts.append(str(segment["text"]))
        support_metadata.append(metadata)

    response_spans = _build_text_spans_by_lines(
        case.response,
        provider=chunking_provider,
        chunk_token_budget=response_chunk_tokens,
    )
    if not response_spans:
        raise ValueError(f"{case.id} produced no response chunks")

    response_gte, _ = _encode_response_chunks(
        case.response,
        provider=gte_provider,
        chunk_token_budget=response_chunk_tokens,
        document_prompt_name=gte_doc_prompt,
        spans=response_spans,
    )
    if not response_gte:
        raise ValueError(f"{case.id} produced no encoded response chunks")

    gte_units = _encode_support_units(
        support_ids=support_ids,
        support_texts=support_texts,
        support_metadata=support_metadata,
        provider=gte_provider,
        document_prompt_name=gte_doc_prompt,
    )
    gte_query_emb, gte_query_tokens = _encode_query(
        case.query or "",
        provider=gte_provider,
        query_prompt_name=gte_query_prompt,
    )

    support_units_by_encoder: Dict[str, List[SupportUnitInput]] = {PRIMARY_SLUG: gte_units}
    response_chunks_by_encoder: Dict[str, List[ResponseChunkInput]] = {PRIMARY_SLUG: response_gte}
    query_embeddings_by_encoder: Dict[str, Optional[torch.Tensor]] = {PRIMARY_SLUG: gte_query_emb}
    query_tokens_by_encoder: Dict[str, Optional[List[str]]] = {PRIMARY_SLUG: gte_query_tokens}
    context_tokens_by_encoder: Dict[str, int] = {
        PRIMARY_SLUG: int(sum(u.embeddings.shape[0] for u in gte_units))
    }

    if code_provider is not None:
        try:
            code_units = _encode_support_units(
                support_ids=support_ids,
                support_texts=support_texts,
                support_metadata=support_metadata,
                provider=code_provider,
                document_prompt_name=code_doc_prompt,
            )
            code_response, _code_spans = _encode_response_chunks(
                case.response,
                provider=code_provider,
                chunk_token_budget=response_chunk_tokens,
                document_prompt_name=code_doc_prompt,
                spans=response_spans,
            )
            code_query_emb, code_query_tokens = _encode_query(
                case.query or "",
                provider=code_provider,
                query_prompt_name=code_query_prompt,
            )
        except Exception as exc:  # pragma: no cover - env-specific
            logger.warning(
                "Code encoder failed on case %s (%s x %s): %s",
                case.id,
                chunker,
                CODE_ENCODER,
                exc,
            )
        else:
            support_units_by_encoder[CODE_SLUG] = code_units
            response_chunks_by_encoder[CODE_SLUG] = code_response
            query_embeddings_by_encoder[CODE_SLUG] = code_query_emb
            query_tokens_by_encoder[CODE_SLUG] = code_query_tokens
            context_tokens_by_encoder[CODE_SLUG] = int(
                sum(u.embeddings.shape[0] for u in code_units)
            )

    return SharedEncodedCase(
        case=case,
        chunker=chunker,
        segmenter_backend=backend,
        raw_context=raw_context,
        support_ids=support_ids,
        support_texts=support_texts,
        support_metadata=support_metadata,
        support_units_by_encoder=support_units_by_encoder,
        response_chunks_by_encoder=response_chunks_by_encoder,
        query_embeddings_by_encoder=query_embeddings_by_encoder,
        query_tokens_by_encoder=query_tokens_by_encoder,
        context_token_count_by_encoder=context_tokens_by_encoder,
    )


def load_or_build_encoded_cases(
    *,
    rebuild: bool,
    chunker: str,
    use_prompts: bool,
    max_swebench_instances: int,
    context_chunk_tokens: int,
    response_chunk_tokens: int,
    gte_provider: Any,
    gte_query_prompt: Optional[str],
    gte_doc_prompt: Optional[str],
    code_provider: Optional[Any],
    code_query_prompt: Optional[str],
    code_doc_prompt: Optional[str],
    chunking_provider_choice: str,
    chunking_provider: Any,
    case_bank: str,
    cases: Optional[Sequence[CodingCase]] = None,
    max_cases: Optional[int] = None,
) -> List[SharedEncodedCase]:
    cache_path = cache_path_for(
        chunker,
        use_prompts=use_prompts,
        max_swebench_instances=max_swebench_instances,
        context_chunk_tokens=context_chunk_tokens,
        response_chunk_tokens=response_chunk_tokens,
        has_code_encoder=code_provider is not None,
        chunking_provider=chunking_provider_choice,
        case_bank=case_bank,
        max_cases=max_cases,
    )
    if not rebuild and cache_path.exists():
        with open(cache_path, "rb") as handle:
            blob = pickle.load(handle)
        if (
            blob.get("cache_schema_version") == _CACHE_SCHEMA_VERSION
            and blob.get("chunker") == chunker
            and blob.get("use_prompts") == use_prompts
            and blob.get("max_swebench_instances") == max_swebench_instances
            and blob.get("context_chunk_tokens") == context_chunk_tokens
            and blob.get("response_chunk_tokens") == response_chunk_tokens
            and blob.get("has_code_encoder") == (code_provider is not None)
            and blob.get("chunking_provider") == chunking_provider_choice
            and blob.get("case_bank") == case_bank
        ):
            return blob["encoded_cases"]

    if chunker == "colgrep":
        backend = describe_segmenter_backend()
        if backend.get("status") != "ready":
            raise RuntimeError(
                "colgrep_parser is required for the 'colgrep' chunker. "
                "Install it first, for example with "
                "`pip install /workspace/next-plaid/colgrep/python-sdk`."
            )

    if cases is None:
        cases = _build_case_bank(
            case_bank=case_bank,
            max_swebench_instances=max_swebench_instances,
        )

    encoded_cases = [
        _build_shared_encoded_case(
            case,
            chunker=chunker,
            gte_provider=gte_provider,
            gte_query_prompt=gte_query_prompt,
            gte_doc_prompt=gte_doc_prompt,
            code_provider=code_provider,
            code_query_prompt=code_query_prompt,
            code_doc_prompt=code_doc_prompt,
            context_chunk_tokens=context_chunk_tokens,
            response_chunk_tokens=response_chunk_tokens,
            chunking_provider=chunking_provider,
        )
        for case in cases
    ]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as handle:
        pickle.dump(
            {
                "cache_schema_version": _CACHE_SCHEMA_VERSION,
                "chunker": chunker,
                "use_prompts": use_prompts,
                "max_swebench_instances": max_swebench_instances,
                "context_chunk_tokens": context_chunk_tokens,
                "response_chunk_tokens": response_chunk_tokens,
                "has_code_encoder": code_provider is not None,
                "chunking_provider": chunking_provider_choice,
                "case_bank": case_bank,
                "encoded_cases": encoded_cases,
            },
            handle,
        )
    return encoded_cases


def _run_scorer(
    encoded_case: SharedEncodedCase,
    encoder_slug: str,
) -> Tuple[Dict[str, Any], float]:
    support_units = encoded_case.support_units_by_encoder[encoder_slug]
    response_chunks = encoded_case.response_chunks_by_encoder[encoder_slug]
    query_embeddings = encoded_case.query_embeddings_by_encoder.get(encoder_slug)
    query_tokens = encoded_case.query_tokens_by_encoder.get(encoder_slug)

    support_batches = partition_support_units(
        support_units, batch_size=_score_batch_units()
    )
    latency_samples: List[float] = []
    scored: Optional[Dict[str, Any]] = None
    for _ in range(_latency_repeats()):
        start = time.perf_counter()
        scored = score_groundedness_response_chunked(
            response_chunks=response_chunks,
            support_batches=support_batches,
            response_text=encoded_case.case.response,
            query_embeddings=query_embeddings,
            query_tokens=query_tokens,
            evidence_limit=5,
            primary_metric="reverse_context",
            debug_dense_matrices=False,
            structured_enabled=False,
        )
        latency_samples.append((time.perf_counter() - start) * 1000.0)
    assert scored is not None
    scored["_latency_samples_ms"] = latency_samples
    return scored, float(np.median(latency_samples))


_AGG_FN = {
    "mean": lambda values: float(np.mean(values)),
    "max": lambda values: float(np.max(values)),
    "min": lambda values: float(np.min(values)),
}


def _fuse_scalar(values: Sequence[float], rule: str) -> float:
    return _AGG_FN[rule](list(values))


def _max_top_evidence(scored: Dict[str, Any]) -> float:
    return max(
        (float(item["score"]) for item in scored.get("top_evidence") or []),
        default=0.0,
    )


def _held_out_from_scored(scored: Dict[str, Any]) -> List[str]:
    return [
        str(u.get("support_id"))
        for u in scored.get("support_units") or []
        if u.get("usage_state") == "unused" and u.get("support_id") is not None
    ]


def _uncertain_from_scored(scored: Dict[str, Any]) -> List[str]:
    return [
        str(u.get("support_id"))
        for u in scored.get("support_units") or []
        if u.get("usage_state") == "uncertain" and u.get("support_id") is not None
    ]


def _used_from_scored(scored: Dict[str, Any]) -> List[str]:
    return [
        str(u.get("support_id"))
        for u in scored.get("support_units") or []
        if u.get("usage_state") == "used" and u.get("support_id") is not None
    ]


def _row_single(
    encoded_case: SharedEncodedCase,
    *,
    scorer_config: str,
    encoder: str,
    scored: Dict[str, Any],
    latency_ms: float,
    phantom_threshold: float,
) -> Dict[str, Any]:
    max_ev = _max_top_evidence(scored)
    held_out_ids = _held_out_from_scored(scored)
    uncertain_ids = _uncertain_from_scored(scored)
    used_ids = _used_from_scored(scored)
    scores = dict(scored["scores"])
    total = int(scores.get("support_units_total") or len(encoded_case.support_ids))
    held_out = {
        "held_out_ids": held_out_ids,
        "held_out_ids_gte": held_out_ids if scorer_config == "gte_only" else [],
        "held_out_ids_code": held_out_ids if scorer_config == "code_only" else [],
        "uncertain_ids": uncertain_ids,
        "uncertain_ids_diff": [],
        "used_ids": used_ids,
        "context_unused_ratio": float(scores.get("context_unused_ratio") or 0.0),
        "context_uncertain_ratio": float(scores.get("context_uncertain_ratio") or 0.0),
        "context_usage_ratio": float(scores.get("context_usage_ratio") or 0.0),
        "support_units_unused": int(scores.get("support_units_unused") or 0),
        "support_units_uncertain": int(scores.get("support_units_uncertain") or 0),
        "support_units_usage_used": int(scores.get("support_units_usage_used") or 0),
        "support_units_total": total,
    }
    context_tokens = encoded_case.context_token_count_by_encoder.get(
        PRIMARY_SLUG if scorer_config == "gte_only" else CODE_SLUG,
        encoded_case.context_token_count_by_encoder.get(PRIMARY_SLUG, 0),
    )
    metadata = dict(encoded_case.case.metadata)
    tier = str(metadata.get("tier") or "").strip() or None
    base_scenario_id = metadata.get("base_scenario_id")
    return {
        "id": encoded_case.case.id,
        "source": encoded_case.case.source,
        "chunker": encoded_case.chunker,
        "segmenter_backend": encoded_case.segmenter_backend,
        "scorer_config": scorer_config,
        "encoder": encoder,
        "label": encoded_case.case.label,
        "subcategory": encoded_case.case.subcategory,
        "tier": tier,
        "base_scenario_id": base_scenario_id,
        "query": encoded_case.case.query,
        "response": encoded_case.case.response,
        "scores": scores,
        "top_evidence": scored.get("top_evidence") or [],
        "warnings": scored.get("warnings") or [],
        "held_out": held_out,
        "latency_ms": float(latency_ms),
        "latency_samples_ms": list(scored.get("_latency_samples_ms") or [latency_ms]),
        "notes": encoded_case.case.notes,
        "context_token_count": int(context_tokens),
        "support_unit_count": len(encoded_case.support_ids),
        "phantom_threshold": float(phantom_threshold),
        "max_top_evidence_score": float(max_ev),
        "phantom_flagged": bool(max_ev < phantom_threshold),
        "metadata": metadata,
    }


def _row_fused(
    encoded_case: SharedEncodedCase,
    *,
    rule: str,
    gte_scored: Dict[str, Any],
    gte_latency_ms: float,
    code_scored: Dict[str, Any],
    code_latency_ms: float,
    phantom_threshold: float,
) -> Dict[str, Any]:
    scorer_config = f"fuse_{rule}"
    gte_scores = gte_scored["scores"]
    code_scores = code_scored["scores"]
    coverage_threshold = float(gte_scores.get("context_coverage_threshold") or 0.0)

    gte_units = list(gte_scored.get("support_units") or [])
    code_units = list(code_scored.get("support_units") or [])
    unit_count = min(len(gte_units), len(code_units))

    fused_units: List[Dict[str, Any]] = []
    fused_used_count = 0
    for idx in range(unit_count):
        gte_unit = gte_units[idx]
        code_unit = code_units[idx]
        support_id = str(gte_unit.get("support_id") or code_unit.get("support_id") or "")
        fused_coverage = _fuse_scalar(
            [float(gte_unit.get("coverage_score") or 0.0), float(code_unit.get("coverage_score") or 0.0)],
            rule,
        )
        fused_score = _fuse_scalar(
            [float(gte_unit.get("score") or 0.0), float(code_unit.get("score") or 0.0)],
            rule,
        )
        fused_matched = int(
            _fuse_scalar(
                [
                    float(gte_unit.get("matched_response_tokens") or 0),
                    float(code_unit.get("matched_response_tokens") or 0),
                ],
                rule,
            )
        )
        fused_used = fused_coverage >= coverage_threshold
        if fused_used:
            fused_used_count += 1
        fused_units.append(
            {
                "support_id": support_id,
                "index": int(gte_unit.get("index", idx)),
                "coverage_score": fused_coverage,
                "score": fused_score,
                "matched_response_tokens": fused_matched,
                "used": bool(fused_used),
                "fusion_rule": rule,
            }
        )

    fused_coverage_ratio = (
        float(fused_used_count) / float(unit_count) if unit_count > 0 else 0.0
    )
    fused_reverse_context = _fuse_scalar(
        [float(gte_scores.get("reverse_context") or 0.0), float(code_scores.get("reverse_context") or 0.0)],
        rule,
    )
    fused_consensus = _fuse_scalar(
        [
            float(gte_scores.get("consensus_hardened") or 0.0),
            float(code_scores.get("consensus_hardened") or 0.0),
        ],
        rule,
    )
    fused_attribution_used = _fuse_scalar(
        [
            float(gte_scores.get("context_attribution_used_count") or 0.0),
            float(code_scores.get("context_attribution_used_count") or 0.0),
        ],
        rule,
    )
    fused_attribution_ratio = (
        float(fused_attribution_used) / float(unit_count) if unit_count > 0 else 0.0
    )
    fused_max_ev = _fuse_scalar(
        [float(_max_top_evidence(gte_scored)), float(_max_top_evidence(code_scored))],
        rule,
    )

    gte_unused = set(_held_out_from_scored(gte_scored))
    code_unused = set(_held_out_from_scored(code_scored))
    consensus_unused = sorted(gte_unused & code_unused)
    diff_unused = sorted(gte_unused.symmetric_difference(code_unused))
    gte_uncertain = set(_uncertain_from_scored(gte_scored))
    code_uncertain = set(_uncertain_from_scored(code_scored))
    consensus_uncertain = sorted(gte_uncertain & code_uncertain)
    gte_used = set(_used_from_scored(gte_scored))
    code_used = set(_used_from_scored(code_scored))
    consensus_used = sorted(gte_used & code_used)

    total = max(unit_count, int(gte_scores.get("support_units_total") or unit_count))
    fused_scores = {
        "reverse_context": fused_reverse_context,
        "consensus_hardened": fused_consensus,
        "context_coverage_ratio": fused_coverage_ratio,
        "context_coverage_threshold": coverage_threshold,
        "support_units_used": int(fused_used_count),
        "support_units_total": int(total),
        "context_attribution_ratio": fused_attribution_ratio,
        "context_attribution_used_count": int(fused_attribution_used),
        "support_units_usage_used": int(len(consensus_used)),
        "support_units_unused": int(len(consensus_unused)),
        "support_units_uncertain": int(len(consensus_uncertain)),
        "context_usage_ratio": float(len(consensus_used)) / float(total) if total > 0 else 0.0,
        "context_unused_ratio": float(len(consensus_unused)) / float(total) if total > 0 else 0.0,
        "context_uncertain_ratio": float(len(consensus_uncertain)) / float(total) if total > 0 else 0.0,
        "_gte_reverse_context": float(gte_scores.get("reverse_context") or 0.0),
        "_code_reverse_context": float(code_scores.get("reverse_context") or 0.0),
    }

    held_out = {
        "held_out_ids": consensus_unused,
        "held_out_ids_gte": sorted(gte_unused),
        "held_out_ids_code": sorted(code_unused),
        "uncertain_ids": consensus_uncertain,
        "uncertain_ids_diff": diff_unused,
        "used_ids": consensus_used,
        "context_unused_ratio": fused_scores["context_unused_ratio"],
        "context_uncertain_ratio": fused_scores["context_uncertain_ratio"],
        "context_usage_ratio": fused_scores["context_usage_ratio"],
        "support_units_unused": int(len(consensus_unused)),
        "support_units_uncertain": int(len(consensus_uncertain)),
        "support_units_usage_used": int(len(consensus_used)),
        "support_units_total": int(total),
    }

    fused_latency = _fuse_scalar([gte_latency_ms, code_latency_ms], rule)
    top_evidence_pool = sorted(
        (gte_scored.get("top_evidence") or []) + (code_scored.get("top_evidence") or []),
        key=lambda item: float(item.get("score", 0.0)),
        reverse=True,
    )[:5]
    context_tokens = _fuse_scalar(
        [
            float(encoded_case.context_token_count_by_encoder.get(PRIMARY_SLUG, 0)),
            float(encoded_case.context_token_count_by_encoder.get(CODE_SLUG, 0)),
        ],
        rule,
    )
    warnings = list(gte_scored.get("warnings") or []) + list(code_scored.get("warnings") or [])
    metadata = dict(encoded_case.case.metadata)
    tier = str(metadata.get("tier") or "").strip() or None
    base_scenario_id = metadata.get("base_scenario_id")
    return {
        "id": encoded_case.case.id,
        "source": encoded_case.case.source,
        "chunker": encoded_case.chunker,
        "segmenter_backend": encoded_case.segmenter_backend,
        "scorer_config": scorer_config,
        "encoder": "stacked",
        "fusion_rule": rule,
        "label": encoded_case.case.label,
        "subcategory": encoded_case.case.subcategory,
        "tier": tier,
        "base_scenario_id": base_scenario_id,
        "query": encoded_case.case.query,
        "response": encoded_case.case.response,
        "scores": fused_scores,
        "fused_units": fused_units,
        "top_evidence": top_evidence_pool,
        "warnings": warnings,
        "held_out": held_out,
        "latency_ms": float(fused_latency),
        "latency_samples_ms": [gte_latency_ms, code_latency_ms],
        "notes": encoded_case.case.notes,
        "context_token_count": int(context_tokens),
        "support_unit_count": len(encoded_case.support_ids),
        "phantom_threshold": float(phantom_threshold),
        "max_top_evidence_score": float(fused_max_ev),
        "phantom_flagged": bool(fused_max_ev < phantom_threshold),
        "metadata": metadata,
    }


def score_encoded_case(
    encoded_case: SharedEncodedCase,
    *,
    phantom_threshold: float,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    has_code = CODE_SLUG in encoded_case.support_units_by_encoder

    gte_scored, gte_latency = _run_scorer(encoded_case, PRIMARY_SLUG)
    rows.append(
        _row_single(
            encoded_case,
            scorer_config="gte_only",
            encoder=PRIMARY_ENCODER,
            scored=gte_scored,
            latency_ms=gte_latency,
            phantom_threshold=phantom_threshold,
        )
    )

    if not has_code:
        return rows

    try:
        code_scored, code_latency = _run_scorer(encoded_case, CODE_SLUG)
    except Exception as exc:  # pragma: no cover - env-specific
        logger.warning(
            "Skipping code scorer for %s x %s: %s",
            encoded_case.case.id,
            encoded_case.chunker,
            exc,
        )
        return rows

    rows.append(
        _row_single(
            encoded_case,
            scorer_config="code_only",
            encoder=CODE_ENCODER,
            scored=code_scored,
            latency_ms=code_latency,
            phantom_threshold=phantom_threshold,
        )
    )

    for rule in FUSION_RULES:
        rows.append(
            _row_fused(
                encoded_case,
                rule=rule,
                gte_scored=gte_scored,
                gte_latency_ms=gte_latency,
                code_scored=code_scored,
                code_latency_ms=code_latency,
                phantom_threshold=phantom_threshold,
            )
        )
    return rows


def _mean_optional(values: Sequence[Optional[float]]) -> Optional[float]:
    filtered = [float(v) for v in values if v is not None]
    if not filtered:
        return None
    return float(np.mean(filtered))


def _distribution_stats(values: Sequence[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"mean": None, "std": None, "p10": None, "p50": None, "p90": None, "max": None}
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(arr.max()),
    }


def _saturation_rate(values: Sequence[float], threshold: float) -> Optional[float]:
    if not values:
        return None
    return float(np.mean([1.0 if float(v) >= threshold else 0.0 for v in values]))


def summarize_cell(
    rows: Sequence[Dict[str, Any]],
    *,
    phantom_threshold: float,
) -> Dict[str, Any]:
    anchors = [row for row in rows if row["label"] in {"grounded", "ungrounded"}]
    labels = [1 if row["label"] == "grounded" else 0 for row in anchors]
    grounded_rows = [row for row in anchors if row["label"] == "grounded"]
    ungrounded_rows = [row for row in anchors if row["label"] == "ungrounded"]
    ambiguous_rows = [row for row in rows if row["label"] == "ambiguous"]
    phantom_rows = [row for row in rows if row["subcategory"] == "phantom_api"]

    correct_rows = [row for row in rows if row.get("tier") == "correct"]
    wrong_rows = [row for row in rows if row.get("tier") == "wrong"]
    tier_ambiguous_rows = [row for row in rows if row.get("tier") == "ambiguous"]

    def _tier_rc(bucket: Sequence[Dict[str, Any]]) -> List[float]:
        return [float(row["scores"]["reverse_context"]) for row in bucket]

    correct_rc = _tier_rc(correct_rows)
    wrong_rc = _tier_rc(wrong_rows)
    tier_ambiguous_rc = _tier_rc(tier_ambiguous_rows)

    correct_rc_stats = _distribution_stats(correct_rc)
    wrong_rc_stats = _distribution_stats(wrong_rc)
    tier_ambiguous_rc_stats = _distribution_stats(tier_ambiguous_rc)

    auroc_correct_vs_wrong = _maybe_auroc(
        [*correct_rc, *wrong_rc],
        [1] * len(correct_rc) + [0] * len(wrong_rc),
    )
    auroc_correct_vs_ambiguous = _maybe_auroc(
        [*correct_rc, *tier_ambiguous_rc],
        [1] * len(correct_rc) + [0] * len(tier_ambiguous_rc),
    )
    auroc_grounded_vs_ungrounded_plus_ambiguous = _maybe_auroc(
        [*correct_rc, *wrong_rc, *tier_ambiguous_rc],
        [1] * len(correct_rc)
        + [0] * len(wrong_rc)
        + [0] * len(tier_ambiguous_rc),
    )

    by_base: Dict[str, Dict[str, float]] = {}
    for row in rows:
        base = row.get("base_scenario_id")
        tier = row.get("tier")
        if not base or tier not in {"correct", "ambiguous", "wrong"}:
            continue
        by_base.setdefault(str(base), {})[tier] = float(row["scores"]["reverse_context"])
    monotonicity_samples: List[int] = []
    partial_monotonicity_samples: List[int] = []
    for base, scores in by_base.items():
        if "correct" in scores and "wrong" in scores:
            partial_monotonicity_samples.append(
                1 if scores["correct"] > scores["wrong"] else 0
            )
        if all(t in scores for t in ("correct", "ambiguous", "wrong")):
            monotonicity_samples.append(
                1
                if scores["correct"] > scores["ambiguous"] > scores["wrong"]
                else 0
            )
    monotonicity_rate = (
        float(np.mean(monotonicity_samples)) if monotonicity_samples else None
    )
    partial_monotonicity_rate = (
        float(np.mean(partial_monotonicity_samples))
        if partial_monotonicity_samples
        else None
    )

    latency_samples = [
        float(sample)
        for row in rows
        for sample in (row.get("latency_samples_ms") or [row["latency_ms"]])
    ]

    def _mean_score(bucket: Sequence[Dict[str, Any]], key: str) -> Optional[float]:
        if not bucket:
            return None
        return float(np.mean([float(item["scores"][key]) for item in bucket]))

    def _mean_optional_score(bucket: Sequence[Dict[str, Any]], key: str) -> Optional[float]:
        values = [item["scores"].get(key) for item in bucket if item["scores"].get(key) is not None]
        if not values:
            return None
        return float(np.mean([float(value) for value in values]))

    grounded_coverage = _mean_optional_score(grounded_rows, "context_coverage_ratio")
    ungrounded_coverage = _mean_optional_score(ungrounded_rows, "context_coverage_ratio")
    grounded_unused = _mean_optional(
        [float(row["held_out"]["context_unused_ratio"]) for row in grounded_rows]
    )
    ungrounded_unused = _mean_optional(
        [float(row["held_out"]["context_unused_ratio"]) for row in ungrounded_rows]
    )
    grounded_uncertain = _mean_optional(
        [float(row["held_out"]["context_uncertain_ratio"]) for row in grounded_rows]
    )
    grounded_usage = _mean_optional(
        [float(row["held_out"]["context_usage_ratio"]) for row in grounded_rows]
    )
    grounded_held_out_nonempty = (
        float(
            np.mean(
                [1.0 if row["held_out"]["held_out_ids"] else 0.0 for row in grounded_rows]
            )
        )
        if grounded_rows
        else None
    )

    grounded_rc = [float(row["scores"]["reverse_context"]) for row in grounded_rows]
    ungrounded_rc = [float(row["scores"]["reverse_context"]) for row in ungrounded_rows]
    all_rc = [float(row["scores"]["reverse_context"]) for row in anchors]
    grounded_rc_stats = _distribution_stats(grounded_rc)
    ungrounded_rc_stats = _distribution_stats(ungrounded_rc)
    reverse_context_separation = (
        float(grounded_rc_stats["mean"] - ungrounded_rc_stats["mean"])
        if grounded_rc_stats["mean"] is not None and ungrounded_rc_stats["mean"] is not None
        else None
    )
    reverse_context_overlap = (
        float(max(0.0, ungrounded_rc_stats["p90"] - grounded_rc_stats["p10"]))
        if grounded_rc_stats["p10"] is not None and ungrounded_rc_stats["p90"] is not None
        else None
    )
    saturation_all = _saturation_rate(all_rc, SATURATION_THRESHOLD)
    saturation_grounded = _saturation_rate(grounded_rc, SATURATION_THRESHOLD)
    saturation_ungrounded = _saturation_rate(ungrounded_rc, SATURATION_THRESHOLD)

    grounded_max_ev = [float(row["max_top_evidence_score"]) for row in grounded_rows]
    ungrounded_max_ev = [float(row["max_top_evidence_score"]) for row in ungrounded_rows]
    max_ev_separation = (
        float(np.mean(grounded_max_ev) - np.mean(ungrounded_max_ev))
        if grounded_max_ev and ungrounded_max_ev
        else None
    )

    metrics = {
        "case_count": len(rows),
        "anchor_count": len(anchors),
        "grounded_count": len(grounded_rows),
        "ungrounded_count": len(ungrounded_rows),
        "ambiguous_count": len(ambiguous_rows),
        "correct_tier_count": len(correct_rows),
        "wrong_tier_count": len(wrong_rows),
        "ambiguous_tier_count": len(tier_ambiguous_rows),
        "reverse_context_auroc": _maybe_auroc(
            [float(row["scores"]["reverse_context"]) for row in anchors],
            labels,
        ),
        "consensus_hardened_auroc": _maybe_auroc(
            [float(row["scores"].get("consensus_hardened") or 0.0) for row in anchors],
            labels,
        ),
        "auroc_correct_vs_wrong": auroc_correct_vs_wrong,
        "auroc_correct_vs_ambiguous": auroc_correct_vs_ambiguous,
        "auroc_grounded_vs_ungrounded_plus_ambiguous": auroc_grounded_vs_ungrounded_plus_ambiguous,
        "monotonicity_rate": monotonicity_rate,
        "partial_monotonicity_rate": partial_monotonicity_rate,
        "correct_reverse_context_mean": correct_rc_stats["mean"],
        "correct_reverse_context_std": correct_rc_stats["std"],
        "ambiguous_reverse_context_mean": tier_ambiguous_rc_stats["mean"],
        "ambiguous_reverse_context_std": tier_ambiguous_rc_stats["std"],
        "wrong_reverse_context_mean": wrong_rc_stats["mean"],
        "wrong_reverse_context_std": wrong_rc_stats["std"],
        "grounded_reverse_context_mean": grounded_rc_stats["mean"],
        "grounded_reverse_context_std": grounded_rc_stats["std"],
        "grounded_reverse_context_p10": grounded_rc_stats["p10"],
        "grounded_reverse_context_p90": grounded_rc_stats["p90"],
        "grounded_reverse_context_p50": grounded_rc_stats["p50"],
        "ungrounded_reverse_context_mean": ungrounded_rc_stats["mean"],
        "ungrounded_reverse_context_std": ungrounded_rc_stats["std"],
        "ungrounded_reverse_context_p10": ungrounded_rc_stats["p10"],
        "ungrounded_reverse_context_p90": ungrounded_rc_stats["p90"],
        "ungrounded_reverse_context_p50": ungrounded_rc_stats["p50"],
        "reverse_context_separation": reverse_context_separation,
        "reverse_context_overlap": reverse_context_overlap,
        "saturation_rate": saturation_all,
        "saturation_grounded": saturation_grounded,
        "saturation_ungrounded": saturation_ungrounded,
        "max_top_evidence_separation": max_ev_separation,
        "grounded_context_coverage_mean": grounded_coverage,
        "ungrounded_context_coverage_mean": ungrounded_coverage,
        "context_coverage_delta": (
            float(grounded_coverage - ungrounded_coverage)
            if grounded_coverage is not None and ungrounded_coverage is not None
            else None
        ),
        "grounded_context_unused_mean": grounded_unused,
        "ungrounded_context_unused_mean": ungrounded_unused,
        "context_unused_delta": (
            float(ungrounded_unused - grounded_unused)
            if grounded_unused is not None and ungrounded_unused is not None
            else None
        ),
        "grounded_context_uncertain_mean": grounded_uncertain,
        "grounded_context_usage_mean": grounded_usage,
        "grounded_held_out_nonempty_rate": grounded_held_out_nonempty,
        "phantom_api_count": len(phantom_rows),
        "phantom_api_precision_at_threshold": (
            float(np.mean([1.0 if row["phantom_flagged"] else 0.0 for row in phantom_rows]))
            if phantom_rows
            else None
        ),
        "phantom_api_mean_max_evidence": (
            float(np.mean([float(row["max_top_evidence_score"]) for row in phantom_rows]))
            if phantom_rows
            else None
        ),
        "phantom_threshold": float(phantom_threshold),
        "latency_p50_ms": (
            float(np.percentile(latency_samples, 50)) if latency_samples else 0.0
        ),
        "latency_p95_ms": (
            float(np.percentile(latency_samples, 95)) if latency_samples else 0.0
        ),
        "mean_context_tokens": (
            float(np.mean([row["context_token_count"] for row in rows])) if rows else 0.0
        ),
        "mean_support_units": (
            float(np.mean([row["support_unit_count"] for row in rows])) if rows else 0.0
        ),
    }
    metrics["gate_anchor_auroc_pass"] = bool(
        metrics["reverse_context_auroc"] is not None
        and metrics["reverse_context_auroc"] >= GO_NO_GO["min_anchor_auroc_reverse_context"]
    )

    by_subcategory: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        name = row["subcategory"] or row["label"]
        bucket = by_subcategory.setdefault(
            name,
            {
                "count": 0,
                "reverse_context_mean": 0.0,
                "context_coverage_mean": 0.0,
                "context_unused_mean": 0.0,
                "max_top_evidence_mean": 0.0,
            },
        )
        bucket["count"] += 1
        bucket["reverse_context_mean"] += float(row["scores"]["reverse_context"])
        bucket["context_coverage_mean"] += float(row["scores"].get("context_coverage_ratio") or 0.0)
        bucket["context_unused_mean"] += float(row["held_out"]["context_unused_ratio"])
        bucket["max_top_evidence_mean"] += float(row["max_top_evidence_score"])
    for bucket in by_subcategory.values():
        bucket["reverse_context_mean"] /= max(1, bucket["count"])
        bucket["context_coverage_mean"] /= max(1, bucket["count"])
        bucket["context_unused_mean"] /= max(1, bucket["count"])
        bucket["max_top_evidence_mean"] /= max(1, bucket["count"])
    return {"metrics": metrics, "by_subcategory": by_subcategory}


def summarize_matrix(
    rows: Sequence[Dict[str, Any]],
    *,
    phantom_threshold: float,
    chunkers: Sequence[str],
    scorer_configs: Sequence[str],
) -> Dict[str, Any]:
    by_cell: Dict[str, Dict[str, Any]] = {}
    for chunker in chunkers:
        for scorer_config in scorer_configs:
            key = f"{chunker}|{scorer_config}"
            cell_rows = [
                row
                for row in rows
                if row["chunker"] == chunker and row["scorer_config"] == scorer_config
            ]
            if not cell_rows:
                continue
            summary = summarize_cell(cell_rows, phantom_threshold=phantom_threshold)
            summary["chunker"] = chunker
            summary["scorer_config"] = scorer_config
            by_cell[key] = summary

    def _metric(chunker: str, scorer: str, name: str) -> Optional[float]:
        entry = by_cell.get(f"{chunker}|{scorer}")
        if entry is None:
            return None
        return entry["metrics"].get(name)

    def _delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
        if a is None or b is None:
            return None
        return float(a - b)

    stacking_lift: Dict[str, Dict[str, Optional[float]]] = {}
    for chunker in chunkers:
        for rule in FUSION_RULES:
            entry = stacking_lift.setdefault(chunker, {})
            entry[f"fuse_{rule}_minus_gte_only__reverse_context_auroc"] = _delta(
                _metric(chunker, f"fuse_{rule}", "reverse_context_auroc"),
                _metric(chunker, "gte_only", "reverse_context_auroc"),
            )
            entry[f"fuse_{rule}_minus_gte_only__context_unused_delta"] = _delta(
                _metric(chunker, f"fuse_{rule}", "context_unused_delta"),
                _metric(chunker, "gte_only", "context_unused_delta"),
            )
            entry[f"fuse_{rule}_minus_gte_only__phantom_precision"] = _delta(
                _metric(chunker, f"fuse_{rule}", "phantom_api_precision_at_threshold"),
                _metric(chunker, "gte_only", "phantom_api_precision_at_threshold"),
            )

    chunker_lift: Dict[str, Dict[str, Optional[float]]] = {}
    for scorer in scorer_configs:
        entry = chunker_lift.setdefault(scorer, {})
        entry["colgrep_minus_sentence_packed__reverse_context_auroc"] = _delta(
            _metric("colgrep", scorer, "reverse_context_auroc"),
            _metric("sentence_packed", scorer, "reverse_context_auroc"),
        )
        entry["colgrep_minus_sentence_packed__context_unused_delta"] = _delta(
            _metric("colgrep", scorer, "context_unused_delta"),
            _metric("sentence_packed", scorer, "context_unused_delta"),
        )
        entry["colgrep_minus_sentence_packed__phantom_precision"] = _delta(
            _metric("colgrep", scorer, "phantom_api_precision_at_threshold"),
            _metric("sentence_packed", scorer, "phantom_api_precision_at_threshold"),
        )

    flagship_key = f"{_FLAGSHIP_CELL[0]}|{_FLAGSHIP_CELL[1]}"
    return {
        "by_cell": by_cell,
        "stacking_lift": stacking_lift,
        "chunker_lift": chunker_lift,
        "flagship_cell": {
            "chunker": _FLAGSHIP_CELL[0],
            "scorer_config": _FLAGSHIP_CELL[1],
            "key": flagship_key,
            "present": flagship_key in by_cell,
        },
    }


def _best_fuse_rule(summary: Dict[str, Any], chunker: str) -> Tuple[Optional[str], Optional[float]]:
    best_rule: Optional[str] = None
    best_auroc: Optional[float] = None
    for rule in FUSION_RULES:
        cell = summary["by_cell"].get(f"{chunker}|fuse_{rule}")
        if cell is None:
            continue
        value = cell["metrics"].get("reverse_context_auroc")
        if value is None:
            continue
        if best_auroc is None or value > best_auroc:
            best_auroc = value
            best_rule = rule
    return best_rule, best_auroc


def _flagship_headline(summary: Dict[str, Any]) -> str:
    info = summary["flagship_cell"]
    key = info["key"]
    flagship = summary["by_cell"].get(key)
    if flagship is None:
        return (
            f"SKIP — the flagship cell ({info['chunker']} x {info['scorer_config']}) "
            "was not evaluated in this run (code encoder likely unavailable)."
        )

    metrics = flagship["metrics"]
    auroc_val = metrics["reverse_context_auroc"]
    phantom = metrics["phantom_api_precision_at_threshold"]
    unused_delta = metrics["context_unused_delta"]
    best_rule, best_auroc = _best_fuse_rule(summary, info["chunker"])
    best_note = ""
    if best_rule is not None and best_auroc is not None:
        if best_rule != "mean":
            best_note = (
                f" Best fusion rule on {info['chunker']} is `fuse_{best_rule}` "
                f"(AUROC={best_auroc:.3f}); consider flipping the flagship."
            )

    gte_only_auroc = summary["by_cell"].get(f"{info['chunker']}|gte_only", {}).get(
        "metrics", {}
    ).get("reverse_context_auroc")
    stacking_note = ""
    if gte_only_auroc is not None and auroc_val is not None:
        lift = auroc_val - gte_only_auroc
        stacking_note = f" Stacking lift vs gte_only={_fmt_delta(lift)}."

    if auroc_val is not None and auroc_val >= GO_NO_GO["min_anchor_auroc_reverse_context"]:
        return (
            "YES — the flagship colgrep x fuse_mean cell clears the gate: "
            f"reverse_context AUROC={auroc_val:.3f}, "
            f"phantom-API precision@{metrics['phantom_threshold']:.2f}="
            f"{_fmt_metric(phantom, width=2)}, "
            f"ungrounded minus grounded unused-ratio delta={_fmt_delta(unused_delta)}."
            f"{stacking_note}{best_note}"
        )
    return (
        "NO — the flagship colgrep x fuse_mean cell did not clear the RAG-style gate: "
        f"reverse_context AUROC={_fmt_metric(auroc_val, width=3)}, "
        f"phantom-API precision@{metrics['phantom_threshold']:.2f}="
        f"{_fmt_metric(phantom, width=2)}, "
        f"ungrounded minus grounded unused-ratio delta={_fmt_delta(unused_delta)}."
        f"{stacking_note}{best_note}"
    )


def render_markdown(
    *,
    summary: Dict[str, Any],
    rows: Sequence[Dict[str, Any]],
    run_warnings: Sequence[str],
    max_swebench_instances: int,
    phantom_threshold: float,
    tag: str,
    shared_chunk_tokens: int,
    code_encoder_available: bool,
    chunkers: Sequence[str],
    scorer_configs: Sequence[str],
    case_bank: str = DEFAULT_CASE_BANK,
) -> str:
    lines = [
        "# Coding-Agent Groundedness: Scorer Stack x Chunker",
        "",
        f"> {_flagship_headline(summary)}",
        "",
        f"- run tag: `{tag}`",
        f"- case bank: `{case_bank}`",
        f"- primary scorer: `{PRIMARY_ENCODER}`",
        f"- orthogonal scorer: `{CODE_ENCODER}` (available: `{code_encoder_available}`)",
        f"- shared chunk budget: `{shared_chunk_tokens}` tokens (GTE tokenizer)",
        f"- max SWE-bench instances: `{max_swebench_instances}` (ignored for `transcripts_v1`)",
        f"- phantom threshold: `{phantom_threshold:.2f}`",
        f"- flagship cell: `{_FLAGSHIP_CELL[0]}` x `{_FLAGSHIP_CELL[1]}` "
        f"(present: `{summary['flagship_cell']['present']}`)",
        "",
    ]
    if run_warnings:
        lines.extend(["## Warnings", ""])
        for warning in run_warnings:
            lines.append(f"- {warning}")
        lines.append("")

    lines.extend(
        [
            "## Headline matrix",
            "",
            "| chunker | scorer | AUROC | grounded cov | cov delta | unused delta | phantom@thr | grounded held-out rate | p95 ms | support units |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for chunker in chunkers:
        for scorer in scorer_configs:
            key = f"{chunker}|{scorer}"
            cell_summary = summary["by_cell"].get(key)
            if cell_summary is None:
                continue
            metrics = cell_summary["metrics"]
            gate_mark = " *" if metrics["gate_anchor_auroc_pass"] else ""
            flagship_mark = (
                " (flagship)"
                if (chunker, scorer) == _FLAGSHIP_CELL
                else ""
            )
            lines.append(
                "| {chunker} | `{scorer}`{mark}{flagship} | {auroc} | {cov} | {cov_delta} | {unused_delta} | {phantom} | {held_out_rate} | {latency:.2f} | {units:.1f} |".format(
                    chunker=chunker,
                    scorer=scorer,
                    mark=gate_mark,
                    flagship=flagship_mark,
                    auroc=_fmt_metric(metrics["reverse_context_auroc"]),
                    cov=_fmt_metric(metrics["grounded_context_coverage_mean"]),
                    cov_delta=_fmt_metric(metrics["context_coverage_delta"]),
                    unused_delta=_fmt_metric(metrics["context_unused_delta"]),
                    phantom=_fmt_metric(metrics["phantom_api_precision_at_threshold"]),
                    held_out_rate=_fmt_metric(metrics["grounded_held_out_nonempty_rate"]),
                    latency=metrics["latency_p95_ms"],
                    units=metrics["mean_support_units"],
                )
            )
    lines.append("")
    lines.append("`*` marks cells that pass `reverse_context AUROC >= 0.90`.")

    lines.extend(
        [
            "",
            "## Score distribution diagnostics (reverse_context)",
            "",
            "| chunker | scorer | grounded mean | grounded p10-p90 | ungrounded mean | ungrounded p10-p90 | separation | saturation@0.95 |",
            "|---|---|---:|---|---:|---|---:|---:|",
        ]
    )
    for chunker in chunkers:
        for scorer in scorer_configs:
            key = f"{chunker}|{scorer}"
            cell_summary = summary["by_cell"].get(key)
            if cell_summary is None:
                continue
            metrics = cell_summary["metrics"]
            gp10 = metrics["grounded_reverse_context_p10"]
            gp90 = metrics["grounded_reverse_context_p90"]
            up10 = metrics["ungrounded_reverse_context_p10"]
            up90 = metrics["ungrounded_reverse_context_p90"]
            lines.append(
                "| {chunker} | `{scorer}` | {g_mean} | {g_range} | {u_mean} | {u_range} | {sep} | {sat} |".format(
                    chunker=chunker,
                    scorer=scorer,
                    g_mean=_fmt_metric(metrics["grounded_reverse_context_mean"]),
                    g_range=(
                        f"{gp10:.4f}-{gp90:.4f}"
                        if gp10 is not None and gp90 is not None
                        else "n/a"
                    ),
                    u_mean=_fmt_metric(metrics["ungrounded_reverse_context_mean"]),
                    u_range=(
                        f"{up10:.4f}-{up90:.4f}"
                        if up10 is not None and up90 is not None
                        else "n/a"
                    ),
                    sep=_fmt_delta(metrics["reverse_context_separation"]),
                    sat=_fmt_metric(metrics["saturation_rate"]),
                )
            )
    lines.append("")
    lines.append(
        "`separation` is mean(grounded) − mean(ungrounded) on `reverse_context`. "
        "`saturation@0.95` is the fraction of anchor cases (grounded + ungrounded) whose score ≥ 0.95 — "
        "a ceiling effect indicator."
    )

    lines.extend(
        [
            "",
            "## Tier metrics (transcripts_v1: correct / ambiguous / wrong)",
            "",
            "`AUROC c-vs-w` = AUROC on `tier=correct` vs `tier=wrong`. "
            "`AUROC c-vs-a` = AUROC on `correct` vs `ambiguous` (hard bucket). "
            "`monotonicity` = fraction of base scenarios where "
            "`correct > ambiguous > wrong` holds on `reverse_context`. "
            "`partial mono` = fraction where `correct > wrong` holds (ignores ambiguous).",
            "",
            "| chunker | scorer | AUROC c-vs-w | AUROC c-vs-a | AUROC c-vs-wa | monotonicity | partial mono | correct mean | ambiguous mean | wrong mean |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for chunker in chunkers:
        for scorer in scorer_configs:
            key = f"{chunker}|{scorer}"
            cell_summary = summary["by_cell"].get(key)
            if cell_summary is None:
                continue
            metrics = cell_summary["metrics"]
            lines.append(
                "| {chunker} | `{scorer}` | {cvw} | {cva} | {cvwa} | {mono} | {pmono} | {c_mean} | {a_mean} | {w_mean} |".format(
                    chunker=chunker,
                    scorer=scorer,
                    cvw=_fmt_metric(metrics.get("auroc_correct_vs_wrong")),
                    cva=_fmt_metric(metrics.get("auroc_correct_vs_ambiguous")),
                    cvwa=_fmt_metric(
                        metrics.get("auroc_grounded_vs_ungrounded_plus_ambiguous")
                    ),
                    mono=_fmt_metric(metrics.get("monotonicity_rate")),
                    pmono=_fmt_metric(metrics.get("partial_monotonicity_rate")),
                    c_mean=_fmt_metric(metrics.get("correct_reverse_context_mean")),
                    a_mean=_fmt_metric(metrics.get("ambiguous_reverse_context_mean")),
                    w_mean=_fmt_metric(metrics.get("wrong_reverse_context_mean")),
                )
            )

    lines.extend(
        [
            "",
            "## Stacking lift (fuse_* minus gte_only, per chunker)",
            "",
            "| chunker | rule | AUROC lift | unused-delta lift | phantom precision lift |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for chunker in chunkers:
        entry = summary["stacking_lift"].get(chunker, {})
        for rule in FUSION_RULES:
            lines.append(
                "| {chunker} | fuse_{rule} | {auroc} | {unused} | {phantom} |".format(
                    chunker=chunker,
                    rule=rule,
                    auroc=_fmt_delta(entry.get(f"fuse_{rule}_minus_gte_only__reverse_context_auroc")),
                    unused=_fmt_delta(entry.get(f"fuse_{rule}_minus_gte_only__context_unused_delta")),
                    phantom=_fmt_delta(entry.get(f"fuse_{rule}_minus_gte_only__phantom_precision")),
                )
            )

    lines.extend(
        [
            "",
            "## Chunker lift (colgrep minus sentence_packed, per scorer)",
            "",
            "| scorer | AUROC lift | unused-delta lift | phantom precision lift |",
            "|---|---:|---:|---:|",
        ]
    )
    for scorer in scorer_configs:
        entry = summary["chunker_lift"].get(scorer, {})
        lines.append(
            "| `{scorer}` | {auroc} | {unused} | {phantom} |".format(
                scorer=scorer,
                auroc=_fmt_delta(entry.get("colgrep_minus_sentence_packed__reverse_context_auroc")),
                unused=_fmt_delta(entry.get("colgrep_minus_sentence_packed__context_unused_delta")),
                phantom=_fmt_delta(entry.get("colgrep_minus_sentence_packed__phantom_precision")),
            )
        )

    lines.extend(
        [
            "",
            "## Subcategory snapshot (per cell)",
            "",
        ]
    )
    for chunker in chunkers:
        for scorer in scorer_configs:
            key = f"{chunker}|{scorer}"
            cell_summary = summary["by_cell"].get(key)
            if cell_summary is None:
                continue
            lines.append(f"### `{chunker}` x `{scorer}`")
            lines.append("")
            lines.append(
                "| subcategory | count | reverse_context | coverage | unused ratio | max evidence |"
            )
            lines.append("|---|---:|---:|---:|---:|---:|")
            for name, bucket in sorted(cell_summary["by_subcategory"].items()):
                lines.append(
                    f"| {name} | {bucket['count']} | {bucket['reverse_context_mean']:.4f} | "
                    f"{bucket['context_coverage_mean']:.4f} | {bucket['context_unused_mean']:.4f} | "
                    f"{bucket['max_top_evidence_mean']:.4f} |"
                )
            lines.append("")

    flagship_info = summary["flagship_cell"]
    flagship_rows = [
        row
        for row in rows
        if row["chunker"] == flagship_info["chunker"]
        and row["scorer_config"] == flagship_info["scorer_config"]
    ]
    if flagship_rows:
        lines.extend(
            [
                "## Held-out support units (flagship cell)",
                "",
                f"Flagship cell: `{flagship_info['chunker']}` x `{flagship_info['scorer_config']}`. "
                "Consensus held-out ids are the intersection of GTE's and "
                "LateOn-Code-edge's independently computed `usage_state == 'unused'` sets. "
                "`gte` / `code` columns show each encoder's raw unused set.",
                "",
                "| case | label | subcategory | consensus held-out | gte unused | code unused | consensus held-out ids |",
                "|---|---|---|---:|---:|---:|---|",
            ]
        )
        for row in flagship_rows:
            held_out = row["held_out"]
            consensus_preview = ", ".join(held_out["held_out_ids"][:4])
            if len(held_out["held_out_ids"]) > 4:
                consensus_preview += ", ..."
            if not consensus_preview:
                consensus_preview = "-"
            lines.append(
                f"| {row['id']} | {row['label']} | {row['subcategory'] or '-'} | "
                f"{len(held_out['held_out_ids'])} | {len(held_out['held_out_ids_gte'])} | "
                f"{len(held_out['held_out_ids_code'])} | {consensus_preview} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Per-case scores",
            "",
            "| id | chunker | scorer | label | subcategory | reverse_context | coverage | unused | phantom | max evidence | units |",
            "|---|---|---|---|---|---:|---:|---:|---|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| {id} | {chunker} | `{scorer}` | {label} | {subcategory} | {rc:.4f} | {cov:.4f} | {unused:.4f} | {flagged} | {evidence:.4f} | {units} |".format(
                id=row["id"],
                chunker=row["chunker"],
                scorer=row["scorer_config"],
                label=row["label"],
                subcategory=row["subcategory"] or "-",
                rc=float(row["scores"]["reverse_context"]),
                cov=float(row["scores"].get("context_coverage_ratio") or 0.0),
                unused=float(row["held_out"]["context_unused_ratio"]),
                flagged=row["phantom_flagged"],
                evidence=float(row["max_top_evidence_score"]),
                units=row["support_unit_count"],
            )
        )
    return "\n".join(lines) + "\n"


def _load_code_provider() -> Tuple[Optional[Any], Optional[str], Optional[str], Optional[str]]:
    try:
        code_provider, code_query_prompt, code_doc_prompt, _transport = _load_provider(
            CODE_ENCODER, False
        )
        return code_provider, code_query_prompt, code_doc_prompt, None
    except Exception as exc:  # pragma: no cover - environment-specific
        return None, None, None, (
            f"Skipping code encoder '{CODE_ENCODER}' ({type(exc).__name__}: {exc}). "
            "Falling back to gte_only."
        )


def _run_matrix(
    *,
    chunkers: Sequence[str],
    scorer_configs: Sequence[str],
    use_prompts: bool,
    rebuild: bool,
    max_swebench_instances: int,
    phantom_threshold: float,
    shared_chunk_tokens: int,
    chunking_provider_choice: str,
    case_bank: str = DEFAULT_CASE_BANK,
    providers: Optional[Dict[str, Any]] = None,
    max_cases: Optional[int] = None,
) -> Tuple[
    List[Dict[str, Any]],
    List[str],
    bool,
    Dict[str, Any],
]:
    warnings: List[str] = []

    if providers is None:
        logger.info("Loading primary encoder %s", PRIMARY_ENCODER)
        gte_provider, gte_query_prompt, gte_doc_prompt, gte_transport = _load_provider(
            PRIMARY_ENCODER, use_prompts
        )
        logger.info("Loading code encoder %s", CODE_ENCODER)
        code_provider, code_query_prompt, code_doc_prompt, load_warning = _load_code_provider()
    else:
        gte_provider = providers["gte_provider"]
        gte_query_prompt = providers.get("gte_query_prompt")
        gte_doc_prompt = providers.get("gte_doc_prompt")
        gte_transport = providers.get("gte_transport")
        code_provider = providers.get("code_provider")
        code_query_prompt = providers.get("code_query_prompt")
        code_doc_prompt = providers.get("code_doc_prompt")
        load_warning = providers.get("code_load_warning")

    gte_token_limit = provider_token_limit(gte_provider, is_query=False)
    code_token_limit = (
        provider_token_limit(code_provider, is_query=False)
        if code_provider is not None
        else None
    )
    if load_warning:
        logger.warning(load_warning)
        warnings.append(load_warning)

    effective_scorer_configs = list(scorer_configs)
    if code_provider is None:
        dropped = [s for s in effective_scorer_configs if s != "gte_only"]
        effective_scorer_configs = [s for s in effective_scorer_configs if s == "gte_only"]
        if dropped:
            warnings.append(
                f"Code encoder unavailable; dropping scorer configs {dropped} for this run."
            )

    if chunking_provider_choice not in CHUNKING_PROVIDER_CHOICES:
        raise ValueError(
            f"chunking_provider must be one of {CHUNKING_PROVIDER_CHOICES}, got "
            f"{chunking_provider_choice!r}"
        )
    if chunking_provider_choice == "code" and code_provider is None:
        warnings.append(
            "chunking_provider='code' requested but code encoder unavailable; "
            "falling back to chunking_provider='gte'."
        )
        chunking_provider_choice = "gte"
    chunking_provider_resolved = (
        gte_provider if chunking_provider_choice == "gte" else code_provider
    )
    assert chunking_provider_resolved is not None

    code_encoder_available = code_provider is not None
    cases = _build_case_bank(
        case_bank=case_bank,
        max_swebench_instances=max_swebench_instances,
        max_cases=max_cases,
    )
    rows: List[Dict[str, Any]] = []
    for chunker in chunkers:
        try:
            encoded_cases = load_or_build_encoded_cases(
                rebuild=rebuild,
                chunker=chunker,
                use_prompts=use_prompts,
                max_swebench_instances=max_swebench_instances,
                context_chunk_tokens=shared_chunk_tokens,
                response_chunk_tokens=shared_chunk_tokens,
                gte_provider=gte_provider,
                gte_query_prompt=gte_query_prompt,
                gte_doc_prompt=gte_doc_prompt,
                code_provider=code_provider,
                code_query_prompt=code_query_prompt,
                code_doc_prompt=code_doc_prompt,
                chunking_provider_choice=chunking_provider_choice,
                chunking_provider=chunking_provider_resolved,
                case_bank=case_bank,
                cases=cases,
                max_cases=max_cases,
            )
        except RuntimeError as exc:
            message = f"Skipping chunker '{chunker}': {type(exc).__name__}: {exc}"
            logger.warning(message)
            warnings.append(message)
            continue

        for encoded in encoded_cases:
            case_rows = score_encoded_case(encoded, phantom_threshold=phantom_threshold)
            for row in case_rows:
                if row["scorer_config"] in effective_scorer_configs:
                    rows.append(row)

    run_info = {
        "primary_encoder": PRIMARY_ENCODER,
        "code_encoder": CODE_ENCODER,
        "code_encoder_available": code_encoder_available,
        "gte_token_limit": gte_token_limit,
        "code_token_limit": code_token_limit,
        "shared_chunk_tokens": int(shared_chunk_tokens),
        "chunking_provider": chunking_provider_choice,
        "transport_primary": gte_transport,
        "case_bank": case_bank,
        "case_count": len(cases),
    }
    return rows, warnings, code_encoder_available, run_info


def run(
    *,
    chunkers: Sequence[str] = CHUNKERS,
    scorer_configs: Sequence[str] = SCORER_CONFIGS,
    use_prompts: bool = False,
    rebuild: bool = False,
    tag: str = "code_stack_v1",
    max_swebench_instances: int = 15,
    phantom_threshold: float = 0.35,
    shared_chunk_tokens: int = SHARED_CHUNK_TOKENS,
    chunking_provider_choice: str = "gte",
    case_bank: str = DEFAULT_CASE_BANK,
    output_dir: Optional[Path] = None,
    providers: Optional[Dict[str, Any]] = None,
    max_cases: Optional[int] = None,
) -> Tuple[Dict[str, Any], Path, Path]:
    rows, warnings, code_available, run_info = _run_matrix(
        chunkers=chunkers,
        scorer_configs=scorer_configs,
        use_prompts=use_prompts,
        rebuild=rebuild,
        max_swebench_instances=max_swebench_instances,
        phantom_threshold=phantom_threshold,
        shared_chunk_tokens=shared_chunk_tokens,
        chunking_provider_choice=chunking_provider_choice,
        case_bank=case_bank,
        providers=providers,
        max_cases=max_cases,
    )
    run_info["code_encoder_available"] = code_available
    effective_scorer_configs = list(scorer_configs if code_available else ["gte_only"])
    summary = summarize_matrix(
        rows,
        phantom_threshold=phantom_threshold,
        chunkers=chunkers,
        scorer_configs=effective_scorer_configs,
    )
    payload = {
        "tag": tag,
        "max_swebench_instances": max_swebench_instances,
        "phantom_threshold": phantom_threshold,
        "use_prompts": use_prompts,
        "run_info": run_info,
        "warnings": warnings,
        "flagship_headline": _flagship_headline(summary),
        "summary": summary,
        "rows": rows,
    }

    target_dir = output_dir or _ARTIFACTS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    json_path = target_dir / f"{tag}.json"
    md_path = target_dir / f"{tag}.md"
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    md_path.write_text(
        render_markdown(
            summary=summary,
            rows=rows,
            run_warnings=warnings,
            max_swebench_instances=max_swebench_instances,
            phantom_threshold=phantom_threshold,
            tag=tag,
            shared_chunk_tokens=int(run_info["shared_chunk_tokens"]),
            code_encoder_available=code_available,
            chunkers=chunkers,
            scorer_configs=effective_scorer_configs,
            case_bank=case_bank,
        ),
        encoding="utf-8",
    )
    return payload, json_path, md_path


def _load_shared_providers(*, use_prompts: bool) -> Dict[str, Any]:
    """Load both encoders once so a sweep can reuse them across budgets."""

    logger.info("Loading primary encoder %s", PRIMARY_ENCODER)
    gte_provider, gte_query_prompt, gte_doc_prompt, gte_transport = _load_provider(
        PRIMARY_ENCODER, use_prompts
    )
    logger.info("Loading code encoder %s", CODE_ENCODER)
    code_provider, code_query_prompt, code_doc_prompt, code_load_warning = _load_code_provider()
    return {
        "gte_provider": gte_provider,
        "gte_query_prompt": gte_query_prompt,
        "gte_doc_prompt": gte_doc_prompt,
        "gte_transport": gte_transport,
        "code_provider": code_provider,
        "code_query_prompt": code_query_prompt,
        "code_doc_prompt": code_doc_prompt,
        "code_load_warning": code_load_warning,
    }


def _render_sweep_markdown(
    sweep_entries: Sequence[Dict[str, Any]],
    *,
    tag: str,
    max_swebench_instances: int,
    phantom_threshold: float,
    chunkers: Sequence[str],
    scorer_configs: Sequence[str],
) -> str:
    lines = [
        f"# Chunk-Size Sweep: `{tag}`",
        "",
        "One row per `(budget, chunker, scorer)` so the ceiling effect is visible.",
        "`separation` is `mean(grounded) − mean(ungrounded)` on `reverse_context`; "
        "`saturation@0.95` is the fraction of anchor cases whose score ≥ 0.95.",
        "",
        f"- max SWE-bench instances: `{max_swebench_instances}`",
        f"- phantom threshold: `{phantom_threshold:.2f}`",
        "",
        "## AUROC across budgets (flagship: colgrep × fuse_mean)",
        "",
        "| budget | chunking | chunker | scorer | AUROC | grounded mean | ungrounded mean | separation | saturation@0.95 |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for entry in sweep_entries:
        budget = int(entry["run_info"]["shared_chunk_tokens"])
        chunking = str(entry["run_info"].get("chunking_provider") or "gte")
        cells = entry["summary"]["by_cell"]
        for chunker in chunkers:
            for scorer in scorer_configs:
                cell_summary = cells.get(f"{chunker}|{scorer}")
                if cell_summary is None:
                    continue
                metrics = cell_summary["metrics"]
                flagship = (chunker, scorer) == _FLAGSHIP_CELL
                flagship_mark = " (flagship)" if flagship else ""
                lines.append(
                    "| {budget} | {chunking} | {chunker} | `{scorer}`{flagship} | {auroc} | {g_mean} | {u_mean} | {sep} | {sat} |".format(
                        budget=budget,
                        chunking=chunking,
                        chunker=chunker,
                        scorer=scorer,
                        flagship=flagship_mark,
                        auroc=_fmt_metric(metrics["reverse_context_auroc"]),
                        g_mean=_fmt_metric(metrics["grounded_reverse_context_mean"]),
                        u_mean=_fmt_metric(metrics["ungrounded_reverse_context_mean"]),
                        sep=_fmt_delta(metrics["reverse_context_separation"]),
                        sat=_fmt_metric(metrics["saturation_rate"]),
                    )
                )

    lines.extend(
        [
            "",
            "## Flagship cell across budgets (colgrep × fuse_mean)",
            "",
            "| budget | chunking | AUROC | grounded mean±std | ungrounded mean±std | separation | overlap | saturation@0.95 |",
            "|---:|---|---:|---|---|---:|---:|---:|",
        ]
    )
    for entry in sweep_entries:
        budget = int(entry["run_info"]["shared_chunk_tokens"])
        chunking = str(entry["run_info"].get("chunking_provider") or "gte")
        cell_summary = entry["summary"]["by_cell"].get(
            f"{_FLAGSHIP_CELL[0]}|{_FLAGSHIP_CELL[1]}"
        )
        if cell_summary is None:
            continue
        m = cell_summary["metrics"]
        g_mean = m["grounded_reverse_context_mean"]
        g_std = m["grounded_reverse_context_std"]
        u_mean = m["ungrounded_reverse_context_mean"]
        u_std = m["ungrounded_reverse_context_std"]
        lines.append(
            "| {budget} | {chunking} | {auroc} | {g} | {u} | {sep} | {overlap} | {sat} |".format(
                budget=budget,
                chunking=chunking,
                auroc=_fmt_metric(m["reverse_context_auroc"]),
                g=(
                    f"{g_mean:.4f}±{g_std:.4f}"
                    if g_mean is not None and g_std is not None
                    else "n/a"
                ),
                u=(
                    f"{u_mean:.4f}±{u_std:.4f}"
                    if u_mean is not None and u_std is not None
                    else "n/a"
                ),
                sep=_fmt_delta(m["reverse_context_separation"]),
                overlap=_fmt_metric(m["reverse_context_overlap"]),
                sat=_fmt_metric(m["saturation_rate"]),
            )
        )

    lines.extend(
        [
            "",
            "## Per-budget artifacts",
            "",
            "Each budget's full report is at:",
            "",
        ]
    )
    for entry in sweep_entries:
        lines.append(
            f"- `{entry['run_info']['shared_chunk_tokens']}` tokens "
            f"({entry['run_info'].get('chunking_provider', 'gte')} chunker): "
            f"[{entry['tag']}.md]({entry['tag']}.md)"
        )
    lines.append("")
    return "\n".join(lines)


def run_sweep(
    *,
    budgets: Sequence[Tuple[int, str]],
    chunkers: Sequence[str] = CHUNKERS,
    scorer_configs: Sequence[str] = SCORER_CONFIGS,
    use_prompts: bool = False,
    rebuild: bool = False,
    tag: str = "chunk_size_sweep_v1",
    max_swebench_instances: int = 15,
    phantom_threshold: float = 0.35,
    case_bank: str = DEFAULT_CASE_BANK,
    output_dir: Optional[Path] = None,
) -> Tuple[Path, Path]:
    """Run the matrix across multiple `(budget, chunking_provider)` pairs.

    Each budget gets its own per-run artifact so cell-level detail is preserved;
    a top-level sweep report lines up budgets side by side for AUROC and
    score-distribution diagnostics.
    """

    target_dir = output_dir or _ARTIFACTS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    providers = _load_shared_providers(use_prompts=use_prompts)

    sweep_entries: List[Dict[str, Any]] = []
    for budget, chunking in budgets:
        run_tag = f"{tag}__b{budget}_{chunking}"
        logger.info(
            "Sweep: running budget=%s chunking_provider=%s tag=%s", budget, chunking, run_tag
        )
        payload, _json_path, _md_path = run(
            chunkers=chunkers,
            scorer_configs=scorer_configs,
            use_prompts=use_prompts,
            rebuild=rebuild,
            tag=run_tag,
            max_swebench_instances=max_swebench_instances,
            phantom_threshold=phantom_threshold,
            shared_chunk_tokens=int(budget),
            chunking_provider_choice=chunking,
            case_bank=case_bank,
            output_dir=target_dir,
            providers=providers,
        )
        sweep_entries.append(payload)

    effective_scorer_configs = list(scorer_configs)
    if not any(e["run_info"].get("code_encoder_available") for e in sweep_entries):
        effective_scorer_configs = ["gte_only"]

    sweep_json = {
        "tag": tag,
        "max_swebench_instances": max_swebench_instances,
        "phantom_threshold": phantom_threshold,
        "budgets": [
            {"shared_chunk_tokens": int(b), "chunking_provider": c} for b, c in budgets
        ],
        "entries": [
            {
                "tag": e["tag"],
                "shared_chunk_tokens": int(e["run_info"]["shared_chunk_tokens"]),
                "chunking_provider": e["run_info"].get("chunking_provider", "gte"),
                "by_cell": e["summary"]["by_cell"],
                "flagship_headline": e["flagship_headline"],
                "warnings": e["warnings"],
            }
            for e in sweep_entries
        ],
    }

    json_path = target_dir / f"{tag}.json"
    md_path = target_dir / f"{tag}.md"
    json_path.write_text(json.dumps(sweep_json, indent=2, default=str), encoding="utf-8")
    md_path.write_text(
        _render_sweep_markdown(
            sweep_entries,
            tag=tag,
            max_swebench_instances=max_swebench_instances,
            phantom_threshold=phantom_threshold,
            chunkers=chunkers,
            scorer_configs=effective_scorer_configs,
        ),
        encoding="utf-8",
    )
    return json_path, md_path


def _parse_budget_spec(spec: str) -> Tuple[int, str]:
    if ":" in spec:
        raw_budget, chunking = spec.split(":", 1)
    else:
        raw_budget, chunking = spec, "gte"
    budget = int(raw_budget)
    if budget <= 0:
        raise ValueError(f"Budget must be positive, got {budget}")
    if chunking not in CHUNKING_PROVIDER_CHOICES:
        raise ValueError(
            f"chunking provider must be one of {CHUNKING_PROVIDER_CHOICES}, got {chunking!r}"
        )
    return budget, chunking


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--chunker", action="append", default=None, help="Restrict chunkers.")
    parser.add_argument(
        "--scorer",
        action="append",
        default=None,
        help="Restrict scorer configs (gte_only, code_only, fuse_mean, fuse_max, fuse_min).",
    )
    parser.add_argument("--prompts", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--tag", default="code_stack_v1")
    parser.add_argument("--max-swebench-instances", type=int, default=15)
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help=(
            "For transcripts_v1/v2 banks, cap the number of BASE scenarios loaded "
            "(all tiers per base are kept). Useful for pilot A/B runs."
        ),
    )
    parser.add_argument("--phantom-threshold", type=float, default=0.35)
    parser.add_argument(
        "--shared-chunk-tokens",
        type=int,
        default=SHARED_CHUNK_TOKENS,
        help="Shared support-unit + response chunk budget (single-run mode).",
    )
    parser.add_argument(
        "--chunking-provider",
        choices=CHUNKING_PROVIDER_CHOICES,
        default="gte",
        help="Which encoder's tokenizer/limit drives chunking (single-run mode).",
    )
    parser.add_argument(
        "--sweep",
        action="append",
        default=None,
        help=(
            "Add a sweep point `budget[:chunking_provider]`, e.g. '128' or '1024:code'. "
            "May be repeated. Triggers the sweep runner; --tag is used as the sweep tag."
        ),
    )
    parser.add_argument(
        "--case-bank",
        choices=CASE_BANK_CHOICES,
        default=DEFAULT_CASE_BANK,
        help=(
            "Which case bank to score. 'transcripts_v1' uses the new real-agent-transcript "
            "bank with correct/ambiguous/wrong tiers; 'v1_diff' uses the legacy diff + "
            "SWE-bench bank; 'both' concatenates."
        ),
    )
    args = parser.parse_args()

    selected_chunkers = tuple(args.chunker) if args.chunker else CHUNKERS
    selected_scorer_configs = tuple(args.scorer) if args.scorer else SCORER_CONFIGS

    if args.sweep:
        budgets = [_parse_budget_spec(spec) for spec in args.sweep]
        json_path, md_path = run_sweep(
            budgets=budgets,
            chunkers=selected_chunkers,
            scorer_configs=selected_scorer_configs,
            use_prompts=args.prompts,
            rebuild=args.rebuild,
            tag=args.tag,
            max_swebench_instances=args.max_swebench_instances,
            phantom_threshold=args.phantom_threshold,
            case_bank=args.case_bank,
        )
        print(f"Wrote {json_path}")
        print(f"Wrote {md_path}")
    else:
        payload, json_path, md_path = run(
            chunkers=selected_chunkers,
            scorer_configs=selected_scorer_configs,
            use_prompts=args.prompts,
            rebuild=args.rebuild,
            tag=args.tag,
            max_swebench_instances=args.max_swebench_instances,
            phantom_threshold=args.phantom_threshold,
            shared_chunk_tokens=args.shared_chunk_tokens,
            chunking_provider_choice=args.chunking_provider,
            case_bank=args.case_bank,
            max_cases=args.max_cases,
        )
        print(f"Wrote {json_path}")
        print(f"Wrote {md_path}")
        print(payload["flagship_headline"])
