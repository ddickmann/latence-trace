"""Compliance redaction service orchestration."""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from latence_trace.api.compliance_models import (
    ComplianceEntity,
    ComplianceRedactionRequest,
    ComplianceRedactionResponse,
    ComplianceUsage,
)
from latence_trace.compliance.chunking import TextChunk, chunk_text_by_tokens
from latence_trace.compliance.custom_regex import (
    extract_custom_entities,
    validate_custom_configs,
)
from latence_trace.compliance.labels import (
    canonicalize_model_label,
    resolve_label_set,
    to_model_label_set,
)
from latence_trace.compliance.redaction import ComplianceRedactionEngine
from latence_trace.compliance.validators import apply_sanity_checks
from latence_trace.providers.gliner import VllmFactoryDebertaGlinerProvider

DEFAULT_COMPLIANCE_TEXT_MAX_TOKENS = 512
DEFAULT_COMPLIANCE_RETRY_MIN_TEXT_TOKENS = 32


class ComplianceServiceError(Exception):
    error_code = "compliance_service_error"
    status_code = 500
    hint = "Inspect the compliance redaction request and GLiNER endpoint health."


class ComplianceValidationError(ComplianceServiceError):
    error_code = "validation_error"
    status_code = 400
    hint = "Send text plus either mode='open' or mode='category' with valid categories."


class ComplianceRedactionService:
    def __init__(
        self,
        *,
        provider: VllmFactoryDebertaGlinerProvider,
        model_name: str,
        tokenizer: Any | None = None,
        dataset_path: str | None = None,
        max_text_tokens: int = DEFAULT_COMPLIANCE_TEXT_MAX_TOKENS,
        max_model_len: int = 768,
        max_chunk_concurrency: int = 16,
        min_retry_text_tokens: int = DEFAULT_COMPLIANCE_RETRY_MIN_TEXT_TOKENS,
    ) -> None:
        self.provider = provider
        self.model_name = model_name
        self.max_text_tokens = max(1, int(max_text_tokens))
        self.max_model_len = max(1, int(max_model_len))
        self.max_chunk_concurrency = max(1, int(max_chunk_concurrency))
        self.min_retry_text_tokens = max(1, int(min_retry_text_tokens))
        self._tokenizer = tokenizer
        self._redaction_engine = ComplianceRedactionEngine(dataset_path)

    @classmethod
    def from_env(cls) -> ComplianceRedactionService:
        endpoint = os.environ.get(
            "LATENCE_TRACE_COMPLIANCE_GLINER_ENDPOINT",
            "http://127.0.0.1:8003",
        )
        model = os.environ.get(
            "LATENCE_TRACE_COMPLIANCE_GLINER_MODEL",
            "knowledgator/gliner-pii-large-v1.0",
        )
        max_concurrency = int(os.environ.get("LATENCE_TRACE_COMPLIANCE_MAX_CONCURRENCY", "16"))
        provider = VllmFactoryDebertaGlinerProvider(
            endpoint=endpoint,
            model=model,
            timeout=float(os.environ.get("LATENCE_TRACE_COMPLIANCE_GLINER_TIMEOUT", "60")),
            max_concurrency=max_concurrency,
        )
        tokenizer = None
        try:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                model,
                use_fast=True,
                trust_remote_code=True,
            )
        except Exception:
            tokenizer = None
        return cls(
            provider=provider,
            model_name=model,
            tokenizer=tokenizer,
            dataset_path=os.environ.get("LATENCE_TRACE_COMPLIANCE_DATASET_PATH"),
            max_text_tokens=int(
                os.environ.get(
                    "LATENCE_TRACE_COMPLIANCE_TEXT_MAX_TOKENS",
                    str(DEFAULT_COMPLIANCE_TEXT_MAX_TOKENS),
                )
            ),
            max_model_len=int(os.environ.get("LATENCE_TRACE_COMPLIANCE_MAX_MODEL_LEN", "768")),
            max_chunk_concurrency=max_concurrency,
            min_retry_text_tokens=int(
                os.environ.get(
                    "LATENCE_TRACE_COMPLIANCE_RETRY_MIN_TEXT_TOKENS",
                    str(DEFAULT_COMPLIANCE_RETRY_MIN_TEXT_TOKENS),
                )
            ),
        )

    def replacement_dataset_stats(self, *, load: bool = False) -> dict[str, Any]:
        """Return privacy-safe synthetic replacement dataset health."""

        return self._redaction_engine.stats(load=load)

    def _estimate_prompt_token_overhead(self, labels: list[str]) -> int:
        if not labels:
            return 0
        if self._tokenizer is None:
            return (len(labels) * 2) + 1
        prompt = []
        for label in labels:
            prompt.append("<<ENT>>")
            prompt.append(label)
        prompt.append("<<SEP>>")
        try:
            encoded = self._tokenizer(
                [prompt],
                is_split_into_words=True,
                add_special_tokens=True,
                truncation=False,
                padding=False,
                return_tensors=None,
            )
            return len(encoded["input_ids"][0])
        except Exception:
            return (len(labels) * 2) + 1

    def _effective_text_token_budget(self, labels: list[str]) -> int:
        prompt_overhead = self._estimate_prompt_token_overhead(labels)
        safety_margin = int(prompt_overhead * 0.15)
        model_budget = self.max_model_len - prompt_overhead - safety_margin
        return max(1, min(self.max_text_tokens, model_budget))

    def _chunk_text(self, text: str, *, labels: list[str]) -> list[TextChunk]:
        return chunk_text_by_tokens(
            text,
            self._tokenizer,
            max_text_tokens=self._effective_text_token_budget(labels),
        )

    @staticmethod
    def _offset_entities(
        text: str,
        chunk: TextChunk,
        entities: list[dict[str, Any]],
        alias_to_canonical: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for entity in entities:
            start = int(entity.get("start", 0)) + chunk.start
            end = int(entity.get("end", 0)) + chunk.start
            if start < 0 or end <= start or start >= len(text):
                continue
            end = min(end, len(text))
            updated = dict(entity)
            updated["start"] = start
            updated["end"] = end
            updated["text"] = text[start:end]
            if alias_to_canonical:
                model_label = str(updated.get("label", ""))
                canonical_label = canonicalize_model_label(model_label, alias_to_canonical)
                updated["label"] = canonical_label
                if canonical_label != model_label:
                    metadata = dict(updated.get("metadata") or {})
                    metadata["model_label"] = model_label
                    updated["metadata"] = metadata
            updated["source"] = updated.get("source") or "model"
            updated["score"] = float(updated.get("score", 0.0))
            out.append(updated)
        return out

    @staticmethod
    def _dedupe_entities(entities: list[dict[str, Any]]) -> list[dict[str, Any]]:
        def priority(entity: dict[str, Any]) -> tuple[int, float, int]:
            source_priority = 1 if entity.get("source") == "custom_regex" else 0
            score = float(entity.get("score", 0.0))
            length = int(entity.get("end", 0)) - int(entity.get("start", 0))
            return (source_priority, score, length)

        kept: list[dict[str, Any]] = []
        for entity in sorted(
            entities,
            key=lambda item: (
                int(item.get("start", 0)),
                -priority(item)[0],
                -priority(item)[1],
                -priority(item)[2],
            ),
        ):
            start = int(entity.get("start", 0))
            end = int(entity.get("end", 0))
            replacement_idx: int | None = None
            skip = False
            for idx, existing in enumerate(kept):
                if start < int(existing["end"]) and end > int(existing["start"]):
                    if priority(entity) > priority(existing):
                        replacement_idx = idx
                    else:
                        skip = True
                    break
            if skip:
                continue
            if replacement_idx is not None:
                kept[replacement_idx] = entity
            else:
                kept.append(entity)
        return sorted(kept, key=lambda item: int(item.get("start", 0)))

    def _detect_chunk(
        self,
        chunk: TextChunk,
        *,
        labels: list[str],
        request: ComplianceRedactionRequest,
    ) -> tuple[list[tuple[TextChunk, list[dict[str, Any]], float]], int]:
        started = time.perf_counter()
        try:
            entities = self.provider.detect(
                text=chunk.text,
                labels=labels,
                threshold=request.threshold,
                flat_ner=request.flat_ner,
                multi_label=request.multi_label,
            )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            return [(chunk, entities, elapsed_ms)], 0
        except Exception:
            retry_chunks = self._retry_subchunks(chunk)
            if not retry_chunks:
                raise
            results: list[tuple[TextChunk, list[dict[str, Any]], float]] = []
            retries = 1
            for retry_chunk in retry_chunks:
                retry_results, retry_count = self._detect_chunk(
                    retry_chunk,
                    labels=labels,
                    request=request,
                )
                results.extend(retry_results)
                retries += retry_count
            return results, retries

    def _retry_subchunks(self, chunk: TextChunk) -> list[TextChunk]:
        if chunk.token_count <= self.min_retry_text_tokens:
            return []
        next_token_budget = max(self.min_retry_text_tokens, chunk.token_count // 2)
        if next_token_budget >= chunk.token_count:
            return []
        subchunks = chunk_text_by_tokens(
            chunk.text,
            self._tokenizer,
            max_text_tokens=next_token_budget,
        )
        if len(subchunks) <= 1:
            return []
        return [
            TextChunk(
                text=subchunk.text,
                start=chunk.start + subchunk.start,
                end=chunk.start + subchunk.end,
                chunk_index=subchunk.chunk_index,
                total_chunks=subchunk.total_chunks,
                token_count=subchunk.token_count,
            )
            for subchunk in subchunks
            if subchunk.end > subchunk.start
        ]

    def redact(self, request: ComplianceRedactionRequest) -> ComplianceRedactionResponse:
        started = time.perf_counter()
        timings: dict[str, float] = {}

        labels_started = time.perf_counter()
        try:
            canonical_labels = resolve_label_set(
                mode=request.mode,
                categories=request.categories,
                labels=request.labels,
            )
        except ValueError as exc:
            raise ComplianceValidationError(str(exc)) from exc
        model_labels, alias_to_canonical = to_model_label_set(canonical_labels)
        timings["label_resolution_ms"] = (time.perf_counter() - labels_started) * 1000.0

        chunk_started = time.perf_counter()
        chunks = self._chunk_text(request.text, labels=model_labels)
        timings["chunking_ms"] = (time.perf_counter() - chunk_started) * 1000.0
        if not chunks:
            raise ComplianceValidationError("text did not produce any chunks")

        infer_started = time.perf_counter()
        model_entities: list[dict[str, Any]] = []
        chunk_latencies: list[float] = []
        detected_chunk_count = 0
        retry_split_count = 0
        with ThreadPoolExecutor(
            max_workers=min(self.max_chunk_concurrency, len(chunks)),
            thread_name_prefix="latence-compliance-chunk",
        ) as pool:
            futures = [
                pool.submit(self._detect_chunk, chunk, labels=model_labels, request=request)
                for chunk in chunks
            ]
            for future in futures:
                results, retries = future.result()
                retry_split_count += retries
                detected_chunk_count += len(results)
                for chunk, entities, elapsed_ms in results:
                    chunk_latencies.append(elapsed_ms)
                    model_entities.extend(
                        self._offset_entities(
                            request.text,
                            chunk,
                            entities,
                            alias_to_canonical,
                        )
                    )
        timings["vllm_request_ms"] = (time.perf_counter() - infer_started) * 1000.0
        timings["max_chunk_vllm_ms"] = max(chunk_latencies) if chunk_latencies else 0.0
        timings["vllm_retry_splits"] = float(retry_split_count)

        sanity_started = time.perf_counter()
        entities = apply_sanity_checks(request.text, model_entities)
        timings["sanity_checks_ms"] = (time.perf_counter() - sanity_started) * 1000.0

        regex_started = time.perf_counter()
        custom_configs, config_errors = validate_custom_configs(
            [item.model_dump() for item in request.custom_labels]
        )
        if config_errors:
            raise ComplianceValidationError("; ".join(config_errors))
        entities.extend(extract_custom_entities(request.text, custom_configs))
        timings["custom_regex_ms"] = (time.perf_counter() - regex_started) * 1000.0

        dedupe_started = time.perf_counter()
        entities = self._dedupe_entities(entities)
        timings["dedupe_ms"] = (time.perf_counter() - dedupe_started) * 1000.0

        redacted_text = None
        if request.redact:
            redact_started = time.perf_counter()
            redacted_text, entities = self._redaction_engine.redact_text(
                request.text,
                entities,
                mode=request.redaction_mode,
                country=request.country,
            )
            timings["redaction_ms"] = (time.perf_counter() - redact_started) * 1000.0
        else:
            timings["redaction_ms"] = 0.0

        processing_time_ms = (time.perf_counter() - started) * 1000.0
        timings["total_ms"] = processing_time_ms
        unique_labels = sorted({str(entity.get("label", "")) for entity in entities})

        response_entities = [ComplianceEntity(**entity) for entity in entities]
        return ComplianceRedactionResponse(
            original_text=request.text if request.include_original_text else None,
            entities=response_entities,
            entity_count=len(response_entities),
            unique_labels=unique_labels,
            redacted_text=redacted_text,
            chunks_processed=detected_chunk_count or len(chunks),
            labels_used=canonical_labels,
            label_mode=request.mode,
            selected_categories=request.categories,
            processing_time_ms=round(processing_time_ms, 2),
            timings_ms={key: round(value, 2) for key, value in timings.items()},
            usage=ComplianceUsage(
                chunks_processed=detected_chunk_count or len(chunks),
                labels_used=len(canonical_labels),
                entity_count=len(response_entities),
                unique_labels=unique_labels,
                redaction_mode=request.redaction_mode if request.redact else None,
                redacted=request.redact,
                mode=request.mode,
                categories=request.categories,
            ),
        )
