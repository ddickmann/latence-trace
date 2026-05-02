from __future__ import annotations

import asyncio
import os
import re
import threading
import time
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from latence_trace.api.compliance_models import (
    ComplianceRedactionRequest,
    compliance_schema_metadata,
)
from latence_trace.api.compliance_routes import create_compliance_router
from latence_trace.api.compliance_service import ComplianceRedactionService
from latence_trace.compliance.labels import (
    GDPR_CATEGORIES,
    all_gdpr_labels,
    model_label_alias,
    resolve_label_set,
    to_model_label_set,
)
from server.main import create_app


class _WhitespaceTokenizer:
    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
        truncation: bool = False,
        padding: bool = False,
        return_tensors: Any = None,
    ) -> dict[str, Any]:
        offsets = [(match.start(), match.end()) for match in re.finditer(r"\S+", text)]
        return {
            "input_ids": list(range(len(offsets))),
            "offset_mapping": offsets,
        }


class _RegexProvider:
    def __init__(self, *, sleep_ms: int = 0) -> None:
        self.sleep_ms = sleep_ms
        self.calls = 0
        self.seen_labels: list[list[str]] = []
        self.inflight = 0
        self.peak_inflight = 0
        self._lock = threading.Lock()

    def detect(self, *, text: str, labels, threshold=0.5, flat_ner=True, multi_label=False):
        with self._lock:
            self.calls += 1
            self.seen_labels.append(list(labels))
            self.inflight += 1
            self.peak_inflight = max(self.peak_inflight, self.inflight)
        try:
            if self.sleep_ms:
                time.sleep(self.sleep_ms / 1000.0)
            entities = []
            for label, pattern in {
                "name": r"\b(?:Jane Doe|Maria Schmidt|Alice Johnson)\b",
                "email": r"\b\S+@\S+\.\S+\b",
                "employee_id": r"\bEMP-\d+\b",
                "phone_number": r"\b555-\d{4}\b",
            }.items():
                if label not in labels:
                    continue
                for match in re.finditer(pattern, text):
                    entities.append(
                        {
                            "start": match.start(),
                            "end": match.end(),
                            "text": match.group(),
                            "label": label,
                            "score": 0.95,
                        }
                    )
            # Obvious invalid structured prediction to prove sanity checks run.
            if "email" in labels and "not-an-email" in text:
                start = text.index("not-an-email")
                entities.append(
                    {
                        "start": start,
                        "end": start + len("not-an-email"),
                        "text": "not-an-email",
                        "label": "email",
                        "score": 0.99,
                    }
                )
            return entities
        finally:
            with self._lock:
                self.inflight -= 1

    def healthcheck(self):
        return {"status": "ok"}

    def close(self) -> None:
        return None


def _service(provider: _RegexProvider | None = None, *, max_text_tokens: int = 512):
    return ComplianceRedactionService(
        provider=provider or _RegexProvider(),
        model_name="knowledgator/gliner-pii-large-v1.0",
        tokenizer=_WhitespaceTokenizer(),
        max_text_tokens=max_text_tokens,
        max_model_len=768,
        max_chunk_concurrency=8,
    )


def test_gdpr_label_catalog_is_ordered_and_category_resolution_is_stable():
    labels = all_gdpr_labels()
    assert labels[:3] == ["person", "date_of_birth", "age"]
    assert len(labels) == len(set(labels))
    assert resolve_label_set(mode="open") == labels
    assert (
        resolve_label_set(mode="category", categories=["financial"]) == GDPR_CATEGORIES["financial"]
    )
    assert resolve_label_set(
        mode="category",
        categories=["financial"],
        labels=["email", "employee_id"],
    ) == ["email", "employee_id"]
    assert model_label_alias("person") == "name"
    assert model_label_alias("social_security_number") == "ssn"
    assert model_label_alias("unknown") == "unknown"
    model_labels, reverse = to_model_label_set(["person", "email"])
    assert model_labels == ["name", "email"]
    assert reverse == {"name": "person", "email": "email"}


def test_compliance_schema_surfaces_model_alias_metadata():
    schema = compliance_schema_metadata()

    assert schema["labels"] == all_gdpr_labels()
    assert schema["model_label_aliases"]["person"]["model_label"] == "name"
    assert schema["model_label_aliases"]["person"]["single_label_benchmark_f1"] == 1.0


def test_compliance_service_uses_model_aliases_but_returns_canonical_labels():
    provider = _RegexProvider()
    service = _service(provider)

    response = service.redact(
        ComplianceRedactionRequest(
            text="Patient Maria Schmidt uses email maria@example.de.",
            mode="category",
            labels=["person", "email"],
            redact=True,
            redaction_mode="mask",
            include_original_text=False,
        )
    )

    assert provider.seen_labels == [["name", "email"]]
    assert response.labels_used == ["person", "email"]
    assert response.unique_labels == ["email", "person"]
    assert response.original_text is None
    assert response.redacted_text == "Patient [PERSON] uses email [EMAIL]."
    assert [(entity.text, entity.label) for entity in response.entities] == [
        ("Maria Schmidt", "person"),
        ("maria@example.de", "email"),
    ]
    assert response.entities[0].metadata == {"model_label": "name"}


def test_compliance_service_redacts_and_applies_sanity_checks():
    service = _service()
    response = service.redact(
        ComplianceRedactionRequest(
            text="Contact jane@example.com but ignore not-an-email.",
            labels=["email"],
            redact=True,
            redaction_mode="mask",
        )
    )

    assert response.entity_count == 1
    assert response.entities[0].text == "jane@example.com"
    assert response.redacted_text == "Contact [EMAIL] but ignore not-an-email."
    assert response.timings_ms["vllm_request_ms"] >= 0
    assert response.chunks_processed == 1


def test_custom_regex_override_wins_overlap_dedupe():
    service = _service()
    response = service.redact(
        ComplianceRedactionRequest(
            text="Employee EMP-123 joined.",
            labels=["employee_id"],
            custom_labels=[{"label_name": "custom.employee_number", "extractor": r"EMP-\d+"}],
        )
    )

    assert response.entity_count == 1
    assert response.entities[0].label == "custom.employee_number"
    assert response.entities[0].source == "custom_regex"


def test_chunk_detection_runs_concurrently_not_sequentially():
    provider = _RegexProvider(sleep_ms=100)
    service = _service(provider, max_text_tokens=1)
    started = time.perf_counter()
    response = service.redact(
        ComplianceRedactionRequest(
            text="one two three four",
            labels=["email"],
        )
    )
    elapsed = time.perf_counter() - started

    assert response.chunks_processed == 4
    assert provider.calls == 4
    assert provider.peak_inflight > 1
    assert elapsed < 0.3, f"chunk inference appears serialized: elapsed={elapsed:.3f}s"


def test_compliance_route_offloads_redaction_and_keeps_health_unblocked():
    provider = _RegexProvider(sleep_ms=150)
    service = _service(provider)
    app = FastAPI()
    app.include_router(create_compliance_router(lambda: service))

    @app.get("/ping")
    async def ping():
        return {"status": "ok"}

    async def _run() -> tuple[float, int, int]:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            task = asyncio.create_task(
                client.post(
                    "/compliance/redact",
                    json={"text": "Contact jane@example.com", "labels": ["email"]},
                    timeout=5.0,
                )
            )
            await asyncio.sleep(0.02)
            started = time.perf_counter()
            ping_response = await client.get("/ping", timeout=2.0)
            ping_elapsed = time.perf_counter() - started
            redact_response = await task
        return ping_elapsed, ping_response.status_code, redact_response.status_code

    ping_elapsed, ping_status, redact_status = asyncio.run(_run())
    assert ping_status == 200
    assert redact_status == 200
    assert ping_elapsed < 0.12


def test_public_v1_compliance_aliases_are_mounted(monkeypatch):
    service = _service()
    monkeypatch.setenv("LATENCE_TRACE_LICENSE_REQUIRE", "false")
    monkeypatch.setenv("LATENCE_TRACE_DISABLE_WARMUP", "1")
    monkeypatch.setattr("server.main._get_compliance_service", lambda: service)

    app = create_app(profile=None)

    async def _run() -> tuple[int, int]:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            schema = await client.get("/v1/compliance/schema")
            redact = await client.post(
                "/v1/compliance/redact",
                json={
                    "text": "Contact jane@example.com",
                    "labels": ["email"],
                    "redact": True,
                },
            )
        return schema.status_code, redact.status_code

    assert asyncio.run(_run()) == (200, 200)


@pytest.mark.skipif(
    os.environ.get("LATENCE_TRACE_COMPLIANCE_PARITY") != "1",
    reason="set LATENCE_TRACE_COMPLIANCE_PARITY=1 with GPU deps to run native GLiNER parity",
)
def test_native_gliner_parity_against_vllm_endpoint():
    from gliner import GLiNER

    from latence_trace.providers.gliner import VllmFactoryDebertaGlinerProvider

    model_id = os.environ.get(
        "LATENCE_TRACE_COMPLIANCE_GLINER_MODEL",
        "knowledgator/gliner-pii-large-v1.0",
    )
    endpoint = os.environ["LATENCE_TRACE_COMPLIANCE_GLINER_ENDPOINT"]
    labels = all_gdpr_labels()
    fixtures = [
        "Jane Doe was born on 1980-01-01 and uses jane@example.com.",
        "The employee EMP-123 owns account DE89370400440532013000.",
    ]
    native = GLiNER.from_pretrained(model_id)
    provider = VllmFactoryDebertaGlinerProvider(endpoint=endpoint, model=model_id)
    try:
        for text in fixtures:
            expected = native.predict_entities(text, labels, threshold=0.5)
            actual = provider.detect(text=text, labels=labels, threshold=0.5)
            expected_triples = {(e["start"], e["end"], e["label"]) for e in expected}
            actual_triples = {(e["start"], e["end"], e["label"]) for e in actual}
            assert actual_triples == expected_triples
    finally:
        provider.close()


@pytest.mark.skipif(
    os.environ.get("LATENCE_TRACE_COMPLIANCE_PARITY") != "1",
    reason="set LATENCE_TRACE_COMPLIANCE_PARITY=1 with GPU deps to run native GLiNER parity",
)
def test_vllm_endpoint_parallel_requests_match_native_gliner():
    from gliner import GLiNER

    model_id = os.environ.get(
        "LATENCE_TRACE_COMPLIANCE_GLINER_MODEL",
        "knowledgator/gliner-pii-large-v1.0",
    )
    native_model_id = os.environ.get("LATENCE_TRACE_COMPLIANCE_NATIVE_MODEL", model_id)
    endpoint = os.environ["LATENCE_TRACE_COMPLIANCE_GLINER_ENDPOINT"]
    labels = ["person", "date_of_birth", "email", "employee_id", "organization"]
    text = (
        "Jane Doe was born on 1980-01-01. Email jane@example.com. "
        "Employee id EMP-123 works at Acme Corp."
    )

    native = GLiNER.from_pretrained(native_model_id)
    expected = native.predict_entities(text, labels, threshold=0.5)
    expected_triples = {(e["start"], e["end"], e["label"], e["text"]) for e in expected}

    async def _run() -> list[set[tuple[int, int, str, str]]]:
        async with httpx.AsyncClient(timeout=120.0) as client:
            payload = {
                "model": model_id,
                "task": "plugin",
                "data": {
                    "text": text,
                    "labels": labels,
                    "threshold": 0.5,
                    "flat_ner": True,
                    "multi_label": False,
                },
            }
            responses = await asyncio.gather(
                *[client.post(f"{endpoint}/pooling", json=payload) for _ in range(8)]
            )
        triples = []
        for response in responses:
            assert response.status_code == 200
            data = response.json()["data"]
            triples.append({(e["start"], e["end"], e["label"], e["text"]) for e in data})
        return triples

    actual_triples = asyncio.run(_run())
    assert all(triples == expected_triples for triples in actual_triples)
    assert len({tuple(sorted(triples)) for triples in actual_triples}) == 1
