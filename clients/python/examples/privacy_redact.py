"""Redact private data with the TRACE privacy product API."""

from latence_trace_client import LatenceTraceClient


with LatenceTraceClient() as client:
    result = client.privacy.redact(
        text="Contact Jane Doe at jane@example.com about invoice DE89370400440532013000.",
        labels=["person", "email", "iban"],
        include_original_text=False,
    )
    print(result.redacted_text)
    print(result.entity_count, result.unique_labels)
