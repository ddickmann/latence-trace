"""Compress text and chat messages."""

from latence_trace_client import LatenceTraceClient


with LatenceTraceClient() as client:
    text = client.compression.text(
        "Keep the refund status, invoice id, and policy exception. Remove repetition.",
        target_token_ratio=0.5,
    )
    messages = client.compression.messages(
        [
            {"role": "system", "content": "You are a careful support agent."},
            {"role": "user", "content": "Please check refund status and policy."},
        ]
    )
    print(text.compressed_text, text.tokens_saved)
    print(messages.compressed_messages)
