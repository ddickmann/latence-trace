"""Score a RAG answer with guard checks enabled and disabled."""

from latence_trace_client import LatenceTraceClient


with LatenceTraceClient() as client:
    guarded = client.grounding.rag(
        query="Can the refund be approved?",
        response_text="The refund is approved and will arrive in 48 hours.",
        raw_context="Refund status: pending manager approval.",
    )
    isolation = client.grounding.rag(
        query="Can the refund be approved?",
        response_text="The refund is approved and will arrive in 48 hours.",
        raw_context="Refund status: pending manager approval.",
        context_trust_enabled=False,
    )
    print(guarded.risk_band, guarded.context_trust_diagnostics)
    print(isolation.risk_band)
