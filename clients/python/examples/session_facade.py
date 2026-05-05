"""Use SDK-managed session state with a stateless TRACE runtime."""

from latence_trace_client import FileSessionStorage, LatenceTraceClient


storage = FileSessionStorage(".trace-sessions")

with LatenceTraceClient() as client:
    session = client.session(session_id="agent-42", storage=storage)
    session.event("retrieval", "Refund policy and invoice status loaded.")
    session.memory_step(
        turn_text="Refund status pending manager approval; keep this constraint hot.",
        memory_domain="support",
    )
    result = session.rag(
        query="Can I approve the refund?",
        response_text="The refund should be reviewed before sending a promise.",
        raw_context="Refund status: pending manager approval.",
    )
    rollup = session.rollup(session_id=session.session_id)

print(result.risk_band)
print(rollup)
