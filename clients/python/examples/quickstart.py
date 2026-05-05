"""End-to-end quickstart for the latence-trace SDK."""

from __future__ import annotations

from latence_trace_client import LatenceTraceClient


def main() -> None:
    with LatenceTraceClient(
        base_url="http://localhost:8090",
        api_key=None,  # set via LATENCE_TRACE_API_KEY in production
    ) as client:
        result = client.grounding.rag(
            query="When was Newton born?",
            response_text="Newton was born in 1643.",
            raw_context=(
                "Sir Isaac Newton (25 December 1642 – 20 March 1726/27 OS) "
                "was an English mathematician, physicist, astronomer, alchemist..."
            ),
        )
        print("risk_band       =", result.risk_band.value)
        print("groundedness_v2 =", result.scores.groundedness_v2)
        print("coverage_score  =", result.scores.coverage_score_u)
        print("request_id      =", result.request_id)


if __name__ == "__main__":
    main()
