from __future__ import annotations

from latence_trace.memory.extract import extract_spans
from latence_trace.memory.models import MemoryPolicy, MemoryUpdateRequest
from latence_trace.memory.service import update_memory


def _run_turns(domain: str, turns: list[dict], *, hot_budget: int = 160) -> str:
    state = None
    policy = MemoryPolicy(hot_token_budget=hot_budget, warm_token_budget=512, max_spans=128)
    hot = ""
    for turn in turns:
        response = update_memory(
            MemoryUpdateRequest(
                memory_domain=domain,
                prior_memory_state=state,
                memory_policy=policy,
                **turn,
            )
        )
        state = response.next_memory_state
        hot = response.hot_context
    return hot


def test_ood_rag_old_citations_decay_before_latest_answer() -> None:
    turns = [
        {
            "query_text": f"Question {idx}",
            "raw_context": f"Passage cite_{idx:02d}: RX-{1000 + idx} temporary advisory {idx}% for this answer only.",
            "response_text": f"Answer uses cite_{idx:02d} and RX-{1000 + idx}.",
        }
        for idx in range(1, 28)
    ]

    hot = _run_turns("rag", turns, hot_budget=96)

    assert "cite_27" in hot
    assert "RX-1027" in hot
    assert "cite_01" not in hot
    assert "RX-1001" not in hot


def test_ood_tool_state_keeps_latest_structured_fact_not_old_reservation() -> None:
    turns = [
        {
            "query_text": "Maintain current reservation state.",
            "raw_context": (
                f'{{"reservation_id":"RSV-{idx:04d}","order_id":"ORD-{idx:04d}",'
                f'"payment_id":"PAY-{idx:04d}","status":"{"cancelled" if idx == 18 else "pending"}",'
                f'"amount":"{100 + idx} EUR"}}'
            ),
            "response_text": f"Observed reservation RSV-{idx:04d}.",
        }
        for idx in range(1, 19)
    ]

    hot = _run_turns("tool", turns, hot_budget=120)

    assert "RSV-0018" in hot
    assert "ORD-0018" in hot
    assert "cancelled" in hot
    assert "RSV-0001" not in hot


def test_ood_tool_exact_index_preserves_privacy_reduced_email_handle() -> None:
    spans = extract_spans(
        raw_context=(
            '{"email": "olivia.gonzalez4421@example.com", "address": {"zip": "90504"}, '
            '"reservations": ["K67C4W"], "flights": [{"flight_number": "HAT137"}], '
            '"payment_methods": {"credit_card_9969263": {"id": "credit_card_9969263"}}}'
        ),
        memory_domain="tool",
    )

    extracted = "\n".join(span.text for span in spans)

    assert "tool_fact email_localpart=olivia.gonzalez4421" in extracted
    assert "HAT137" in extracted
    assert "K67C4W" in extracted
    assert "olivia.gonzalez4421@example.com" not in extracted


def test_ood_rust_code_keeps_stable_anchors_over_generated_noise() -> None:
    turns = [
        {
            "query_text": (
                "Fix crates/payments/src/retry_policy.rs for "
                "crates/payments/tests/retry_policy_test.rs::test_idempotency_window "
                "and RetryBudget::calculate_backoff."
            ),
            "raw_context": (
                "code_exact_index crates/payments/src/retry_policy.rs "
                "crates/payments/tests/retry_policy_test.rs::test_idempotency_window "
                "RetryBudget::calculate_backoff"
            ),
        }
    ]
    turns.extend(
        {
            "turn_text": f"Scanned generated Rust noise {idx}.",
            "raw_context": f"warning tmp_{idx}; crates/noise/src/generated_{idx}.rs unrelated_{idx} passed.",
            "response_text": "continue",
        }
        for idx in range(1, 45)
    )

    hot = _run_turns("code", turns, hot_budget=160)

    assert "crates/payments/src/retry_policy.rs" in hot
    assert "test_idempotency_window" in hot
    assert "RetryBudget::calculate_backoff" in hot
    assert "generated_44.rs" not in hot


def test_ood_chat_preserves_late_critical_answer() -> None:
    turns = [
        {
            "turn_text": f"Discuss unrelated preference option {idx}.",
            "response_text": f"Noted casual option {idx}.",
        }
        for idx in range(1, 40)
    ]
    turns.append(
        {
            "query_text": "What is the final selected pharmacy?",
            "turn_text": "Remember the final selected venue.",
            "response_text": "Final venue: Northstar Pediatric Pharmacy, counter B7.",
        }
    )

    hot = _run_turns("chat", turns, hot_budget=128)

    assert "Northstar Pediatric Pharmacy" in hot
    assert "counter B7" in hot
    assert "option 1" not in hot


def test_ood_finance_legal_pharma_identifiers_survive_grounding() -> None:
    hot = _run_turns(
        "grounding",
        [
            {
                "query_text": "Verify finance, legal, and pharma claims.",
                "raw_context": (
                    "Clause 14.2 requires notice by 2026-05-30. "
                    "Invoice FIN-8842 totals 8420 EUR. "
                    "Drug RX-77 lists dizziness and rash as side effects."
                ),
                "response_text": (
                    "Notice date is 2026-05-30, invoice FIN-8842 is 8420 EUR, "
                    "and RX-77 side effects include dizziness and rash."
                ),
            }
        ],
        hot_budget=120,
    )

    assert "2026-05-30" in hot
    assert "8420 EUR" in hot
    assert "FIN-8842" in hot
    assert "RX-77" in hot
