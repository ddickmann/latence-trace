from __future__ import annotations

from latence_trace.api.models import GroundednessRequest
from latence_trace.memory.models import MemoryPolicy


def test_memory_fields_are_opt_in_on_groundedness_request() -> None:
    request = GroundednessRequest(response_text="hello")

    assert request.memory_state is None
    assert request.memory_policy is None
    assert request.enable_memory_shadow is False
    assert request.apply_memory_context is False


def test_memory_policy_validates_on_groundedness_request() -> None:
    request = GroundednessRequest(
        response_text="hello",
        enable_memory_shadow=True,
        memory_policy=MemoryPolicy(hot_token_budget=32),
    )

    assert request.memory_policy is not None
    assert request.memory_policy.hot_token_budget == 32
