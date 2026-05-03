from __future__ import annotations

import pytest
from pydantic import ValidationError

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


def test_memory_policy_accepts_context_ratio_on_groundedness_request() -> None:
    request = GroundednessRequest(
        response_text="hello",
        enable_memory_shadow=True,
        memory_policy=MemoryPolicy(
            context_window_tokens=270_000,
            memory_context_ratio=0.4,
        ),
    )

    assert request.memory_policy is not None
    assert request.memory_policy.context_window_tokens == 270_000
    assert request.memory_policy.memory_context_ratio == 0.4


def test_memory_policy_rejects_full_context_ratio() -> None:
    with pytest.raises(ValidationError):
        MemoryPolicy(context_window_tokens=270_000, memory_context_ratio=1.0)


def test_memory_policy_accepts_adaptive_budget_controls() -> None:
    policy = MemoryPolicy(
        memory_budget_mode="adaptive",
        target_token_reduction=0.9,
        min_exact_critical_recall=0.99,
        min_survival_mass=0.95,
        recent_tail_token_budget=2_048,
    )

    assert policy.memory_budget_mode == "adaptive"
    assert policy.target_token_reduction == 0.9
    assert policy.recent_tail_token_budget == 2_048


def test_memory_policy_rejects_invalid_budget_mode() -> None:
    with pytest.raises(ValidationError):
        MemoryPolicy(memory_budget_mode="tiny")  # type: ignore[arg-type]
