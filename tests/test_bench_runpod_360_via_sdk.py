from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts import bench_runpod_360_via_sdk as via_sdk


class _FakeGrounding:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def rag(self, **kwargs):
        self.calls.append(("rag", kwargs))
        return {"risk_band": "green"}

    async def code(self, **kwargs):
        self.calls.append(("code", kwargs))
        return {"risk_band": "green"}


class _FakeSdk:
    def __init__(self) -> None:
        self.grounding = _FakeGrounding()
        self.rollup_calls: list[dict] = []

    async def rollup(self, **kwargs):
        self.rollup_calls.append(kwargs)
        return {"success": True}


@pytest.mark.asyncio
async def test_via_sdk_routes_rag_payload_through_new_latence_surface(monkeypatch) -> None:
    sdk = _FakeSdk()
    monkeypatch.setattr(via_sdk, "_SDK", sdk)

    result = await via_sdk._sdk_submit(
        SimpleNamespace(),
        None,
        {
            "input": {
                "query": "q",
                "response": "answer",
                "context": "ctx",
                "guard_check_enabled": False,
                "profile": "quality",
            }
        },
    )

    assert result["status"] == "COMPLETED"
    assert sdk.grounding.calls == [
        (
            "rag",
            {
                "query": "q",
                "response_text": "answer",
                "raw_context": "ctx",
                "context_trust_enabled": False,
                "extra": {"profile": "quality"},
            },
        )
    ]


@pytest.mark.asyncio
async def test_via_sdk_routes_code_and_rollup_through_new_latence_surface(monkeypatch) -> None:
    sdk = _FakeSdk()
    monkeypatch.setattr(via_sdk, "_SDK", sdk)

    code_result = await via_sdk._sdk_submit(
        SimpleNamespace(),
        None,
        {
            "input": {
                "scoring_mode": "code",
                "response": "answer",
                "context": "code",
                "response_language_hint": "python",
            }
        },
    )
    rollup_result = await via_sdk._sdk_submit(
        SimpleNamespace(),
        None,
        {"input": {"action": "rollup", "turns": [{"risk_band": "green"}]}},
    )

    assert code_result["status"] == "COMPLETED"
    assert rollup_result["status"] == "COMPLETED"
    assert sdk.grounding.calls[0] == (
        "code",
        {
            "response_text": "answer",
            "raw_context": "code",
            "extra": {"response_language_hint": "python"},
        },
    )
    assert sdk.rollup_calls == [{"turns": [{"risk_band": "green"}]}]
