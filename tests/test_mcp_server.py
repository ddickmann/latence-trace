"""PA6 stdio MCP adapter tests.

Drives ``run_stdio_loop`` with an in-memory stdin/stdout pair and a
fake :class:`GroundednessService` so we can verify the JSON-RPC
handshake, tool discovery, and tool invocation contracts without
booting the real encoder / NLI stack.
"""

from __future__ import annotations

import io
import json
from typing import Any, Dict, Iterator, List, Sequence

import pytest
import torch

from latence_trace.api.service import GroundednessService
from latence_trace.mcp.server import run_stdio_loop


class _DeterministicEncoder:
    @property
    def model_name(self) -> str:
        return "mcp-test-encoder"

    @property
    def model_name_or_path(self) -> str:
        return self.model_name

    def encode(self, sentences, *, is_query=False, prompt_name=None, **kwargs):
        outs = []
        for sentence in sentences:
            seed = abs(hash(sentence)) % (2**31 - 1)
            generator = torch.Generator().manual_seed(seed)
            token_count = max(2, min(8, len((sentence or "x").split())))
            outs.append(torch.randn(token_count, 32, generator=generator))
        return outs


def _service_factory():
    return GroundednessService(encoder_factory=lambda _name: _DeterministicEncoder())


def _drive_stdio(messages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    stdin_text = "\n".join(json.dumps(m) for m in messages) + "\n"
    stdin = io.StringIO(stdin_text)
    stdout = io.StringIO()
    rc = run_stdio_loop(service_factory=_service_factory, stdin=stdin, stdout=stdout)
    assert rc == 0
    out_lines = [line for line in stdout.getvalue().splitlines() if line.strip()]
    return [json.loads(line) for line in out_lines]


@pytest.fixture(autouse=True)
def _disable_nli(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("VOYAGER_GROUNDEDNESS_NLI_ENABLED", "0")
    yield


def test_initialize_handshake_returns_protocol_version_and_server_info():
    responses = _drive_stdio(
        [{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}]
    )
    assert len(responses) == 1
    body = responses[0]
    assert body["jsonrpc"] == "2.0"
    assert body["id"] == 1
    assert "protocolVersion" in body["result"]
    assert body["result"]["serverInfo"]["name"] == "latence-trace"
    assert body["result"]["serverInfo"]["version"] == "1.0.0"
    assert body["result"]["capabilities"]["tools"]["listChanged"] is False


def test_tools_list_advertises_score_groundedness_with_required_response_text():
    responses = _drive_stdio(
        [{"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}]
    )
    assert len(responses) == 1
    tools = responses[0]["result"]["tools"]
    assert len(tools) >= 1
    tool = next(t for t in tools if t["name"] == "score_groundedness")
    assert "inputSchema" in tool
    assert "response_text" in tool["inputSchema"]["properties"]
    assert "response_text" in tool["inputSchema"]["required"]
    assert "support_units" in tool["inputSchema"]["properties"]
    assert "attribution_mode" in tool["inputSchema"]["properties"]


def test_tools_call_invokes_score_groundedness_and_returns_structured_content():
    arguments = {
        "query_text": "When was Heinrich born?",
        "raw_context": "Heinrich was born in 1851 in Augsburg.",
        "response_text": "Heinrich was born in Augsburg in 1851.",
        "primary_metric": "reverse_context",
    }
    responses = _drive_stdio(
        [
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "score_groundedness", "arguments": arguments},
            }
        ]
    )
    assert len(responses) == 1
    body = responses[0]
    assert body["id"] == 3
    assert "result" in body
    result = body["result"]
    assert result["isError"] is False
    assert isinstance(result["content"], list) and result["content"]
    assert result["content"][0]["type"] == "text"
    structured = result["structuredContent"]
    assert "scores" in structured
    assert "support_units" in structured


def test_tools_call_unknown_tool_returns_jsonrpc_error():
    responses = _drive_stdio(
        [
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {"name": "do_my_taxes", "arguments": {}},
            }
        ]
    )
    assert len(responses) == 1
    body = responses[0]
    assert body["id"] == 4
    assert "error" in body
    # Per JSON-RPC 2.0 the tool name is a *parameter* of ``tools/call``,
    # so an unknown tool maps to "Invalid params" (-32602), not
    # "Method not found" (-32601).
    assert body["error"]["code"] == -32602
    assert "do_my_taxes" in body["error"]["message"]


def test_tools_call_invalid_arguments_returns_validation_error():
    responses = _drive_stdio(
        [
            {
                "jsonrpc": "2.0",
                "id": 5,
                "method": "tools/call",
                "params": {
                    "name": "score_groundedness",
                    "arguments": {"primary_metric": "triangular"},  # missing query_text + response_text
                },
            }
        ]
    )
    assert len(responses) == 1
    body = responses[0]
    assert body["id"] == 5
    assert "error" in body
    assert body["error"]["code"] == -32602


def test_initialized_notification_does_not_produce_a_response():
    responses = _drive_stdio(
        [
            {"jsonrpc": "2.0", "method": "initialized", "params": {}},
            {"jsonrpc": "2.0", "id": 6, "method": "ping", "params": {}},
        ]
    )
    # Only the ping should produce a reply.
    assert len(responses) == 1
    assert responses[0]["id"] == 6


def test_malformed_json_returns_parse_error_without_crashing_loop():
    stdin = io.StringIO('this is not json\n{"jsonrpc": "2.0", "id": 7, "method": "ping"}\n')
    stdout = io.StringIO()
    rc = run_stdio_loop(service_factory=_service_factory, stdin=stdin, stdout=stdout)
    assert rc == 0
    lines = [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]
    assert len(lines) == 2
    assert "error" in lines[0]
    assert lines[0]["error"]["code"] == -32700
    assert lines[1]["id"] == 7


def test_unknown_method_with_id_returns_method_not_found():
    responses = _drive_stdio(
        [{"jsonrpc": "2.0", "id": 8, "method": "tools/unknown", "params": {}}]
    )
    assert len(responses) == 1
    assert "error" in responses[0]
    assert responses[0]["error"]["code"] == -32601


def test_closed_book_refusal_round_trips_through_mcp():
    arguments = {
        "response_text": "Heinrich was born in 1851.",
        "primary_metric": "reverse_context",
        "attribution_mode": "closed_book",
    }
    responses = _drive_stdio(
        [
            {
                "jsonrpc": "2.0",
                "id": 9,
                "method": "tools/call",
                "params": {"name": "score_groundedness", "arguments": arguments},
            }
        ]
    )
    assert len(responses) == 1
    structured = responses[0]["result"]["structuredContent"]
    assert structured.get("reason") == "no_premise_supplied"
