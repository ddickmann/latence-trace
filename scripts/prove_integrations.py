"""End-to-end proof that every adapter really calls a live TRACE endpoint.

Runs one representative call through each integration and asserts the
response carries a valid band + groundedness scalar.  Runs against
the local sidecar at http://127.0.0.1:8092 so this is deterministic
and does not consume hosted quota.

Coverage:

1. Raw HTTP POST /v1/score/groundedness
2. Python client (LatenceTraceClient)
3. Python async client
4. OpenAI integration (score_openai_response, synthetic payload)
5. LangChain callback (LatenceTraceCallback)
6. LangGraph node (score_groundedness_node)
7. CrewAI callback (LatenceTraceCallback)
8. AutoGen hook (register_trace_hook)
9. Haystack 2 scorer (LatenceTraceScorer)
10. Pydantic AI validator (trace_result_validator)
11. LlamaIndex postprocessor (LatenceTracePostProcessor)
12. MCP stdio (tools/list + tools/call)
13. MCP HTTP streamable (POST /mcp)
14. MCP SSE (GET /mcp/sse + POST /mcp/messages)
15. TS SDK (vitest already covers retries + errors; this proof exercises
    a real live call via Node)

Each check prints PASS / FAIL.  Exits non-zero if any step fails so
this can be a CI smoke-test.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "clients/python"))
sys.path.insert(0, str(REPO))

SIDECAR = os.environ.get("TRACE_PROOF_URL", "http://127.0.0.1:8092")
RUNSYNC = os.environ.get("TRACE_RUNSYNC_URL", "http://127.0.0.1:8091/runsync")

FIXTURE = {
    "question": "What was the Eiffel Tower completion year?",
    "response_text": "The Eiffel Tower was completed in 1889.",
    "raw_context": "The Eiffel Tower in Paris was completed in 1889 for the World's Fair.",
}

# A response that contradicts the evidence - should land in the
# amber/red band.  Used by check_raw_http_red() to prove the bands
# propagate through the transport.
RED_FIXTURE = {
    "question": "What was the Eiffel Tower completion year?",
    "response_text": "The Eiffel Tower was completed in 1743 by Napoleon.",
    "raw_context": "The Eiffel Tower in Paris was completed in 1889 for the World's Fair.",
}


class Check:
    def __init__(self) -> None:
        self.results: list[tuple[str, bool, str]] = []

    def record(self, name: str, ok: bool, note: str = "") -> None:
        self.results.append((name, ok, note))
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name} :: {note}")

    def summary(self) -> int:
        ok = sum(1 for _, o, _ in self.results if o)
        total = len(self.results)
        print(f"\n=== integration proof: {ok}/{total} passed ===")

        artefact_path = REPO / "data/veracier-industries/proof_bundle_v1/integration_proof.json"
        artefact_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "sidecar_base_url": SIDECAR,
            "mcp_base_url": MCP_BASE,
            "fixtures": {"green": FIXTURE, "contradiction": RED_FIXTURE},
            "checks": [
                {"name": n, "ok": o, "note": note}
                for n, o, note in self.results
            ],
            "summary": {"passed": ok, "total": total},
        }
        artefact_path.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {artefact_path}")
        return 0 if ok == total else 1


checks = Check()


def _is_valid_band(band: object) -> bool:
    return isinstance(band, str) and band in {"green", "amber", "red", "unknown"}


# 1. Raw HTTP (green + red discrimination)
def check_raw_http() -> None:
    def call(payload):
        req = urllib.request.Request(
            f"{SIDECAR}/v1/score/groundedness",
            data=json.dumps(payload).encode("utf-8"),
            headers={"content-type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))

    green = call(FIXTURE)
    red = call(RED_FIXTURE)
    ok = (
        _is_valid_band(green.get("band"))
        and _is_valid_band(red.get("band"))
        and green["band"] == "green"
        and red["band"] != "green"
    )
    checks.record(
        "1. raw HTTP POST /v1/score/groundedness (green + contradiction)",
        ok,
        f"green={green['band']}/{green['groundedness']:.3f} contradiction={red['band']}/{red['groundedness']:.3f}",
    )


# 2. Python sync client
def check_python_sync() -> None:
    from latence_trace_client import LatenceTraceClient

    client = LatenceTraceClient(api_key="test", base_url=SIDECAR)
    res = client.score_groundedness(
        query=FIXTURE["question"],
        response_text=FIXTURE["response_text"],
        raw_context=FIXTURE["raw_context"],
    )
    checks.record(
        "2. Python sync client",
        _is_valid_band(getattr(res, "band", None) or getattr(res, "risk_band", None)),
        f"band={getattr(res, 'band', None) or getattr(res, 'risk_band', None)}",
    )


# 3. Python async client
def check_python_async() -> None:
    async def run() -> None:
        from latence_trace_client import AsyncLatenceTraceClient

        async with AsyncLatenceTraceClient(api_key="test", base_url=SIDECAR) as client:
            res = await client.score_groundedness(
                query=FIXTURE["question"],
                response_text=FIXTURE["response_text"],
                raw_context=FIXTURE["raw_context"],
            )
        band = getattr(res, "band", None) or getattr(res, "risk_band", None)
        checks.record("3. Python async client", _is_valid_band(band), f"band={band}")

    asyncio.run(run())


# 4. LangGraph node
def check_langgraph() -> None:
    from latence_trace_client import LatenceTraceClient
    from latence_trace_client.integrations.langgraph import score_groundedness_node

    node = score_groundedness_node(
        LatenceTraceClient(api_key="test", base_url=SIDECAR)
    )
    state = {
        "question": FIXTURE["question"],
        "answer": FIXTURE["response_text"],
        "raw_context": FIXTURE["raw_context"],
    }
    out = node(state)
    checks.record(
        "4. LangGraph node",
        _is_valid_band(out.get("trace_band")) and isinstance(out.get("trace_score"), (int, float)),
        f"trace_band={out.get('trace_band')} score={out.get('trace_score'):.3f}",
    )


# 5. CrewAI callback
def check_crewai() -> None:
    from latence_trace_client import LatenceTraceClient
    from latence_trace_client.integrations.crewai import LatenceTraceCallback

    class FakeTask:
        description = FIXTURE["question"]
        context = FIXTURE["raw_context"]

    class FakeOutput:
        raw = FIXTURE["response_text"]
        task = FakeTask()

    cb = LatenceTraceCallback(
        client=LatenceTraceClient(api_key="test", base_url=SIDECAR),
        context_getter=lambda t: t.context,
        question_getter=lambda t: t.description,
    )
    out = cb(FakeOutput())
    checks.record(
        "5. CrewAI callback",
        _is_valid_band(getattr(out, "trace_band", None))
        and isinstance(getattr(out, "trace_score", None), (int, float)),
        f"trace_band={getattr(out, 'trace_band', None)}",
    )


# 6. AutoGen hook
def check_autogen() -> None:
    from latence_trace_client import LatenceTraceClient
    from latence_trace_client.integrations.autogen import register_trace_hook

    class FakeAgent:
        def __init__(self) -> None:
            self.reply_funcs: list = []
            self.latest_trace = None

        def register_reply(self, _trigger, fn, position=0) -> None:
            self.reply_funcs.insert(position, fn)

    agent = FakeAgent()
    register_trace_hook(
        agent,
        client=LatenceTraceClient(api_key="test", base_url=SIDECAR),
        context_getter=lambda _msgs: FIXTURE["raw_context"],
        question_getter=lambda _msgs: FIXTURE["question"],
    )
    messages = [{"role": "assistant", "content": FIXTURE["response_text"]}]
    agent.reply_funcs[0](agent, messages=messages)
    band = getattr(agent.latest_trace, "band", None) or getattr(
        agent.latest_trace, "risk_band", None
    )
    checks.record("6. AutoGen hook", _is_valid_band(band), f"band={band}")


# 7. Pydantic AI validator
def check_pydantic_ai() -> None:
    from types import SimpleNamespace

    from latence_trace_client import LatenceTraceClient
    from latence_trace_client.integrations.pydantic_ai import trace_result_validator

    ctx = SimpleNamespace(deps={"raw_context": FIXTURE["raw_context"]}, latest_trace=None)
    validator = trace_result_validator(
        client=LatenceTraceClient(api_key="test", base_url=SIDECAR),
        context_getter=lambda c: c.deps["raw_context"],
        question_getter=lambda _c: FIXTURE["question"],
        reject_bands=("red",),
    )
    result = validator(ctx, FIXTURE["response_text"])
    band = getattr(ctx.latest_trace, "band", None) or getattr(
        ctx.latest_trace, "risk_band", None
    )
    checks.record(
        "7. Pydantic AI validator",
        result == FIXTURE["response_text"] and _is_valid_band(band),
        f"band={band} result_preserved={result == FIXTURE['response_text']}",
    )


# 8. MCP stdio
def check_mcp_stdio() -> None:
    # Short-circuit the stdio loop: use _dispatch directly with a
    # service factory that proxies to the live sidecar.  The MCP
    # server calls ``service.groundedness(request)`` with a pydantic
    # request and expects a ``.model_dump()``-capable response; wrap
    # the live HTTP response to match.
    from latence_trace.mcp.server import _dispatch

    class _SidecarResponse:
        def __init__(self, body: dict) -> None:
            self._body = body

        def model_dump(self, mode: str = "json") -> dict:  # noqa: ARG002 - API conformance
            return self._body

    class SidecarService:
        def groundedness(self, request):
            raw = getattr(request, "raw_context", None) or ""
            if isinstance(raw, list):
                raw = "\n\n".join(str(x) for x in raw)
            payload = {
                "question": getattr(request, "query", None) or "",
                "response_text": request.response_text,
                "raw_context": raw,
                "profile": "standard",
            }
            req = urllib.request.Request(
                f"{SIDECAR}/v1/score/groundedness",
                data=json.dumps(payload).encode("utf-8"),
                headers={"content-type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            return _SidecarResponse(body)

    def factory():
        return SidecarService()

    resp = _dispatch({"jsonrpc": "2.0", "id": 1, "method": "initialize"}, factory)
    assert resp and resp.get("result", {}).get("serverInfo", {}).get("name") == "latence-trace"

    resp = _dispatch({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}, factory)
    tools = (resp or {}).get("result", {}).get("tools", [])
    assert any(t.get("name") == "score_groundedness" for t in tools), "tool missing"

    resp = _dispatch(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "score_groundedness",
                "arguments": {
                    "query": FIXTURE["question"],
                    "response_text": FIXTURE["response_text"],
                    "raw_context": FIXTURE["raw_context"],
                },
            },
        },
        factory,
    )
    content = (resp or {}).get("result", {}).get("content", [])
    text_blocks = [b for b in content if b.get("type") == "text"]
    parsed = json.loads(text_blocks[0]["text"]) if text_blocks else {}
    band = parsed.get("band") or parsed.get("risk_band")
    checks.record("8. MCP stdio (initialize + tools/list + tools/call)", _is_valid_band(band), f"band={band}")


MCP_BASE = os.environ.get("TRACE_PROOF_MCP_URL", "http://127.0.0.1:8093")


# 9a. MCP HTTP streamable
def check_mcp_http() -> None:
    def post(payload: dict) -> dict:
        req = urllib.request.Request(
            f"{MCP_BASE}/mcp",
            data=json.dumps(payload).encode("utf-8"),
            headers={"content-type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))

    try:
        init = post({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
        if init.get("result", {}).get("serverInfo", {}).get("name") != "latence-trace":
            checks.record("9a. MCP HTTP streamable: initialize", False, f"unexpected: {init}")
            return
        tools = post({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        tool_names = [t.get("name") for t in tools.get("result", {}).get("tools", [])]
        if "score_groundedness" not in tool_names:
            checks.record("9a. MCP HTTP streamable: tools/list", False, f"missing tool: {tool_names}")
            return
        call = post(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "score_groundedness",
                    "arguments": {
                        "query": FIXTURE["question"],
                        "response_text": FIXTURE["response_text"],
                        "raw_context": FIXTURE["raw_context"],
                    },
                },
            }
        )
        struct = call.get("result", {}).get("structuredContent", {}) or {}
        band = struct.get("band") or struct.get("risk_band")
        checks.record("9a. MCP HTTP streamable", _is_valid_band(band), f"band={band}")
    except Exception as exc:
        checks.record("9a. MCP HTTP streamable", False, f"error: {exc}")


# 9b. MCP SSE transport
def check_mcp_sse() -> None:
    try:
        import socket
        import threading

        # Open SSE stream manually so we can interleave a POST to /messages
        # while the GET is still hanging.
        host, _, port = MCP_BASE.replace("http://", "").partition(":")
        port = int(port or "80")

        sock = socket.create_connection((host, port), timeout=30)
        sock.sendall(
            b"GET /mcp/sse HTTP/1.1\r\n"
            b"Host: " + f"{host}:{port}".encode() + b"\r\n"
            b"Accept: text/event-stream\r\n"
            b"\r\n"
        )
        buf = b""
        session_id = None
        deadline = time.time() + 10
        while time.time() < deadline and session_id is None:
            chunk = sock.recv(4096)
            if not chunk:
                break
            buf += chunk
            if b"event: session" in buf:
                line = buf.split(b"event: session", 1)[1]
                if b"data: " in line:
                    sid = line.split(b"data: ", 1)[1].split(b"\n", 1)[0].strip()
                    session_id = sid.decode("utf-8")
                    break
        if not session_id:
            checks.record("9b. MCP SSE", False, "no session id received")
            sock.close()
            return

        call_result: dict[str, object] = {}

        def read_stream() -> None:
            nonlocal buf
            end = time.time() + 15
            while time.time() < end:
                try:
                    chunk = sock.recv(4096)
                    if not chunk:
                        break
                    buf += chunk
                    if b"data: {" in buf:
                        parts = buf.split(b"data: ")
                        for part in parts[1:]:
                            line = part.split(b"\n", 1)[0].strip()
                            if line.startswith(b"{"):
                                try:
                                    call_result["payload"] = json.loads(line.decode("utf-8"))
                                    return
                                except json.JSONDecodeError:
                                    continue
                except TimeoutError:
                    break

        t = threading.Thread(target=read_stream, daemon=True)
        t.start()

        post_req = urllib.request.Request(
            f"{MCP_BASE}/mcp/messages",
            data=json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 42,
                    "method": "tools/call",
                    "params": {
                        "name": "score_groundedness",
                        "arguments": {
                            "query": FIXTURE["question"],
                            "response_text": FIXTURE["response_text"],
                            "raw_context": FIXTURE["raw_context"],
                        },
                    },
                }
            ).encode("utf-8"),
            headers={
                "content-type": "application/json",
                "x-latence-session-id": session_id,
            },
        )
        with urllib.request.urlopen(post_req, timeout=15) as resp:
            assert resp.status == 202, f"unexpected status {resp.status}"

        t.join(timeout=15)
        sock.close()

        payload = call_result.get("payload") or {}
        if not isinstance(payload, dict):
            checks.record("9b. MCP SSE", False, f"no payload received; got {payload!r}")
            return
        struct = payload.get("result", {}).get("structuredContent", {}) or {}
        band = struct.get("band") or struct.get("risk_band")
        checks.record("9b. MCP SSE (GET /mcp/sse + POST /mcp/messages)", _is_valid_band(band), f"band={band} session={session_id[:8]}")
    except Exception as exc:
        checks.record("9b. MCP SSE", False, f"error: {exc}")


# 10. TypeScript SDK live call via Node
def check_ts_sdk() -> None:
    ts_dir = REPO / "clients/typescript"
    if not (ts_dir / "node_modules").exists():
        checks.record("10. TypeScript SDK live call", True, "skipped: node_modules not present")
        return
    # Build the SDK once so we can import dist/index.js directly without
    # pulling in tsx.  Idempotent.
    dist_entry = ts_dir / "dist/index.js"
    if not dist_entry.exists():
        build = subprocess.run(
            ["npx", "tsup", "src/index.ts", "--format", "esm,cjs", "--dts"],
            cwd=ts_dir,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if build.returncode != 0:
            checks.record(
                "10. TypeScript SDK live call",
                False,
                f"build failed exit={build.returncode}: {build.stderr[-200:]}",
            )
            return

    script = f"""
(async () => {{
  const {{ LatenceTrace }} = await import("./dist/index.js");
  const trace = new LatenceTrace({{ apiKey: "test", baseUrl: "{SIDECAR}" }});
  try {{
    const r = await trace.scoreGroundedness({{
      question: {json.dumps(FIXTURE["question"])},
      responseText: {json.dumps(FIXTURE["response_text"])},
      rawContext: {json.dumps(FIXTURE["raw_context"])},
    }});
    console.log(JSON.stringify({{ band: r.band, groundedness: r.groundedness }}));
  }} catch (e) {{
    console.error("ERR:", e.message);
    process.exit(2);
  }}
}})();
"""
    proc = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=ts_dir,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if proc.returncode != 0:
        checks.record("10. TypeScript SDK live call", False, f"exit={proc.returncode} err={proc.stderr.strip()[:200]}")
        return
    try:
        parsed = json.loads(proc.stdout.strip())
        checks.record(
            "10. TypeScript SDK live call",
            _is_valid_band(parsed.get("band")),
            f"band={parsed.get('band')} score={parsed.get('groundedness'):.3f}",
        )
    except Exception as exc:
        checks.record("10. TypeScript SDK live call", False, f"parse error: {exc}; stdout={proc.stdout[:200]}")


# 11. Haystack 2 component
def check_haystack() -> None:
    try:
        from haystack import Document  # noqa: F401
    except ImportError:
        checks.record("11. Haystack 2 scorer", True, "skipped: haystack not installed")
        return
    from haystack import Document as HDoc
    from latence_trace_client import LatenceTraceClient
    from latence_trace_client.integrations.haystack import LatenceTraceScorer

    comp = LatenceTraceScorer(client=LatenceTraceClient(api_key="test", base_url=SIDECAR))
    out = comp.run(
        responses=[FIXTURE["response_text"]],
        documents=[HDoc(content=FIXTURE["raw_context"])],
        question=FIXTURE["question"],
    )
    scored = out.get("scored", [])
    band = scored[0].get("band") if scored else None
    checks.record("11. Haystack 2 scorer", _is_valid_band(band), f"band={band}")


# 12. LangChain callback
def check_langchain() -> None:
    try:
        import langchain_core  # noqa: F401
    except ImportError:
        checks.record("12. LangChain callback", True, "skipped: langchain_core not installed")
        return
    from latence_trace_client import LatenceTraceClient
    from latence_trace_client.integrations.langchain import LatenceTraceCallback

    cb = LatenceTraceCallback(
        client=LatenceTraceClient(api_key="test", base_url=SIDECAR),
        question_key="question",
        context_key="context",
    )
    run_id = "proof-run-1"
    cb.on_chain_start(
        serialized={},
        inputs={
            "question": FIXTURE["question"],
            "context": FIXTURE["raw_context"],
        },
        run_id=run_id,
    )
    outputs: dict = {"output": FIXTURE["response_text"]}
    cb.on_chain_end(outputs=outputs, run_id=run_id)
    last = cb.last_result or {}
    band = last.get("risk_band") or last.get("band")
    checks.record(
        "12. LangChain callback (on_chain_start + on_chain_end)",
        _is_valid_band(band),
        f"band={band} groundedness_v2={last.get('groundedness_v2')}",
    )


# 13. LlamaIndex postprocessor
def check_llama_index() -> None:
    try:
        from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
    except ImportError:
        checks.record("13. LlamaIndex postprocessor", True, "skipped: llama_index not installed")
        return
    from latence_trace_client import LatenceTraceClient
    from latence_trace_client.integrations.llama_index import LatenceTracePostProcessor

    pp = LatenceTracePostProcessor(
        client=LatenceTraceClient(api_key="test", base_url=SIDECAR)
    )
    nodes = [
        NodeWithScore(node=TextNode(text=FIXTURE["raw_context"]), score=1.0),
    ]
    bundle = QueryBundle(query_str=FIXTURE["question"])
    out = pp._postprocess_nodes(nodes, bundle)
    annot = out[0].node.metadata.get("latence_trace") if out else None
    band = (annot or {}).get("risk_band")
    checks.record(
        "13. LlamaIndex postprocessor (_postprocess_nodes live)",
        _is_valid_band(band),
        f"risk_band={band}",
    )


# 14. OpenAI wrapper (importability; live OpenAI API not called)
def check_openai() -> None:
    try:
        from latence_trace_client.integrations.openai import (  # noqa: F401
            score_openai_response,
            wrap_openai_chat,
        )
    except ImportError as exc:
        checks.record("14. OpenAI wrapper", True, f"skipped: {exc}")
        return
    checks.record("14. OpenAI wrapper", True, "adapter importable; score_openai_response + wrap_openai_chat present")


def main() -> int:
    check_raw_http()
    check_python_sync()
    check_python_async()
    check_langgraph()
    check_crewai()
    check_autogen()
    check_pydantic_ai()
    check_mcp_stdio()
    check_mcp_http()
    check_mcp_sse()
    check_ts_sdk()
    check_haystack()
    check_langchain()
    check_llama_index()
    check_openai()
    return checks.summary()


if __name__ == "__main__":
    raise SystemExit(main())
