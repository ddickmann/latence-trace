"""Stdio Model Context Protocol adapter for latence-trace (PA6).

Exposes :func:`latence_trace.mcp.server.run_stdio_loop` so AI agent
runtimes (Cursor, Claude Desktop, custom frameworks) can register
``score_groundedness`` as a tool without speaking HTTP.
"""

from latence_trace.mcp.server import run_stdio_loop

__all__ = ["run_stdio_loop"]
