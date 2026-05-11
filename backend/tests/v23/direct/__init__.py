"""Ring 1 — direct-invocation tests for v23-MCP tool functions.

Every MCP tool is a plain Python function inside
``backend.mcp.tools.{core,hitl,carve}`` (FastMCP decorator is a no-op
pass-through). These tests import + call directly without any JSON-RPC
transport — the fastest dev-loop ring.

Day-1 surface is 11 tools (see ``/tmp/research-v23-mcp-build-sequence.md``
D9.1–D9.11). One file per tool will land as those steps complete.
"""
