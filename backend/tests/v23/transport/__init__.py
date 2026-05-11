"""Ring 2 — JSON-RPC over stdio transport tests for v23-MCP.

Boots ``backend.mcp.v23_server`` as a subprocess, frames JSON-RPC over
stdin/stdout, asserts on raw envelopes. Catches schema mismatches,
encoding edges, GPU-lock semantics, concurrent-request ordering that
Ring 1 can never see.

Lands in D19 (FastMCP wiring) per
``/tmp/research-v23-mcp-build-sequence.md``.
"""
