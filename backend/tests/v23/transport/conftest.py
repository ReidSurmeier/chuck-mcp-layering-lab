"""Ring 2 fixtures — stdio MCP server subprocess + tiny JSON-RPC client.

The real FastMCP wiring lands in D19. Until then the fixture launches a
subprocess pointed at the current stub ``backend.mcp.v23_server`` which
does not yet speak JSON-RPC. That is fine: the only Ring 2 placeholder
test xfails until D19.1 lands the handshake.
"""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]


@dataclass
class MCPStdioClient:
    """Minimal JSON-RPC framing over a subprocess' stdin/stdout.

    Lives here in the conftest so tests can import the shape without
    pulling in FastMCP machinery. Real impl matches MCP framing
    (LSP-style ``Content-Length:`` headers) once D19.1 lands.
    """

    proc: subprocess.Popen
    _id: int = 0

    def call(self, method: str, params: dict[str, Any] | None = None, timeout: float = 30.0) -> dict[str, Any]:
        self._id += 1
        req = {
            "jsonrpc": "2.0",
            "id": self._id,
            "method": method,
            "params": params or {},
        }
        assert self.proc.stdin is not None
        assert self.proc.stdout is not None
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()
        # NOTE: real impl will read until matching ``id`` and obey
        # Content-Length framing. Placeholder reads one line so tests
        # can xfail cleanly until D19.1.
        line = self.proc.stdout.readline()
        return json.loads(line) if line.strip() else {}

    def close(self) -> None:
        try:
            self.proc.terminate()
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


@pytest.fixture
def mcp_stdio_client() -> Any:
    """Start the v23 server as a subprocess and yield a stdio client.

    XFAIL-safe: if the server can't be launched in stdio mode yet
    (pre-D19), the fixture still yields a client whose ``call()`` will
    return ``{}`` so xfail'd tests record a clean failure rather than
    an error during setup.
    """
    cmd = [sys.executable, "-m", "backend.mcp.v23_server"]
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    client = MCPStdioClient(proc=proc)
    try:
        yield client
    finally:
        client.close()
