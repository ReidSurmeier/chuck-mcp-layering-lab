"""Ring 2 placeholder — JSON-RPC ``tools/list`` over stdio.

Lands green at D19.2 (``test_tools_list_returns_11_day1_tools``). Until
the FastMCP lifespan + stdio glue arrives in D19.1 the v23_server stub
prints a banner and exits, so this xfails by design.
"""
from __future__ import annotations

import pytest


@pytest.mark.xfail(reason="awaits D19.1 — FastMCP stdio handshake")
def test_server_responds_to_tools_list(mcp_stdio_client) -> None:
    response = mcp_stdio_client.call("tools/list")
    assert isinstance(response, dict) and response, "no JSON-RPC response received"
    assert response.get("jsonrpc") == "2.0"
    assert "result" in response, response
    tools = response["result"].get("tools", [])
    names = {t.get("name") for t in tools}
    expected_subset = {"analyze_image", "propose_stack", "export_print_plan"}
    assert expected_subset <= names, f"missing tools: {expected_subset - names}"
