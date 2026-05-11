"""Ring 3 fixtures — scripted MockOpus driver.

Until D21.1 the real ``MockOpus`` (research-v23-mcp-testing.md §4)
doesn't exist; this conftest provides a minimal scriptable driver so
placeholder flow tests can xfail with the right shape.

The :class:`ScriptedMockOpus` collects an expected ordered list of
``(tool_name, args_predicate)`` steps and replays them against a
callable (Ring 1 direct call or Ring 2 stdio call). Each ``step()``
returns the tool result; assertions live in the test body.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import pytest


@dataclass
class ScriptedMockOpus:
    """Replays a fixed sequence of tool calls against a dispatcher.

    Used by Ring 3 placeholder tests to assert the canonical Pattern-A
    cold-start flow (analyze → propose → inspect → export) survives any
    refactor of the underlying tools.
    """

    dispatcher: Callable[[str, dict[str, Any]], Any]
    transcript: list[tuple[str, dict[str, Any], Any]] = field(default_factory=list)

    def step(
        self,
        tool: str,
        args: dict[str, Any],
        expect: Callable[[Any], bool] | None = None,
    ) -> Any:
        result = self.dispatcher(tool, args)
        self.transcript.append((tool, args, result))
        if expect is not None:
            assert expect(result), f"{tool} response failed: {result!r}"
        return result


def _placeholder_dispatcher(tool: str, args: dict[str, Any]) -> Any:
    """Deliberate stub — every call raises ``NotImplementedError``.

    Forces tests that rely on real tool behaviour to xfail until D10+
    wires the real ``propose_stack`` / ``export_print_plan`` pipeline.
    """
    raise NotImplementedError(
        f"MockOpus.dispatch({tool!r}) — real tool wiring lands in D10+"
    )


@pytest.fixture
def mock_opus() -> ScriptedMockOpus:
    """Yield a :class:`ScriptedMockOpus` wired to a placeholder dispatcher."""
    return ScriptedMockOpus(dispatcher=_placeholder_dispatcher)
