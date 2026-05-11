"""Ring 3 placeholder — Pattern-A cold-start flow.

Pattern A (research-v23-mcp-testing.md §4): the user hands Opus an
image and asks for a print plan. Canonical tool sequence:

    analyze_image → propose_stack → inspect_plan → export_print_plan

Lands green at D21.1 (``test_full_golden_path_on_reid_untitled_01``).
"""
from __future__ import annotations

import pytest


EXPECTED_FLOW: tuple[str, ...] = (
    "analyze_image",
    "propose_stack",
    "inspect_plan",
    "export_print_plan",
)


@pytest.mark.xfail(reason="awaits D10+ — real tool wiring through D21.1 flow harness")
def test_pattern_a_cold_start_flow(mock_opus) -> None:
    image_path = "/home/reidsurmeier/src/woodblock-reidsurmeier-wtf/corpus/reid_untitled_01/original.png"

    mock_opus.step("analyze_image", {"path": image_path})
    plan_id = mock_opus.step("propose_stack", {"image_path": image_path, "solve_profile": "fast"})
    mock_opus.step("inspect_plan", {"plan_id": plan_id, "focus": "heatmap"})
    mock_opus.step("export_print_plan", {"plan_id": plan_id})

    actual = tuple(name for name, _args, _r in mock_opus.transcript)
    assert actual == EXPECTED_FLOW, f"flow drifted: {actual}"
