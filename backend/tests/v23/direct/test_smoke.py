"""Ring 1 smoke — the 11 day-1 MCP tools are importable.

Authority: ``/tmp/research-v23-mcp-build-sequence.md`` D9.1–D9.11.

Tool surface (day-1, 11 names):
  ingest_reference_image, analyze_image, build_hue_family_map,
  propose_stack, inspect_plan, alternative_stacks, compare_plans,
  simplify_masks_for_carving, score_candidate_stack,
  generate_print_recipe_report, export_print_plan.

This test stays xfail until D9 wires the decorators, then flips green.
"""
from __future__ import annotations

import importlib

import pytest

DAY1_TOOLS: tuple[str, ...] = (
    "ingest_reference_image",
    "analyze_image",
    "build_hue_family_map",
    "propose_stack",
    "inspect_plan",
    "alternative_stacks",
    "compare_plans",
    "simplify_masks_for_carving",
    "score_candidate_stack",
    "generate_print_recipe_report",
    "export_print_plan",
)

# Modules the day-1 tools live in per D9 (`tools/{core,hitl,carve}.py`).
TOOL_MODULES: tuple[str, ...] = (
    "backend.mcp.tools.core",
    "backend.mcp.tools.hitl",
    "backend.mcp.tools.carve",
)


@pytest.mark.xfail(reason="awaits D9.1-D9.11 — tools/{core,hitl,carve}.py")
def test_day1_tools_importable() -> None:
    found: set[str] = set()
    for mod_name in TOOL_MODULES:
        mod = importlib.import_module(mod_name)
        for name in DAY1_TOOLS:
            if hasattr(mod, name):
                found.add(name)
    missing = set(DAY1_TOOLS) - found
    assert not missing, f"day-1 tools not importable: {sorted(missing)}"
