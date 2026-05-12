"""D9B — Tier 5 carve handoff tools (3 tools)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.mcp.errors import ToolResult, WoodblockError


def _impl_pending(code: str, hint: str) -> WoodblockError:
    return WoodblockError(
        tier="degraded", code=code,
        message=f"{code} — real CNC export lands at D13 (Tier 5 carve real)",
        hint=hint, recoverable=True,
    )


def export_svg(plan_id: str, impression_ids: list[str] | None = None) -> ToolResult[dict[str, Any]]:
    """Per-impression carveable SVG (min-island + kerf + registration dilation)."""
    return ToolResult(
        ok=True,
        data={"plan_id": plan_id, "impression_ids": impression_ids or [],
              "svg_paths": []},
        errors=[_impl_pending("IMPL_PENDING_SVG",
                              "real potrace + min-island + kerf compensation lands at D13")],
    )


def export_block_svgs(plan_id: str) -> ToolResult[dict[str, Any]]:
    """Per-physical-block aggregated SVG with kento marks + reg halos."""
    return ToolResult(
        ok=True,
        data={"plan_id": plan_id, "block_svg_paths": {}, "kento_marks": []},
        errors=[_impl_pending("IMPL_PENDING_BLOCK_SVG",
                              "real per-Block aggregate + kento + reg halos lands at D13")],
    )


def generate_carve_order(plan_id: str) -> ToolResult[dict[str, Any]]:
    """CarveOrderManifest: ordered carving steps + ShopBot validation + hour estimate."""
    return ToolResult(
        ok=True,
        data={"plan_id": plan_id, "steps": [], "estimated_hours": 0.0,
              "shopbot_nodes_total": 0, "tool_recommendations": []},
        errors=[_impl_pending("IMPL_PENDING_CARVE_ORDER",
                              "real scheduling + tool-path validation lands at D13/D14")],
    )


__all__ = ["export_svg", "export_block_svgs", "generate_carve_order"]
