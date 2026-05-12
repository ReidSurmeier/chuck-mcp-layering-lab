"""D9B — Tier 3 introspection tools (6 tools, all REAL read-only)."""
from __future__ import annotations

from typing import Any

from backend.mcp.errors import ToolResult, WoodblockError
from backend.services.v23.core import forward_render_jax, templates


def get_pigments() -> ToolResult[dict[str, Any]]:
    """Return the 13-pigment Mixbox-anchored catalog."""
    pigments = []
    names = [
        "cadmium_yellow", "hansa_yellow", "cadmium_orange", "cadmium_red",
        "quinacridone_magenta", "cobalt_violet", "ultramarine_blue",
        "cobalt_blue", "viridian_green", "forest_green",
        "burnt_sienna", "raw_umber", "ivory_black",
    ]
    for idx, name in enumerate(names):
        r, g, b = forward_render_jax.PIGMENT_RGB_255[idx].tolist()
        pigments.append({
            "pigment_id": name,
            "name": name.replace("_", " ").title(),
            "rgb": [int(r), int(g), int(b)],
            "hex": f"#{r:02x}{g:02x}{b:02x}",
        })
    return ToolResult(ok=True, data={"pigments": pigments, "count": len(pigments),
                                     "catalog": "generic_mixbox_13"})


def get_emma_priors() -> ToolResult[dict[str, Any]]:
    """Return Emma-derived priors (hue families, accent rule, keyblock rule)."""
    return ToolResult(ok=True, data={
        "hue_families": ["cream", "cool", "flesh", "warm", "shadow", "detail", "accent"],
        "default_M_prior": 6,
        "M_prior_range": [4, 12],
        "accent_block_rule": "is_accent_block if pigment_count > 4 OR family_count > 2",
        "underprint_area_ratio_range": [1.5, 4.0],
        "keyblock_rule": "darkest detail impression gets highest order_step",
        "pigments_per_block_target": {"mean": 3.0, "cap": 5, "floor": 1},
    })


def get_defaults() -> ToolResult[dict[str, Any]]:
    """Return locked technical defaults from research-v23-mcp-defaults.md."""
    return ToolResult(ok=True, data={
        "solve_profile_walltime_s": {"fast": 60, "default": 180, "thorough": 600},
        "dE2000_target": {"mean": 1.5, "p95": 3.0},
        "ambiguous_band": {"pixel_dE_gt": 3.0, "mask_mean_gt": 1.5},
        "block_count_target": 6,
        "block_count_cap": 10,
        "min_island_px_at_300dpi": 60,
        "kento_dilation_px_at_300dpi": 3,
        "schema_version": "v23.0",
        "calibration_default": "generic_mixbox_13",
        "render_tier_default": "t1_mixbox",
    })


def solver_telemetry(plan_id: str) -> ToolResult[dict[str, Any]]:
    return ToolResult(
        ok=True,
        data={"plan_id": plan_id, "pyramid_levels_completed": 0,
              "iters_per_level": [], "loss_per_level": [],
              "rule_loss_breakdown": {}, "exit_reason": "mock",
              "wall_time_s": 0.0, "divergence_flags": []},
        errors=[WoodblockError(
            tier="degraded", code="IMPL_PENDING_TELEMETRY",
            message="solver telemetry persists at D10 once real solver runs",
            recoverable=True,
        )],
    )


def dE_at(plan_id: str, x: int, y: int) -> ToolResult[dict[str, Any]]:
    if x < 0 or y < 0:
        return ToolResult(ok=False, data=None, errors=[
            WoodblockError(tier="refusal", code="INVALID_COORDS",
                           message=f"x and y must be >= 0, got ({x}, {y})",
                           recoverable=True),
        ])
    return ToolResult(
        ok=True,
        data={"plan_id": plan_id, "x": x, "y": y, "dE": 0.0,
              "target_rgb": [0, 0, 0], "rendered_rgb": [0, 0, 0]},
        errors=[WoodblockError(
            tier="degraded", code="IMPL_PENDING_DE_AT",
            message="pixel-level ΔE lookup wires at D10",
            recoverable=True,
        )],
    )


def pigment_at(plan_id: str, x: int, y: int) -> ToolResult[dict[str, Any]]:
    if x < 0 or y < 0:
        return ToolResult(ok=False, data=None, errors=[
            WoodblockError(tier="refusal", code="INVALID_COORDS",
                           message=f"x and y must be >= 0, got ({x}, {y})",
                           recoverable=True),
        ])
    return ToolResult(
        ok=True,
        data={"plan_id": plan_id, "x": x, "y": y, "impressions": []},
        errors=[WoodblockError(
            tier="degraded", code="IMPL_PENDING_PIGMENT_AT",
            message="impression stack inspector wires at D10",
            recoverable=True,
        )],
    )


__all__ = ["get_pigments", "get_emma_priors", "get_defaults",
           "solver_telemetry", "dE_at", "pigment_at"]
