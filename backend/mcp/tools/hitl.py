"""D9B — Tier 1 HITL refinement tools (8 tools)."""
from __future__ import annotations

import time
from typing import Any, Literal

from backend.mcp.errors import ToolResult, WoodblockError

_PIN_ACTIONS = ("force", "forbid", "merge")


def _impl_pending(code: str, hint: str) -> WoodblockError:
    return WoodblockError(
        tier="degraded", code=code,
        message=f"{code} — backing implementation lands at D11 (Tier 1 HITL real logic)",
        hint=hint, recoverable=True,
    )


def _new_plan_id(parent: str) -> str:
    return f"{parent}_hitl_{int(time.time() * 1000) % 100000}"


def pin_region(
    plan_id: str,
    region: dict[str, Any],
    action: Literal["force", "forbid", "merge"],
    pigment_id: str | None = None,
) -> ToolResult[dict[str, Any]]:
    if action not in _PIN_ACTIONS:
        return ToolResult(ok=False, data=None, errors=[
            WoodblockError(tier="refusal", code="INVALID_PIN_ACTION",
                           message=f"action must be one of {_PIN_ACTIONS}, got {action!r}",
                           recoverable=True),
        ])
    return ToolResult(
        ok=True,
        data={"plan_id": plan_id, "new_plan_id": _new_plan_id(plan_id),
              "action": action, "region": region, "pigment_id": pigment_id},
        errors=[_impl_pending("IMPL_PENDING_PIN", "real pin/re-solve loop lands at D11")],
    )


def alternative_stacks(plan_id: str, n: int = 3) -> ToolResult[dict[str, Any]]:
    if n < 1 or n > 10:
        return ToolResult(ok=False, data=None, errors=[
            WoodblockError(tier="refusal", code="INVALID_N",
                           message=f"n must be in [1, 10], got {n}", recoverable=True),
        ])
    alts = [{"plan_id": f"{plan_id}_alt_{i}", "rank": i, "overall_score": 0.5 - 0.02 * i}
            for i in range(n)]
    return ToolResult(ok=True, data={"plan_id": plan_id, "alternatives": alts},
                      errors=[_impl_pending("IMPL_PENDING_ALTS",
                                            "real alternative generation lands at D11")])


def generate_stack_candidates(plan_id: str, n: int = 3) -> ToolResult[dict[str, Any]]:
    """Alias for alternative_stacks (user-named per addendum-v3)."""
    return alternative_stacks(plan_id, n=n)


def compare_plans(plan_a: str, plan_b: str) -> ToolResult[dict[str, Any]]:
    return ToolResult(
        ok=True,
        data={"plan_a": plan_a, "plan_b": plan_b,
              "dE_delta_mean": 0.0, "dE_delta_p95": 0.0,
              "impression_diff": [], "composite_diff_path": None},
        errors=[_impl_pending("IMPL_PENDING_COMPARE", "real diff lands at D11")],
    )


def compare_alternate_recipes(plan_a: str, plan_b: str) -> ToolResult[dict[str, Any]]:
    """Alias for compare_plans."""
    return compare_plans(plan_a, plan_b)


def merge_impressions(plan_id: str, impression_ids: list[str]) -> ToolResult[dict[str, Any]]:
    if len(impression_ids) < 2:
        return ToolResult(ok=False, data=None, errors=[
            WoodblockError(tier="refusal", code="INSUFFICIENT_IMPRESSIONS",
                           message="merge_impressions requires >= 2 impression_ids",
                           recoverable=True),
        ])
    return ToolResult(
        ok=True,
        data={"plan_id": plan_id, "new_plan_id": _new_plan_id(plan_id),
              "merged_count": len(impression_ids)},
        errors=[_impl_pending("IMPL_PENDING_MERGE", "real merge lands at D11")],
    )


def merge_impressions_by_hue_family(
    plan_id: str, family_name: str,
) -> ToolResult[dict[str, Any]]:
    return ToolResult(
        ok=True,
        data={"plan_id": plan_id, "new_plan_id": _new_plan_id(plan_id),
              "merged_family": family_name, "merged_count": 0},
        errors=[_impl_pending("IMPL_PENDING_MERGE_FAMILY",
                              "real family-aware merge lands at D11")],
    )


def split_impression(
    plan_id: str, impression_id: str,
    by: Literal["mask_island", "hue_subcluster"] = "mask_island",
) -> ToolResult[dict[str, Any]]:
    if by not in ("mask_island", "hue_subcluster"):
        return ToolResult(ok=False, data=None, errors=[
            WoodblockError(tier="refusal", code="INVALID_SPLIT_MODE",
                           message=f"by must be mask_island or hue_subcluster, got {by!r}",
                           recoverable=True),
        ])
    return ToolResult(
        ok=True,
        data={"plan_id": plan_id, "new_plan_id": _new_plan_id(plan_id),
              "split_impression": impression_id, "by": by, "child_count": 2},
        errors=[_impl_pending("IMPL_PENDING_SPLIT", "real split lands at D11")],
    )


def adjust_pull_groups(plan_id: str, hints: dict[str, Any]) -> ToolResult[dict[str, Any]]:
    return ToolResult(
        ok=True,
        data={"plan_id": plan_id, "new_plan_id": _new_plan_id(plan_id), "hints": hints},
        errors=[_impl_pending("IMPL_PENDING_ADJUST_PULLS",
                              "real DSATUR re-pack with constraints lands at D11")],
    )


def simplify_masks_for_carving(plan_id: str) -> ToolResult[dict[str, Any]]:
    """Post-solve topology repair (addendum-v3 fix 1 home)."""
    return ToolResult(
        ok=True,
        data={"plan_id": plan_id, "new_plan_id": _new_plan_id(plan_id),
              "tiny_islands_removed": 0, "morph_open_radius": 1, "morph_close_radius": 1},
        errors=[_impl_pending("IMPL_PENDING_SIMPLIFY",
                              "real topology_score + morph_open/close repair lands at D11")],
    )


__all__ = [
    "pin_region", "alternative_stacks", "generate_stack_candidates",
    "compare_plans", "compare_alternate_recipes",
    "merge_impressions", "merge_impressions_by_hue_family",
    "split_impression", "adjust_pull_groups", "simplify_masks_for_carving",
]
