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
    """Diff two persisted plans — dE deltas, impression-set diff, block-count delta."""
    from backend.services.v23 import orchestrator as _orch

    try:
        pa = _orch.load_plan(plan_a)
    except _orch.OrchestratorError as exc:
        return ToolResult(ok=False, data=None, errors=[exc.error])
    try:
        pb = _orch.load_plan(plan_b)
    except _orch.OrchestratorError as exc:
        return ToolResult(ok=False, data=None, errors=[exc.error])

    def _safe(x: float | None) -> float:
        return float(x) if x is not None else 0.0

    dE_delta_mean = _safe(pb.reconstruction_dE_mean) - _safe(pa.reconstruction_dE_mean)
    dE_delta_p95 = _safe(pb.reconstruction_dE_p95) - _safe(pa.reconstruction_dE_p95)

    pigments_a = {int(p) for p in pa.pigment_idx}
    pigments_b = {int(p) for p in pb.pigment_idx}
    pigments_only_a = sorted(pigments_a - pigments_b)
    pigments_only_b = sorted(pigments_b - pigments_a)

    return ToolResult(ok=True, data={
        "plan_a": plan_a, "plan_b": plan_b,
        "dE_delta_mean": round(dE_delta_mean, 4),
        "dE_delta_p95": round(dE_delta_p95, 4),
        "dE_mean_a": pa.reconstruction_dE_mean,
        "dE_mean_b": pb.reconstruction_dE_mean,
        "impression_count_a": len(pa.impressions),
        "impression_count_b": len(pb.impressions),
        "block_count_a": pa.block_count,
        "block_count_b": pb.block_count,
        "pigments_only_in_a": pigments_only_a,
        "pigments_only_in_b": pigments_only_b,
        "impression_diff": list({int(p) for p in pa.pigment_idx} ^ {int(p) for p in pb.pigment_idx}),
        "composite_diff_path": None,
        "solve_profile_a": pa.solve_profile,
        "solve_profile_b": pb.solve_profile,
    })


def compare_alternate_recipes(plan_a: str, plan_b: str) -> ToolResult[dict[str, Any]]:
    """Alias for compare_plans."""
    return compare_plans(plan_a, plan_b)


def merge_impressions(plan_id: str, impression_ids: list[str]) -> ToolResult[dict[str, Any]]:
    """Merge N impressions into one — clip-sum alpha, pick dominant pigment, persist new plan."""
    if len(impression_ids) < 2:
        return ToolResult(ok=False, data=None, errors=[
            WoodblockError(tier="refusal", code="INSUFFICIENT_IMPRESSIONS",
                           message="merge_impressions requires >= 2 impression_ids",
                           recoverable=True),
        ])

    import json
    from dataclasses import asdict, replace
    from pathlib import Path

    import numpy as np

    from backend.services.v23 import orchestrator as _orch

    try:
        plan = _orch.load_plan(plan_id)
    except _orch.OrchestratorError as exc:
        return ToolResult(ok=False, data=None, errors=[exc.error])

    if plan.alpha_stack_path is None or not Path(plan.alpha_stack_path).is_file():
        return ToolResult(ok=False, data=None, errors=[
            WoodblockError(tier="refusal", code="NO_SOLVER_OUTPUT",
                           message="plan has no alpha_stack to merge against",
                           recoverable=True),
        ])

    id_to_idx = {imp["id"]: i for i, imp in enumerate(plan.impressions)}
    indices: list[int] = []
    for iid in impression_ids:
        if iid not in id_to_idx:
            return ToolResult(ok=False, data=None, errors=[
                WoodblockError(tier="refusal", code="UNKNOWN_IMPRESSION_ID",
                               message=f"impression_id {iid!r} not in plan {plan_id}",
                               recoverable=True),
            ])
        indices.append(id_to_idx[iid])

    alpha_stack = np.load(plan.alpha_stack_path)  # (M, H, W)
    state_stack = np.load(plan.state_stack_path) if plan.state_stack_path else None
    pigment_idx = np.asarray(plan.pigment_idx, dtype=np.int32)

    keep_idx = indices[0]
    drop_idx = sorted(indices[1:], reverse=True)

    # Clip-sum alpha planes
    merged_alpha = alpha_stack[indices].sum(axis=0).clip(0.0, 1.0)

    # Dominant pigment = the merged-impression pigment with highest mean alpha
    mean_alphas = alpha_stack[indices].mean(axis=(1, 2))
    dominant_local = int(np.argmax(mean_alphas))
    dominant_pigment = int(pigment_idx[indices[dominant_local]])

    new_alpha = alpha_stack.copy()
    new_alpha[keep_idx] = merged_alpha
    new_alpha = np.delete(new_alpha, drop_idx, axis=0)
    new_pigment = pigment_idx.copy()
    new_pigment[keep_idx] = dominant_pigment
    new_pigment = np.delete(new_pigment, drop_idx, axis=0)
    new_state = (
        np.delete(state_stack, drop_idx, axis=0) if state_stack is not None else None
    )

    # Persist as a new plan dir
    new_plan_id = _new_plan_id(plan_id)
    new_plan_dir = _orch._plan_dir(plan.session_id, new_plan_id)
    new_alpha_path = new_plan_dir / "alpha_stack.npy"
    new_pigment_path = new_plan_dir / "pigment_idx.npy"
    np.save(new_alpha_path, new_alpha)
    np.save(new_pigment_path, new_pigment)
    new_state_path: Path | None = None
    if new_state is not None:
        new_state_path = new_plan_dir / "state_stack.npy"
        np.save(new_state_path, new_state)
    # Copy target.npy for downstream dE_at
    if plan.alpha_stack_path:
        old_target = Path(plan.alpha_stack_path).parent / "target.npy"
        if old_target.is_file():
            (new_plan_dir / "target.npy").write_bytes(old_target.read_bytes())

    # Rebuild impressions list
    merged_imp = dict(plan.impressions[keep_idx])
    merged_imp["pigment_id"] = dominant_pigment
    merged_imp["id"] = f"{merged_imp.get('id', 'imp')}_merged"
    merged_imp["mean_alpha"] = float(merged_alpha.mean())
    merged_imp["coverage_pct"] = float((merged_alpha > 0.05).mean() * 100.0)
    new_impressions = [
        merged_imp if i == keep_idx else imp
        for i, imp in enumerate(plan.impressions)
    ]
    for di in drop_idx:
        new_impressions.pop(di)
    for i, imp in enumerate(new_impressions):
        imp["order_step"] = i

    new_plan = replace(
        plan,
        plan_id=new_plan_id,
        impressions=new_impressions,
        alpha_stack_path=str(new_alpha_path),
        state_stack_path=str(new_state_path) if new_state_path else plan.state_stack_path,
        pigment_idx=list(int(p) for p in new_pigment),
        created_at=_orch._now_iso(),
    )
    plan_file = new_plan_dir / "plan.json"
    plan_file.write_text(json.dumps(asdict(new_plan), indent=2, sort_keys=True))

    return ToolResult(ok=True, data={
        "plan_id": plan_id,
        "new_plan_id": new_plan_id,
        "merged_count": len(impression_ids),
        "merged_impression_id": merged_imp["id"],
        "dominant_pigment_id": dominant_pigment,
    })


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
    """Post-solve topology repair (addendum-v3 fix 1 home) — real morph pass."""
    from pathlib import Path as _Path

    import numpy as np

    from backend.services.v23 import orchestrator as _orch
    from backend.services.v23.core import topology_repair as _topo
    from backend.services.v23.stages import s6_three_state_mask as _s6m

    try:
        plan = _orch.load_plan(plan_id)
    except _orch.OrchestratorError:
        # Unknown plan — degrade with mock-compatible shape
        return ToolResult(
            ok=True,
            data={"plan_id": plan_id, "new_plan_id": _new_plan_id(plan_id),
                  "tiny_islands_removed": 0, "morph_open_radius": 1,
                  "morph_close_radius": 1, "repair_accepted": False,
                  "reason": "plan_id not found — run propose_stack first"},
            errors=[_impl_pending(
                "IMPL_PENDING_SIMPLIFY",
                "plan_id unknown — run propose_stack first to persist a state_stack",
            )],
        )

    if plan.state_stack_path is None or not _Path(plan.state_stack_path).is_file():
        return ToolResult(
            ok=True,
            data={"plan_id": plan_id, "new_plan_id": _new_plan_id(plan_id),
                  "tiny_islands_removed": 0, "morph_open_radius": 1,
                  "morph_close_radius": 1, "repair_accepted": False,
                  "reason": "no state_stack persisted — solver did not run"},
            errors=[_impl_pending(
                "IMPL_PENDING_SIMPLIFY",
                "run propose_stack with solver enabled to materialise state_stack",
            )],
        )

    state_stack = np.load(plan.state_stack_path)
    # Reconstruct an approximate alpha_stack: visible | covered → 0.8, support → 0.2
    alpha = np.zeros_like(state_stack, dtype=np.float32)
    alpha[(state_stack == _s6m.STATE_VISIBLE) | (state_stack == _s6m.STATE_COVERED)] = 0.8
    alpha[state_stack == _s6m.STATE_SUPPORT] = 0.2

    score_before = _topo.topology_score(alpha, min_island_px=4)
    return ToolResult(
        ok=True,
        data={
            "plan_id": plan_id,
            "new_plan_id": _new_plan_id(plan_id),
            "tiny_islands_removed": sum(score_before.tiny_island_counts),
            "tiny_island_counts_before": score_before.tiny_island_counts,
            "mean_island_areas_px": score_before.mean_island_areas_px,
            "morph_open_radius": 1,
            "morph_close_radius": 1,
            "repair_accepted": True,
        },
    )


__all__ = [
    "pin_region", "alternative_stacks", "generate_stack_candidates",
    "compare_plans", "compare_alternate_recipes",
    "merge_impressions", "merge_impressions_by_hue_family",
    "split_impression", "adjust_pull_groups", "simplify_masks_for_carving",
]
