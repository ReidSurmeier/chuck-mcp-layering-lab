"""Production planning tools for batch/block expansion."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from backend.mcp.errors import ToolResult, WoodblockError
from backend.services.v23 import orchestrator as _orch
from backend.services.v23.stages import s6d_production_batches, s6e_adaptive_ink_stack


def plan_production_batches(
    plan_id: str,
    *,
    detail_slots: int = 16,
) -> ToolResult[dict[str, Any]]:
    """Propose 4 + 4 + detail production batches from the cell graph."""
    try:
        plan = _orch.load_plan(plan_id)
    except _orch.OrchestratorError as exc:
        return ToolResult(ok=False, data=None, errors=[exc.error])
    if not plan.alpha_stack_path or not Path(plan.alpha_stack_path).is_file():
        return ToolResult(ok=False, data=None, errors=[
            WoodblockError(
                tier="refusal",
                code="NO_SOLVER_OUTPUT",
                message="plan has no alpha_stack to expand into batches",
                recoverable=True,
            )
        ])
    if not plan.cell_graph_path or not Path(plan.cell_graph_path).is_file():
        return ToolResult(ok=False, data=None, errors=[
            WoodblockError(
                tier="refusal",
                code="NO_CELL_GRAPH",
                message="plan has no persisted cell graph",
                recoverable=True,
            )
        ])
    if not plan.cell_labels_path or not Path(plan.cell_labels_path).is_file():
        return ToolResult(ok=False, data=None, errors=[
            WoodblockError(
                tier="refusal",
                code="NO_CELL_LABELS",
                message="plan has no persisted cell label map",
                recoverable=True,
            )
        ])
    if not 1 <= int(detail_slots) <= 64:
        return ToolResult(ok=False, data=None, errors=[
            WoodblockError(
                tier="refusal",
                code="INVALID_DETAIL_SLOTS",
                message="detail_slots must be between 1 and 64",
                recoverable=True,
            )
        ])

    alpha_stack = np.load(plan.alpha_stack_path)
    pigment_idx = np.asarray(plan.pigment_idx, dtype=np.int32)
    graph = json.loads(Path(plan.cell_graph_path).read_text())
    labels = np.load(plan.cell_labels_path)
    result = s6d_production_batches.plan_production_batches(
        alpha_stack,
        pigment_idx,
        cell_graph=graph,
        cell_labels=labels,
        detail_slots=int(detail_slots),
    )
    plan_dir = Path(plan.alpha_stack_path).parent
    out_path = plan_dir / "production_batch_plan.json"
    payload = {
        "plan_id": plan_id,
        "cell_labels_path": plan.cell_labels_path,
        "diagnostics": result.diagnostics,
        "batches": result.batches,
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return ToolResult(ok=True, data={
        "plan_id": plan_id,
        "batch_plan_path": str(out_path),
        "cell_labels_path": plan.cell_labels_path,
        "diagnostics": result.diagnostics,
        "batches": result.batches,
    })


def plan_adaptive_ink_stack(
    plan_id: str,
    *,
    max_plates: int = 36,
) -> ToolResult[dict[str, Any]]:
    """Propose a solved adaptive ink-batch stack from target cells."""
    try:
        plan = _orch.load_plan(plan_id)
    except _orch.OrchestratorError as exc:
        return ToolResult(ok=False, data=None, errors=[exc.error])
    plan_dir = Path(plan.alpha_stack_path).parent if plan.alpha_stack_path else None
    if plan_dir is None or not (plan_dir / "target.npy").is_file():
        return ToolResult(ok=False, data=None, errors=[
            WoodblockError(
                tier="refusal",
                code="NO_TARGET_CACHE",
                message="plan has no target.npy for adaptive ink planning",
                recoverable=True,
            )
        ])
    if not plan.cell_graph_path or not Path(plan.cell_graph_path).is_file():
        return ToolResult(ok=False, data=None, errors=[
            WoodblockError(tier="refusal", code="NO_CELL_GRAPH",
                           message="plan has no persisted cell graph", recoverable=True)
        ])
    if not plan.cell_labels_path or not Path(plan.cell_labels_path).is_file():
        return ToolResult(ok=False, data=None, errors=[
            WoodblockError(tier="refusal", code="NO_CELL_LABELS",
                           message="plan has no persisted cell label map", recoverable=True)
        ])
    target = np.load(plan_dir / "target.npy")
    graph = json.loads(Path(plan.cell_graph_path).read_text())
    labels = np.load(plan.cell_labels_path)
    result = s6e_adaptive_ink_stack.plan_adaptive_ink_stack(
        target.astype(np.float32),
        cell_graph=graph,
        cell_labels=labels.astype(np.int32),
        max_plates=int(max_plates),
    )
    out_path = plan_dir / "adaptive_ink_stack_plan.json"
    payload = {
        "plan_id": plan_id,
        "cell_labels_path": plan.cell_labels_path,
        "diagnostics": result.diagnostics,
        "batches": result.batches,
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return ToolResult(ok=True, data={
        "plan_id": plan_id,
        "adaptive_plan_path": str(out_path),
        "cell_labels_path": plan.cell_labels_path,
        "diagnostics": result.diagnostics,
        "batches": result.batches,
    })


__all__ = ["plan_production_batches", "plan_adaptive_ink_stack"]
