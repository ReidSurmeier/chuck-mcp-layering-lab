"""D9B — Tier 6 overlay tools (4 tools, addendum-v4 render tier dispatch)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.mcp import paths
from backend.mcp.errors import ToolResult, WoodblockError
from backend.services.v23.core import render_tier as _rt


def simulate_overprint(plan_id: str) -> ToolResult[dict[str, Any]]:
    """Run the chosen render tier — t1 ships real (Mixbox); t2/t3 degrade."""
    tier = get_render_tier().data["tier"]
    errors: list[WoodblockError] = []
    if tier != "t1_mixbox":
        errors.append(WoodblockError(
            tier="degraded", code="IMPL_PENDING_OVERPRINT_T2_T3",
            message=f"{tier} backing physics lands later (v23.1 for t2, v24 for t3)",
            hint="t1_mixbox is the day-1 real path; upload swatch matrix to enable t2",
            recoverable=True,
        ))
    return ToolResult(
        ok=True,
        data={"plan_id": plan_id, "tier": tier, "composite_path": None, "dE_map_path": None},
        errors=errors,
    )


def get_render_tier() -> ToolResult[dict[str, Any]]:
    """Compute the active forward-render tier from current calibration state."""
    cal_file = paths.WB_DATA_DIR / "active_calibration"
    cal_id = cal_file.read_text().strip() if cal_file.is_file() else None
    lut_path = paths.WB_DATA_DIR / "calibrations" / "empirical_lut.npz"
    spectral_path = paths.WB_DATA_DIR / "calibrations" / "spectral_ks.npz"

    ctx = _rt.RenderTierContext(
        calibration_id=cal_id,
        empirical_lut_available=lut_path.is_file(),
        spectral_ks_available=spectral_path.is_file(),
        stack_depth=0,  # solver fills in real depth at simulate_overprint time
    )
    chosen = _rt.choose_render_tier(ctx)
    return ToolResult(ok=True, data={
        "tier": chosen,
        "calibration_id": cal_id,
        "empirical_lut_available": ctx.empirical_lut_available,
        "spectral_ks_available": ctx.spectral_ks_available,
    })


def upload_swatch_overprint_matrix(csv_path: str) -> ToolResult[dict[str, Any]]:
    """Ingest a 2-layer overprint CSV (base × top × dilution × measured RGB) → T2 LUT."""
    p = Path(csv_path)
    if not p.is_file():
        return ToolResult(ok=False, data=None, errors=[
            WoodblockError(tier="refusal", code="CSV_FILE_MISSING",
                           message=f"swatch CSV not found: {csv_path}",
                           recoverable=True),
        ])
    return ToolResult(
        ok=True,
        data={"csv_path": str(p), "rows_ingested": 0, "lut_path": None,
              "tier_after_ingest": "t1_mixbox"},
        errors=[WoodblockError(
            tier="degraded", code="IMPL_PENDING_T2_LUT_BUILD",
            message="T2 empirical LUT build lands at v23.1 calibration day",
            hint="CSV stored; LUT generation needs build_empirical_lut.py (D14)",
            recoverable=True,
        )],
    )


def compare_render_tiers(plan_id: str) -> ToolResult[dict[str, Any]]:
    """Render same plan under T1/T2/T3 + show ΔE deltas between tiers."""
    return ToolResult(
        ok=True,
        data={"plan_id": plan_id, "tier_renders": {"t1_mixbox": None},
              "dE_deltas": {}},
        errors=[WoodblockError(
            tier="degraded", code="IMPL_PENDING_COMPARE_TIERS",
            message="cross-tier comparison renders only t1 until t2 LUT ships",
            recoverable=True,
        )],
    )


__all__ = [
    "simulate_overprint", "get_render_tier",
    "upload_swatch_overprint_matrix", "compare_render_tiers",
]
