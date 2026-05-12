"""D9B — Tier 2 calibration tools (5 tools)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backend.mcp import paths
from backend.mcp.errors import ToolResult, WoodblockError


def _calibrations_dir() -> Path:
    d = paths.WB_DATA_DIR / "calibrations"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _impl_pending(code: str, hint: str) -> WoodblockError:
    return WoodblockError(
        tier="degraded", code=code,
        message=f"{code} — full calibration fit lands at D12",
        hint=hint, recoverable=True,
    )


def capture_swatch(
    swatch_image_path: str,
    layout: dict[str, Any],
    colorchecker: dict[str, Any],
) -> ToolResult[dict[str, Any]]:
    if not Path(swatch_image_path).is_file():
        return ToolResult(ok=False, data=None, errors=[
            WoodblockError(tier="refusal", code="SWATCH_FILE_MISSING",
                           message=f"swatch image not found: {swatch_image_path}",
                           recoverable=True),
        ])
    candidate_id = f"cal_candidate_{Path(swatch_image_path).stem}"
    return ToolResult(
        ok=True,
        data={"candidate_id": candidate_id, "swatch_image": swatch_image_path,
              "layout": layout, "colorchecker": colorchecker,
              "detected_patches": 0},
        errors=[_impl_pending("IMPL_PENDING_SWATCH",
                              "ColorChecker + ArUco detection lands at D12")],
    )


def fit_pigments(candidate_id: str) -> ToolResult[dict[str, Any]]:
    """Solve per-pigment Mixbox z anchor + opacity curve from captured swatch."""
    calibration_id = candidate_id.replace("cal_candidate_", "cal_fitted_")
    return ToolResult(
        ok=True,
        data={"candidate_id": candidate_id, "calibration_id": calibration_id,
              "pigments_fitted": 0, "swatch_fit_dE_median": 0.0},
        errors=[_impl_pending("IMPL_PENDING_FIT",
                              "JAX L-BFGS Newton-inverse fit lands at D12")],
    )


def apply_calibration(calibration_id: str) -> ToolResult[dict[str, Any]]:
    state_file = paths.WB_DATA_DIR / "active_calibration"
    paths.WB_DATA_DIR.mkdir(parents=True, exist_ok=True)
    state_file.write_text(calibration_id)
    return ToolResult(
        ok=True,
        data={"calibration_id": calibration_id, "applied": True,
              "previous_calibration": "generic_mixbox_13"},
    )


def list_calibrations() -> ToolResult[dict[str, Any]]:
    d = _calibrations_dir()
    entries = []
    for p in sorted(d.glob("*.json")):
        try:
            entries.append(json.loads(p.read_text()))
        except json.JSONDecodeError:
            continue
    active_file = paths.WB_DATA_DIR / "active_calibration"
    active = active_file.read_text().strip() if active_file.is_file() else None
    return ToolResult(ok=True, data={"calibrations": entries, "active": active})


def inspect_calibration(calibration_id: str) -> ToolResult[dict[str, Any]]:
    f = _calibrations_dir() / f"{calibration_id}.json"
    if not f.is_file():
        return ToolResult(ok=False, data=None, errors=[
            WoodblockError(tier="refusal", code="CALIBRATION_NOT_FOUND",
                           message=f"calibration {calibration_id!r} not found",
                           hint="call list_calibrations() to see what's available",
                           recoverable=True),
        ])
    return ToolResult(ok=True, data=json.loads(f.read_text()))


__all__ = [
    "capture_swatch", "fit_pigments", "apply_calibration",
    "list_calibrations", "inspect_calibration",
]
