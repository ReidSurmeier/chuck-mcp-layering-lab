"""S6.d production batch planning from solved roles and cell graph."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from backend.services.v23.core import forward_render_jax


@dataclass(frozen=True)
class ProductionBatchPlanResult:
    """Read-only production expansion proposal."""

    batches: list[dict[str, Any]]
    diagnostics: dict[str, Any]


def _hue_between(hue: float, start: float, end: float) -> bool:
    hue = hue % 360.0
    start = start % 360.0
    end = end % 360.0
    if start <= end:
        return start <= hue <= end
    return hue >= start or hue <= end


def _rgb_to_hex(rgb: NDArray[np.float32]) -> str:
    vals = np.clip(np.round(rgb * 255.0), 0, 255).astype(int).tolist()
    return f"#{vals[0]:02x}{vals[1]:02x}{vals[2]:02x}"


def _family(cell: dict[str, Any]) -> str:
    hue = float(cell.get("hue_deg", 0.0))
    lum = float(cell.get("luminance_L", 0.0))
    chroma = float(cell.get("chroma_ab", 0.0))
    if lum < 38.0:
        return "key_shadow"
    if chroma < 10.0:
        return "neutral_grey"
    if _hue_between(hue, 340.0, 24.0):
        return "red_pink"
    if _hue_between(hue, 24.0, 72.0):
        return "orange"
    if _hue_between(hue, 72.0, 112.0):
        return "yellow"
    if _hue_between(hue, 112.0, 178.0):
        return "green"
    if _hue_between(hue, 178.0, 258.0):
        return "blue_teal"
    if _hue_between(hue, 258.0, 340.0):
        return "violet_blue"
    return "neutral_grey"


def _tone(cell: dict[str, Any]) -> str:
    lum = float(cell.get("luminance_L", 0.0))
    if lum >= 74.0:
        return "light"
    if lum >= 48.0:
        return "mid"
    return "dark"


def _cell_area_pct(cells: list[dict[str, Any]], ids: set[int], total_area: float) -> float:
    area = sum(int(c.get("area_px", 0)) for c in cells if int(c["cell_id"]) in ids)
    return round(float(area / max(total_area, 1.0) * 100.0), 3)


def _mean_rgb(cells: list[dict[str, Any]], ids: set[int]) -> NDArray[np.float32]:
    values: list[NDArray[np.float32]] = []
    weights: list[float] = []
    for cell in cells:
        if int(cell["cell_id"]) not in ids:
            continue
        values.append(np.asarray(cell.get("mean_rgb", [0.0, 0.0, 0.0]), dtype=np.float32))
        weights.append(float(cell.get("area_px", 1)))
    if not values:
        return forward_render_jax.PAPER_RGB.astype(np.float32)
    arr = np.stack(values, axis=0)
    w = np.asarray(weights, dtype=np.float32)
    return np.average(arr, axis=0, weights=w).astype(np.float32)


def _candidate_ids(role: str) -> list[int]:
    if "key_shadow" in role or role.endswith("_dark"):
        return [12, 20, 19, 15, 11, 34, 35]
    if "pink" in role or "red" in role:
        return [24, 32, 27, 17, 3, 16, 18, 25]
    if "orange" in role or "brown" in role or "yellow" in role:
        return [25, 26, 2, 13, 14, 10, 31, 17]
    if "green" in role:
        return [28, 31, 23, 8, 22, 9]
    if "blue" in role or "violet" in role:
        return [29, 34, 30, 21, 33, 7, 20, 19]
    if "grey" in role or "neutral" in role:
        return [35, 34, 11, 12]
    return list(range(len(forward_render_jax.PIGMENT_NAMES)))


def _nearest_pigments(rgb: NDArray[np.float32], role: str, limit: int = 3) -> list[dict[str, Any]]:
    pigments = forward_render_jax.PIGMENT_TABLE.astype(np.float32)
    candidates = np.asarray(_candidate_ids(role), dtype=np.int32)
    d2 = np.sum((pigments[candidates] - rgb[None, :]) ** 2, axis=1)
    local_order = np.argsort(d2)[:limit]
    out: list[dict[str, Any]] = []
    for local_idx in local_order.tolist():
        idx = int(candidates[int(local_idx)])
        out.append({
            "pigment_id": idx,
            "pigment_name": forward_render_jax.PIGMENT_NAMES[idx],
            "distance_rgb": round(float(np.sqrt(d2[int(local_idx)])), 4),
        })
    return out


def _cell_alpha_means(labels: NDArray[np.int32], alpha_stack: NDArray[np.float32]) -> NDArray[np.float32]:
    flat_labels = labels.ravel().astype(np.int32)
    count = np.bincount(flat_labels).astype(np.float32)
    count = np.maximum(count, 1.0)
    means = []
    for alpha in alpha_stack:
        sums = np.bincount(
            flat_labels,
            weights=alpha.ravel().astype(np.float32),
            minlength=count.shape[0],
        ).astype(np.float32)
        means.append(sums / count)
    return np.stack(means, axis=0)


def _source_impressions(
    cell_ids: set[int],
    cell_alpha: NDArray[np.float32],
    pigment_idx: NDArray[np.int32],
) -> list[dict[str, Any]]:
    if not cell_ids:
        return []
    ids = np.asarray(sorted(cell_ids), dtype=np.int32)
    ids = ids[ids < cell_alpha.shape[1]]
    if ids.size == 0:
        return []
    means = cell_alpha[:, ids].mean(axis=1)
    order = np.argsort(-means)
    out: list[dict[str, Any]] = []
    for slot in order[:4].tolist():
        if float(means[slot]) < 0.015:
            continue
        pid = int(pigment_idx[slot])
        out.append({
            "impression_id": f"imp_{slot + 1:03d}",
            "pigment_name": forward_render_jax.PIGMENT_NAMES[pid],
            "mean_alpha_on_cells": round(float(means[slot]), 4),
        })
    return out


def _suggested_alpha(batch_id: str, role: str) -> float:
    if batch_id == "batch_01_light_support":
        return 0.30
    if batch_id == "batch_02_color_build":
        return 0.42
    if "key_shadow" in role or role.endswith("_dark"):
        return 0.62
    if role.endswith("_light"):
        return 0.34
    return 0.48


def _plate(
    plate_id: str,
    batch_id: str,
    role: str,
    cells: list[dict[str, Any]],
    cell_ids: set[int],
    total_area: float,
    cell_alpha: NDArray[np.float32],
    pigment_idx: NDArray[np.int32],
) -> dict[str, Any]:
    rgb = _mean_rgb(cells, cell_ids)
    return {
        "plate_id": plate_id,
        "batch_id": batch_id,
        "role": role,
        "cell_count": len(cell_ids),
        "area_pct": _cell_area_pct(cells, cell_ids, total_area),
        "mean_hex": _rgb_to_hex(rgb),
        "target_mean_rgb": [round(float(v), 4) for v in rgb.tolist()],
        "suggested_alpha": _suggested_alpha(batch_id, role),
        "suggested_pigments": _nearest_pigments(rgb, role),
        "cell_ids": sorted(int(cid) for cid in cell_ids),
        "source_impressions": _source_impressions(cell_ids, cell_alpha, pigment_idx),
        "geometry_note": "jigsaw cell group; vectorization should preserve clear cell boundaries",
    }


def _select(cells: list[dict[str, Any]], predicate) -> set[int]:
    return {int(c["cell_id"]) for c in cells if predicate(c)}


def _detail_groups(cells: list[dict[str, Any]], detail_slots: int) -> list[tuple[str, set[int], float]]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for cell in cells:
        if float(cell.get("paper_delta_e", 0.0)) < 4.0:
            continue
        fam = _family(cell)
        tone = _tone(cell)
        if tone == "light" and fam in {"neutral_grey", "yellow"}:
            continue
        buckets.setdefault((fam, tone), []).append(cell)
    ranked: list[tuple[str, set[int], float]] = []
    for (fam, tone), group in buckets.items():
        area = float(sum(int(c.get("area_px", 0)) for c in group))
        ids = {int(c["cell_id"]) for c in group}
        ranked.append((f"detail_{fam}_{tone}", ids, area))
    ranked.sort(key=lambda item: item[2], reverse=True)
    return ranked[: max(0, int(detail_slots))]


def plan_production_batches(
    alpha_stack: NDArray[np.float32],
    pigment_idx: NDArray[np.int32],
    *,
    cell_graph: dict[str, Any],
    cell_labels: NDArray[np.int32],
    detail_slots: int = 16,
) -> ProductionBatchPlanResult:
    """Expand a compressed solver stack into production-style color batches."""
    cells = list(cell_graph.get("cells", []))
    if not cells:
        raise ValueError("cell_graph has no cells")
    total_area = float(sum(int(c.get("area_px", 0)) for c in cells))
    cell_alpha = _cell_alpha_means(cell_labels.astype(np.int32), alpha_stack.astype(np.float32))

    batch1_specs = [
        ("light_pink_support", lambda c: _tone(c) == "light" and _family(c) == "red_pink"),
        ("light_blue_support", lambda c: _tone(c) == "light" and _family(c) in {"blue_teal", "violet_blue"}),
        ("light_orange_support", lambda c: _tone(c) == "light" and _family(c) in {"orange", "yellow"}),
        ("light_green_support", lambda c: _tone(c) == "light" and _family(c) == "green"),
    ]
    batch2_specs = [
        ("orange_red_build", lambda c: _tone(c) == "mid" and _family(c) in {"orange", "red_pink"}),
        ("blue_teal_build", lambda c: _tone(c) == "mid" and _family(c) in {"blue_teal", "violet_blue"}),
        ("green_build", lambda c: _tone(c) == "mid" and _family(c) == "green"),
        ("brown_shadow_build", lambda c: _tone(c) != "light" and _family(c) in {"orange", "yellow", "neutral_grey"}),
    ]

    batches: list[dict[str, Any]] = []
    for batch_id, name, specs in (
        ("batch_01_light_support", "first four light support blocks", batch1_specs),
        ("batch_02_color_build", "second four color/depth blocks", batch2_specs),
    ):
        plates = [
            _plate(
                f"{batch_id}_plate_{i:02d}",
                batch_id,
                role,
                cells,
                _select(cells, predicate),
                total_area,
                cell_alpha,
                pigment_idx,
            )
            for i, (role, predicate) in enumerate(specs, start=1)
        ]
        batches.append({"batch_id": batch_id, "name": name, "plates": plates})

    detail_plates = [
        _plate(
            f"batch_03_regional_detail_plate_{i:02d}",
            "batch_03_regional_detail",
            role,
            cells,
            ids,
            total_area,
            cell_alpha,
            pigment_idx,
        )
        for i, (role, ids, _area) in enumerate(_detail_groups(cells, detail_slots), start=1)
    ]
    batches.append({
        "batch_id": "batch_03_regional_detail",
        "name": "regional hue shifts, contours, and key/detail blocks",
        "plates": detail_plates,
    })

    plate_count = sum(len(batch["plates"]) for batch in batches)
    diagnostics = {
        "template": "close_4_4_16",
        "batch_count": len(batches),
        "plate_count": plate_count,
        "target_first_batch_plates": 4,
        "target_second_batch_plates": 4,
        "target_detail_plates": int(detail_slots),
        "cell_count": len(cells),
        "note": "production expansion proposal; does not mutate alpha masks",
    }
    return ProductionBatchPlanResult(batches=batches, diagnostics=diagnostics)


__all__ = ["ProductionBatchPlanResult", "plan_production_batches"]
