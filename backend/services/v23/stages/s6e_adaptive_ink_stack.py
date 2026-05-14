"""S6.e adaptive ink-batch stack from target cells."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.cluster.vq import kmeans2

from backend.services.v23.core import color, forward_render_jax


@dataclass(frozen=True)
class AdaptiveInkStackResult:
    """A solved print-color stack, still expressed as jigsaw cell groups."""

    batches: list[dict[str, Any]]
    diagnostics: dict[str, Any]
    final_rgb: NDArray[np.float32]


def _hex(rgb: NDArray[np.float32]) -> str:
    vals = np.clip(np.round(rgb * 255.0), 0, 255).astype(int).tolist()
    return f"#{vals[0]:02x}{vals[1]:02x}{vals[2]:02x}"


def _cell_ids(cells: list[dict[str, Any]]) -> set[int]:
    return {int(c["cell_id"]) for c in cells}


def _mean(values: list[NDArray[np.float32]], weights: list[float]) -> NDArray[np.float32]:
    arr = np.stack(values, axis=0)
    w = np.asarray(weights, dtype=np.float32)
    return np.average(arr, axis=0, weights=w).astype(np.float32)


def _hue_between(hue: float, start: float, end: float) -> bool:
    hue, start, end = hue % 360.0, start % 360.0, end % 360.0
    return start <= hue <= end if start <= end else hue >= start or hue <= end


def _family(cell: dict[str, Any]) -> str:
    hue = float(cell.get("hue_deg", 0.0))
    lum = float(cell.get("luminance_L", 0.0))
    chroma = float(cell.get("chroma_ab", 0.0))
    if lum < 38.0:
        return "key"
    if chroma < 9.0:
        return "neutral"
    if _hue_between(hue, 340.0, 24.0):
        return "red_pink"
    if _hue_between(hue, 24.0, 72.0):
        return "orange"
    if _hue_between(hue, 72.0, 112.0):
        return "yellow"
    if _hue_between(hue, 112.0, 178.0):
        return "green"
    if _hue_between(hue, 178.0, 258.0):
        return "blue"
    return "violet"


def _cluster_cells(
    cells: list[dict[str, Any]],
    *,
    k: int,
) -> list[list[dict[str, Any]]]:
    if not cells:
        return []
    if len(cells) <= k:
        return [[cell] for cell in cells]
    lab = np.asarray([cell["mean_lab"] for cell in cells], dtype=np.float32)
    features = lab.copy()
    features[:, 0] *= 0.72
    unique_features = np.unique(np.round(features, 3), axis=0)
    k = min(int(k), len(unique_features))
    if k <= 1:
        return [cells]
    try:
        _, labels = kmeans2(features, k, minit="++", seed=19)
    except Exception:
        labels = np.arange(len(cells), dtype=np.int32) % k
    groups: list[list[dict[str, Any]]] = [[] for _ in range(k)]
    for cell, label in zip(cells, labels.tolist(), strict=True):
        groups[int(label)].append(cell)
    return [group for group in groups if group]


def _group_stats(group: list[dict[str, Any]]) -> dict[str, Any]:
    weights = [float(c.get("area_px", 1)) for c in group]
    rgb = _mean(
        [np.asarray(c["mean_rgb"], dtype=np.float32) for c in group],
        weights,
    )
    lab = _mean(
        [np.asarray(c["mean_lab"], dtype=np.float32) for c in group],
        weights,
    )
    area = int(sum(int(c.get("area_px", 0)) for c in group))
    return {"cells": group, "rgb": rgb, "lab": lab, "area": area}


def _merge_close(groups: list[dict[str, Any]], max_groups: int) -> list[dict[str, Any]]:
    groups = sorted(groups, key=lambda g: -int(g["area"]))
    changed = True
    while changed or len(groups) > max_groups:
        changed = False
        best: tuple[float, int, int] | None = None
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                d = float(np.linalg.norm(groups[i]["lab"] - groups[j]["lab"]))
                if best is None or d < best[0]:
                    best = (d, i, j)
        if best is None:
            break
        if best[0] > 4.25 and len(groups) <= max_groups:
            break
        _, i, j = best
        merged = groups[i]["cells"] + groups[j]["cells"]
        groups = [g for idx, g in enumerate(groups) if idx not in {i, j}]
        groups.append(_group_stats(merged))
        groups.sort(key=lambda g: -int(g["area"]))
        changed = True
    return groups


def _mask(labels: NDArray[np.int32], ids: set[int]) -> NDArray[np.bool_]:
    return np.isin(labels, np.asarray(sorted(ids), dtype=np.int32))


def _alpha_for_lab(lab: NDArray[np.float32]) -> float:
    lum = float(lab[0])
    if lum < 34.0:
        return 0.94
    if lum < 48.0:
        return 0.86
    if lum > 78.0:
        return 0.56
    return 0.74


def _solve_plate_ink(
    current: NDArray[np.float32],
    target: NDArray[np.float32],
    mask: NDArray[np.bool_],
    alpha: float,
) -> NDArray[np.float32]:
    if not mask.any():
        return forward_render_jax.PAPER_RGB.astype(np.float32)
    need = (target[mask] - current[mask] * (1.0 - alpha)) / max(alpha, 1e-4)
    return np.clip(np.mean(need, axis=0), 0.0, 1.0).astype(np.float32)


def _apply(
    current: NDArray[np.float32],
    mask: NDArray[np.bool_],
    ink: NDArray[np.float32],
    alpha: float,
) -> NDArray[np.float32]:
    a = np.zeros(mask.shape + (1,), dtype=np.float32)
    a[mask] = float(np.clip(alpha, 0.0, 1.0))
    return current * (1.0 - a) + ink[None, None, :] * a


def _plate(
    idx: int,
    stage: str,
    role: str,
    group: dict[str, Any],
    labels: NDArray[np.int32],
    current: NDArray[np.float32],
    target: NDArray[np.float32],
    total_px: float,
    alpha: float,
) -> tuple[dict[str, Any], NDArray[np.float32]]:
    ids = _cell_ids(group["cells"])
    m = _mask(labels, ids)
    ink = _solve_plate_ink(current, target, m, alpha)
    after = _apply(current, m, ink, alpha)
    plate = {
        "plate_id": f"adaptive_plate_{idx:03d}",
        "stage": stage,
        "role": role,
        "cell_ids": sorted(ids),
        "cell_count": len(ids),
        "area_pct": round(float(m.mean() * 100.0), 3),
        "suggested_alpha": round(float(alpha), 4),
        "ink_rgb": [round(float(v), 5) for v in ink.tolist()],
        "ink_hex": _hex(ink),
        "target_mean_hex": _hex(group["rgb"]),
        "mean_lab": [round(float(v), 3) for v in group["lab"].tolist()],
        "geometry_note": "adaptive ink batch; same color may include separated jigsaw islands",
    }
    return plate, after.astype(np.float32)


def plan_adaptive_ink_stack(
    target_rgb: NDArray[np.float32],
    *,
    cell_graph: dict[str, Any],
    cell_labels: NDArray[np.int32],
    max_plates: int = 36,
) -> AdaptiveInkStackResult:
    """Build a flexible solved ink-batch stack from target cell colors."""
    cells = [
        c for c in cell_graph.get("cells", [])
        if float(c.get("paper_delta_e", 0.0)) >= 3.2
    ]
    max_plates = max(8, min(int(max_plates), 72))
    total_px = float(np.prod(cell_labels.shape))
    current = np.broadcast_to(
        forward_render_jax.PAPER_RGB.astype(np.float32),
        target_rgb.shape,
    ).copy()

    support_specs = [
        ("support_light_pink", lambda c: float(c["luminance_L"]) > 70 and _family(c) == "red_pink"),
        ("support_light_warm", lambda c: float(c["luminance_L"]) > 70 and _family(c) in {"orange", "yellow"}),
        ("support_light_cool", lambda c: float(c["luminance_L"]) > 70 and _family(c) in {"blue", "violet"}),
        ("support_light_green", lambda c: float(c["luminance_L"]) > 70 and _family(c) == "green"),
    ]
    plates: list[dict[str, Any]] = []
    batches: list[dict[str, Any]] = []
    support_plates: list[dict[str, Any]] = []
    idx = 1
    for role, pred in support_specs:
        group_cells = [c for c in cells if pred(c)]
        if not group_cells:
            continue
        group = _group_stats(group_cells)
        area_pct = group["area"] / total_px * 100.0
        if area_pct < 0.35:
            continue
        plate, current = _plate(
            idx, "support", role, group, cell_labels, current, target_rgb, total_px, 0.28,
        )
        support_plates.append(plate)
        plates.append(plate)
        idx += 1
    if support_plates:
        batches.append({"batch_id": "adaptive_01_support", "name": "light support ink batches", "plates": support_plates})

    remaining_slots = max(4, max_plates - len(plates))
    seed_groups = [_group_stats(g) for g in _cluster_cells(cells, k=remaining_slots)]
    color_groups = _merge_close(seed_groups, remaining_slots)
    color_groups.sort(key=lambda g: (float(g["lab"][0]) < 42.0, -float(g["lab"][0]), -float(g["area"])))

    color_plates: list[dict[str, Any]] = []
    for group in color_groups:
        lab = group["lab"]
        role = f"ink_{_family(group['cells'][0])}_{'dark' if lab[0] < 42 else 'light' if lab[0] > 74 else 'mid'}"
        plate, current = _plate(
            idx,
            "adaptive_color",
            role,
            group,
            cell_labels,
            current,
            target_rgb,
            total_px,
            _alpha_for_lab(lab),
        )
        color_plates.append(plate)
        plates.append(plate)
        idx += 1
    batches.append({"batch_id": "adaptive_02_color_solution", "name": "merged adaptive ink batches", "plates": color_plates})

    d_e = color.delta_e_summary(current, target_rgb)
    diagnostics = {
        "template": "adaptive_ink_batches",
        "plate_count": len(plates),
        "support_plate_count": len(support_plates),
        "color_plate_count": len(color_plates),
        "cell_count": len(cells),
        "max_plates": max_plates,
        "mean_delta_e76": round(float(d_e["dE_mean"]), 3),
        "p95_delta_e76": round(float(d_e["dE_p95"]), 3),
        "note": "solved adaptive ink colors from target/current composite; not limited to fixed pigment names",
    }
    return AdaptiveInkStackResult(batches=batches, diagnostics=diagnostics, final_rgb=current.astype(np.float32))


__all__ = ["AdaptiveInkStackResult", "plan_adaptive_ink_stack"]
