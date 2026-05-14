"""S3.b — target-image cell graph for region-first print planning.

The inverse solver works on alpha tensors, but Emma-style planning needs a
stable set of printable local regions first: cells, their adjacency, target
color statistics, and a paper/tint classification. This stage creates that
graph before warm-starting or solving so later stages can organize plates by
regions instead of retrofitting pixel masks after the fact.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from skimage import color, filters, segmentation

from backend.services.v23.core import forward_render_jax

CellGraphMode = Literal["emma_lattice", "slic"]

_MAX_SEGMENT_PIXELS = 850_000


@dataclass(frozen=True)
class CellGraphResult:
    """Region graph extracted from a target image."""

    labels: NDArray[np.int32]
    cells: list[dict[str, Any]]
    adjacency: list[dict[str, Any]]
    diagnostics: dict[str, Any]


def _rgb_float(rgb: NDArray[np.uint8] | NDArray[np.float32]) -> NDArray[np.float32]:
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"rgb must be HWC RGB, got {arr.shape!r}")
    if np.issubdtype(arr.dtype, np.integer):
        return (arr.astype(np.float32) / 255.0).clip(0.0, 1.0)
    return arr.astype(np.float32).clip(0.0, 1.0)


def _resize_label_map(labels: NDArray[np.int32], size: tuple[int, int]) -> NDArray[np.int32]:
    """Resize labels with nearest-neighbor. ``size`` is ``(width, height)``."""
    return np.asarray(
        Image.fromarray(labels.astype(np.int32), mode="I").resize(size, Image.Resampling.NEAREST),
        dtype=np.int32,
    )


def _paper_lab_from_border(lab: NDArray[np.float32]) -> NDArray[np.float32]:
    h, w = lab.shape[:2]
    margin = max(4, min(h, w) // 32)
    border = np.concatenate([
        lab[:margin].reshape(-1, 3),
        lab[-margin:].reshape(-1, 3),
        lab[:, :margin].reshape(-1, 3),
        lab[:, -margin:].reshape(-1, 3),
    ], axis=0)
    return np.median(border, axis=0).astype(np.float32)


def _default_n_segments(h: int, w: int, mode: CellGraphMode) -> int:
    area = h * w
    divisor = 420.0 if mode == "emma_lattice" else 580.0
    return max(96, min(2400, int(round(area / divisor))))


def _segment_cells(
    rgb: NDArray[np.float32],
    *,
    mode: CellGraphMode,
    n_cells: int | None,
    max_pixels: int,
) -> NDArray[np.int32]:
    h, w = rgb.shape[:2]
    seg_rgb = rgb
    scale = 1.0
    if h * w > max_pixels:
        scale = (max_pixels / float(h * w)) ** 0.5
        new_w = max(96, int(round(w * scale)))
        new_h = max(96, int(round(h * scale)))
        seg_rgb = np.asarray(
            Image.fromarray((rgb * 255.0).astype(np.uint8), "RGB").resize(
                (new_w, new_h),
                Image.Resampling.LANCZOS,
            ),
            dtype=np.float32,
        ) / 255.0

    seg_h, seg_w = seg_rgb.shape[:2]
    segments = int(n_cells or _default_n_segments(seg_h, seg_w, mode))
    compactness = 19.0 if mode == "emma_lattice" else 13.0
    sigma = 0.65 if mode == "emma_lattice" else 1.0
    labels = segmentation.slic(
        seg_rgb,
        n_segments=segments,
        compactness=compactness,
        sigma=sigma,
        start_label=0,
        channel_axis=-1,
        convert2lab=True,
        enforce_connectivity=True,
        min_size_factor=0.22,
        max_size_factor=3.5,
    ).astype(np.int32)
    if scale != 1.0:
        labels = _resize_label_map(labels, (w, h))
    return labels.astype(np.int32)


def _adjacency(
    labels: NDArray[np.int32],
    edge_strength: NDArray[np.float32],
) -> list[dict[str, Any]]:
    pairs: dict[tuple[int, int], list[float]] = {}

    right = labels[:, 1:] != labels[:, :-1]
    if right.any():
        a = labels[:, :-1][right]
        b = labels[:, 1:][right]
        e = edge_strength[:, :-1][right]
        for ai, bi, ei in zip(a.tolist(), b.tolist(), e.tolist(), strict=True):
            key = (ai, bi) if ai < bi else (bi, ai)
            slot = pairs.setdefault(key, [0.0, 0.0])
            slot[0] += 1.0
            slot[1] += float(ei)

    down = labels[1:, :] != labels[:-1, :]
    if down.any():
        a = labels[:-1, :][down]
        b = labels[1:, :][down]
        e = edge_strength[:-1, :][down]
        for ai, bi, ei in zip(a.tolist(), b.tolist(), e.tolist(), strict=True):
            key = (ai, bi) if ai < bi else (bi, ai)
            slot = pairs.setdefault(key, [0.0, 0.0])
            slot[0] += 1.0
            slot[1] += float(ei)

    out = []
    for (a, b), (count, edge_sum) in sorted(pairs.items()):
        out.append({
            "a": int(a),
            "b": int(b),
            "boundary_px": int(round(count)),
            "mean_edge_strength": round(float(edge_sum / max(count, 1.0)), 5),
        })
    return out


def _role_hint(luminance: float, chroma: float, paper_delta_e: float) -> str:
    if paper_delta_e < 3.2 and luminance > 84.0:
        return "paper_or_unprinted_margin"
    if luminance > 78.0 and chroma < 18.0:
        return "subtle_tint"
    if luminance < 42.0:
        return "key_or_shadow"
    if chroma > 34.0:
        return "local_chroma"
    return "regional_mass"


def _summarise_cells(
    rgb: NDArray[np.float32],
    labels: NDArray[np.int32],
    lab: NDArray[np.float32],
    paper_lab: NDArray[np.float32],
) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for label_id in np.unique(labels).tolist():
        cell = labels == int(label_id)
        ys, xs = np.where(cell)
        if ys.size == 0:
            continue
        mean_rgb = rgb[cell].mean(axis=0)
        mean_lab = lab[cell].mean(axis=0)
        chroma = float(np.sqrt(mean_lab[1] ** 2 + mean_lab[2] ** 2))
        hue = float(np.degrees(np.arctan2(mean_lab[2], mean_lab[1])) % 360.0)
        paper_delta = float(np.linalg.norm(mean_lab - paper_lab))
        cells.append({
            "cell_id": int(label_id),
            "area_px": int(ys.size),
            "bbox": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
            "centroid": [round(float(xs.mean()), 2), round(float(ys.mean()), 2)],
            "mean_rgb": [round(float(v), 5) for v in mean_rgb.tolist()],
            "mean_hex": _rgb_to_hex(mean_rgb),
            "mean_lab": [round(float(v), 4) for v in mean_lab.tolist()],
            "luminance_L": round(float(mean_lab[0]), 4),
            "chroma_ab": round(chroma, 4),
            "hue_deg": round(hue, 2),
            "paper_delta_e": round(paper_delta, 4),
            "role_hint": _role_hint(float(mean_lab[0]), chroma, paper_delta),
        })
    return cells


def _rgb_to_hex(rgb: NDArray[np.float32]) -> str:
    vals = np.clip(np.round(rgb * 255.0), 0, 255).astype(int).tolist()
    return f"#{vals[0]:02x}{vals[1]:02x}{vals[2]:02x}"


def build_cell_graph(
    rgb: NDArray[np.uint8] | NDArray[np.float32],
    *,
    mode: CellGraphMode = "emma_lattice",
    n_cells: int | None = None,
    max_pixels: int = _MAX_SEGMENT_PIXELS,
) -> CellGraphResult:
    """Build a printable-region graph from the target image."""
    if mode not in ("emma_lattice", "slic"):
        raise ValueError(f"mode must be 'emma_lattice' or 'slic', got {mode!r}")
    target = _rgb_float(rgb)
    h, w = target.shape[:2]
    labels = _segment_cells(target, mode=mode, n_cells=n_cells, max_pixels=max_pixels)
    lab = color.rgb2lab(target).astype(np.float32)
    paper_lab = _paper_lab_from_border(lab)
    luminance = lab[..., 0] / 100.0
    edge_strength = filters.sobel(luminance).astype(np.float32)
    cells = _summarise_cells(target, labels, lab, paper_lab)
    adjacency = _adjacency(labels, edge_strength)

    role_counts: dict[str, int] = {}
    for cell in cells:
        role = str(cell["role_hint"])
        role_counts[role] = role_counts.get(role, 0) + 1
    area_values = [int(c["area_px"]) for c in cells]
    diagnostics = {
        "mode": mode,
        "cell_count": len(cells),
        "adjacency_count": len(adjacency),
        "shape": [int(h), int(w)],
        "paper_rgb": [round(float(v), 5) for v in forward_render_jax.PAPER_RGB.tolist()],
        "estimated_paper_lab": [round(float(v), 4) for v in paper_lab.tolist()],
        "role_counts": role_counts,
        "area_px_mean": round(float(np.mean(area_values)) if area_values else 0.0, 3),
        "area_px_p95": round(float(np.percentile(area_values, 95)) if area_values else 0.0, 3),
    }
    return CellGraphResult(
        labels=labels.astype(np.int32),
        cells=cells,
        adjacency=adjacency,
        diagnostics=diagnostics,
    )


def persist_cell_graph(result: CellGraphResult, plan_dir: Path) -> dict[str, str]:
    """Persist graph JSON and dense label map under ``plan_dir``."""
    plan_dir.mkdir(parents=True, exist_ok=True)
    labels_path = plan_dir / "cell_labels.npy"
    graph_path = plan_dir / "cell_graph.json"
    np.save(labels_path, result.labels.astype(np.int32))
    payload = {
        "diagnostics": result.diagnostics,
        "cells": result.cells,
        "adjacency": result.adjacency,
        "labels_path": str(labels_path),
    }
    graph_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return {"cell_graph_path": str(graph_path), "cell_labels_path": str(labels_path)}


__all__ = ["CellGraphResult", "build_cell_graph", "persist_cell_graph"]
