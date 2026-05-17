"""Real SNIC (Simple Non-Iterative Clustering) superpixel cell proposer.

Replaces the fixed-grid placeholder in chuck_mcp_v2.plan_emma._grid_cell_graph.

Pipeline:
    1. Load image, downscale to max_dim.
    2. Run SNIC (via pysnic) on the Lab-converted image; if pysnic is missing,
       fall back to skimage.segmentation.slic (still real, image-driven).
    3. Build per-cell statistics:
         - mean_rgb, mean_lab
         - chroma = sqrt(a*^2 + b*^2)
         - role_hint via heuristic on L*, chroma, and rough spatial prior
         - adjacency_list — 4-connected neighbors via label diff
         - pixels (count), centroid_yx
    4. Return cell_graph dict compatible with production_plan_builder:
         {"cells": {cell_id: {...}},
          "meta":  {"source": "snic"|"slic", "wall_time_seconds": ...,
                    "n_cells": N, "image_shape": (H, W)}}

The exposed function ``propose_cells(image_path, target_cells, max_dim)`` is
the only API consumers should depend on. ``hue_cluster_count`` is exposed for
the test suite and the integration patch.
"""
from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Lab conversion (sRGB D65 → CIELAB). Vectorised; matches colour-science to
# within ~0.3 ΔE — adequate for hue clustering.
# ---------------------------------------------------------------------------
def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = np.clip(c, 0.0, 1.0)
    a = 0.055
    return np.where(c <= 0.04045, c / 12.92, ((c + a) / (1 + a)) ** 2.4)


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Vectorised sRGB → CIELAB. rgb shape (..., 3) in [0, 1]."""
    lin = _srgb_to_linear(rgb)
    M = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float64,
    )
    xyz = lin @ M.T
    # D65 reference white
    Xn, Yn, Zn = 0.95047, 1.0, 1.08883
    xyz_n = xyz / np.array([Xn, Yn, Zn], dtype=np.float64)
    eps = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    f = np.where(xyz_n > eps, np.cbrt(xyz_n), (kappa * xyz_n + 16.0) / 116.0)
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
# SNIC dispatcher
# ---------------------------------------------------------------------------
def _run_snic(lab: np.ndarray, target_cells: int) -> tuple[np.ndarray, str]:
    """Run SNIC over a Lab image, returning (label_map, backend_tag).

    Falls back to skimage.segmentation.slic if pysnic is not importable.
    Both produce dense int label maps of shape (H, W).
    """
    try:
        from pysnic.algorithms.snic import snic  # type: ignore
    except ImportError:
        snic = None  # type: ignore[assignment]

    H, W = lab.shape[:2]
    compactness = 10.0

    if snic is not None:
        # pysnic expects a list-of-list-of-(L,a,b) tuples
        lab_list = lab.tolist()
        try:
            segmentation, _, centroids = snic(
                lab_list,
                target_cells,
                compactness,
            )
            labels = np.asarray(segmentation, dtype=np.int64)
            # pysnic occasionally returns -1 (unassigned border pixels). Heal
            # by nearest-neighbour from the right/down neighbour.
            if (labels < 0).any():
                mask = labels < 0
                idx_y, idx_x = np.where(mask)
                for y, x in zip(idx_y.tolist(), idx_x.tolist()):
                    for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < H and 0 <= nx < W and labels[ny, nx] >= 0:
                            labels[y, x] = labels[ny, nx]
                            break
            return labels, "snic"
        except Exception:
            # If SNIC chokes on edge cases, fall through to SLIC
            pass

    # SLIC fallback (still real, image-driven; no grid)
    from skimage.segmentation import slic

    labels = slic(
        lab,
        n_segments=target_cells,
        compactness=compactness,
        sigma=0.0,
        start_label=0,
        channel_axis=-1,
        convert2lab=False,  # we already gave Lab
        enforce_connectivity=True,
    )
    return labels.astype(np.int64), "slic"


# ---------------------------------------------------------------------------
# Role hint heuristic
#
# Used only as a coarse prior — the production_plan_builder still re-derives
# role assignments downstream. We want the per-cell role_hint distribution
# to span >= 5 named buckets so the planner can warm-start sensibly.
# ---------------------------------------------------------------------------
def _role_hint(
    mean_lab: np.ndarray,
    chroma: float,
    centroid_yx: tuple[float, float],
    shape: tuple[int, int],
) -> str:
    L, a, b = float(mean_lab[0]), float(mean_lab[1]), float(mean_lab[2])
    H, W = shape
    y, x = centroid_yx
    fy, fx = y / max(1, H - 1), x / max(1, W - 1)

    # Background: near-white, low chroma, OR clearly outside the face quad
    # (Chuck Close prints have a paper margin).
    is_margin = fy < 0.04 or fy > 0.97 or fx < 0.04 or fx > 0.97
    if (L > 86 and chroma < 10) or (is_margin and chroma < 18):
        return "background"

    # Hair: dark + cool, typically upper portion / sides
    if L < 38 and (fy < 0.35 or fx < 0.12 or fx > 0.88):
        return "hair"

    # Lip: reddish, mid lightness, lower-middle third
    hue_deg = np.degrees(np.arctan2(b, a))
    in_lip_band = 0.55 < fy < 0.85 and 0.30 < fx < 0.70
    if in_lip_band and 30 < L < 70 and chroma > 22 and (-30 < hue_deg < 45):
        return "lip"

    # Eyes: very dark or cool-saturated within the eye band
    in_eye_band = 0.30 < fy < 0.55 and 0.20 < fx < 0.80
    if in_eye_band and (L < 30 or (chroma > 25 and 150 < abs(hue_deg) % 360)):
        return "eyes"
    if in_eye_band and L < 32 and chroma < 30:
        return "eyes"

    # Skin: warm hues in the face quad
    if 0.10 < fy < 0.95 and 0.10 < fx < 0.95:
        if 35 < L < 92 and 5 < chroma < 55 and -25 < hue_deg < 75:
            return "skin"

    return "other"


# ---------------------------------------------------------------------------
# Adjacency from label map
# ---------------------------------------------------------------------------
def _build_adjacency(labels: np.ndarray) -> dict[int, list[int]]:
    """4-connected adjacency between superpixel labels."""
    adj: dict[int, set[int]] = defaultdict(set)
    # Horizontal neighbours
    left = labels[:, :-1]
    right = labels[:, 1:]
    diff = left != right
    for a_lbl, b_lbl in zip(left[diff].tolist(), right[diff].tolist()):
        if a_lbl >= 0 and b_lbl >= 0:
            adj[int(a_lbl)].add(int(b_lbl))
            adj[int(b_lbl)].add(int(a_lbl))
    # Vertical neighbours
    top = labels[:-1, :]
    bot = labels[1:, :]
    diff = top != bot
    for a_lbl, b_lbl in zip(top[diff].tolist(), bot[diff].tolist()):
        if a_lbl >= 0 and b_lbl >= 0:
            adj[int(a_lbl)].add(int(b_lbl))
            adj[int(b_lbl)].add(int(a_lbl))
    return {k: sorted(v) for k, v in adj.items()}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def propose_cells(
    image_path: str,
    *,
    target_cells: int = 2000,
    max_dim: int = 2048,
) -> dict[str, Any]:
    """Run real SNIC on the input image and return a cell_graph dict.

    Args:
        image_path: path to JPG/PNG/etc. — anything PIL can open.
        target_cells: SNIC target superpixel count (will be approximate).
        max_dim: cap on the longer image side before SNIC (perf safety).

    Returns:
        {"cells": {cell_id: {mean_rgb, mean_lab, chroma, role_hint,
                              adjacency_list, pixels, centroid_yx}},
         "meta": {source, wall_time_seconds, n_cells, image_shape}}
    """
    t0 = time.time()

    image = Image.open(image_path).convert("RGB")
    if max(image.size) > max_dim:
        image.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
    rgb = np.asarray(image, dtype=np.uint8)
    H, W = rgb.shape[:2]
    rgb01 = rgb.astype(np.float32) / 255.0
    lab = _rgb_to_lab(rgb01)

    labels, backend = _run_snic(lab, target_cells=target_cells)

    # Vectorised per-cell aggregation
    flat_labels = labels.ravel()
    n_labels = int(flat_labels.max()) + 1

    sums_rgb = np.zeros((n_labels, 3), dtype=np.float64)
    sums_lab = np.zeros((n_labels, 3), dtype=np.float64)
    counts = np.zeros(n_labels, dtype=np.int64)
    sums_y = np.zeros(n_labels, dtype=np.float64)
    sums_x = np.zeros(n_labels, dtype=np.float64)

    np.add.at(sums_rgb, flat_labels, rgb01.reshape(-1, 3))
    np.add.at(sums_lab, flat_labels, lab.reshape(-1, 3))
    np.add.at(counts, flat_labels, 1)

    yy, xx = np.mgrid[0:H, 0:W]
    np.add.at(sums_y, flat_labels, yy.ravel())
    np.add.at(sums_x, flat_labels, xx.ravel())

    adj = _build_adjacency(labels)

    cells: dict[int, dict[str, Any]] = {}
    for cid in range(n_labels):
        n = int(counts[cid])
        if n == 0:
            continue
        mean_rgb = (sums_rgb[cid] / n).astype(np.float32)
        mean_lab = (sums_lab[cid] / n).astype(np.float32)
        chroma = float(np.hypot(mean_lab[1], mean_lab[2]))
        cy = float(sums_y[cid] / n)
        cx = float(sums_x[cid] / n)
        role = _role_hint(mean_lab, chroma, (cy, cx), (H, W))
        cells[cid] = {
            "mean_rgb": mean_rgb,                       # in [0, 1]
            "mean_lab": mean_lab,
            "chroma": chroma,
            "role_hint": role,
            "adjacency_list": adj.get(cid, []),
            "pixels": n,
            "centroid_yx": (cy, cx),
        }

    wall = time.time() - t0
    return {
        "cells": cells,
        "meta": {
            "source": backend,
            "wall_time_seconds": wall,
            "n_cells": len(cells),
            "image_shape": (H, W),
            "target_cells": target_cells,
        },
    }


def hue_cluster_count(
    cell_graph: dict[str, Any],
    *,
    min_cells_per_cluster: int = 30,
    n_bins: int = 10,
) -> int:
    """Count distinct hue bins (atan2(b*, a*) split into ``n_bins``) that
    contain at least ``min_cells_per_cluster`` cells.

    Returns the number of populated bins — the planner uses this as a sanity
    floor on chromatic diversity (Emma >= 5).
    """
    cells = cell_graph["cells"]
    if not cells:
        return 0
    a = np.fromiter(
        (float(c["mean_lab"][1]) for c in cells.values()),
        dtype=np.float32,
        count=len(cells),
    )
    b = np.fromiter(
        (float(c["mean_lab"][2]) for c in cells.values()),
        dtype=np.float32,
        count=len(cells),
    )
    chroma = np.hypot(a, b)
    hue = np.degrees(np.arctan2(b, a)) % 360.0

    # Ignore near-neutral cells (chroma < 6) — they have no meaningful hue.
    chromatic = chroma >= 6.0
    if not chromatic.any():
        return 0

    bin_idx = (hue[chromatic] / (360.0 / n_bins)).astype(np.int64) % n_bins
    counts = np.bincount(bin_idx, minlength=n_bins)
    return int((counts >= min_cells_per_cluster).sum())


__all__ = ["propose_cells", "hue_cluster_count"]
