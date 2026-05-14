"""S6.b — region-level jigsaw organization for printable color plates.

The inverse solver optimizes pigment alphas pixel-by-pixel. That can produce
technically useful color masks that are hard to carve or ink: many hue decisions
inside one local patch, fuzzy overlap between middle colors, and no clear
jigsaw boundaries. This pass runs after S5 and before state classification,
block packing, and vector export. It preserves broad underlayers and final
detail/key work, but makes middle color roles region-exclusive over target
superpixels so disjoint hue zones can share a physical block face.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from skimage import morphology, segmentation

from backend.services.v23.core import forward_render_jax

_ACTIVE_THRESHOLD = 0.035
_MAX_SEGMENT_PIXELS = 850_000
_TINT_RECOVERY_PIGMENTS = frozenset({0, 1, 3, 13, 14, 17, 21, 23})
_ADAPTIVE_PIGMENT_START = 24
_SUPPORT_PIGMENTS = frozenset({0, 1, 2, 13, 14, 21, 23, 26, 31})


@dataclass(frozen=True)
class JigsawOrganizationResult:
    """Output of :func:`organize_jigsaw_regions`."""

    alpha_stack: NDArray[np.float32]
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class _RoleLayout:
    under_count: int
    mid_count: int
    detail_count: int

    @property
    def under_end(self) -> int:
        return self.under_count

    @property
    def mid_end(self) -> int:
        return self.under_count + self.mid_count


def _role_layout(m: int, pigment_idx: NDArray[np.int32] | None = None) -> _RoleLayout:
    if m <= 2:
        return _RoleLayout(under_count=0, mid_count=0, detail_count=m)
    if m <= 4:
        return _RoleLayout(under_count=1, mid_count=max(0, m - 2), detail_count=1)
    if pigment_idx is not None and any(
        int(pid) >= _ADAPTIVE_PIGMENT_START for pid in pigment_idx.tolist()
    ):
        detail_count = 2
        support_count = sum(1 for pid in pigment_idx.tolist() if int(pid) in _SUPPORT_PIGMENTS)
        under_count = min(3, max(1, support_count), m - detail_count)
        return _RoleLayout(
            under_count=under_count,
            mid_count=max(0, m - under_count - detail_count),
            detail_count=detail_count,
        )
    detail_count = 2
    under_count = min(3, m - detail_count)
    mid_count = max(0, m - under_count - detail_count)
    return _RoleLayout(
        under_count=under_count,
        mid_count=mid_count,
        detail_count=detail_count,
    )


def _pairwise_overlap(alpha: NDArray[np.float32]) -> float:
    m = int(alpha.shape[0])
    if m <= 1:
        return 0.0
    total = 0.0
    pairs = 0
    for i in range(m):
        for j in range(i + 1, m):
            total += float(np.mean(alpha[i] * alpha[j]))
            pairs += 1
    return total / float(max(pairs, 1))


def _segment_image(
    target_rgb: NDArray[np.float32],
    *,
    n_segments: int | None,
) -> NDArray[np.int32]:
    h, w = target_rgb.shape[:2]
    pixels = h * w
    seg_rgb = target_rgb
    scale = 1.0
    if pixels > _MAX_SEGMENT_PIXELS:
        scale = (_MAX_SEGMENT_PIXELS / float(pixels)) ** 0.5
        new_w = max(96, int(round(w * scale)))
        new_h = max(96, int(round(h * scale)))
        arr = np.clip(target_rgb * 255.0, 0, 255).astype(np.uint8)
        seg_rgb = np.asarray(
            Image.fromarray(arr, "RGB").resize((new_w, new_h), Image.Resampling.LANCZOS),
            dtype=np.float32,
        ) / 255.0
    seg_h, seg_w = seg_rgb.shape[:2]
    if n_segments is None:
        n_segments = max(96, min(1800, int(round((seg_h * seg_w) / 520.0))))
    labels = segmentation.slic(
        seg_rgb,
        n_segments=int(n_segments),
        compactness=14.0,
        sigma=1.0,
        start_label=0,
        channel_axis=-1,
        convert2lab=True,
    ).astype(np.int32)
    if scale != 1.0:
        labels = np.asarray(
            Image.fromarray(labels, mode="I").resize((w, h), Image.Resampling.NEAREST),
            dtype=np.int32,
        )
    return labels


def _cleanup_mid_mask(
    alpha: NDArray[np.float32],
    *,
    min_region_px: int,
) -> NDArray[np.float32]:
    active = alpha >= _ACTIVE_THRESHOLD
    active = morphology.remove_small_objects(
        active,
        max_size=max(0, int(min_region_px) - 1),
    )
    active = morphology.closing(active, footprint=morphology.disk(1))
    visible = alpha[alpha >= _ACTIVE_THRESHOLD]
    fill = float(np.median(visible)) if visible.size else _ACTIVE_THRESHOLD
    filled = np.where(active & (alpha < _ACTIVE_THRESHOLD), fill, alpha)
    return np.where(active, filled, 0.0).astype(np.float32)


def _forward_numpy(
    alpha_stack: NDArray[np.float32],
    pigment_idx: NDArray[np.int32],
) -> NDArray[np.float32]:
    h, w = alpha_stack.shape[1:]
    composite = np.broadcast_to(forward_render_jax.PAPER_RGB, (h, w, 3)).copy()
    pigments = forward_render_jax.PIGMENT_TABLE[pigment_idx]
    for alpha, pigment in zip(alpha_stack, pigments, strict=True):
        a = np.clip(alpha[..., None], 0.0, 1.0)
        composite = composite * (1.0 - a) + pigment[None, None, :] * a
    return composite.astype(np.float32)


def _recover_near_paper_tints(
    alpha_stack: NDArray[np.float32],
    pigment_idx: NDArray[np.int32],
    target_rgb: NDArray[np.float32],
    labels: NDArray[np.int32],
) -> tuple[NDArray[np.float32], dict[str, Any]]:
    out = alpha_stack.copy()
    current = _forward_numpy(out, pigment_idx)
    paper = forward_render_jax.PAPER_RGB.astype(np.float32)
    target_lum = (
        0.299 * target_rgb[..., 0]
        + 0.587 * target_rgb[..., 1]
        + 0.114 * target_rgb[..., 2]
    )
    target_delta = np.linalg.norm(target_rgb - paper[None, None, :], axis=-1)
    current_delta = np.linalg.norm(current - paper[None, None, :], axis=-1)
    needs_tint = (
        (target_lum > 0.50)
        & (target_delta > 0.018)
        & (current_delta < target_delta * 0.92)
    )
    candidate_slots = [
        i for i, pid in enumerate(pigment_idx.tolist())
        if int(pid) in _TINT_RECOVERY_PIGMENTS
    ]
    if not candidate_slots:
        return out, {
            "tint_recovery_cells": 0,
            "tint_recovery_by_slot": [],
        }

    pigments = forward_render_jax.PIGMENT_TABLE[pigment_idx]
    recovered_cells = 0
    by_slot = [0 for _ in range(alpha_stack.shape[0])]
    for label_id in np.unique(labels).tolist():
        cell = labels == int(label_id)
        active = cell & needs_tint
        if float(active.mean()) < 0.00002:
            continue
        target_mean = target_rgb[active].mean(axis=0)
        current_mean = current[active].mean(axis=0)
        before = float(np.sum((current_mean - target_mean) ** 2))
        best_slot: int | None = None
        best_boost = 0.0
        best_gain = 0.0
        for slot in candidate_slots:
            # Tint recovery is a gap-filler for pale regions the solver left
            # too close to paper. Do not thicken an already established plate.
            if float(out[slot, active].mean()) >= _ACTIVE_THRESHOLD:
                continue
            direction = pigments[slot] - current_mean
            denom = float(np.sum(direction * direction))
            if denom <= 1e-8:
                continue
            optimum = float(np.dot(target_mean - current_mean, direction) / denom)
            boost = float(np.clip(optimum, 0.0, 0.16))
            if boost <= 0.008:
                continue
            after = current_mean * (1.0 - boost) + pigments[slot] * boost
            gain = before - float(np.sum((after - target_mean) ** 2))
            if gain > best_gain:
                best_gain = gain
                best_boost = boost
                best_slot = slot
        if best_slot is None or best_gain <= 0.0:
            continue
        out[best_slot, active] = np.clip(
            out[best_slot, active] + best_boost * (1.0 - out[best_slot, active]),
            0.0,
            1.0,
        )
        recovered_cells += 1
        by_slot[best_slot] += 1

    return out.astype(np.float32), {
        "tint_recovery_cells": int(recovered_cells),
        "tint_recovery_by_slot": by_slot,
    }


def organize_jigsaw_regions(
    alpha_stack: NDArray[np.float32],
    pigment_idx: NDArray[np.int32],
    *,
    target_rgb: NDArray[np.float32],
    cell_labels: NDArray[np.int32] | None = None,
    n_segments: int | None = None,
    min_region_px: int = 96,
    activity_threshold: float = _ACTIVE_THRESHOLD,
) -> JigsawOrganizationResult:
    """Make middle color plates region-exclusive over perceptual image cells.

    Underlayers stay overlapping and broad. Final detail/key plates stay detailed.
    Middle color plates are converted into jigsaw-like regions: for each target
    superpixel, the strongest middle role owns that whole cell and competing
    middle roles are zeroed there.
    """
    if alpha_stack.ndim != 3:
        raise ValueError(f"alpha_stack must be MHW, got {alpha_stack.shape!r}")
    if target_rgb.ndim != 3 or target_rgb.shape[-1] != 3:
        raise ValueError(f"target_rgb must be HWC RGB, got {target_rgb.shape!r}")
    m, h, w = alpha_stack.shape
    if target_rgb.shape[:2] != (h, w):
        raise ValueError(
            f"target_rgb shape {target_rgb.shape[:2]} must match alpha {(h, w)}"
        )

    pigment_idx = pigment_idx.astype(np.int32)
    layout = _role_layout(m, pigment_idx)
    if cell_labels is not None:
        if cell_labels.shape != (h, w):
            raise ValueError(f"cell_labels shape {cell_labels.shape!r} must match {(h, w)!r}")
        labels = cell_labels.astype(np.int32)
    else:
        labels = _segment_image(target_rgb.astype(np.float32), n_segments=n_segments)
    if layout.mid_count <= 1:
        recovered, recovery = _recover_near_paper_tints(
            alpha_stack.astype(np.float32),
            pigment_idx,
            target_rgb.astype(np.float32),
            labels,
        )
        return JigsawOrganizationResult(
            alpha_stack=recovered,
            diagnostics={
                "applied": False,
                "reason": "not enough middle color roles",
                "mid_count": layout.mid_count,
                **recovery,
            },
        )

    out = alpha_stack.astype(np.float32).copy()
    mid_slice = slice(layout.under_end, layout.mid_end)
    mid_before = out[mid_slice].copy()
    label_ids = np.unique(labels)
    assigned_cells = [0 for _ in range(layout.mid_count)]
    inactive_cells = 0

    out[mid_slice] = 0.0
    for label_id in label_ids.tolist():
        cell = labels == int(label_id)
        means = mid_before[:, cell].mean(axis=1)
        winner = int(np.argmax(means))
        winner_mean = float(means[winner])
        if winner_mean < activity_threshold:
            inactive_cells += 1
            continue
        winner_alpha = mid_before[winner, cell]
        fill = max(winner_mean * 0.85, float(np.percentile(winner_alpha, 70)) * 0.65)
        out[layout.under_end + winner, cell] = np.clip(
            0.68 * winner_alpha + 0.32 * fill,
            0.0,
            1.0,
        )
        assigned_cells[winner] += 1

    for offset in range(layout.mid_count):
        idx = layout.under_end + offset
        out[idx] = _cleanup_mid_mask(out[idx], min_region_px=min_region_px)

    out, recovery = _recover_near_paper_tints(
        out,
        pigment_idx,
        target_rgb.astype(np.float32),
        labels,
    )
    mid_after = out[mid_slice]
    diagnostics: dict[str, Any] = {
        "applied": True,
        "cell_count": int(label_ids.size),
        "inactive_cells": int(inactive_cells),
        "under_count": layout.under_count,
        "mid_count": layout.mid_count,
        "detail_count": layout.detail_count,
        "assigned_cells_by_mid_slot": assigned_cells,
        "mid_pairwise_overlap_before": round(_pairwise_overlap(mid_before), 6),
        "mid_pairwise_overlap_after": round(_pairwise_overlap(mid_after), 6),
        "mid_coverage_before": [
            round(float((mid_before[i] >= activity_threshold).mean() * 100.0), 3)
            for i in range(layout.mid_count)
        ],
        "mid_coverage_after": [
            round(float((mid_after[i] >= activity_threshold).mean() * 100.0), 3)
            for i in range(layout.mid_count)
        ],
        **recovery,
    }
    return JigsawOrganizationResult(alpha_stack=out.astype(np.float32), diagnostics=diagnostics)


__all__ = ["JigsawOrganizationResult", "organize_jigsaw_regions"]
