"""
chuck-mcp v3 — Region & cell-assignment visualizer.

Produces three PNG sheets to /tmp (or a chosen out_dir):

  emma_regions_polygons.png      — image + face mesh + colored polygon overlays
  emma_regions_masks.png         — 4x5 grid of per-region binary masks
  emma_cell_assignment.png       — original image colored by primary region
                                    each SNIC cell is filled with that region's
                                    color. Falls through to gray if no region.

Run:

  python3 visualize_regions.py [image_path] [out_dir]
"""

from __future__ import annotations
import sys
from pathlib import Path

import cv2
import numpy as np
from matplotlib import pyplot as plt

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
from face_region_mapper import extract_face_regions
from merge_regions_with_cells import (
    merge_face_regions_with_snic_cells, resolve_cell_to_primary_region,
)

# Distinguishable palette — 22 entries, hand-picked, BGR.
REGION_COLORS = {
    "face":          (180, 180, 180),
    "left_cheek":    (40, 180, 255),     # orange
    "right_cheek":   (40, 220, 255),     # paler orange
    "left_temple":   (200, 100, 0),      # cyan-blue
    "right_temple":  (200, 140, 0),
    "forehead":      (90, 200, 90),      # green
    "chin":          (30, 30, 200),      # red
    "left_jaw":      (180, 60, 200),     # magenta
    "right_jaw":     (220, 60, 200),
    "upper_lip":     (60, 60, 230),
    "lower_lip":     (100, 60, 230),
    "lips":          (40, 40, 200),
    "left_eye":      (0, 0, 0),
    "right_eye":     (0, 0, 0),
    "left_eyebrow":  (60, 30, 130),
    "right_eyebrow": (60, 30, 130),
    "nose":          (130, 200, 230),
    "hair":          (100, 50, 30),      # dark brown
    "background":    (240, 220, 200),    # pale
}


def _draw_polygon_overlay(img: np.ndarray, regions: dict, alpha: float = 0.45) -> np.ndarray:
    H, W = img.shape[:2]
    overlay = img.copy()
    for name, r in regions.items():
        color = REGION_COLORS.get(name, (200, 200, 200))
        # Fill the mask
        layer = np.zeros_like(img)
        layer[r.mask > 0] = color
        overlay = cv2.addWeighted(overlay, 1.0, layer, alpha, 0)
        # Draw polygon outline if available
        if r.polygon is not None:
            pts = np.array(r.polygon, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(overlay, [pts], isClosed=True, color=color, thickness=3)
    return overlay


def make_polygon_sheet(image_path: str, out_path: str):
    img = cv2.imread(image_path)
    regs = extract_face_regions(image_path)
    overlay = _draw_polygon_overlay(img, regs)
    # Draw confidence / source legend on the side
    legend_h = img.shape[0]
    legend_w = 700
    legend = np.full((legend_h, legend_w, 3), 245, dtype=np.uint8)
    y = 40
    cv2.putText(legend, "FACE REGIONS", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    y += 50
    for name in sorted(regs):
        r = regs[name]
        color = REGION_COLORS.get(name, (200, 200, 200))
        cv2.rectangle(legend, (20, y - 25), (60, y + 5), color, -1)
        n_px = int((r.mask > 0).sum())
        cv2.putText(legend, f"{name}", (75, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(legend, f"src={r.source[:40]} conf={r.confidence:.2f} px={n_px}",
                    (75, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)
        y += 55
    sheet = np.hstack([overlay, legend])
    cv2.imwrite(out_path, sheet)
    print(f"wrote {out_path}  ({sheet.shape[1]}x{sheet.shape[0]})")


def make_mask_grid(image_path: str, out_path: str):
    img = cv2.imread(image_path)
    regs = extract_face_regions(image_path)
    names = sorted(regs)
    n = len(names)
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.6, rows * 4.2))
    axes = np.array(axes).reshape(rows, cols)
    for i, name in enumerate(names):
        r = regs[name]
        ax = axes[i // cols, i % cols]
        # overlay mask onto a dimmed image
        dimmed = (img * 0.35).astype(np.uint8)
        col = REGION_COLORS.get(name, (200, 200, 200))
        bright = np.where(r.mask[..., None] > 0, np.array(col), dimmed)
        bright = bright.astype(np.uint8)
        ax.imshow(cv2.cvtColor(bright, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{name}\n{r.source[:35]} conf={r.confidence:.2f}", fontsize=8)
        ax.axis("off")
    for j in range(n, rows * cols):
        axes[j // cols, j % cols].axis("off")
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def make_cell_assignment_sheet(image_path: str, out_path: str, n_segments: int = 1200):
    from skimage.segmentation import slic

    img = cv2.imread(image_path)
    H, W = img.shape[:2]
    regs = extract_face_regions(image_path)
    print(f"running SLIC (stand-in for SNIC) with n_segments={n_segments} ...")
    labels = slic(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                  n_segments=n_segments, compactness=12.0, start_label=0).astype(np.int32)
    print(f"got {labels.max() + 1} superpixels")
    assign = merge_face_regions_with_snic_cells(regs, labels, strategy="centroid")
    primary = resolve_cell_to_primary_region(assign)

    out_img = np.full_like(img, 180)  # gray for unassigned cells
    for cid, name in primary.items():
        col = REGION_COLORS.get(name, (200, 200, 200))
        out_img[labels == cid] = col
    # Overlay the original image at 35% on top of the colored cells
    blended = cv2.addWeighted(out_img, 0.7, img, 0.3, 0)
    # Draw SNIC cell boundaries lightly
    edges = np.zeros((H, W), dtype=np.uint8)
    edges[1:, :] |= (labels[1:, :] != labels[:-1, :]).astype(np.uint8)
    edges[:, 1:] |= (labels[:, 1:] != labels[:, :-1]).astype(np.uint8)
    blended[edges > 0] = (40, 40, 40)

    # Three-panel layout: original | regions | cell assignment
    poly_overlay = _draw_polygon_overlay(img, regs, alpha=0.40)
    panel = np.hstack([img, poly_overlay, blended])
    cv2.imwrite(out_path, panel)
    print(f"wrote {out_path}  ({panel.shape[1]}x{panel.shape[0]})")

    # Also print region->cell count summary
    print("\nRegion -> #SNIC cells assigned:")
    for name in sorted(assign):
        print(f"  {name:14s}  {len(assign[name]):4d} cells")


if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else str(_HERE / "close_emma_2002_2048.jpg")
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else _HERE
    out_dir.mkdir(exist_ok=True, parents=True)

    print(f"input: {image_path}")
    print(f"output dir: {out_dir}")
    make_polygon_sheet(image_path, str(out_dir / "emma_regions_polygons.png"))
    make_mask_grid(image_path, str(out_dir / "emma_regions_masks.png"))
    make_cell_assignment_sheet(image_path, str(out_dir / "emma_cell_assignment.png"))
